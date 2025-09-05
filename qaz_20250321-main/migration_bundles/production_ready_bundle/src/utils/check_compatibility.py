#!/usr/bin/env python3
"""
Version Compatibility Checker for VectorQA Sage
Checks for package updates and compatibility issues
"""

import subprocess
import sys
import json
from typing import Dict, List, Tuple

def get_installed_packages() -> Dict[str, str]:
    """Get currently installed packages and their versions."""
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "list", "--format=json"], 
                              capture_output=True, text=True, check=True)
        packages = json.loads(result.stdout)
        return {pkg["name"].lower(): pkg["version"] for pkg in packages}
    except Exception as e:
        print(f"Error getting installed packages: {e}")
        return {}

def get_latest_version(package_name: str) -> str:
    """Get latest version of a package from PyPI."""
    try:
        import urllib.request
        import urllib.parse
        
        url = f"https://pypi.org/pypi/{package_name}/json"
        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read().decode())
            return data["info"]["version"]
    except Exception as e:
        print(f"Warning: Could not check latest version for {package_name}: {e}")
    return "unknown"

def check_python_compatibility() -> Dict[str, any]:
    """Check Python version compatibility."""
    python_version = sys.version_info
    compatibility = {
        "version": f"{python_version.major}.{python_version.minor}.{python_version.micro}",
        "supported": python_version >= (3, 8),
        "recommended": python_version >= (3, 10),
        "issues": []
    }
    
    if python_version < (3, 8):
        compatibility["issues"].append("Python 3.8+ required for modern ML libraries")
    elif python_version < (3, 10):
        compatibility["issues"].append("Python 3.10+ recommended for best performance")
    
    return compatibility

def check_critical_packages() -> List[Dict[str, any]]:
    """Check critical packages for updates and compatibility."""
    critical_packages = [
        "dspy", "litellm", "streamlit", "pandas", "numpy", 
        "torch", "transformers", "sentence-transformers",
        "boto3", "scikit-learn", "matplotlib", "pytest"
    ]
    
    installed = get_installed_packages()
    package_status = []
    
    for package in critical_packages:
        if package.lower() in installed:
            current = installed[package.lower()]
            latest = get_latest_version(package)
            
            status = {
                "name": package,
                "current": current,
                "latest": latest,
                "needs_update": latest != "unknown" and current != latest,
                "critical": package in ["dspy", "streamlit", "pandas", "numpy"]
            }
            package_status.append(status)
        else:
            package_status.append({
                "name": package,
                "current": "not installed",
                "latest": get_latest_version(package),
                "needs_update": True,
                "critical": True
            })
    
    return package_status

def run_security_check() -> Dict[str, any]:
    """Run security vulnerability check."""
    try:
        # Install safety if not available
        subprocess.run([sys.executable, "-m", "pip", "install", "safety"], 
                      capture_output=True, check=False)
        
        result = subprocess.run([sys.executable, "-m", "safety", "check", "--json"], 
                              capture_output=True, text=True, check=False)
        
        if result.returncode == 0:
            return {"status": "safe", "vulnerabilities": []}
        else:
            try:
                vulnerabilities = json.loads(result.stdout)
                return {"status": "vulnerable", "vulnerabilities": vulnerabilities}
            except:
                return {"status": "error", "message": result.stderr}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def generate_upgrade_plan(package_status: List[Dict[str, any]]) -> List[str]:
    """Generate upgrade commands."""
    upgrade_commands = []
    
    # Critical updates first
    critical_updates = [p for p in package_status if p.get("critical") and p.get("needs_update")]
    if critical_updates:
        packages = [f"{p['name']}=={p['latest']}" for p in critical_updates if p['latest'] != "unknown"]
        if packages:
            upgrade_commands.append(f"pip install --upgrade {' '.join(packages)}")
    
    # Non-critical updates
    other_updates = [p for p in package_status if not p.get("critical") and p.get("needs_update")]
    if other_updates:
        packages = [f"{p['name']}=={p['latest']}" for p in other_updates if p['latest'] != "unknown"]
        if packages:
            upgrade_commands.append(f"pip install --upgrade {' '.join(packages)}")
    
    return upgrade_commands

def main():
    print("🔍 VectorQA Sage - Version Compatibility Check")
    print("=" * 50)
    
    # Check Python compatibility
    print("\n📍 Python Version Check:")
    python_compat = check_python_compatibility()
    print(f"Current: Python {python_compat['version']}")
    print(f"Supported: {'✅' if python_compat['supported'] else '❌'}")
    print(f"Recommended: {'✅' if python_compat['recommended'] else '⚠️'}")
    
    if python_compat['issues']:
        print("Issues:")
        for issue in python_compat['issues']:
            print(f"  - {issue}")
    
    # Check package versions
    print("\n📦 Package Version Check:")
    package_status = check_critical_packages()
    
    needs_update = [p for p in package_status if p.get("needs_update")]
    critical_updates = [p for p in needs_update if p.get("critical")]
    
    print(f"Total packages checked: {len(package_status)}")
    print(f"Packages needing updates: {len(needs_update)}")
    print(f"Critical updates: {len(critical_updates)}")
    
    if critical_updates:
        print("\n⚠️  Critical Updates Available:")
        for pkg in critical_updates:
            print(f"  {pkg['name']}: {pkg['current']} → {pkg['latest']}")
    
    if needs_update and not critical_updates:
        print("\n📈 Optional Updates Available:")
        for pkg in needs_update:
            print(f"  {pkg['name']}: {pkg['current']} → {pkg['latest']}")
    
    # Generate upgrade plan
    print("\n🚀 Upgrade Plan:")
    upgrade_commands = generate_upgrade_plan(package_status)
    if upgrade_commands:
        for i, cmd in enumerate(upgrade_commands, 1):
            print(f"{i}. {cmd}")
        
        print("\n💡 To upgrade all packages:")
        print("python3 -m pip install --upgrade -r requirements_updated.txt")
    else:
        print("✅ All packages are up to date!")
    
    # Security check
    print("\n🛡️  Security Check:")
    security = run_security_check()
    if security["status"] == "safe":
        print("✅ No known vulnerabilities found")
    elif security["status"] == "vulnerable":
        print(f"⚠️  {len(security['vulnerabilities'])} vulnerabilities found")
        for vuln in security["vulnerabilities"][:3]:  # Show first 3
            print(f"  - {vuln.get('package', 'Unknown')}: {vuln.get('advisory', 'Security issue')}")
    else:
        print(f"❓ Security check failed: {security.get('message', 'Unknown error')}")
    
    print("\n" + "=" * 50)
    print("✅ Compatibility check complete!")

if __name__ == "__main__":
    main()
