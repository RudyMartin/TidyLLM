#!/usr/bin/env python3
"""
Safe Library Upgrade Script for VectorQA Sage
Performs incremental upgrades with testing at each step
"""

import subprocess
import sys
import os
from typing import List, Tuple

def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}")
    print(f"   Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"   Error: {e.stderr}")
        return False

def run_tests() -> bool:
    """Run the test suite and return success status."""
    print("\n🧪 Running test suite...")
    return run_command([sys.executable, "run_tests.py", "--fast"], "Test suite")

def backup_requirements() -> bool:
    """Backup current requirements."""
    print("\n💾 Creating backup...")
    return run_command([
        sys.executable, "-m", "pip", "freeze"
    ], "Backup current environment to requirements_backup.txt") and \
    run_command([
        sys.executable, "-c", 
        "import subprocess; open('requirements_backup.txt', 'w').write(subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True).stdout)"
    ], "Save backup")

def upgrade_package(package: str, version: str = None) -> bool:
    """Upgrade a specific package."""
    if version:
        package_spec = f"{package}=={version}"
        description = f"Upgrading {package} to {version}"
    else:
        package_spec = package
        description = f"Upgrading {package} to latest"
    
    return run_command([
        sys.executable, "-m", "pip", "install", "--upgrade", package_spec
    ], description)

def main():
    print("🚀 VectorQA Sage - Safe Library Upgrade")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("run_tests.py"):
        print("❌ Error: run_tests.py not found. Please run from project root.")
        sys.exit(1)
    
    # Backup current state
    print("\n📋 Step 1: Backup Current Environment")
    if not backup_requirements():
        print("❌ Failed to backup environment")
        sys.exit(1)
    
    # Run initial tests
    print("\n📋 Step 2: Verify Current State")
    if not run_tests():
        print("❌ Current tests are failing. Fix tests before upgrading.")
        sys.exit(1)
    
    print("✅ Environment backed up and tests passing. Ready to upgrade!")
    
    # Phase 1: Safe updates
    print("\n📋 Step 3: Phase 1 - Safe Updates")
    safe_packages = [
        ("pip", None),
        ("boto3", None),
        ("botocore", None),
        ("dill", None),
        ("fsspec", None),
        ("multiprocess", None),
        ("pydantic-core", None)
    ]
    
    phase1_success = True
    for package, version in safe_packages:
        if not upgrade_package(package, version):
            phase1_success = False
            break
    
    if not phase1_success:
        print("❌ Phase 1 upgrades failed")
        return
    
    # Test after Phase 1
    print("\n🧪 Testing after Phase 1...")
    if not run_tests():
        print("❌ Tests failed after Phase 1. Rolling back...")
        run_command([sys.executable, "-m", "pip", "install", "-r", "requirements_backup.txt"], 
                   "Rolling back to backup")
        return
    
    print("✅ Phase 1 complete and tests passing!")
    
    # Phase 2: Critical updates (with user confirmation)
    print("\n📋 Step 4: Phase 2 - Critical Updates")
    print("⚠️  This phase includes major version updates that may break compatibility.")
    
    response = input("Do you want to proceed with Phase 2? (y/N): ").lower().strip()
    if response != 'y':
        print("✅ Stopping at Phase 1. Your environment is safely updated!")
        return
    
    critical_packages = [
        ("dspy", "3.0.1"),
        ("litellm", "1.75.9")
    ]
    
    for package, version in critical_packages:
        print(f"\n🔄 Upgrading {package} to {version}")
        
        if not upgrade_package(package, version):
            print(f"❌ Failed to upgrade {package}")
            break
        
        # Test after each critical upgrade
        print(f"\n🧪 Testing after {package} upgrade...")
        if not run_tests():
            print(f"❌ Tests failed after {package} upgrade. Rolling back...")
            run_command([sys.executable, "-m", "pip", "install", "-r", "requirements_backup.txt"], 
                       "Rolling back to backup")
            return
        
        print(f"✅ {package} upgrade successful!")
    
    # Final test
    print("\n🏁 Final Verification")
    if run_tests():
        print("🎉 All upgrades completed successfully!")
        print("📝 Updating requirements.txt...")
        
        # Update requirements.txt with new versions
        result = subprocess.run([sys.executable, "-m", "pip", "freeze"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            with open("requirements_new.txt", "w") as f:
                f.write(result.stdout)
            print("✅ New requirements saved to requirements_new.txt")
        
        print("\n📊 Summary:")
        print("✅ All packages upgraded successfully")
        print("✅ All tests passing")
        print("✅ Environment is stable")
        print("\n💡 Next steps:")
        print("1. Review requirements_new.txt")
        print("2. Test the Streamlit app manually")
        print("3. Replace requirements.txt with requirements_new.txt if satisfied")
        
    else:
        print("❌ Final tests failed. This shouldn't happen!")

if __name__ == "__main__":
    main()
