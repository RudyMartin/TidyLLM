#!/usr/bin/env python3
"""
Safe Package Cleanup Script for VectorQA Sage
Removes unnecessary packages while preserving functionality
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

def backup_environment() -> bool:
    """Backup current environment."""
    print("\n💾 Creating environment backup...")
    return run_command([
        sys.executable, "-m", "pip", "freeze"
    ], "Backup current environment to requirements_backup_cleanup.txt") and \
    run_command([
        sys.executable, "-c", 
        "import subprocess; open('requirements_backup_cleanup.txt', 'w').write(subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True).stdout)"
    ], "Save backup")

def uninstall_packages(packages: List[str], phase: str) -> bool:
    """Uninstall a list of packages."""
    if not packages:
        return True
    
    print(f"\n📦 {phase}")
    print(f"   Removing: {', '.join(packages)}")
    
    cmd = [sys.executable, "-m", "pip", "uninstall", "-y"] + packages
    return run_command(cmd, f"Uninstall {len(packages)} packages")

def check_conflicts() -> bool:
    """Check for package conflicts."""
    print("\n🔍 Checking for package conflicts...")
    result = subprocess.run([sys.executable, "-m", "pip", "check"], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ No package conflicts found")
        return True
    else:
        print("⚠️  Package conflicts found:")
        print(result.stdout)
        return False

def main():
    print("🧹 VectorQA Sage - Safe Package Cleanup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("run_tests.py"):
        print("❌ Error: run_tests.py not found. Please run from project root.")
        sys.exit(1)
    
    # Backup current state
    print("\n📋 Step 1: Backup Current Environment")
    if not backup_environment():
        print("❌ Failed to backup environment")
        sys.exit(1)
    
    # Run initial tests
    print("\n📋 Step 2: Verify Current State")
    if not run_tests():
        print("❌ Current tests are failing. Fix tests before cleanup.")
        sys.exit(1)
    
    print("✅ Environment backed up and tests passing. Ready to cleanup!")
    
    # Phase 1: Remove duplicates
    print("\n📋 Step 3: Phase 1 - Remove Duplicates")
    duplicate_removals = ["PyPDF2"]  # Keep pypdf (newer version)
    
    if not uninstall_packages(duplicate_removals, "Phase 2: Duplicates"):
        print("❌ Failed to remove duplicates")
        return
    
    # Test after Phase 2
    print("\n🧪 Testing after Phase 2...")
    if not run_tests():
        print("❌ Tests failed after Phase 2. Rolling back...")
        run_command([sys.executable, "-m", "pip", "install", "-r", "requirements_backup_cleanup.txt"], 
                   "Rolling back to backup")
        return
    
    # Phase 2: Remove unused major packages
    print("\n📋 Step 4: Phase 2 - Remove Unused Major Packages")
    major_removals = [
        "SQLAlchemy", "psycopg2-binary", "alembic", "greenlet",  # Database
        "fastapi", "uvicorn", "starlette", "h11",  # Web framework
        "GitPython", "gitdb", "smmap",  # Git
        "optuna", "narwhals"  # Optimization
    ]
    
    if not uninstall_packages(major_removals, "Phase 3: Unused Major Packages"):
        print("❌ Failed to remove unused major packages")
        return
    
    # Test after Phase 2
    print("\n🧪 Testing after Phase 2...")
    if not run_tests():
        print("❌ Tests failed after Phase 2. Rolling back...")
        run_command([sys.executable, "-m", "pip", "install", "-r", "requirements_backup_cleanup.txt"], 
                   "Rolling back to backup")
        return
    
    # Phase 3: Remove unused utilities (with user confirmation)
    print("\n📋 Step 5: Phase 3 - Remove Unused Utilities")
    print("⚠️  This phase removes smaller utility packages that are not used.")
    
    response = input("Do you want to proceed with Phase 4? (y/N): ").lower().strip()
    if response != 'y':
        print("✅ Stopping at Phase 3. Your environment is cleaned up!")
        return
    
    utility_removals = [
        "aiohappyeyeballs", "aiohttp", "aiosignal", "frozenlist", "multidict", "yarl",  # Async/HTTP
        "magicattr", "propcache", "jiter"  # Utilities
    ]
    
    if not uninstall_packages(utility_removals, "Phase 4: Unused Utilities"):
        print("❌ Failed to remove unused utilities")
        return
    
    # Test after Phase 3
    print("\n🧪 Testing after Phase 3...")
    if not run_tests():
        print("❌ Tests failed after Phase 3. Rolling back...")
        run_command([sys.executable, "-m", "pip", "install", "-r", "requirements_backup_cleanup.txt"], 
                   "Rolling back to backup")
        return
    
    # Final verification
    print("\n🏁 Final Verification")
    if check_conflicts() and run_tests():
        print("🎉 Package cleanup completed successfully!")
        
        # Show package count
        result = subprocess.run([sys.executable, "-m", "pip", "list"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            package_count = len([line for line in result.stdout.split('\n') 
                               if line.strip() and 'Package' not in line and '---' not in line])
            print(f"📊 Total packages after cleanup: {package_count}")
        
        print("\n📊 Summary:")
        print("✅ Critical conflicts resolved")
        print("✅ Duplicates removed")
        print("✅ Unused packages removed")
        print("✅ All tests passing")
        print("✅ No package conflicts")
        print("\n💡 Your environment is now clean and optimized!")
        print("📝 Backup saved as: requirements_backup_cleanup.txt")
        
    else:
        print("❌ Final verification failed. This shouldn't happen!")

if __name__ == "__main__":
    main()
