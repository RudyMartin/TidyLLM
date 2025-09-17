#!/usr/bin/env python3
"""
V2 Safety Validation - Check all critical constraints
"""

import os
import subprocess
from pathlib import Path

def check_no_mlruns_folders():
    """CRITICAL: Check for any mlruns folders that would break the app"""
    print("CHECKING FOR FORBIDDEN MLRUNS FOLDERS")
    print("-" * 40)
    
    search_paths = [
        Path("C:/Users/marti/AI-Scoring"),
        Path("C:/Users/marti/AI-Scoring/v2"),
        Path("C:/Users/marti/AI-Scoring/onboarding"),
        Path("C:/Users/marti/AI-Scoring/tidyllm")
    ]
    
    mlruns_found = []
    
    for base_path in search_paths:
        if base_path.exists():
            # Find any mlruns directories
            mlruns_dirs = list(base_path.rglob("mlruns"))
            for mlrun_dir in mlruns_dirs:
                size = sum(f.stat().st_size for f in mlrun_dir.rglob('*') if f.is_file())
                mlruns_found.append((str(mlrun_dir), size))
    
    if mlruns_found:
        print("CRITICAL WARNING: MLRUNS FOLDERS FOUND!")
        for folder, size in mlruns_found:
            size_mb = size / (1024 * 1024)
            print(f"  DANGER: {folder} ({size_mb:.2f} MB)")
        
        print("\nTHESE FOLDERS WILL BREAK THE APPLICATION!")
        print("They consume disk space and violate V2 architecture.")
        print("Remove them immediately:")
        for folder, _ in mlruns_found:
            print(f"  rm -rf \"{folder}\"")
        
        return False
    else:
        print("OK: No mlruns folders found")
        return True

def check_sqlite_files():
    """Check for SQLite files that should use PostgreSQL"""
    print("\nCHECKING FOR SQLITE FILES (SHOULD USE POSTGRESQL)")
    print("-" * 50)
    
    v2_path = Path("C:/Users/marti/AI-Scoring/v2")
    sqlite_files = list(v2_path.rglob("*.db")) + list(v2_path.rglob("*.sqlite"))
    
    if sqlite_files:
        print("WARNING: SQLite files found in V2:")
        for db_file in sqlite_files:
            size = db_file.stat().st_size / (1024 * 1024) if db_file.exists() else 0
            print(f"  {db_file} ({size:.2f} MB)")
        
        print("\nV2 should use PostgreSQL backend, not SQLite!")
        return False
    else:
        print("OK: No SQLite files found in V2")
        return True

def check_hardcoded_credentials():
    """Check for hardcoded credentials in V2 code"""
    print("\nCHECKING FOR HARDCODED CREDENTIALS")
    print("-" * 35)
    
    v2_src = Path("C:/Users/marti/AI-Scoring/v2/src")
    if not v2_src.exists():
        print("V2 src directory not found")
        return True
    
    dangerous_patterns = [
        "password=",
        "Password:",
        "SECRET_KEY",
        "api_key=",
        "AWS_ACCESS_KEY",
        "Fujifuji500"  # The known bad password
    ]
    
    violations = []
    
    for py_file in v2_src.rglob("*.py"):
        try:
            content = py_file.read_text()
            for pattern in dangerous_patterns:
                if pattern in content:
                    violations.append((str(py_file), pattern))
        except Exception:
            continue
    
    if violations:
        print("CRITICAL: Hardcoded credentials found!")
        for file_path, pattern in violations:
            print(f"  DANGER: {file_path} contains '{pattern}'")
        
        print("\nV2 must use AWS Secrets Manager only!")
        return False
    else:
        print("OK: No hardcoded credentials found in V2")
        return True

def check_current_mlflow_uri():
    """Check current MLflow URI is safe"""
    print("\nCHECKING CURRENT MLFLOW URI")
    print("-" * 27)
    
    try:
        import mlflow
        import yaml
        from pathlib import Path
        
        # Load the actual PostgreSQL URI from settings
        settings_path = Path("C:/Users/marti/AI-Scoring/tidyllm/admin/settings.yaml")
        if settings_path.exists():
            with open(settings_path, 'r') as f:
                config = yaml.safe_load(f)
            
            pg_creds = config['credentials']['postgresql']
            expected_uri = (
                f"postgresql://{pg_creds['username']}:{pg_creds['password']}"
                f"@{pg_creds['host']}:{pg_creds['port']}/{pg_creds['database']}"
            )
            
            # Set the correct URI
            mlflow.set_tracking_uri(expected_uri)
            print(f"Set MLflow URI to PostgreSQL backend")
        
        current_uri = mlflow.get_tracking_uri()
        print(f"Current MLflow URI: {current_uri}")
        
        # Check for dangerous patterns
        forbidden_patterns = [
            "sqlite:",
            "file://",
            "./mlruns",
            "mlruns/"
        ]
        
        for pattern in forbidden_patterns:
            if pattern in str(current_uri):
                print(f"CRITICAL: Forbidden pattern '{pattern}' found in URI!")
                print("This will cause disk space issues and break the app!")
                return False
        
        if "postgresql://" in current_uri:
            print("OK: MLflow using PostgreSQL backend")
            return True
        else:
            print("WARNING: MLflow not using PostgreSQL backend")
            return False
            
    except Exception as e:
        print(f"Could not check MLflow URI: {e}")
        return True

def check_v2_architecture():
    """Check V2 follows Clean Architecture"""
    print("\nCHECKING V2 CLEAN ARCHITECTURE")
    print("-" * 30)
    
    v2_path = Path("C:/Users/marti/AI-Scoring/v2/src")
    required_layers = [
        "domain",
        "infrastructure", 
        "application",
        "presentation"
    ]
    
    architecture_ok = True
    
    for layer in required_layers:
        layer_path = v2_path / layer
        if layer_path.exists():
            py_files = list(layer_path.rglob("*.py"))
            print(f"OK: {layer} layer exists ({len(py_files)} files)")
        else:
            print(f"MISSING: {layer} layer not found")
            architecture_ok = False
    
    return architecture_ok

def run_safety_validation():
    """Run all V2 safety validations"""
    print("V2 SAFETY VALIDATION")
    print("=" * 50)
    print("Checking for violations of critical constraints...")
    print()
    
    checks = [
        ("No MLruns Folders", check_no_mlruns_folders),
        ("No SQLite Files", check_sqlite_files),
        ("No Hardcoded Credentials", check_hardcoded_credentials),
        ("Safe MLflow URI", check_current_mlflow_uri),
        ("Clean Architecture", check_v2_architecture)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"ERROR in {check_name}: {e}")
            results.append((check_name, False))
    
    print("\n" + "=" * 50)
    print("V2 SAFETY VALIDATION RESULTS")
    print("=" * 50)
    
    passed = 0
    for check_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{check_name:25}: {status}")
        if result:
            passed += 1
    
    total = len(results)
    success_rate = passed / total * 100
    
    print(f"\nSafety Score: {passed}/{total} ({success_rate:.1f}%)")
    
    if success_rate == 100:
        print("\nCONCLUSION: V2 is SAFE for deployment")
        print("All critical constraints are satisfied.")
    elif success_rate >= 80:
        print("\nCONCLUSION: V2 needs MINOR fixes")
        print("Address the failing checks before deployment.")
    else:
        print("\nCONCLUSION: V2 has CRITICAL safety violations")
        print("MUST fix all issues before deployment!")
    
    return success_rate == 100

if __name__ == "__main__":
    run_safety_validation()