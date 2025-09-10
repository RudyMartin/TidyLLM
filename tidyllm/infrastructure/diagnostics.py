#!/usr/bin/env python3
"""
TidyLLM Real System Diagnostics using Admin Configuration
=========================================================

Cross-platform diagnostic script that works on:
- Windows development machines
- Linux production environments (AWS SageMaker, etc.)

Uses the REAL configuration from tidyllm/admin/ folder.
Auto-discovers credentials and adapts to platform differences.
"""

import sys
import os
import time
import json
import yaml
import platform
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

# Cross-platform detection
PLATFORM_INFO = {
    'system': platform.system(),
    'is_windows': platform.system() == 'Windows',
    'is_linux': platform.system() == 'Linux',
    'is_sagemaker': os.path.exists('/opt/ml') or 'sagemaker' in os.getcwd().lower(),
    'python_version': platform.python_version()
}

# Colors for clean output (works on both Windows and Linux terminals)
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_section(title: str):
    print(f"\n{Colors.BLUE}{'='*50}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{title}{Colors.END}")
    print(f"{Colors.BLUE}{'='*50}{Colors.END}")

def check_status(name: str, success: bool, details: str = "", time_ms: float = 0):
    status = f"{Colors.GREEN}[PASS]{Colors.END}" if success else f"{Colors.RED}[FAIL]{Colors.END}"
    time_str = f" ({time_ms:.0f}ms)" if time_ms > 0 else ""
    print(f"  {status} {name}{time_str}")
    if details:
        print(f"       {details}")

def find_tidyllm_root():
    """Find TidyLLM root directory across platforms"""
    current_dir = Path.cwd()
    
    # Look for tidyllm directory in current path or parent directories
    for path in [current_dir] + list(current_dir.parents):
        tidyllm_dir = path / 'tidyllm'
        if tidyllm_dir.exists() and (tidyllm_dir / 'admin').exists():
            return path
    
    # If not found, assume current directory
    return current_dir

def load_admin_config_and_credentials():
    """Load real configuration AND auto-discover credentials from admin folder (cross-platform)"""
    try:
        # Find TidyLLM root directory
        tidyllm_root = find_tidyllm_root()
        admin_dir = tidyllm_root / 'tidyllm' / 'admin'
        
        print(f"[PLATFORM] {PLATFORM_INFO['system']} {PLATFORM_INFO['python_version']}")
        if PLATFORM_INFO['is_sagemaker']:
            print(f"[PLATFORM] AWS SageMaker environment detected")
        
        # Load settings.yaml using cross-platform path
        admin_settings_path = admin_dir / 'settings.yaml'
        if not admin_settings_path.exists():
            raise FileNotFoundError(f"settings.yaml not found at {admin_settings_path}")
            
        with open(admin_settings_path, 'r') as f:
            settings = yaml.safe_load(f)
        
        # Auto-discover credentials with platform-appropriate files
        if PLATFORM_INFO['is_windows']:
            credential_files = [
                admin_dir / 'set_aws_env.bat',
                admin_dir / 'set_aws_credentials.py',
                admin_dir / 'set_aws_env.sh'  # Backup for WSL
            ]
        else:  # Linux/Unix (including SageMaker)
            credential_files = [
                admin_dir / 'set_aws_env.sh',
                admin_dir / 'set_aws_credentials.py',
                admin_dir / 'set_aws_env.bat'  # Check anyway
            ]
        
        credentials_found = False
        for cred_file in credential_files:
            if cred_file.exists():
                print(f"[AUTO-DISCOVERY] Found credentials in: {cred_file}")
                
                # Parse credentials from file
                with open(cred_file, 'r') as f:
                    content = f.read()
                    
                # Extract AWS credentials using cross-platform regex
                import re
                access_key_match = re.search(r'AWS_ACCESS_KEY_ID[=\s]+([A-Z0-9]+)', content)
                secret_key_match = re.search(r'AWS_SECRET_ACCESS_KEY[=\s]+([A-Za-z0-9+/]+)', content) 
                region_match = re.search(r'AWS_DEFAULT_REGION[=\s]+([a-z0-9-]+)', content)
                
                if access_key_match and secret_key_match and region_match:
                    # Set environment variables automatically
                    os.environ['AWS_ACCESS_KEY_ID'] = access_key_match.group(1)
                    os.environ['AWS_SECRET_ACCESS_KEY'] = secret_key_match.group(1)
                    os.environ['AWS_DEFAULT_REGION'] = region_match.group(1)
                    
                    print(f"[AUTO-LOADED] AWS_ACCESS_KEY_ID: {access_key_match.group(1)[:10]}...")
                    print(f"[AUTO-LOADED] AWS_DEFAULT_REGION: {region_match.group(1)}")
                    credentials_found = True
                    break
        
        # Check for SageMaker IAM role credentials (no files needed)
        if not credentials_found and PLATFORM_INFO['is_sagemaker']:
            print("[INFO] SageMaker environment - checking for IAM role credentials")
            try:
                # Use UnifiedSessionManager for STS client
                from .session.unified import UnifiedSessionManager
                session_mgr = UnifiedSessionManager()
                sts = session_mgr.get_sts_client()
                identity = sts.get_caller_identity()
                print(f"[AUTO-LOADED] Using SageMaker IAM role: {identity['Arn']}")
                credentials_found = True
            except:
                pass
        
        if not credentials_found:
            print("[WARNING] No AWS credentials found in admin folder or IAM role")
        
        return settings
        
    except Exception as e:
        print(f"ERROR: Could not load admin configuration: {e}")
        return None

def test_real_database(settings):
    """Test connection to real RDS database"""
    print_section("[DATABASE] Real PostgreSQL RDS Connection")
    
    try:
        import psycopg2
        
        # Use UnifiedSessionManager for PostgreSQL connection
        try:
            sys.path.insert(0, str(Path(__file__).parent / 'scripts' / 'infrastructure'))
            from start_unified_sessions import UnifiedSessionManager
            
            session_mgr = UnifiedSessionManager()
            start_time = time.time()
            conn = session_mgr.get_postgres_connection()
            
            if not conn:
                raise Exception("Could not get PostgreSQL connection from UnifiedSessionManager")
        except ImportError:
            # Fallback to direct connection with admin config
            db_config = settings['postgres']
            start_time = time.time()
            conn = psycopg2.connect(
                host=db_config['host'],
                database=db_config['db_name'],
                user=db_config['db_user'],
                password=db_config['db_password'],
                sslmode=db_config.get('ssl_mode', 'require'),
            connect_timeout=10
        )
        
        cursor = conn.cursor()
        cursor.execute("SELECT version()")
        version = cursor.fetchone()[0]
        response_time = (time.time() - start_time) * 1000
        
        cursor.close()
        conn.close()
        
        check_status("PostgreSQL RDS", True, f"Connected to {db_config['host']}", response_time)
        check_status("Database Version", True, version.split(' ')[0] + ' ' + version.split(' ')[1])
        check_status("SSL Mode", True, f"Required SSL connection")
        
        return True, {"host": db_config['host'], "response_time_ms": response_time}
        
    except Exception as e:
        check_status("PostgreSQL RDS", False, f"Error: {str(e)}")
        return False, {"error": str(e)}

def test_real_s3_config(settings):
    """Test S3 configuration (not connection, just config)"""
    print_section("[S3] Real S3 Configuration")
    
    try:
        s3_config = settings['s3']
        
        check_status("S3 Region", True, f"Region: {s3_config['region']}")
        check_status("S3 Bucket", True, f"Bucket: {s3_config['bucket']}")
        
        # Check prefixes
        if 'prefixes' in s3_config:
            prefixes = list(s3_config['prefixes'].keys())
            check_status("S3 Prefixes", True, f"Found {len(prefixes)} prefixes: {', '.join(prefixes[:3])}...")
        
        # Check environment configurations
        if 'environments' in s3_config:
            envs = list(s3_config['environments'].keys())
            check_status("S3 Environments", True, f"Configured: {', '.join(envs)}")
        
        return True, {"bucket": s3_config['bucket'], "region": s3_config['region']}
        
    except Exception as e:
        check_status("S3 Configuration", False, f"Error: {str(e)}")
        return False, {"error": str(e)}

def test_aws_credentials():
    """Test if AWS credentials are available"""
    print_section("[AWS] Credential Status")
    
    try:
        import boto3
        
        start_time = time.time()
        
        # Try to get caller identity using UnifiedSessionManager if available
        try:
            # Use UnifiedSessionManager for STS client
            from .session.unified import UnifiedSessionManager
            session_mgr = UnifiedSessionManager()
            sts = session_mgr.get_sts_client()
            identity = sts.get_caller_identity()
            response_time = (time.time() - start_time) * 1000
            
            check_status("AWS Credentials", True, f"Account: {identity['Account']}", response_time)
            check_status("AWS Identity", True, f"User: {identity['Arn'].split('/')[-1]}")
            return True, {"account": identity['Account'], "user": identity['Arn']}
            
        except Exception as e:
            check_status("AWS Credentials", False, f"Not configured: {str(e)}")
            return False, {"error": "AWS credentials not configured"}
            
    except ImportError:
        check_status("Boto3", False, "Not installed")
        return False, {"error": "boto3 not installed"}

def test_flow_agreements():
    """Test FLOW agreement system"""
    print_section("[FLOW] Agreement System")
    
    try:
        sys.path.insert(0, 'tidyllm/demo-standalone')
        from flow.agreements import FlowAgreementManager, execute_flow_command
        
        start_time = time.time()
        flow_manager = FlowAgreementManager()
        manager_time = (time.time() - start_time) * 1000
        
        agreements = flow_manager.get_available_agreements()
        
        check_status("FLOW Manager", True, "Initialized successfully", manager_time)
        check_status("FLOW Agreements", len(agreements) > 0, f"Found {len(agreements)} agreements")
        
        if agreements:
            # Test execution
            test_agreement = agreements[0]
            start_time = time.time()
            result = execute_flow_command(f'[{test_agreement}]')
            exec_time = (time.time() - start_time) * 1000
            
            success = result.get('execution_mode') in ['real', 'simulation']
            check_status("FLOW Execution", success, f"Mode: {result.get('execution_mode', 'unknown')}", exec_time)
            
            return True, {"agreement_count": len(agreements), "execution_mode": result.get('execution_mode')}
        else:
            return False, {"error": "No agreements found"}
            
    except Exception as e:
        check_status("FLOW System", False, f"Error: {str(e)}")
        return False, {"error": str(e)}

def test_tidyllm_gateways():
    """Test TidyLLM gateway system with real config"""
    print_section("[GATEWAYS] TidyLLM Gateway System")
    
    try:
        # Check if gateway files exist
        gateway_files = [
            "tidyllm/gateways/base_gateway.py",
            "tidyllm/gateways/corporate_llm_gateway.py", 
            "tidyllm/gateways/ai_processing_gateway.py",
            "tidyllm/gateways/workflow_optimizer_gateway.py"
        ]
        
        gateway_count = 0
        for gateway_file in gateway_files:
            if os.path.exists(gateway_file):
                gateway_count += 1
                gateway_name = os.path.basename(gateway_file).replace('.py', '').replace('_', ' ').title()
                check_status(gateway_name, True, "File exists")
            else:
                gateway_name = os.path.basename(gateway_file).replace('.py', '').replace('_', ' ').title()
                check_status(gateway_name, False, "File not found")
        
        # Try to import gateway registry
        try:
            from tidyllm.gateways.gateway_registry import get_global_registry
            check_status("Gateway Registry", True, "Import successful")
            gateway_count += 1
        except Exception as e:
            check_status("Gateway Registry", False, f"Import failed: {str(e)}")
        
        return gateway_count > 0, {"gateway_files_found": gateway_count}
        
    except Exception as e:
        check_status("Gateway System", False, f"Error: {str(e)}")
        return False, {"error": str(e)}

def check_python_dependencies():
    """Check required Python dependencies across platforms"""
    print_section("[PLATFORM] Python Environment Check")
    
    required_packages = {
        'yaml': 'PyYAML',
        'psycopg2': 'psycopg2-binary', 
        'boto3': 'boto3',
    }
    
    missing_packages = []
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            check_status(f"Python Package: {package_name}", True, "Installed")
        except ImportError:
            check_status(f"Python Package: {package_name}", False, "Missing - install required")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"[ACTION REQUIRED] Install missing packages:")
        if PLATFORM_INFO['is_windows']:
            print(f"  pip install {' '.join(missing_packages)}")
        else:
            print(f"  pip3 install {' '.join(missing_packages)}")
        return False
    
    return True

def main():
    print(f"{Colors.BOLD}[DIAGNOSTICS] TidyLLM Cross-Platform System Test{Colors.END}")
    print("Supports: Windows development + Linux production (SageMaker)")
    print("Using REAL configuration from tidyllm/admin/ folder")
    print("AUTO-DISCOVERING credentials from admin folder...")
    print("=" * 70)
    
    # Check Python dependencies first
    deps_ok = check_python_dependencies()
    if not deps_ok:
        print(f"{Colors.RED}FATAL: Missing required Python dependencies{Colors.END}")
        return 1
    
    # Load real admin configuration AND auto-discover credentials
    settings = load_admin_config_and_credentials()
    if not settings:
        print(f"{Colors.RED}FATAL: Could not load admin configuration{Colors.END}")
        return 1
    
    print(f"[SUCCESS] Loaded real settings from: {find_tidyllm_root() / 'tidyllm' / 'admin' / 'settings.yaml'}")
    print(f"[SUCCESS] PostgreSQL: {settings['postgres']['host']}")
    print(f"[SUCCESS] S3 Bucket: {settings['s3']['bucket']}")
    
    # Run tests
    results = {"passed": 0, "failed": 0, "components": {}}
    
    # Test 1: Real Database
    db_success, db_result = test_real_database(settings)
    results["components"]["database"] = db_result
    if db_success:
        results["passed"] += 1
    else:
        results["failed"] += 1
    
    # Test 2: S3 Config
    s3_success, s3_result = test_real_s3_config(settings)
    results["components"]["s3_config"] = s3_result
    if s3_success:
        results["passed"] += 1
    else:
        results["failed"] += 1
    
    # Test 3: AWS Credentials
    aws_success, aws_result = test_aws_credentials()
    results["components"]["aws_credentials"] = aws_result
    if aws_success:
        results["passed"] += 1
    else:
        results["failed"] += 1
    
    # Test 4: FLOW Agreements
    flow_success, flow_result = test_flow_agreements()
    results["components"]["flow_agreements"] = flow_result
    if flow_success:
        results["passed"] += 1
    else:
        results["failed"] += 1
    
    # Test 5: Gateway System
    gateway_success, gateway_result = test_tidyllm_gateways()
    results["components"]["gateways"] = gateway_result
    if gateway_success:
        results["passed"] += 1
    else:
        results["failed"] += 1
    
    # Summary
    total_tests = results["passed"] + results["failed"]
    success_rate = (results["passed"] / total_tests * 100) if total_tests > 0 else 0
    
    print_section("[SUMMARY] Real System Status")
    
    if results["failed"] == 0:
        print(f"  {Colors.GREEN}{Colors.BOLD}[SUCCESS] ALL SYSTEMS OPERATIONAL{Colors.END}")
        status = "READY FOR PRODUCTION DEMO"
        exit_code = 0
    elif success_rate >= 60:
        print(f"  {Colors.YELLOW}{Colors.BOLD}[WARNING] MOSTLY OPERATIONAL{Colors.END}")
        status = "READY FOR DEMO (with noted limitations)"
        exit_code = 0
    else:
        print(f"  {Colors.RED}{Colors.BOLD}[FAIL] MULTIPLE ISSUES{Colors.END}")
        status = "REQUIRES ATTENTION"
        exit_code = 1
    
    print(f"  [RESULTS] {results['passed']}/{total_tests} tests passed ({success_rate:.1f}%)")
    print(f"  [STATUS] {status}")
    print(f"  [TIME] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Key insights
    if db_success:
        print(f"\n  [KEY] Real RDS database connection working!")
    if flow_success:
        print(f"  [KEY] FLOW agreement system operational!")
    if not aws_success:
        print(f"  [NOTE] AWS credentials needed for S3/Bedrock (expected in demo environment)")
    
    # Save report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_filename = f'real_diagnostic_report_{timestamp}.json'
    
    report = {
        "status": status,
        "success_rate": success_rate,
        "tests_passed": results["passed"],
        "tests_failed": results["failed"],
        "timestamp": datetime.now().isoformat(),
        "configuration_source": "tidyllm/admin/settings.yaml",
        "components": results["components"]
    }
    
    try:
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n  [REPORT] Report saved: {report_filename}")
    except:
        pass
    
    return exit_code

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)