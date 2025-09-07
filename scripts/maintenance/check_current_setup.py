#!/usr/bin/env python3
"""
Current Setup Checker - See what works right now
================================================
Tests what components are available in current state.
"""

import os
import sys
import subprocess
from datetime import datetime

class CurrentSetupChecker:
    def __init__(self):
        self.results = {"passed": 0, "failed": 0, "components": []}
        
    def check_status(self, name, success, details=""):
        status = "[PASS]" if success else "[FAIL]"
        print(f"  {status} {name}")
        if details:
            print(f"       {details}")
        
        self.results["components"].append({
            "name": name,
            "status": success,
            "details": details
        })
        
        if success:
            self.results["passed"] += 1
        else:
            self.results["failed"] += 1
    
    def check_python_environment(self):
        """Check Python and basic dependencies"""
        print("\n=== PYTHON ENVIRONMENT ===")
        
        # Python version
        version = sys.version.split()[0]
        python_ok = sys.version_info >= (3, 11)
        self.check_status("Python Version", python_ok, f"Version {version} (need 3.11+)")
        
        # Check pip
        try:
            import pip
            self.check_status("Pip", True, f"Available")
        except ImportError:
            self.check_status("Pip", False, "Not found")
        
        # Check basic dependencies
        deps = [("python-dotenv", "dotenv"), ("PyYAML", "yaml")]
        for dep_name, import_name in deps:
            try:
                __import__(import_name)
                self.check_status(f"{dep_name}", True, "Installed")
            except ImportError:
                self.check_status(f"{dep_name}", False, "Not installed")
    
    def check_aws_tools(self):
        """Check AWS CLI and boto3"""
        print("\n=== AWS TOOLS ===")
        
        # AWS CLI
        try:
            result = subprocess.run(['aws', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                self.check_status("AWS CLI", True, result.stdout.strip())
            else:
                self.check_status("AWS CLI", False, "Command failed")
        except FileNotFoundError:
            self.check_status("AWS CLI", False, "Not installed - need to install AWS CLI")
        
        # Boto3
        try:
            import boto3
            self.check_status("Boto3", True, "Available for AWS SDK")
        except ImportError:
            self.check_status("Boto3", False, "Not installed - need: pip install boto3")
    
    def check_database_tools(self):
        """Check PostgreSQL client tools"""
        print("\n=== DATABASE TOOLS ===")
        
        # PostgreSQL client
        try:
            result = subprocess.run(['psql', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                self.check_status("PostgreSQL Client", True, result.stdout.strip())
            else:
                self.check_status("PostgreSQL Client", False, "Command failed")
        except FileNotFoundError:
            self.check_status("PostgreSQL Client", False, "Not installed - need PostgreSQL client")
        
        # psycopg2
        try:
            import psycopg2
            self.check_status("psycopg2", True, "Available for PostgreSQL connections")
        except ImportError:
            self.check_status("psycopg2", False, "Not installed - need: pip install psycopg2-binary")
    
    def check_tidyllm_components(self):
        """Check TidyLLM specific components"""
        print("\n=== TIDYLLM COMPONENTS ===")
        
        # Check if TidyLLM directories exist
        tidyllm_dirs = ["tidyllm", "scripts", "tidyllm/gateways"]
        for dir_name in tidyllm_dirs:
            if os.path.exists(dir_name):
                files_count = len([f for f in os.listdir(dir_name) if f.endswith('.py')])
                self.check_status(f"{dir_name}/ directory", True, f"{files_count} Python files")
            else:
                self.check_status(f"{dir_name}/ directory", False, "Directory not found")
        
        # Check for key files
        key_files = [
            "settings.yaml",
            ".env", 
            "run_diagnostics.py",
            "tidyllm/demo-standalone/flow_agreements.py"
        ]
        
        for file_path in key_files:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                self.check_status(f"{file_path}", True, f"{size} bytes")
            else:
                self.check_status(f"{file_path}", False, "File not found")
        
        # Check cache directory
        cache_dir = ".bedrock_cache"
        if os.path.exists(cache_dir):
            cache_files = sum([len(files) for r, d, files in os.walk(cache_dir)])
            self.check_status("Cache Directory", True, f"{cache_files} cached files")
        else:
            self.check_status("Cache Directory", False, "Not created yet")
    
    def check_flow_agreements(self):
        """Test FLOW agreement system if available"""
        print("\n=== FLOW AGREEMENT SYSTEM ===")
        
        try:
            sys.path.insert(0, 'tidyllm/demo-standalone')
            from flow_agreements import FlowAgreementManager
            
            manager = FlowAgreementManager()
            agreements = manager.get_available_agreements()
            
            self.check_status("FLOW Manager", True, f"Loaded successfully")
            self.check_status("FLOW Agreements", len(agreements) > 0, f"Found {len(agreements)} agreements")
            
            if agreements:
                print(f"       Available: {', '.join(agreements[:3])}...")
                
        except Exception as e:
            self.check_status("FLOW Agreement System", False, f"Error: {str(e)}")
    
    def generate_report(self):
        """Generate final report"""
        total = self.results["passed"] + self.results["failed"]
        success_rate = (self.results["passed"] / total * 100) if total > 0 else 0
        
        print(f"\n{'='*60}")
        print("CURRENT SETUP STATUS REPORT")
        print(f"{'='*60}")
        
        if self.results["failed"] == 0:
            print(f"[SUCCESS] ALL AVAILABLE COMPONENTS WORKING")
            status = "READY FOR AWS SETUP"
        elif success_rate >= 70:
            print(f"[WARNING] MOSTLY READY - {self.results['failed']} components need attention")
            status = "NEEDS MINOR FIXES"
        else:
            print(f"[FAIL] MULTIPLE ISSUES - {self.results['failed']} components failed")
            status = "NEEDS MAJOR SETUP"
        
        print(f"Passed: {self.results['passed']}/{total} ({success_rate:.1f}%)")
        print(f"Status: {status}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Next steps
        print(f"\nNEXT STEPS:")
        if self.results["failed"] == 0:
            print("1. Install AWS CLI: https://aws.amazon.com/cli/")
            print("2. Configure AWS credentials: aws configure")
            print("3. Run the full setup guide")
        else:
            failed_components = [c["name"] for c in self.results["components"] if not c["status"]]
            print("Fix these components first:")
            for i, component in enumerate(failed_components, 1):
                print(f"{i}. {component}")
        
        return success_rate >= 70

def main():
    print("[DIAGNOSTICS] TidyLLM Current Setup Checker")
    print("Checking what works in current environment...")
    
    checker = CurrentSetupChecker()
    
    # Run all checks
    checker.check_python_environment()
    checker.check_aws_tools()
    checker.check_database_tools()
    checker.check_tidyllm_components()
    checker.check_flow_agreements()
    
    # Generate report
    ready = checker.generate_report()
    
    return 0 if ready else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)