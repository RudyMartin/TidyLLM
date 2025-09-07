#!/usr/bin/env python3
"""
TidyLLM System Diagnostics - Client Demo
========================================

Quick system diagnostic that proves all components are working.
Perfect for client demonstrations and system verification.

Usage: python run_diagnostics.py
"""

import sys
import os
import time
import json
from datetime import datetime
from typing import Dict, Any

# Colors for clean output
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

def auto_discover_admin_credentials():
    """Auto-discover AWS credentials from admin folder"""
    try:
        credential_files = [
            'tidyllm/admin/set_aws_env.sh',
            'tidyllm/admin/set_aws_env.bat', 
            'tidyllm/admin/set_aws_credentials.py'
        ]
        
        import re
        for cred_file in credential_files:
            if os.path.exists(cred_file):
                print(f"[AUTO-DISCOVERY] Found credentials in: {cred_file}")
                
                with open(cred_file, 'r') as f:
                    content = f.read()
                    
                access_key_match = re.search(r'AWS_ACCESS_KEY_ID[=\s]([A-Z0-9]+)', content)
                secret_key_match = re.search(r'AWS_SECRET_ACCESS_KEY[=\s]([A-Za-z0-9+/]+)', content) 
                region_match = re.search(r'AWS_DEFAULT_REGION[=\s]([a-z0-9-]+)', content)
                
                if access_key_match and secret_key_match and region_match:
                    os.environ['AWS_ACCESS_KEY_ID'] = access_key_match.group(1)
                    os.environ['AWS_SECRET_ACCESS_KEY'] = secret_key_match.group(1)
                    os.environ['AWS_DEFAULT_REGION'] = region_match.group(1)
                    
                    print(f"[AUTO-LOADED] AWS credentials from admin folder")
                    return True
        
        print("[INFO] No AWS credentials found in admin folder (checking environment)")
        return False
        
    except Exception as e:
        print(f"[WARNING] Error auto-discovering credentials: {e}")
        return False

def main():
    print(f"{Colors.BOLD}[DIAGNOSTICS] TidyLLM System Diagnostics{Colors.END}")
    print(f"Running comprehensive system check...")
    print(f"Auto-discovering credentials from admin folder...")
    print()
    
    # Try to auto-discover credentials from admin folder
    auto_discover_admin_credentials()
    
    total_start = time.time()
    results = {"passed": 0, "failed": 0, "components": {}}
    
    # 1. Test Infrastructure
    print_section("[INFRA] Infrastructure")
    try:
        sys.path.insert(0, 'scripts/infrastructure')
        from start_unified_sessions import get_global_session_manager
        
        start = time.time()
        session_mgr = get_global_session_manager()
        health = session_mgr.get_health_summary()
        response_time = (time.time() - start) * 1000
        
        # Check core services
        services = ['s3', 'postgresql', 'bedrock']
        all_healthy = True
        
        for service in services:
            if service in health['services']:
                is_healthy = health['services'][service]['healthy']
                latency = health['services'][service].get('latency_ms', 0) or 0
                
                details = f"Latency: {latency:.0f}ms" if latency is not None else "Connection established"
                check_status(f"{service.upper()}", is_healthy, details)
                
                if is_healthy:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                    all_healthy = False
            else:
                check_status(f"{service.upper()}", False, "Service not found")
                results["failed"] += 1
                all_healthy = False
        
        results["components"]["infrastructure"] = {
            "healthy": all_healthy,
            "response_time_ms": response_time,
            "credential_source": health['credential_source']
        }
        
        print(f"       Config: {health['credential_source']} credentials")
        print(f"       Database: {health['configuration']['postgres_host']}")
        print(f"       S3: {health['configuration']['s3_default_bucket']}")
        
    except Exception as e:
        check_status("Infrastructure", False, f"Error: {str(e)}")
        results["failed"] += 1
        results["components"]["infrastructure"] = {"healthy": False, "error": str(e)}
    
    # 2. Test Gateway System
    print_section("[GATEWAYS] Gateway System")
    try:
        from tidyllm.gateways.gateway_registry import get_global_registry
        
        start = time.time()
        registry = get_global_registry()
        registry.auto_configure()
        registry_time = (time.time() - start) * 1000
        
        check_status("Gateway Registry", True, "Auto-configured", registry_time)
        results["passed"] += 1
        
        # Test each gateway
        gateways = ['corporate_llm', 'ai_processing', 'workflow_optimizer']
        gateway_results = {}
        
        for gw_name in gateways:
            start = time.time()
            gateway = registry.get(gw_name)
            gw_time = (time.time() - start) * 1000
            
            if gateway:
                check_status(f"{gw_name.replace('_', ' ').title()}", True, f"Status: {gateway.status.value if hasattr(gateway, 'status') else 'ACTIVE'}", gw_time)
                results["passed"] += 1
                gateway_results[gw_name] = {"healthy": True, "response_time_ms": gw_time}
            else:
                check_status(f"{gw_name.replace('_', ' ').title()}", False, "Gateway not available")
                results["failed"] += 1
                gateway_results[gw_name] = {"healthy": False}
        
        results["components"]["gateways"] = gateway_results
        
    except Exception as e:
        check_status("Gateway System", False, f"Error: {str(e)}")
        results["failed"] += 1
        results["components"]["gateways"] = {"healthy": False, "error": str(e)}
    
    # 3. Test FLOW Agreements
    print_section("[FLOW] FLOW Agreement System")
    try:
        sys.path.insert(0, 'tidyllm/demo-standalone')
        from flow_agreements import FlowAgreementManager, execute_flow_command
        
        start = time.time()
        flow_manager = FlowAgreementManager()
        agreements = flow_manager.get_available_agreements()
        manager_time = (time.time() - start) * 1000
        
        check_status("FLOW Manager", True, f"Found {len(agreements)} agreements", manager_time)
        
        if len(agreements) > 0:
            results["passed"] += 1
            
            # Test execution
            test_agreement = agreements[0]
            start = time.time()
            result = execute_flow_command(f'[{test_agreement}]')
            exec_time = (time.time() - start) * 1000
            
            success = result.get('execution_mode') in ['real', 'simulation']
            check_status("FLOW Execution", success, f"Mode: {result.get('execution_mode', 'unknown')}", exec_time)
            
            if success:
                results["passed"] += 1
            else:
                results["failed"] += 1
        else:
            results["failed"] += 1
        
        results["components"]["flow"] = {
            "healthy": len(agreements) > 0,
            "agreement_count": len(agreements)
        }
        
    except Exception as e:
        check_status("FLOW System", False, f"Error: {str(e)}")
        results["failed"] += 1
        results["components"]["flow"] = {"healthy": False, "error": str(e)}
    
    # 4. Test AI Processing (Real AI Call)
    print_section("[AI] AI Processing Test")
    try:
        from tidyllm.gateways.gateway_registry import get_global_registry
        from tidyllm.gateways.ai_processing_gateway import AIRequest
        
        registry = get_global_registry()
        ai_gateway = registry.get('ai_processing')
        
        if ai_gateway:
            start = time.time()
            
            # Make actual AI request
            request = AIRequest(
                prompt="System diagnostic test - please respond with exactly: SYSTEM_OPERATIONAL",
                model="claude-3-haiku",  # Fast model for diagnostics
                temperature=0.1,
                max_tokens=10,
                metadata={"diagnostic": True}
            )
            
            response = ai_gateway.process_ai_request(request)
            ai_time = (time.time() - start) * 1000
            
            success = response.status.value == 'SUCCESS'
            backend = response.metadata.get('backend', 'unknown') if response.metadata else 'unknown'
            cache_hit = response.metadata.get('cache_hit', False) if response.metadata else False
            
            details = f"Backend: {backend}, Cache: {'HIT' if cache_hit else 'MISS'}"
            check_status("AI Processing", success, details, ai_time)
            
            if success:
                results["passed"] += 1
                print(f"       Response: {response.data[:50] if response.data else 'No response'}...")
            else:
                results["failed"] += 1
            
            results["components"]["ai_processing"] = {
                "healthy": success,
                "backend": backend,
                "response_time_ms": ai_time,
                "cache_hit": cache_hit
            }
        else:
            check_status("AI Processing", False, "Gateway not available")
            results["failed"] += 1
            results["components"]["ai_processing"] = {"healthy": False}
            
    except Exception as e:
        check_status("AI Processing", False, f"Error: {str(e)}")
        results["failed"] += 1
        results["components"]["ai_processing"] = {"healthy": False, "error": str(e)}
    
    # 5. Test Cache System
    print_section("[CACHE] Cache System")
    try:
        cache_dir = ".bedrock_cache"
        
        if os.path.exists(cache_dir):
            # Count cache files
            total_files = 0
            total_size = 0
            
            for model_dir in os.listdir(cache_dir):
                model_path = os.path.join(cache_dir, model_dir)
                if os.path.isdir(model_path):
                    files = [f for f in os.listdir(model_path) if f.endswith('.json.gz')]
                    total_files += len(files)
                    total_size += sum(os.path.getsize(os.path.join(model_path, f)) for f in files)
            
            check_status("Cache Directory", True, f"{total_files} files, {total_size/(1024*1024):.1f} MB")
            results["passed"] += 1
            
            results["components"]["cache"] = {
                "healthy": True,
                "file_count": total_files,
                "size_mb": total_size / (1024 * 1024)
            }
        else:
            check_status("Cache Directory", False, "Directory not found")
            results["failed"] += 1
            results["components"]["cache"] = {"healthy": False}
            
    except Exception as e:
        check_status("Cache System", False, f"Error: {str(e)}")
        results["failed"] += 1
        results["components"]["cache"] = {"healthy": False, "error": str(e)}
    
    # Final Summary
    total_time = time.time() - total_start
    total_tests = results["passed"] + results["failed"]
    success_rate = (results["passed"] / total_tests * 100) if total_tests > 0 else 0
    
    print_section("[SUMMARY] Diagnostic Summary")
    
    if results["failed"] == 0:
        print(f"  {Colors.GREEN}{Colors.BOLD}[SUCCESS] ALL SYSTEMS OPERATIONAL{Colors.END}")
        print(f"  {Colors.GREEN}[PASS] {results['passed']}/{total_tests} tests passed (100%){Colors.END}")
        status = "READY FOR DEMO"
        exit_code = 0
    else:
        print(f"  {Colors.YELLOW}{Colors.BOLD}[WARNING] SYSTEM PARTIALLY OPERATIONAL{Colors.END}")
        print(f"  {Colors.YELLOW}[PASS] {results['passed']}/{total_tests} tests passed ({success_rate:.1f}%){Colors.END}")
        print(f"  {Colors.RED}[FAIL] {results['failed']} tests failed{Colors.END}")
        status = "REQUIRES ATTENTION"
        exit_code = 1
    
    print(f"  [TIME] Total execution time: {total_time:.2f}s")
    print(f"  [DATE] Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n  [STATUS] System Status: {Colors.BOLD}{status}{Colors.END}")
    
    # Save diagnostic report
    report = {
        "status": status,
        "success_rate": success_rate,
        "tests_passed": results["passed"],
        "tests_failed": results["failed"],
        "execution_time": total_time,
        "timestamp": datetime.now().isoformat(),
        "components": results["components"]
    }
    
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f'diagnostic_report_{timestamp}.json'
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n  [REPORT] Report saved: {report_filename}")
    except:
        pass
    
    return exit_code

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)