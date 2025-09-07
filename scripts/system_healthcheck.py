#!/usr/bin/env python3
"""
TidyLLM System Health Check - Client Demo Script
===============================================

Comprehensive health check that demonstrates the system is fully operational.
This script touches all real components and provides clear pass/fail results.

Usage:
    python scripts/system_healthcheck.py

Output:
    - Real-time health status of all components
    - Performance metrics and response times
    - Clear pass/fail indicators for client demonstrations
"""

import sys
import os
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.WARNING)  # Suppress noise for demo
logger = logging.getLogger(__name__)

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(title: str):
    """Print formatted section header"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{title.center(60)}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.END}")

def print_status(component: str, status: bool, details: str = "", response_time: float = 0):
    """Print formatted status line"""
    icon = f"{Colors.GREEN}✅ PASS{Colors.END}" if status else f"{Colors.RED}❌ FAIL{Colors.END}"
    time_str = f" ({response_time:.2f}s)" if response_time > 0 else ""
    print(f"  {icon} {component}{time_str}")
    if details:
        print(f"      {details}")

def test_infrastructure() -> Dict[str, Any]:
    """Test core infrastructure components"""
    print_header("🏗️  INFRASTRUCTURE HEALTH CHECK")
    
    results = {"overall": True, "components": {}}
    
    try:
        # Add scripts/infrastructure to path
        sys.path.insert(0, 'scripts/infrastructure')
        from start_unified_sessions import get_global_session_manager
        
        start_time = time.time()
        session_mgr = get_global_session_manager()
        health = session_mgr.get_health_summary()
        response_time = time.time() - start_time
        
        # Test each service
        services = ['s3', 'postgresql', 'bedrock']
        for service in services:
            if service in health['services']:
                service_health = health['services'][service]
                is_healthy = service_health['healthy']
                latency = service_health.get('latency_ms', 0)
                
                details = f"Latency: {latency:.1f}ms"
                if not is_healthy and service_health.get('error'):
                    details = f"Error: {service_health['error']}"
                
                print_status(f"{service.upper()} Connection", is_healthy, details)
                results["components"][service] = {
                    "healthy": is_healthy,
                    "latency_ms": latency,
                    "error": service_health.get('error')
                }
                
                if not is_healthy:
                    results["overall"] = False
            else:
                print_status(f"{service.upper()} Connection", False, "Service not found")
                results["components"][service] = {"healthy": False, "error": "Service not found"}
                results["overall"] = False
        
        # Show configuration details
        config = health['configuration']
        print(f"\n  📋 Configuration:")
        print(f"      Credential Source: {health['credential_source']}")
        print(f"      S3 Region: {config['s3_region']}")
        print(f"      S3 Bucket: {config['s3_default_bucket']}")
        print(f"      PostgreSQL: {config['postgres_host']}")
        print(f"      Bedrock Region: {config['bedrock_region']}")
        
        results["config"] = config
        results["credential_source"] = health['credential_source']
        results["response_time"] = response_time
        
    except Exception as e:
        print_status("UnifiedSessionManager", False, f"Error: {str(e)}")
        results["overall"] = False
        results["error"] = str(e)
    
    return results

def test_gateways() -> Dict[str, Any]:
    """Test all three gateways"""
    print_header("🚪 GATEWAY SYSTEM HEALTH CHECK")
    
    results = {"overall": True, "gateways": {}}
    
    try:
        # Import gateway registry
        from tidyllm.gateways.gateway_registry import get_global_registry
        
        start_time = time.time()
        registry = get_global_registry()
        registry.auto_configure()
        registry_time = time.time() - start_time
        
        print_status("Gateway Registry", True, "Auto-configuration completed", registry_time)
        
        # Test each gateway
        gateways = ['corporate_llm', 'ai_processing', 'workflow_optimizer']
        
        for gateway_name in gateways:
            start_time = time.time()
            gateway = registry.get(gateway_name)
            
            if gateway:
                # Test gateway capabilities
                try:
                    capabilities = gateway.get_capabilities() if hasattr(gateway, 'get_capabilities') else {}
                    dependencies = gateway.get_required_services() if hasattr(gateway, 'get_required_services') else []
                    response_time = time.time() - start_time
                    
                    details = f"Status: {gateway.status.value if hasattr(gateway, 'status') else 'ACTIVE'}"
                    if capabilities and 'cache_enabled' in capabilities:
                        details += f", Cache: {'ON' if capabilities['cache_enabled'] else 'OFF'}"
                    if dependencies:
                        details += f", Dependencies: {len(dependencies)}"
                    
                    print_status(f"{gateway_name.replace('_', ' ').title()} Gateway", True, details, response_time)
                    
                    results["gateways"][gateway_name] = {
                        "healthy": True,
                        "status": gateway.status.value if hasattr(gateway, 'status') else 'ACTIVE',
                        "capabilities": capabilities,
                        "dependencies": dependencies,
                        "response_time": response_time
                    }
                    
                except Exception as e:
                    print_status(f"{gateway_name.replace('_', ' ').title()} Gateway", False, f"Capability test failed: {str(e)}")
                    results["gateways"][gateway_name] = {"healthy": False, "error": str(e)}
                    results["overall"] = False
            else:
                print_status(f"{gateway_name.replace('_', ' ').title()} Gateway", False, "Gateway not available")
                results["gateways"][gateway_name] = {"healthy": False, "error": "Gateway not available"}
                results["overall"] = False
        
        # Show registry statistics
        try:
            stats = registry.get_registry_stats()
            available_services = registry.get_available_services()
            print(f"\n  📊 Registry Statistics:")
            print(f"      Available Services: {len(available_services)}")
            print(f"      Registry Stats: {stats}")
        except:
            pass  # Non-critical
            
    except Exception as e:
        print_status("Gateway Registry", False, f"Error: {str(e)}")
        results["overall"] = False
        results["error"] = str(e)
    
    return results

def test_flow_agreements() -> Dict[str, Any]:
    """Test FLOW agreement system"""
    print_header("⛓️  FLOW AGREEMENT SYSTEM CHECK")
    
    results = {"overall": True, "agreements": {}}
    
    try:
        # Import FLOW system
        sys.path.insert(0, 'tidyllm/demo-standalone')
        from flow_agreements import FlowAgreementManager, execute_flow_command
        
        start_time = time.time()
        flow_manager = FlowAgreementManager()
        manager_time = time.time() - start_time
        
        print_status("FLOW Agreement Manager", True, "Manager initialized", manager_time)
        
        # Test available agreements
        agreements = flow_manager.get_available_agreements()
        print_status("Available Agreements", len(agreements) > 0, f"Found {len(agreements)} agreements")
        
        results["agreements"]["count"] = len(agreements)
        results["agreements"]["list"] = agreements
        
        if len(agreements) == 0:
            results["overall"] = False
        
        # Test execution of a sample agreement
        if agreements:
            test_agreement = agreements[0]  # Test first available agreement
            try:
                start_time = time.time()
                result = execute_flow_command(f'[{test_agreement}]')
                execution_time = time.time() - start_time
                
                success = result.get('execution_mode') in ['real', 'simulation']
                details = f"Mode: {result.get('execution_mode', 'unknown')}, Confidence: {result.get('confidence', 0)}"
                
                print_status(f"FLOW Execution Test", success, details, execution_time)
                
                results["agreements"]["test_execution"] = {
                    "success": success,
                    "agreement": test_agreement,
                    "result": result,
                    "response_time": execution_time
                }
                
                if not success:
                    results["overall"] = False
                    
            except Exception as e:
                print_status("FLOW Execution Test", False, f"Execution failed: {str(e)}")
                results["agreements"]["test_execution"] = {"success": False, "error": str(e)}
                results["overall"] = False
        
        # Check execution history
        try:
            history = flow_manager.get_execution_history()
            recent_executions = len([h for h in history if time.time() - h.get('timestamp', 0) < 3600])
            print_status("Execution History", True, f"Recent executions: {recent_executions} in last hour")
            results["agreements"]["history_count"] = len(history)
            results["agreements"]["recent_count"] = recent_executions
        except:
            pass  # Non-critical
            
    except Exception as e:
        print_status("FLOW Agreement System", False, f"Error: {str(e)}")
        results["overall"] = False
        results["error"] = str(e)
    
    return results

def test_ai_processing() -> Dict[str, Any]:
    """Test actual AI processing through the gateway chain"""
    print_header("🤖 AI PROCESSING INTEGRATION TEST")
    
    results = {"overall": True, "processing": {}}
    
    try:
        from tidyllm.gateways.gateway_registry import get_global_registry
        from tidyllm.gateways.ai_processing_gateway import AIRequest
        from tidyllm.gateways.corporate_llm_gateway import LLMRequest
        
        registry = get_global_registry()
        registry.auto_configure()
        
        # Test Corporate LLM Gateway first
        corporate_gateway = registry.get('corporate_llm')
        if corporate_gateway:
            try:
                start_time = time.time()
                llm_request = LLMRequest(
                    prompt="System health check - respond with 'OPERATIONAL'",
                    audit_reason="System health verification",
                    user_id="system_healthcheck"
                )
                
                corp_result = corporate_gateway.execute_llm_request(llm_request)
                corp_time = time.time() - start_time
                
                success = corp_result.status.value == 'SUCCESS'
                details = f"Status: {corp_result.status.value}"
                if corp_result.data:
                    details += f", Response: {corp_result.data[:30]}..."
                
                print_status("Corporate LLM Processing", success, details, corp_time)
                
                results["processing"]["corporate_llm"] = {
                    "success": success,
                    "status": corp_result.status.value,
                    "response_time": corp_time,
                    "has_response": bool(corp_result.data)
                }
                
                if not success:
                    results["overall"] = False
                    
            except Exception as e:
                print_status("Corporate LLM Processing", False, f"Error: {str(e)}")
                results["processing"]["corporate_llm"] = {"success": False, "error": str(e)}
                results["overall"] = False
        
        # Test AI Processing Gateway
        ai_gateway = registry.get('ai_processing')
        if ai_gateway:
            try:
                start_time = time.time()
                ai_request = AIRequest(
                    prompt="System health check - respond with 'AI_OPERATIONAL'",
                    model="claude-3-sonnet",
                    temperature=0.1,
                    max_tokens=50,
                    metadata={"test": "health_check"}
                )
                
                ai_result = ai_gateway.process_ai_request(ai_request)
                ai_time = time.time() - start_time
                
                success = ai_result.status.value == 'SUCCESS'
                details = f"Status: {ai_result.status.value}"
                if ai_result.metadata:
                    backend = ai_result.metadata.get('backend', 'unknown')
                    cache_hit = ai_result.metadata.get('cache_hit', False)
                    details += f", Backend: {backend}, Cache: {'HIT' if cache_hit else 'MISS'}"
                
                print_status("AI Processing Gateway", success, details, ai_time)
                
                results["processing"]["ai_processing"] = {
                    "success": success,
                    "status": ai_result.status.value,
                    "response_time": ai_time,
                    "backend": ai_result.metadata.get('backend') if ai_result.metadata else None,
                    "cache_hit": ai_result.metadata.get('cache_hit') if ai_result.metadata else False
                }
                
                if not success:
                    results["overall"] = False
                    
            except Exception as e:
                print_status("AI Processing Gateway", False, f"Error: {str(e)}")
                results["processing"]["ai_processing"] = {"success": False, "error": str(e)}
                results["overall"] = False
        
    except Exception as e:
        print_status("AI Processing Test", False, f"Error: {str(e)}")
        results["overall"] = False
        results["error"] = str(e)
    
    return results

def test_cache_system() -> Dict[str, Any]:
    """Test caching system"""
    print_header("💾 CACHE SYSTEM CHECK")
    
    results = {"overall": True, "cache": {}}
    
    try:
        cache_dir = ".bedrock_cache"
        
        # Check cache directory exists
        if os.path.exists(cache_dir):
            print_status("Cache Directory", True, f"Found: {cache_dir}")
            
            # Get cache statistics
            cache_stats = {}
            total_files = 0
            total_size = 0
            
            for model_dir in os.listdir(cache_dir):
                model_path = os.path.join(cache_dir, model_dir)
                if os.path.isdir(model_path):
                    files = [f for f in os.listdir(model_path) if f.endswith('.json.gz')]
                    file_count = len(files)
                    size = sum(os.path.getsize(os.path.join(model_path, f)) for f in files)
                    
                    cache_stats[model_dir] = {
                        'files': file_count,
                        'size_bytes': size,
                        'size_mb': size / (1024 * 1024)
                    }
                    
                    total_files += file_count
                    total_size += size
            
            if cache_stats:
                print_status("Cache Contents", True, f"{total_files} files, {total_size / (1024 * 1024):.1f} MB")
                
                # Show breakdown by model
                for model, stats in cache_stats.items():
                    print(f"      {model}: {stats['files']} files ({stats['size_mb']:.1f} MB)")
                
                results["cache"] = {
                    "directory_exists": True,
                    "total_files": total_files,
                    "total_size_mb": total_size / (1024 * 1024),
                    "models": cache_stats
                }
            else:
                print_status("Cache Contents", True, "Empty (no cached responses yet)")
                results["cache"] = {"directory_exists": True, "total_files": 0}
                
        else:
            print_status("Cache Directory", False, f"Directory {cache_dir} not found")
            results["overall"] = False
            results["cache"] = {"directory_exists": False}
            
    except Exception as e:
        print_status("Cache System", False, f"Error: {str(e)}")
        results["overall"] = False
        results["error"] = str(e)
    
    return results

def generate_summary_report(all_results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate overall system health summary"""
    print_header("📊 SYSTEM HEALTH SUMMARY")
    
    # Calculate overall health
    component_health = {
        "Infrastructure": all_results.get("infrastructure", {}).get("overall", False),
        "Gateway System": all_results.get("gateways", {}).get("overall", False),
        "FLOW Agreements": all_results.get("flow", {}).get("overall", False),
        "AI Processing": all_results.get("ai_processing", {}).get("overall", False),
        "Cache System": all_results.get("cache", {}).get("overall", False)
    }
    
    overall_healthy = all(component_health.values())
    healthy_count = sum(component_health.values())
    total_count = len(component_health)
    
    print(f"\n  🎯 Overall System Status:")
    if overall_healthy:
        print(f"      {Colors.GREEN}{Colors.BOLD}✅ SYSTEM FULLY OPERATIONAL{Colors.END}")
        print(f"      All {total_count} components: HEALTHY")
    else:
        print(f"      {Colors.YELLOW}{Colors.BOLD}⚠️  SYSTEM PARTIALLY OPERATIONAL{Colors.END}")
        print(f"      {healthy_count}/{total_count} components healthy")
    
    print(f"\n  📋 Component Status:")
    for component, health in component_health.items():
        status = f"{Colors.GREEN}✅ HEALTHY{Colors.END}" if health else f"{Colors.RED}❌ FAILED{Colors.END}"
        print(f"      {component}: {status}")
    
    # Performance metrics
    print(f"\n  ⚡ Performance Metrics:")
    
    # Infrastructure response times
    if "infrastructure" in all_results and "response_time" in all_results["infrastructure"]:
        infra_time = all_results["infrastructure"]["response_time"]
        print(f"      Infrastructure Init: {infra_time:.2f}s")
    
    # Gateway response times
    if "gateways" in all_results and "gateways" in all_results["gateways"]:
        for gw_name, gw_data in all_results["gateways"]["gateways"].items():
            if "response_time" in gw_data:
                print(f"      {gw_name.replace('_', ' ').title()}: {gw_data['response_time']:.2f}s")
    
    # AI processing times
    if "ai_processing" in all_results and "processing" in all_results["ai_processing"]:
        for proc_name, proc_data in all_results["ai_processing"]["processing"].items():
            if "response_time" in proc_data:
                print(f"      {proc_name.replace('_', ' ').title()}: {proc_data['response_time']:.2f}s")
    
    # Configuration summary
    print(f"\n  🔧 Configuration:")
    if "infrastructure" in all_results and "credential_source" in all_results["infrastructure"]:
        print(f"      Credential Source: {all_results['infrastructure']['credential_source']}")
    
    if "infrastructure" in all_results and "config" in all_results["infrastructure"]:
        config = all_results["infrastructure"]["config"]
        print(f"      Database: {config.get('postgres_host', 'Unknown')}")
        print(f"      S3 Bucket: {config.get('s3_default_bucket', 'Unknown')}")
    
    # Create summary for return
    summary = {
        "overall_healthy": overall_healthy,
        "component_health": component_health,
        "healthy_components": healthy_count,
        "total_components": total_count,
        "timestamp": datetime.now().isoformat(),
        "system_operational": overall_healthy
    }
    
    return summary

def main():
    """Main health check execution"""
    print(f"{Colors.BLUE}{Colors.BOLD}")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║                TIDYLLM SYSTEM HEALTH CHECK               ║")
    print("║                   Client Demo Script                     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}")
    
    start_time = time.time()
    
    # Run all health checks
    all_results = {}
    
    try:
        all_results["infrastructure"] = test_infrastructure()
    except Exception as e:
        print(f"{Colors.RED}❌ Infrastructure test failed: {e}{Colors.END}")
        all_results["infrastructure"] = {"overall": False, "error": str(e)}
    
    try:
        all_results["gateways"] = test_gateways()
    except Exception as e:
        print(f"{Colors.RED}❌ Gateway test failed: {e}{Colors.END}")
        all_results["gateways"] = {"overall": False, "error": str(e)}
    
    try:
        all_results["flow"] = test_flow_agreements()
    except Exception as e:
        print(f"{Colors.RED}❌ FLOW test failed: {e}{Colors.END}")
        all_results["flow"] = {"overall": False, "error": str(e)}
    
    try:
        all_results["ai_processing"] = test_ai_processing()
    except Exception as e:
        print(f"{Colors.RED}❌ AI processing test failed: {e}{Colors.END}")
        all_results["ai_processing"] = {"overall": False, "error": str(e)}
    
    try:
        all_results["cache"] = test_cache_system()
    except Exception as e:
        print(f"{Colors.RED}❌ Cache test failed: {e}{Colors.END}")
        all_results["cache"] = {"overall": False, "error": str(e)}
    
    # Generate summary
    summary = generate_summary_report(all_results)
    
    total_time = time.time() - start_time
    
    print_header("🏁 HEALTH CHECK COMPLETE")
    print(f"  Total execution time: {total_time:.2f}s")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if summary["system_operational"]:
        print(f"\n  {Colors.GREEN}{Colors.BOLD}🎉 SYSTEM READY FOR CLIENT DEMONSTRATION{Colors.END}")
        exit_code = 0
    else:
        print(f"\n  {Colors.YELLOW}{Colors.BOLD}⚠️  SYSTEM REQUIRES ATTENTION BEFORE DEMO{Colors.END}")
        exit_code = 1
    
    # Save detailed results for debugging
    try:
        with open('system_health_report.json', 'w') as f:
            json.dump({
                "summary": summary,
                "detailed_results": all_results,
                "execution_time": total_time
            }, f, indent=2, default=str)
        print(f"\n  📄 Detailed report saved: system_health_report.json")
    except:
        pass
    
    return exit_code

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)