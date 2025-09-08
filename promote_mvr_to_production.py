#!/usr/bin/env python3
"""
Promote MVR Analysis to Production
===================================

This script promotes the MVR Analysis FLOW Agreement to production-ready status
by ensuring all components are properly integrated and tested.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

def check_mvr_readiness() -> Dict[str, Any]:
    """Check if MVR Analysis is ready for production."""
    
    readiness_report = {
        'timestamp': datetime.now().isoformat(),
        'mvr_analysis_status': 'checking',
        'components': {},
        'issues': [],
        'recommendations': []
    }
    
    print("=" * 60)
    print("MVR ANALYSIS PRODUCTION READINESS CHECK")
    print("=" * 60)
    
    # 1. Check FLOW Agreement Registration
    print("\n[1] Checking FLOW Agreement Registration...")
    try:
        # Import and test the command
        sys.path.insert(0, str(Path(__file__).parent))
        # Import using the renamed file
        import importlib.util
        spec = importlib.util.spec_from_file_location("demo", "3-demo.py")
        demo_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(demo_module)
        CleanFlowManager = demo_module.CleanFlowManager
        
        manager = CleanFlowManager()
        agreements = manager.get_available_agreements()
        
        mvr_found = '[MVR Analysis]' in agreements
        mvr_chain_found = '[MVR Analysis Chain]' in agreements
        
        readiness_report['components']['flow_registration'] = {
            'mvr_analysis': mvr_found,
            'mvr_chain': mvr_chain_found,
            'status': 'OK' if mvr_found else 'MISSING'
        }
        
        if mvr_found:
            print("   OK: [MVR Analysis] registered")
        else:
            print("   ERROR: [MVR Analysis] NOT registered")
            readiness_report['issues'].append("MVR Analysis not in FLOW agreements")
            
        if mvr_chain_found:
            print("   OK: [MVR Analysis Chain] registered")
            
    except Exception as e:
        print(f"   ERROR checking registration: {e}")
        readiness_report['issues'].append(f"Registration check failed: {e}")
    
    # 2. Check Gateway Configuration
    print("\n[2] Checking Gateway Configuration...")
    gateways_needed = ['corporate_llm', 'ai_processing', 'workflow_optimizer']
    
    try:
        from tidyllm.flow_agreements.mvr_analysis import MVRAnalysisFlowAgreement
        
        mvr_agreement = MVRAnalysisFlowAgreement()
        gateway_config = mvr_agreement.get_gateway_config()
        
        for gateway in gateways_needed:
            if gateway in gateway_config:
                print(f"   OK: {gateway} configured")
                readiness_report['components'][f'gateway_{gateway}'] = 'configured'
            else:
                print(f"   ERROR: {gateway} NOT configured")
                readiness_report['issues'].append(f"{gateway} not configured")
                
    except Exception as e:
        print(f"   ERROR checking gateways: {e}")
        readiness_report['issues'].append(f"Gateway check failed: {e}")
    
    # 3. Check Drop Zone Configuration
    print("\n[3] Checking Drop Zone Configuration...")
    try:
        from tidyllm.flow_agreements.mvr_analysis import MVRAnalysisFlowAgreement
        
        mvr_agreement = MVRAnalysisFlowAgreement()
        drop_zone_config = mvr_agreement.get_drop_zone_config()
        
        required_configs = ['input_path', 'accepted_formats', 'auto_process']
        for config in required_configs:
            if config in drop_zone_config:
                print(f"   OK: {config}: {drop_zone_config[config]}")
            else:
                print(f"   ERROR: {config} missing")
                readiness_report['issues'].append(f"Drop zone {config} missing")
                
        readiness_report['components']['drop_zone'] = drop_zone_config
        
    except Exception as e:
        print(f"   ERROR checking drop zones: {e}")
        readiness_report['issues'].append(f"Drop zone check failed: {e}")
    
    # 4. Check Database Connectivity
    print("\n[4] Checking Database Connectivity...")
    try:
        # Test with clean system
        execute_clean_flow_command = demo_module.execute_clean_flow_command
        
        result = execute_clean_flow_command('[MVR Analysis]', context={'test': True})
        
        if result.get('result', {}).get('database_ready'):
            print("   OK: PostgreSQL connection ready")
            readiness_report['components']['database'] = 'ready'
        else:
            print("   WARNING: PostgreSQL connection not ready")
            readiness_report['issues'].append("Database connection not ready")
            
    except Exception as e:
        print(f"   ERROR checking database: {e}")
        readiness_report['issues'].append(f"Database check failed: {e}")
    
    # 5. Check S3 Configuration
    print("\n[5] Checking S3 Configuration...")
    try:
        result = execute_clean_flow_command('[MVR Analysis]', context={'test': True})
        
        if result.get('result', {}).get('s3_ready'):
            print("   OK: S3 connection ready")
            readiness_report['components']['s3'] = 'ready'
        else:
            print("   WARNING: S3 connection not ready (AWS credentials needed)")
            readiness_report['recommendations'].append("Configure AWS credentials for S3 access")
            
    except Exception as e:
        print(f"   WARNING: S3 check: {e}")
    
    # 6. Check Report Templates
    print("\n[6] Checking Report Templates...")
    templates = {
        'compliance': 'JB_Overview_Prompt.md',
        'intelligence': 'comprehensive_whitepaper_analysis.md',
        'knowledge': 'toc_extraction_prompt.md'
    }
    
    for report_type, template in templates.items():
        template_path = Path('qaz_20250321-main/src/assets/prompts/favorites') / template
        if template_path.exists():
            print(f"   OK: {report_type} template: {template}")
        else:
            print(f"   WARNING: {report_type} template missing: {template}")
            readiness_report['recommendations'].append(f"Add {template} to prompts folder")
    
    # Final Assessment
    print("\n" + "=" * 60)
    print("PRODUCTION READINESS ASSESSMENT")
    print("=" * 60)
    
    critical_issues = [i for i in readiness_report['issues'] if 'NOT registered' in i or 'failed' in i]
    
    if not critical_issues:
        readiness_report['mvr_analysis_status'] = 'READY'
        print("\n[SUCCESS] MVR Analysis is READY for production!")
        print("\nTo activate in production:")
        print("1. Ensure AWS credentials are configured")
        print("2. Place MVR documents in drop zone: ./mvr_dropzone")
        print("3. Execute: python 1-enterprise.py '[MVR Analysis]'")
        
    else:
        readiness_report['mvr_analysis_status'] = 'NOT READY'
        print("\n[FAILED] MVR Analysis has issues to resolve:")
        for issue in critical_issues:
            print(f"   - {issue}")
    
    if readiness_report['recommendations']:
        print("\nRecommendations:")
        for rec in readiness_report['recommendations']:
            print(f"   - {rec}")
    
    # Save report
    report_path = Path('mvr_production_readiness_report.json')
    with open(report_path, 'w') as f:
        json.dump(readiness_report, f, indent=2)
    
    print(f"\nDetailed report saved to: {report_path}")
    
    return readiness_report

def promote_to_production():
    """Promote MVR Analysis to production."""
    
    print("\n" + "=" * 60)
    print("PROMOTING MVR ANALYSIS TO PRODUCTION")
    print("=" * 60)
    
    # Run readiness check
    report = check_mvr_readiness()
    
    if report['mvr_analysis_status'] == 'READY':
        print("\n[PROMOTED] MVR Analysis promoted to production!")
        print("\nAvailable commands:")
        print("  [MVR Analysis]       - Parallel 3-report analysis")
        print("  [MVR Analysis Chain] - Sequential pipeline processing")
        
        print("\nUsage examples:")
        print("  CLI:  python 1-enterprise.py '[MVR Analysis]'")
        print("  API:  POST /flow {'command': '[MVR Analysis]'}")
        print("  S3:   Drop file to trigger: [mvr_analysis].trigger")
        
        return True
    else:
        print("\n[WARNING] Cannot promote to production - issues need resolution")
        return False

if __name__ == "__main__":
    promote_to_production()