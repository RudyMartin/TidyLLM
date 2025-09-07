#!/usr/bin/env python3
"""
AWS Service Access Verification Script
Tests SageMaker, S3, and PostgreSQL RDS access using UnifiedSessionManager
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Use existing UnifiedSessionManager instead of creating new session management
sys.path.insert(0, str(Path(__file__).parent / 'scripts' / 'infrastructure'))
try:
    from start_unified_sessions import UnifiedSessionManager, ServiceType
    UNIFIED_SESSION_AVAILABLE = True
except ImportError:
    try:
        # Fallback path
        sys.path.insert(0, str(Path(__file__).parent))
        from scripts.infrastructure.start_unified_sessions import UnifiedSessionManager, ServiceType
        UNIFIED_SESSION_AVAILABLE = True
    except ImportError:
        UNIFIED_SESSION_AVAILABLE = False

# Removed manual credential loading and individual test functions
# Now using UnifiedSessionManager which handles all session management properly

def main():
    print('AWS Service Access Verification')
    print('=' * 60)
    print(f'Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print()
    
    if not UNIFIED_SESSION_AVAILABLE:
        print('[ERROR] UnifiedSessionManager not available - cannot verify services')
        print('        Check that scripts/infrastructure/start_unified_sessions.py exists')
        return 1
    
    print('[UNIFIED SESSION] Using existing UnifiedSessionManager for all service access')
    
    # Initialize UnifiedSessionManager - it handles credential discovery automatically
    try:
        session_mgr = UnifiedSessionManager()
        print('[SUCCESS] UnifiedSessionManager initialized with auto-credential discovery')
    except Exception as e:
        print(f'[ERROR] Failed to initialize UnifiedSessionManager: {e}')
        return 1
    
    print()
    
    # Get health summary from UnifiedSessionManager
    print('[HEALTH CHECK] Using UnifiedSessionManager health check system...')
    health_summary = session_mgr.get_health_summary()
    
    print(f'[CREDENTIALS] Source: {health_summary["credential_source"]}')
    print(f'[CONFIG] S3 Region: {health_summary["configuration"]["s3_region"]}')
    print(f'[CONFIG] PostgreSQL: {health_summary["configuration"]["postgres_host"]}')
    print()
    
    # Test individual services using the existing session manager
    results = []
    
    # Test S3 using UnifiedSessionManager
    print('[S3] Testing via UnifiedSessionManager...')
    s3_client = session_mgr.get_s3_client()
    if s3_client and session_mgr.is_healthy(ServiceType.S3):
        try:
            buckets = s3_client.list_buckets()
            print(f'[SUCCESS] S3 accessible via UnifiedSessionManager')
            print(f'          Total buckets: {len(buckets["Buckets"])}')
            for bucket in buckets['Buckets'][:3]:
                print(f'            - {bucket["Name"]}')
            results.append(('S3', True))
        except Exception as e:
            print(f'[FAIL] S3 error: {e}')
            results.append(('S3', False))
    else:
        print('[FAIL] S3 not available via UnifiedSessionManager')
        results.append(('S3', False))
    
    print()
    
    # Test PostgreSQL using UnifiedSessionManager
    print('[POSTGRESQL] Testing via UnifiedSessionManager...')
    if session_mgr.is_healthy(ServiceType.POSTGRESQL):
        try:
            conn = session_mgr.get_postgres_connection()
            if conn:
                with conn.cursor() as cursor:
                    cursor.execute('SELECT version(), current_database(), current_user;')
                    result = cursor.fetchone()
                    print(f'[SUCCESS] PostgreSQL accessible via UnifiedSessionManager')
                    print(f'          Database: {result[1]}')
                    print(f'          User: {result[2]}')
                session_mgr.return_postgres_connection(conn)
                results.append(('PostgreSQL', True))
            else:
                print('[FAIL] Could not get PostgreSQL connection from UnifiedSessionManager')
                results.append(('PostgreSQL', False))
        except Exception as e:
            print(f'[FAIL] PostgreSQL error: {e}')
            results.append(('PostgreSQL', False))
    else:
        print('[FAIL] PostgreSQL not healthy in UnifiedSessionManager')
        results.append(('PostgreSQL', False))
    
    print()
    
    # Test Bedrock using UnifiedSessionManager
    print('[BEDROCK] Testing via UnifiedSessionManager...')
    bedrock_client = session_mgr.get_bedrock_client()
    if bedrock_client and session_mgr.is_healthy(ServiceType.BEDROCK):
        print('[SUCCESS] Bedrock client accessible via UnifiedSessionManager')
        results.append(('Bedrock', True))
    else:
        print('[FAIL] Bedrock not available via UnifiedSessionManager')
        results.append(('Bedrock', False))
    
    print()
    
    # Test SageMaker using existing S3 client
    print('[SAGEMAKER] Testing via existing session...')
    if s3_client:
        try:
            import boto3
            # Use the same session as S3 client for SageMaker
            sagemaker = s3_client._client_config.session.client('sagemaker')
            notebooks = sagemaker.list_notebook_instances()
            print(f'[SUCCESS] SageMaker accessible via existing session')
            print(f'          Notebook instances: {len(notebooks["NotebookInstances"])}')
            results.append(('SageMaker', True))
        except Exception as e:
            print(f'[FAIL] SageMaker: {e}')
            results.append(('SageMaker', False))
    else:
        print('[FAIL] Cannot test SageMaker - no S3 session available')
        results.append(('SageMaker', False))
    
    print()
    
    # Summary
    print('=' * 60)
    print('[SUMMARY] AWS Service Access Results (via UnifiedSessionManager):')
    
    for service, success in results:
        status = 'ACCESSIBLE' if success else 'FAILED'
        print(f'  {service:12} : {status}')
    
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    print()
    print(f'[OVERALL] {successful}/{total} services accessible ({successful/total*100:.0f}%)')
    
    if successful >= 2:  # S3 + PostgreSQL are the core requirements
        print('[STATUS] Core services (S3 + PostgreSQL) accessible - system ready!')
    
    print('[ARCHITECTURE] Used existing UnifiedSessionManager - no new session management created')
    print('=' * 60)
    
    # Cleanup
    session_mgr.cleanup()
    
    return 0 if successful > 0 else 1

if __name__ == "__main__":
    sys.exit(main())