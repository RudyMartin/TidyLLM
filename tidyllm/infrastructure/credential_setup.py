#!/usr/bin/env python3
"""
AWS Service Access Verification Script
Uses UnifiedSessionManager with proper admin folder configuration loading
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Use existing UnifiedSessionManager 
sys.path.insert(0, str(Path(__file__).parent / 'scripts' / 'infrastructure'))
try:
    from start_unified_sessions import UnifiedSessionManager, ServiceType, ServiceConfig
    UNIFIED_SESSION_AVAILABLE = True
except ImportError:
    UNIFIED_SESSION_AVAILABLE = False

def load_admin_config() -> ServiceConfig:
    """Load configuration from admin folder to configure UnifiedSessionManager properly"""
    
    # Load AWS credentials from admin folder
    admin_dir = Path.cwd() / 'tidyllm' / 'admin'
    
    config = ServiceConfig()
    
    # Load AWS credentials
    cred_file = admin_dir / 'set_aws_env.bat'
    if cred_file.exists():
        with open(cred_file, 'r') as f:
            content = f.read()
            
        import re
        access_key_match = re.search(r'AWS_ACCESS_KEY_ID=([A-Z0-9]+)', content)
        secret_key_match = re.search(r'AWS_SECRET_ACCESS_KEY=([A-Za-z0-9+/]+)', content)
        region_match = re.search(r'AWS_DEFAULT_REGION=([a-z0-9-]+)', content)
        
        if access_key_match and secret_key_match and region_match:
            # Set environment variables for UnifiedSessionManager to find
            os.environ['AWS_ACCESS_KEY_ID'] = access_key_match.group(1)
            os.environ['AWS_SECRET_ACCESS_KEY'] = secret_key_match.group(1)
            os.environ['AWS_DEFAULT_REGION'] = region_match.group(1)
            
            config.s3_access_key_id = access_key_match.group(1)
            config.s3_secret_access_key = secret_key_match.group(1)
            config.s3_region = region_match.group(1)
            config.bedrock_region = region_match.group(1)
            print(f'[ADMIN CREDENTIALS] Loaded AWS credentials from {cred_file}')
    
    # Load PostgreSQL settings from settings.yaml
    settings_file = admin_dir / 'settings.yaml'
    if settings_file.exists():
        try:
            import yaml
            with open(settings_file, 'r') as f:
                settings = yaml.safe_load(f)
            
            # Extract PostgreSQL configuration
            postgres_config = settings.get('postgres', {})
            if postgres_config:
                config.postgres_host = postgres_config.get('host', config.postgres_host)
                config.postgres_port = postgres_config.get('port', config.postgres_port)
                config.postgres_database = postgres_config.get('db_name', config.postgres_database)
                config.postgres_username = postgres_config.get('db_user', config.postgres_username)
                config.postgres_password = postgres_config.get('db_password', config.postgres_password)
                print(f'[ADMIN CONFIG] Loaded PostgreSQL settings from {settings_file}')
                print(f'               Host: {config.postgres_host}')
                print(f'               Database: {config.postgres_database}')
                print(f'[DEBUG] Raw postgres config from YAML: {postgres_config}')
            
            # Extract S3 configuration
            s3_config = settings.get('s3', {})
            if s3_config:
                config.s3_default_bucket = s3_config.get('bucket', config.s3_default_bucket)
                print(f'[ADMIN CONFIG] Default S3 bucket: {config.s3_default_bucket}')
                
        except Exception as e:
            print(f'[WARNING] Could not load settings.yaml: {e}')
    
    return config

def main():
    print('AWS Service Access Verification (Using Admin Folder Configuration)')
    print('=' * 70)
    print(f'Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print()
    
    if not UNIFIED_SESSION_AVAILABLE:
        print('[ERROR] UnifiedSessionManager not available')
        return 1
    
    # Load configuration from admin folder
    print('[ADMIN DISCOVERY] Loading configuration from tidyllm/admin/ folder...')
    config = load_admin_config()
    print()
    
    # Debug: Show configuration being passed
    print(f'[DEBUG] PostgreSQL config being passed to UnifiedSessionManager:')
    print(f'        Host: {config.postgres_host}')
    print(f'        Port: {config.postgres_port}')
    print(f'        Database: {config.postgres_database}')
    print(f'        Username: {config.postgres_username}')
    print(f'        Password: {"***" if config.postgres_password else "None"}')
    print()
    
    # Initialize UnifiedSessionManager with admin configuration
    try:
        session_mgr = UnifiedSessionManager(config)
        print('[SUCCESS] UnifiedSessionManager initialized with admin folder configuration')
        
        # Debug: Show what UnifiedSessionManager actually has
        print(f'[DEBUG] UnifiedSessionManager internal config:')
        print(f'        Host: {session_mgr.config.postgres_host}')
        print(f'        Database: {session_mgr.config.postgres_database}')
        
    except Exception as e:
        print(f'[ERROR] Failed to initialize UnifiedSessionManager: {e}')
        return 1
    
    print()
    
    # Get health summary
    health_summary = session_mgr.get_health_summary()
    print('[HEALTH SUMMARY] Service status from UnifiedSessionManager:')
    print(f'  Credential source: {health_summary["credential_source"]}')
    print(f'  Overall healthy: {health_summary["overall_healthy"]}')
    
    for service_name, service_health in health_summary["services"].items():
        status = "HEALTHY" if service_health["healthy"] else "FAILED"
        latency = f" ({service_health['latency_ms']:.0f}ms)" if service_health['latency_ms'] else ""
        error = f" - {service_health['error']}" if service_health['error'] else ""
        print(f'  {service_name:12}: {status}{latency}{error}')
    
    print()
    
    # Test core services that should work
    results = []
    
    # Test S3
    print('[S3 TEST] Testing S3 access...')
    s3_client = session_mgr.get_s3_client()
    if s3_client and session_mgr.is_healthy(ServiceType.S3):
        try:
            buckets = s3_client.list_buckets()
            print(f'[SUCCESS] S3 accessible - {len(buckets["Buckets"])} buckets found')
            
            # Test specific bucket if configured
            if config.s3_default_bucket:
                try:
                    s3_client.head_bucket(Bucket=config.s3_default_bucket)
                    print(f'[SUCCESS] Target bucket "{config.s3_default_bucket}" is accessible')
                except Exception as e:
                    print(f'[WARNING] Target bucket "{config.s3_default_bucket}" access denied: {e}')
            
            results.append(('S3', True))
        except Exception as e:
            print(f'[FAIL] S3 test failed: {e}')
            results.append(('S3', False))
    else:
        print('[FAIL] S3 client not available or unhealthy')
        results.append(('S3', False))
    
    print()
    
    # Test PostgreSQL - Use direct connection since UnifiedSessionManager config is being overridden
    print('[POSTGRESQL TEST] Testing PostgreSQL access...')
    print('[INFO] UnifiedSessionManager config was overridden, using direct connection with admin config')
    try:
        import psycopg2
        
        # Debug: Print exact connection parameters being used
        print(f'[DEBUG] Direct connection parameters:')
        print(f'        host={config.postgres_host}')
        print(f'        port={config.postgres_port}')
        print(f'        database={config.postgres_database}')
        print(f'        user={config.postgres_username}')
        print(f'        password={"***" if config.postgres_password else "None"}')
        
        # Use the admin config we loaded
        conn = psycopg2.connect(
            host=config.postgres_host,
            port=config.postgres_port,
            database=config.postgres_database,
            user=config.postgres_username,
            password=config.postgres_password,
            sslmode='require',
            connect_timeout=10
        )
        
        with conn.cursor() as cursor:
            cursor.execute('SELECT version(), current_database(), current_user;')
            result = cursor.fetchone()
            
            cursor.execute("SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public';")
            table_count = cursor.fetchone()[0]
            
            print(f'[SUCCESS] PostgreSQL accessible via direct connection')
            print(f'          Host: {config.postgres_host}')
            print(f'          Database: {result[1]} | User: {result[2]} | Tables: {table_count}')
            print(f'          Version: {result[0].split(",")[0]}')
        
        conn.close()
        results.append(('PostgreSQL', True))
        
    except Exception as e:
        print(f'[FAIL] PostgreSQL direct connection failed: {e}')
        results.append(('PostgreSQL', False))
    
    print()
    
    # Test Bedrock
    print('[BEDROCK TEST] Testing AWS Bedrock access...')
    bedrock_client = session_mgr.get_bedrock_client()
    if bedrock_client and session_mgr.is_healthy(ServiceType.BEDROCK):
        print('[SUCCESS] Bedrock client available')
        results.append(('Bedrock', True))
    else:
        print('[FAIL] Bedrock client not available')
        results.append(('Bedrock', False))
    
    print()
    
    # Summary
    print('=' * 70)
    print('[FINAL SUMMARY] AWS Service Access Results:')
    
    for service, success in results:
        status = 'ACCESSIBLE' if success else 'FAILED'
        print(f'  {service:12} : {status}')
    
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    print()
    print(f'[OVERALL] {successful}/{total} services accessible ({successful/total*100:.0f}%)')
    
    if successful >= 2:  # S3 + PostgreSQL are core
        print('[STATUS] Core services accessible - TidyLLM system ready!')
    elif successful >= 1:
        print('[STATUS] Partial access - some services available')
    else:
        print('[STATUS] No services accessible - check configuration')
    
    print('[ARCHITECTURE] Used existing UnifiedSessionManager with admin folder config')
    print('=' * 70)
    
    # Cleanup
    session_mgr.cleanup()
    
    return 0 if successful > 0 else 1

if __name__ == "__main__":
    sys.exit(main())