#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TidyLLM Services - Central Import Module

This module eliminates confusion by providing a single import point for all services.
No more scattered imports or multiple names for the same functionality.

BEFORE (confusing):
    from tidyllm.demo_standalone.credential_manager import CredentialManager
    from tidyllm_vectorqa.whitepapers.s3_session_manager import S3SessionManager  
    from tidyllm.demo_standalone.connection_manager import DemoConnectionManager
    from transfer.qaz_final_20250404.core.client_bundle import ClientBundle

AFTER (clean):
    from tidyllm_services import get_services, get_credentials, get_s3, get_database

Usage Examples:
    # Get unified services (recommended)
    services = get_services()
    s3_client = services.get_s3_client()
    bedrock_client = services.get_bedrock_client()
    
    # Quick access methods
    creds = get_credentials()
    s3 = get_s3()
    db = get_database()
    
    # Session management for CLI/API hybrid
    from tidyllm_services import store_session, get_session
    store_session('user123', 'preferences', {'theme': 'dark'})
    prefs = get_session('user123', 'preferences')
"""

# Import the unified services manager
try:
    from tidyllm_unified_services import (
        UnifiedServiceManager, 
        get_services as _get_unified_services,
        get_s3_client as _get_s3_quick,
        get_bedrock_client as _get_bedrock_quick,
        get_database_connection as _get_db_quick,
        store_session_data as _store_session_quick,
        get_session_data as _get_session_quick
    )
    UNIFIED_SERVICES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ UnifiedServiceManager not available: {e}")
    UNIFIED_SERVICES_AVAILABLE = False

# Fallback imports for individual managers if unified services fail
try:
    import sys
    sys.path.append('tidyllm/demo-standalone')
    from credential_manager import CredentialManager, get_credential_manager
    CREDENTIAL_MANAGER_AVAILABLE = True
except ImportError:
    CREDENTIAL_MANAGER_AVAILABLE = False

try:
    sys.path.append('tidyllm-vectorqa/tidyllm_vectorqa/whitepapers')
    from s3_session_manager import S3SessionManager, get_s3_session_manager
    S3_SESSION_MANAGER_AVAILABLE = True
except ImportError:
    S3_SESSION_MANAGER_AVAILABLE = False

try:
    sys.path.append('tidyllm/tidyllm')
    from connection_manager import DemoConnectionManager, get_connection_manager
    CONNECTION_MANAGER_AVAILABLE = True
except ImportError:
    CONNECTION_MANAGER_AVAILABLE = False

try:
    sys.path.append('transfer/qaz_final_20250404/core')
    from client_bundle import get_enhanced_client_bundle, EnhancedClientBundle
    CLIENT_BUNDLE_AVAILABLE = True
except ImportError:
    CLIENT_BUNDLE_AVAILABLE = False


# === PRIMARY API (UNIFIED SERVICES) ===

def get_services() -> 'UnifiedServiceManager':
    """
    Get the unified services manager - PRIMARY RECOMMENDED METHOD
    
    This is the main entry point that consolidates all AWS and database services.
    Use this instead of individual manager imports.
    
    Returns:
        UnifiedServiceManager: Single interface to all services
        
    Examples:
        services = get_services()
        s3 = services.get_s3_client()
        bedrock = services.get_bedrock_client()
        services.store_data('key', 'value')
    """
    if UNIFIED_SERVICES_AVAILABLE:
        return _get_unified_services()
    else:
        raise ImportError("UnifiedServiceManager not available. Install required dependencies.")


def get_service_status() -> dict:
    """Get comprehensive status of all services"""
    if UNIFIED_SERVICES_AVAILABLE:
        return get_services().get_service_status()
    else:
        return {
            'error': 'UnifiedServiceManager not available',
            'fallback_available': {
                'credentials': CREDENTIAL_MANAGER_AVAILABLE,
                's3': S3_SESSION_MANAGER_AVAILABLE,
                'database': CONNECTION_MANAGER_AVAILABLE,
                'client_bundle': CLIENT_BUNDLE_AVAILABLE
            }
        }


# === QUICK ACCESS METHODS ===

def get_s3_client(**kwargs):
    """Quick access to S3 client"""
    if UNIFIED_SERVICES_AVAILABLE:
        return _get_s3_quick(**kwargs)
    elif S3_SESSION_MANAGER_AVAILABLE:
        return get_s3_session_manager().get_s3_client(**kwargs)
    else:
        raise ImportError("No S3 client available")


def get_bedrock_client():
    """Quick access to Bedrock client"""
    if UNIFIED_SERVICES_AVAILABLE:
        return _get_bedrock_quick()
    else:
        raise ImportError("No Bedrock client available")


def get_database_connection():
    """Quick access to database connection"""
    if UNIFIED_SERVICES_AVAILABLE:
        return _get_db_quick()
    elif CONNECTION_MANAGER_AVAILABLE:
        return get_connection_manager().get_connection()
    else:
        raise ImportError("No database connection available")


# === INDIVIDUAL SERVICE ACCESS (FALLBACK) ===

def get_credentials():
    """Get credential manager"""
    if UNIFIED_SERVICES_AVAILABLE:
        return get_services().credentials
    elif CREDENTIAL_MANAGER_AVAILABLE:
        return get_credential_manager()
    else:
        raise ImportError("No credential manager available")


def get_s3():
    """Get S3 session manager"""
    if UNIFIED_SERVICES_AVAILABLE:
        return get_services().s3_manager
    elif S3_SESSION_MANAGER_AVAILABLE:
        return get_s3_session_manager()
    else:
        raise ImportError("No S3 session manager available")


def get_database():
    """Get database connection manager"""
    if UNIFIED_SERVICES_AVAILABLE:
        return get_services().db_manager
    elif CONNECTION_MANAGER_AVAILABLE:
        return get_connection_manager()
    else:
        raise ImportError("No database manager available")


def get_client_bundle():
    """Get enhanced client bundle"""
    if UNIFIED_SERVICES_AVAILABLE:
        return get_services().client_bundle
    elif CLIENT_BUNDLE_AVAILABLE:
        return get_enhanced_client_bundle()
    else:
        raise ImportError("No client bundle available")


# === SESSION MANAGEMENT (CLI/API HYBRID) ===

def store_session(session_id: str, key: str, data, ttl: int = 3600):
    """Store session data for CLI/API hybrid mode"""
    if UNIFIED_SERVICES_AVAILABLE:
        return _store_session_quick(session_id, key, data)
    elif CONNECTION_MANAGER_AVAILABLE:
        return get_connection_manager().store_session_data(session_id, key, data, ttl)
    else:
        raise ImportError("No session storage available")


def get_session(session_id: str, key: str):
    """Get session data for CLI/API hybrid mode"""
    if UNIFIED_SERVICES_AVAILABLE:
        return _get_session_quick(session_id, key)
    elif CONNECTION_MANAGER_AVAILABLE:
        return get_connection_manager().get_session_data(session_id, key)
    else:
        raise ImportError("No session storage available")


def clear_session(session_id: str):
    """Clear all data for a session"""
    if UNIFIED_SERVICES_AVAILABLE:
        return get_services().db_manager.clear_session(session_id)
    elif CONNECTION_MANAGER_AVAILABLE:
        return get_connection_manager().clear_session(session_id)
    else:
        raise ImportError("No session storage available")


def get_active_sessions():
    """Get list of active session IDs"""
    if UNIFIED_SERVICES_AVAILABLE:
        return get_services().db_manager.get_active_sessions()
    elif CONNECTION_MANAGER_AVAILABLE:
        return get_connection_manager().get_active_sessions()
    else:
        return []


# === UTILITY FUNCTIONS ===

def test_all_connections() -> dict:
    """Test all service connections"""
    if UNIFIED_SERVICES_AVAILABLE:
        return get_services().test_all_connections()
    else:
        results = {}
        
        # Test individual services
        if S3_SESSION_MANAGER_AVAILABLE:
            try:
                s3_mgr = get_s3_session_manager()
                s3_status = s3_mgr.test_connection()
                results['s3'] = s3_status.get('success', False)
            except:
                results['s3'] = False
        
        if CONNECTION_MANAGER_AVAILABLE:
            try:
                db_mgr = get_connection_manager()
                results['database'] = db_mgr.test_connection()
            except:
                results['database'] = False
                
        return results


def cleanup_all_services():
    """Clean up all services"""
    if UNIFIED_SERVICES_AVAILABLE:
        get_services().cleanup()
    else:
        # Clean up individual services
        if CONNECTION_MANAGER_AVAILABLE:
            try:
                get_connection_manager().close()
            except:
                pass


# === ALIASES FOR BACKWARD COMPATIBILITY ===

# Common alternative names that might be used
get_aws_services = get_services  # For AWS-specific contexts
get_service_manager = get_services  # Generic service manager
get_unified_manager = get_services  # Explicit unified manager
get_central_services = get_services  # Central services

# Storage aliases
store_data = lambda key, data: get_services().store_data(key, data)
get_data = lambda key: get_services().get_data(key)

# AWS client aliases  
get_aws_s3 = get_s3_client
get_aws_bedrock = get_bedrock_client

# Database aliases
get_db = get_database
get_db_connection = get_database_connection


# === MODULE INFO ===

__version__ = "1.0.0"
__author__ = "TidyLLM Services Team"
__description__ = "Centralized service access for TidyLLM applications"

# Available services summary
AVAILABLE_SERVICES = {
    'unified_services': UNIFIED_SERVICES_AVAILABLE,
    'credential_manager': CREDENTIAL_MANAGER_AVAILABLE,
    's3_session_manager': S3_SESSION_MANAGER_AVAILABLE,
    'connection_manager': CONNECTION_MANAGER_AVAILABLE,
    'client_bundle': CLIENT_BUNDLE_AVAILABLE
}


def print_service_summary():
    """Print summary of available services"""
    print("🚀 TidyLLM Services Summary")
    print("=" * 40)
    
    if UNIFIED_SERVICES_AVAILABLE:
        print("✅ Unified Services Manager: AVAILABLE")
        status = get_service_status()
        print(f"   Overall Status: {status['overall_status']}")
        for service, healthy in status['health'].items():
            icon = "✅" if healthy else "❌"
            print(f"   {icon} {service}")
    else:
        print("⚠️  Unified Services Manager: NOT AVAILABLE")
        print("📋 Individual Services:")
        for service, available in AVAILABLE_SERVICES.items():
            if service != 'unified_services':
                icon = "✅" if available else "❌"
                print(f"   {icon} {service}")
    
    print("=" * 40)
    print("💡 Usage: from tidyllm_services import get_services")


if __name__ == "__main__":
    print_service_summary()
    
    # Test basic functionality
    try:
        print("\n🧪 Testing Services...")
        if UNIFIED_SERVICES_AVAILABLE:
            services = get_services()
            print(f"✅ Got unified services: {type(services).__name__}")
            
            connections = test_all_connections()
            print("🔗 Connection Tests:")
            for service, connected in connections.items():
                icon = "✅" if connected else "❌"
                print(f"   {icon} {service}")
        else:
            print("⚠️  Unified services not available, testing fallbacks...")
            
    except Exception as e:
        print(f"❌ Error testing services: {e}")
    
    print("\n✅ TidyLLM Services module ready!")