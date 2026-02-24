"""
TidyLLM Unified Credential Manager - DEPRECATED
================================================

⚠️ DEPRECATED: This file is now obsolete. The unified sessions system 
(tidyllm/infrastructure/session/unified.py) already handles ALL credential 
management with Polars backend.

DO NOT USE THIS FILE. Use unified sessions system instead:
- from tidyllm.infrastructure.session.unified import get_global_session_manager

This was an incorrect approach that mixed validators with credential management.
The correct architecture is:
- Credentials: unified_sessions system (Polars backend)
- Validators: Just validation logic, get connections from unified_sessions

LEGACY CODE KEPT FOR REFERENCE ONLY
====================================
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class ServiceCredentials:
    """Container for service-specific credentials."""
    service_name: str
    credentials: Dict[str, str]
    connection_params: Dict[str, Any]
    is_complete: bool
    missing_fields: List[str]

class UnifiedCredentialManager:
    """
    Unified credential manager using Polars backend for ALL TidyLLM components.
    
    This replaces all the inconsistent configuration approaches:
    - Database Validator: Uses this instead of various config managers
    - AWS Validator: Uses this instead of environment variables  
    - MLflow Service: Uses this instead of hardcoded MLflowConfig
    - Onboarding Session Manager: Uses this instead of unified_sessions + settings_manager
    """
    
    _instance: Optional['UnifiedCredentialManager'] = None
    _initialized: bool = False
    
    def __new__(cls):
        """Singleton pattern - ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.loader = None
        self.credentials_df = None
        self.aws_services_df = None
        self.config_df = None
        self._service_cache = {}
        
        # Initialize Polars backend
        self._initialize_polars_backend()
        self._initialized = True
    
    def _initialize_polars_backend(self):
        """Initialize the Polars configuration backend."""
        if not POLARS_AVAILABLE:
            logger.error("Polars not available - unified credential manager cannot function")
            return
        
        try:
            from scripts.infrastructure.config_loader_polars import load_settings_with_polars
            
            # Load configuration using Polars backend
            self.loader, dataframes = load_settings_with_polars()
            
            if dataframes['credentials'].is_empty():
                logger.warning("No credentials found in Polars configuration")
                return
            
            self.credentials_df = dataframes['credentials']
            self.aws_services_df = dataframes['aws_services']
            self.config_df = dataframes['config']
            
            logger.info(f"✅ Unified Credential Manager initialized with {len(self.credentials_df)} credentials")
            
        except Exception as e:
            logger.error(f"Failed to initialize Polars backend: {e}")
    
    def get_aws_credentials(self) -> ServiceCredentials:
        """Get AWS credentials from Polars backend."""
        if 'aws' in self._service_cache:
            return self._service_cache['aws']
        
        if self.credentials_df is None:
            return self._empty_service_credentials('aws', 'Polars backend not initialized')
        
        # Get AWS credentials from Polars DataFrame
        aws_creds = self.credentials_df.filter(
            pl.col('service') == 'aws'
        )
        
        if aws_creds.is_empty():
            return self._empty_service_credentials('aws', 'No AWS credentials found')
        
        # Convert to credential dictionary
        creds = {}
        for row in aws_creds.to_dicts():
            key_parts = row['credential_key'].split('.')
            field_name = key_parts[-1]  # Get last part
            creds[field_name] = row['credential_value']
        
        # Check for required AWS fields
        required_fields = ['access_key_id', 'secret_access_key', 'region']
        missing = [field for field in required_fields if field not in creds]
        
        service_creds = ServiceCredentials(
            service_name='aws',
            credentials=creds,
            connection_params={
                'aws_access_key_id': creds.get('access_key_id'),
                'aws_secret_access_key': creds.get('secret_access_key'),
                'region_name': creds.get('region', 'us-east-1'),
                'aws_session_token': creds.get('session_token')
            },
            is_complete=len(missing) == 0,
            missing_fields=missing
        )
        
        self._service_cache['aws'] = service_creds
        return service_creds
    
    def get_postgresql_credentials(self) -> ServiceCredentials:
        """Get PostgreSQL credentials from Polars backend."""
        if 'postgresql' in self._service_cache:
            return self._service_cache['postgresql']
        
        if self.credentials_df is None:
            return self._empty_service_credentials('postgresql', 'Polars backend not initialized')
        
        # Get PostgreSQL credentials from Polars DataFrame
        pg_creds = self.credentials_df.filter(
            pl.col('service') == 'postgresql'
        )
        
        if pg_creds.is_empty():
            return self._empty_service_credentials('postgresql', 'No PostgreSQL credentials found')
        
        # Convert to credential dictionary
        creds = {}
        for row in pg_creds.to_dicts():
            key_parts = row['credential_key'].split('.')
            field_name = key_parts[-1]  # Get last part
            creds[field_name] = row['credential_value']
        
        # Check for required PostgreSQL fields
        required_fields = ['host', 'database', 'username', 'password']
        missing = [field for field in required_fields if field not in creds]
        
        service_creds = ServiceCredentials(
            service_name='postgresql',
            credentials=creds,
            connection_params={
                'host': creds.get('host'),
                'port': int(creds.get('port', 5432)),
                'database': creds.get('database'),
                'user': creds.get('username'),
                'password': creds.get('password'),
                'sslmode': creds.get('ssl_mode', 'require')
            },
            is_complete=len(missing) == 0,
            missing_fields=missing
        )
        
        self._service_cache['postgresql'] = service_creds
        return service_creds
    
    def get_mlflow_credentials(self) -> ServiceCredentials:
        """Get MLflow credentials from Polars backend."""
        if 'mlflow' in self._service_cache:
            return self._service_cache['mlflow']
        
        if self.credentials_df is None:
            return self._empty_service_credentials('mlflow', 'Polars backend not initialized')
        
        # Get MLflow credentials from Polars DataFrame
        mlflow_creds = self.credentials_df.filter(
            pl.col('service') == 'mlflow'
        )
        
        # If no specific MLflow credentials, use defaults
        creds = {}
        if not mlflow_creds.is_empty():
            for row in mlflow_creds.to_dicts():
                key_parts = row['credential_key'].split('.')
                field_name = key_parts[-1]
                creds[field_name] = row['credential_value']
        
        # Set defaults for MLflow
        tracking_uri = creds.get('tracking_uri', 'http://localhost:5000')
        
        service_creds = ServiceCredentials(
            service_name='mlflow',
            credentials=creds,
            connection_params={
                'tracking_uri': tracking_uri,
                'gateway_uri': tracking_uri,  # For backward compatibility
                'timeout': int(creds.get('timeout', 30)),
                'retry_count': int(creds.get('retry_count', 3))
            },
            is_complete=True,  # MLflow can work with defaults
            missing_fields=[]
        )
        
        self._service_cache['mlflow'] = service_creds
        return service_creds
    
    def get_service_credentials(self, service_name: str) -> ServiceCredentials:
        """Get credentials for any service."""
        if service_name == 'aws':
            return self.get_aws_credentials()
        elif service_name in ['postgresql', 'postgres', 'database']:
            return self.get_postgresql_credentials()
        elif service_name == 'mlflow':
            return self.get_mlflow_credentials()
        else:
            return self._empty_service_credentials(service_name, f'Service {service_name} not supported')
    
    def _empty_service_credentials(self, service_name: str, reason: str) -> ServiceCredentials:
        """Return empty service credentials with error reason."""
        return ServiceCredentials(
            service_name=service_name,
            credentials={},
            connection_params={},
            is_complete=False,
            missing_fields=[f'Error: {reason}']
        )
    
    def get_credential_summary(self) -> Dict[str, Any]:
        """Get summary of all available credentials."""
        if self.credentials_df is None:
            return {'error': 'Polars backend not initialized'}
        
        summary = {
            'total_credentials': len(self.credentials_df),
            'services_available': [],
            'services_complete': [],
            'services_incomplete': []
        }
        
        # Check each service
        for service in ['aws', 'postgresql', 'mlflow']:
            creds = self.get_service_credentials(service)
            summary['services_available'].append(service)
            
            if creds.is_complete:
                summary['services_complete'].append(service)
            else:
                summary['services_incomplete'].append({
                    'service': service,
                    'missing_fields': creds.missing_fields
                })
        
        return summary
    
    def validate_all_credentials(self) -> Dict[str, Any]:
        """Validate all credentials for completeness."""
        validation_results = {
            'overall_status': 'unknown',
            'services': {},
            'issues': [],
            'warnings': []
        }
        
        services = ['aws', 'postgresql', 'mlflow']
        complete_services = 0
        
        for service in services:
            creds = self.get_service_credentials(service)
            validation_results['services'][service] = {
                'complete': creds.is_complete,
                'missing_fields': creds.missing_fields,
                'credential_count': len(creds.credentials)
            }
            
            if creds.is_complete:
                complete_services += 1
            else:
                validation_results['issues'].extend([
                    f"{service}: {field}" for field in creds.missing_fields
                ])
        
        # Determine overall status
        if complete_services == len(services):
            validation_results['overall_status'] = 'complete'
        elif complete_services > 0:
            validation_results['overall_status'] = 'partial'
        else:
            validation_results['overall_status'] = 'incomplete'
        
        return validation_results

# Global instance functions
_global_manager: Optional[UnifiedCredentialManager] = None

def get_unified_credential_manager() -> UnifiedCredentialManager:
    """Get the global unified credential manager instance."""
    global _global_manager
    if _global_manager is None:
        _global_manager = UnifiedCredentialManager()
    return _global_manager

def get_aws_credentials() -> ServiceCredentials:
    """Convenience function to get AWS credentials."""
    return get_unified_credential_manager().get_aws_credentials()

def get_postgresql_credentials() -> ServiceCredentials:
    """Convenience function to get PostgreSQL credentials."""
    return get_unified_credential_manager().get_postgresql_credentials()

def get_mlflow_credentials() -> ServiceCredentials:
    """Convenience function to get MLflow credentials."""
    return get_unified_credential_manager().get_mlflow_credentials()

if __name__ == "__main__":
    # Demo the unified credential manager
    print("UNIFIED CREDENTIAL MANAGER DEMO")
    print("=" * 50)
    
    manager = get_unified_credential_manager()
    
    # Show credential summary
    summary = manager.get_credential_summary()
    print("Credential Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Validate all credentials
    validation = manager.validate_all_credentials()
    print(f"\nOverall Status: {validation['overall_status']}")
    
    # Test each service
    for service in ['aws', 'postgresql', 'mlflow']:
        creds = manager.get_service_credentials(service)
        print(f"\n{service.upper()} Credentials:")
        print(f"  Complete: {creds.is_complete}")
        print(f"  Credential count: {len(creds.credentials)}")
        if not creds.is_complete:
            print(f"  Missing: {creds.missing_fields}")