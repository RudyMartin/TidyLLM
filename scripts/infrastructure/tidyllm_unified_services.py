#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

# S3 Configuration Management
sys.path.append(str(Path(__file__).parent.parent / 'tidyllm' / 'admin') if 'tidyllm' in str(Path(__file__)) else str(Path(__file__).parent / 'tidyllm' / 'admin'))
from credential_loader import get_s3_config, build_s3_path

# Get S3 configuration (bucket and path builder)
s3_config = get_s3_config()  # Add environment parameter for dev/staging/prod

TidyLLM Unified Services Manager

Consolidates all scattered AWS and database managers into a single, coherent system.
This eliminates the naming confusion and provides a single point of access for all services.

Integrates:
- CredentialManager (demo-standalone) 
- S3SessionManager (tidyllm-vectorqa)
- DemoConnectionManager (demo-standalone)
- ClientBundle (transfer/qaz_final_20250404)
- BedrockEnhancedWrapper (tidyllm)
"""

import os
import sys
import logging
import threading
import time
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Add paths for existing managers
sys.path.append('tidyllm/demo-standalone')
sys.path.append('tidyllm-vectorqa/tidyllm_vectorqa/whitepapers')
sys.path.append('transfer/qaz_final_20250404/core')
sys.path.append('tidyllm/tidyllm')

# Import existing managers - DO NOT REPLACE
try:
    from credential_manager import CredentialManager
    CREDENTIAL_MANAGER_AVAILABLE = True
except ImportError:
    CREDENTIAL_MANAGER_AVAILABLE = False
    print("⚠️ CredentialManager not available")

try:
    from s3_session_manager import S3SessionManager, S3Utils
    S3_SESSION_MANAGER_AVAILABLE = True
except ImportError:
    S3_SESSION_MANAGER_AVAILABLE = False
    print("⚠️ S3SessionManager not available")

try:
    from connection_manager import DemoConnectionManager
    CONNECTION_MANAGER_AVAILABLE = True
except ImportError:
    CONNECTION_MANAGER_AVAILABLE = False
    print("⚠️ DemoConnectionManager not available")

try:
    from client_bundle import ClientBundle
    CLIENT_BUNDLE_AVAILABLE = True
except ImportError:
    CLIENT_BUNDLE_AVAILABLE = False
    print("⚠️ ClientBundle not available")

try:
    from dspy_bedrock_enhanced import DSPyBedrockEnhancedWrapper
    BEDROCK_ENHANCED_AVAILABLE = True
except ImportError:
    BEDROCK_ENHANCED_AVAILABLE = False
    print("⚠️ BedrockEnhancedWrapper not available")

# Optional Redis for session caching
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("⚠️ Redis not available - using memory cache")

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service status enumeration"""
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error" 
    DEGRADED = "degraded"


@dataclass
class ServiceHealth:
    """Health status for each service component"""
    credentials: bool = False
    s3: bool = False
    database: bool = False
    bedrock: bool = False
    cache: bool = False
    client_bundle: bool = False


class UnifiedServiceManager:
    """
    Central orchestrator for all AWS and database services.
    Eliminates the confusion of multiple managers with similar names.
    
    Single point of access for:
    - AWS credentials and sessions
    - S3 operations and file management  
    - Database connections and pooling
    - Bedrock LLM operations
    - Session caching for CLI/API hybrid mode
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern for global service access"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(UnifiedServiceManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
            
        self.status = ServiceStatus.INITIALIZING
        self.health = ServiceHealth()
        self.session_cache = {}
        self.cache_lock = threading.Lock()
        
        # Initialize existing managers
        self.credentials = None
        self.s3_manager = None  
        self.s3_utils = None
        self.db_manager = None
        self.client_bundle = None
        self.bedrock_wrapper = None
        self.redis_client = None
        
        # Initialize all services
        self._initialize_services()
        self._initialized = True
    
    def _initialize_services(self):
        """Initialize all available services"""
        try:
            # 1. Initialize credentials first
            self._init_credentials()
            
            # 2. Initialize S3 services
            self._init_s3_services()
            
            # 3. Initialize database services  
            self._init_database_services()
            
            # 4. Initialize Bedrock services
            self._init_bedrock_services()
            
            # 5. Initialize client bundle
            self._init_client_bundle()
            
            # 6. Initialize caching
            self._init_caching()
            
            # Determine overall status
            self._update_status()
            
            logger.info(f"✅ UnifiedServiceManager initialized - Status: {self.status.value}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize UnifiedServiceManager: {e}")
            self.status = ServiceStatus.ERROR
    
    def _init_credentials(self):
        """Initialize credential management with admin settings"""
        if CREDENTIAL_MANAGER_AVAILABLE:
            try:
                # Pass admin config path to credential manager
                admin_config_path = self._get_admin_config_path()
                if admin_config_path:
                    self.credentials = CredentialManager(str(admin_config_path))
                    logger.info(f"✅ Credentials manager initialized with admin config: {admin_config_path}")
                else:
                    self.credentials = CredentialManager()
                    logger.info("✅ Credentials manager initialized with default config")
                self.health.credentials = True
            except Exception as e:
                logger.warning(f"⚠️ Credential manager failed: {e}")
        else:
            logger.warning("⚠️ Credential manager not available")
    
    def _get_admin_config_path(self):
        """Get path to admin settings.yaml"""
        from pathlib import Path
        admin_paths = [
            Path("tidyllm/tidyllm/admin/settings.yaml"),
            Path("../tidyllm/tidyllm/admin/settings.yaml"),
            Path("../../tidyllm/tidyllm/admin/settings.yaml"),
            Path("tidyllm/admin/settings.yaml"),
        ]
        
        for path in admin_paths:
            if path.exists():
                return path
        return None
    
    def _load_admin_settings(self):
        """Load settings from admin folder"""
        import yaml
        config_path = self._get_admin_config_path()
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.warning(f"Failed to load admin settings: {e}")
        return {}
    
    def _init_s3_services(self):
        """Initialize S3 services with admin settings"""
        if S3_SESSION_MANAGER_AVAILABLE:
            try:
                # Load admin settings for S3 configuration
                admin_settings = self._load_admin_settings()
                
                # Get AWS config from credentials if available
                aws_config = {}
                if self.credentials and self.credentials.has_aws_credentials():
                    aws_config = self.credentials.get_aws_config()
                
                # Get domain/prefix from admin settings
                s3_admin_config = admin_settings.get('s3', {})
                aws_s3_admin_config = admin_settings.get('aws', {}).get('s3', {})
                
                # Determine region
                region = (s3_admin_config.get('region') or 
                         aws_config.get('region') or 
                         admin_settings.get('aws', {}).get('region') or 
                         'us-east-1')
                
                # Create S3SessionManager with admin config
                self.s3_manager = S3SessionManager(
                    region=region,
                    profile=aws_config.get('profile')
                )
                
                # Pass admin domain/prefix settings to S3 manager
                if hasattr(self.s3_manager, 'credentials'):
                    # Use admin settings for bucket and prefix (domain mapping)
                    self.s3_manager.credentials.default_bucket = (
                        s3_admin_config.get('bucket') or 
                        aws_s3_admin_config.get('default_bucket') or 
                        s3_config["bucket"]
                    )
                    self.s3_manager.credentials.default_prefix = (
                        s3_admin_config.get('prefix') or 
                        aws_s3_admin_config.get('default_prefix') or 
                        'pages/'
                    )
                    self.s3_manager.credentials.region = region
                    
                    # Pass KMS key from admin settings
                    kms_key = admin_settings.get('aws', {}).get('kms_key_id')
                    if kms_key:
                        self.s3_manager.credentials.kms_key_id = kms_key
                    
                    logger.info(f"🎯 S3 configured - bucket: {self.s3_manager.credentials.default_bucket}, prefix: {self.s3_manager.credentials.default_prefix}")
                
                self.s3_utils = S3Utils(self.s3_manager)
                self.health.s3 = True
                logger.info("✅ S3 services initialized with admin settings")
            except Exception as e:
                logger.warning(f"⚠️ S3 services failed: {e}")
        else:
            logger.warning("⚠️ S3 session manager not available")
    
    def _init_database_services(self):
        """Initialize database services with admin settings"""
        if CONNECTION_MANAGER_AVAILABLE:
            try:
                # Load admin settings for database configuration
                admin_settings = self._load_admin_settings()
                postgres_config = admin_settings.get('postgres', {})
                
                # Create connection config from admin settings
                if postgres_config:
                    from connection_manager import ConnectionConfig
                    config = ConnectionConfig(
                        host=postgres_config.get('host', 'localhost'),
                        port=postgres_config.get('port', 5432),
                        database=postgres_config.get('db_name', 'demo_db'),
                        username=postgres_config.get('db_user', 'demo_user'),
                        password=postgres_config.get('db_password', ''),
                        pool_size=postgres_config.get('connection_pool_size', 5),
                        timeout=postgres_config.get('timeout', 30),
                        ssl_mode=postgres_config.get('ssl_mode', 'prefer')
                    )
                    self.db_manager = DemoConnectionManager(config)
                    logger.info(f"✅ Database services initialized with admin config: {postgres_config.get('host')}")
                else:
                    self.db_manager = DemoConnectionManager()
                    logger.info("✅ Database services initialized with default config")
                    
                self.health.database = self.db_manager.is_connected()
            except Exception as e:
                logger.warning(f"⚠️ Database services failed: {e}")
        else:
            logger.warning("⚠️ Connection manager not available")
    
    def _init_bedrock_services(self):
        """Initialize Bedrock services with admin settings"""
        if BEDROCK_ENHANCED_AVAILABLE:
            try:
                # Load admin settings for Bedrock configuration
                admin_settings = self._load_admin_settings()
                bedrock_admin_config = admin_settings.get('aws', {}).get('bedrock', {})
                
                # Use credentials if available
                bedrock_config = None
                if self.credentials and self.credentials.has_aws_credentials():
                    aws_config = self.credentials.get_aws_config()
                    try:
                        from dspy_bedrock_enhanced import BedrockConfig
                        bedrock_config = BedrockConfig(
                            region=bedrock_admin_config.get('region', aws_config.get('region', 'us-east-1')),
                            profile=bedrock_admin_config.get('credentials', {}).get('profile', aws_config.get('profile')),
                            access_key_id=aws_config.get('access_key_id'),
                            secret_access_key=aws_config.get('secret_access_key'),
                            session_token=aws_config.get('session_token'),
                            default_model=bedrock_admin_config.get('default_model', 'anthropic.claude-3-sonnet-20240229-v1:0')
                        )
                        logger.info(f"🤖 Bedrock configured with model: {bedrock_config.default_model}")
                    except ImportError:
                        logger.warning("BedrockConfig not available, using basic config")
                        bedrock_config = {
                            'region': bedrock_admin_config.get('region', 'us-east-1'),
                            'default_model': bedrock_admin_config.get('default_model', 'anthropic.claude-3-sonnet-20240229-v1:0')
                        }
                
                # Pass admin settings path to wrapper if it supports it
                admin_config_path = self._get_admin_config_path()
                if admin_config_path:
                    self.bedrock_wrapper = DSPyBedrockEnhancedWrapper(
                        bedrock_config=bedrock_config,
                        settings_path=str(admin_config_path)
                    )
                else:
                    self.bedrock_wrapper = DSPyBedrockEnhancedWrapper(
                        bedrock_config=bedrock_config
                    )
                
                self.health.bedrock = True
                logger.info("✅ Bedrock services initialized with admin settings")
            except Exception as e:
                logger.warning(f"⚠️ Bedrock services failed: {e}")
        else:
            logger.warning("⚠️ Bedrock wrapper not available")
    
    def _init_client_bundle(self):
        """Initialize client bundle"""
        if CLIENT_BUNDLE_AVAILABLE:
            try:
                self.client_bundle = ClientBundle()
                self.health.client_bundle = True
                logger.info("✅ Client bundle initialized")
            except Exception as e:
                logger.warning(f"⚠️ Client bundle failed: {e}")
        else:
            logger.warning("⚠️ Client bundle not available")
    
    def _init_caching(self):
        """Initialize caching system"""
        try:
            if REDIS_AVAILABLE:
                # Try Redis first
                redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()  # Test connection
                self.health.cache = True
                logger.info("✅ Redis cache initialized")
            else:
                # Fall back to in-memory cache
                self.health.cache = True
                logger.info("✅ Memory cache initialized")
        except Exception as e:
            logger.warning(f"⚠️ Cache initialization failed: {e}")
            # Still use memory cache
            self.health.cache = True
    
    def _update_status(self):
        """Update overall service status"""
        if all([self.health.credentials, self.health.s3, self.health.database, 
                self.health.bedrock, self.health.cache]):
            self.status = ServiceStatus.READY
        elif any([self.health.credentials, self.health.s3, self.health.database]):
            self.status = ServiceStatus.DEGRADED
        else:
            self.status = ServiceStatus.ERROR
    
    # === UNIFIED API METHODS ===
    
    def get_s3_client(self, **kwargs):
        """Get S3 client - unified access point"""
        if self.s3_manager:
            return self.s3_manager.get_s3_client(**kwargs)
        elif self.client_bundle:
            return self.client_bundle.s3
        else:
            raise RuntimeError("No S3 client available")
    
    def get_bedrock_client(self):
        """Get Bedrock client - unified access point"""
        if self.bedrock_wrapper:
            return self.bedrock_wrapper.bedrock_client
        elif self.client_bundle:
            return self.client_bundle.bedrock
        else:
            raise RuntimeError("No Bedrock client available")
    
    def get_database_connection(self):
        """Get database connection - unified access point"""
        if self.db_manager:
            if self.db_manager.is_connected():
                return self.db_manager.get_connection()
        raise RuntimeError("No database connection available")
    
    def list_s3_objects(self, bucket: str, prefix: str = "", extension: str = None) -> List[str]:
        """List S3 objects - unified method"""
        if self.s3_utils:
            return self.s3_utils.list_objects_s3(bucket, prefix, extension)
        else:
            raise RuntimeError("S3 utilities not available")
    
    def store_data(self, key: str, data: Any, storage_type: str = "auto"):
        """Store data with automatic fallback"""
        if storage_type == "database" or (storage_type == "auto" and self.db_manager and self.db_manager.is_connected()):
            return self.db_manager.store_in_memory(key, data)
        else:
            return self._store_in_cache(key, data)
    
    def _store_in_cache(self, key: str, data: Any):
        """Store data in cache (Redis or memory)"""
        if self.redis_client:
            try:
                import pickle
                self.redis_client.set(key, pickle.dumps(data), ex=3600)  # 1 hour TTL
                return True
            except Exception as e:
                logger.warning(f"Redis storage failed, falling back to memory: {e}")
        
        # Memory fallback
        with self.cache_lock:
            self.session_cache[key] = {
                'data': data,
                'timestamp': datetime.now(),
                'ttl': 3600
            }
        return True
    
    def get_data(self, key: str, storage_type: str = "auto") -> Optional[Any]:
        """Get data with automatic fallback"""
        if storage_type == "database" or (storage_type == "auto" and self.db_manager and self.db_manager.is_connected()):
            return self.db_manager.get_from_memory(key)
        else:
            return self._get_from_cache(key)
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get data from cache (Redis or memory)"""
        if self.redis_client:
            try:
                import pickle
                data = self.redis_client.get(key)
                return pickle.loads(data) if data else None
            except Exception as e:
                logger.warning(f"Redis retrieval failed, trying memory: {e}")
        
        # Memory fallback
        with self.cache_lock:
            if key in self.session_cache:
                item = self.session_cache[key]
                if datetime.now() - item['timestamp'] < timedelta(seconds=item['ttl']):
                    return item['data']
                else:
                    del self.session_cache[key]  # Expired
        return None
    
    def create_bedrock_module(self, name: str, signature: str, model_id: str, **kwargs):
        """Create Bedrock module - unified method"""
        if self.bedrock_wrapper:
            return self.bedrock_wrapper.create_bedrock_module(
                name=name, signature=signature, model_id=model_id, **kwargs
            )
        else:
            raise RuntimeError("Bedrock wrapper not available")
    
    def estimate_bedrock_cost(self, model_id: str, input_tokens: int, output_tokens: int) -> Dict[str, float]:
        """Estimate Bedrock costs - unified method"""
        if self.bedrock_wrapper:
            return self.bedrock_wrapper.estimate_cost(model_id, input_tokens, output_tokens)
        else:
            raise RuntimeError("Bedrock wrapper not available")
    
    def test_all_connections(self) -> Dict[str, bool]:
        """Test all service connections"""
        results = {}
        
        # Test S3
        if self.s3_manager:
            s3_status = self.s3_manager.test_connection()
            results['s3'] = s3_status.get('success', False)
        else:
            results['s3'] = False
        
        # Test Database  
        if self.db_manager:
            results['database'] = self.db_manager.test_connection()
        else:
            results['database'] = False
        
        # Test Cache
        if self.redis_client:
            try:
                self.redis_client.ping()
                results['cache'] = True
            except:
                results['cache'] = False
        else:
            results['cache'] = True  # Memory cache always works
        
        return results
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status"""
        connection_tests = self.test_all_connections()
        
        return {
            'overall_status': self.status.value,
            'health': {
                'credentials': self.health.credentials,
                's3': self.health.s3,
                'database': self.health.database,
                'bedrock': self.health.bedrock,
                'cache': self.health.cache,
                'client_bundle': self.health.client_bundle
            },
            'connections': connection_tests,
            'available_services': {
                'credential_manager': CREDENTIAL_MANAGER_AVAILABLE,
                's3_session_manager': S3_SESSION_MANAGER_AVAILABLE,
                'connection_manager': CONNECTION_MANAGER_AVAILABLE,
                'client_bundle': CLIENT_BUNDLE_AVAILABLE,
                'bedrock_enhanced': BEDROCK_ENHANCED_AVAILABLE,
                'redis': REDIS_AVAILABLE
            },
            'cache_stats': {
                'memory_keys': len(self.session_cache),
                'redis_available': self.redis_client is not None
            }
        }
    
    def cleanup(self):
        """Clean up all services"""
        if self.db_manager:
            self.db_manager.close()
        if self.redis_client:
            self.redis_client.close()
        logger.info("✅ UnifiedServiceManager cleanup completed")


# === GLOBAL ACCESS FUNCTIONS ===

_unified_services = None

def get_services() -> UnifiedServiceManager:
    """
    Get the global unified services instance.
    This replaces all the scattered get_*_manager() functions.
    """
    global _unified_services
    if _unified_services is None:
        _unified_services = UnifiedServiceManager()
    return _unified_services

def get_s3_client(**kwargs):
    """Quick access to S3 client"""
    return get_services().get_s3_client(**kwargs)

def get_bedrock_client():
    """Quick access to Bedrock client"""
    return get_services().get_bedrock_client()

def get_database_connection():
    """Quick access to database connection"""
    return get_services().get_database_connection()

def store_session_data(key: str, data: Any):
    """Store session data for CLI/API hybrid mode"""
    return get_services().store_data(f"session_{key}", data)

def get_session_data(key: str):
    """Get session data for CLI/API hybrid mode"""
    return get_services().get_data(f"session_{key}")


if __name__ == "__main__":
    print("🚀 Testing TidyLLM Unified Services")
    
    # Test unified services
    services = get_services()
    status = services.get_service_status()
    
    print(f"\n📊 Service Status: {status['overall_status']}")
    print("\n🏥 Health Check:")
    for service, healthy in status['health'].items():
        status_icon = "✅" if healthy else "❌"
        print(f"  {status_icon} {service}: {healthy}")
    
    print("\n🔗 Connection Tests:")
    for service, connected in status['connections'].items():
        status_icon = "✅" if connected else "❌"  
        print(f"  {status_icon} {service}: {connected}")
    
    print("\n📦 Available Services:")
    for service, available in status['available_services'].items():
        status_icon = "✅" if available else "⚠️"
        print(f"  {status_icon} {service}: {available}")
    
    # Test session caching
    print("\n💾 Testing Session Cache:")
    store_session_data("test_key", {"message": "Hello from unified services!"})
    cached_data = get_session_data("test_key")
    print(f"  Cached data: {cached_data}")
    
    print(f"\n🎯 Unified Services Ready: {services.status.value}")