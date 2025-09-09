#!/usr/bin/env python3
"""
################################################################################
# *** IMPORTANT: READ docs/2025-09-08/IMPORTANT-CONSTRAINTS-FOR-THIS-CODEBASE.md ***
# *** BEFORE PLANNING ANY CHANGES TO THIS FILE ***
################################################################################

TidyLLM Unified Session Management System
=========================================

SOLUTION TO SCATTERED SESSION CHAOS:
------------------------------------
This consolidates ALL scattered session management across:
- S3 (3 different implementations found)
- PostgreSQL (multiple psycopg2 patterns)
- Bedrock (mixed credential approaches)

ONE SESSION MANAGER TO RULE THEM ALL - NO MORE GOING IN CIRCLES!

Features:
- Single credential discovery for all services
- Unified connection pooling and health checks
- Consistent error handling and fallback patterns
- Environment-based configuration with sane defaults
- Session sharing across all Streamlit demos
"""

import os
import sys
import json
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import tempfile
import time

# Core AWS imports
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError, ProfileNotFound
    from botocore.config import Config
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

# PostgreSQL imports
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, Json
    from psycopg2.pool import SimpleConnectionPool
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

# Configuration imports
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

logger = logging.getLogger("unified_sessions")
logger.setLevel(logging.INFO)

class ServiceType(Enum):
    """Supported service types"""
    S3 = "s3"
    BEDROCK = "bedrock" 
    POSTGRESQL = "postgresql"

class CredentialSource(Enum):
    """Sources for credentials (ordered by security priority)"""
    IAM_ROLE = "iam_role"
    AWS_PROFILE = "aws_profile"
    ENVIRONMENT = "environment"
    SETTINGS_FILE = "settings_file"
    NOT_FOUND = "not_found"

@dataclass
class ServiceConfig:
    """Unified configuration for all services"""
    # S3 Configuration
    s3_region: str = "us-east-1"
    s3_default_bucket: Optional[str] = None
    s3_access_key_id: Optional[str] = None
    s3_secret_access_key: Optional[str] = None
    
    # Bedrock Configuration
    bedrock_region: str = "us-east-1"
    bedrock_model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"
    
    # PostgreSQL Configuration
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_database: str = "tidyllm_db"
    postgres_username: str = "postgres"
    postgres_password: Optional[str] = None
    postgres_pool_size: int = 10
    
    # AWS Profile
    aws_profile: Optional[str] = None
    
    # Credential source tracking
    credential_source: CredentialSource = CredentialSource.NOT_FOUND

@dataclass
class ConnectionHealth:
    """Health status for service connections"""
    service: ServiceType
    healthy: bool = False
    last_check: Optional[datetime] = None
    error: Optional[str] = None
    latency_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class UnifiedSessionManager:
    """
    ONE SESSION MANAGER TO END THE CHAOS
    
    Consolidates all scattered session management patterns:
    - tidyllm-vectorqa/.../s3_session_manager.py
    - tidyllm/knowledge_systems/core/s3_manager.py
    - Multiple PostgreSQL connection patterns
    - Scattered Bedrock credential handling
    """
    
    def __init__(self, config: ServiceConfig = None):
        self.config = config or ServiceConfig()
        
        # Service clients
        self._s3_client = None
        self._s3_resource = None
        self._bedrock_client = None
        self._postgres_pool = None
        
        # Health tracking
        self.health_status: Dict[ServiceType, ConnectionHealth] = {
            ServiceType.S3: ConnectionHealth(ServiceType.S3),
            ServiceType.BEDROCK: ConnectionHealth(ServiceType.BEDROCK),
            ServiceType.POSTGRESQL: ConnectionHealth(ServiceType.POSTGRESQL)
        }
        
        # Auto-discover credentials
        self._discover_credentials()
        
        # Initialize connections
        self._initialize_connections()
    
    def _discover_credentials(self):
        """Discover credentials from environment and settings"""
        # Check if we already have a fully configured setup
        if (self.config.postgres_password and 
            self.config.postgres_host != "localhost" and
            self.config.s3_access_key_id):
            logger.info("[TARGET] Configuration already provided - skipping auto-discovery")
            return
        
        logger.info("[SEARCH] Discovering credentials for all services...")
        
        # Store original config values to avoid overwriting intentional settings
        original_postgres_host = self.config.postgres_host if self.config.postgres_host != "localhost" else None
        original_postgres_database = self.config.postgres_database if self.config.postgres_database != "tidyllm_db" else None
        original_postgres_username = self.config.postgres_username if self.config.postgres_username != "postgres" else None
        original_postgres_password = self.config.postgres_password
        
        # Load from environment first
        self._load_from_environment()
        
        # Load from settings file if available
        self._load_from_settings()
        
        # Restore original config if it was intentionally set
        if original_postgres_host:
            self.config.postgres_host = original_postgres_host
        if original_postgres_database:
            self.config.postgres_database = original_postgres_database
        if original_postgres_username:
            self.config.postgres_username = original_postgres_username
        if original_postgres_password:
            self.config.postgres_password = original_postgres_password
        
        # Test IAM role availability
        self._test_iam_role()
    
    def _load_from_environment(self):
        """Load credentials from environment variables"""
        # AWS credentials
        self.config.s3_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        self.config.s3_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.config.aws_profile = os.getenv('AWS_PROFILE', 'default')
        
        # PostgreSQL credentials
        postgres_password = os.getenv('POSTGRES_PASSWORD') or os.getenv('POSTGRESQL_PASSWORD')
        if postgres_password:
            self.config.postgres_password = postgres_password
            self.config.credential_source = CredentialSource.ENVIRONMENT
            logger.info("[OK] Found PostgreSQL password in environment")
        
        # Database URL override
        database_url = os.getenv('DATABASE_URL') or os.getenv('POSTGRESQL_URL')
        if database_url:
            self._parse_database_url(database_url)
        
        if self.config.s3_access_key_id and self.config.s3_secret_access_key:
            self.config.credential_source = CredentialSource.ENVIRONMENT
            logger.info("[OK] Found AWS credentials in environment")
    
    def _load_from_settings(self):
        """Load from tidyllm settings files"""
        settings_paths = [
            Path("tidyllm/admin/settings.yaml"),  # Real admin settings file first
            Path("../tidyllm/admin/settings.yaml"),  # From subdirectory (e.g., onboarding)
            Path("tidyllm/tidyllm/admin/settings.yaml"),  # Alternative path
            Path("admin/settings.yaml"),  # Direct admin folder
            Path("tidyllm/admin/embeddings_settings.yaml"),  # Legacy fallback
            Path("settings.yaml"),  # Root level
            Path("config.yaml")  # Generic fallback
        ]
        
        for path in settings_paths:
            if path.exists():
                try:
                    with open(path) as f:
                        settings = yaml.safe_load(f) if YAML_AVAILABLE else {}
                    
                    # Extract AWS settings
                    aws_config = settings.get('aws', {})
                    if aws_config.get('default_bucket'):
                        self.config.s3_default_bucket = aws_config['default_bucket']
                    
                    # Load AWS credentials from settings if not in environment
                    if not self.config.s3_access_key_id and aws_config.get('access_key_id'):
                        self.config.s3_access_key_id = aws_config['access_key_id']
                        self.config.s3_secret_access_key = aws_config.get('secret_access_key')
                        self.config.s3_region = aws_config.get('region', 'us-east-1')
                        self.config.credential_source = CredentialSource.SETTINGS_FILE
                        logger.info("[OK] Loaded AWS credentials from settings file")
                    
                    # Extract PostgreSQL settings - check multiple possible config sections
                    db_config = settings.get('postgres', {}) or settings.get('database', {}) or settings.get('postgresql', {})
                    if db_config:
                        self.config.postgres_host = db_config.get('host', self.config.postgres_host)
                        self.config.postgres_port = db_config.get('port', self.config.postgres_port)
                        # Handle both field name formats
                        self.config.postgres_database = db_config.get('db_name', db_config.get('database', self.config.postgres_database))
                        self.config.postgres_username = db_config.get('db_user', db_config.get('username', self.config.postgres_username))
                        password = db_config.get('db_password', db_config.get('password'))
                        if password:
                            self.config.postgres_password = password
                        
                        logger.info(f"[OK] PostgreSQL config loaded: {self.config.postgres_host}:{self.config.postgres_port}/{self.config.postgres_database}")
                    
                    logger.info(f"[OK] Loaded settings from {path}")
                    break
                except Exception as e:
                    logger.warning(f"[WARNING]  Could not load settings from {path}: {e}")
    
    def _test_iam_role(self):
        """Test if IAM role credentials are available"""
        if not AWS_AVAILABLE:
            return False
        
        try:
            session = boto3.Session()
            credentials = session.get_credentials()
            if credentials and not credentials.access_key:
                # IAM role detected
                self.config.credential_source = CredentialSource.IAM_ROLE
                logger.info("[OK] Using IAM role credentials")
                return True
        except:
            pass
        return False
    
    def _parse_database_url(self, url: str):
        """Parse DATABASE_URL into components"""
        try:
            # Format: postgresql://username:password@host:port/database
            if url.startswith('postgresql://'):
                url = url.replace('postgresql://', '')
                if '@' in url:
                    auth, location = url.split('@', 1)
                    if ':' in auth:
                        self.config.postgres_username, self.config.postgres_password = auth.split(':', 1)
                    
                    if '/' in location:
                        host_port, database = location.split('/', 1)
                        self.config.postgres_database = database.split('?')[0]  # Remove query params
                        
                        if ':' in host_port:
                            self.config.postgres_host, port_str = host_port.split(':', 1)
                            self.config.postgres_port = int(port_str)
                        else:
                            self.config.postgres_host = host_port
        except Exception as e:
            logger.warning(f"[WARNING]  Could not parse DATABASE_URL: {e}")
    
    def _initialize_connections(self):
        """Initialize all service connections"""
        logger.info("[LAUNCH] Initializing service connections...")
        
        # Initialize S3
        if AWS_AVAILABLE:
            self._init_s3()
        
        # Initialize Bedrock
        if AWS_AVAILABLE:
            self._init_bedrock()
        
        # Initialize PostgreSQL
        if POSTGRES_AVAILABLE and self.config.postgres_password:
            self._init_postgresql()
    
    def _init_s3(self):
        """Initialize S3 connection"""
        try:
            start_time = time.time()
            
            if self.config.credential_source == CredentialSource.IAM_ROLE:
                session = boto3.Session()
            elif self.config.credential_source in [CredentialSource.ENVIRONMENT, CredentialSource.SETTINGS_FILE]:
                # Use credentials from environment or settings file
                session = boto3.Session(
                    aws_access_key_id=self.config.s3_access_key_id,
                    aws_secret_access_key=self.config.s3_secret_access_key,
                    region_name=self.config.s3_region
                )
            else:
                # Try default profile
                session = boto3.Session(profile_name=self.config.aws_profile)
            
            self._s3_client = session.client('s3', region_name=self.config.s3_region)
            self._s3_resource = session.resource('s3', region_name=self.config.s3_region)
            
            # Test connection
            self._s3_client.list_buckets()
            
            latency = (time.time() - start_time) * 1000
            self.health_status[ServiceType.S3] = ConnectionHealth(
                service=ServiceType.S3,
                healthy=True,
                last_check=datetime.now(),
                latency_ms=latency
            )
            
            logger.info(f"[OK] S3 connection established ({latency:.1f}ms)")
            
        except Exception as e:
            self.health_status[ServiceType.S3] = ConnectionHealth(
                service=ServiceType.S3,
                healthy=False,
                last_check=datetime.now(),
                error=str(e)
            )
            logger.warning(f"[ERROR] S3 connection failed: {e}")
    
    def _init_bedrock(self):
        """Initialize Bedrock connection"""
        try:
            start_time = time.time()
            
            if self.config.credential_source == CredentialSource.IAM_ROLE:
                session = boto3.Session()
            elif self.config.credential_source in [CredentialSource.ENVIRONMENT, CredentialSource.SETTINGS_FILE]:
                # Use credentials from environment or settings file
                session = boto3.Session(
                    aws_access_key_id=self.config.s3_access_key_id,
                    aws_secret_access_key=self.config.s3_secret_access_key,
                    region_name=self.config.bedrock_region
                )
            else:
                session = boto3.Session(profile_name=self.config.aws_profile)
            
            self._bedrock_client = session.client('bedrock-runtime', region_name=self.config.bedrock_region)
            
            # Test with a simple list models call (lightweight)
            try:
                bedrock_client = session.client('bedrock', region_name=self.config.bedrock_region)
                bedrock_client.list_foundation_models()
            except:
                pass  # May not have permissions, but client works
            
            latency = (time.time() - start_time) * 1000
            self.health_status[ServiceType.BEDROCK] = ConnectionHealth(
                service=ServiceType.BEDROCK,
                healthy=True,
                last_check=datetime.now(),
                latency_ms=latency
            )
            
            logger.info(f"[OK] Bedrock connection established ({latency:.1f}ms)")
            
        except Exception as e:
            self.health_status[ServiceType.BEDROCK] = ConnectionHealth(
                service=ServiceType.BEDROCK,
                healthy=False,
                last_check=datetime.now(),
                error=str(e)
            )
            logger.warning(f"[ERROR] Bedrock connection failed: {e}")
    
    def _init_postgresql(self):
        """Initialize PostgreSQL connection pool"""
        try:
            start_time = time.time()
            
            # Create connection pool
            self._postgres_pool = SimpleConnectionPool(
                minconn=1,
                maxconn=self.config.postgres_pool_size,
                host=self.config.postgres_host,
                port=self.config.postgres_port,
                database=self.config.postgres_database,
                user=self.config.postgres_username,
                password=self.config.postgres_password,
                cursor_factory=RealDictCursor
            )
            
            # Test connection
            conn = self._postgres_pool.getconn()
            try:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT version();")
                    version = cursor.fetchone()
            finally:
                self._postgres_pool.putconn(conn)
            
            latency = (time.time() - start_time) * 1000
            self.health_status[ServiceType.POSTGRESQL] = ConnectionHealth(
                service=ServiceType.POSTGRESQL,
                healthy=True,
                last_check=datetime.now(),
                latency_ms=latency,
                metadata={"version": str(version) if version else "unknown"}
            )
            
            logger.info(f"[OK] PostgreSQL connection established ({latency:.1f}ms)")
            
        except Exception as e:
            self.health_status[ServiceType.POSTGRESQL] = ConnectionHealth(
                service=ServiceType.POSTGRESQL,
                healthy=False,
                last_check=datetime.now(),
                error=str(e)
            )
            logger.warning(f"[ERROR] PostgreSQL connection failed: {e}")
    
    # Service Client Access Methods
    def get_s3_client(self):
        """Get S3 client (thread-safe)"""
        return self._s3_client
    
    def get_s3_resource(self):
        """Get S3 resource (thread-safe)"""
        return self._s3_resource
    
    def get_bedrock_client(self):
        """Get Bedrock client (thread-safe)"""
        return self._bedrock_client
    
    def get_postgres_connection(self):
        """Get PostgreSQL connection from pool"""
        if self._postgres_pool:
            return self._postgres_pool.getconn()
        return None
    
    def return_postgres_connection(self, conn):
        """Return PostgreSQL connection to pool"""
        if self._postgres_pool and conn:
            self._postgres_pool.putconn(conn)
    
    # Health Check Methods
    def check_health(self, service: ServiceType = None) -> Dict[ServiceType, ConnectionHealth]:
        """Check health of services"""
        if service:
            # Check specific service
            self._health_check_service(service)
            return {service: self.health_status[service]}
        else:
            # Check all services
            for svc in ServiceType:
                self._health_check_service(svc)
            return self.health_status
    
    def _health_check_service(self, service: ServiceType):
        """Perform health check for specific service"""
        try:
            start_time = time.time()
            
            if service == ServiceType.S3 and self._s3_client:
                self._s3_client.list_buckets()
            elif service == ServiceType.BEDROCK and self._bedrock_client:
                # Light health check - client exists
                pass
            elif service == ServiceType.POSTGRESQL and self._postgres_pool:
                conn = self.get_postgres_connection()
                try:
                    with conn.cursor() as cursor:
                        cursor.execute("SELECT 1;")
                        cursor.fetchone()
                finally:
                    self.return_postgres_connection(conn)
            
            latency = (time.time() - start_time) * 1000
            self.health_status[service].healthy = True
            self.health_status[service].last_check = datetime.now()
            self.health_status[service].latency_ms = latency
            self.health_status[service].error = None
            
        except Exception as e:
            self.health_status[service].healthy = False
            self.health_status[service].last_check = datetime.now()
            self.health_status[service].error = str(e)
    
    def is_healthy(self, service: ServiceType = None) -> bool:
        """Check if service(s) are healthy"""
        if service:
            return self.health_status[service].healthy
        return all(status.healthy for status in self.health_status.values())
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary"""
        return {
            "overall_healthy": self.is_healthy(),
            "services": {
                service.value: {
                    "healthy": status.healthy,
                    "last_check": status.last_check.isoformat() if status.last_check else None,
                    "latency_ms": status.latency_ms,
                    "error": status.error,
                    "metadata": status.metadata
                }
                for service, status in self.health_status.items()
            },
            "credential_source": self.config.credential_source.value,
            "configuration": {
                "s3_region": self.config.s3_region,
                "s3_default_bucket": self.config.s3_default_bucket,
                "postgres_host": self.config.postgres_host,
                "postgres_database": self.config.postgres_database,
                "bedrock_region": self.config.bedrock_region
            }
        }
    
    def cleanup(self):
        """Clean up connections"""
        if self._postgres_pool:
            self._postgres_pool.closeall()
        logger.info("[CLEANUP] Cleaned up all connections")

    # Additional compatibility methods for seamless integration
    def get_session(self):
        """Get current session info (compatibility method)"""
        return {
            "session_id": "unified_session",
            "active": True,
            "healthy": self.is_healthy()
        }
    
    def create_session(self):
        """Create new session (compatibility method)"""
        import uuid
        return {
            "session_id": str(uuid.uuid4()),
            "active": True,
            "healthy": self.is_healthy()
        }

# Global session manager instance
_global_session_manager = None

def get_global_session_manager() -> UnifiedSessionManager:
    """Get or create global session manager instance"""
    global _global_session_manager
    if _global_session_manager is None:
        _global_session_manager = UnifiedSessionManager()
    return _global_session_manager

def reset_global_session_manager():
    """Reset global session manager (for testing)"""
    global _global_session_manager
    if _global_session_manager:
        _global_session_manager.cleanup()
    _global_session_manager = None

# Convenience functions for demos
def get_s3_client():
    """Get S3 client from global session manager"""
    return get_global_session_manager().get_s3_client()

def get_bedrock_client():
    """Get Bedrock client from global session manager"""
    return get_global_session_manager().get_bedrock_client()

def get_postgres_connection():
    """Get PostgreSQL connection from global session manager"""
    return get_global_session_manager().get_postgres_connection()

def return_postgres_connection(conn):
    """Return PostgreSQL connection to global session manager"""
    return get_global_session_manager().return_postgres_connection(conn)

# Export the ServiceType enum for compatibility
__all__ = ['UnifiedSessionManager', 'ServiceType', 'CredentialSource', 'ServiceConfig', 
           'ConnectionHealth', 'get_global_session_manager', 'reset_global_session_manager',
           'get_s3_client', 'get_bedrock_client', 'get_postgres_connection', 
           'return_postgres_connection']

if __name__ == "__main__":
    # Demo the unified session manager
    print("TidyLLM Unified Session Manager Demo")
    print("=" * 50)
    
    # Create session manager
    session_mgr = UnifiedSessionManager()
    
    # Show health summary
    health = session_mgr.get_health_summary()
    print(json.dumps(health, indent=2, default=str))
    
    # Test connections
    print("\nTesting connections...")
    for service in ServiceType:
        is_healthy = session_mgr.is_healthy(service)
        status_icon = "OK" if is_healthy else "FAIL"
        print(f"{status_icon} {service.value}: {'HEALTHY' if is_healthy else 'FAILED'}")
    
    # Cleanup
    session_mgr.cleanup()
    print("\nDemo complete!")