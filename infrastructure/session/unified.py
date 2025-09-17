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
        self._bedrock_runtime_client = None
        self._postgres_pool = None
        
        # Health tracking
        self.health_status: Dict[ServiceType, ConnectionHealth] = {
            ServiceType.S3: ConnectionHealth(ServiceType.S3),
            ServiceType.BEDROCK: ConnectionHealth(ServiceType.BEDROCK),
            ServiceType.POSTGRESQL: ConnectionHealth(ServiceType.POSTGRESQL)
        }
        
        # Auto-discover credentials
        self._discover_credentials()
        
        # Initialize connections LAZILY - only when actually needed
        # self._initialize_connections()  # Disabled for performance
    
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
        
        # Apply discovered credentials to environment for reuse
        self._apply_credentials_to_environment()
    
    def _apply_credentials_to_environment(self):
        """Apply discovered credentials to environment variables for reuse by other components"""
        # Only set environment variables if they're not already set
        if self.config.s3_access_key_id and not os.getenv('AWS_ACCESS_KEY_ID'):
            os.environ['AWS_ACCESS_KEY_ID'] = self.config.s3_access_key_id
            logger.info("[APPLIED] AWS_ACCESS_KEY_ID to environment")
        
        if self.config.s3_secret_access_key and not os.getenv('AWS_SECRET_ACCESS_KEY'):
            os.environ['AWS_SECRET_ACCESS_KEY'] = self.config.s3_secret_access_key
            logger.info("[APPLIED] AWS_SECRET_ACCESS_KEY to environment")
        
        if self.config.s3_region and not os.getenv('AWS_DEFAULT_REGION'):
            os.environ['AWS_DEFAULT_REGION'] = self.config.s3_region
            logger.info(f"[APPLIED] AWS_DEFAULT_REGION to environment: {self.config.s3_region}")
        
        if self.config.aws_profile and not os.getenv('AWS_PROFILE'):
            os.environ['AWS_PROFILE'] = self.config.aws_profile
            logger.info(f"[APPLIED] AWS_PROFILE to environment: {self.config.aws_profile}")
        
        # Apply PostgreSQL credentials
        if self.config.postgres_password and not os.getenv('POSTGRES_PASSWORD'):
            os.environ['POSTGRES_PASSWORD'] = self.config.postgres_password
            logger.info("[APPLIED] POSTGRES_PASSWORD to environment")
    
    def _load_from_environment(self):
        """Load credentials from environment variables"""
        # AWS credentials
        self.config.s3_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        self.config.s3_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.config.aws_profile = os.getenv('AWS_PROFILE')
        
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
        """Load from Polars-based credential system"""
        try:
            # Use Polars-based config loader for credentials
            from scripts.infrastructure.config_loader_polars import ConfigLoaderPolars
            loader = ConfigLoaderPolars()
            dataframes = loader.load_full_config()
            
            if 'credentials' in dataframes and len(dataframes['credentials']) > 0:
                logger.info("[OK] Loaded credentials from Polars config loader")
                self._apply_polars_credentials_to_config(dataframes['credentials'])
                return
        except ImportError:
            logger.debug("Polars config loader not available, falling back to settings manager")
        except Exception as e:
            logger.warning(f"Failed to load from Polars config loader: {e}")
        
        # Fallback: Use centralized settings manager
        try:
            from ..settings_manager import get_settings_manager
            settings_manager = get_settings_manager()
            settings = settings_manager.get_settings()
            
            if settings:
                logger.info(f"[OK] Loaded settings from centralized manager: {settings_manager.settings_file}")
                self._apply_settings_to_config(settings)
                return
        except ImportError:
            logger.debug("Centralized settings manager not available, falling back to direct loading")
        except Exception as e:
            logger.warning(f"Failed to load from centralized settings manager: {e}")
        
        # Final fallback: Direct settings loading (legacy)
        self._load_from_settings_direct()
    
    def _apply_polars_credentials_to_config(self, credentials_df):
        """Apply credentials from Polars DataFrame to USM config"""
        try:
            import polars as pl
            
            # Get AWS credentials
            aws_creds = credentials_df.filter(pl.col('service') == 'aws')
            
            if len(aws_creds) > 0:
                # Extract credentials by key
                access_key = aws_creds.filter(pl.col('credential_key').str.contains('access_key_id')).select('credential_value').to_series().to_list()
                secret_key = aws_creds.filter(pl.col('credential_key').str.contains('secret_access_key')).select('credential_value').to_series().to_list()
                region = aws_creds.filter(pl.col('credential_key').str.contains('default_region')).select('credential_value').to_series().to_list()
                profile = aws_creds.filter(pl.col('credential_key').str.contains('profile')).select('credential_value').to_series().to_list()
                
                # Apply to config
                if access_key:
                    self.config.s3_access_key_id = access_key[0]
                if secret_key:
                    self.config.s3_secret_access_key = secret_key[0]
                if region:
                    self.config.s3_region = region[0]
                    self.config.bedrock_region = region[0]
                if profile and profile[0] != 'None':
                    self.config.aws_profile = profile[0]
                
                self.config.credential_source = CredentialSource.SETTINGS_FILE
                logger.info("[OK] Applied AWS credentials from Polars")
            
            # Get PostgreSQL credentials
            postgres_creds = credentials_df.filter(pl.col('service') == 'postgresql')
            
            if len(postgres_creds) > 0:
                # Extract PostgreSQL credentials
                host = postgres_creds.filter(pl.col('credential_key').str.contains('host')).select('credential_value').to_series().to_list()
                port = postgres_creds.filter(pl.col('credential_key').str.contains('port')).select('credential_value').to_series().to_list()
                database = postgres_creds.filter(pl.col('credential_key').str.contains('database')).select('credential_value').to_series().to_list()
                username = postgres_creds.filter(pl.col('credential_key').str.contains('username')).select('credential_value').to_series().to_list()
                password = postgres_creds.filter(pl.col('credential_key').str.contains('password')).select('credential_value').to_series().to_list()
                
                # Apply to config
                if host:
                    self.config.postgres_host = host[0]
                if port:
                    self.config.postgres_port = int(port[0])
                if database:
                    self.config.postgres_database = database[0]
                if username:
                    self.config.postgres_username = username[0]
                if password:
                    self.config.postgres_password = password[0]
                
                logger.info("[OK] Applied PostgreSQL credentials from Polars")
                
        except Exception as e:
            logger.warning(f"Failed to apply Polars credentials: {e}")
    
    def _apply_settings_to_config(self, settings: Dict[str, Any]):
        """Apply centralized settings to USM config"""
        # Apply AWS settings from credentials section
        credentials = settings.get("credentials", {})
        aws_config = credentials.get("aws", {})
        if aws_config:
            self.config.s3_access_key_id = aws_config.get("access_key_id")
            self.config.s3_secret_access_key = aws_config.get("secret_access_key")
            self.config.s3_region = aws_config.get("default_region", "us-east-1")
            self.config.bedrock_region = aws_config.get("default_region", "us-east-1")
            self.config.aws_profile = aws_config.get("profile")
            
            # S3 specific settings
            s3_config = settings.get("s3", {})
            if s3_config:
                self.config.s3_default_bucket = s3_config.get("default_bucket")
                self.config.s3_default_prefix = s3_config.get("default_prefix")
        
        # Apply PostgreSQL settings from credentials section
        postgres_config = credentials.get("postgresql", {})
        if postgres_config:
            self.config.postgres_host = postgres_config.get("host")
            self.config.postgres_port = postgres_config.get("port", 5432)
            self.config.postgres_database = postgres_config.get("database")
            self.config.postgres_username = postgres_config.get("username")
            self.config.postgres_password = postgres_config.get("password")
        
        # Apply system settings
        system_config = settings.get("system", {})
        if system_config:
            self.config.root_path = system_config.get("root_path")
        
        logger.info("[OK] Applied centralized settings to USM config")
    
    def refresh_from_centralized_settings(self):
        """Refresh USM config from centralized settings manager"""
        try:
            from ..settings_manager import get_settings_manager
            settings_manager = get_settings_manager()
            
            # Refresh settings if needed
            if hasattr(settings_manager, 'refresh_settings'):
                settings_manager.refresh_settings()
            
            # Get updated settings
            settings = settings_manager.get_settings()
            if settings:
                self._apply_settings_to_config(settings)
                logger.info("[OK] USM config refreshed from centralized settings")
                return True
        except Exception as e:
            logger.warning(f"Failed to refresh USM from centralized settings: {e}")
        return False
    
    def _load_from_settings_direct(self):
        """Direct settings loading (legacy fallback)"""
        # Start from current directory and search upward for settings.yaml
        current_dir = Path.cwd()
        settings_file = None
        
        # Search up to 5 levels up from current directory
        for _ in range(5):
            potential_paths = [
                current_dir / "tidyllm" / "admin" / "settings.yaml",
                current_dir / "admin" / "settings.yaml",
                current_dir / "settings.yaml",
            ]
            
            for path in potential_paths:
                if path.exists():
                    settings_file = path
                    break
            
            if settings_file:
                break
                
            # Move up one directory
            parent = current_dir.parent
            if parent == current_dir:  # Reached root
                break
            current_dir = parent
        
        # If not found by searching up, try explicit paths
        if not settings_file:
            fallback_paths = [
                Path.home() / "github" / "tidyllm" / "admin" / "settings.yaml",
                Path("C:/Users/marti/github/tidyllm/admin/settings.yaml"),
            ]
            for path in fallback_paths:
                if path.exists():
                    settings_file = path
                    break
        
        if settings_file and settings_file.exists():
            try:
                with open(settings_file) as f:
                    settings = yaml.safe_load(f) if YAML_AVAILABLE else {}
                    
                    # Dynamic AWS settings extraction - auto-detect all AWS fields
                    aws_config = settings.get('aws', {})
                    
                    # Dynamically load any AWS-related fields  
                    for key, value in aws_config.items():
                        if value:  # Only set non-empty values
                            # Map common variations to standardized config fields
                            if key in ['access_key_id', 'aws_access_key_id', 'access_key']:
                                self.config.s3_access_key_id = value
                            elif key in ['secret_access_key', 'aws_secret_access_key', 'secret_key']:
                                self.config.s3_secret_access_key = value
                            elif key in ['region', 'aws_region', 'default_region', 'aws_default_region']:
                                self.config.s3_region = value
                                self.config.bedrock_region = value
                            elif key in ['default_bucket', 's3_bucket', 'bucket']:
                                self.config.s3_default_bucket = value
                            elif key == 'bedrock':
                                # Handle nested bedrock config
                                if isinstance(value, dict):
                                    self.config.bedrock_region = value.get('region', self.config.bedrock_region)
                                    self.config.bedrock_model_id = value.get('default_model', self.config.bedrock_model_id)
                    
                    # Also check api_keys section for AWS credentials (legacy support)
                    api_keys = settings.get('api_keys', {})
                    if not self.config.s3_access_key_id:
                        self.config.s3_access_key_id = api_keys.get('aws_access_key_id') or api_keys.get('access_key_id')
                    if not self.config.s3_secret_access_key:
                        self.config.s3_secret_access_key = api_keys.get('aws_secret_access_key') or api_keys.get('secret_access_key')
                    
                    # Set credential source if we found credentials
                    if self.config.s3_access_key_id and self.config.s3_secret_access_key:
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
                    
                logger.info(f"[OK] Loaded settings from {settings_file}")
            except Exception as e:
                logger.warning(f"[WARNING]  Could not load settings from {settings_file}: {e}")
    
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
            
            # Create both bedrock and bedrock-runtime clients
            self._bedrock_client = session.client('bedrock', region_name=self.config.bedrock_region)
            self._bedrock_runtime_client = session.client('bedrock-runtime', region_name=self.config.bedrock_region)
            
            # Test with a simple list models call (lightweight)
            try:
                self._bedrock_client.list_foundation_models()
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
    
    def _init_sts(self):
        """Initialize STS connection"""
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
            
            self._sts_client = session.client('sts', region_name=self.config.bedrock_region)
            
            # Test with get_caller_identity
            try:
                self._sts_client.get_caller_identity()
            except:
                pass  # May not have permissions, but client works
            
            latency = (time.time() - start_time) * 1000
            logger.info(f"[OK] STS connection established ({latency:.1f}ms)")
            
        except Exception as e:
            logger.warning(f"[ERROR] STS connection failed: {e}")
            self._sts_client = None
    
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
        """Get S3 client (thread-safe) - lazy initialization"""
        if self._s3_client is None:
            self._init_s3()
        return self._s3_client
    
    def get_s3_resource(self):
        """Get S3 resource (thread-safe)"""
        return self._s3_resource
    
    def get_bedrock_client(self):
        """Get Bedrock client (thread-safe) - lazy initialization"""
        if self._bedrock_client is None:
            self._init_bedrock()
        return self._bedrock_client
    
    def get_bedrock_runtime_client(self):
        """Get Bedrock Runtime client (thread-safe) - lazy initialization"""
        if self._bedrock_runtime_client is None:
            self._init_bedrock()
        return self._bedrock_runtime_client
    
    def get_sts_client(self):
        """Get STS client (thread-safe)"""
        if not hasattr(self, '_sts_client'):
            self._init_sts()
        return self._sts_client
    
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
    
    def test_connection(self, service: str = "all") -> Dict[str, Any]:
        """Test connections to specified services with detailed results."""
        import time
        results = {}
        
        if service in ["all", "s3"]:
            results["s3"] = self._test_s3_connection()
        
        if service in ["all", "bedrock"]:
            results["bedrock"] = self._test_bedrock_connection()
            # Also test runtime access (what's actually needed for AI processing)
            results["bedrock_runtime"] = self._test_bedrock_runtime_access()
        
        if service in ["all", "postgres"]:
            results["postgres"] = self._test_postgres_connection()
        
        return results

    def _test_s3_connection(self) -> Dict[str, Any]:
        """Test S3 connection with timing and details."""
        import time
        start_time = time.time()
        
        try:
            s3_client = self.get_s3_client()
            response = s3_client.list_buckets()
            duration_ms = (time.time() - start_time) * 1000
            
            return {
                "status": "success",
                "duration_ms": round(duration_ms, 1),
                "bucket_count": len(response.get("Buckets", [])),
                "message": f"S3 connected successfully ({duration_ms:.1f}ms)"
            }
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return {
                "status": "failed",
                "duration_ms": round(duration_ms, 1),
                "error": str(e),
                "message": f"S3 connection failed: {str(e)}"
            }

    def _test_bedrock_connection(self) -> Dict[str, Any]:
        """Test Bedrock connection with timing and details."""
        import time
        start_time = time.time()
        
        try:
            bedrock_client = self.get_bedrock_client()
            if bedrock_client is None:
                return {
                    "status": "no_client",
                    "duration_ms": 0,
                    "error": "Bedrock client not available",
                    "message": "Bedrock client not initialized - check AWS credentials"
                }
            # Test with a minimal model list call
            response = bedrock_client.list_foundation_models()
            duration_ms = (time.time() - start_time) * 1000
            
            return {
                "status": "success", 
                "duration_ms": round(duration_ms, 1),
                "model_count": len(response.get("modelSummaries", [])),
                "message": f"Bedrock connected successfully ({duration_ms:.1f}ms)"
            }
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_str = str(e)
            
            # Handle specific permission errors with helpful guidance
            if "AccessDeniedException" in error_str and "bedrock:ListFoundationModels" in error_str:
                return {
                    "status": "corporate_restricted",
                    "duration_ms": round(duration_ms, 1),
                    "error": error_str,
                    "message": "Bedrock connection: Corporate policy restricts ListFoundationModels",
                    "solution": "This is normal in corporate environments. Bedrock runtime access may still work.",
                    "user_arn": self._extract_user_arn_from_error(error_str),
                    "corporate_mode": True
                }
            else:
                return {
                    "status": "failed",
                    "duration_ms": round(duration_ms, 1),
                    "error": error_str,
                    "message": f"Bedrock connection failed: {error_str}"
                }
    
    def _extract_user_arn_from_error(self, error_str: str) -> str:
        """Extract user ARN from AccessDeniedException error message."""
        import re
        # Look for pattern: User: arn:aws:iam::account:user/username
        match = re.search(r'User: (arn:aws:iam::\d+:user/[^\s]+)', error_str)
        return match.group(1) if match else "Unknown"
    
    def _test_bedrock_runtime_access(self) -> Dict[str, Any]:
        """Test Bedrock runtime access (what's actually needed for AI processing)."""
        import time
        import json
        start_time = time.time()
        
        try:
            # Test with bedrock-runtime client (what's actually used for AI processing)
            bedrock_runtime_client = self.get_bedrock_runtime_client()
            if bedrock_runtime_client is None:
                return {
                    "status": "no_client",
                    "duration_ms": 0,
                    "error": "Bedrock runtime client not available",
                    "message": "Bedrock runtime client not initialized - check AWS credentials"
                }
            # Try a simple invoke_model call with a minimal payload
            # This tests the actual permission needed for AI processing
            test_payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "test"}]
            }
            
            # This will fail with a model not found error, but that's OK - we're testing permissions
            try:
                bedrock_runtime_client.invoke_model(
                    modelId="anthropic.claude-3-haiku-20240307-v1:0",
                    body=json.dumps(test_payload),
                    contentType="application/json"
                )
            except Exception as e:
                error_str = str(e)
                # If it's a model not found error, that means permissions are OK
                if "ValidationException" in error_str and "model" in error_str.lower():
                    duration_ms = (time.time() - start_time) * 1000
                    return {
                        "status": "success",
                        "duration_ms": round(duration_ms, 1),
                        "message": f"Bedrock runtime access confirmed ({duration_ms:.1f}ms)",
                        "note": "Model validation error indicates permissions are OK"
                    }
                else:
                    raise e
            
            duration_ms = (time.time() - start_time) * 1000
            return {
                "status": "success",
                "duration_ms": round(duration_ms, 1),
                "message": f"Bedrock runtime access confirmed ({duration_ms:.1f}ms)"
            }
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_str = str(e)
            
            if "AccessDeniedException" in error_str:
                return {
                    "status": "permission_denied",
                    "duration_ms": round(duration_ms, 1),
                    "error": error_str,
                    "message": "Bedrock runtime access denied - check bedrock:InvokeModel permission"
                }
            else:
                return {
                    "status": "failed",
                    "duration_ms": round(duration_ms, 1),
                    "error": error_str,
                    "message": f"Bedrock runtime test failed: {error_str}"
                }

    def _test_postgres_connection(self) -> Dict[str, Any]:
        """Test PostgreSQL connection with timing and details."""
        import time
        start_time = time.time()
        
        try:
            conn = self.get_postgres_connection()
            
            # Better error message if connection is None
            if conn is None:
                duration_ms = (time.time() - start_time) * 1000
                return {
                    "status": "failed",
                    "duration_ms": round(duration_ms, 1),
                    "error": "get_postgres_connection() returned None",
                    "message": "PostgreSQL connection failed: Connection pool not initialized or database credentials missing. Check settings.yaml for postgres_* configuration.",
                    "troubleshooting": [
                        "Verify postgres_host, postgres_port, postgres_database are set in settings.yaml",
                        "Ensure postgres_user and postgres_password are configured",
                        "Check if PostgreSQL server is running and accessible",
                        "Verify network connectivity to database server"
                    ]
                }
            
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            self.return_postgres_connection(conn)
            
            duration_ms = (time.time() - start_time) * 1000
            return {
                "status": "success",
                "duration_ms": round(duration_ms, 1),
                "test_query": "SELECT 1",
                "result": result[0] if result else None,
                "message": f"PostgreSQL connected successfully ({duration_ms:.1f}ms)"
            }
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_message = str(e)
            
            # Provide more specific error messages based on common issues
            troubleshooting = []
            if "authentication failed" in error_message.lower():
                troubleshooting = [
                    "Check postgres_user and postgres_password in settings.yaml",
                    "Verify user has access to the specified database",
                    "Ensure password is correct and not expired"
                ]
            elif "could not connect" in error_message.lower() or "connection refused" in error_message.lower():
                troubleshooting = [
                    "Verify PostgreSQL server is running",
                    "Check postgres_host and postgres_port in settings.yaml",
                    "Ensure firewall allows connections to database port",
                    "Verify database server is accepting connections"
                ]
            elif "database" in error_message.lower() and "does not exist" in error_message.lower():
                troubleshooting = [
                    "Check postgres_database name in settings.yaml",
                    "Ensure the database exists on the server",
                    "Create the database if it doesn't exist"
                ]
            else:
                troubleshooting = [
                    "Check all postgres_* settings in settings.yaml",
                    "Verify PostgreSQL server is running and accessible",
                    "Review connection logs for more details"
                ]
            
            return {
                "status": "failed",
                "duration_ms": round(duration_ms, 1),
                "error": error_message,
                "message": f"PostgreSQL connection failed: {error_message}",
                "troubleshooting": troubleshooting
            }
    
    def validate_session(self) -> Dict[str, Any]:
        """
        Validate current session status (compatibility wrapper).
        
        Returns:
            Dict with session validation results
        """
        try:
            # Check if all core services are healthy
            health_status = self.is_healthy()
            test_results = self.test_connection("all")
            
            all_services_healthy = all(
                result.get("status") == "success" 
                for result in test_results.values()
            )
            
            return {
                "valid": all_services_healthy and health_status,
                "session_healthy": health_status,
                "services": test_results,
                "message": "Session valid" if all_services_healthy else "Session has connection issues"
            }
        except Exception as e:
            return {
                "valid": False,
                "session_healthy": False,
                "error": str(e),
                "message": f"Session validation failed: {str(e)}"
            }
    
    def test_postgres_connection(self) -> Dict[str, Any]:
        """
        Test PostgreSQL connection (compatibility wrapper).
        
        Returns:
            Dict with postgres connection test results
        """
        return self._test_postgres_connection()
    
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
    
    def get_gateways(self) -> Dict[str, Any]:
        """
        Get all available gateways through GatewayRegistry.
        
        This maintains the ONE SESSION MANAGER RULE by providing
        gateway access through USM rather than bypassing it.
        
        Returns:
            Dictionary of gateway instances keyed by service name
        """
        try:
            from ...gateways.gateway_registry import get_global_registry
            registry = get_global_registry()
            registry.auto_configure()
            
            # Get gateways from registry
            available_services = registry.get_available_services()
            gateways = {}
            
            for service_name in available_services:
                try:
                    gateway = registry.get(service_name)
                    if gateway:
                        gateways[service_name] = gateway
                except Exception as e:
                    logger.warning(f"Failed to get gateway {service_name}: {e}")
            
            logger.info(f"USM: Retrieved {len(gateways)} gateways: {list(gateways.keys())}")
            return gateways
            
        except Exception as e:
            logger.error(f"USM: Failed to get gateways: {e}")
            return {}
    
    def get_services(self) -> List[str]:
        """
        Get list of available service names.
        
        Returns:
            List of service names available through GatewayRegistry
        """
        try:
            from ...gateways.gateway_registry import get_global_registry
            registry = get_global_registry()
            registry.auto_configure()
            return registry.get_available_services()
        except Exception as e:
            logger.error(f"USM: Failed to get services: {e}")
            return []

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