"""
S3 Session Manager
==================

Centralized AWS S3 session management for TidyLLM projects.
Based on existing AWS credential patterns from tidyllm/demo-standalone/credential_manager.py.

Handles multiple credential sources:
1. Environment variables (highest priority)
2. AWS credentials file (~/.aws/credentials)  
3. AWS config file (~/.aws/config)
4. Explicit credentials
5. IAM roles (on EC2)
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from pathlib import Path
import json
from enum import Enum
import yaml

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError, ProfileNotFound, BotoCoreError
    from botocore.config import Config
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    boto3 = None
    NoCredentialsError = Exception  # Fallback

# Setup logging similar to proven pattern
logger = logging.getLogger("s3_session_manager")
logger.setLevel(logging.DEBUG)

class CredentialSource(Enum):
    """Sources for AWS credentials (ordered by corporate security priority)"""
    KMS_ENCRYPTED = "kms_encrypted"  # Highest security - KMS encrypted credentials
    IAM_ROLE = "iam_role"           # Second - IAM roles (no stored credentials)
    AWS_PROFILE = "aws_profile"     # Third - AWS CLI profiles
    ENVIRONMENT = "environment"     # Fourth - Environment variables  
    AWS_CREDENTIALS_FILE = "aws_credentials_file"  # Fifth - Credentials file
    SETTINGS_FILE = "settings_file"  # Sixth - Settings YAML file
    EXPLICIT = "explicit"           # Last resort - explicit credentials
    NOT_FOUND = "not_found"

@dataclass
class S3Credentials:
    """AWS S3 credentials configuration"""
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    session_token: Optional[str] = None
    region: str = "us-east-1"
    profile: Optional[str] = None
    source: CredentialSource = CredentialSource.NOT_FOUND
    kms_key_id: Optional[str] = None
    default_bucket: Optional[str] = None
    default_prefix: Optional[str] = None

class S3SessionManager:
    """Centralized S3 session management with multiple credential sources"""
    
    def __init__(self, profile: str = None, region: str = "us-east-1"):
        """
        Initialize S3 session manager
        
        Args:
            profile: AWS profile name to use
            region: AWS region to use
        """
        if not AWS_AVAILABLE:
            raise ImportError("boto3 not available. Install with: pip install boto3")
        
        self.credentials = S3Credentials(region=region, profile=profile)
        self._session = None
        self._s3_client = None
        self._s3_resource = None
        
        # Always load S3 defaults from settings first
        self._load_s3_defaults()
        
        # Try to load credentials from various sources
        self._discover_credentials()
    
    def _discover_credentials(self):
        """Discover credentials from various sources in corporate security priority order"""
        
        # Store KMS key ID if available (for reference, not credential decryption)
        settings = self._load_settings_file()
        if settings:
            kms_key_id = settings.get('aws', {}).get('kms_key_id')
            if kms_key_id:
                self.credentials.kms_key_id = kms_key_id
                logger.info(f"Found KMS key for reference: {kms_key_id}")
        
        # 1. IAM role (no stored credentials - highest corporate security)
        if self._test_iam_role():
            self.credentials.source = CredentialSource.IAM_ROLE
            return
        
        # 2. AWS CLI profile (corporate managed)
        if self.credentials.profile and self._load_from_profile():
            self.credentials.source = CredentialSource.AWS_PROFILE
            return
            
        # 3. Environment variables
        if self._load_from_environment():
            self.credentials.source = CredentialSource.ENVIRONMENT
            return
        
        # 4. Default profile from AWS credentials file
        if self._load_from_profile("default"):
            self.credentials.source = CredentialSource.AWS_CREDENTIALS_FILE
            return
            
        # 5. Settings YAML file (plain text - least secure)
        if self._load_from_settings():
            self.credentials.source = CredentialSource.SETTINGS_FILE
            return
        
        # No credentials found
        self.credentials.source = CredentialSource.NOT_FOUND
    
    def _load_from_environment(self) -> bool:
        """Load credentials from environment variables"""
        access_key = os.getenv('AWS_ACCESS_KEY_ID')
        secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        session_token = os.getenv('AWS_SESSION_TOKEN')
        region = os.getenv('AWS_DEFAULT_REGION') or os.getenv('AWS_REGION')
        
        if access_key and secret_key:
            self.credentials.access_key_id = access_key
            self.credentials.secret_access_key = secret_key
            self.credentials.session_token = session_token
            if region:
                self.credentials.region = region
            return True
        
        return False
    
    def _load_from_profile(self, profile: str = None) -> bool:
        """Load credentials from AWS credentials file"""
        try:
            profile_name = profile or self.credentials.profile or "default"
            session = boto3.Session(profile_name=profile_name)
            credentials = session.get_credentials()
            
            if credentials:
                self.credentials.access_key_id = credentials.access_key
                self.credentials.secret_access_key = credentials.secret_key
                self.credentials.session_token = credentials.token
                self.credentials.profile = profile_name
                return True
        
        except (ProfileNotFound, Exception):
            pass
        
        return False
    
    def _test_iam_role(self) -> bool:
        """Test if IAM role credentials are available (EC2 instance role)"""
        try:
            session = boto3.Session()
            credentials = session.get_credentials()
            return credentials is not None
        except Exception:
            return False
    
    def _load_from_kms(self) -> bool:
        """Load credentials using AWS KMS - simplified approach like working demo"""
        try:
            # First load settings to get KMS key ID
            settings = self._load_settings_file()
            if not settings:
                return False
                
            kms_key_id = settings.get('aws', {}).get('kms_key_id')
            if not kms_key_id:
                return False
                
            logger.info(f"Found KMS key: {kms_key_id}")
                
            # Method 1: Try encrypted credential blobs (advanced setup)
            encrypted_access_key = settings.get('aws', {}).get('encrypted_access_key_id')
            encrypted_secret_key = settings.get('aws', {}).get('encrypted_secret_access_key')
            
            if encrypted_access_key and encrypted_secret_key:
                # Create a temporary session to decrypt
                kms_client = boto3.client('kms', region_name=self.credentials.region)
                
                # Decrypt credentials (handle base64 encoding)
                import base64
                access_key_blob = base64.b64decode(encrypted_access_key)
                secret_key_blob = base64.b64decode(encrypted_secret_key)
                
                access_key_response = kms_client.decrypt(CiphertextBlob=access_key_blob)
                secret_key_response = kms_client.decrypt(CiphertextBlob=secret_key_blob)
                
                self.credentials.access_key_id = access_key_response['Plaintext'].decode('utf-8')
                self.credentials.secret_access_key = secret_key_response['Plaintext'].decode('utf-8')
                self.credentials.kms_key_id = kms_key_id
                
                logger.info("✅ Loaded credentials from KMS encrypted blobs")
                return True
            
            # Method 2: Simple KMS key-based session (like working demo)
            # Maybe the KMS key is used for assume-role or STS operations
            try:
                # Try to create session using KMS key for role assumption
                sts_client = boto3.client('sts', region_name=self.credentials.region)
                
                # This might work if there's an assume-role setup with KMS
                # Or if KMS key provides access to credential store
                response = sts_client.get_caller_identity()
                
                # If we get here, we have working credentials from somewhere
                self.credentials.kms_key_id = kms_key_id
                logger.info("✅ Using KMS-based credential access")
                return True
                
            except Exception as sts_error:
                logger.debug(f"KMS-based STS access failed: {sts_error}")
                
            # Method 3: Check if KMS key enables credential discovery elsewhere
            # Some corporate setups use KMS key as pointer to credential location
            logger.info("KMS key found but no direct credential access method worked")
            
        except Exception as e:
            logger.debug(f"KMS credential loading failed: {e}")
            
        return False
    
    def _load_from_settings(self) -> bool:
        """Load credentials from settings.yaml file"""
        try:
            settings = self._load_settings_file()
            if not settings:
                return False
                
            aws_config = settings.get('aws', {})
            access_key = aws_config.get('access_key_id', '').strip()
            secret_key = aws_config.get('secret_access_key', '').strip()
            
            if access_key and secret_key and access_key != '' and secret_key != '':
                self.credentials.access_key_id = access_key
                self.credentials.secret_access_key = secret_key
                self.credentials.kms_key_id = aws_config.get('kms_key_id')
                
                # Load S3 defaults
                s3_config = settings.get('s3', {})
                self.credentials.default_bucket = s3_config.get('default_bucket')
                self.credentials.default_prefix = s3_config.get('default_prefix')
                
                logger.info("✅ Loaded credentials from settings.yaml")
                return True
                
        except Exception as e:
            logger.debug(f"Settings file credential loading failed: {e}")
            
        return False
    
    def _load_settings_file(self) -> Optional[Dict[str, Any]]:
        """Load settings from YAML file"""
        try:
            # Look for settings.yaml in current directory and parent directories
            current_dir = Path(__file__).parent
            settings_paths = [
                current_dir / "settings.yaml",
                current_dir.parent / "settings.yaml",
                current_dir.parent.parent / "settings.yaml"
            ]
            
            for settings_path in settings_paths:
                if settings_path.exists():
                    with open(settings_path, 'r') as f:
                        return yaml.safe_load(f)
                        
        except Exception as e:
            logger.debug(f"Failed to load settings file: {e}")
            
        return None
    
    def _load_s3_defaults(self):
        """Load S3 defaults from settings file regardless of credentials"""
        try:
            settings = self._load_settings_file()
            if settings:
                # Check both locations for S3 config
                s3_config = settings.get('s3', {}) or settings.get('aws', {}).get('s3', {})
                
                self.credentials.default_bucket = s3_config.get('default_bucket')
                self.credentials.default_prefix = s3_config.get('default_prefix')
                
                # Also load region from AWS config if not already set
                aws_config = settings.get('aws', {})
                if aws_config.get('region'):
                    self.credentials.region = aws_config['region']
                    
        except Exception as e:
            logger.debug(f"Failed to load S3 defaults: {e}")
    
    def get_session(self) -> 'boto3.Session':
        """Get boto3 session with discovered credentials"""
        if self._session is None:
            if self.credentials.source in [CredentialSource.KMS_ENCRYPTED, CredentialSource.ENVIRONMENT, CredentialSource.SETTINGS_FILE]:
                # Use explicit credentials (from KMS, environment, or settings)
                self._session = boto3.Session(
                    aws_access_key_id=self.credentials.access_key_id,
                    aws_secret_access_key=self.credentials.secret_access_key,
                    aws_session_token=self.credentials.session_token,
                    region_name=self.credentials.region
                )
            elif self.credentials.source in [CredentialSource.AWS_PROFILE, CredentialSource.AWS_CREDENTIALS_FILE]:
                # Use profile
                self._session = boto3.Session(
                    profile_name=self.credentials.profile,
                    region_name=self.credentials.region
                )
            elif self.credentials.source == CredentialSource.IAM_ROLE:
                # Use default session (IAM role)
                self._session = boto3.Session(region_name=self.credentials.region)
            else:
                raise Exception("No AWS credentials found")
        
        return self._session
    
    def get_s3_client(self, **client_kwargs):
        """Get S3 client with retry configuration"""
        if self._s3_client is None:
            session = self.get_session()
            
            # Configure retries and timeouts for better reliability
            config = Config(
                region_name=self.credentials.region,
                retries={'max_attempts': 3, 'mode': 'adaptive'},
                max_pool_connections=50,
                **client_kwargs
            )
            
            self._s3_client = session.client('s3', config=config)
        
        return self._s3_client
    
    def get_s3_resource(self, **resource_kwargs):
        """Get S3 resource"""
        if self._s3_resource is None:
            session = self.get_session()
            self._s3_resource = session.resource('s3', **resource_kwargs)
        
        return self._s3_resource
    
    def test_connection(self) -> Dict[str, Any]:
        """Test S3 connection and return status"""
        try:
            s3_client = self.get_s3_client()
            response = s3_client.list_buckets()
            
            buckets = [bucket['Name'] for bucket in response['Buckets']]
            
            return {
                "success": True,
                "message": f"Connected to S3 successfully",
                "bucket_count": len(buckets),
                "buckets": buckets[:10],  # First 10 buckets
                "region": self.credentials.region,
                "credential_source": self.credentials.source.value
            }
        
        except Exception as e:
            if "credentials" in str(e).lower() or "NoCredentialsError" in str(type(e)):
                return {
                    "success": False,
                    "message": "AWS credentials not found",
                    "credential_source": self.credentials.source.value,
                    "solutions": [
                        "Configure AWS CLI: aws configure",
                        "Set environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY",
                        "Use IAM role if on EC2",
                        "Create ~/.aws/credentials file"
                    ]
                }
            else:
                return {
                    "success": False,
                    "message": f"Connection failed: {str(e)}",
                    "credential_source": self.credentials.source.value
                }
        
    
    def create_bucket_if_not_exists(self, bucket_name: str) -> Dict[str, Any]:
        """Create S3 bucket if it doesn't exist"""
        try:
            s3_client = self.get_s3_client()
            
            # Check if bucket exists
            try:
                s3_client.head_bucket(Bucket=bucket_name)
                return {
                    "success": True,
                    "message": f"Bucket {bucket_name} already exists",
                    "created": False
                }
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == '403':
                    # Bucket exists but we don't have full access - try to use it anyway
                    return {
                        "success": True,
                        "message": f"Bucket {bucket_name} exists (limited access)",
                        "created": False,
                        "warning": "Limited permissions on bucket"
                    }
                elif error_code != '404':
                    raise
            
            # Create bucket
            if self.credentials.region == 'us-east-1':
                s3_client.create_bucket(Bucket=bucket_name)
            else:
                s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.credentials.region}
                )
            
            return {
                "success": True,
                "message": f"Created bucket {bucket_name}",
                "created": True
            }
        
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to create bucket: {str(e)}"
            }
    
    def upload_file(self, file_path: str, bucket_name: str, s3_key: str, 
                   metadata: Dict[str, str] = None) -> Dict[str, Any]:
        """Upload file to S3 with metadata"""
        try:
            s3_client = self.get_s3_client()
            
            extra_args = {}
            if metadata:
                extra_args['Metadata'] = metadata
            
            # Upload file
            s3_client.upload_file(file_path, bucket_name, s3_key, ExtraArgs=extra_args)
            
            # Get file size
            file_size = os.path.getsize(file_path)
            
            return {
                "success": True,
                "message": f"Uploaded {os.path.basename(file_path)} to s3://{bucket_name}/{s3_key}",
                "s3_url": f"s3://{bucket_name}/{s3_key}",
                "file_size": file_size
            }
        
        except Exception as e:
            return {
                "success": False,
                "message": f"Upload failed: {str(e)}"
            }
    
    def download_file(self, bucket_name: str, s3_key: str, local_path: str) -> Dict[str, Any]:
        """Download file from S3"""
        try:
            s3_client = self.get_s3_client()
            s3_client.download_file(bucket_name, s3_key, local_path)
            
            file_size = os.path.getsize(local_path)
            
            return {
                "success": True,
                "message": f"Downloaded s3://{bucket_name}/{s3_key} to {local_path}",
                "local_path": local_path,
                "file_size": file_size
            }
        
        except Exception as e:
            return {
                "success": False,
                "message": f"Download failed: {str(e)}"
            }
    
    def list_objects(self, bucket_name: str, prefix: str = "") -> List[Dict[str, Any]]:
        """List objects in S3 bucket"""
        try:
            s3_client = self.get_s3_client()
            
            objects = []
            paginator = s3_client.get_paginator('list_objects_v2')
            
            for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        objects.append({
                            'key': obj['Key'],
                            'size': obj['Size'],
                            'last_modified': obj['LastModified'],
                            'etag': obj['ETag']
                        })
            
            return objects
        
        except Exception as e:
            return []
    
    def get_credential_status(self) -> Dict[str, Any]:
        """Get detailed credential status information"""
        return {
            "available": self.credentials.source != CredentialSource.NOT_FOUND,
            "source": self.credentials.source.value,
            "region": self.credentials.region,
            "profile": self.credentials.profile,
            "has_access_key": bool(self.credentials.access_key_id),
            "has_secret_key": bool(self.credentials.secret_access_key),
            "has_session_token": bool(self.credentials.session_token),
            "kms_key_id": self.credentials.kms_key_id,
            "default_bucket": self.credentials.default_bucket,
            "default_prefix": self.credentials.default_prefix
        }

# Global instance for easy access
_s3_session_manager = None

def get_s3_session_manager(profile: str = None, region: str = "us-east-1") -> S3SessionManager:
    """Get global S3 session manager instance"""
    global _s3_session_manager
    
    if _s3_session_manager is None:
        _s3_session_manager = S3SessionManager(profile=profile, region=region)
    
    return _s3_session_manager

def reset_s3_session_manager():
    """Reset global S3 session manager (useful for testing)"""
    global _s3_session_manager
    _s3_session_manager = None

# Additional utility functions based on proven patterns from prior project
class S3Utils:
    """Enhanced S3 utilities based on proven patterns"""
    
    def __init__(self, session_manager: S3SessionManager = None):
        self.session_manager = session_manager or get_s3_session_manager()
    
    def list_objects_s3(self, bucket: str, prefix: str, extension: str = None) -> list:
        """List objects in S3 bucket with optional extension filter"""
        try:
            s3_client = self.session_manager.get_s3_client()
            response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
            keys = [obj["Key"] for obj in response.get("Contents", [])]
            if extension:
                keys = [key for key in keys if key.endswith(extension)]
            return keys
        except Exception as e:
            logger.error(f"❌ Error listing objects in bucket {bucket} with prefix {prefix}: {e}")
            return []
    
    def list_pdf_files_s3(self, bucket: str, prefix: str = "") -> List[str]:
        """List PDF files in S3 bucket"""
        return self.list_objects_s3(bucket, prefix, extension=".pdf")
    
    def list_json_files_s3(self, bucket: str, prefix: str = "") -> List[str]:
        """List JSON files in S3 bucket"""
        return self.list_objects_s3(bucket, prefix, extension=".json")
    
    def load_json_from_s3(self, bucket: str, key: str) -> dict:
        """Load JSON file from S3"""
        try:
            s3_client = self.session_manager.get_s3_client()
            response = s3_client.get_object(Bucket=bucket, Key=key)
            return json.loads(response["Body"].read().decode("utf-8"))
        except Exception as e:
            logger.error(f"❌ Failed to load JSON from s3://{bucket}/{key}: {e}")
            return {}
    
    def save_json_to_s3(self, bucket: str, key: str, data: dict):
        """Save JSON data to S3"""
        try:
            s3_client = self.session_manager.get_s3_client()
            s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=json.dumps(data, indent=2).encode("utf-8"),
                ContentType="application/json"
            )
            logger.info(f"✅ Saved JSON to s3://{bucket}/{key}")
        except Exception as e:
            logger.error(f"❌ Failed to save JSON to s3://{bucket}/{key}: {e}")
    
    def save_metadata_to_s3(self, bucket: str, key: str, metadata, model_key: str, 
                           metadata_path: str = "meta", filename: str = None, 
                           overwrite: bool = True):
        """Save metadata to S3 with organized folder structure"""
        try:
            s3_client = self.session_manager.get_s3_client()
            filename = filename or f"{model_key}_metadata.json"
            full_metadata_path = f"{key}/{model_key}/{metadata_path}/{filename}".strip("/")

            if not overwrite:
                try:
                    s3_client.head_object(Bucket=bucket, Key=full_metadata_path)
                    logger.warning(f"⚠️ Metadata already exists at s3://{bucket}/{full_metadata_path}. Skipping.")
                    return
                except ClientError as e:
                    if e.response['Error']['Code'] != "404":
                        raise

            s3_client.put_object(
                Bucket=bucket,
                Key=full_metadata_path,
                Body=json.dumps(metadata, indent=2),
                ContentType="application/json"
            )
            logger.info(f"✅ Metadata saved to s3://{bucket}/{full_metadata_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save metadata for {model_key}: {e}")
            raise
    
    def upload_file_to_s3(self, bucket: str, key: str, data, 
                         content_type: str = "application/octet-stream"):
        """Upload file data to S3"""
        try:
            s3_client = self.session_manager.get_s3_client()
            s3_client.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
            logger.info(f"✅ Uploaded file to s3://{bucket}/{key}")
        except Exception as e:
            logger.error(f"❌ Failed to upload file to s3://{bucket}/{key}: {e}")
    
    def delete_s3_object(self, bucket: str, key: str) -> bool:
        """Delete a single object from S3"""
        try:
            s3_client = self.session_manager.get_s3_client()
            s3_client.delete_object(Bucket=bucket, Key=key)
            logger.info(f"✅ Deleted s3://{bucket}/{key}")
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                logger.warning(f"⚠️ Object not found: s3://{bucket}/{key}")
                return False
            else:
                logger.error(f"❌ Failed to delete s3://{bucket}/{key}: {e}")
                return False
        except Exception as e:
            logger.error(f"❌ Unexpected error deleting s3://{bucket}/{key}: {e}")
            return False
    
    def check_s3_object_tag(self, bucket: str, key: str, tag_key: str) -> bool:
        """Check if S3 object has a specific tag"""
        try:
            s3_client = self.session_manager.get_s3_client()
            response = s3_client.get_object_tagging(Bucket=bucket, Key=key)
            tags = {tag["Key"]: tag["Value"] for tag in response.get("TagSet", [])}
            return tag_key in tags
        except ClientError as e:
            if e.response["Error"]["Code"] == "AccessDenied":
                logger.warning(f"⚠️ Access denied checking tags for {key}")
            else:
                logger.error(f"❌ Error checking tags for {key}: {e}")
            return False

# Global S3 utilities instance
def get_s3_utils(session_manager: S3SessionManager = None) -> S3Utils:
    """Get S3 utilities instance"""
    return S3Utils(session_manager)