"""
AWS Delegate - Borrows ALL AWS Services from Parent Infrastructure
==================================================================

Single delegate for ALL AWS operations (S3, Bedrock, STS, etc.)
TidyLLM doesn't import boto3 - it uses the parent's unified AWS service.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Add parent infrastructure to path
qa_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(qa_root))

# Try to import parent AWS service
try:
    from infrastructure.services.aws_service import get_aws_service, AWSService, BedrockModel
    PARENT_AWS_AVAILABLE = True
except ImportError:
    PARENT_AWS_AVAILABLE = False
    logger.warning("Parent AWS service not available - AWS operations will be disabled")

    # Mock AWSService for when parent isn't available
    class AWSService:
        """Mock AWS service when parent infrastructure isn't available."""

        def __init__(self, config=None):
            self.config = config or {}

        def is_available(self):
            return False

        # S3 operations
        def upload_file(self, *args, **kwargs):
            logger.warning("AWS not available - upload skipped")
            return False

        def download_file(self, *args, **kwargs):
            logger.warning("AWS not available - download skipped")
            return False

        def list_s3_objects(self, *args, **kwargs):
            return []

        def read_json_from_s3(self, *args, **kwargs):
            return None

        def write_json_to_s3(self, *args, **kwargs):
            return False

        def s3_object_exists(self, *args, **kwargs):
            return False

        def delete_s3_object(self, *args, **kwargs):
            return False

        def get_s3_presigned_url(self, *args, **kwargs):
            return None

        # Bedrock operations
        def list_foundation_models(self):
            return []

        def invoke_model(self, *args, **kwargs):
            return None

        def create_embedding(self, *args, **kwargs):
            return None

        # STS operations
        def get_caller_identity(self):
            return None

        def assume_role(self, *args, **kwargs):
            return None

        # Client getters
        def get_s3_client(self):
            return None

        def get_bedrock_client(self):
            return None

        def get_bedrock_runtime_client(self):
            return None

        def get_sts_client(self):
            return None

        def health_check(self):
            return {'available': False, 'services': {}}

    # Mock BedrockModel enum
    class BedrockModel:
        CLAUDE_3_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"
        CLAUDE_3_SONNET = "anthropic.claude-3-sonnet-20240229-v1:0"
        CLAUDE_3_OPUS = "anthropic.claude-3-opus-20240229-v1:0"
        TITAN_TEXT_EXPRESS = "amazon.titan-text-express-v1"
        TITAN_EMBED_TEXT = "amazon.titan-embed-text-v1"

    def get_aws_service(config=None):
        return AWSService(config)


class AWSDelegate:
    """
    Unified delegate for ALL AWS operations.

    This class provides a clean interface for TidyLLM components
    to use ANY AWS service without directly importing boto3.
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize AWS delegate with optional configuration."""
        self.config = config or {}
        self._service = None

        if PARENT_AWS_AVAILABLE:
            self._service = get_aws_service(config)
            logger.info("AWS delegate connected to parent infrastructure")
        else:
            self._service = AWSService(config)
            logger.warning("AWS delegate using mock service (parent not available)")

    @property
    def service(self) -> AWSService:
        """Get the underlying AWS service."""
        return self._service

    def is_available(self) -> bool:
        """Check if AWS operations are available."""
        return self._service.is_available()

    # ==================== S3 Operations ====================

    def upload_file(self, file_path: str, s3_key: str, bucket: Optional[str] = None) -> bool:
        """Upload file to S3."""
        return self._service.upload_file(file_path, s3_key, bucket)

    def download_file(self, s3_key: str, local_path: str, bucket: Optional[str] = None) -> bool:
        """Download file from S3."""
        return self._service.download_file(s3_key, local_path, bucket)

    def list_objects(self, prefix: str = "", bucket: Optional[str] = None) -> List[str]:
        """List S3 objects."""
        return self._service.list_s3_objects(prefix, bucket)

    def read_json(self, s3_key: str, bucket: Optional[str] = None) -> Optional[Dict]:
        """Read JSON from S3."""
        return self._service.read_json_from_s3(s3_key, bucket)

    def write_json(self, data: Dict, s3_key: str, bucket: Optional[str] = None) -> bool:
        """Write JSON to S3."""
        return self._service.write_json_to_s3(data, s3_key, bucket)

    def exists(self, s3_key: str, bucket: Optional[str] = None) -> bool:
        """Check if S3 object exists."""
        return self._service.s3_object_exists(s3_key, bucket)

    def delete(self, s3_key: str, bucket: Optional[str] = None) -> bool:
        """Delete S3 object."""
        return self._service.delete_s3_object(s3_key, bucket)

    def get_presigned_url(self, s3_key: str, expires_in: int = 3600, bucket: Optional[str] = None) -> Optional[str]:
        """Get presigned URL for S3 object."""
        return self._service.get_s3_presigned_url(s3_key, expires_in, bucket)

    # ==================== Bedrock Operations ====================

    def list_foundation_models(self) -> List[Dict]:
        """List available foundation models."""
        return self._service.list_foundation_models()

    def invoke_model(self, prompt: str, model_id: Optional[str] = None, **kwargs) -> Optional[str]:
        """Invoke a Bedrock model."""
        return self._service.invoke_model(prompt, model_id, **kwargs)

    def invoke_claude(self, prompt: str, model: str = None, **kwargs) -> Optional[str]:
        """Invoke Claude model."""
        model = model or BedrockModel.CLAUDE_3_HAIKU.value if PARENT_AWS_AVAILABLE else "anthropic.claude-3-haiku"
        return self._service.invoke_model(prompt, model, **kwargs)

    def invoke_titan(self, prompt: str, model: str = None, **kwargs) -> Optional[str]:
        """Invoke Titan model."""
        model = model or BedrockModel.TITAN_TEXT_EXPRESS.value if PARENT_AWS_AVAILABLE else "amazon.titan-text-express-v1"
        return self._service.invoke_model(prompt, model, **kwargs)

    def create_embedding(self, text: str, model_id: Optional[str] = None) -> Optional[List[float]]:
        """Create text embedding."""
        return self._service.create_embedding(text, model_id)

    # ==================== STS Operations ====================

    def get_caller_identity(self) -> Optional[Dict]:
        """Get AWS caller identity."""
        return self._service.get_caller_identity()

    def assume_role(self, role_arn: str, session_name: str) -> Optional[Dict]:
        """Assume an IAM role."""
        return self._service.assume_role(role_arn, session_name)

    # ==================== Client Access (Compatibility) ====================

    def get_s3_client(self):
        """Get S3 client interface for compatibility."""
        return self._service.get_s3_client() if self.is_available() else self

    def get_bedrock_client(self):
        """Get Bedrock client interface for compatibility."""
        return self._service.get_bedrock_client() if self.is_available() else self

    def get_bedrock_runtime_client(self):
        """Get Bedrock Runtime client interface for compatibility."""
        return self._service.get_bedrock_runtime_client() if self.is_available() else self

    def get_sts_client(self):
        """Get STS client interface for compatibility."""
        return self._service.get_sts_client() if self.is_available() else self

    # ==================== Health & Status ====================

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on AWS services."""
        return self._service.health_check()


# Singleton instance
_aws_delegate = None

def get_aws_delegate(config: Optional[Dict] = None) -> AWSDelegate:
    """
    Get singleton AWS delegate instance.

    This is the main entry point for TidyLLM components to access AWS services.
    """
    global _aws_delegate
    if _aws_delegate is None:
        _aws_delegate = AWSDelegate(config)
    return _aws_delegate


# ==================== Compatibility Functions ====================
# These maintain compatibility with existing code

def get_s3_delegate(config: Optional[Dict] = None) -> AWSDelegate:
    """Compatibility: Get AWS delegate for S3 operations."""
    return get_aws_delegate(config)

def get_bedrock_delegate(config: Optional[Dict] = None) -> AWSDelegate:
    """Compatibility: Get AWS delegate for Bedrock operations."""
    return get_aws_delegate(config)

def get_s3_config() -> Dict[str, Any]:
    """Compatibility: Get S3 configuration."""
    delegate = get_aws_delegate()
    if delegate.is_available():
        return {
            "bucket": delegate.config.get('bucket', 'default-bucket'),
            "region": delegate.config.get('region', 'us-east-1'),
            "available": True
        }
    return {
        "bucket": "default-bucket",
        "region": "us-east-1",
        "available": False
    }

def build_s3_path(*args) -> str:
    """Compatibility: Build S3 path from components."""
    return "/".join(str(arg) for arg in args if arg)


# Export commonly used models
if PARENT_AWS_AVAILABLE:
    from infrastructure.services.aws_service import BedrockModel
else:
    # Use mock BedrockModel defined above
    pass