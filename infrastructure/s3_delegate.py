"""
S3 Delegate - Borrows from Parent Infrastructure
=================================================

This module delegates all S3 operations to the parent infrastructure.
TidyLLM doesn't import boto3 directly - it uses the parent's S3 service.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Add parent infrastructure to path
qa_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(qa_root))

# Try to import parent S3 service
try:
    from infrastructure.services.s3_service import get_s3_service, S3Service
    PARENT_S3_AVAILABLE = True
except ImportError:
    PARENT_S3_AVAILABLE = False
    logger.warning("Parent S3 service not available - S3 operations will be disabled")

    # Real S3Service implementation for when parent isn't available
    import boto3
    import json
    import os
    from botocore.exceptions import ClientError, NoCredentialsError

    class S3Service:
        """Real S3 service implementation using boto3 directly."""

        def __init__(self, config=None):
            self.config = config or {}
            self.region = self.config.get('region', 'us-east-1')
            self.bucket = self.config.get('bucket', os.environ.get('S3_BUCKET', 'nsc-mvp1'))
            self._client = None
            self._resource = None

        def _get_client(self):
            """Get or create S3 client."""
            if not self._client:
                try:
                    self._client = boto3.client('s3', region_name=self.region)
                except Exception as e:
                    logger.error(f"Failed to create S3 client: {e}")
            return self._client

        def _get_resource(self):
            """Get or create S3 resource."""
            if not self._resource:
                try:
                    self._resource = boto3.resource('s3', region_name=self.region)
                except Exception as e:
                    logger.error(f"Failed to create S3 resource: {e}")
            return self._resource

        def is_available(self):
            """Check if S3 service is available."""
            try:
                client = self._get_client()
                if client:
                    # Try to list buckets to verify connectivity
                    client.head_bucket(Bucket=self.bucket)
                    return True
            except (ClientError, NoCredentialsError):
                pass
            return False

        def upload_file(self, file_path, s3_key, bucket=None):
            """Upload file to S3."""
            bucket = bucket or self.bucket
            try:
                client = self._get_client()
                if client:
                    with open(file_path, 'rb') as f:
                        client.put_object(
                            Bucket=bucket,
                            Key=s3_key,
                            Body=f
                        )
                    logger.info(f"Uploaded {file_path} to s3://{bucket}/{s3_key}")
                    return True
            except (ClientError, FileNotFoundError) as e:
                logger.error(f"Error uploading file: {e}")
            return False

        def download_file(self, s3_key, local_path, bucket=None):
            """Download file from S3."""
            bucket = bucket or self.bucket
            try:
                client = self._get_client()
                if client:
                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)

                    response = client.get_object(Bucket=bucket, Key=s3_key)
                    with open(local_path, 'wb') as f:
                        f.write(response['Body'].read())
                    logger.info(f"Downloaded s3://{bucket}/{s3_key} to {local_path}")
                    return True
            except ClientError as e:
                logger.error(f"Error downloading file: {e}")
            return False

        def list_objects(self, prefix="", bucket=None):
            """List objects in S3 bucket."""
            bucket = bucket or self.bucket
            objects = []
            try:
                client = self._get_client()
                if client:
                    paginator = client.get_paginator('list_objects_v2')
                    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

                    for page in pages:
                        if 'Contents' in page:
                            for obj in page['Contents']:
                                objects.append(obj['Key'])
            except ClientError as e:
                logger.error(f"Error listing objects: {e}")
            return objects

        def read_json(self, s3_key, bucket=None):
            """Read JSON file from S3."""
            bucket = bucket or self.bucket
            try:
                client = self._get_client()
                if client:
                    response = client.get_object(Bucket=bucket, Key=s3_key)
                    content = response['Body'].read().decode('utf-8')
                    return json.loads(content)
            except (ClientError, json.JSONDecodeError) as e:
                logger.error(f"Error reading JSON: {e}")
            return None

        def write_json(self, data, s3_key, bucket=None):
            """Write JSON data to S3."""
            bucket = bucket or self.bucket
            try:
                client = self._get_client()
                if client:
                    json_data = json.dumps(data, indent=2)
                    client.put_object(
                        Bucket=bucket,
                        Key=s3_key,
                        Body=json_data,
                        ContentType='application/json'
                    )
                    logger.info(f"Wrote JSON to s3://{bucket}/{s3_key}")
                    return True
            except (ClientError, TypeError) as e:
                logger.error(f"Error writing JSON: {e}")
            return False

        def exists(self, s3_key, bucket=None):
            """Check if object exists in S3."""
            bucket = bucket or self.bucket
            try:
                client = self._get_client()
                if client:
                    client.head_object(Bucket=bucket, Key=s3_key)
                    return True
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    return False
                logger.error(f"Error checking existence: {e}")
            return False

        def delete(self, s3_key, bucket=None):
            """Delete object from S3."""
            bucket = bucket or self.bucket
            try:
                client = self._get_client()
                if client:
                    client.delete_object(Bucket=bucket, Key=s3_key)
                    logger.info(f"Deleted s3://{bucket}/{s3_key}")
                    return True
            except ClientError as e:
                logger.error(f"Error deleting object: {e}")
            return False

        def get_presigned_url(self, s3_key, expires_in=3600, bucket=None):
            """Generate presigned URL for S3 object."""
            bucket = bucket or self.bucket
            try:
                client = self._get_client()
                if client:
                    url = client.generate_presigned_url(
                        'get_object',
                        Params={'Bucket': bucket, 'Key': s3_key},
                        ExpiresIn=expires_in
                    )
                    return url
            except ClientError as e:
                logger.error(f"Error generating presigned URL: {e}")
            return None

    def get_s3_service(config=None):
        return S3Service(config)


class S3Delegate:
    """
    Delegates S3 operations to parent infrastructure.

    This class provides a clean interface for TidyLLM components
    to use S3 without directly importing boto3.
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize S3 delegate with optional configuration."""
        self.config = config or {}
        self._service = None

        if PARENT_S3_AVAILABLE:
            self._service = get_s3_service(config)
            logger.info("S3 delegate connected to parent infrastructure")
        else:
            self._service = S3Service(config)
            logger.warning("S3 delegate using mock service (parent not available)")

    @property
    def service(self) -> S3Service:
        """Get the underlying S3 service."""
        return self._service

    def is_available(self) -> bool:
        """Check if S3 operations are available."""
        return self._service.is_available()

    # Delegate all methods to parent service
    def upload_file(self, file_path: str, s3_key: str, bucket: Optional[str] = None) -> bool:
        """Upload file to S3 via parent infrastructure."""
        return self._service.upload_file(file_path, s3_key, bucket)

    def download_file(self, s3_key: str, local_path: str, bucket: Optional[str] = None) -> bool:
        """Download file from S3 via parent infrastructure."""
        return self._service.download_file(s3_key, local_path, bucket)

    def list_objects(self, prefix: str = "", bucket: Optional[str] = None) -> List[str]:
        """List S3 objects via parent infrastructure."""
        return self._service.list_objects(prefix, bucket)

    def read_json(self, s3_key: str, bucket: Optional[str] = None) -> Optional[Dict]:
        """Read JSON from S3 via parent infrastructure."""
        return self._service.read_json(s3_key, bucket)

    def write_json(self, data: Dict, s3_key: str, bucket: Optional[str] = None) -> bool:
        """Write JSON to S3 via parent infrastructure."""
        return self._service.write_json(data, s3_key, bucket)

    def exists(self, s3_key: str, bucket: Optional[str] = None) -> bool:
        """Check if S3 object exists via parent infrastructure."""
        return self._service.exists(s3_key, bucket)

    def delete(self, s3_key: str, bucket: Optional[str] = None) -> bool:
        """Delete S3 object via parent infrastructure."""
        return self._service.delete(s3_key, bucket)

    def get_presigned_url(self, s3_key: str, expires_in: int = 3600, bucket: Optional[str] = None) -> Optional[str]:
        """Get presigned URL via parent infrastructure."""
        return self._service.get_presigned_url(s3_key, expires_in, bucket)


# Singleton instance
_s3_delegate = None

def get_s3_delegate(config: Optional[Dict] = None) -> S3Delegate:
    """
    Get singleton S3 delegate instance.

    This is the main entry point for TidyLLM components to access S3.
    """
    global _s3_delegate
    if _s3_delegate is None:
        _s3_delegate = S3Delegate(config)
    return _s3_delegate


# Compatibility functions for existing code
def get_s3_config() -> Dict[str, Any]:
    """
    Get S3 configuration from parent infrastructure.

    Maintains compatibility with existing code.
    """
    delegate = get_s3_delegate()
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
    """
    Build S3 path from components.

    Maintains compatibility with existing code.
    """
    return "/".join(str(arg) for arg in args if arg)