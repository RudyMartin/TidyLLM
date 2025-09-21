"""
AWS Validator for TidyLLM
========================

Corporate-safe AWS service validation (S3, Bedrock, STS) with timeout protection.
"""

from typing import Dict, Any
from .base import BaseValidator

# For S3 operations, use parent infrastructure
try:
    from tidyllm.infrastructure.s3_delegate import get_s3_delegate
    S3_DELEGATE_AVAILABLE = True
except ImportError:
    S3_DELEGATE_AVAILABLE = False

# Keep boto3 for non-S3 services (Bedrock, STS)
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False


class AWSValidator(BaseValidator):
    """Corporate-safe AWS service validator."""
    
    def validate_aws_connectivity(self) -> Dict[str, Any]:
        """
        Validate AWS connectivity without hanging in corporate environments.
        
        Returns:
            Validation results dict with S3, Bedrock, STS status
        """
        
        # Detect corporate environment first
        env_info = self.detect_corporate_environment()
        
        if self.corporate_mode:
            return self._validate_corporate_mode()
        else:
            return self._validate_standard_mode()
    
    def _validate_corporate_mode(self) -> Dict[str, Any]:
        """Validate in corporate mode - no hanging service calls."""
        
        result = {
            's3': self.corporate_safe_result('S3', 'Corporate environment detected - skipping list_buckets() to prevent hanging'),
            'bedrock': self.corporate_safe_result('Bedrock', 'Corporate environment detected - skipping list_foundation_models() to prevent hanging'),
            'sts': self.corporate_safe_result('STS', 'Corporate environment detected - skipping get_caller_identity() to prevent hanging'),
            'overall_status': 'corporate_safe',
            'corporate_mode': True
        }
        
        # Do basic credential detection without service calls
        if AWS_AVAILABLE:
            try:
                # Test if we can create clients (but don't call services)
                boto3.client('s3')
                result['s3']['client_creation'] = 'success'
                
                boto3.client('bedrock', region_name='us-east-1')
                result['bedrock']['client_creation'] = 'success'
                
                boto3.client('sts')
                result['sts']['client_creation'] = 'success'
                
            except Exception as e:
                result['client_creation_error'] = str(e)
        
        return result
    
    def _validate_standard_mode(self) -> Dict[str, Any]:
        """Validate in standard mode with timeouts."""
        
        result = {
            's3': {'status': 'unknown', 'message': '', 'latency': 0},
            'bedrock': {'status': 'unknown', 'message': '', 'latency': 0}, 
            'sts': {'status': 'unknown', 'message': '', 'latency': 0},
            'overall_status': 'testing',
            'corporate_mode': False
        }
        
        if not AWS_AVAILABLE:
            for service in ['s3', 'bedrock', 'sts']:
                result[service] = {
                    'status': 'error',
                    'message': 'boto3 not available',
                    'latency': 0
                }
            return result
        
        # Test each service with timeout protection
        result['s3'] = self.run_with_timeout(self._test_s3_service, 'S3')
        result['bedrock'] = self.run_with_timeout(self._test_bedrock_service, 'Bedrock')
        result['sts'] = self.run_with_timeout(self._test_sts_service, 'STS')
        
        # Determine overall status
        statuses = [result[service]['status'] for service in ['s3', 'bedrock', 'sts']]
        if all(status == 'success' for status in statuses):
            result['overall_status'] = 'success'
        elif any(status == 'success' for status in statuses):
            result['overall_status'] = 'partial'
        else:
            result['overall_status'] = 'failed'
        
        return result
    
    def _test_s3_service(self) -> Dict[str, Any]:
        """Test S3 service - THIS is where hanging happens."""
        try:
            s3_client = boto3.client('s3')
            
            # THIS is the call that hangs in corporate environments
            buckets = s3_client.list_buckets()
            
            return {
                'status': 'success',
                'message': f'S3 connection successful - {len(buckets["Buckets"])} buckets found',
                'bucket_count': len(buckets["Buckets"])
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'S3 connection failed: {e}'
            }
    
    def _test_bedrock_service(self) -> Dict[str, Any]:
        """Test Bedrock service - THIS is where hanging happens."""
        try:
            bedrock_client = boto3.client('bedrock', region_name='us-east-1')
            
            # THIS is the call that hangs in corporate environments
            models = bedrock_client.list_foundation_models()
            
            return {
                'status': 'success',
                'message': f'Bedrock connection successful - {len(models["modelSummaries"])} models available',
                'model_count': len(models["modelSummaries"])
            }
            
        except Exception as e:
            return {
                'status': 'error', 
                'message': f'Bedrock connection failed: {e}'
            }
    
    def _test_sts_service(self) -> Dict[str, Any]:
        """Test STS service - THIS is where hanging happens."""
        try:
            sts_client = boto3.client('sts')
            
            # THIS is the call that hangs in corporate environments
            identity = sts_client.get_caller_identity()
            
            return {
                'status': 'success',
                'message': f'STS connection successful - Account: {identity["Account"]}',
                'account_id': identity["Account"],
                'user_arn': identity.get("Arn", "unknown")
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'STS connection failed: {e}'
            }