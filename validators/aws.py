"""
AWS Validator for TidyLLM
========================

Corporate-safe AWS service validation (S3, Bedrock, STS) with timeout protection.
"""

from typing import Dict, Any
from .base import BaseValidator

# Use infrastructure delegate for ALL AWS operations
try:
    from packages.tidyllm.infrastructure.infra_delegate import get_infra_delegate
    INFRA_DELEGATE_AVAILABLE = True
except ImportError:
    INFRA_DELEGATE_AVAILABLE = False

# No longer import boto3 directly - all through infra_delegate
AWS_AVAILABLE = INFRA_DELEGATE_AVAILABLE


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

        # Test infra_delegate availability
        if INFRA_DELEGATE_AVAILABLE:
            try:
                infra = get_infra_delegate()
                result['infra_delegate'] = 'available'

                # Check if we can get Bedrock config
                bedrock_config = infra.get_bedrock_config()
                if bedrock_config:
                    result['bedrock']['config'] = 'available'

            except Exception as e:
                result['infra_delegate_error'] = str(e)

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
        
        if not INFRA_DELEGATE_AVAILABLE:
            for service in ['s3', 'bedrock', 'sts']:
                result[service] = {
                    'status': 'error',
                    'message': 'Infrastructure delegate not available',
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
        """Test S3 service through infra_delegate."""
        try:
            infra = get_infra_delegate()

            # S3 operations through delegate (if supported)
            # For now, just check availability
            return {
                'status': 'partial',
                'message': 'S3 operations pending infra_delegate implementation'
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'S3 delegate not available: {e}'
            }
    
    def _test_bedrock_service(self) -> Dict[str, Any]:
        """Test Bedrock service through infra_delegate."""
        try:
            infra = get_infra_delegate()

            # Test Bedrock with a simple prompt
            result = infra.invoke_bedrock(
                prompt="Test connection",
                model_id='anthropic.claude-3-haiku-20240307-v1:0'
            )

            if result.get('success'):
                return {
                    'status': 'success',
                    'message': 'Bedrock connection successful via infra_delegate'
                }
            else:
                return {
                    'status': 'error',
                    'message': f'Bedrock test failed: {result.get("error", "Unknown error")}'
                }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Bedrock connection failed: {e}'
            }
    
    def _test_sts_service(self) -> Dict[str, Any]:
        """Test STS service through infra_delegate."""
        try:
            infra = get_infra_delegate()

            # STS operations through delegate (if supported)
            # For now, just check availability
            return {
                'status': 'partial',
                'message': 'STS operations pending infra_delegate implementation'
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'STS delegate not available: {e}'
            }