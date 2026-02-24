"""
Bedrock Delegate - Proper delegate pattern for AWS Bedrock access
==================================================================

This delegate provides Bedrock access without importing parent infrastructure.
It loads credentials from settings.yaml and creates its own boto3 client.

This follows the architecture rule: Domain/Application layers use delegates,
never import infrastructure directly.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import boto3
try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logger.warning("boto3 not available - Bedrock delegate will use mock")


class BedrockDelegate:
    """
    Delegate for AWS Bedrock operations.

    Provides a clean interface for domain services to use Bedrock
    without knowing about infrastructure details.
    """

    def __init__(self):
        """Initialize the Bedrock delegate."""
        self._client = None
        self._initialized = False

    def _initialize(self):
        """Lazy initialization of Bedrock client."""
        if self._initialized:
            return

        self._initialized = True

        if not BOTO3_AVAILABLE:
            logger.warning("boto3 not available - using mock")
            return

        try:
            # Try to get credentials from environment first
            if os.getenv('AWS_ACCESS_KEY_ID'):
                self._client = boto3.client(
                    'bedrock-runtime',
                    region_name=os.getenv('AWS_REGION', 'us-east-1'),
                    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
                )
                logger.info("Bedrock delegate initialized with environment credentials")
                return

            # Try to use default credentials (IAM role, etc)
            self._client = boto3.client(
                'bedrock-runtime',
                region_name='us-east-1'
            )
            logger.info("Bedrock delegate initialized with default credentials")

        except Exception as e:
            logger.warning(f"Could not initialize Bedrock client: {e}")
            self._client = None

    def invoke_model(self, prompt: str, model_id: str = None, **kwargs) -> Dict[str, Any]:
        """
        Invoke a Bedrock model.

        Args:
            prompt: The prompt to send to the model
            model_id: The model ID (defaults to Claude 3 Haiku)
            **kwargs: Additional parameters

        Returns:
            Dict with success status and response text
        """
        # Initialize if needed
        if not self._initialized:
            self._initialize()

        # Default model
        if not model_id:
            model_id = 'anthropic.claude-3-haiku-20240307-v1:0'

        # If no client available, return mock
        if not self._client:
            return {
                'success': False,
                'error': 'Bedrock not available (no credentials or boto3)',
                'text': f'[MOCK] Would process: {prompt[:50]}...'
            }

        try:
            # Format request for Claude models
            if 'claude' in model_id.lower():
                request_body = {
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": kwargs.get('max_tokens', 1000),
                    "temperature": kwargs.get('temperature', 0.7),
                    "anthropic_version": "bedrock-2023-05-31"
                }
            else:
                # Generic format for other models
                request_body = {
                    "prompt": prompt,
                    "max_tokens": kwargs.get('max_tokens', 1000),
                    "temperature": kwargs.get('temperature', 0.7)
                }

            # Invoke the model
            response = self._client.invoke_model(
                modelId=model_id,
                body=json.dumps(request_body),
                contentType='application/json'
            )

            # Parse response
            response_body = json.loads(response['body'].read())

            # Extract text based on model type
            if 'content' in response_body and isinstance(response_body['content'], list):
                # Claude 3 response format
                text = response_body['content'][0].get('text', '')
            elif 'completion' in response_body:
                # Claude 2 format
                text = response_body['completion']
            else:
                # Fallback - just stringify
                text = str(response_body)

            return {
                'success': True,
                'text': text,
                'model': model_id
            }

        except Exception as e:
            logger.error(f"Bedrock invocation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'text': ''
            }

    def list_models(self) -> Dict[str, Any]:
        """List available Bedrock models."""
        if not self._initialized:
            self._initialize()

        if not self._client:
            return {
                'success': False,
                'error': 'Bedrock not available',
                'models': []
            }

        try:
            # Create regular bedrock client for listing
            bedrock = boto3.client('bedrock', region_name='us-east-1')
            response = bedrock.list_foundation_models()

            models = [m['modelId'] for m in response.get('modelSummaries', [])]

            return {
                'success': True,
                'models': models
            }
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return {
                'success': False,
                'error': str(e),
                'models': []
            }


# Singleton instance
_delegate = None

def get_bedrock_delegate() -> BedrockDelegate:
    """
    Get the Bedrock delegate singleton.

    This is the function that domain services should use:

    from tidyllm.infrastructure.bedrock_delegate import get_bedrock_delegate
    delegate = get_bedrock_delegate()
    response = delegate.invoke_model(prompt)
    """
    global _delegate
    if _delegate is None:
        _delegate = BedrockDelegate()
    return _delegate