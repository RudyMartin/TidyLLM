"""
Bedrock Delegate - Borrows from Parent Infrastructure
======================================================

This module delegates all Bedrock operations to the parent infrastructure.
TidyLLM doesn't import boto3 directly for Bedrock - it uses the parent's Bedrock service.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Add parent infrastructure to path
qa_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(qa_root))

# Try to import parent Bedrock service
try:
    from infrastructure.services.bedrock_service import get_bedrock_service, BedrockService, BedrockModel
    PARENT_BEDROCK_AVAILABLE = True
except ImportError:
    PARENT_BEDROCK_AVAILABLE = False
    logger.warning("Parent Bedrock service not available - Bedrock operations will be disabled")

    # Real BedrockService implementation for when parent isn't available
    import boto3
    import json
    from botocore.exceptions import ClientError

    class BedrockService:
        """Real Bedrock service implementation using boto3 directly."""

        def __init__(self, config=None):
            self.config = config or {}
            self.region = self.config.get('region', 'us-east-1')
            self._client = None
            self._runtime_client = None

        def _get_client(self):
            """Get or create Bedrock client."""
            if not self._client:
                try:
                    self._client = boto3.client('bedrock', region_name=self.region)
                except Exception as e:
                    logger.error(f"Failed to create Bedrock client: {e}")
            return self._client

        def _get_runtime_client(self):
            """Get or create Bedrock runtime client."""
            if not self._runtime_client:
                try:
                    self._runtime_client = boto3.client('bedrock-runtime', region_name=self.region)
                except Exception as e:
                    logger.error(f"Failed to create Bedrock runtime client: {e}")
            return self._runtime_client

        def is_available(self):
            """Check if Bedrock service is available."""
            try:
                client = self._get_client()
                if client:
                    # Try a simple API call to verify connectivity
                    client.list_foundation_models(maxResults=1)
                    return True
            except:
                pass
            return False

        def list_foundation_models(self):
            """List available foundation models."""
            try:
                client = self._get_client()
                if client:
                    response = client.list_foundation_models()
                    return response.get('modelSummaries', [])
            except ClientError as e:
                logger.error(f"Error listing models: {e}")
            return []

        def invoke_model(self, prompt, model_id=None, **kwargs):
            """Invoke a model with the given prompt."""
            if not model_id:
                model_id = BedrockModel.CLAUDE_3_HAIKU  # Default model

            try:
                runtime = self._get_runtime_client()
                if runtime:
                    # Format request based on model type
                    if 'claude' in model_id:
                        body = json.dumps({
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": kwargs.get('max_tokens', 1000),
                            "anthropic_version": "bedrock-2023-05-31"
                        })
                    elif 'titan' in model_id:
                        body = json.dumps({
                            "inputText": prompt,
                            "textGenerationConfig": {
                                "maxTokenCount": kwargs.get('max_tokens', 1000)
                            }
                        })
                    else:
                        body = json.dumps({"prompt": prompt})

                    response = runtime.invoke_model(
                        modelId=model_id,
                        body=body,
                        contentType='application/json',
                        accept='application/json'
                    )

                    result = json.loads(response['body'].read())

                    # Extract text based on model response format
                    if 'claude' in model_id:
                        return result.get('content', [{}])[0].get('text', '')
                    elif 'titan' in model_id:
                        return result.get('results', [{}])[0].get('outputText', '')
                    else:
                        return str(result)

            except ClientError as e:
                logger.error(f"Error invoking model {model_id}: {e}")
            return None

        def invoke_claude(self, prompt, **kwargs):
            """Invoke Claude model."""
            model_id = kwargs.get('model_id', BedrockModel.CLAUDE_3_HAIKU)
            return self.invoke_model(prompt, model_id, **kwargs)

        def invoke_titan(self, prompt, **kwargs):
            """Invoke Titan model."""
            model_id = kwargs.get('model_id', BedrockModel.TITAN_TEXT_EXPRESS)
            return self.invoke_model(prompt, model_id, **kwargs)

        def create_embedding(self, text, model_id=None):
            """Create text embedding."""
            if not model_id:
                model_id = BedrockModel.TITAN_EMBED_TEXT

            try:
                runtime = self._get_runtime_client()
                if runtime:
                    body = json.dumps({"inputText": text})
                    response = runtime.invoke_model(
                        modelId=model_id,
                        body=body,
                        contentType='application/json',
                        accept='application/json'
                    )

                    result = json.loads(response['body'].read())
                    return result.get('embedding', None)

            except ClientError as e:
                logger.error(f"Error creating embedding: {e}")
            return None

        def get_model_info(self, model_id):
            """Get information about a specific model."""
            try:
                client = self._get_client()
                if client:
                    models = self.list_foundation_models()
                    for model in models:
                        if model.get('modelId') == model_id:
                            return model
            except ClientError as e:
                logger.error(f"Error getting model info: {e}")
            return None

    # Mock BedrockModel enum
    class BedrockModel:
        CLAUDE_3_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"
        CLAUDE_3_SONNET = "anthropic.claude-3-sonnet-20240229-v1:0"
        CLAUDE_3_OPUS = "anthropic.claude-3-opus-20240229-v1:0"
        CLAUDE_2_1 = "anthropic.claude-v2:1"
        TITAN_TEXT_EXPRESS = "amazon.titan-text-express-v1"
        TITAN_EMBED_TEXT = "amazon.titan-embed-text-v1"

    def get_bedrock_service(config=None):
        return BedrockService(config)


class BedrockDelegate:
    """
    Delegates Bedrock operations to parent infrastructure.

    This class provides a clean interface for TidyLLM components
    to use Bedrock without directly importing boto3.
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize Bedrock delegate with optional configuration."""
        self.config = config or {}
        self._service = None

        if PARENT_BEDROCK_AVAILABLE:
            self._service = get_bedrock_service(config)
            logger.info("Bedrock delegate connected to parent infrastructure")
        else:
            self._service = BedrockService(config)
            logger.warning("Bedrock delegate using mock service (parent not available)")

    @property
    def service(self) -> BedrockService:
        """Get the underlying Bedrock service."""
        return self._service

    def is_available(self) -> bool:
        """Check if Bedrock operations are available."""
        return self._service.is_available()

    # Delegate all methods to parent service
    def list_foundation_models(self) -> List[Dict]:
        """List foundation models via parent infrastructure."""
        return self._service.list_foundation_models()

    def invoke_model(self, prompt: str, model_id: Optional[str] = None, **kwargs) -> Optional[str]:
        """Invoke model via parent infrastructure."""
        return self._service.invoke_model(prompt, model_id, **kwargs)

    def invoke_claude(self, prompt: str, **kwargs) -> Optional[str]:
        """Invoke Claude via parent infrastructure."""
        return self._service.invoke_claude(prompt, **kwargs)

    def invoke_titan(self, prompt: str, **kwargs) -> Optional[str]:
        """Invoke Titan via parent infrastructure."""
        return self._service.invoke_titan(prompt, **kwargs)

    def create_embedding(self, text: str, model_id: Optional[str] = None) -> Optional[List[float]]:
        """Create embedding via parent infrastructure."""
        return self._service.create_embedding(text, model_id)

    def get_model_info(self, model_id: str) -> Optional[Dict]:
        """Get model info via parent infrastructure."""
        return self._service.get_model_info(model_id)

    # Compatibility methods for existing code
    def get_bedrock_client(self):
        """
        Get a client-like interface for compatibility.

        Note: This returns the delegate itself which provides
        similar methods to the boto3 bedrock client.
        """
        return self

    def get_bedrock_runtime_client(self):
        """
        Get a runtime client-like interface for compatibility.

        Note: This returns the delegate itself which provides
        similar methods to the boto3 bedrock-runtime client.
        """
        return self


# Singleton instance
_bedrock_delegate = None

def get_bedrock_delegate(config: Optional[Dict] = None) -> BedrockDelegate:
    """
    Get singleton Bedrock delegate instance.

    This is the main entry point for TidyLLM components to access Bedrock.
    """
    global _bedrock_delegate
    if _bedrock_delegate is None:
        _bedrock_delegate = BedrockDelegate(config)
    return _bedrock_delegate


# Export commonly used models for convenience
if PARENT_BEDROCK_AVAILABLE:
    # Use parent's BedrockModel enum
    from infrastructure.services.bedrock_service import BedrockModel
else:
    # Use mock BedrockModel
    pass  # Already defined above