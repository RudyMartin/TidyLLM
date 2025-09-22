#!/usr/bin/env python3
"""
LLM Delegate - Infrastructure Layer
====================================

Delegate for LLM operations following hexagonal architecture.
Provides clean interface for adapters without exposing infrastructure details.
"""

from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class LLMDelegate:
    """
    Delegate for LLM operations.

    Encapsulates all LLM infrastructure access.
    Adapters use this delegate instead of direct imports.
    """

    def __init__(self):
        """Initialize LLM delegate with lazy loading."""
        self._gateway = None
        self._initialized = False

    def _initialize(self):
        """Lazy initialization of LLM infrastructure."""
        if self._initialized:
            return True

        try:
            # Import only when needed (lazy loading)
            from tidyllm.gateways.corporate_llm_gateway import CorporateLLMGateway
            from tidyllm.infrastructure.standards import TidyLLMStandardRequest

            self._gateway = CorporateLLMGateway()
            self._request_class = TidyLLMStandardRequest
            self._initialized = True

            logger.info("LLM delegate initialized successfully")
            return True

        except ImportError as e:
            logger.warning(f"LLM infrastructure not available: {e}")
            self._initialized = False
            return False

    def generate_response(self, prompt: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate LLM response.

        Args:
            prompt: Input prompt
            config: Optional configuration (model, temperature, etc.)

        Returns:
            Response dictionary with text and metadata
        """
        if not self._initialize():
            return {
                'text': 'LLM service unavailable',
                'success': False,
                'error': 'Infrastructure not initialized'
            }

        config = config or {}

        try:
            # Create standard request
            request = self._request_class(
                model_id=config.get('model', 'claude-3-sonnet'),
                user_id='rag_system',
                session_id=f"rag_{config.get('session_id', 'default')}",
                prompt=prompt,
                temperature=config.get('temperature', 0.7),
                max_tokens=config.get('max_tokens', 1500)
            )

            # Process through gateway
            response = self._gateway.process_llm_request(request)

            if response.status == 'SUCCESS':
                return {
                    'text': response.data,
                    'success': True,
                    'model': request.model_id,
                    'tokens_used': response.tokens_used
                }
            else:
                return {
                    'text': 'Generation failed',
                    'success': False,
                    'error': response.error
                }

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return {
                'text': 'Generation error',
                'success': False,
                'error': str(e)
            }

    def generate_structured_response(self, prompt: str, schema: Dict[str, Any],
                                    config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate structured LLM response following a schema.

        Args:
            prompt: Input prompt
            schema: Expected response schema
            config: Optional configuration

        Returns:
            Structured response following schema
        """
        # Add schema instructions to prompt
        structured_prompt = f"{prompt}\n\nRespond in JSON format following this schema: {schema}"

        response = self.generate_response(structured_prompt, config)

        if response.get('success'):
            try:
                import json
                structured = json.loads(response['text'])
                return {
                    'data': structured,
                    'success': True,
                    'model': response.get('model')
                }
            except json.JSONDecodeError:
                return {
                    'data': {'text': response['text']},
                    'success': False,
                    'error': 'Failed to parse structured response'
                }
        else:
            return response

    def get_available_models(self) -> List[str]:
        """Get list of available LLM models."""
        if not self._initialize():
            return []

        # Return known models
        return [
            'claude-3-sonnet',
            'claude-3-opus',
            'claude-3-haiku',
            'gpt-4',
            'gpt-3.5-turbo'
        ]

    def is_available(self) -> bool:
        """Check if LLM service is available."""
        return self._initialize()


class LLMDelegateFactory:
    """Factory for creating LLM delegates."""

    _instance = None

    @classmethod
    def get_delegate(cls) -> LLMDelegate:
        """Get singleton LLM delegate instance."""
        if cls._instance is None:
            cls._instance = LLMDelegate()
        return cls._instance


def get_llm_delegate() -> LLMDelegate:
    """Get LLM delegate instance."""
    return LLMDelegateFactory.get_delegate()