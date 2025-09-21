"""
Corporate LLM Gateway
====================

Corporate-compliant LLM gateway for direct Bedrock integration.
Provides controlled access to AWS Bedrock models with corporate governance.

Features:
- Direct AWS Bedrock integration
- Corporate firewall compatibility
- Cost tracking and budgeting
- Audit logging and compliance
- Model ID resolution and management
- Temperature and parameter control

Architecture: Service → CorporateLLMGateway → UnifiedSessionManager → AWS Bedrock

Usage:
    from tidyllm.gateways import CorporateLLMGateway, LLMRequest

    gateway = CorporateLLMGateway()
    request = LLMRequest(
        prompt="Hello, how are you?",
        model_id="claude-3-sonnet",
        temperature=0.7
    )
    response = gateway.process_request(request)
"""

import json
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Import infrastructure
try:
    from tidyllm.infrastructure.session.unified import UnifiedSessionManager
    USM_AVAILABLE = True
except ImportError:
    USM_AVAILABLE = False

# Import MLflow integration (avoid circular import)
MLFLOW_AVAILABLE = False
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    pass


@dataclass
class LLMRequest:
    """Request object for LLM processing."""
    prompt: str
    model_id: str = "claude-3-sonnet"
    temperature: float = 0.7
    max_tokens: int = 4000
    user_id: Optional[str] = None
    audit_reason: Optional[str] = None


@dataclass
class LLMResponse:
    """Response object from LLM processing."""
    content: str
    success: bool
    model_used: str
    processing_time_ms: float
    token_usage: Dict[str, int]
    error: Optional[str] = None
    audit_trail: Optional[Dict] = None


class CorporateLLMGateway:
    """
    Corporate-compliant LLM gateway for AWS Bedrock integration.

    Provides controlled access to LLM models with corporate governance,
    cost tracking, and audit logging.
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize Corporate LLM Gateway."""
        self.config = config or {}
        self.usm = None
        self.mlflow_service = None
        self.cost_tracker = CostTracker()

        # Model ID mappings (friendly -> Bedrock)
        self.model_mappings = {
            "claude-3-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
            "claude-3-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
            "claude-3-5-sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "claude-3-opus": "anthropic.claude-3-opus-20240229-v1:0",
            "gpt-4": "anthropic.claude-3-sonnet-20240229-v1:0",  # Fallback to Claude
            "titan-text": "amazon.titan-text-express-v1"
        }

        # Initialize components
        self._initialize_usm()
        self._initialize_mlflow()

        logger.info("CorporateLLMGateway initialized")

    def _initialize_usm(self):
        """Initialize UnifiedSessionManager."""
        if not USM_AVAILABLE:
            logger.warning("UnifiedSessionManager not available")
            return

        try:
            # Use direct USM - bypass corporate wrapper for now
            self.usm = UnifiedSessionManager()
            logger.info("Direct USM initialized for gateway")

        except Exception as e:
            logger.error(f"USM initialization failed: {e}")
            self.usm = None

    def _initialize_mlflow(self):
        """Initialize MLflow integration."""
        if MLFLOW_AVAILABLE:
            try:
                # Lazy import to avoid circular dependency
                from tidyllm.services.mlflow_integration_service import MLflowIntegrationService
                self.mlflow_service = MLflowIntegrationService()
                logger.info("MLflow integration initialized")
            except Exception as e:
                logger.warning(f"MLflow initialization failed: {e}")
                self.mlflow_service = None

    def process_request(self, request: LLMRequest) -> LLMResponse:
        """
        Process LLM request with corporate compliance.

        Args:
            request: LLM request object

        Returns:
            LLM response object
        """
        start_time = datetime.now()

        try:
            # Validate request
            self._validate_request(request)

            # Check budget and rate limits
            self._check_budget_limits(request)

            # Resolve model ID
            bedrock_model_id = self._resolve_model_id(request.model_id)

            # Get Bedrock client
            bedrock_client = self._get_bedrock_client()

            # Prepare request body
            body = self._prepare_request_body(request)

            # Log request for audit
            self._log_request(request)

            # Call Bedrock using delegate or client
            if hasattr(bedrock_client, 'invoke_model'):
                # It's either a delegate or boto3 client
                if hasattr(bedrock_client, 'is_available'):
                    # It's a delegate - use its invoke_model method
                    response_text = bedrock_client.invoke_model(
                        prompt=request.prompt,
                        model_id=bedrock_model_id,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature
                    )
                    # Create response format to match expected structure
                    response = {
                        'body': type('obj', (object,), {
                            'read': lambda: json.dumps({'content': [{'text': response_text}]}).encode()
                        })()
                    }
                else:
                    # It's a boto3 client - use original format
                    response = bedrock_client.invoke_model(
                        modelId=bedrock_model_id,
                        body=json.dumps(body),
                        contentType="application/json"
                    )

            # Parse response
            response_body = json.loads(response['body'].read())
            content = response_body['content'][0]['text']

            # Calculate metrics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            token_usage = self._calculate_token_usage(request.prompt, content)

            # Update cost tracking
            self._update_cost_tracking(request, token_usage)

            # Create response
            llm_response = LLMResponse(
                content=content,
                success=True,
                model_used=bedrock_model_id,
                processing_time_ms=processing_time,
                token_usage=token_usage,
                audit_trail=self._create_audit_trail(request, processing_time)
            )

            # Log to MLflow if available
            self._log_to_mlflow(request, llm_response)

            logger.info(f"LLM request processed successfully in {processing_time:.1f}ms")
            return llm_response

        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            return LLMResponse(
                content="",
                success=False,
                model_used=request.model_id,
                processing_time_ms=processing_time,
                token_usage={"input": 0, "output": 0},
                error=str(e),
                audit_trail=self._create_audit_trail(request, processing_time, error=str(e))
            )

    def _validate_request(self, request: LLMRequest):
        """Validate LLM request."""
        if not request.prompt or not request.prompt.strip():
            raise ValueError("Prompt cannot be empty")

        if request.temperature < 0 or request.temperature > 1:
            raise ValueError("Temperature must be between 0 and 1")

        if request.max_tokens < 1 or request.max_tokens > 100000:
            raise ValueError("Max tokens must be between 1 and 100000")

    def _check_budget_limits(self, request: LLMRequest):
        """Check budget and rate limits."""
        # In production, this would check actual budget constraints
        if self.cost_tracker.daily_spend > 1000:  # $1000 daily limit
            raise RuntimeError("Daily budget limit exceeded")

    def _resolve_model_id(self, model_id: str) -> str:
        """
        Resolve model ID to Bedrock format - accepts both short names and full IDs.

        Examples:
        - 'claude-3-haiku' -> 'anthropic.claude-3-haiku-20240307-v1:0'
        - 'anthropic.claude-3-haiku-20240307-v1:0' -> 'anthropic.claude-3-haiku-20240307-v1:0'
        """
        # If it's already a full Bedrock ID (contains dots), return as-is
        if '.' in model_id and 'anthropic.' in model_id:
            logger.debug(f"Using full Bedrock model ID: {model_id}")
            return model_id

        # If it's in our mapping, use the mapping
        if model_id in self.model_mappings:
            bedrock_id = self.model_mappings[model_id]
            logger.debug(f"Mapped {model_id} -> {bedrock_id}")
            return bedrock_id

        # Try to auto-detect common patterns
        if model_id.startswith('claude-3-'):
            # Auto-generate Bedrock ID for Claude 3 models
            version_map = {
                'claude-3-haiku': 'anthropic.claude-3-haiku-20240307-v1:0',
                'claude-3-sonnet': 'anthropic.claude-3-sonnet-20240229-v1:0',
                'claude-3-5-sonnet': 'anthropic.claude-3-5-sonnet-20240620-v1:0',
                'claude-3-opus': 'anthropic.claude-3-opus-20240229-v1:0'
            }
            if model_id in version_map:
                bedrock_id = version_map[model_id]
                logger.info(f"Auto-resolved {model_id} -> {bedrock_id}")
                return bedrock_id

        # Fallback: assume it's already a valid Bedrock ID
        logger.warning(f"Unknown model ID '{model_id}', passing through as-is")
        return model_id

    def _get_bedrock_client(self):
        """Get Bedrock client from USM with fallback support."""
        if not self.usm:
            logger.warning("UnifiedSessionManager not available, using fallback mode")
            return self._create_fallback_bedrock_client()

        try:
            bedrock_client = self.usm.get_bedrock_runtime_client()
            if not bedrock_client:
                logger.warning("USM Bedrock client not available, using fallback")
                return self._create_fallback_bedrock_client()
            return bedrock_client
        except Exception as e:
            logger.warning(f"USM Bedrock client failed: {e}, using fallback")
            return self._create_fallback_bedrock_client()

    def _create_fallback_bedrock_client(self):
        """Create fallback Bedrock client for testing when USM unavailable."""
        try:
            # Use bedrock delegate instead of direct boto3
            from tidyllm.infrastructure.bedrock_delegate import get_bedrock_delegate
            delegate = get_bedrock_delegate()
            if delegate.is_available():
                # Return the delegate which provides bedrock operations
                return delegate
            else:
                raise Exception("Bedrock delegate not available")
        except Exception as e:
            logger.warning(f"Direct boto3 client failed: {e}, using mock client")
            return self._create_mock_bedrock_client()

    def _create_mock_bedrock_client(self):
        """Create mock Bedrock client for testing without AWS credentials."""
        class MockBedrockClient:
            def invoke_model(self, modelId, body, contentType):
                """Mock Bedrock model invocation with realistic responses."""
                import json

                request_body = json.loads(body)
                prompt = request_body.get('messages', [{}])[0].get('content', '')

                # Generate mock response based on model and prompt
                if 'haiku' in modelId.lower():
                    content = f"Mock Haiku response: {prompt[:100]}... [Fast initial analysis]"
                elif 'sonnet' in modelId.lower():
                    content = f"Mock Sonnet response: {prompt[:100]}... [Enhanced detailed analysis with recommendations]"
                elif 'opus' in modelId.lower():
                    content = f"Mock Opus response: {prompt[:100]}... [Premium comprehensive analysis with strategic insights]"
                else:
                    content = f"Mock response: Analysis of workflow requirements with actionable recommendations."

                # Mock response structure matching Bedrock format
                response_body = {
                    'content': [{
                        'text': content
                    }]
                }

                class MockResponse:
                    def __init__(self, body_content):
                        self.body_content = body_content

                    def read(self):
                        return json.dumps(self.body_content).encode()

                return {
                    'body': MockResponse(response_body)
                }

        logger.info("Using mock Bedrock client for testing")
        return MockBedrockClient()

    def _prepare_request_body(self, request: LLMRequest) -> Dict:
        """Prepare request body for Bedrock."""
        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "messages": [
                {
                    "role": "user",
                    "content": request.prompt
                }
            ]
        }

    def _calculate_token_usage(self, prompt: str, response: str) -> Dict[str, int]:
        """Calculate approximate token usage."""
        # Rough estimation: 1 token ≈ 4 characters
        input_tokens = len(prompt) // 4
        output_tokens = len(response) // 4

        return {
            "input": input_tokens,
            "output": output_tokens,
            "total": input_tokens + output_tokens
        }

    def _update_cost_tracking(self, request: LLMRequest, token_usage: Dict):
        """Update cost tracking."""
        # Rough cost estimation (varies by model)
        cost_per_1k_tokens = 0.003  # ~$3 per 1M tokens
        estimated_cost = (token_usage["total"] / 1000) * cost_per_1k_tokens

        self.cost_tracker.daily_spend += estimated_cost
        self.cost_tracker.request_count += 1

        if request.user_id:
            user_cost = self.cost_tracker.user_costs.get(request.user_id, 0)
            self.cost_tracker.user_costs[request.user_id] = user_cost + estimated_cost

    def _create_audit_trail(self, request: LLMRequest, processing_time: float, error: str = None) -> Dict:
        """Create audit trail for compliance."""
        return {
            "timestamp": datetime.now().isoformat(),
            "user_id": request.user_id,
            "audit_reason": request.audit_reason,
            "model_id": request.model_id,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "processing_time_ms": processing_time,
            "success": error is None,
            "error": error
        }

    def _log_request(self, request: LLMRequest):
        """Log request for audit."""
        logger.info(f"LLM Request: model={request.model_id}, user={request.user_id}, reason={request.audit_reason}")

    def _log_to_mlflow(self, request: LLMRequest, response: LLMResponse):
        """Log to MLflow if available."""
        if self.mlflow_service:
            try:
                self.mlflow_service.log_llm_request(
                    model=request.model_id,
                    prompt=request.prompt[:100] + "..." if len(request.prompt) > 100 else request.prompt,
                    response=response.content[:100] + "..." if len(response.content) > 100 else response.content,
                    processing_time=response.processing_time_ms,
                    token_usage=response.token_usage,
                    success=response.success
                )
            except Exception as e:
                logger.warning(f"MLflow logging failed: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get adapter status."""
        return {
            "gateway_name": "CorporateLLMGateway",
            "usm_available": self.usm is not None,
            "mlflow_available": self.mlflow_service is not None,
            "supported_models": list(self.model_mappings.keys()),
            "cost_tracking": {
                "daily_spend": self.cost_tracker.daily_spend,
                "request_count": self.cost_tracker.request_count
            },
            "timestamp": datetime.now().isoformat()
        }


class CostTracker:
    """Simple cost tracking for corporate compliance."""

    def __init__(self):
        self.daily_spend = 0.0
        self.monthly_spend = 0.0
        self.request_count = 0
        self.user_costs = {}


# ==================== MAIN ====================

if __name__ == "__main__":
    print("Corporate LLM Gateway")
    print("=" * 30)

    # Test gateway functionality
    try:
        gateway = CorporateLLMGateway()

        # Check status
        print("\nGateway Status:")
        status = gateway.get_status()
        for key, value in status.items():
            if key not in ["timestamp"]:
                print(f"+ {key}: {value}")

        # Test request
        print("\nTesting LLM Request:")
        request = LLMRequest(
            prompt="Hello, how are you?",
            model_id="claude-3-sonnet",
            temperature=0.7,
            user_id="test_user",
            audit_reason="gateway_test"
        )

        response = gateway.process_request(request)
        print(f"Success: {response.success}")
        print(f"Content: {response.content[:100]}...")
        print(f"Processing time: {response.processing_time_ms:.1f}ms")

        print("\n+ Corporate LLM Gateway test completed!")

    except Exception as e:
        print(f"- Corporate LLM Gateway test failed: {e}")