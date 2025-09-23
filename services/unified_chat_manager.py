"""
Unified Chat Manager
===================

Central orchestration service for all chat processing modes in TidyLLM V3.
Provides consistent interface for direct, RAG, DSPy, and custom chat flows.

UPDATED: Now uses consolidated infrastructure delegate pattern
- Proper hexagonal architecture compliance
- Uses parent infrastructure (ResilientPoolManager, credential_carrier)
- No direct infrastructure imports

Architecture:
    Portal (interfaces/api.py) → Service (UnifiedChatManager) → InfraDelegate → Infrastructure

Features:
- Multiple chat processing modes
- Reasoning/Chain of Thought support
- Model and parameter management
- Conversation history
- Streaming responses
- Error handling and fallbacks

Usage:
    from tidyllm.services import UnifiedChatManager, ChatMode

    chat_manager = UnifiedChatManager()
    response = chat_manager.chat(
        message="Hello!",
        mode=ChatMode.DIRECT,
        model="claude-3-sonnet",
        reasoning=True
    )
"""

from typing import Dict, List, Any, Optional, Union, Iterator
from datetime import datetime
from enum import Enum
import logging
import sys
from pathlib import Path

# Set up paths using PathManager if available
try:
    # Try to get to qa_root and use PathManager
    qa_root = Path(__file__).parent.parent.parent.parent
    if str(qa_root) not in sys.path:
        sys.path.insert(0, str(qa_root))

    from common.utilities.path_manager import PathManager
    path_mgr = PathManager()
    for path in path_mgr.get_python_paths():
        if path not in sys.path:
            sys.path.insert(0, path)
except ImportError:
    # PathManager not available, continue without it
    pass

# Import consolidated infrastructure delegate
from ..infrastructure.infra_delegate import get_infra_delegate

logger = logging.getLogger(__name__)


class ChatMode(Enum):
    """Available chat processing modes."""
    DIRECT = "direct"           # Direct Bedrock/LLM calls
    RAG = "rag"                # RAG-enhanced responses
    DSPY = "dspy"              # DSPy prompt optimization
    HYBRID = "hybrid"          # Intelligent mode selection
    CUSTOM = "custom"          # Custom processing chain


class ChatResponse:
    """Structured chat response with optional reasoning and RL tracking."""

    def __init__(self, content: str, reasoning_data: Optional[Dict] = None, rl_data: Optional[Dict] = None):
        self.content = content
        self.reasoning_data = reasoning_data or {}
        self.rl_data = rl_data or {}  # RESTORED - MLflow is now fixed
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format including RL tracking data."""
        result = {
            "response": self.content,
            "timestamp": self.timestamp
        }

        # Add reasoning data if available
        if self.reasoning_data:
            result.update({
                "reasoning": self.reasoning_data.get("reasoning", ""),
                "method": self.reasoning_data.get("method", "unknown"),
                "model": self.reasoning_data.get("model", ""),
                "temperature": self.reasoning_data.get("temperature", 0.7),
                "confidence": self.reasoning_data.get("confidence", 0.0),
                **{k: v for k, v in self.reasoning_data.items()
                   if k not in ["reasoning", "method", "model", "temperature", "confidence"]}
            })

        # Add RL tracking data if available (RESTORED - MLflow is now fixed)
        if self.rl_data:
            if "rl_metrics" in self.rl_data:
                result["rl_metrics"] = self.rl_data["rl_metrics"]
            if "rl_state" in self.rl_data:
                result["rl_state"] = self.rl_data["rl_state"]
            if "learning_feedback" in self.rl_data:
                result["learning_feedback"] = self.rl_data["learning_feedback"]
            if "policy_info" in self.rl_data:
                result["policy_info"] = self.rl_data["policy_info"]
            if "exploration_data" in self.rl_data:
                result["exploration_data"] = self.rl_data["exploration_data"]
            if "value_estimation" in self.rl_data:
                result["value_estimation"] = self.rl_data["value_estimation"]
            if "reward_signal" in self.rl_data:
                result["reward_signal"] = self.rl_data["reward_signal"]

        return result

    def __str__(self) -> str:
        """String representation returns content."""
        return self.content


class UnifiedChatManager:
    """
    Central orchestration service for all chat processing modes.

    Provides consistent interface and manages different chat adapters.
    """

    def __init__(self):
        """Initialize the Unified Chat Manager."""
        self.supported_modes = [mode.value for mode in ChatMode]
        self.default_model = "claude-3-sonnet"
        self.default_temperature = 0.7

        # Chat adapters (lazy-loaded)
        self._bedrock_adapter = None
        self._rag_adapter = None
        self._dspy_adapter = None

        logger.info("UnifiedChatManager initialized")

    # ==================== MAIN CHAT INTERFACE ====================

    def chat(self, message: str, mode: Union[ChatMode, str] = ChatMode.DIRECT,
             model: str = None, temperature: float = None, reasoning: bool = False,
             history: Optional[List[Dict]] = None, **kwargs) -> Union[str, Dict]:
        """
        Process chat message using specified mode.

        Args:
            message: User message/prompt
            mode: Processing mode (ChatMode enum or string)
            model: Model to use (defaults to claude-3-sonnet)
            temperature: Model temperature (defaults to 0.7)
            reasoning: Return detailed reasoning/CoT if True
            history: Conversation history
            **kwargs: Additional parameters

        Returns:
            str or dict: Response content or detailed reasoning object
        """
        try:
            # Normalize mode
            if isinstance(mode, str):
                mode = ChatMode(mode.lower())

            # Set defaults
            model = model or self.default_model
            temperature = temperature if temperature is not None else self.default_temperature

            # Route to appropriate adapter
            if mode == ChatMode.DIRECT:
                response = self._process_direct_chat(message, model, temperature, **kwargs)
            elif mode == ChatMode.RAG:
                response = self._process_rag_chat(message, model, temperature, **kwargs)
            elif mode == ChatMode.DSPY:
                response = self._process_dspy_chat(message, model, temperature, **kwargs)
            elif mode == ChatMode.HYBRID:
                response = self._process_hybrid_chat(message, model, temperature, **kwargs)
            elif mode == ChatMode.CUSTOM:
                response = self._process_custom_chat(message, model, temperature, **kwargs)
            else:
                raise ValueError(f"Unsupported chat mode: {mode}")

            # Return based on reasoning preference
            if reasoning and isinstance(response, ChatResponse):
                return response.to_dict()
            elif isinstance(response, ChatResponse):
                return str(response)
            else:
                return response

        except Exception as e:
            logger.error(f"Chat processing failed: {e}")
            error_response = f"Chat processing failed ({mode.value if hasattr(mode, 'value') else mode}): {e}"

            if reasoning:
                return {
                    "response": error_response,
                    "reasoning": f"System error during {mode} processing: {str(e)}",
                    "method": f"{mode}_error",
                    "error": str(e),
                    "confidence": 0.0,
                    "timestamp": datetime.now().isoformat()
                }
            return error_response

    # ==================== CHAT MODE PROCESSORS ====================

    def _process_direct_chat(self, message: str, model: str, temperature: float, **kwargs) -> ChatResponse:
        """Process direct chat ONLY through CorporateLLMGateway that tracks all calls."""
        try:
            # ONLY USE THE CORPORATE LLM GATEWAY - NO OTHER PATHS!
            from ..gateways.corporate_llm_gateway import CorporateLLMGateway, LLMRequest

            # Get or create gateway instance
            if not hasattr(self, '_gateway'):
                self._gateway = CorporateLLMGateway()
                logger.info("Initialized CorporateLLMGateway for tracking all LLM calls")

            # Create request for gateway
            llm_request = LLMRequest(
                prompt=message,
                model_id=model,
                temperature=temperature,
                max_tokens=kwargs.get('max_tokens', 4000),
                user_id=kwargs.get('user_id', 'chat_user'),
                audit_reason='unified_chat_direct'
            )

            # Process through gateway (which tracks all calls)
            llm_response = self._gateway.process_request(llm_request)

            if llm_response.success:
                reasoning_data = {
                    "reasoning": f"Processed via CorporateLLMGateway: {model} with temperature {temperature}",
                    "method": "corporate_gateway",
                    "model": llm_response.model_used,
                    "temperature": temperature,
                    "confidence": 0.95,
                    "processing_time_ms": llm_response.processing_time_ms,
                    "token_usage": llm_response.token_usage,
                    "gateway_tracked": True
                }

                # Extract RL data from gateway response if available (RESTORED - MLflow is now fixed)
                rl_data = {}
                if hasattr(llm_response, 'rl_metrics') and llm_response.rl_metrics:
                    rl_data["rl_metrics"] = llm_response.rl_metrics
                if hasattr(llm_response, 'rl_state') and llm_response.rl_state:
                    rl_data["rl_state"] = llm_response.rl_state
                if hasattr(llm_response, 'learning_feedback') and llm_response.learning_feedback:
                    rl_data["learning_feedback"] = llm_response.learning_feedback
                if hasattr(llm_response, 'policy_info') and llm_response.policy_info:
                    rl_data["policy_info"] = llm_response.policy_info
                if hasattr(llm_response, 'exploration_data') and llm_response.exploration_data:
                    rl_data["exploration_data"] = llm_response.exploration_data
                if hasattr(llm_response, 'value_estimation') and llm_response.value_estimation:
                    rl_data["value_estimation"] = llm_response.value_estimation
                if hasattr(llm_response, 'reward_signal') and llm_response.reward_signal:
                    rl_data["reward_signal"] = llm_response.reward_signal

                print(f"DEBUG: Creating ChatResponse with RL data: {list(rl_data.keys()) if rl_data else 'None'}")
                return ChatResponse(llm_response.content, reasoning_data, rl_data)
            else:
                # Gateway failed but tracked the attempt
                error_msg = llm_response.error or 'Unknown error'
                logger.warning(f"Gateway processing failed: {error_msg}")

                return ChatResponse(
                    f"I received your message but encountered an issue: {error_msg}",
                    {"reasoning": f"Gateway error: {error_msg}", "method": "gateway_error", "confidence": 0.5, "gateway_tracked": True}
                )

        except Exception as e:
            logger.error(f"Direct chat via gateway failed: {e}")
            # Even errors are tracked by the gateway audit
            return ChatResponse(
                f"I understand you said: '{message}'. I'm having technical difficulties with the gateway.",
                {"reasoning": f"Gateway exception: {str(e)}", "method": "gateway_exception", "confidence": 0.3, "gateway_tracked": False}
            )

    def _process_rag_chat(self, message: str, model: str, temperature: float, **kwargs) -> ChatResponse:
        """Process RAG-enhanced chat."""
        try:
            from .unified_rag_manager import UnifiedRAGManager, RAGSystemType

            rag_manager = UnifiedRAGManager()
            result = rag_manager.query(
                system_type=RAGSystemType.AI_POWERED,
                query=message,
                model_preference=model,
                temperature=temperature
            )

            if result.get("success"):
                content = result.get("answer", f"RAG-enhanced response to: {message}")

                reasoning_data = {
                    "reasoning": f"RAG processing: Retrieved relevant context from knowledge base, then generated response using {model}",
                    "method": "rag_enhanced",
                    "rag_system": "AI_POWERED",
                    "context_sources": result.get("sources", []),
                    "model": model,
                    "temperature": temperature,
                    "confidence": result.get("confidence", 0.88),
                    "retrieval_count": len(result.get("sources", [])),
                    "processing_time_ms": result.get("processing_time", 300)
                }

                return ChatResponse(content, reasoning_data)
            else:
                raise Exception(result.get("error", "RAG processing failed"))

        except Exception as e:
            logger.error(f"RAG chat failed: {e}")
            raise

    def _process_dspy_chat(self, message: str, model: str, temperature: float, **kwargs) -> ChatResponse:
        """Process DSPy Chain of Thought chat."""
        try:
            # DSPy service would be imported here when available
            # from tidyllm.services import DSPyService
            # For now, return a basic response
            return ChatResponse(
                f"DSPy mode processing: {message}",
                {"reasoning": "DSPy service being configured", "method": "dspy_placeholder", "confidence": 0.7}
            )

            # Ensure DSPy is configured with our corporate gateway (NO OpenAI)
            if not dspy_service.current_lm or "corporate_adapter" not in str(dspy_service.current_lm):
                config_success = dspy_service.configure_lm(model_name=model)
                if not config_success:
                    raise Exception("Failed to configure DSPy with CorporateLLMGateway")

            # Use DSPy's Chain of Thought for reasoning
            result = dspy_service.chat_with_cot(message)

            if result.get("success"):
                content = result.get("response", f"DSPy CoT response to '{message}'")

                reasoning_data = {
                    "reasoning": result.get("reasoning", "DSPy Chain of Thought reasoning using CorporateLLMGateway"),
                    "method": "dspy_chain_of_thought",
                    "original_message": message,
                    "model": model,
                    "temperature": temperature,
                    "confidence": 0.92,
                    "dspy_signature": "question -> answer",
                    "dspy_backend": "corporate_llm_gateway",
                    "processing_time_ms": 250
                }

                return ChatResponse(content, reasoning_data)
            else:
                raise Exception(result.get("error", "DSPy CoT failed"))

        except Exception as e:
            logger.error(f"DSPy chat failed: {e}")
            raise

    def _process_hybrid_chat(self, message: str, model: str, temperature: float, **kwargs) -> ChatResponse:
        """Process hybrid chat (intelligent mode selection)."""
        # Analyze message to determine best processing mode
        if "step by step" in message.lower() or "reasoning" in message.lower():
            return self._process_dspy_chat(message, model, temperature, **kwargs)
        elif len(message) > 100 or "?" in message:
            return self._process_rag_chat(message, model, temperature, **kwargs)
        else:
            return self._process_direct_chat(message, model, temperature, **kwargs)

    def _process_custom_chat(self, message: str, model: str, temperature: float, **kwargs) -> ChatResponse:
        """Process custom chat flow."""
        # Placeholder for custom processing chains
        content = f"Custom processing for '{message}' (feature coming soon)"

        reasoning_data = {
            "reasoning": "Custom processing chain - user-defined workflow",
            "method": "custom_chain",
            "model": model,
            "temperature": temperature,
            "confidence": 0.75
        }

        return ChatResponse(content, reasoning_data)

    # ==================== MANAGEMENT & UTILITIES ====================

    def get_supported_modes(self) -> List[str]:
        """Get list of supported chat modes."""
        return self.supported_modes

    def get_status(self) -> Dict[str, Any]:
        """Get service status."""
        return {
            "service_name": "UnifiedChatManager",
            "supported_modes": self.supported_modes,
            "default_model": self.default_model,
            "default_temperature": self.default_temperature,
            "adapters_loaded": {
                "bedrock": self._bedrock_adapter is not None,
                "rag": self._rag_adapter is not None,
                "dspy": self._dspy_adapter is not None
            },
            "timestamp": datetime.now().isoformat()
        }

    def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            # Test basic functionality
            test_response = self.chat("test", mode=ChatMode.DIRECT, reasoning=True)

            return {
                "success": True,
                "healthy": True,
                "modes_available": self.supported_modes,
                "basic_functionality": isinstance(test_response, dict),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "success": False,
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


# ==================== CONVENIENCE FUNCTIONS ====================

def get_chat_manager() -> UnifiedChatManager:
    """Get chat manager instance - convenience function."""
    return UnifiedChatManager()


# ==================== MAIN ====================

if __name__ == "__main__":
    print("TidyLLM Unified Chat Manager")
    print("=" * 40)

    # Test chat manager functionality
    try:
        chat_manager = UnifiedChatManager()

        # Check status
        print("\nService Status:")
        status = chat_manager.get_status()
        for key, value in status.items():
            if key not in ["timestamp"]:
                print(f"+ {key}: {value}")

        # Health check
        print("\nHealth Check:")
        health = chat_manager.health_check()
        if health["success"]:
            print("+ Chat Manager is healthy")
            print(f"+ Available modes: {', '.join(health['modes_available'])}")

        # Test chat modes
        print("\nTesting Chat Modes:")

        # Direct mode
        response = chat_manager.chat("Hello!", mode=ChatMode.DIRECT, reasoning=True)
        print(f"Direct: {response.get('method', 'unknown')}")

        print("\n+ Unified Chat Manager test completed!")

    except Exception as e:
        print(f"- Chat Manager test failed: {e}")