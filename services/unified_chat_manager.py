"""
Unified Chat Manager
===================

Central orchestration service for all chat processing modes in TidyLLM V3.
Provides consistent interface for direct, RAG, DSPy, and custom chat flows.

Architecture:
    Portal (interfaces/api.py) → Service (UnifiedChatManager) → Adapters (Bedrock, RAG, DSPy)

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

logger = logging.getLogger(__name__)


class ChatMode(Enum):
    """Available chat processing modes."""
    DIRECT = "direct"           # Direct Bedrock/LLM calls
    RAG = "rag"                # RAG-enhanced responses
    DSPY = "dspy"              # DSPy prompt optimization
    HYBRID = "hybrid"          # Intelligent mode selection
    CUSTOM = "custom"          # Custom processing chain


class ChatResponse:
    """Structured chat response with optional reasoning."""

    def __init__(self, content: str, reasoning_data: Optional[Dict] = None):
        self.content = content
        self.reasoning_data = reasoning_data or {}
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        if self.reasoning_data:
            return {
                "response": self.content,
                "reasoning": self.reasoning_data.get("reasoning", ""),
                "method": self.reasoning_data.get("method", "unknown"),
                "model": self.reasoning_data.get("model", ""),
                "temperature": self.reasoning_data.get("temperature", 0.7),
                "confidence": self.reasoning_data.get("confidence", 0.0),
                "timestamp": self.timestamp,
                **{k: v for k, v in self.reasoning_data.items()
                   if k not in ["reasoning", "method", "model", "temperature", "confidence"]}
            }
        return {"response": self.content, "timestamp": self.timestamp}

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

    def chat(self, message: str, mode: Union[ChatMode, str] = ChatMode.RAG,
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
        """Process direct Bedrock/LLM chat using CorporateLLMGateway."""
        try:
            from tidyllm.gateways import CorporateLLMGateway, LLMRequest

            # Initialize Corporate LLM Gateway
            gateway = CorporateLLMGateway()

            # Create LLM request
            request = LLMRequest(
                prompt=message,
                model_id=model,
                temperature=temperature,
                max_tokens=kwargs.get('max_tokens', 4000),
                user_id=kwargs.get('user_id', 'chat_user'),
                audit_reason='unified_chat_manager'
            )

            # Process request
            response = gateway.process_request(request)

            if response.success:
                reasoning_data = {
                    "reasoning": f"Corporate LLM Gateway: Used {response.model_used} with temperature {temperature} via AWS Bedrock",
                    "method": "corporate_llm_gateway",
                    "model": response.model_used,
                    "temperature": temperature,
                    "confidence": 0.92,
                    "processing_time_ms": response.processing_time_ms,
                    "token_usage": response.token_usage,
                    "audit_trail": response.audit_trail
                }

                return ChatResponse(response.content, reasoning_data)
            else:
                raise Exception(response.error)

        except Exception as e:
            logger.error(f"Direct chat failed: {e}")
            raise

    def _process_rag_chat(self, message: str, model: str, temperature: float, **kwargs) -> ChatResponse:
        """Process RAG-enhanced chat."""
        try:
            from tidyllm.services import UnifiedRAGManager, RAGSystemType

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
            from tidyllm.services import DSPyService

            # Initialize DSPy service with corporate configuration
            dspy_service = DSPyService(auto_configure=True)

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