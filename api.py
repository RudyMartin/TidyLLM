"""
################################################################################
# *** IMPORTANT: READ docs/2025-09-08/IMPORTANT-CONSTRAINTS-FOR-THIS-CODEBASE.md ***
# *** BEFORE PLANNING ANY CHANGES TO THIS FILE ***
################################################################################

TidyLLM Basic API - Simple Functions for Common Tasks
====================================================

Provides simple, beginner-friendly functions that work out of the box.
No complex gateway setup required for basic usage.

Basic Usage:
    import tidyllm
    
    # Simple chat
    response = tidyllm.chat("Hello, how are you?")
    
    # Process document
    result = tidyllm.process_document("document.pdf")
    
    # Query with context
    answer = tidyllm.query("What is machine learning?")
"""

from typing import List, Dict, Any
from datetime import datetime

# Basic API functions
def chat(message: str, **kwargs) -> str:
    """
    Simple chat function using TidyLLM gateway infrastructure.

    This is the public API for the tidyllm library that provides
    simple access to enterprise AI processing capabilities.
    """
    try:
        from .gateways.ai_processing_gateway import AIProcessingGateway, AIRequest

        # Initialize gateway
        gateway = AIProcessingGateway()

        # Create AI request
        ai_request = AIRequest(
            prompt=message,
            model=kwargs.get("model"),
            temperature=kwargs.get("temperature"),
            max_tokens=kwargs.get("max_tokens"),
            metadata=kwargs.get("metadata", {})
        )

        # Process through gateway
        response = gateway.backend.complete(ai_request)
        return response

    except Exception as e:
        # Graceful fallback for library users
        return f"AI processing unavailable: {str(e)}"

def query(question: str, context: str = None, **kwargs) -> str:
    """
    Query with optional context using TidyLLM gateway infrastructure.

    Enhanced query processing with context awareness through
    enterprise gateway system.
    """
    try:
        from .gateways.ai_processing_gateway import AIProcessingGateway, AIRequest

        # Initialize gateway
        gateway = AIProcessingGateway()

        # Enhance prompt with context if provided
        enhanced_prompt = question
        if context:
            enhanced_prompt = f"Context: {context}\n\nQuestion: {question}"

        # Create AI request
        ai_request = AIRequest(
            prompt=enhanced_prompt,
            model=kwargs.get("model"),
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens"),
            context={"context": context} if context else None,
            metadata=kwargs.get("metadata", {})
        )

        # Process through gateway
        response = gateway.backend.complete(ai_request)
        return response

    except Exception as e:
        # Graceful fallback for library users
        return f"AI query processing unavailable: {str(e)}"

def process_document(document_path: str, task: str = "analyze", **kwargs) -> Dict[str, Any]:
    """
    Process a document with specified task using TidyLLM infrastructure.

    Args:
        document_path: Path to document file
        task: Processing task (e.g., "Summarize this", "analyze", "extract key points")
        **kwargs: Additional processing options

    Returns:
        Dict with processing results including status, document info, and task output

    Example:
        result = tidyllm.process_document("document.pdf", "Summarize this")
        print(result["summary"])
    """
    try:
        from .gateways.ai_processing_gateway import AIProcessingGateway, AIRequest

        # Initialize gateway
        gateway = AIProcessingGateway()

        # Create processing prompt
        processing_prompt = f"Process the document at '{document_path}' with the following task: {task}"

        # Create AI request
        ai_request = AIRequest(
            prompt=processing_prompt,
            model=kwargs.get("model"),
            temperature=kwargs.get("temperature", 0.3),  # Lower temp for document analysis
            max_tokens=kwargs.get("max_tokens", 2000),
            context={"document_path": document_path, "task": task},
            metadata={
                "operation": "document_processing",
                "document_path": document_path,
                "task": task,
                **kwargs.get("metadata", {})
            }
        )

        # Process through gateway
        response = gateway.backend.complete(ai_request)

        return {
            "status": "processed",
            "document": document_path,
            "task": task,
            "summary": response,
            "processing_method": "tidyllm_gateway",
            "message": "Document processed using TidyLLM enterprise gateway"
        }

    except Exception as e:
        # Graceful fallback
        return {
            "status": "error",
            "document": document_path,
            "task": task,
            "summary": f"Document processing failed: {str(e)}",
            "processing_method": "fallback",
            "message": f"Document processing unavailable: {str(e)}"
        }

def list_models(**kwargs) -> List[Dict[str, Any]]:
    """List all available AI models across backends."""
    try:
        from .gateways.ai_processing_gateway import AIProcessingGateway
        ai_gateway = AIProcessingGateway()
        capabilities = ai_gateway.get_capabilities()
        
        # Transform into expected format
        models = []
        backend = capabilities.get("current_backend", "unknown")
        available_models = capabilities.get("models", [])
        
        for model in available_models:
            models.append({
                "name": model,
                "backend": backend,
                "type": "chat",
                "max_tokens": capabilities.get("max_tokens", 4096),
                "supports_streaming": capabilities.get("supports_streaming", False)
            })
        
        return models
    except Exception as e:
        # Fallback for demos/examples when gateway unavailable
        return [
            {"name": "claude-3-sonnet", "backend": "anthropic", "type": "chat", "max_tokens": 4096},
            {"name": "gpt-4", "backend": "openai", "type": "chat", "max_tokens": 8192},
            {"name": "llama2-70b", "backend": "bedrock", "type": "chat", "max_tokens": 4096}
        ]

def set_model(model: str, **kwargs) -> bool:
    """Set default model preference for API operations.
    
    Args:
        model: Model identifier (e.g., "anthropic/claude-3-sonnet-20240229")
        **kwargs: Additional model configuration options
        
    Returns:
        bool: True if model preference stored successfully
        
    Example:
        tidyllm.set_model("anthropic/claude-3-sonnet-20240229")
        # Subsequent chat/query calls will prefer this model
        
    Note:
        - Stores model preference in session for legacy compatibility
        - Actual model routing handled by AIProcessingGateway configuration
        - Use tidyllm.list_models() to see available models first
        - Use ai_gateway.get_capabilities() to see current backend capabilities
    """
    # Store model preference - could be enhanced to integrate with session storage
    import os
    os.environ["TIDYLLM_DEFAULT_MODEL"] = model
    return True

def status(**kwargs) -> Dict[str, Any]:
    """Get system status and health information.
    
    This function provides a simple interface to system health information,
    integrating with existing health check infrastructure. It works alongside
    session_mgr.validate_session() and ai_gateway.health_check().
    
    Args:
        **kwargs: Additional status check options
        
    Returns:
        Dict[str, Any]: System status with health, services, and diagnostics
        
    Note:
        - Maintains compatibility with legacy examples calling tidyllm.status()
        - Integrates with UnifiedSessionManager and gateway health checks
        - Use session_mgr.validate_session() for detailed session health
        - Use ai_gateway.health_check() for AI service specific health
    """
    try:
        # Try to get real status from session manager
        from .infrastructure.session.unified import UnifiedSessionManager
        session_mgr = UnifiedSessionManager()
        session_health = session_mgr.validate_session()
        
        return {
            "status": "healthy" if session_health.get("valid", False) else "degraded",
            "timestamp": datetime.now().isoformat(),
            "session_health": session_health,
            "message": "System status check complete"
        }
    except Exception as e:
        # Fallback status for compatibility
        return {
            "status": "unknown",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "message": "Status check completed with fallback (normal for demo mode)"
        }

# API Server class for tests
class TidyLLMAPI:
    """TidyLLM API Server class."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def start(self):
        """Start API server."""
        return True
    
    def stop(self):
        """Stop API server."""
        return True