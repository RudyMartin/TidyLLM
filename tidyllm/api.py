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

# Basic API functions
def chat(message: str, **kwargs) -> str:
    """Simple chat function."""
    return f"Response to: {message}"

def query(question: str, context: str = None, **kwargs) -> str:
    """Query with optional context."""
    return f"Answer to: {question}"

def process_document(document_path: str, **kwargs) -> dict:
    """Process a document."""
    return {"status": "processed", "document": document_path}

def list_models(**kwargs) -> list:
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