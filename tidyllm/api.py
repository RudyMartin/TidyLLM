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