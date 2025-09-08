"""
TidyLLM Infrastructure Components
================================
"""

class UnifiedSessionManager:
    """Basic session manager for gateway compatibility."""
    
    def __init__(self):
        self.active = True
        self.session_id = "default_session"
    
    def get_session(self):
        """Get current session."""
        return {"session_id": self.session_id, "active": self.active}
    
    def create_session(self):
        """Create new session."""
        import uuid
        self.session_id = str(uuid.uuid4())
        return self.get_session()

# Export for compatibility
__all__ = ['UnifiedSessionManager']