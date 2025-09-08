"""
TidyLLM Unified Session Manager
==============================
Provides session management for gateway systems and infrastructure components.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class UnifiedSessionManager:
    """
    Unified session manager for TidyLLM infrastructure.
    
    Provides session tracking, resource management, and coordination
    across gateway systems and infrastructure components.
    """
    
    def __init__(self):
        """Initialize the session manager."""
        self.session_id = str(uuid.uuid4())
        self.created_at = datetime.now()
        self.active = True
        self.sessions = {}
        self.resources = {}
        
        logger.info(f"UnifiedSessionManager initialized: {self.session_id}")
    
    def create_session(self, session_type: str = "default") -> str:
        """Create a new session."""
        session_id = str(uuid.uuid4())
        session_info = {
            "id": session_id,
            "type": session_type,
            "created_at": datetime.now(),
            "active": True,
            "resources": [],
            "metadata": {}
        }
        
        self.sessions[session_id] = session_info
        logger.info(f"Created session: {session_id} ({session_type})")
        return session_id
    
    def get_session(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get session information."""
        if session_id is None:
            session_id = self.session_id
        
        if session_id in self.sessions:
            return self.sessions[session_id]
        
        # Return default session info
        return {
            "id": self.session_id,
            "type": "default",
            "created_at": self.created_at,
            "active": self.active,
            "resources": [],
            "metadata": {}
        }
    
    def close_session(self, session_id: str) -> bool:
        """Close a session."""
        if session_id in self.sessions:
            self.sessions[session_id]["active"] = False
            logger.info(f"Closed session: {session_id}")
            return True
        return False
    
    def register_resource(self, resource_id: str, resource_type: str, metadata: Dict[str, Any] = None):
        """Register a resource with the session manager."""
        self.resources[resource_id] = {
            "id": resource_id,
            "type": resource_type,
            "metadata": metadata or {},
            "registered_at": datetime.now()
        }
        logger.debug(f"Registered resource: {resource_id} ({resource_type})")
    
    def unregister_resource(self, resource_id: str) -> bool:
        """Unregister a resource."""
        if resource_id in self.resources:
            del self.resources[resource_id]
            logger.debug(f"Unregistered resource: {resource_id}")
            return True
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get session manager status."""
        return {
            "session_manager_id": self.session_id,
            "active": self.active,
            "created_at": self.created_at.isoformat(),
            "total_sessions": len(self.sessions),
            "active_sessions": sum(1 for s in self.sessions.values() if s["active"]),
            "total_resources": len(self.resources),
            "available": True
        }
    
    def cleanup(self):
        """Clean up resources and sessions."""
        inactive_sessions = [sid for sid, info in self.sessions.items() if not info["active"]]
        for session_id in inactive_sessions:
            del self.sessions[session_id]
        
        logger.info(f"Cleaned up {len(inactive_sessions)} inactive sessions")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


# Global instance
_global_session_manager = None


def get_unified_session_manager() -> UnifiedSessionManager:
    """Get global session manager instance."""
    global _global_session_manager
    
    if _global_session_manager is None:
        _global_session_manager = UnifiedSessionManager()
    
    return _global_session_manager


def create_session_manager() -> UnifiedSessionManager:
    """Create a new session manager instance."""
    return UnifiedSessionManager()


# Export main class for compatibility
__all__ = ['UnifiedSessionManager', 'get_unified_session_manager', 'create_session_manager']