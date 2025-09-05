"""
MCP Context Store

Persistent storage for MCP contexts.
"""

import logging
import json
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from .context_manager import MCPContext


class ContextStore:
    """Persistent storage for MCP contexts"""
    
    def __init__(self, storage_path: str = "contexts"):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.contexts_file = self.storage_path / "contexts.json"
        self.history_file = self.storage_path / "history.json"
        self.metadata_file = self.storage_path / "metadata.json"

    def save_context(self, context: MCPContext) -> bool:
        """Save context to persistent storage"""
        try:
            # Load existing contexts
            contexts = self._load_contexts()
            
            # Add/update context
            contexts[context.context_id] = context.to_dict()
            
            # Save back to file
            with open(self.contexts_file, 'w') as f:
                json.dump(contexts, f, indent=2, default=str)
            
            self.logger.info(f"Saved context {context.context_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save context {context.context_id}: {e}")
            return False

    def load_context(self, context_id: str) -> Optional[MCPContext]:
        """Load context from persistent storage"""
        try:
            contexts = self._load_contexts()
            context_data = contexts.get(context_id)
            
            if not context_data:
                return None
            
            return self._dict_to_context(context_data)
            
        except Exception as e:
            self.logger.error(f"Failed to load context {context_id}: {e}")
            return None

    def load_all_contexts(self) -> Dict[str, MCPContext]:
        """Load all contexts from persistent storage"""
        try:
            contexts = self._load_contexts()
            return {
                context_id: self._dict_to_context(context_data)
                for context_id, context_data in contexts.items()
            }
        except Exception as e:
            self.logger.error(f"Failed to load contexts: {e}")
            return {}

    def delete_context(self, context_id: str) -> bool:
        """Delete context from persistent storage"""
        try:
            contexts = self._load_contexts()
            
            if context_id in contexts:
                # Move to history before deleting
                self._add_to_history(contexts[context_id])
                del contexts[context_id]
                
                # Save updated contexts
                with open(self.contexts_file, 'w') as f:
                    json.dump(contexts, f, indent=2, default=str)
                
                self.logger.info(f"Deleted context {context_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to delete context {context_id}: {e}")
            return False

    def save_contexts_batch(self, contexts: List[MCPContext]) -> bool:
        """Save multiple contexts in batch"""
        try:
            contexts_dict = self._load_contexts()
            
            for context in contexts:
                contexts_dict[context.context_id] = context.to_dict()
            
            with open(self.contexts_file, 'w') as f:
                json.dump(contexts_dict, f, indent=2, default=str)
            
            self.logger.info(f"Saved {len(contexts)} contexts in batch")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save contexts batch: {e}")
            return False

    def get_context_statistics(self) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            contexts = self._load_contexts()
            history = self._load_history()
            
            total_contexts = len(contexts)
            total_history = len(history)
            
            # Calculate storage size
            contexts_size = os.path.getsize(self.contexts_file) if self.contexts_file.exists() else 0
            history_size = os.path.getsize(self.history_file) if self.history_file.exists() else 0
            
            return {
                "total_contexts": total_contexts,
                "total_history": total_history,
                "storage_size_bytes": contexts_size + history_size,
                "contexts_file_size": contexts_size,
                "history_file_size": history_size,
                "storage_path": str(self.storage_path)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get storage statistics: {e}")
            return {}

    def cleanup_expired_contexts(self) -> int:
        """Clean up expired contexts from storage"""
        try:
            contexts = self._load_contexts()
            expired_count = 0
            
            for context_id, context_data in list(contexts.items()):
                context = self._dict_to_context(context_data)
                if context and context.is_expired():
                    # Move to history
                    self._add_to_history(context_data)
                    del contexts[context_id]
                    expired_count += 1
            
            # Save updated contexts
            if expired_count > 0:
                with open(self.contexts_file, 'w') as f:
                    json.dump(contexts, f, indent=2, default=str)
                
                self.logger.info(f"Cleaned up {expired_count} expired contexts")
            
            return expired_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup expired contexts: {e}")
            return 0

    def backup_contexts(self, backup_path: str) -> bool:
        """Create backup of all contexts"""
        try:
            backup_dir = Path(backup_path)
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy context files
            import shutil
            shutil.copy2(self.contexts_file, backup_dir / "contexts.json")
            shutil.copy2(self.history_file, backup_dir / "history.json")
            shutil.copy2(self.metadata_file, backup_dir / "metadata.json")
            
            self.logger.info(f"Created backup at {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            return False

    def restore_contexts(self, backup_path: str) -> bool:
        """Restore contexts from backup"""
        try:
            backup_dir = Path(backup_path)
            
            # Verify backup files exist
            required_files = ["contexts.json", "history.json", "metadata.json"]
            for file in required_files:
                if not (backup_dir / file).exists():
                    raise FileNotFoundError(f"Backup file {file} not found")
            
            # Restore files
            import shutil
            shutil.copy2(backup_dir / "contexts.json", self.contexts_file)
            shutil.copy2(backup_dir / "history.json", self.history_file)
            shutil.copy2(backup_dir / "metadata.json", self.metadata_file)
            
            self.logger.info(f"Restored contexts from {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore contexts: {e}")
            return False

    def _load_contexts(self) -> Dict[str, Any]:
        """Load contexts from file"""
        if not self.contexts_file.exists():
            return {}
        
        try:
            with open(self.contexts_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load contexts file: {e}")
            return {}

    def _load_history(self) -> List[Dict[str, Any]]:
        """Load history from file"""
        if not self.history_file.exists():
            return []
        
        try:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load history file: {e}")
            return []

    def _add_to_history(self, context_data: Dict[str, Any]):
        """Add context to history"""
        try:
            history = self._load_history()
            history.append({
                **context_data,
                "deleted_at": datetime.now().isoformat()
            })
            
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Failed to add to history: {e}")

    def _dict_to_context(self, context_data: Dict[str, Any]) -> Optional[MCPContext]:
        """Convert dictionary to MCPContext"""
        try:
            return MCPContext(
                context_id=context_data["context_id"],
                context_data=context_data["context_data"],
                source_layer=context_data["source_layer"],
                created_at=datetime.fromisoformat(context_data["created_at"]),
                updated_at=datetime.fromisoformat(context_data["updated_at"]),
                expiry_time=datetime.fromisoformat(context_data["expiry_time"]) if context_data["expiry_time"] else None,
                metadata=context_data["metadata"],
                version=context_data["version"]
            )
        except Exception as e:
            self.logger.error(f"Failed to convert dict to context: {e}")
            return None
