"""
MCP Context Manager

Manages context flow through the MCP hierarchy.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import uuid
from dataclasses import dataclass, field


@dataclass
class MCPContext:
    """MCP context data structure"""
    context_id: str
    context_data: Dict[str, Any]
    source_layer: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    expiry_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: int = 1

    def is_expired(self) -> bool:
        """Check if context has expired"""
        if self.expiry_time is None:
            return False
        return datetime.now() > self.expiry_time

    def update(self, new_data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
        """Update context data"""
        self.context_data.update(new_data)
        self.updated_at = datetime.now()
        self.version += 1
        if metadata:
            self.metadata.update(metadata)

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary"""
        return {
            "context_id": self.context_id,
            "context_data": self.context_data,
            "source_layer": self.source_layer,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expiry_time": self.expiry_time.isoformat() if self.expiry_time else None,
            "metadata": self.metadata,
            "version": self.version
        }


class MCPContextManager:
    """Manages context flow through the MCP hierarchy"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.contexts: Dict[str, MCPContext] = {}
        self.context_history: List[MCPContext] = []
        self.max_contexts = 1000
        self.default_expiry_hours = 24

    def create_context(self,
                      context_data: Dict[str, Any],
                      source_layer: str,
                      expiry_hours: Optional[int] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> MCPContext:
        """Create a new context"""
        context_id = str(uuid.uuid4())
        expiry_time = None
        if expiry_hours:
            expiry_time = datetime.now() + timedelta(hours=expiry_hours)
        elif self.default_expiry_hours:
            expiry_time = datetime.now() + timedelta(hours=self.default_expiry_hours)

        context = MCPContext(
            context_id=context_id,
            context_data=context_data,
            source_layer=source_layer,
            expiry_time=expiry_time,
            metadata=metadata or {}
        )

        self.contexts[context_id] = context
        self._cleanup_expired_contexts()
        
        self.logger.info(f"Created context {context_id} from {source_layer}")
        return context

    def get_context(self, context_id: str) -> Optional[MCPContext]:
        """Get context by ID"""
        context = self.contexts.get(context_id)
        if context and context.is_expired():
            self.logger.warning(f"Context {context_id} has expired")
            return None
        return context

    def update_context(self,
                      context_id: str,
                      new_data: Dict[str, Any],
                      metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update existing context"""
        context = self.get_context(context_id)
        if not context:
            self.logger.error(f"Context {context_id} not found or expired")
            return False

        context.update(new_data, metadata)
        self.logger.info(f"Updated context {context_id}")
        return True

    def delete_context(self, context_id: str) -> bool:
        """Delete context"""
        if context_id in self.contexts:
            context = self.contexts.pop(context_id)
            self.context_history.append(context)
            self.logger.info(f"Deleted context {context_id}")
            return True
        return False

    def get_contexts_by_layer(self, source_layer: str) -> List[MCPContext]:
        """Get all contexts from a specific layer"""
        return [
            context for context in self.contexts.values()
            if context.source_layer == source_layer and not context.is_expired()
        ]

    def get_contexts_by_type(self, context_type: str) -> List[MCPContext]:
        """Get contexts by type (from metadata)"""
        return [
            context for context in self.contexts.values()
            if context.metadata.get("type") == context_type and not context.is_expired()
        ]

    def merge_contexts(self, context_ids: List[str], merge_strategy: str = "union") -> Optional[MCPContext]:
        """Merge multiple contexts"""
        contexts = [self.get_context(cid) for cid in context_ids]
        contexts = [c for c in contexts if c is not None]
        
        if not contexts:
            return None

        if merge_strategy == "union":
            merged_data = {}
            for context in contexts:
                merged_data.update(context.context_data)
        elif merge_strategy == "intersection":
            merged_data = contexts[0].context_data.copy()
            for context in contexts[1:]:
                merged_data = {k: v for k, v in merged_data.items() if k in context.context_data}
        else:
            self.logger.error(f"Unknown merge strategy: {merge_strategy}")
            return None

        # Create new merged context
        merged_metadata = {
            "merged_from": context_ids,
            "merge_strategy": merge_strategy,
            "type": "merged"
        }
        
        return self.create_context(
            context_data=merged_data,
            source_layer="context_manager",
            metadata=merged_metadata
        )

    def enrich_context(self, context_id: str, enrichment_data: Dict[str, Any]) -> bool:
        """Enrich context with additional data"""
        context = self.get_context(context_id)
        if not context:
            return False

        # Add enrichment metadata
        enrichment_metadata = {
            "enriched_at": datetime.now().isoformat(),
            "enrichment_data": enrichment_data
        }
        
        context.update(enrichment_data, enrichment_metadata)
        return True

    def validate_context(self, context_id: str, validation_rules: Dict[str, Any]) -> Dict[str, Any]:
        """Validate context against rules"""
        context = self.get_context(context_id)
        if not context:
            return {"valid": False, "error": "Context not found or expired"}

        validation_result = {"valid": True, "errors": []}
        
        for field, rule in validation_rules.items():
            if field not in context.context_data:
                if rule.get("required", False):
                    validation_result["valid"] = False
                    validation_result["errors"].append(f"Required field '{field}' missing")
            else:
                value = context.context_data[field]
                if "type" in rule:
                    expected_type = rule["type"]
                    if not isinstance(value, expected_type):
                        validation_result["valid"] = False
                        validation_result["errors"].append(f"Field '{field}' should be {expected_type}")

        return validation_result

    def get_context_statistics(self) -> Dict[str, Any]:
        """Get context statistics"""
        total_contexts = len(self.contexts)
        expired_contexts = len([c for c in self.contexts.values() if c.is_expired()])
        active_contexts = total_contexts - expired_contexts
        
        layer_counts = {}
        for context in self.contexts.values():
            layer = context.source_layer
            layer_counts[layer] = layer_counts.get(layer, 0) + 1

        return {
            "total_contexts": total_contexts,
            "active_contexts": active_contexts,
            "expired_contexts": expired_contexts,
            "layer_distribution": layer_counts,
            "history_size": len(self.context_history)
        }

    def _cleanup_expired_contexts(self):
        """Clean up expired contexts"""
        expired_ids = [
            context_id for context_id, context in self.contexts.items()
            if context.is_expired()
        ]
        
        for context_id in expired_ids:
            self.delete_context(context_id)

        # Limit total contexts
        if len(self.contexts) > self.max_contexts:
            # Remove oldest contexts
            sorted_contexts = sorted(
                self.contexts.items(),
                key=lambda x: x[1].created_at
            )
            contexts_to_remove = len(self.contexts) - self.max_contexts
            
            for i in range(contexts_to_remove):
                context_id, _ = sorted_contexts[i]
                self.delete_context(context_id)

    def export_contexts(self) -> Dict[str, Any]:
        """Export all contexts for backup/transfer"""
        return {
            "contexts": {cid: ctx.to_dict() for cid, ctx in self.contexts.items()},
            "history": [ctx.to_dict() for ctx in self.context_history],
            "exported_at": datetime.now().isoformat()
        }

    def import_contexts(self, data: Dict[str, Any]) -> bool:
        """Import contexts from backup/transfer"""
        try:
            # Clear existing contexts
            self.contexts.clear()
            self.context_history.clear()
            
            # Import contexts
            for context_id, context_data in data.get("contexts", {}).items():
                context = MCPContext(
                    context_id=context_data["context_id"],
                    context_data=context_data["context_data"],
                    source_layer=context_data["source_layer"],
                    created_at=datetime.fromisoformat(context_data["created_at"]),
                    updated_at=datetime.fromisoformat(context_data["updated_at"]),
                    expiry_time=datetime.fromisoformat(context_data["expiry_time"]) if context_data["expiry_time"] else None,
                    metadata=context_data["metadata"],
                    version=context_data["version"]
                )
                self.contexts[context_id] = context
            
            # Import history
            for context_data in data.get("history", []):
                context = MCPContext(
                    context_id=context_data["context_id"],
                    context_data=context_data["context_data"],
                    source_layer=context_data["source_layer"],
                    created_at=datetime.fromisoformat(context_data["created_at"]),
                    updated_at=datetime.fromisoformat(context_data["updated_at"]),
                    expiry_time=datetime.fromisoformat(context_data["expiry_time"]) if context_data["expiry_time"] else None,
                    metadata=context_data["metadata"],
                    version=context_data["version"]
                )
                self.context_history.append(context)
            
            self.logger.info(f"Imported {len(data.get('contexts', {}))} contexts and {len(data.get('history', []))} history items")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to import contexts: {e}")
            return False
