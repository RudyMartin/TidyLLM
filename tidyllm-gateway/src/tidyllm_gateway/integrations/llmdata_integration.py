#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLMData Integration - Unified Gateway with Backward Compatibility

Provides enterprise governance to existing LLMData verbs and macro system
while maintaining full backward compatibility. Existing code continues
to work unchanged while gaining enterprise features.

Architecture:
Existing Macros → LLMData Verbs → TidyLLM Gateway → MLFlow → Providers
                               (Enterprise Layer Added Here)
"""

import logging
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime
import inspect
import functools

# TidyLLM Gateway imports
from ..litellm_clone import completion as gateway_completion, embedding as gateway_embedding
from ..core.provider_registry import get_provider_registry
from ..enterprise.spend_tracking import EnterpriseSpendTracker

# Try to import existing LLMData components
try:
    from tidyllm.core import LLMMessage, Provider
    from tidyllm.gateway import get_default_gateway as get_original_gateway
    LLMDATA_AVAILABLE = True
except ImportError:
    LLMDATA_AVAILABLE = False
    # Create placeholder classes for compatibility
    class LLMMessage:
        def __init__(self, content: str, *files, **kwargs):
            self.content = content
            self.files = files
            self.metadata = kwargs
    
    class Provider:
        def __init__(self, provider: str, model: str = None, **kwargs):
            self.provider = provider
            self.model = model
            self.config = kwargs

logger = logging.getLogger(__name__)


class EnterpriseContext:
    """Context for enterprise governance in LLMData workflows"""
    
    def __init__(self):
        self.current_user = "llmdata_system"
        self.current_department = "data_processing" 
        self.current_audit_reason = "LLMData workflow execution"
        self.spend_tracker = EnterpriseSpendTracker()
        self.workflow_id = None
        self.macro_name = None
    
    def set_context(
        self, 
        user_id: str = None,
        department: str = None, 
        audit_reason: str = None,
        workflow_id: str = None,
        macro_name: str = None
    ):
        """Set enterprise context for LLMData operations"""
        if user_id:
            self.current_user = user_id
        if department:
            self.current_department = department
        if audit_reason:
            self.current_audit_reason = audit_reason
        if workflow_id:
            self.workflow_id = workflow_id
        if macro_name:
            self.macro_name = macro_name
    
    def get_audit_reason(self) -> str:
        """Get contextual audit reason"""
        base_reason = self.current_audit_reason
        
        if self.macro_name:
            base_reason = f"{self.macro_name} macro execution"
        if self.workflow_id:
            base_reason = f"{base_reason} (workflow: {self.workflow_id})"
            
        return base_reason


# Global enterprise context
_enterprise_context = EnterpriseContext()


def with_enterprise_governance(llmdata_func):
    """Decorator to add enterprise governance to LLMData functions"""
    
    @functools.wraps(llmdata_func)
    def wrapper(*args, **kwargs):
        # Extract enterprise parameters if provided
        user_id = kwargs.pop('user_id', _enterprise_context.current_user)
        department = kwargs.pop('department', _enterprise_context.current_department) 
        audit_reason = kwargs.pop('audit_reason', _enterprise_context.get_audit_reason())
        max_cost_usd = kwargs.pop('max_cost_usd', None)
        
        # Execute original function with enterprise context
        try:
            result = llmdata_func(*args, **kwargs)
            
            # If result contains LLM response, add enterprise metadata
            if isinstance(result, dict) and 'choices' in result:
                result['enterprise'] = {
                    'user_id': user_id,
                    'department': department,
                    'audit_reason': audit_reason,
                    'timestamp': datetime.utcnow().isoformat(),
                    'governed': True
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Enterprise-governed LLMData function failed: {e}")
            raise
    
    return wrapper


def chat(provider: Provider) -> Callable:
    """
    Enhanced chat verb with enterprise governance
    
    Backward compatible with existing LLMData chat() while adding
    enterprise controls, audit trails, and cost tracking.
    
    Usage (unchanged from LLMData):
        response = llm_message("Hello") | chat(claude())
        
    New enterprise features (optional):
        response = llm_message("Hello") | chat(
            claude(),
            user_id="analyst@company.com",
            audit_reason="Customer analysis"
        )
    """
    
    def chat_executor(message: LLMMessage, **enterprise_kwargs) -> Dict[str, Any]:
        """Execute chat through enterprise gateway"""
        
        # Extract enterprise context
        user_id = enterprise_kwargs.get('user_id', _enterprise_context.current_user)
        department = enterprise_kwargs.get('department', _enterprise_context.current_department)
        audit_reason = enterprise_kwargs.get('audit_reason', _enterprise_context.get_audit_reason())
        max_cost_usd = enterprise_kwargs.get('max_cost_usd')
        fallbacks = enterprise_kwargs.get('fallbacks')
        
        # Convert LLMMessage to messages format
        messages = [{"role": "user", "content": message.content}]
        
        # Add file attachments if present
        if hasattr(message, 'files') and message.files:
            # Add file information to message content
            file_info = f"\n\nAttached files: {', '.join(str(f) for f in message.files)}"
            messages[0]["content"] += file_info
        
        # Map provider to model
        model = _map_provider_to_model(provider)
        
        # Execute through enterprise gateway
        try:
            response = gateway_completion(
                model=model,
                messages=messages,
                user_id=user_id,
                audit_reason=audit_reason,
                department=department,
                max_cost_usd=max_cost_usd,
                fallbacks=fallbacks,
                **{k: v for k, v in enterprise_kwargs.items() 
                   if k not in ['user_id', 'department', 'audit_reason', 'max_cost_usd', 'fallbacks']}
            )
            
            # Add enterprise metadata
            response['llmdata_compatible'] = True
            response['enterprise_governed'] = True
            
            return response
            
        except Exception as e:
            logger.error(f"Enterprise chat execution failed: {e}")
            
            # Fallback to original LLMData gateway if available
            if LLMDATA_AVAILABLE:
                logger.info("Falling back to original LLMData gateway")
                original_gateway = get_original_gateway()
                return original_gateway.query("chat", {
                    "messages": messages,
                    "model": model
                })
            else:
                raise e
    
    return chat_executor


def embed(provider: Provider) -> Callable:
    """
    Enhanced embedding verb with enterprise governance
    
    Backward compatible with existing LLMData embed() while adding
    enterprise controls and audit trails.
    """
    
    def embed_executor(message: LLMMessage, **enterprise_kwargs) -> Dict[str, Any]:
        """Execute embedding through enterprise gateway"""
        
        user_id = enterprise_kwargs.get('user_id', _enterprise_context.current_user)
        department = enterprise_kwargs.get('department', _enterprise_context.current_department)
        audit_reason = enterprise_kwargs.get('audit_reason', _enterprise_context.get_audit_reason())
        
        # Map provider to embedding model
        model = _map_provider_to_embedding_model(provider)
        
        try:
            response = gateway_embedding(
                model=model,
                input=message.content,
                user_id=user_id,
                audit_reason=audit_reason,
                department=department
            )
            
            response['llmdata_compatible'] = True
            response['enterprise_governed'] = True
            
            return response
            
        except Exception as e:
            logger.error(f"Enterprise embedding execution failed: {e}")
            raise e
    
    return embed_executor


def analyze_data(data_source: str, provider: Provider) -> Callable:
    """
    Enhanced data analysis verb with enterprise governance
    
    Analyzes data through enterprise gateway with audit trails
    """
    
    def analyze_executor(message: LLMMessage, **enterprise_kwargs) -> Dict[str, Any]:
        """Execute data analysis through enterprise gateway"""
        
        user_id = enterprise_kwargs.get('user_id', _enterprise_context.current_user)
        audit_reason = f"Data analysis of {data_source} - {_enterprise_context.get_audit_reason()}"
        
        # Enhance message with data context
        analysis_prompt = f"""
        {message.content}
        
        Data Source: {data_source}
        Analysis Request: Provide comprehensive analysis including:
        1. Key insights and patterns
        2. Statistical summary
        3. Recommendations
        4. Risk factors or concerns
        """
        
        enhanced_message = LLMMessage(analysis_prompt)
        
        # Route through chat with enhanced audit context
        return chat(provider)(enhanced_message, 
                             user_id=user_id, 
                             audit_reason=audit_reason,
                             **enterprise_kwargs)
    
    return analyze_executor


def send_batch(messages: List[LLMMessage], provider: Provider, **enterprise_kwargs) -> List[Dict[str, Any]]:
    """
    Enhanced batch processing with enterprise governance
    
    Processes multiple messages with budget controls and audit trails
    """
    
    user_id = enterprise_kwargs.get('user_id', _enterprise_context.current_user)
    department = enterprise_kwargs.get('department', _enterprise_context.current_department)
    max_batch_cost = enterprise_kwargs.get('max_batch_cost_usd', 100.0)
    
    # Create batch budget
    batch_id = f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    
    _enterprise_context.spend_tracker.create_budget(
        name=f"Batch Processing {len(messages)} messages",
        limit_usd=max_batch_cost,
        user_id=user_id,
        department=department,
        hard_limit=True
    )
    
    results = []
    total_cost = 0.0
    
    for i, message in enumerate(messages):
        try:
            # Check budget before each request
            budget_check = _enterprise_context.spend_tracker.check_budget_approval(
                user_id=user_id,
                department=department,
                tenant_id="llmdata",
                model=_map_provider_to_model(provider),
                provider=provider.provider,
                estimated_cost_usd=1.0  # Conservative estimate
            )
            
            if not budget_check["approved"]:
                results.append({
                    "error": f"Batch budget exceeded: {budget_check['reason']}",
                    "message_index": i,
                    "enterprise_governed": True
                })
                break
            
            # Process message
            audit_reason = f"Batch processing item {i+1}/{len(messages)} - {batch_id}"
            
            response = chat(provider)(message, 
                                    user_id=user_id,
                                    department=department,
                                    audit_reason=audit_reason)
            
            response["message_index"] = i
            response["batch_id"] = batch_id
            results.append(response)
            
            # Track cost
            cost = response.get("cost_usd", 0.0)
            total_cost += cost
            
        except Exception as e:
            results.append({
                "error": str(e),
                "message_index": i,
                "batch_id": batch_id,
                "enterprise_governed": True
            })
    
    return results


def _map_provider_to_model(provider: Provider) -> str:
    """Map LLMData Provider to TidyLLM Gateway model"""
    
    # Get provider registry for mappings
    registry = get_provider_registry()
    
    # Provider mapping logic
    if provider.provider == "claude":
        if provider.model:
            return provider.model
        return "claude-3-5-sonnet"  # Default Claude model
    elif provider.provider == "openai":
        if provider.model:
            return provider.model
        return "gpt-4"  # Default OpenAI model
    elif provider.provider == "ollama":
        if provider.model:
            return provider.model
        return "llama-3.1-70b"  # Default local model
    else:
        # Try to find model in registry
        approved_models = registry.get_approved_models()
        if approved_models:
            return approved_models[0].model_id
        return "gpt-4"  # Final fallback


def _map_provider_to_embedding_model(provider: Provider) -> str:
    """Map LLMData Provider to embedding model"""
    
    if provider.provider == "openai":
        return "text-embedding-3-large"
    elif provider.provider == "claude":
        return "text-embedding-3-large"  # Route through OpenAI for embeddings
    else:
        return "text-embedding-3-large"  # Default embedding model


# Provider convenience functions (backward compatible)
def claude(model: str = "claude-3-5-sonnet", **kwargs) -> Provider:
    """Create Claude provider (enhanced with enterprise routing)"""
    return Provider("claude", model, **kwargs)


def openai(model: str = "gpt-4", **kwargs) -> Provider:
    """Create OpenAI provider (enhanced with enterprise routing)"""  
    return Provider("openai", model, **kwargs)


def ollama(model: str = "llama-3.1-70b", **kwargs) -> Provider:
    """Create Ollama provider (enhanced with enterprise routing)"""
    return Provider("ollama", model, **kwargs)


def llm_message(content: str, *files, **kwargs) -> LLMMessage:
    """
    Create LLM message (fully backward compatible)
    
    Enhanced with enterprise metadata support
    """
    return LLMMessage(content, *files, **kwargs)


# Context management functions
def set_enterprise_context(
    user_id: str = None,
    department: str = None,
    audit_reason: str = None,
    workflow_id: str = None,
    macro_name: str = None
):
    """
    Set enterprise context for all LLMData operations
    
    Usage:
        set_enterprise_context(
            user_id="analyst@company.com",
            department="risk-management", 
            audit_reason="Q4 model validation",
            macro_name="mvr_peer_review"
        )
        
        # All subsequent LLMData calls use this context
        response = llm_message("Analyze model") | chat(claude())
    """
    _enterprise_context.set_context(
        user_id=user_id,
        department=department,
        audit_reason=audit_reason,
        workflow_id=workflow_id,
        macro_name=macro_name
    )
    
    logger.info(f"Enterprise context set: {user_id} / {department} / {macro_name}")


def get_enterprise_context() -> Dict[str, Any]:
    """Get current enterprise context"""
    return {
        "user_id": _enterprise_context.current_user,
        "department": _enterprise_context.current_department,
        "audit_reason": _enterprise_context.current_audit_reason,
        "workflow_id": _enterprise_context.workflow_id,
        "macro_name": _enterprise_context.macro_name
    }


def clear_enterprise_context():
    """Clear enterprise context (reset to defaults)"""
    global _enterprise_context
    _enterprise_context = EnterpriseContext()


# Backward compatibility - monkey patch existing LLMData if available
if LLMDATA_AVAILABLE:
    try:
        import tidyllm.verbs as original_verbs
        
        # Replace original verbs with enterprise-enhanced versions
        original_verbs.chat = chat
        original_verbs.embed = embed
        original_verbs.analyze_data = analyze_data
        original_verbs.send_batch = send_batch
        
        # Add enterprise context functions
        original_verbs.set_enterprise_context = set_enterprise_context
        original_verbs.get_enterprise_context = get_enterprise_context
        
        logger.info("✅ Enhanced existing LLMData verbs with enterprise governance")
        
    except Exception as e:
        logger.warning(f"Could not enhance existing LLMData verbs: {e}")


# Export all functions for direct import
__all__ = [
    # Core verbs (enhanced)
    'chat', 'embed', 'analyze_data', 'send_batch',
    
    # Provider functions (enhanced)
    'claude', 'openai', 'ollama',
    
    # Message creation (enhanced)
    'llm_message',
    
    # Enterprise context management
    'set_enterprise_context',
    'get_enterprise_context', 
    'clear_enterprise_context',
    
    # Data classes (for compatibility)
    'LLMMessage', 'Provider'
]