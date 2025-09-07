"""
TidyLLM - The Great Walled City of Enterprise AI
===============================================

A lightweight, transparent AI/ML library focused on educational value and user sovereignty.

Core Modules:
- gateways: Enterprise AI processing engines
- knowledge_resource_server: MCP-based knowledge provider
- knowledge_systems: Domain-specific knowledge management

Main Interface:
- TidyLLMInterface: Primary enterprise AI workflow interface
"""

# Core gateway imports
try:
    from . import gateways
    from .gateways import (
        init_gateways,
        get_global_registry,
        AIProcessingGateway,
        CorporateLLMGateway,
        WorkflowOptimizerGateway,
        DatabaseGateway,
        FileStorageGateway
    )
    GATEWAYS_AVAILABLE = True
except ImportError:
    GATEWAYS_AVAILABLE = False

# Knowledge resource server imports
try:
    from . import knowledge_resource_server
    from .knowledge_resource_server import KnowledgeMCPServer
    KNOWLEDGE_SERVER_AVAILABLE = True
except ImportError:
    KNOWLEDGE_SERVER_AVAILABLE = False

# Knowledge systems imports
try:
    from . import knowledge_systems
    KNOWLEDGE_SYSTEMS_AVAILABLE = True
except ImportError:
    KNOWLEDGE_SYSTEMS_AVAILABLE = False

__version__ = "1.0.0"
__all__ = [
    # Gateway exports
    "gateways",
    "init_gateways", 
    "get_global_registry",
    "AIProcessingGateway",
    "CorporateLLMGateway", 
    "WorkflowOptimizerGateway",
    "DatabaseGateway",
    "FileStorageGateway",
    
    # Knowledge server exports
    "knowledge_resource_server",
    "KnowledgeMCPServer",
    
    # Knowledge systems exports
    "knowledge_systems",
    
    # Availability flags
    "GATEWAYS_AVAILABLE",
    "KNOWLEDGE_SERVER_AVAILABLE", 
    "KNOWLEDGE_SYSTEMS_AVAILABLE",
    
    # Main TidyLLM Interface
    "TidyLLMInterface"
]

# Main TidyLLM Interface Import
try:
    import sys
    import os
    from pathlib import Path
    import importlib.util
    
    # Import TidyLLMInterface from the main tidyllm.py file using direct module loading
    root_dir = Path(__file__).parent.parent
    main_script_path = root_dir / "tidyllm.py"
    
    if main_script_path.exists():
        spec = importlib.util.spec_from_file_location("tidyllm_main", main_script_path)
        tidyllm_main = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tidyllm_main)
        
        TidyLLMInterface = tidyllm_main.TidyLLMInterface
        TIDYLLM_INTERFACE_AVAILABLE = True
    else:
        TidyLLMInterface = None
        TIDYLLM_INTERFACE_AVAILABLE = False
        
except Exception as e:
    TidyLLMInterface = None
    TIDYLLM_INTERFACE_AVAILABLE = False