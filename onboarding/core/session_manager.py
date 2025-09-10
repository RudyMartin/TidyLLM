"""
TidyLLM Onboarding Session Manager
=================================

Unified session management for the onboarding system.
Handles TidyLLM component initialization and session state.
"""

import streamlit as st
from typing import Optional
import sys
from pathlib import Path

# Add parent directory to Python path for tidyllm imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class SessionManager:
    """Unified session manager for TidyLLM onboarding system."""
    
    _instance: Optional[object] = None
    _initialized: bool = False
    
    @classmethod
    def get_instance(cls):
        """Get singleton UnifiedSessionManager instance for Streamlit."""
        
        # Check if we already have it in Streamlit's global session state
        if hasattr(st, 'session_state') and hasattr(st.session_state, '_unified_session_manager'):
            if st.session_state._unified_session_manager is not None:
                return st.session_state._unified_session_manager
        
        # Try to create new instance if imports available
        try:
            from tidyllm.infrastructure.session.unified import UnifiedSessionManager
            
            if cls._instance is None:
                cls._instance = UnifiedSessionManager()
                cls._initialized = True
                
            # Store in Streamlit session state for persistence
            if hasattr(st, 'session_state'):
                st.session_state._unified_session_manager = cls._instance
                
            return cls._instance
            
        except ImportError as e:
            st.error(f"TidyLLM imports not available: {e}")
            return None
        except Exception as e:
            st.error(f"Failed to initialize UnifiedSessionManager: {e}")
            return None
    
    @classmethod
    def get_gateways(cls):
        """Get all TidyLLM CORE GATEWAYS - REQUIRES AWS CONNECTION."""
        try:
            from tidyllm.gateways.corporate_llm_gateway import CorporateLLMGateway
            from tidyllm.gateways.ai_processing_gateway import AIProcessingGateway
            from tidyllm.gateways.workflow_optimizer_gateway import WorkflowOptimizerGateway
            
            session_manager = cls.get_instance()
            
            # CRITICAL: AWS connection is required for gateways
            if not session_manager:
                st.error("❌ AWS connection required for gateways - configure connections first!")
                return {}
            
            # Verify AWS services are available
            if not session_manager.get_s3_client():
                st.error("❌ S3 connection required for gateways")
                return {}
            
            if not session_manager.get_bedrock_client():
                st.error("❌ Bedrock connection required for gateways")
                return {}
            
            gateways = {}
            
            # Initialize CORE GATEWAYS with proper error handling
            try:
                gateway = CorporateLLMGateway()
                gateway.session_manager = session_manager
                gateways['corporate_llm'] = gateway
                st.success("✅ CorporateLLMGateway initialized")
            except Exception as e:
                st.error(f"❌ CorporateLLMGateway failed: {e}")
                gateways['corporate_llm'] = None
            
            try:
                gateway = AIProcessingGateway()
                gateway.session_manager = session_manager
                gateways['ai_processing'] = gateway
                st.success("✅ AIProcessingGateway initialized")
            except Exception as e:
                st.error(f"❌ AIProcessingGateway failed: {e}")
                gateways['ai_processing'] = None
            
            try:
                gateway = WorkflowOptimizerGateway()
                gateway.session_manager = session_manager
                gateways['workflow_optimizer'] = gateway
                st.success("✅ WorkflowOptimizerGateway initialized")
            except Exception as e:
                st.error(f"❌ WorkflowOptimizerGateway failed: {e}")
                gateways['workflow_optimizer'] = None
            
            # Initialize Knowledge MCP Server (4th core gateway)
            try:
                from tidyllm.knowledge_resource_server.mcp_server import KnowledgeMCPServer
                gateway = KnowledgeMCPServer()
                # Note: KnowledgeMCPServer doesn't need session_manager like other gateways
                gateways['knowledge_resources'] = gateway
                st.success("✅ KnowledgeMCPServer initialized")
            except Exception as e:
                st.error(f"❌ KnowledgeMCPServer failed: {e}")
                gateways['knowledge_resources'] = None
            
            return gateways
            
        except Exception as e:
            st.error(f"❌ Gateway initialization failed: {e}")
            return {}
    
    @classmethod
    def get_services(cls):
        """Get TidyLLM utility services (not gateways)."""
        try:
            from tidyllm.gateways.database_gateway import DatabaseGateway, DatabaseGatewayConfig
            from tidyllm.gateways.file_storage_gateway import FileStorageGateway, FileStorageConfig
            from tidyllm.gateways.mvr_gateway import MVRAnalysisGateway
            
            session_manager = cls.get_instance()
            
            if not session_manager:
                st.error("❌ AWS connection required for services - configure connections first!")
                return {}
            
            services = {}
            
            # Initialize Database Utility Service
            try:
                config = DatabaseGatewayConfig()
                service = DatabaseGateway(config)
                service.session_manager = session_manager
                services['database'] = service
                st.success("✅ Database Service initialized")
            except Exception as e:
                st.error(f"❌ Database Service failed: {e}")
                services['database'] = None
            
            # Initialize File Storage Utility Service
            try:
                config = FileStorageConfig()
                service = FileStorageGateway(config)
                service.session_manager = session_manager
                # Set S3 client from USM
                if session_manager.get_s3_client():
                    service.set_s3_client(session_manager.get_s3_client())
                services['file_storage'] = service
                st.success("✅ File Storage Service initialized")
            except Exception as e:
                st.error(f"❌ File Storage Service failed: {e}")
                services['file_storage'] = None
            
            # Initialize MVR Document Service
            try:
                service = MVRAnalysisGateway()
                service.session_manager = session_manager
                services['mvr_document'] = service
                st.success("✅ MVR Document Service initialized")
            except Exception as e:
                st.error(f"❌ MVR Document Service failed: {e}")
                services['mvr_document'] = None
            
            return services
            
        except Exception as e:
            st.error(f"❌ Service initialization failed: {e}")
            return {}
    
    @classmethod
    def get_knowledge_systems(cls):
        """Get TidyLLM knowledge systems."""
        try:
            from tidyllm.knowledge_systems.core.domain_rag import DomainRAG
            from tidyllm.flow.examples.bracket_registry import BracketRegistry
            
            return {
                'domain_rag': DomainRAG,
                'bracket_registry': BracketRegistry
            }
        except Exception as e:
            st.error(f"Failed to initialize knowledge systems: {e}")
            return {}

def get_session_manager():
    """Get the session manager instance."""
    return SessionManager.get_instance()

def init_streamlit_session_state():
    """Initialize Streamlit session state for onboarding."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.session_manager = SessionManager.get_instance()
        st.session_state.knowledge_systems = SessionManager.get_knowledge_systems()
    
    # ALWAYS refresh gateways and services on every page load to get latest status
    st.session_state.gateways = SessionManager.get_gateways()
    st.session_state.services = SessionManager.get_services()
