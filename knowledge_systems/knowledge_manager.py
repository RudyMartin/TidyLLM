"""
Knowledge Manager
=================

Central orchestrator for all knowledge systems in TidyLLM:
- Manages multiple domain RAG systems
- Coordinates S3 and vector database operations
- Provides unified interface for knowledge operations
- Handles Flow Agreement integration

This is the main entry point for all knowledge management operations.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict

# S3 Configuration Management
sys.path.append(str(Path(__file__).parent.parent.parent / 'admin'))
try:
    from credential_loader import get_s3_config, build_s3_path
    s3_config = get_s3_config()
except ImportError:
    s3_config = None
    def build_s3_path(*args): return '/'.join(args)

try:
    from tidyllm.infrastructure.session import S3Manager, S3Config
    from tidyllm.knowledge_systems.vector_manager import VectorManager, VectorConfig
    from tidyllm.knowledge_systems.domain_rag import DomainRAG, DomainRAGConfig, RAGQuery, RAGResponse
except ImportError:
    # Fallback for test scenarios - create minimal classes
    class S3Manager:
        def __init__(self, config=None): pass
        def get_status(self): return {"status": "mock"}
        def test_connection(self): return {"success": False, "error": "Mock S3Manager"}
        def upload_file(self, *args, **kwargs): return type('obj', (object,), {"success": False, "error": "Mock upload"})()

    class S3Config: pass
    class VectorManager:
        def __init__(self, config=None): pass
        def connect(self): return {"success": False, "error": "Mock VectorManager"}
        def get_database_stats(self): return {"status": "mock"}
        def check_extensions(self): return {"success": False}
        def close(self): pass

    class VectorConfig: pass
    class DomainRAG:
        def __init__(self, *args, **kwargs): pass
        def get_stats(self): return {"status": "mock"}
        def export_config(self): return {"mock": True}

    class DomainRAGConfig: pass
    class RAGQuery: pass
    class RAGResponse: pass

logger = logging.getLogger("knowledge_manager")

@dataclass
class KnowledgeSystemConfig:
    """Configuration for the entire knowledge system"""
    s3_config: Optional[S3Config] = None
    vector_config: Optional[VectorConfig] = None
    domain_configs: Optional[Dict[str, DomainRAGConfig]] = None
    auto_setup: bool = True

class KnowledgeManager:
    """Central manager for all knowledge systems"""
    
    def __init__(self, config: KnowledgeSystemConfig = None):
        self.config = config or KnowledgeSystemConfig()
        
        # Initialize core managers
        self.s3_manager = None
        self.vector_manager = None
        self.domain_rags: Dict[str, DomainRAG] = {}
        
        # Initialize if auto_setup is enabled
        if self.config.auto_setup:
            self._setup_managers()
    
    def _setup_managers(self):
        """Setup core managers with configuration"""
        try:
            # Initialize S3 manager
            self.s3_manager = S3Manager(self.config.s3_config)
            logger.info(f"S3 Manager initialized: {self.s3_manager.get_status()}")
            
            # Initialize Vector manager
            self.vector_manager = VectorManager(self.config.vector_config)
            logger.info("Vector Manager initialized")
            
            # Initialize domain RAGs if configured
            if self.config.domain_configs:
                for domain_name, domain_config in self.config.domain_configs.items():
                    self.add_domain_rag(domain_name, domain_config)
            
        except Exception as e:
            logger.error(f"Failed to setup managers: {e}")
    
    def add_domain_rag(self, domain_name: str, config: DomainRAGConfig) -> DomainRAG:
        """Add a new domain RAG system"""
        domain_rag = DomainRAG(
            config=config,
            s3_manager=self.s3_manager,
            vector_manager=self.vector_manager
        )
        
        self.domain_rags[domain_name] = domain_rag
        logger.info(f"Added domain RAG: {domain_name}")
        
        return domain_rag
    
    def get_domain_rag(self, domain_name: str) -> Optional[DomainRAG]:
        """Get domain RAG by name"""
        return self.domain_rags.get(domain_name)
    
    def list_domains(self) -> List[str]:
        """List all available domain names"""
        return list(self.domain_rags.keys())
    
    def create_model_validation_rag(self, knowledge_base_path: Union[str, Path]) -> DomainRAG:
        """Create Model Validation domain RAG from knowledge base PDFs"""
        knowledge_base_path = Path(knowledge_base_path)
        
        if not knowledge_base_path.exists():
            raise ValueError(f"Knowledge base path not found: {knowledge_base_path}")
        
        # Configure domain RAG for model validation
        config = DomainRAGConfig(
            domain_name="model_validation",
            description="Model validation and risk management knowledge base",
            s3_bucket=self.s3_manager.config.default_bucket if self.s3_manager else None,
            s3_prefix = build_s3_path("knowledge_base", "model_validation/"),
            metadata_schema={
                "regulation_type": {"pattern": r"(Basel|CCAR|DFAST|SR11-7)", "description": "Financial regulation type"},
                "risk_category": {"pattern": r"(credit|market|operational|model)", "description": "Risk category"},
                "validation_stage": {"pattern": r"(development|validation|implementation|monitoring)", "description": "Model lifecycle stage"}
            }
        )
        
        # Add to system
        domain_rag = self.add_domain_rag("model_validation", config)
        
        # Process knowledge base documents
        logger.info(f"Processing knowledge base from: {knowledge_base_path}")
        results = domain_rag.process_directory(
            knowledge_base_path,
            file_patterns=["*.pdf", "*.txt", "*.md", "*.docx"]
        )
        
        successful_docs = [r for r in results if r.success]
        failed_docs = [r for r in results if not r.success]
        
        logger.info(f"Model Validation RAG created:")
        logger.info(f"  ✅ Successfully processed: {len(successful_docs)} documents")
        logger.info(f"  ❌ Failed: {len(failed_docs)} documents")
        logger.info(f"  📊 Total chunks: {sum(r.chunks_created for r in successful_docs)}")
        
        if failed_docs:
            logger.warning("Failed documents:")
            for doc in failed_docs:
                logger.warning(f"  - {doc.source_file}: {doc.error}")
        
        return domain_rag
    
    def query_domain(self, domain_name: str, query: str, **kwargs) -> RAGResponse:
        """Query a specific domain RAG"""
        domain_rag = self.get_domain_rag(domain_name)
        if not domain_rag:
            raise ValueError(f"Domain RAG not found: {domain_name}")
        
        rag_query = RAGQuery(query=query, **kwargs)
        return domain_rag.query(rag_query)
    
    def query_all_domains(self, query: str, **kwargs) -> Dict[str, RAGResponse]:
        """Query all available domain RAGs"""
        results = {}
        
        for domain_name, domain_rag in self.domain_rags.items():
            try:
                rag_query = RAGQuery(query=query, **kwargs)
                response = domain_rag.query(rag_query)
                results[domain_name] = response
            except Exception as e:
                logger.error(f"Failed to query domain {domain_name}: {e}")
                results[domain_name] = RAGResponse(
                    query=query,
                    answer=f"Error querying {domain_name}: {str(e)}",
                    sources=[],
                    confidence=0.0,
                    processing_time=0.0,
                    metadata={"error": str(e)}
                )
        
        return results
    
    def get_best_answer(self, query: str, **kwargs) -> RAGResponse:
        """Get the best answer across all domains"""
        all_responses = self.query_all_domains(query, **kwargs)
        
        if not all_responses:
            return RAGResponse(
                query=query,
                answer="No domain RAG systems available to answer this query.",
                sources=[],
                confidence=0.0,
                processing_time=0.0,
                metadata={"error": "No domains available"}
            )
        
        # Find response with highest confidence
        best_response = max(all_responses.values(), key=lambda r: r.confidence)
        
        # Add metadata about other domain responses
        best_response.metadata["all_domain_scores"] = {
            domain: response.confidence 
            for domain, response in all_responses.items()
        }
        best_response.metadata["best_domain"] = max(
            all_responses.keys(), 
            key=lambda d: all_responses[d].confidence
        )
        
        return best_response
    
    def upload_document_to_s3(self, file_path: Union[str, Path], 
                             s3_key: str = None, metadata: Dict[str, str] = None) -> Dict[str, Any]:
        """Upload document to S3"""
        if not self.s3_manager:
            return {"success": False, "error": "S3 manager not initialized"}
        
        result = self.s3_manager.upload_file(file_path, s3_key=s3_key, metadata=metadata)
        return {
            "success": result.success,
            "s3_url": result.s3_url,
            "error": result.error,
            "metadata": result.metadata
        }
    
    def test_connections(self) -> Dict[str, Any]:
        """Test all system connections"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "s3_connection": None,
            "vector_connection": None,
            "domain_rags": {}
        }
        
        # Test S3 connection
        if self.s3_manager:
            results["s3_connection"] = self.s3_manager.test_connection()
        else:
            results["s3_connection"] = {"success": False, "error": "S3 manager not initialized"}
        
        # Test vector connection
        if self.vector_manager:
            connect_result = self.vector_manager.connect()
            results["vector_connection"] = connect_result
            
            if connect_result["success"]:
                # Test extensions
                ext_result = self.vector_manager.check_extensions()
                results["vector_connection"]["extensions"] = ext_result
        else:
            results["vector_connection"] = {"success": False, "error": "Vector manager not initialized"}
        
        # Test domain RAGs
        for domain_name, domain_rag in self.domain_rags.items():
            results["domain_rags"][domain_name] = domain_rag.get_stats()
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "managers": {
                "s3_manager": self.s3_manager.get_status() if self.s3_manager else None,
                "vector_manager": self.vector_manager.get_database_stats() if self.vector_manager else None
            },
            "domain_rags": {
                domain_name: domain_rag.get_stats()
                for domain_name, domain_rag in self.domain_rags.items()
            },
            "summary": {
                "total_domains": len(self.domain_rags),
                "s3_available": bool(self.s3_manager),
                "vector_available": bool(self.vector_manager)
            }
        }
        
        return status
    
    def create_flow_agreement(self, domain_name: str) -> Dict[str, Any]:
        """Create Flow Agreement configuration for domain RAG"""
        domain_rag = self.get_domain_rag(domain_name)
        if not domain_rag:
            raise ValueError(f"Domain RAG not found: {domain_name}")
        
        stats = domain_rag.get_stats()
        
        flow_agreement = {
            "flow_type": "domain_rag_query",
            "domain": domain_name,
            "description": domain_rag.config.description,
            "version": "1.0",
            "operations": [
                {
                    "name": "query_knowledge_base",
                    "type": "rag_query",
                    "parameters": {
                        "domain": domain_name,
                        "max_results": 5,
                        "similarity_threshold": 0.7
                    }
                },
                {
                    "name": "generate_answer",
                    "type": "llm_generation",
                    "depends_on": ["query_knowledge_base"]
                }
            ],
            "metadata": {
                "documents_available": stats.get("documents_processed", 0),
                "total_chunks": stats.get("total_chunks", 0),
                "last_updated": stats.get("last_updated"),
                "s3_location": domain_rag.config.s3_prefix
            },
            "integration": {
                "chat_interface": True,
                "bracket_command": f"[{domain_name}_rag]",
                "mvr_analysis_compatible": domain_name == "model_validation"
            }
        }
        
        return flow_agreement
    
    def export_system_config(self) -> Dict[str, Any]:
        """Export complete system configuration"""
        return {
            "knowledge_system_config": asdict(self.config),
            "domain_configs": {
                domain_name: domain_rag.export_config()
                for domain_name, domain_rag in self.domain_rags.items()
            },
            "system_status": self.get_system_status()
        }
    
    def close(self):
        """Close all connections and cleanup"""
        if self.vector_manager:
            self.vector_manager.close()
        
        # Reset global instances
        try:
            from tidyllm.infrastructure.session import reset_s3_manager
            from tidyllm.knowledge_systems.vector_manager import reset_vector_manager
        except ImportError:
            def reset_s3_manager(): pass
            def reset_vector_manager(): pass
        reset_s3_manager()
        reset_vector_manager()

# Global instance for easy access
_knowledge_manager_instance = None

def get_knowledge_manager(config: KnowledgeSystemConfig = None) -> KnowledgeManager:
    """Get global knowledge manager instance"""
    global _knowledge_manager_instance
    
    if _knowledge_manager_instance is None or config is not None:
        _knowledge_manager_instance = KnowledgeManager(config)
    
    return _knowledge_manager_instance

def reset_knowledge_manager():
    """Reset global knowledge manager instance"""
    global _knowledge_manager_instance
    if _knowledge_manager_instance:
        _knowledge_manager_instance.close()
    _knowledge_manager_instance = None