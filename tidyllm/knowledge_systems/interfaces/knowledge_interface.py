"""

import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# S3 Configuration Management - Import from admin
try:
    from ...admin.credential_loader import get_s3_config, build_s3_path
except ImportError:
    # Fallback import path
    import sys
    from pathlib import Path
    admin_path = Path(__file__).parent.parent.parent / 'admin'
    sys.path.insert(0, str(admin_path))
    from credential_loader import get_s3_config, build_s3_path

# Get S3 configuration (bucket and path builder)
s3_config = get_s3_config()  # Add environment parameter for dev/staging/prod

Knowledge Interface
===================

Unified interface for all knowledge operations in TidyLLM.
Provides simple, consistent API that abstracts the complexity of:
- S3 document storage
- Vector database operations  
- Domain RAG systems
- Flow Agreement integration

This is the main interface used by chat interfaces, workflows, and applications.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from ..core.knowledge_manager import KnowledgeManager, KnowledgeSystemConfig
from ..core.domain_rag import RAGQuery, RAGResponse

logger = logging.getLogger("knowledge_interface")

class KnowledgeInterface:
    """
    Simplified interface for knowledge operations.
    
    This class provides the main API that applications should use for:
    - Creating and managing domain RAG systems
    - Querying knowledge bases
    - Processing documents
    - Integration with chat and workflow systems
    """
    
    def __init__(self, config: KnowledgeSystemConfig = None):
        self.manager = KnowledgeManager(config)
        logger.info("Knowledge Interface initialized")
    
    # Domain RAG Management
    
    def create_domain_rag(self, domain_name: str, knowledge_base_path: Union[str, Path] = None,
                         s3_bucket: str = None, s3_prefix: str = None,
                         description: str = "") -> Dict[str, Any]:
        """
        Create a new domain RAG system from documents (local directory or S3).
        
        Args:
            domain_name: Unique name for the domain (e.g., "model_validation")
            knowledge_base_path: Path to local directory containing documents (optional)
            s3_bucket: S3 bucket containing documents (optional)
            s3_prefix: S3 prefix/folder path (optional)
            description: Human-readable description of the domain
            
        Returns:
            Dict with creation results and statistics
        """
        try:
            # Determine source: local directory or S3
            if knowledge_base_path:
                # Local directory processing
                if domain_name == "model_validation":
                    domain_rag = self.manager.create_model_validation_rag(knowledge_base_path)
                else:
                    # Generic domain RAG creation from local directory
                    from ..core.domain_rag import DomainRAGConfig
                    config = DomainRAGConfig(
                        domain_name=domain_name,
                        description=description,
                        s3_prefix=build_s3_path("knowledge_base", f"{domain_name}/"),
                        s3_bucket=self.manager.s3_manager.config.default_bucket if self.manager.s3_manager else None
                    )
                    
                    domain_rag = self.manager.add_domain_rag(domain_name, config)
                    results = domain_rag.process_directory(knowledge_base_path)
                    
                    successful_docs = [r for r in results if r.success]
                    failed_docs = [r for r in results if not r.success]
                    
                    logger.info(f"Domain RAG '{domain_name}' created from local directory:")
                    logger.info(f"  Successfully processed: {len(successful_docs)} documents")
                    logger.info(f"  Failed: {len(failed_docs)} documents")
                    
            elif s3_bucket and s3_prefix:
                # S3-based processing
                domain_rag = self._create_s3_domain_rag(domain_name, s3_bucket, s3_prefix, description)
                
            else:
                raise ValueError("Must provide either knowledge_base_path or s3_bucket+s3_prefix")
            
            stats = domain_rag.get_stats()
            
            return {
                "success": True,
                "domain_name": domain_name,
                "description": description,
                "source": "local" if knowledge_base_path else "s3",
                "stats": stats,
                "flow_agreement": self.manager.create_flow_agreement(domain_name)
            }
            
        except Exception as e:
            logger.error(f"Failed to create domain RAG '{domain_name}': {e}")
            return {
                "success": False,
                "domain_name": domain_name,
                "error": str(e)
            }
    
    def _create_s3_domain_rag(self, domain_name: str, s3_bucket: str, s3_prefix: str, 
                             description: str) -> "DomainRAG":
        """Create domain RAG from S3-stored documents"""
        import tempfile
        import shutil
        from ..core.domain_rag import DomainRAGConfig
        
        # Configure S3-based domain RAG
        config = DomainRAGConfig(
            domain_name=domain_name,
            description=description,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            processing_config={
                "source": "s3",
                "download_temp": True,
                "cleanup_temp": True
            }
        )
        
        # Create domain RAG instance
        domain_rag = self.manager.add_domain_rag(domain_name, config)
        
        # Download documents from S3 to temporary directory
        temp_dir = Path(tempfile.mkdtemp(prefix=f"s3_domain_rag_{domain_name}_"))
        
        try:
            logger.info(f"Downloading documents from s3://{s3_bucket}/{s3_prefix}")
            
            download_results = self.manager.s3_manager.download_and_process_documents(
                bucket=s3_bucket,
                prefix=s3_prefix,
                temp_dir=temp_dir
            )
            
            successful_downloads = [r for r in download_results if r["success"]]
            failed_downloads = [r for r in download_results if not r["success"]]
            
            logger.info(f"Downloaded {len(successful_downloads)} documents from S3")
            if failed_downloads:
                logger.warning(f"Failed to download {len(failed_downloads)} documents")
            
            # Process downloaded documents
            if successful_downloads:
                process_results = []
                
                for download in successful_downloads:
                    local_path = Path(download["local_path"])
                    result = domain_rag.process_document(local_path)
                    process_results.append(result)
                    
                    if result.success:
                        logger.info(f"Processed {local_path.name}: {result.chunks_created} chunks")
                    else:
                        logger.error(f"Failed to process {local_path.name}: {result.error}")
                
                successful_processing = [r for r in process_results if r.success]
                logger.info(f"Successfully processed {len(successful_processing)}/{len(download_results)} documents")
            
        finally:
            # Cleanup temporary directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
        
        return domain_rag
    
    def upload_knowledge_base_to_s3(self, local_path: Union[str, Path], 
                                   domain_name: str = "default",
                                   s3_prefix: str = None) -> Dict[str, Any]:
        """Upload local knowledge base to S3 for domain RAG creation"""
        try:
            if not self.manager.s3_manager:
                return {
                    "success": False,
                    "error": "S3 manager not available"
                }
            
            # Set default s3_prefix if not provided
            if s3_prefix is None:
                s3_prefix = build_s3_path("knowledge_base", "")
            
            result = self.manager.s3_manager.upload_knowledge_base(
                local_path=local_path,
                s3_prefix=s3_prefix,
                domain_name=domain_name
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to upload knowledge base to S3: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def list_domains(self) -> List[str]:
        """List all available domain RAG systems"""
        return self.manager.list_domains()
    
    def get_domain_info(self, domain_name: str) -> Dict[str, Any]:
        """Get information about a specific domain RAG"""
        domain_rag = self.manager.get_domain_rag(domain_name)
        if not domain_rag:
            return {"success": False, "error": f"Domain not found: {domain_name}"}
        
        return {
            "success": True,
            "domain_name": domain_name,
            "config": domain_rag.export_config(),
            "stats": domain_rag.get_stats(),
            "flow_agreement": self.manager.create_flow_agreement(domain_name)
        }
    
    # Query Operations
    
    def query(self, query: str, domain: str = None, max_results: int = 5, 
              similarity_threshold: float = 0.7) -> RAGResponse:
        """
        Query knowledge base(s) with natural language.
        
        Args:
            query: Natural language question
            domain: Specific domain to query (None for best across all domains)
            max_results: Maximum number of source documents to return
            similarity_threshold: Minimum similarity score for results
            
        Returns:
            RAGResponse with answer, sources, and metadata
        """
        try:
            if domain:
                # Query specific domain
                return self.manager.query_domain(
                    domain, query, 
                    max_results=max_results,
                    similarity_threshold=similarity_threshold
                )
            else:
                # Query all domains and return best answer
                return self.manager.get_best_answer(
                    query,
                    max_results=max_results,
                    similarity_threshold=similarity_threshold
                )
                
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return RAGResponse(
                query=query,
                answer=f"I encountered an error processing your query: {str(e)}",
                sources=[],
                confidence=0.0,
                processing_time=0.0,
                metadata={"error": str(e)}
            )
    
    def query_all_domains(self, query: str, **kwargs) -> Dict[str, RAGResponse]:
        """Query all domains and return responses from each"""
        return self.manager.query_all_domains(query, **kwargs)
    
    # Chat Interface Integration
    
    def process_chat_message(self, message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process chat message with bracket command support.
        
        Supports commands like:
        - [model_validation_rag] What are Basel requirements?
        - [domain_name] Query text
        - Regular queries without brackets
        """
        context = context or {}
        
        # Check for bracket commands
        import re
        bracket_match = re.match(r'\[(\w+)\]\s*(.*)', message.strip())
        
        if bracket_match:
            command, query_text = bracket_match.groups()
            
            # Handle RAG commands
            if command.endswith('_rag'):
                domain_name = command[:-4]  # Remove '_rag' suffix
                if domain_name in self.list_domains():
                    response = self.query(query_text, domain=domain_name)
                    return {
                        "type": "domain_rag_query",
                        "domain": domain_name,
                        "query": query_text,
                        "response": response,
                        "success": True
                    }
                else:
                    return {
                        "type": "error",
                        "error": f"Domain RAG not found: {domain_name}",
                        "available_domains": self.list_domains(),
                        "success": False
                    }
            
            # Handle other bracket commands
            elif command in self.list_domains():
                response = self.query(query_text, domain=command)
                return {
                    "type": "domain_query",
                    "domain": command,
                    "query": query_text,
                    "response": response,
                    "success": True
                }
        
        # Regular query without bracket command
        response = self.query(message)
        return {
            "type": "general_query",
            "query": message,
            "response": response,
            "success": True
        }
    
    # Document Management
    
    def upload_document(self, file_path: Union[str, Path], domain: str = None) -> Dict[str, Any]:
        """Upload document to S3 and optionally process for domain RAG"""
        try:
            # Upload to S3
            upload_result = self.manager.upload_document_to_s3(file_path)
            
            if not upload_result["success"]:
                return upload_result
            
            result = {
                "upload": upload_result,
                "processing": None
            }
            
            # Process for domain if specified
            if domain and domain in self.list_domains():
                domain_rag = self.manager.get_domain_rag(domain)
                process_result = domain_rag.process_document(file_path)
                result["processing"] = {
                    "success": process_result.success,
                    "document_id": process_result.document_id,
                    "chunks_created": process_result.chunks_created,
                    "error": process_result.error
                }
            
            return {"success": True, "result": result}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # System Management
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return self.manager.get_system_status()
    
    def test_connections(self) -> Dict[str, Any]:
        """Test all system connections"""
        return self.manager.test_connections()
    
    def get_flow_agreements(self) -> Dict[str, Dict[str, Any]]:
        """Get Flow Agreement configurations for all domains"""
        agreements = {}
        for domain_name in self.list_domains():
            try:
                agreements[domain_name] = self.manager.create_flow_agreement(domain_name)
            except Exception as e:
                logger.error(f"Failed to create flow agreement for {domain_name}: {e}")
                agreements[domain_name] = {"error": str(e)}
        
        return agreements
    
    # MVR Analysis Integration
    
    def setup_mvr_integration(self) -> Dict[str, Any]:
        """Setup integration with MVR Analysis workflow"""
        try:
            # Ensure model validation domain exists
            if "model_validation" not in self.list_domains():
                # Look for knowledge base
                knowledge_base_path = Path(__file__).parent.parent.parent.parent / "knowledge_base"
                if knowledge_base_path.exists():
                    result = self.create_domain_rag("model_validation", knowledge_base_path, 
                                                   "Model validation and risk management knowledge base")
                    if not result["success"]:
                        return result
                else:
                    return {
                        "success": False,
                        "error": f"Knowledge base not found at {knowledge_base_path}"
                    }
            
            # Create MVR-specific flow agreement
            flow_agreement = self.manager.create_flow_agreement("model_validation")
            flow_agreement["mvr_integration"] = {
                "stages": ["mvr_tag", "mvr_qa", "mvr_peer", "mvr_report"],
                "knowledge_injection_points": {
                    "mvr_qa": "Inject model validation knowledge for Q&A generation",
                    "mvr_peer": "Provide regulatory context for peer review",
                    "mvr_report": "Supply templates and standards for reporting"
                }
            }
            
            return {
                "success": True,
                "flow_agreement": flow_agreement,
                "domain_stats": self.get_domain_info("model_validation")
            }
            
        except Exception as e:
            logger.error(f"Failed to setup MVR integration: {e}")
            return {"success": False, "error": str(e)}

# Global interface instance
_knowledge_interface_instance = None

def get_knowledge_interface(config: KnowledgeSystemConfig = None) -> KnowledgeInterface:
    """Get global knowledge interface instance"""
    global _knowledge_interface_instance
    
    if _knowledge_interface_instance is None or config is not None:
        _knowledge_interface_instance = KnowledgeInterface(config)
    
    return _knowledge_interface_instance