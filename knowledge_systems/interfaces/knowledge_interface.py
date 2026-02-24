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
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from ..knowledge_manager import KnowledgeManager, KnowledgeSystemConfig
from ..domain_rag import RAGQuery, RAGResponse

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
              similarity_threshold: float = 0.7, authority_tier: int = None,
              collection_name: str = None) -> RAGResponse:
        """
        Query knowledge base(s) with natural language.

        Args:
            query: Natural language question
            domain: Specific domain to query (None for best across all domains)
            max_results: Maximum number of source documents to return
            similarity_threshold: Minimum similarity score for results
            authority_tier: Authority level for compliance queries (1=Regulatory, 2=SOP, 3=Technical)
            collection_name: Specific collection to query within domain

        Returns:
            RAGResponse with answer, sources, and metadata (enhanced with authority info if applicable)
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

    # ═══════════════════════════════════════════════════════════════════════════════
    # FIVE RAG SYSTEMS - Authority-Based Query Methods (Postgres Integration)
    # ═══════════════════════════════════════════════════════════════════════════════

    def query_compliance_rag(self, query: str, domain: str, authority_tier: int = None,
                           confidence_threshold: float = 0.8) -> RAGResponse:
        """
        [RAG 1/5] ComplianceRAG query with authority-based precedence.

        Authority-based regulatory decisions with tiered compliance:
        - Tier 1: Regulatory (highest precedence)
        - Tier 2: Standard Operating Procedures
        - Tier 3: Technical guidance

        Args:
            query: Natural language question
            domain: Domain name for compliance context
            authority_tier: Specific authority tier (None for highest available)
            confidence_threshold: Minimum confidence for results

        Returns:
            RAGResponse with authority metadata
        """
        try:
            # Use existing query method with authority tier
            response = self.query(
                query=query,
                domain=domain,
                authority_tier=authority_tier,
                similarity_threshold=confidence_threshold
            )

            # Enhance response metadata with authority information
            response.metadata = response.metadata or {}
            response.metadata.update({
                "query_type": "compliance_rag",
                "authority_tier": authority_tier,
                "precedence_level": 1.0 if authority_tier == 1 else (0.8 if authority_tier == 2 else 0.6),
                "compliance_context": True
            })

            return response

        except Exception as e:
            logger.error(f"ComplianceRAG query failed: {e}")
            return RAGResponse(
                query=query,
                answer=f"No authoritative guidance found for this compliance query: {str(e)}",
                sources=[],
                confidence=0.0,
                processing_time=0.0,
                metadata={
                    "error": str(e),
                    "query_type": "compliance_rag",
                    "authority_tier": 0
                }
            )

    def query_postgres_rag(self, query: str, collection_name: str = None,
                          confidence_threshold: float = 0.7) -> RAGResponse:
        """
        [RAG 5/5] PostgresRAG query for direct database vector search.

        Direct postgres+pgvector search without domain restrictions.
        Uses postgres_rag_adapter for raw vector similarity search.

        Args:
            query: Natural language search query
            collection_name: Specific collection to search
            confidence_threshold: Minimum confidence score

        Returns:
            RAGResponse with postgres vector search results
        """
        from ..adapters.postgres_rag import PostgresRAGAdapter, RAGQuery

        try:
            adapter = PostgresRAGAdapter()
            rag_query = RAGQuery(
                query=query,
                collection_name=collection_name,
                confidence_threshold=confidence_threshold
            )

            response = adapter.query_collection(rag_query)
            return response

        except Exception as e:
            logger.error(f"PostgresRAG query failed: {e}")
            from ..core.domain_rag import RAGResponse
            return RAGResponse(
                query=query,
                response=f"PostgresRAG error: {str(e)}",
                sources=[],
                confidence=0.0,
                collection_name=collection_name or "postgres_error",
                authority_tier=5
            )

    def query_document_rag(self, query: str, domain: str = None, collection_name: str = None,
                          confidence_threshold: float = 0.8) -> RAGResponse:
        """
        [RAG 2/5] DocumentRAG query for general document search and retrieval.

        Simple semantic search across document collections for information discovery.

        Args:
            query: Natural language question
            domain: Domain to search (None for all domains)
            collection_name: Specific collection to query
            confidence_threshold: Minimum confidence for results

        Returns:
            RAGResponse with document search results
        """
        try:
            # Use existing query method for document search
            response = self.query(
                query=query,
                domain=domain,
                collection_name=collection_name,
                similarity_threshold=confidence_threshold,
                max_results=10  # More results for document discovery
            )

            # Enhance response metadata
            response.metadata = response.metadata or {}
            response.metadata.update({
                "query_type": "document_rag",
                "search_scope": collection_name or domain or "all_domains",
                "document_discovery": True
            })

            return response

        except Exception as e:
            logger.error(f"DocumentRAG query failed: {e}")
            return RAGResponse(
                query=query,
                answer=f"No relevant knowledge found for this document query: {str(e)}",
                sources=[],
                confidence=0.0,
                processing_time=0.0,
                metadata={
                    "error": str(e),
                    "query_type": "document_rag"
                }
            )

    def query_expert_rag(self, query: str, domain: str = None, collection_name: str = None,
                        confidence_threshold: float = 0.8) -> RAGResponse:
        """
[RAG 3/5] ExpertRAG query for specialized subject matter expertise.

        Leverages expert knowledge collections for high-level analysis and recommendations.

        Args:
            query: Natural language question requiring expert analysis
            domain: Domain for expert knowledge
            collection_name: Specific expert collection
            confidence_threshold: Minimum confidence for expert opinions

        Returns:
            RAGResponse with expert analysis and high precedence level
        """
        try:
            # Use existing query method for expert knowledge
            response = self.query(
                query=query,
                domain=domain,
                collection_name=collection_name,
                similarity_threshold=confidence_threshold,
                max_results=5  # Focused expert responses
            )

            # Enhance response with expert context
            if response.sources and len(response.sources) > 0:
                expert_analysis = f"Expert Analysis: {response.answer}"
            else:
                expert_analysis = "No expert knowledge available for this query."

            response.answer = expert_analysis
            response.metadata = response.metadata or {}
            response.metadata.update({
                "query_type": "expert_rag",
                "authority_tier": 99,  # Expert level but not regulatory authority
                "precedence_level": 0.9,  # High expertise precedence
                "expert_analysis": True
            })

            return response

        except Exception as e:
            logger.error(f"ExpertRAG query failed: {e}")
            return RAGResponse(
                query=query,
                answer=f"No expert knowledge available for this query: {str(e)}",
                sources=[],
                confidence=0.0,
                processing_time=0.0,
                metadata={
                    "error": str(e),
                    "query_type": "expert_rag",
                    "authority_tier": 99
                }
            )

    def query_unified_rag(self, query: str, domain: str = None, authority_tier: int = None,
                         collection_name: str = None, confidence_threshold: float = 0.8) -> RAGResponse:
        """
[RAG 4/5] Unified RAG query that automatically selects the best approach.

        Logic:
        1. If authority_tier specified -> ComplianceRAG
        2. If collection_name specified -> DocumentRAG/ExpertRAG based on metadata
        3. Otherwise -> Try ComplianceRAG first, fallback to DocumentRAG

        Args:
            query: Natural language question
            domain: Domain context
            authority_tier: Authority level (triggers ComplianceRAG)
            collection_name: Specific collection (triggers DocumentRAG/ExpertRAG)
            confidence_threshold: Minimum confidence threshold

        Returns:
            RAGResponse from the most appropriate RAG type
        """
        logger.info(f"Unified RAG Query: {query}")

        try:
            # Route to appropriate RAG type
            if authority_tier:
                return self.query_compliance_rag(query, domain, authority_tier, confidence_threshold)
            elif collection_name:
                # For now, default to DocumentRAG - could be enhanced to detect expert collections
                return self.query_document_rag(query, domain, collection_name, confidence_threshold)
            else:
                # Try compliance first (authoritative), fallback to document
                compliance_response = self.query_compliance_rag(query, domain, None, confidence_threshold)
                if compliance_response.confidence > 0.3:
                    return compliance_response
                else:
                    return self.query_document_rag(query, domain, None, confidence_threshold)

        except Exception as e:
            logger.error(f"Unified RAG query failed: {e}")
            return RAGResponse(
                query=query,
                answer=f"Unified query processing failed: {str(e)}",
                sources=[],
                confidence=0.0,
                processing_time=0.0,
                metadata={"error": str(e), "query_type": "unified_rag"}
            )
    
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
    
    # Domain Registration (Migrated Systems Integration)

    def register_migrated_domain(self, domain_name: str, domain_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register a migrated domain RAG system with the unified interface.

        This allows migrated systems from tidyllm/knowledge_systems/migrated/ to
        integrate with the unified KnowledgeInterface while maintaining their
        specific configurations.

        Args:
            domain_name: Unique domain identifier (e.g., "compliance_regulatory", "model_risk")
            domain_config: Domain configuration including:
                - source: Path to migrated system
                - authority_tier: Authority level (1-3 for compliance, 99 for expert)
                - s3_config: S3 bucket and prefix info
                - description: Human readable description
                - adapter_class: Class reference for the migrated adapter

        Returns:
            Dict with registration results
        """
        try:
            # Validate domain config
            required_fields = ["source", "description"]
            missing_fields = [f for f in required_fields if f not in domain_config]
            if missing_fields:
                return {
                    "success": False,
                    "error": f"Missing required fields: {missing_fields}"
                }

            # Store domain configuration for future use
            if not hasattr(self, '_migrated_domains'):
                self._migrated_domains = {}

            self._migrated_domains[domain_name] = {
                "config": domain_config,
                "registered_at": datetime.now().isoformat(),
                "status": "registered"
            }

            logger.info(f"Registered migrated domain: {domain_name}")
            return {
                "success": True,
                "domain_name": domain_name,
                "registration_time": datetime.now().isoformat(),
                "available_via": [
                    "query_compliance_rag()" if domain_config.get("authority_tier") else None,
                    "query_expert_rag()" if domain_config.get("authority_tier") == 99 else None,
                    "query_document_rag()",
                    "query_unified_rag()"
                ]
            }

        except Exception as e:
            logger.error(f"Failed to register migrated domain '{domain_name}': {e}")
            return {
                "success": False,
                "domain_name": domain_name,
                "error": str(e)
            }

    def list_migrated_domains(self) -> Dict[str, Any]:
        """List all registered migrated domains"""
        if not hasattr(self, '_migrated_domains'):
            return {"migrated_domains": {}, "count": 0}

        return {
            "migrated_domains": self._migrated_domains,
            "count": len(self._migrated_domains),
            "registration_summary": {
                domain_name: {
                    "description": config["config"].get("description", ""),
                    "authority_tier": config["config"].get("authority_tier"),
                    "source": config["config"].get("source", ""),
                    "status": config["status"]
                }
                for domain_name, config in self._migrated_domains.items()
            }
        }

    def get_available_domain_configs(self) -> List[Dict[str, Any]]:
        """
        Get pre-defined domain configurations for common migrated systems.

        This provides standard configurations for integrating common migrated
        RAG systems without requiring manual configuration.

        Returns:
            List of domain configuration templates
        """
        return [
            {
                "domain_name": "regulatory_compliance",
                "description": "Regulatory compliance and authoritative guidance",
                "authority_tier": 1,
                "source": "knowledge_systems.migrated.compliance.domain_rag.authoritative_rag",
                "s3_config": {
                    "bucket": "nsc-mvp1",
                    "prefix": "knowledge_base/regulatory"
                },
                "adapter_type": "compliance_rag"
            },
            {
                "domain_name": "business_analysis",
                "description": "Business analysis and regulatory research",
                "authority_tier": 2,
                "source": "knowledge_systems.migrated.compliance.research_papers.business_analysis_rag",
                "s3_config": {
                    "bucket": "nsc-mvp1",
                    "prefix": "knowledge_base/business_analysis"
                },
                "adapter_type": "document_rag"
            },
            {
                "domain_name": "intelligent_document_processing",
                "description": "Intelligent document processing with Bedrock embeddings",
                "authority_tier": None,
                "source": "knowledge_systems.migrated.scattered_rag.adapters.intelligent_rag_adapter",
                "s3_config": {
                    "bucket": "dsai-2025-asu",
                    "prefix": "knowledge_base/documents"
                },
                "adapter_type": "document_rag"
            },
            {
                "domain_name": "ai_powered_expert_analysis",
                "description": "AI-powered expert analysis and recommendations",
                "authority_tier": 99,
                "source": "knowledge_systems.migrated.scattered_rag.adapters.ai_powered_rag_adapter",
                "s3_config": {
                    "bucket": "dsai-2025-asu",
                    "prefix": "knowledge_base/expert"
                },
                "adapter_type": "expert_rag"
            },
            {
                "domain_name": "unified_postgres_rag",
                "description": "Unified PostgreSQL-based RAG with authority tiers",
                "authority_tier": None,  # Supports all tiers
                "source": "knowledge_systems.migrated.scattered_rag.adapters.postgres_rag_adapter",
                "s3_config": {
                    "bucket": "dsai-2025-asu",
                    "prefix": "knowledge_base/unified"
                },
                "adapter_type": "unified_rag"
            }
        ]

    def register_standard_domains(self) -> Dict[str, Any]:
        """
        Register all standard domain configurations.

        This is a convenience method to register common migrated systems
        with their standard configurations.

        Returns:
            Dict with registration results for all standard domains
        """
        results = {}
        domain_configs = self.get_available_domain_configs()

        for config in domain_configs:
            domain_name = config.pop("domain_name")
            result = self.register_migrated_domain(domain_name, config)
            results[domain_name] = result

        return {
            "registered_count": len([r for r in results.values() if r["success"]]),
            "failed_count": len([r for r in results.values() if not r["success"]]),
            "results": results
        }

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