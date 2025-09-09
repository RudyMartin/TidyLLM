"""
Knowledge Resource Manager - Core Resource Management
===================================================

Manages knowledge resources, domains, and provides unified access to
knowledge operations for the MCP server.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from .sources import KnowledgeSource
from .interfaces import KnowledgeResource

logger = logging.getLogger("knowledge_resource_manager")


@dataclass
class SearchResult:
    """Result from knowledge search operation."""
    document_id: str
    title: str
    content: str
    similarity_score: float
    domain: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_id": self.document_id,
            "title": self.title,
            "content": self.content,
            "similarity_score": self.similarity_score,
            "domain": self.domain,
            "metadata": self.metadata
        }


class KnowledgeResourceManager:
    """
    Core manager for knowledge resources and operations.
    
    Provides unified interface for:
    - Domain registration and management
    - Document search and retrieval
    - Embedding generation
    - Structured data extraction
    - Natural language querying
    """
    
    def __init__(self):
        """Initialize Knowledge Resource Manager."""
        self.domains: Dict[str, KnowledgeSource] = {}
        self.domain_stats: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Knowledge Resource Manager initialized")
    
    def register_domain(self, domain_name: str, source: KnowledgeSource) -> None:
        """
        Register a knowledge domain.
        
        Args:
            domain_name: Unique domain identifier
            source: Knowledge source for the domain
        """
        try:
            # Initialize the source
            source.initialize()
            
            # Register domain
            self.domains[domain_name] = source
            
            # Initialize statistics
            self.domain_stats[domain_name] = {
                "registered_at": datetime.now().isoformat(),
                "document_count": 0,
                "last_accessed": None,
                "search_count": 0,
                "retrieval_count": 0
            }
            
            # Update document count
            try:
                doc_count = source.get_document_count()
                self.domain_stats[domain_name]["document_count"] = doc_count
            except:
                logger.warning(f"Could not get document count for domain '{domain_name}'")
            
            logger.info(f"Registered knowledge domain '{domain_name}' with {self.domain_stats[domain_name]['document_count']} documents")
            
        except Exception as e:
            logger.error(f"Failed to register domain '{domain_name}': {e}")
            raise
    
    def search(self, 
               query: str,
               domain: Optional[str] = None,
               max_results: int = 5,
               similarity_threshold: float = 0.7) -> List[SearchResult]:
        """
        Search across knowledge domains using semantic similarity.
        
        Args:
            query: Search query
            domain: Specific domain to search (None for all domains)
            max_results: Maximum number of results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of search results
        """
        try:
            all_results = []
            
            # Determine domains to search
            domains_to_search = [domain] if domain else list(self.domains.keys())
            
            for domain_name in domains_to_search:
                if domain_name not in self.domains:
                    logger.warning(f"Domain '{domain_name}' not found")
                    continue
                
                try:
                    source = self.domains[domain_name]
                    domain_results = source.search(query, max_results, similarity_threshold)
                    
                    # Convert to SearchResult objects
                    for result in domain_results:
                        search_result = SearchResult(
                            document_id=result.get("id", "unknown"),
                            title=result.get("title", ""),
                            content=result.get("content", ""),
                            similarity_score=result.get("similarity_score", 0.0),
                            domain=domain_name,
                            metadata=result.get("metadata", {})
                        )
                        all_results.append(search_result)
                    
                    # Update stats
                    self.domain_stats[domain_name]["search_count"] += 1
                    self.domain_stats[domain_name]["last_accessed"] = datetime.now().isoformat()
                    
                except Exception as e:
                    logger.error(f"Search failed for domain '{domain_name}': {e}")
                    continue
            
            # Sort by similarity score and limit results
            all_results.sort(key=lambda x: x.similarity_score, reverse=True)
            return all_results[:max_results]
            
        except Exception as e:
            logger.error(f"Search operation failed: {e}")
            return []
    
    def retrieve_document(self, document_id: str) -> Optional[KnowledgeResource]:
        """
        Retrieve a specific document by ID.
        
        Args:
            document_id: Document identifier
            
        Returns:
            KnowledgeResource if found, None otherwise
        """
        try:
            # Search across all domains for the document
            for domain_name, source in self.domains.items():
                try:
                    document = source.retrieve_by_id(document_id)
                    if document:
                        # Update stats
                        self.domain_stats[domain_name]["retrieval_count"] += 1
                        self.domain_stats[domain_name]["last_accessed"] = datetime.now().isoformat()
                        
                        return KnowledgeResource(
                            id=document_id,
                            title=document.get("title", ""),
                            content=document.get("content", ""),
                            domain=domain_name,
                            metadata=document.get("metadata", {}),
                            source_uri=document.get("source_uri", ""),
                            last_updated=document.get("last_updated")
                        )
                except Exception as e:
                    logger.debug(f"Document '{document_id}' not found in domain '{domain_name}': {e}")
                    continue
            
            logger.warning(f"Document '{document_id}' not found in any domain")
            return None
            
        except Exception as e:
            logger.error(f"Document retrieval failed for '{document_id}': {e}")
            return None
    
    def retrieve_by_criteria(self, domain: Optional[str], criteria: Dict[str, Any]) -> List[KnowledgeResource]:
        """
        Retrieve documents by search criteria.
        
        Args:
            domain: Specific domain to search (None for all domains)
            criteria: Search criteria
            
        Returns:
            List of matching documents
        """
        try:
            results = []
            domains_to_search = [domain] if domain else list(self.domains.keys())
            
            for domain_name in domains_to_search:
                if domain_name not in self.domains:
                    continue
                
                try:
                    source = self.domains[domain_name]
                    documents = source.retrieve_by_criteria(criteria)
                    
                    for doc in documents:
                        resource = KnowledgeResource(
                            id=doc.get("id", "unknown"),
                            title=doc.get("title", ""),
                            content=doc.get("content", ""),
                            domain=domain_name,
                            metadata=doc.get("metadata", {}),
                            source_uri=doc.get("source_uri", ""),
                            last_updated=doc.get("last_updated")
                        )
                        results.append(resource)
                    
                    # Update stats
                    self.domain_stats[domain_name]["retrieval_count"] += 1
                    self.domain_stats[domain_name]["last_accessed"] = datetime.now().isoformat()
                    
                except Exception as e:
                    logger.error(f"Criteria retrieval failed for domain '{domain_name}': {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Criteria retrieval failed: {e}")
            return []
    
    def generate_embedding(self, text: str, model: str = "sentence-transformers") -> List[float]:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
            model: Embedding model to use
            
        Returns:
            Embedding vector
        """
        try:
            # Use first available domain's embedding capability
            for domain_name, source in self.domains.items():
                try:
                    if hasattr(source, 'generate_embedding'):
                        return source.generate_embedding(text, model)
                except:
                    continue
            
            # Fallback: mock embedding
            logger.warning(f"No embedding capability available, returning mock embedding")
            return [0.1] * 384  # Mock 384-dimensional embedding
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return []
    
    def extract_structured_data(self, document_id: str, extraction_type: str) -> Dict[str, Any]:
        """
        Extract structured data from document.
        
        Args:
            document_id: Document identifier
            extraction_type: Type of extraction (entities, keywords, summary, etc.)
            
        Returns:
            Extracted structured data
        """
        try:
            # Find document and extract data
            for domain_name, source in self.domains.items():
                try:
                    if hasattr(source, 'extract_structured_data'):
                        result = source.extract_structured_data(document_id, extraction_type)
                        if result:
                            return result
                except:
                    continue
            
            # Fallback: mock extraction
            return {
                "extraction_type": extraction_type,
                "document_id": document_id,
                "status": "mock_extraction",
                "data": f"Mock {extraction_type} extraction for document {document_id}"
            }
            
        except Exception as e:
            logger.error(f"Structured data extraction failed: {e}")
            return {"error": str(e)}
    
    def query_knowledge(self, 
                       question: str, 
                       domain: Optional[str] = None,
                       context_length: int = 2000) -> Tuple[str, str]:
        """
        Query knowledge using natural language.
        
        Args:
            question: Natural language question
            domain: Specific domain to query
            context_length: Maximum context length
            
        Returns:
            Tuple of (answer, context)
        """
        try:
            # First, search for relevant documents
            search_results = self.search(question, domain, max_results=3, similarity_threshold=0.6)
            
            if not search_results:
                return "No relevant information found.", ""
            
            # Build context from search results
            context_parts = []
            total_length = 0
            
            for result in search_results:
                content = result.content
                if total_length + len(content) > context_length:
                    # Truncate to fit
                    remaining = context_length - total_length
                    content = content[:remaining] + "..."
                    context_parts.append(content)
                    break
                else:
                    context_parts.append(content)
                    total_length += len(content)
            
            context = "\n\n".join(context_parts)
            
            # Generate answer (mock implementation)
            answer = f"Based on the available knowledge, here's what I found regarding '{question}': " + \
                    f"The information suggests that {search_results[0].content[:200]}..."
            
            return answer, context
            
        except Exception as e:
            logger.error(f"Knowledge query failed: {e}")
            return f"Error processing query: {str(e)}", ""
    
    def get_domain_stats(self, domain_name: str) -> Dict[str, Any]:
        """Get statistics for a specific domain."""
        return self.domain_stats.get(domain_name, {})
    
    def get_all_domains(self) -> List[str]:
        """Get list of all registered domains."""
        return list(self.domains.keys())
    
    def get_status(self) -> Dict[str, Any]:
        """Get overall resource manager status."""
        return {
            "registered_domains": len(self.domains),
            "total_documents": sum(stats.get("document_count", 0) for stats in self.domain_stats.values()),
            "total_searches": sum(stats.get("search_count", 0) for stats in self.domain_stats.values()),
            "total_retrievals": sum(stats.get("retrieval_count", 0) for stats in self.domain_stats.values()),
            "domains": list(self.domains.keys()),
            "last_updated": datetime.now().isoformat()
        }