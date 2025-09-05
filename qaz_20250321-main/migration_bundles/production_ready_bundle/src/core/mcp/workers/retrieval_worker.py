"""
Retrieval Worker - Document Retrieval and Search

Specialized worker for document retrieval, search, and information gathering
operations using vector search and semantic similarity.
"""

from typing import List, Dict, Any, Optional
from ..worker import Worker
import json
import logging
from datetime import datetime


# Robust import setup
import sys
from pathlib import Path
_src_dir = Path(__file__).parent
while _src_dir.name != "src" and _src_dir.parent != _src_dir:
    _src_dir = _src_dir.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

logger = logging.getLogger(__name__)

class RetrievalWorker(Worker):
    """Specialized worker for document retrieval and search"""
    
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__("retriever", "document retrieval and search", model_config)
        self.vector_store = None  # Will be initialized when needed
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    def _create_execution_prompt(self, task: Dict[str, Any], context: MCPContext) -> str:
        """Create specialized execution prompt for retrieval tasks"""
        
        task_description = task.get("task", "")
        input_data = task.get("input_data", {})
        constraints = task.get("constraints", {})
        
        # Extract search parameters
        query = input_data.get("query", "")
        top_k = input_data.get("top_k", 10)
        similarity_threshold = input_data.get("similarity_threshold", 0.7)
        
        prompt = f"""
        As a document retrieval specialist, execute this search task:
        
        TASK: {task_description}
        
        SEARCH QUERY: {query}
        
        SEARCH PARAMETERS:
        - Top K results: {top_k}
        - Similarity threshold: {similarity_threshold}
        
        CONSTRAINTS: {json.dumps(constraints, indent=2)}
        
        RETRIEVAL INSTRUCTIONS:
        1. Understand the search query and intent
        2. Identify relevant search terms and concepts
        3. Perform semantic search if vector store is available
        4. Rank results by relevance and quality
        5. Provide context and explanations for each result
        6. Include metadata about the search process
        
        If vector search is not available, provide intelligent search guidance
        and recommendations for finding relevant information.
        """
        
        return prompt
    
    def _format_result(self, result: str, task: Dict[str, Any], execution_duration: float) -> Dict[str, Any]:
        """Format retrieval result with specialized metadata"""
        
        # Extract search-specific metadata
        search_metadata = self._extract_search_metadata(result, task)
        
        base_result = super()._format_result(result, task, execution_duration)
        
        # Add retrieval-specific fields
        base_result.update({
            "retrieval_metadata": {
                "search_type": search_metadata.get("search_type", "semantic"),
                "results_count": search_metadata.get("results_count", 0),
                "query_terms": search_metadata.get("query_terms", []),
                "search_scope": search_metadata.get("search_scope", "general"),
                "relevance_score": search_metadata.get("relevance_score", 0.8)
            },
            "search_parameters": {
                "query": task.get("input_data", {}).get("query", ""),
                "top_k": task.get("input_data", {}).get("top_k", 10),
                "similarity_threshold": task.get("input_data", {}).get("similarity_threshold", 0.7)
            }
        })
        
        return base_result
    
    def _extract_search_metadata(self, result: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Extract search-specific metadata from result"""
        metadata = {
            "search_type": "semantic",
            "results_count": 0,
            "query_terms": [],
            "search_scope": "general",
            "relevance_score": 0.8
        }
        
        # Extract query terms from task
        query = task.get("input_data", {}).get("query", "")
        if query:
            metadata["query_terms"] = query.split()
        
        # Analyze result for search patterns
        result_lower = result.lower()
        
        # Count results (look for numbered lists or result indicators)
        if "result" in result_lower or "found" in result_lower:
            # Simple heuristic to count results
            lines = result.split('\n')
            result_lines = [line for line in lines if any(char.isdigit() for char in line[:3])]
            metadata["results_count"] = len(result_lines)
        
        # Detect search type
        if "vector" in result_lower or "semantic" in result_lower:
            metadata["search_type"] = "semantic"
        elif "keyword" in result_lower or "exact" in result_lower:
            metadata["search_type"] = "keyword"
        elif "fuzzy" in result_lower:
            metadata["search_type"] = "fuzzy"
        
        # Detect search scope
        if any(word in result_lower for word in ["document", "file", "pdf"]):
            metadata["search_scope"] = "documents"
        elif any(word in result_lower for word in ["database", "index"]):
            metadata["search_scope"] = "database"
        elif any(word in result_lower for word in ["web", "internet"]):
            metadata["search_scope"] = "web"
        
        return metadata
    
    def _get_specialization_capabilities(self) -> list:
        """Get retrieval-specific capabilities"""
        return [
            "Document search and retrieval",
            "Vector similarity search",
            "Semantic search",
            "Information gathering",
            "Content indexing",
            "Query optimization",
            "Result ranking and filtering",
            "Search result summarization",
            "Multi-modal search (text, images, documents)",
            "Search analytics and insights"
        ]
    
    def perform_vector_search(self, query: str, top_k: int = 10, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Perform vector search using the vector store"""
        try:
            if self.vector_store is None:
                # Initialize vector store if needed
                from core.vector_store import VectorStore
                self.vector_store = VectorStore()
            
            # Perform similarity search
            search_results = self.vector_store.similarity_search(
                query, k=top_k, similarity_threshold=similarity_threshold
            )
            
            # Process and format results
            processed_results = []
            for result in search_results:
                processed_result = {
                    "content": result.content,
                    "metadata": result.metadata,
                    "relevance_score": result.score,
                    "summary": self._generate_content_summary(result.content, query)
                }
                processed_results.append(processed_result)
            
            return processed_results
            
        except Exception as e:
            self.logger.error(f"Vector search error: {e}")
            return []
    
    def _generate_content_summary(self, content: str, query: str) -> str:
        """Generate summary of content relevant to query"""
        try:
            summary_prompt = f"""
            Summarize the following content in relation to the query:
            
            Query: {query}
            Content: {content[:1000]}...
            
            Provide a concise summary highlighting relevance to the query.
            Focus on the most important points that address the search query.
            """
            
            return self.llm_manager.generate_response(summary_prompt)
            
        except Exception as e:
            self.logger.error(f"Content summary generation error: {e}")
            return content[:200] + "..." if len(content) > 200 else content
    
    def get_search_analytics(self) -> Dict[str, Any]:
        """Get search analytics and performance metrics"""
        metrics = self.get_performance_metrics()
        
        # Add retrieval-specific analytics
        search_analytics = {
            "total_searches": metrics["total_tasks"],
            "average_search_duration": metrics["average_duration"],
            "search_success_rate": metrics["success_rate"],
            "search_types": {
                "semantic": 0,
                "keyword": 0,
                "fuzzy": 0
            },
            "search_scopes": {
                "documents": 0,
                "database": 0,
                "web": 0,
                "general": 0
            }
        }
        
        # Analyze task history for search patterns
        for task_record in self.task_history:
            task = task_record["task"]
            result = task_record["result"]
            
            # Count search types
            search_type = result.get("retrieval_metadata", {}).get("search_type", "semantic")
            search_analytics["search_types"][search_type] = search_analytics["search_types"].get(search_type, 0) + 1
            
            # Count search scopes
            search_scope = result.get("retrieval_metadata", {}).get("search_scope", "general")
            search_analytics["search_scopes"][search_scope] = search_analytics["search_scopes"].get(search_scope, 0) + 1
        
        return search_analytics
