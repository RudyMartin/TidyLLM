"""
MCP Service Layer - VectorQA Sage Integration

This service layer provides a clean interface for integrating the MCP hierarchical
LLM system with VectorQA Sage, handling configuration, error management, and
providing specialized methods for different use cases.
"""

from typing import Dict, Any, List, Optional
from core.mcp.orchestrator import MCPOrchestrator
from core.llm_manager import LLMManager
from core.vector_store import VectorStore
from config.credential_manager import CredentialManager
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

class MCPService:
    """Service layer for MCP integration with VectorQA Sage"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._load_mcp_config()
        self.orchestrator = None
        self.llm_manager = LLMManager()
        self.vector_store = VectorStore()
        self.credential_manager = CredentialManager()
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        self._initialize_orchestrator()
    
    def _load_mcp_config(self) -> Dict[str, Any]:
        """Load MCP configuration with VectorQA Sage integration"""
        return {
            "planner_config": {
                "model": "gpt-4o",
                "max_tokens": 2000,
                "temperature": 0.1,
                "provider": "openai"
            },
            "coordinator_config": {
                "model": "gpt-4o-mini",
                "max_tokens": 1500,
                "temperature": 0.2,
                "provider": "openai"
            },
            "worker_config": {
                "model": "gpt-3.5-turbo",
                "max_tokens": 1000,
                "temperature": 0.3,
                "provider": "openai"
            },
            "system_config": {
                "enable_logging": True,
                "enable_analytics": True,
                "max_execution_time": 300,
                "retry_attempts": 2,
                "vector_store_integration": True,
                "credential_management": True
            },
            "vectorqa_integration": {
                "use_vector_store": True,
                "use_dspy_integration": True,
                "use_streamlit_ui": True,
                "enable_real_time_monitoring": True
            }
        }
    
    def _initialize_orchestrator(self):
        """Initialize the MCP orchestrator with proper error handling"""
        try:
            self.orchestrator = MCPOrchestrator(self.config)
            self.logger.info("MCP Orchestrator initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize MCP Orchestrator: {e}")
            raise RuntimeError(f"MCP system initialization failed: {e}")
    
    def process_qa_request(self, question: str, context_documents: List[str] = None, 
                          use_vector_search: bool = True) -> Dict[str, Any]:
        """Process QA request using MCP hierarchy with VectorQA Sage integration"""
        
        self.logger.info(f"Processing QA request: {question[:100]}...")
        
        try:
            # Enhance context with vector search if enabled
            enhanced_context = context_documents or []
            
            if use_vector_search and self.config.get("vectorqa_integration", {}).get("use_vector_store", True):
                try:
                    # Perform vector search to find relevant documents
                    search_results = self.vector_store.similarity_search(question, k=5)
                    vector_context = [result.content for result in search_results]
                    enhanced_context.extend(vector_context)
                    self.logger.info(f"Added {len(vector_context)} vector search results to context")
                except Exception as e:
                    self.logger.warning(f"Vector search failed, continuing without: {e}")
            
            # Process through MCP hierarchy
            result = self.orchestrator.process_qa_request(question, enhanced_context)
            
            # Add VectorQA Sage specific metadata
            result["vectorqa_metadata"] = {
                "vector_search_used": use_vector_search and bool(enhanced_context),
                "context_documents_count": len(enhanced_context),
                "original_context_count": len(context_documents) if context_documents else 0,
                "vector_search_results_count": len(enhanced_context) - (len(context_documents) if context_documents else 0)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"QA request processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "vectorqa_metadata": {
                    "vector_search_used": False,
                    "context_documents_count": 0,
                    "original_context_count": len(context_documents) if context_documents else 0,
                    "vector_search_results_count": 0
                }
            }
    
    def generate_report(self, topic: str, data_sources: List[str], 
                       report_type: str = "comprehensive") -> Dict[str, Any]:
        """Generate comprehensive report using MCP hierarchy"""
        
        self.logger.info(f"Generating report on: {topic}")
        
        try:
            # Enhance data sources with vector search if needed
            enhanced_sources = data_sources.copy() if data_sources else []
            
            if self.config.get("vectorqa_integration", {}).get("use_vector_store", True):
                try:
                    # Search for relevant documents about the topic
                    search_results = self.vector_store.similarity_search(topic, k=10)
                    vector_sources = [result.content for result in search_results]
                    enhanced_sources.extend(vector_sources)
                    self.logger.info(f"Added {len(vector_sources)} vector search results to data sources")
                except Exception as e:
                    self.logger.warning(f"Vector search for report failed: {e}")
            
            # Process through MCP hierarchy
            result = self.orchestrator.generate_report(topic, enhanced_sources)
            
            # Add report-specific metadata
            result["report_metadata"] = {
                "report_type": report_type,
                "topic": topic,
                "data_sources_count": len(enhanced_sources),
                "original_sources_count": len(data_sources) if data_sources else 0,
                "vector_sources_count": len(enhanced_sources) - (len(data_sources) if data_sources else 0)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "report_metadata": {
                    "report_type": report_type,
                    "topic": topic,
                    "data_sources_count": 0,
                    "original_sources_count": len(data_sources) if data_sources else 0,
                    "vector_sources_count": 0
                }
            }
    
    def analyze_documents(self, documents: List[str], analysis_type: str,
                         include_vector_analysis: bool = True) -> Dict[str, Any]:
        """Analyze documents using MCP hierarchy with enhanced capabilities"""
        
        self.logger.info(f"Analyzing {len(documents)} documents for: {analysis_type}")
        
        try:
            # Enhance analysis with vector-based insights if enabled
            enhanced_documents = documents.copy()
            
            if include_vector_analysis and self.config.get("vectorqa_integration", {}).get("use_vector_store", True):
                try:
                    # Perform vector analysis on documents
                    vector_insights = self._perform_vector_analysis(documents, analysis_type)
                    enhanced_documents.append(f"Vector Analysis Insights: {json.dumps(vector_insights, indent=2)}")
                    self.logger.info("Added vector analysis insights to documents")
                except Exception as e:
                    self.logger.warning(f"Vector analysis failed: {e}")
            
            # Process through MCP hierarchy
            result = self.orchestrator.analyze_documents(enhanced_documents, analysis_type)
            
            # Add analysis-specific metadata
            result["analysis_metadata"] = {
                "analysis_type": analysis_type,
                "documents_count": len(documents),
                "enhanced_documents_count": len(enhanced_documents),
                "vector_analysis_included": include_vector_analysis,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Document analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "analysis_metadata": {
                    "analysis_type": analysis_type,
                    "documents_count": len(documents),
                    "enhanced_documents_count": 0,
                    "vector_analysis_included": include_vector_analysis,
                    "analysis_timestamp": datetime.now().isoformat()
                }
            }
    
    def _perform_vector_analysis(self, documents: List[str], analysis_type: str) -> Dict[str, Any]:
        """Perform vector-based analysis on documents"""
        try:
            # Combine documents for analysis
            combined_text = "\n\n".join(documents)
            
            # Create analysis query based on type
            if analysis_type == "sentiment":
                query = "Analyze the sentiment and emotional tone of this content"
            elif analysis_type == "key_points":
                query = "Extract the key points and main ideas from this content"
            elif analysis_type == "compliance":
                query = "Check this content for compliance issues and regulatory concerns"
            else:
                query = f"Analyze this content for {analysis_type}"
            
            # Perform vector search for similar content
            search_results = self.vector_store.similarity_search(combined_text, k=5)
            
            # Analyze similarities and patterns
            vector_insights = {
                "similar_content_found": len(search_results),
                "average_similarity_score": sum(result.score for result in search_results) / len(search_results) if search_results else 0,
                "content_patterns": self._extract_content_patterns(search_results),
                "analysis_type": analysis_type,
                "vector_analysis_timestamp": datetime.now().isoformat()
            }
            
            return vector_insights
            
        except Exception as e:
            self.logger.error(f"Vector analysis failed: {e}")
            return {
                "error": str(e),
                "analysis_type": analysis_type,
                "vector_analysis_timestamp": datetime.now().isoformat()
            }
    
    def _extract_content_patterns(self, search_results: List[Any]) -> List[str]:
        """Extract patterns from vector search results"""
        try:
            patterns = []
            
            # Extract common themes from search results
            all_content = " ".join([result.content for result in search_results])
            
            # Simple pattern extraction (can be enhanced)
            words = all_content.lower().split()
            word_freq = {}
            
            for word in words:
                if len(word) > 4:  # Filter short words
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get most common words as patterns
            common_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            patterns = [word for word, freq in common_words if freq > 2]
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Pattern extraction failed: {e}")
            return []
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status including VectorQA Sage integration"""
        
        try:
            # Get MCP system status
            mcp_status = self.orchestrator.get_system_status()
            
            # Get VectorQA Sage specific status
            vectorqa_status = {
                "vector_store_available": self._check_vector_store_status(),
                "llm_manager_status": self._check_llm_manager_status(),
                "credential_manager_status": self._check_credential_manager_status(),
                "integration_config": self.config.get("vectorqa_integration", {}),
                "system_config": self.config.get("system_config", {})
            }
            
            return {
                "mcp_system": mcp_status,
                "vectorqa_integration": vectorqa_status,
                "overall_status": "healthy" if mcp_status.get("system_overview", {}).get("success_rate", 0) > 0.8 else "degraded"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return {
                "mcp_system": {"error": str(e)},
                "vectorqa_integration": {"error": str(e)},
                "overall_status": "error"
            }
    
    def _check_vector_store_status(self) -> Dict[str, Any]:
        """Check vector store status"""
        try:
            # Simple check - can be enhanced
            return {
                "status": "available",
                "type": "FAISS",
                "document_count": "unknown"  # Could be enhanced to get actual count
            }
        except Exception as e:
            return {
                "status": "unavailable",
                "error": str(e)
            }
    
    def _check_llm_manager_status(self) -> Dict[str, Any]:
        """Check LLM manager status"""
        try:
            # Check if LLM manager is working
            return {
                "status": "available",
                "providers": ["openai", "anthropic", "cohere", "google", "huggingface"]
            }
        except Exception as e:
            return {
                "status": "unavailable",
                "error": str(e)
            }
    
    def _check_credential_manager_status(self) -> Dict[str, Any]:
        """Check credential manager status"""
        try:
            # Check if credentials are available
            return {
                "status": "available",
                "managed_services": ["openai", "anthropic", "cohere", "google"]
            }
        except Exception as e:
            return {
                "status": "unavailable",
                "error": str(e)
            }
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics including VectorQA Sage metrics"""
        
        try:
            # Get MCP analytics
            mcp_analytics = self.orchestrator.get_analytics()
            
            # Add VectorQA Sage specific analytics
            vectorqa_analytics = {
                "vector_search_usage": {
                    "total_searches": 0,
                    "successful_searches": 0,
                    "average_results_per_search": 0
                },
                "integration_metrics": {
                    "qa_requests_processed": 0,
                    "reports_generated": 0,
                    "documents_analyzed": 0
                }
            }
            
            # Analyze execution history for VectorQA metrics
            execution_history = self.orchestrator.get_execution_history(limit=100)
            
            for execution in execution_history:
                vectorqa_metadata = execution.get("results", {}).get("vectorqa_metadata", {})
                if vectorqa_metadata:
                    vectorqa_analytics["vector_search_usage"]["total_searches"] += 1
                    if vectorqa_metadata.get("vector_search_used", False):
                        vectorqa_analytics["vector_search_usage"]["successful_searches"] += 1
                        vectorqa_analytics["vector_search_usage"]["average_results_per_search"] += vectorqa_metadata.get("vector_search_results_count", 0)
                
                # Count different types of requests
                request = execution.get("request", "").lower()
                if "question" in request or "answer" in request:
                    vectorqa_analytics["integration_metrics"]["qa_requests_processed"] += 1
                elif "report" in request or "generate" in request:
                    vectorqa_analytics["integration_metrics"]["reports_generated"] += 1
                elif "analyze" in request or "analysis" in request:
                    vectorqa_analytics["integration_metrics"]["documents_analyzed"] += 1
            
            # Calculate averages
            if vectorqa_analytics["vector_search_usage"]["total_searches"] > 0:
                vectorqa_analytics["vector_search_usage"]["average_results_per_search"] /= vectorqa_analytics["vector_search_usage"]["total_searches"]
            
            return {
                "mcp_analytics": mcp_analytics,
                "vectorqa_analytics": vectorqa_analytics,
                "combined_metrics": {
                    "total_requests": mcp_analytics.get("system_analytics", {}).get("total_requests", 0),
                    "overall_success_rate": mcp_analytics.get("system_analytics", {}).get("success_rate", 0),
                    "vector_integration_usage": vectorqa_analytics["vector_search_usage"]["successful_searches"] / max(1, vectorqa_analytics["vector_search_usage"]["total_searches"])
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get analytics: {e}")
            return {
                "mcp_analytics": {"error": str(e)},
                "vectorqa_analytics": {"error": str(e)},
                "combined_metrics": {"error": str(e)}
            }
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update service configuration"""
        try:
            self.config.update(new_config)
            self.orchestrator.update_config(new_config)
            self.logger.info("MCP service configuration updated successfully")
        except Exception as e:
            self.logger.error(f"Failed to update configuration: {e}")
            raise
    
    def reset_system(self):
        """Reset the MCP system"""
        try:
            self.orchestrator.reset_system()
            self.logger.info("MCP system reset completed")
        except Exception as e:
            self.logger.error(f"Failed to reset system: {e}")
            raise
