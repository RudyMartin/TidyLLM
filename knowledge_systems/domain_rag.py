"""
Domain RAG System
=================

Domain-specific Retrieval Augmented Generation system that combines:
- S3 document storage and management
- Vector database for semantic search  
- Domain-specific knowledge processing
- RAG query and answer generation

Each domain RAG represents a specialized knowledge area (e.g., Model Validation, 
Legal Documents, Technical Manuals) with optimized processing and retrieval.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict

try:
    from ..infrastructure.session import S3Manager, UploadResult
except ImportError:
    class S3Manager:
        def __init__(self, *args, **kwargs): pass
        def upload_file(self, *args, **kwargs): return type('obj', (object,), {"success": False, "error": "Mock S3Manager"})()
    class UploadResult:
        def __init__(self, *args, **kwargs): pass
from .vector_manager import VectorManager, Document, SearchResult

logger = logging.getLogger("domain_rag")

@dataclass
class DomainRAGConfig:
    """Configuration for domain-specific RAG system"""
    domain_name: str
    description: str = ""
    s3_bucket: Optional[str] = None
    s3_prefix: str = ""
    vector_config: Optional[Dict[str, Any]] = None
    processing_config: Optional[Dict[str, Any]] = None
    metadata_schema: Optional[Dict[str, Any]] = None

@dataclass
class ProcessedDocument:
    """Document after domain-specific processing"""
    source_file: str
    document_id: str
    title: str
    content: str
    extracted_metadata: Dict[str, Any]
    chunks_created: int
    processing_time: float
    s3_location: Optional[str] = None
    success: bool = True
    error: Optional[str] = None

@dataclass
class RAGQuery:
    """RAG query request"""
    query: str
    domain_context: Optional[str] = None
    max_results: int = 5
    similarity_threshold: float = 0.7
    include_sources: bool = True

@dataclass
class RAGResponse:
    """RAG query response"""
    query: str
    answer: str
    sources: List[SearchResult]
    confidence: float
    processing_time: float
    metadata: Dict[str, Any]

class DomainRAG:
    """Domain-specific RAG system implementation"""
    
    def __init__(self, config: DomainRAGConfig, s3_manager: S3Manager = None, 
                 vector_manager: VectorManager = None):
        self.config = config
        self.s3_manager = s3_manager
        self.vector_manager = vector_manager
        
        # Initialize processing stats
        self.stats = {
            "documents_processed": 0,
            "total_chunks": 0,
            "last_updated": None,
            "processing_errors": 0
        }
        
        logger.info(f"Initialized DomainRAG for: {config.domain_name}")
    
    def process_document(self, file_path: Union[str, Path], 
                        metadata: Dict[str, Any] = None) -> ProcessedDocument:
        """Process a single document for the domain"""
        start_time = datetime.now()
        file_path = Path(file_path)
        
        try:
            # Read document content
            content = self._extract_content(file_path)
            if not content:
                return ProcessedDocument(
                    source_file=str(file_path),
                    document_id="",
                    title="",
                    content="",
                    extracted_metadata={},
                    chunks_created=0,
                    processing_time=0,
                    success=False,
                    error="Could not extract content from file"
                )
            
            # Extract title and metadata
            title = self._extract_title(file_path, content)
            extracted_metadata = self._extract_metadata(content, metadata or {})
            
            # Create document in vector database
            document = Document(
                title=title,
                content=content,
                source=str(file_path),
                doc_type=file_path.suffix.lower().lstrip('.'),
                metadata={
                    **extracted_metadata,
                    "domain": self.config.domain_name,
                    "processing_timestamp": datetime.now().isoformat()
                }
            )
            
            # Add to vector database
            if self.vector_manager:
                add_result = self.vector_manager.add_document(document)
                if not add_result["success"]:
                    raise Exception(f"Failed to add to vector DB: {add_result['error']}")
                
                document_id = add_result["document_id"]
                
                # Add chunks with embeddings
                chunk_result = self.vector_manager.add_document_chunks(document_id, content)
                if not chunk_result["success"]:
                    raise Exception(f"Failed to add chunks: {chunk_result['error']}")
                
                chunks_created = chunk_result["chunks_added"]
            else:
                document_id = f"mock_{hash(content)}"
                chunks_created = len(content) // 512  # Rough estimate
            
            # Upload to S3 if manager available
            s3_location = None
            if self.s3_manager:
                s3_key = f"{self.config.s3_prefix}{self.config.domain_name}/{file_path.name}"
                upload_result = self.s3_manager.upload_file(
                    file_path, 
                    bucket=self.config.s3_bucket,
                    s3_key=s3_key,
                    metadata={
                        "domain": self.config.domain_name,
                        "document_id": document_id,
                        "title": title
                    }
                )
                if upload_result.success:
                    s3_location = upload_result.s3_url
            
            # Update stats
            self.stats["documents_processed"] += 1
            self.stats["total_chunks"] += chunks_created
            self.stats["last_updated"] = datetime.now().isoformat()
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessedDocument(
                source_file=str(file_path),
                document_id=document_id,
                title=title,
                content=content,
                extracted_metadata=extracted_metadata,
                chunks_created=chunks_created,
                processing_time=processing_time,
                s3_location=s3_location,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            self.stats["processing_errors"] += 1
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessedDocument(
                source_file=str(file_path),
                document_id="",
                title=file_path.name,
                content="",
                extracted_metadata={},
                chunks_created=0,
                processing_time=processing_time,
                success=False,
                error=str(e)
            )
    
    def process_directory(self, directory_path: Union[str, Path], 
                         file_patterns: List[str] = None) -> List[ProcessedDocument]:
        """Process all documents in a directory"""
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            logger.error(f"Directory not found: {directory_path}")
            return []
        
        # Default patterns for common document types
        if not file_patterns:
            file_patterns = ["*.pdf", "*.txt", "*.md", "*.docx", "*.html"]
        
        # Find all matching files
        files_to_process = []
        for pattern in file_patterns:
            files_to_process.extend(directory_path.glob(pattern))
        
        logger.info(f"Found {len(files_to_process)} files to process in {directory_path}")
        
        # Process each file
        results = []
        for file_path in files_to_process:
            result = self.process_document(file_path)
            results.append(result)
            
            if result.success:
                logger.info(f"✅ Processed: {file_path.name} ({result.chunks_created} chunks)")
            else:
                logger.error(f"❌ Failed: {file_path.name} - {result.error}")
        
        return results
    
    def query(self, rag_query: RAGQuery) -> RAGResponse:
        """Execute RAG query against domain knowledge"""
        start_time = datetime.now()
        
        try:
            # Search for relevant documents
            if self.vector_manager:
                search_results = self.vector_manager.search_similar(
                    rag_query.query,
                    top_k=rag_query.max_results,
                    similarity_threshold=rag_query.similarity_threshold
                )
            else:
                # Mock results for testing
                search_results = [SearchResult(
                    document_id="mock_doc_1",
                    chunk_id="mock_chunk_1", 
                    content=f"Mock result for domain '{self.config.domain_name}' query: {rag_query.query}",
                    score=0.85,
                    metadata={"domain": self.config.domain_name}
                )]
            
            # Generate answer from search results
            answer = self._generate_answer(rag_query.query, search_results, rag_query.domain_context)
            
            # Calculate confidence based on search scores and result quality
            confidence = self._calculate_confidence(search_results, answer)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return RAGResponse(
                query=rag_query.query,
                answer=answer,
                sources=search_results if rag_query.include_sources else [],
                confidence=confidence,
                processing_time=processing_time,
                metadata={
                    "domain": self.config.domain_name,
                    "results_count": len(search_results),
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return RAGResponse(
                query=rag_query.query,
                answer=f"I apologize, but I encountered an error processing your query: {str(e)}",
                sources=[],
                confidence=0.0,
                processing_time=processing_time,
                metadata={
                    "domain": self.config.domain_name,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    def _extract_content(self, file_path: Path) -> str:
        """Extract text content from file"""
        try:
            if file_path.suffix.lower() == '.txt':
                return file_path.read_text(encoding='utf-8', errors='ignore')
            elif file_path.suffix.lower() == '.md':
                return file_path.read_text(encoding='utf-8', errors='ignore')
            elif file_path.suffix.lower() == '.html':
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                # Basic HTML tag removal (for basic processing)
                import re
                return re.sub(r'<[^>]+>', '', content)
            else:
                # For other formats, try to read as text
                return file_path.read_text(encoding='utf-8', errors='ignore')
        except Exception as e:
            logger.warning(f"Could not extract content from {file_path}: {e}")
            return ""
    
    def _extract_title(self, file_path: Path, content: str) -> str:
        """Extract title from document"""
        # Try to find title in content (first line or heading)
        lines = content.split('\n')
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if line and len(line) < 200:  # Reasonable title length
                # Remove common markdown/markup
                title = line.lstrip('#').strip()
                if title:
                    return title
        
        # Fallback to filename
        return file_path.stem.replace('_', ' ').replace('-', ' ').title()
    
    def _extract_metadata(self, content: str, base_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract domain-specific metadata from content"""
        metadata = base_metadata.copy()
        
        # Basic content analysis
        metadata.update({
            "content_length": len(content),
            "word_count": len(content.split()),
            "line_count": len(content.split('\n')),
            "domain": self.config.domain_name
        })
        
        # Domain-specific extraction based on config
        if self.config.metadata_schema:
            # Apply domain-specific metadata extraction rules
            for field, rules in self.config.metadata_schema.items():
                if isinstance(rules, dict) and "pattern" in rules:
                    import re
                    pattern = rules["pattern"]
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        metadata[field] = matches[0] if len(matches) == 1 else matches
        
        return metadata
    
    def _generate_answer(self, query: str, search_results: List[SearchResult], 
                        context: Optional[str] = None) -> str:
        """Generate answer from search results"""
        if not search_results:
            return f"I don't have specific information about '{query}' in the {self.config.domain_name} domain."
        
        # Combine relevant context
        relevant_content = []
        for result in search_results:
            relevant_content.append(f"({result.score:.2f}) {result.content}")
        
        context_text = "\n\n".join(relevant_content)
        
        # Generate answer (this would typically use an LLM)
        # For now, provide a structured response
        answer = f"Based on the {self.config.domain_name} domain knowledge:\n\n"
        
        if len(search_results) == 1:
            answer += f"According to the available information: {search_results[0].content}"
        else:
            answer += f"From {len(search_results)} relevant sources, the key information is:\n"
            for i, result in enumerate(search_results[:3], 1):
                snippet = result.content[:200] + "..." if len(result.content) > 200 else result.content
                answer += f"\n{i}. {snippet} (relevance: {result.score:.2f})"
        
        return answer
    
    def _calculate_confidence(self, search_results: List[SearchResult], answer: str) -> float:
        """Calculate confidence score for RAG response"""
        if not search_results:
            return 0.0
        
        # Base confidence on average search scores
        avg_score = sum(result.score for result in search_results) / len(search_results)
        
        # Adjust based on number of results
        result_factor = min(len(search_results) / 3.0, 1.0)  # Optimal at 3+ results
        
        # Adjust based on answer quality indicators
        answer_factor = 1.0
        if "I don't have" in answer or "I apologize" in answer:
            answer_factor = 0.3
        
        confidence = avg_score * result_factor * answer_factor
        return min(confidence, 1.0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get domain RAG statistics"""
        stats = self.stats.copy()
        stats.update({
            "domain_name": self.config.domain_name,
            "description": self.config.description,
            "s3_bucket": self.config.s3_bucket,
            "s3_prefix": self.config.s3_prefix
        })
        
        # Add vector database stats if available
        if self.vector_manager:
            vector_stats = self.vector_manager.get_database_stats()
            if vector_stats["success"]:
                stats["vector_stats"] = vector_stats
        
        return stats
    
    def export_config(self) -> Dict[str, Any]:
        """Export domain RAG configuration"""
        return asdict(self.config)
    
    # Generic convenience methods for onboarding/demo compatibility
    def add_document(self, content: Union[str, bytes], filename: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generic wrapper for document processing"""
        # Create a temporary file if we have bytes/string content
        if isinstance(content, (str, bytes)):
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(mode='w' if isinstance(content, str) else 'wb', 
                                             suffix=Path(filename).suffix, 
                                             delete=False) as f:
                f.write(content)
                temp_path = f.name
            
            try:
                result = self.process_document(temp_path, metadata)
                return {
                    "success": True,
                    "filename": filename,
                    "processed": result.document_id,
                    "chunks": len(result.chunks) if hasattr(result, 'chunks') else 0
                }
            finally:
                os.unlink(temp_path)
        else:
            # Assume it's already a path
            result = self.process_document(content, metadata)
            return {
                "success": True,
                "filename": filename,
                "processed": result.document_id,
                "chunks": len(result.chunks) if hasattr(result, 'chunks') else 0
            }
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Generic search wrapper"""
        from .rag_query import RAGQuery
        rag_query = RAGQuery(query=query, top_k=top_k)
        response = self.query(rag_query)
        
        # Convert to simple format for onboarding
        results = []
        for result in getattr(response, 'search_results', []):
            results.append({
                "content": result.content[:200] if hasattr(result, 'content') else str(result)[:200],
                "score": getattr(result, 'score', 0.0),
                "metadata": getattr(result, 'metadata', {})
            })
        return results
    
    def retrain_vectors(self) -> Dict[str, Any]:
        """Mock vector retraining - not implemented in core DomainRAG"""
        return {
            "success": True,
            "message": "Vector retraining not implemented in core DomainRAG",
            "stats": self.get_stats()
        }