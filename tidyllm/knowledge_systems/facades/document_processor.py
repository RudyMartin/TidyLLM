"""
DocumentProcessor Facade  
=======================

Simple interface for document processing with smart chunking and extraction.
Provides easy API over the complex document extraction and enhanced processing backend.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import logging
import io

try:
    from tidyllm.vectorqa.documents.extraction.text import TextExtractor
    VECTORQA_EXTRACTOR_AVAILABLE = True
except ImportError:
    VECTORQA_EXTRACTOR_AVAILABLE = False

try:
    from ..core.enhanced_extraction import EnhancedDocumentExtractor
    ENHANCED_EXTRACTOR_AVAILABLE = True
except ImportError:
    ENHANCED_EXTRACTOR_AVAILABLE = False

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Simple facade for document processing with smart chunking"""
    
    def __init__(self, 
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 200,
                 min_chunk_size: int = 100):
        """
        Initialize document processor
        
        Args:
            chunk_size: Target size for text chunks
            chunk_overlap: Overlap between consecutive chunks
            min_chunk_size: Minimum chunk size (smaller chunks discarded)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap  
        self.min_chunk_size = min_chunk_size
        
        # Initialize backend extractors
        self.vectorqa_extractor = None
        self.enhanced_extractor = None
        
        if VECTORQA_EXTRACTOR_AVAILABLE:
            self.vectorqa_extractor = TextExtractor()
            logger.info("VectorQA TextExtractor initialized")
            
        if ENHANCED_EXTRACTOR_AVAILABLE:
            self.enhanced_extractor = EnhancedDocumentExtractor()
            logger.info("Enhanced document extractor initialized")
        
        if not (VECTORQA_EXTRACTOR_AVAILABLE or ENHANCED_EXTRACTOR_AVAILABLE):
            logger.warning("No document extractors available - using basic text processing")
    
    def process_text(self, text: str, source: str = "direct_input") -> Dict[str, Any]:
        """
        Process plain text into chunks with metadata
        
        Args:
            text: Text content to process
            source: Source identifier for metadata
            
        Returns:
            Dictionary with chunks, metadata, and processing info
        """
        if not text or not text.strip():
            return {
                "chunks": [],
                "metadata": {"error": "Empty text provided"},
                "source": source,
                "total_chunks": 0
            }
        
        try:
            # Use smart chunking from existing backend
            chunks = self._smart_chunk_text(text)
            
            # Filter out chunks that are too small
            filtered_chunks = [chunk for chunk in chunks if len(chunk.strip()) >= self.min_chunk_size]
            
            return {
                "chunks": filtered_chunks,
                "metadata": {
                    "original_length": len(text),
                    "total_chunks": len(filtered_chunks),
                    "avg_chunk_size": sum(len(c) for c in filtered_chunks) / len(filtered_chunks) if filtered_chunks else 0,
                    "chunk_size_config": self.chunk_size,
                    "overlap_config": self.chunk_overlap,
                    "processing_method": self._get_processing_method()
                },
                "source": source,
                "total_chunks": len(filtered_chunks)
            }
            
        except Exception as e:
            logger.error(f"Text processing failed: {e}")
            return {
                "chunks": [],
                "metadata": {"error": str(e)},
                "source": source,
                "total_chunks": 0
            }
    
    def process_document(self, file_path: Union[str, Path], max_pages: int = None) -> Dict[str, Any]:
        """
        Process document file into chunks with metadata
        
        Args:
            file_path: Path to document file
            max_pages: Maximum pages to process (for PDF/DOCX)
            
        Returns:
            Dictionary with chunks, metadata, and processing info
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {
                "chunks": [],
                "metadata": {"error": f"File not found: {file_path}"},
                "source": str(file_path),
                "total_chunks": 0
            }
        
        try:
            # Extract text using existing backend
            text_content, extraction_metadata = self._extract_document_text(file_path, max_pages)
            
            if not text_content:
                return {
                    "chunks": [],
                    "metadata": {"error": "No text extracted", "extraction_metadata": extraction_metadata},
                    "source": str(file_path),
                    "total_chunks": 0
                }
            
            # Process extracted text
            result = self.process_text(text_content, str(file_path))
            
            # Add extraction metadata
            result["metadata"]["extraction_metadata"] = extraction_metadata
            result["metadata"]["file_info"] = {
                "file_path": str(file_path),
                "file_size": file_path.stat().st_size,
                "file_extension": file_path.suffix.lower()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Document processing failed for {file_path}: {e}")
            return {
                "chunks": [],
                "metadata": {"error": str(e)},
                "source": str(file_path),
                "total_chunks": 0
            }
    
    def process_s3_content(self, content_bytes: bytes, filename: str) -> Dict[str, Any]:
        """
        Process document content from S3 bytes
        
        Args:
            content_bytes: Document content as bytes
            filename: Original filename for format detection
            
        Returns:
            Dictionary with chunks, metadata, and processing info
        """
        try:
            # Use enhanced extractor for S3 content if available
            if self.enhanced_extractor and hasattr(self.enhanced_extractor, 'extract_from_s3_content'):
                text_content = self.enhanced_extractor.extract_from_s3_content(content_bytes, filename)
                extraction_method = "enhanced_s3_extraction"
            else:
                # Fallback to basic text extraction
                text_content = self._extract_from_bytes(content_bytes, filename)
                extraction_method = "basic_bytes_extraction"
            
            # Process extracted text
            result = self.process_text(text_content, f"s3://{filename}")
            result["metadata"]["extraction_method"] = extraction_method
            
            return result
            
        except Exception as e:
            logger.error(f"S3 content processing failed for {filename}: {e}")
            return {
                "chunks": [],
                "metadata": {"error": str(e), "filename": filename},
                "source": f"s3://{filename}",
                "total_chunks": 0
            }
    
    def _smart_chunk_text(self, text: str) -> List[str]:
        """Use smart chunking from existing backend"""
        if self.vectorqa_extractor and hasattr(self.vectorqa_extractor, 'smart_chunking'):
            # Use enhanced smart chunking from VectorQA TextExtractor
            return self.vectorqa_extractor.smart_chunking(text, self.chunk_size)
        elif self.enhanced_extractor and hasattr(self.enhanced_extractor, 'smart_chunking'):
            # Use enhanced extraction smart chunking
            return self.enhanced_extractor.smart_chunking(text, self.chunk_size)
        else:
            # Fallback to basic chunking
            return self._basic_chunk_text(text)
    
    def _extract_document_text(self, file_path: Path, max_pages: Optional[int]) -> Tuple[str, Dict]:
        """Extract text from document using existing backend"""
        if self.vectorqa_extractor:
            # Use VectorQA TextExtractor (your enhanced version)
            return self.vectorqa_extractor.extract_text(str(file_path), max_pages or 5)
        else:
            # Fallback extraction
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                return text, {"method": "basic_text_read"}
            except Exception as e:
                return "", {"error": str(e)}
    
    def _extract_from_bytes(self, content_bytes: bytes, filename: str) -> str:
        """Basic extraction from bytes"""
        try:
            # Try UTF-8 decoding first
            return content_bytes.decode('utf-8')
        except UnicodeDecodeError:
            try:
                # Try latin-1 as fallback
                return content_bytes.decode('latin-1')
            except Exception:
                return ""
    
    def _basic_chunk_text(self, text: str) -> List[str]:
        """Basic fallback chunking when no smart chunking available"""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            if end >= len(text):
                # Last chunk
                chunks.append(text[start:])
                break
            
            # Try to break at sentence or word boundary
            chunk_text = text[start:end]
            
            # Look for sentence endings near the end
            for i in range(len(chunk_text) - 1, max(0, len(chunk_text) - 100), -1):
                if chunk_text[i] in '.!?':
                    chunk_text = chunk_text[:i + 1]
                    break
            
            chunks.append(chunk_text)
            start = start + len(chunk_text) - self.chunk_overlap
        
        return chunks
    
    def _get_processing_method(self) -> str:
        """Get description of processing method used"""
        if self.vectorqa_extractor and hasattr(self.vectorqa_extractor, 'smart_chunking'):
            return "vectorqa_smart_chunking"
        elif self.enhanced_extractor:
            return "enhanced_extraction"
        else:
            return "basic_processing"
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported document formats
        
        Returns:
            List of supported file extensions
        """
        formats = ['.txt', '.text']
        
        if self.vectorqa_extractor:
            if hasattr(self.vectorqa_extractor, 'pdf_available') and self.vectorqa_extractor.pdf_available:
                formats.append('.pdf')
            if hasattr(self.vectorqa_extractor, 'docx_available') and self.vectorqa_extractor.docx_available:
                formats.append('.docx')
        
        return formats
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check health of document processor
        
        Returns:
            Health status dictionary
        """
        status = {
            "vectorqa_extractor_available": VECTORQA_EXTRACTOR_AVAILABLE,
            "enhanced_extractor_available": ENHANCED_EXTRACTOR_AVAILABLE,
            "supported_formats": self.get_supported_formats(),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "processing_method": self._get_processing_method(),
            "status": "healthy"
        }
        
        # Test with simple text processing
        try:
            test_result = self.process_text("This is a test document for health check.")
            status["test_chunks"] = test_result["total_chunks"]
            status["test_passed"] = test_result["total_chunks"] > 0
        except Exception as e:
            status["status"] = "degraded"
            status["error"] = str(e)
            status["test_passed"] = False
        
        return status


# Convenience functions for quick access
def process_text(text: str, chunk_size: int = 1000) -> List[str]:
    """
    Quick function to process text into chunks
    
    Args:
        text: Text to process
        chunk_size: Target chunk size
        
    Returns:
        List of text chunks
    """
    processor = DocumentProcessor(chunk_size=chunk_size)
    result = processor.process_text(text)
    return result["chunks"]


def process_document(file_path: Union[str, Path], chunk_size: int = 1000) -> List[str]:
    """
    Quick function to process document into chunks
    
    Args:
        file_path: Path to document
        chunk_size: Target chunk size
        
    Returns:
        List of text chunks
    """
    processor = DocumentProcessor(chunk_size=chunk_size)
    result = processor.process_document(file_path)
    return result["chunks"]