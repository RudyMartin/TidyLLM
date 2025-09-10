"""
Extraction Worker - Document Content Extraction Agent
=====================================================

Dedicated worker for document content extraction and parsing operations.
Extracts functionality from existing extraction components and organizes
it as a scalable agent worker.

Capabilities:
- PDF document extraction with smart chunking
- Text processing and normalization
- Structured content extraction
- Blank page detection and filtering
- Unicode normalization and cleaning
- Multi-format document support
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import io

from .base_worker import BaseWorker, TaskPriority
from ..session.unified import UnifiedSessionManager

logger = logging.getLogger("extraction_worker")


@dataclass
class ExtractionRequest:
    """Request for document extraction."""
    document_id: str
    content_source: Union[str, bytes, io.BytesIO]  # File path, raw content, or stream
    document_type: str = "pdf"  # pdf, txt, docx, etc.
    extraction_options: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extraction_options is None:
            self.extraction_options = {}


@dataclass 
class ExtractionResult:
    """Result from document extraction."""
    document_id: str
    extracted_text: str
    chunks: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    page_count: Optional[int] = None
    extraction_time: Optional[float] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_id": self.document_id,
            "extracted_text": self.extracted_text,
            "chunks": self.chunks,
            "metadata": self.metadata,
            "page_count": self.page_count,
            "extraction_time": self.extraction_time,
            "warnings": self.warnings,
            "text_length": len(self.extracted_text),
            "chunk_count": len(self.chunks)
        }


class ExtractionWorker(BaseWorker[ExtractionRequest, ExtractionResult]):
    """
    Worker for document content extraction operations.
    
    Integrates existing extraction capabilities:
    - EnhancedDocumentExtractor for advanced PDF processing
    - TextExtractor from vectorqa for basic text extraction
    - Smart chunking and content normalization
    - S3 document retrieval and processing
    
    Task Types:
    - extract_document: Extract content from document
    - extract_s3_document: Extract from S3-stored document
    - extract_chunks: Extract and chunk document content
    - extract_metadata: Extract document metadata only
    """
    
    def __init__(self, 
                 worker_name: str = "extraction_worker",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 min_chunk_size: int = 100,
                 **kwargs):
        """
        Initialize Extraction Worker.
        
        Args:
            worker_name: Worker identifier
            chunk_size: Target chunk size for text splitting
            chunk_overlap: Overlap between chunks
            min_chunk_size: Minimum chunk size (smaller discarded)
        """
        super().__init__(worker_name, **kwargs)
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        # Extraction backends
        self.enhanced_extractor = None
        self.vectorqa_extractor = None
        self.session_manager = None
        
        # Supported document types
        self.supported_types = {"pdf", "txt", "docx", "html", "md"}
        
        logger.info(f"Extraction Worker '{worker_name}' configured")
    
    async def _initialize_worker(self) -> None:
        """Initialize extraction backends."""
        try:
            # Initialize UnifiedSessionManager for S3 access
            try:
                self.session_manager = UnifiedSessionManager()
                logger.info("Extraction Worker: UnifiedSessionManager initialized")
            except Exception as e:
                logger.warning(f"Extraction Worker: UnifiedSessionManager not available: {e}")
            
            # Initialize Enhanced Document Extractor
            try:
                from ...knowledge_systems.core.enhanced_extraction import EnhancedDocumentExtractor
                self.enhanced_extractor = EnhancedDocumentExtractor()
                logger.info("Extraction Worker: EnhancedDocumentExtractor initialized")
            except ImportError as e:
                logger.warning(f"Extraction Worker: EnhancedDocumentExtractor not available: {e}")
            
            # Initialize VectorQA Text Extractor
            try:
                from ...vectorqa.documents.extraction.text import TextExtractor
                self.vectorqa_extractor = TextExtractor()
                logger.info("Extraction Worker: VectorQA TextExtractor initialized")
            except ImportError as e:
                logger.warning(f"Extraction Worker: VectorQA TextExtractor not available: {e}")
            
            if not (self.enhanced_extractor or self.vectorqa_extractor):
                raise RuntimeError("No extraction backends available")
                
        except Exception as e:
            logger.error(f"Extraction Worker initialization failed: {e}")
            raise
    
    def validate_input(self, task_input: Any) -> bool:
        """Validate extraction request input."""
        if not isinstance(task_input, ExtractionRequest):
            return False
        
        if not task_input.document_id:
            return False
        
        if task_input.document_type not in self.supported_types:
            return False
        
        if not task_input.content_source:
            return False
        
        return True
    
    async def process_task(self, task_input: ExtractionRequest) -> ExtractionResult:
        """Process document extraction request."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            logger.info(f"Extracting document '{task_input.document_id}' type '{task_input.document_type}'")
            
            # Route to appropriate extraction method
            if task_input.document_type == "pdf":
                result = await self._extract_pdf(task_input)
            elif task_input.document_type == "txt":
                result = await self._extract_text(task_input)
            else:
                result = await self._extract_generic(task_input)
            
            # Set extraction time
            result.extraction_time = asyncio.get_event_loop().time() - start_time
            
            logger.info(f"Document '{task_input.document_id}' extracted: "
                       f"{len(result.extracted_text)} chars, {len(result.chunks)} chunks")
            
            return result
            
        except Exception as e:
            logger.error(f"Document extraction failed for '{task_input.document_id}': {e}")
            raise
    
    async def _extract_pdf(self, request: ExtractionRequest) -> ExtractionResult:
        """Extract PDF document using enhanced extractor."""
        if not self.enhanced_extractor:
            raise RuntimeError("Enhanced PDF extractor not available")
        
        try:
            # Get document content
            content = await self._get_document_content(request)
            
            # Extract using enhanced extractor
            if isinstance(content, bytes):
                content_stream = io.BytesIO(content)
            else:
                content_stream = content
            
            # Run extraction in thread pool to avoid blocking
            extracted_data = await asyncio.get_event_loop().run_in_executor(
                None, 
                self.enhanced_extractor.extract_from_stream,
                content_stream,
                request.extraction_options
            )
            
            # Process extracted data into chunks
            chunks = await self._create_chunks(
                extracted_data.get("text", ""), 
                request.document_id
            )
            
            return ExtractionResult(
                document_id=request.document_id,
                extracted_text=extracted_data.get("text", ""),
                chunks=chunks,
                metadata=extracted_data.get("metadata", {}),
                page_count=extracted_data.get("page_count"),
                warnings=extracted_data.get("warnings", [])
            )
            
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise
    
    async def _extract_text(self, request: ExtractionRequest) -> ExtractionResult:
        """Extract plain text document."""
        try:
            # Get document content
            content = await self._get_document_content(request)
            
            if isinstance(content, bytes):
                text = content.decode('utf-8', errors='ignore')
            else:
                text = str(content)
            
            # Basic text cleaning
            text = self._normalize_text(text)
            
            # Create chunks
            chunks = await self._create_chunks(text, request.document_id)
            
            return ExtractionResult(
                document_id=request.document_id,
                extracted_text=text,
                chunks=chunks,
                metadata={"document_type": "text", "encoding": "utf-8"}
            )
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            raise
    
    async def _extract_generic(self, request: ExtractionRequest) -> ExtractionResult:
        """Extract generic document using available extractors."""
        if self.vectorqa_extractor:
            return await self._extract_with_vectorqa(request)
        else:
            # Fallback to basic text extraction
            return await self._extract_text(request)
    
    async def _extract_with_vectorqa(self, request: ExtractionRequest) -> ExtractionResult:
        """Extract using VectorQA text extractor."""
        try:
            # Get document content
            content = await self._get_document_content(request)
            
            # Convert to appropriate format for VectorQA
            if isinstance(content, bytes):
                content_stream = io.BytesIO(content)
            else:
                content_stream = content
            
            # Extract using VectorQA
            extracted_data = await asyncio.get_event_loop().run_in_executor(
                None,
                self.vectorqa_extractor.extract,
                content_stream
            )
            
            # Process into standard format
            text = extracted_data if isinstance(extracted_data, str) else str(extracted_data)
            chunks = await self._create_chunks(text, request.document_id)
            
            return ExtractionResult(
                document_id=request.document_id,
                extracted_text=text,
                chunks=chunks,
                metadata={"extractor": "vectorqa", "document_type": request.document_type}
            )
            
        except Exception as e:
            logger.error(f"VectorQA extraction failed: {e}")
            raise
    
    async def _get_document_content(self, request: ExtractionRequest) -> Union[bytes, io.BytesIO]:
        """Get document content from various sources."""
        content_source = request.content_source
        
        if isinstance(content_source, bytes):
            return content_source
        elif isinstance(content_source, io.BytesIO):
            return content_source
        elif isinstance(content_source, str):
            # Check if it's S3 path or file path
            if content_source.startswith("s3://"):
                return await self._get_s3_content(content_source)
            else:
                # File path
                return await self._get_file_content(content_source)
        else:
            raise ValueError(f"Unsupported content source type: {type(content_source)}")
    
    async def _get_s3_content(self, s3_path: str) -> bytes:
        """Get document content from S3."""
        if not self.session_manager:
            raise RuntimeError("S3 access requires UnifiedSessionManager")
        
        try:
            # Parse S3 path
            parts = s3_path.replace("s3://", "").split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""
            
            # Get S3 client
            s3_client = self.session_manager.get_s3_client()
            
            # Download content
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: s3_client.get_object(Bucket=bucket, Key=key)
            )
            
            return response['Body'].read()
            
        except Exception as e:
            logger.error(f"S3 content retrieval failed for '{s3_path}': {e}")
            raise
    
    async def _get_file_content(self, file_path: str) -> bytes:
        """Get document content from file system."""
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            return await asyncio.get_event_loop().run_in_executor(
                None,
                path.read_bytes
            )
            
        except Exception as e:
            logger.error(f"File content retrieval failed for '{file_path}': {e}")
            raise
    
    async def _create_chunks(self, text: str, document_id: str) -> List[Dict[str, Any]]:
        """Create text chunks from extracted text."""
        if not text or len(text) < self.min_chunk_size:
            return []
        
        chunks = []
        text_length = len(text)
        
        for i in range(0, text_length, self.chunk_size - self.chunk_overlap):
            # Extract chunk
            chunk_end = min(i + self.chunk_size, text_length)
            chunk_text = text[i:chunk_end]
            
            # Skip if chunk is too small
            if len(chunk_text) < self.min_chunk_size:
                continue
            
            # Create chunk metadata
            chunk = {
                "chunk_id": f"{document_id}_chunk_{len(chunks)}",
                "document_id": document_id,
                "text": chunk_text,
                "start_offset": i,
                "end_offset": chunk_end,
                "chunk_index": len(chunks),
                "text_length": len(chunk_text)
            }
            
            chunks.append(chunk)
        
        return chunks
    
    def _normalize_text(self, text: str) -> str:
        """Normalize and clean extracted text."""
        if not text:
            return ""
        
        # Basic normalization
        import unicodedata
        text = unicodedata.normalize('NFKD', text)
        
        # Remove excessive whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    # Task submission convenience methods
    async def extract_document(self, 
                              document_id: str,
                              content_source: Union[str, bytes, io.BytesIO],
                              document_type: str = "pdf",
                              extraction_options: Dict[str, Any] = None,
                              priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """
        Submit document extraction task.
        
        Returns:
            Task ID for tracking
        """
        request = ExtractionRequest(
            document_id=document_id,
            content_source=content_source,
            document_type=document_type,
            extraction_options=extraction_options or {}
        )
        
        task = await self.submit_task(
            task_type="extract_document",
            task_input=request,
            priority=priority
        )
        
        return task.task_id
    
    async def extract_s3_document(self,
                                 document_id: str,
                                 s3_path: str,
                                 document_type: str = "pdf",
                                 priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """
        Submit S3 document extraction task.
        
        Returns:
            Task ID for tracking
        """
        return await self.extract_document(
            document_id=document_id,
            content_source=s3_path,
            document_type=document_type,
            priority=priority
        )