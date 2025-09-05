"""
Enhanced Extraction for S3-First Domain RAG
===========================================

Extracts the core smart parsing logic from the ZIP file and adapts it 
for our clean S3-first architecture, removing old metadata dependencies.

Key improvements over basic text extraction:
- Smart chunking with continuity validation
- Direct S3 processing without temp files
- Blank page detection
- Unicode normalization
- Structured output for vector storage
"""

import re
import unicodedata
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import nltk
    # Download required data if not present
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

logger = logging.getLogger("enhanced_extraction")

class EnhancedDocumentExtractor:
    """Enhanced document extraction adapted for S3-first architecture"""
    
    def __init__(self):
        self.pdf_available = PDF_AVAILABLE
        self.nltk_available = NLTK_AVAILABLE
        
        if not PDF_AVAILABLE:
            logger.warning("PyPDF2 not available - PDF extraction limited")
        if not NLTK_AVAILABLE:
            logger.warning("NLTK not available - using basic text processing")
    
    def clean_text(self, text: str) -> str:
        """
        Advanced text cleaning adapted from ZIP file
        Removes old metadata dependencies
        """
        if not text:
            return ""
        
        # Unicode normalization (from ZIP)
        text = unicodedata.normalize('NFKD', text)
        
        # Remove control characters except newlines and tabs
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\t')
        
        # Clean up whitespace (from ZIP logic)
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Clean paragraph breaks
        text = text.strip()
        
        return text
    
    def smart_chunking(self, text: str, max_words: int = 200, overlap_words: int = 20) -> List[Dict[str, Any]]:
        """
        Smart text chunking adapted from ZIP file
        Returns structured chunks suitable for S3-first vector storage
        """
        if not text:
            return []
        
        words = text.split()
        if len(words) <= max_words:
            return [{
                "chunk_index": 0,
                "content": text,
                "word_count": len(words),
                "start_word": 0,
                "end_word": len(words),
                "overlap_info": None
            }]
        
        chunks = []
        start_idx = 0
        chunk_index = 0
        
        while start_idx < len(words):
            # Calculate chunk boundaries
            end_idx = min(start_idx + max_words, len(words))
            chunk_words = words[start_idx:end_idx]
            chunk_text = ' '.join(chunk_words)
            
            # Find natural break point (sentence boundary)
            if end_idx < len(words) and self.nltk_available:
                chunk_text = self._find_sentence_boundary(chunk_text, words[end_idx:end_idx+20])
                actual_word_count = len(chunk_text.split())
            else:
                actual_word_count = len(chunk_words)
            
            chunk_data = {
                "chunk_index": chunk_index,
                "content": self.clean_text(chunk_text),
                "word_count": actual_word_count,
                "start_word": start_idx,
                "end_word": start_idx + actual_word_count,
                "overlap_info": {
                    "prev_overlap": overlap_words if chunk_index > 0 else 0,
                    "next_overlap": overlap_words if end_idx < len(words) else 0
                }
            }
            
            chunks.append(chunk_data)
            
            # Move to next chunk with overlap
            start_idx = max(start_idx + max_words - overlap_words, start_idx + 1)
            chunk_index += 1
        
        return chunks
    
    def _find_sentence_boundary(self, chunk_text: str, next_words: List[str]) -> str:
        """Find natural sentence boundary to improve chunking quality"""
        if not self.nltk_available:
            return chunk_text
        
        try:
            # Add some next words to find sentence boundary
            extended_text = chunk_text + ' ' + ' '.join(next_words[:20])
            sentences = nltk.sent_tokenize(extended_text)
            
            # Find last complete sentence that fits in chunk
            accumulated = ""
            for sentence in sentences:
                test_text = accumulated + (' ' if accumulated else '') + sentence
                if len(test_text.split()) <= len(chunk_text.split()) + 10:  # Small buffer
                    accumulated = test_text
                else:
                    break
            
            return accumulated if accumulated else chunk_text
            
        except Exception:
            return chunk_text
    
    def is_blank_page(self, text: str, min_length: int = 10) -> bool:
        """
        Detect blank or near-empty pages (from ZIP logic)
        Adapted to remove old metadata dependencies
        """
        if not text:
            return True
        
        cleaned = self.clean_text(text)
        
        # Check various blank page indicators
        if len(cleaned) < min_length:
            return True
        
        # Check for pages with only whitespace, numbers, or common PDF artifacts
        content_words = [word for word in cleaned.split() 
                        if not re.match(r'^\d+$', word)  # Page numbers
                        and len(word) > 2]  # Short artifacts
        
        return len(content_words) < 3
    
    def extract_from_s3_content(self, content: bytes, filename: str = "document") -> Dict[str, Any]:
        """
        Extract text directly from S3 content bytes
        No temp files, clean output for vector storage
        """
        result = {
            "filename": filename,
            "success": False,
            "text": "",
            "chunks": [],
            "metadata": {
                "content_length": len(content),
                "processing_method": "s3_direct",
                "extractor": "enhanced"
            }
        }
        
        try:
            if filename.lower().endswith('.pdf'):
                result = self._extract_pdf_from_bytes(content, filename)
            elif filename.lower().endswith('.txt'):
                result = self._extract_text_from_bytes(content, filename)
            else:
                # Try as text first
                try:
                    text = content.decode('utf-8', errors='ignore')
                    result = self._extract_text_from_bytes(text.encode(), filename)
                except Exception:
                    result["error"] = f"Unsupported file type: {filename}"
        
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Extraction failed for {filename}: {e}")
        
        return result
    
    def _extract_pdf_from_bytes(self, content: bytes, filename: str) -> Dict[str, Any]:
        """Extract text from PDF bytes (adapted from ZIP logic)"""
        if not self.pdf_available:
            return {
                "filename": filename,
                "success": False,
                "error": "PyPDF2 not available",
                "text": "",
                "chunks": []
            }
        
        try:
            import io
            pdf_file = io.BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            all_text = ""
            page_texts = []
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    cleaned_text = self.clean_text(page_text)
                    
                    # Skip blank pages
                    if not self.is_blank_page(cleaned_text):
                        page_texts.append(cleaned_text)
                        all_text += cleaned_text + "\n\n"
                    
                except Exception as e:
                    logger.warning(f"Failed to extract page {page_num} from {filename}: {e}")
            
            # Smart chunking of combined text
            chunks = self.smart_chunking(all_text.strip()) if all_text.strip() else []
            
            return {
                "filename": filename,
                "success": True,
                "text": all_text.strip(),
                "chunks": chunks,
                "metadata": {
                    "total_pages": len(pdf_reader.pages),
                    "processed_pages": len(page_texts),
                    "blank_pages_skipped": len(pdf_reader.pages) - len(page_texts),
                    "total_chunks": len(chunks),
                    "processing_method": "enhanced_pdf_extraction"
                }
            }
            
        except Exception as e:
            return {
                "filename": filename,
                "success": False,
                "error": f"PDF processing failed: {str(e)}",
                "text": "",
                "chunks": []
            }
    
    def _extract_text_from_bytes(self, content: bytes, filename: str) -> Dict[str, Any]:
        """Extract text from text file bytes"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            text = None
            
            for encoding in encodings:
                try:
                    text = content.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if text is None:
                text = content.decode('utf-8', errors='ignore')
            
            cleaned_text = self.clean_text(text)
            chunks = self.smart_chunking(cleaned_text) if cleaned_text else []
            
            return {
                "filename": filename,
                "success": True,
                "text": cleaned_text,
                "chunks": chunks,
                "metadata": {
                    "original_length": len(text),
                    "cleaned_length": len(cleaned_text),
                    "total_chunks": len(chunks),
                    "processing_method": "enhanced_text_extraction"
                }
            }
            
        except Exception as e:
            return {
                "filename": filename,
                "success": False,
                "error": f"Text processing failed: {str(e)}",
                "text": "",
                "chunks": []
            }
    
    def validate_extraction_quality(self, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate extraction quality and provide feedback
        Clean implementation without old metadata dependencies
        """
        if not extraction_result.get("success", False):
            return {
                "quality_score": 0.0,
                "issues": ["Extraction failed"],
                "recommendations": ["Check file format and content"]
            }
        
        text = extraction_result.get("text", "")
        chunks = extraction_result.get("chunks", [])
        
        issues = []
        recommendations = []
        quality_factors = []
        
        # Check text length
        if len(text) < 50:
            issues.append("Very short content extracted")
            recommendations.append("Verify document contains readable text")
            quality_factors.append(0.3)
        else:
            quality_factors.append(1.0)
        
        # Check chunk distribution
        if chunks:
            avg_chunk_size = sum(chunk["word_count"] for chunk in chunks) / len(chunks)
            if avg_chunk_size < 20:
                issues.append("Very small chunks detected")
                recommendations.append("Consider increasing chunk size")
                quality_factors.append(0.5)
            else:
                quality_factors.append(0.9)
        
        # Calculate overall quality score
        quality_score = sum(quality_factors) / len(quality_factors) if quality_factors else 0.0
        
        return {
            "quality_score": quality_score,
            "issues": issues,
            "recommendations": recommendations,
            "metrics": {
                "text_length": len(text),
                "chunk_count": len(chunks),
                "avg_chunk_size": sum(chunk["word_count"] for chunk in chunks) / len(chunks) if chunks else 0
            }
        }

# Factory function for easy integration
def get_enhanced_extractor() -> EnhancedDocumentExtractor:
    """Get enhanced document extractor instance"""
    return EnhancedDocumentExtractor()