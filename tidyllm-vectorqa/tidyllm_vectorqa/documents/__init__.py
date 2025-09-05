"""
Document Processing Module
=========================

Text extraction and metadata processing for business documents.
Supports PDF, DOCX, TXT with confidence scoring and pattern matching.
"""

from .extraction.text import TextExtractor
from .extraction.metadata import MetadataExtractor

# Try to import business processor if available
try:
    from .templates.business import BusinessDocumentProcessor
except ImportError:
    # Create a placeholder if business.py doesn't exist
    class BusinessDocumentProcessor:
        def __init__(self):
            self.text_extractor = TextExtractor()
            self.metadata_extractor = MetadataExtractor()
        
        def process_document(self, file_path):
            text, text_meta = self.text_extractor.extract_text(file_path)
            metadata = self.metadata_extractor.extract_metadata(text)
            
            return {
                'text': text,
                'text_metadata': text_meta,
                'extracted_metadata': metadata,
                'document_type': 'unknown',
                'confidence': 0.5
            }

__all__ = ["TextExtractor", "MetadataExtractor", "BusinessDocumentProcessor"]