"""
tidyllm-documents: Document processing and classification toolkit

This package provides comprehensive document processing capabilities:
1. Document Classification (categorization with confidence scoring)
2. Text Extraction (PDF, DOCX, TXT processing)
3. Business Templates (20+ business document patterns and metadata extraction)

Part of the tidyllm-verse: Educational ML tools with complete algorithmic transparency.
"""

from .classification import DocumentClassifier
from .extraction import TextExtractor, MetadataExtractor
from .templates import BusinessDocumentProcessor

__version__ = "0.1.0"
__author__ = "Rudy Martin"

__all__ = [
    "DocumentClassifier",
    "TextExtractor", 
    "MetadataExtractor",
    "BusinessDocumentProcessor"
]

# Package metadata
DESCRIPTION = "Document processing and classification toolkit with complete algorithmic transparency"
LICENSE = "CC-BY-4.0"
HOMEPAGE = "https://github.com/tidyllm-verse/tidyllm-documents"