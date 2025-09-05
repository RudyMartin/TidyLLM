"""
Text and Metadata Extraction Module

Handles extraction of text content and structured metadata from various document formats:
- PDF documents (first N pages)
- DOCX Word documents  
- Plain text files
- Business document pattern matching
"""

from .text import TextExtractor
from .metadata import MetadataExtractor

__all__ = ["TextExtractor", "MetadataExtractor"]