"""
Business Document Templates Module

Specialized processing for common business document types:
- Invoice processing and metadata extraction
- Contract analysis and term identification  
- Financial document classification
- Legal document processing
- Purchase order handling
"""

from .business import BusinessDocumentProcessor

__all__ = ["BusinessDocumentProcessor"]