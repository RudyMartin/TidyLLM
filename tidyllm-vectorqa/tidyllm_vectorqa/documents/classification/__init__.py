"""
Document Classification Module

Multi-category document classification with confidence scoring using:
- TF-IDF embeddings for semantic similarity
- Custom category training capabilities
- Confidence-based validation
- Support for business document types
"""

from .classifier import DocumentClassifier

__all__ = ["DocumentClassifier"]