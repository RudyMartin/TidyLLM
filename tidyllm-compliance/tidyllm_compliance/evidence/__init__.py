"""
Evidence Validation Module

Automated assessment of document authenticity and evidential value for audit and investigation purposes.
Evaluates documents for authenticity indicators, completeness, and quality markers.
"""

from .validation import EvidenceValidator

__all__ = ["EvidenceValidator"]