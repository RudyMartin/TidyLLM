"""
SOP Golden Answers Module
========================

Implements the SOP (Standard Operating Procedures) Golden Answers system
that provides authoritative compliance guidance with precedence over 
general domain knowledge.

This module handles:
- SOP-specific domain RAG construction
- Golden answers precedence validation  
- MVR workflow SOP compliance checking
"""

from .sop_validator import SOPValidator

__all__ = [
    "SOPValidator"
]