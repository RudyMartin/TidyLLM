"""
Y=R+S+N Mathematical Framework
=============================

Mathematical decomposition framework for content quality analysis and context collapse prevention.
"""

from .research_framework import (
    ResearchFramework,
    DecompositionScore,
    PaperAnalysis, 
    get_demo_papers,
    analyze_context_collapse_types,
    extract_table_of_contents,
    extract_bibliography
)

__all__ = [
    "ResearchFramework",
    "DecompositionScore",
    "PaperAnalysis",
    "get_demo_papers", 
    "analyze_context_collapse_types",
    "extract_table_of_contents",
    "extract_bibliography"
]
