"""
TidyLLM Vector QA - Comprehensive ML Package
===========================================

A unified package consolidating all TidyLLM ecosystem functionality:
- Document processing and extraction
- Sentence embeddings and similarity  
- Core ML algorithms
- Y=R+S+N mathematical framework
- Whitepapers analysis and research tools

Part of the tidyllm-verse: Educational ML with complete transparency
"""

__version__ = "0.0.1"
__author__ = "Rudy Martin, TidyLLM Research Project"

# Core exports from submodules
try:
    from .documents import TextExtractor, MetadataExtractor, BusinessDocumentProcessor
except ImportError:
    TextExtractor = MetadataExtractor = BusinessDocumentProcessor = None

try:
    from .sentence.utils.similarity import cosine_similarity as SimilarityCalculator
    from .sentence.tfidf.embeddings import TfIdfEmbeddings as TidyEmbeddings
except ImportError:
    SimilarityCalculator = TidyEmbeddings = None

try:
    from .core.attention.scaled_dot_product import scaled_dot_product_attention as AttentionMechanism
    MLAlgorithms = "Available in core submodules"
except ImportError:
    AttentionMechanism = MLAlgorithms = None

from .yrsn import (
    ResearchFramework, 
    DecompositionScore, 
    extract_table_of_contents, 
    extract_bibliography
)

try:
    from .whitepapers.research_framework import ResearchFramework as WhitepaperAnalyzer
except ImportError:
    WhitepaperAnalyzer = ResearchFramework  # Use YRSN framework as fallback

# Convenience imports
from .yrsn.research_framework import (
    ResearchFramework,
    DecompositionScore, 
    PaperAnalysis,
    get_demo_papers,
    analyze_context_collapse_types,
    extract_table_of_contents,
    extract_bibliography
)

__all__ = [
    # Documents
    "TextExtractor",
    "MetadataExtractor", 
    "BusinessDocumentProcessor",
    
    # Sentence/Embeddings
    "TidyEmbeddings",
    "SimilarityCalculator",
    
    # Core ML
    "AttentionMechanism",
    "MLAlgorithms",
    
    # Y=R+S+N Framework
    "ResearchFramework",
    "DecompositionScore",
    "PaperAnalysis", 
    "get_demo_papers",
    "analyze_context_collapse_types",
    "extract_table_of_contents",
    "extract_bibliography",
    
    # Whitepapers
    "WhitepaperAnalyzer",
]