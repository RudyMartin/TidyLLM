#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TidyLLM-Papers - Research Paper Processing for TidyLLM Ecosystem

A specialized tool package for research paper discovery, processing, and analysis 
that integrates seamlessly with the TidyLLM ecosystem using the same grammar patterns.

Features:
- TidyLLM-style paper processing: papers(query) | discover.arxiv() | analyze.content()
- ArXiv integration with automatic paper discovery
- Enhanced citation extraction and reference analysis
- Seamless integration with existing LLMData attachments
- Pipeline-based processing with familiar | operator syntax
- Direct attachment to LLM models for paper analysis

TidyLLM Integration:
    import tidyllm
    from tidyllm_papers import papers, discover, analyze, cite
    
    # Discover and analyze papers
    research = (papers("machine learning attention mechanisms")
               | discover.arxiv(limit=5)
               | analyze.content()
               | cite.extract_references())
    
    # Attach to LLM for analysis
    insights = (llm_message("Summarize key innovations", research.as_attachments())
               | chat(claude(model="claude-3-5-sonnet")))

Architecture:
    tidyllm-papers/
    ├── __init__.py          # Main API and TidyLLM integration
    ├── core.py              # PaperCollection, Paper classes
    ├── discovery.py         # ArXiv, Google Scholar discovery verbs  
    ├── analysis.py          # Content analysis verbs
    ├── citations.py         # Citation extraction and formatting verbs
    ├── attachments.py       # LLMData attachments integration
    └── examples/            # Usage demonstrations
"""

__version__ = "0.1.0"
__author__ = "TidyLLM Papers Development Team"

# Core API (TidyLLM-style interface)
from .core import (
    PaperCollection,
    Paper,
    papers
)

# Discovery verbs
from .discovery import (
    discover
)

# Analysis verbs  
from .analysis import (
    analyze
)

# Citation verbs
from .citations import (
    cite
)

# LLMData attachments integration
from .attachments import (
    as_attachments,
    to_llmdata
)

# Check for required dependencies
try:
    import arxiv
    ARXIV_AVAILABLE = True
except ImportError:
    ARXIV_AVAILABLE = False
    print("⚠️ arxiv package not installed - ArXiv discovery disabled")

try:
    import tidyllm
    LLMDATA_INTEGRATION = True
except ImportError:
    LLMDATA_INTEGRATION = False
    print("⚠️ llmdata not found - TidyLLM integration limited")

__all__ = [
    # Core classes
    "PaperCollection",
    "Paper", 
    "papers",
    
    # Discovery verbs
    "discover",
    
    # Analysis verbs
    "analyze",
    
    # Citation verbs
    "cite",
    
    # Integration
    "as_attachments",
    "to_llmdata",
    
    # Availability flags
    "ARXIV_AVAILABLE",
    "LLMDATA_INTEGRATION",
]

# Package metadata
__package_info__ = {
    "name": "tidyllm-papers",
    "description": "Research paper processing for TidyLLM ecosystem",
    "keywords": ["llm", "papers", "research", "arxiv", "citations", "tidyllm"],
    "license": "MIT", 
    "python_requires": ">=3.8",
    "dependencies": ["arxiv>=2.2.0", "requests>=2.31.0", "llmdata>=0.1.0"],
}

# Compatibility with TidyLLM ecosystem
TIDYLLM_COMPAT_VERSION = "0.1.0"  # Compatible with llmdata version

if LLMDATA_INTEGRATION:
    print("🔗 TidyLLM-Papers loaded with LLMData integration")
else:
    print("📄 TidyLLM-Papers loaded (standalone mode)")

if ARXIV_AVAILABLE:
    print("🔍 ArXiv discovery enabled")