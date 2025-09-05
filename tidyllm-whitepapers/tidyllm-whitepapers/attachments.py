#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TidyLLM-Papers Attachments Integration

Integration with LLMData attachments system for seamless paper processing
within TidyLLM workflows. Converts paper collections to attachment format
for direct LLM analysis.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from .core import Paper, PaperCollection

logger = logging.getLogger(__name__)

# Import LLMData attachments if available
try:
    import tidyllm
    from tidyllm.attachments_enhanced import AttachmentCollection, attach, load, present, refine
    LLMDATA_AVAILABLE = True
except ImportError:
    LLMDATA_AVAILABLE = False
    logger.info("⚠️ LLMData not available - attachment integration limited")

def as_attachments(collection: PaperCollection, 
                  include_pdfs: bool = True,
                  include_content: bool = True,
                  include_abstracts: bool = True,
                  max_papers: int = None) -> 'AttachmentCollection':
    """
    Convert PaperCollection to LLMData AttachmentCollection
    
    This enables seamless integration with TidyLLM workflows:
    
        papers("ai") | discover.arxiv(5) | analyze.download() 
                    | as_attachments() 
                    | chat(claude())
    
    Args:
        collection: PaperCollection to convert
        include_pdfs: Include PDF files as attachments (if downloaded)
        include_content: Include extracted text content
        include_abstracts: Include paper abstracts in text
        max_papers: Maximum number of papers to include
        
    Returns:
        AttachmentCollection ready for LLM processing
    """
    
    if not LLMDATA_AVAILABLE:
        logger.error("LLMData not available - cannot create attachments")
        return None
    
    # Limit papers if specified
    papers_to_include = collection.papers[:max_papers] if max_papers else collection.papers
    
    # Collect file paths for attachment
    attachment_files = []
    text_content_parts = []
    
    # Add paper collection summary
    text_content_parts.append("# Research Paper Collection")
    text_content_parts.append(f"Query: {collection.query}")
    text_content_parts.append(f"Total Papers: {len(papers_to_include)}")
    text_content_parts.append("")
    
    for i, paper in enumerate(papers_to_include, 1):
        # Add PDF file if available and requested
        if include_pdfs and paper.downloaded and paper.local_path:
            if Path(paper.local_path).exists():
                attachment_files.append(paper.local_path)
        
        # Add text content
        if include_content or include_abstracts:
            text_content_parts.append(f"## Paper {i}: {paper.title}")
            text_content_parts.append(f"**Authors:** {', '.join(paper.authors)}")
            
            if paper.published_date:
                text_content_parts.append(f"**Published:** {paper.published_date}")
            
            if paper.categories:
                text_content_parts.append(f"**Categories:** {', '.join(paper.categories)}")
            
            if paper.arxiv_id:
                text_content_parts.append(f"**ArXiv ID:** {paper.arxiv_id}")
            
            text_content_parts.append("")
            
            # Add abstract if requested
            if include_abstracts and paper.abstract:
                text_content_parts.append("**Abstract:**")
                text_content_parts.append(paper.abstract)
                text_content_parts.append("")
            
            # Add extracted content if requested and available
            if include_content and paper.content:
                text_content_parts.append("**Content:**")
                text_content_parts.append(paper.content[:5000])  # Limit content length
                if len(paper.content) > 5000:
                    text_content_parts.append("[Content truncated...]")
                text_content_parts.append("")
            
            # Add references if available
            if paper.references:
                text_content_parts.append("**References:**")
                for ref in paper.references[:10]:  # Limit references
                    text_content_parts.append(f"- {ref}")
                if len(paper.references) > 10:
                    text_content_parts.append(f"[... and {len(paper.references) - 10} more references]")
                text_content_parts.append("")
            
            text_content_parts.append("---")
            text_content_parts.append("")
    
    # Create temporary text file with all content
    combined_text = "\n".join(text_content_parts)
    temp_text_file = f"/tmp/papers_collection_{collection.query.replace(' ', '_')[:20]}.md"
    
    try:
        with open(temp_text_file, 'w', encoding='utf-8') as f:
            f.write(combined_text)
        attachment_files.append(temp_text_file)
    except Exception as e:
        logger.warning(f"Could not create temporary text file: {e}")
    
    # Create AttachmentCollection
    if attachment_files:
        attachment_collection = attach(*attachment_files)
        
        # Process attachments using LLMData pipeline
        processed_collection = (attachment_collection
                               | load.auto()
                               | present.markdown() + present.data_summary()
                               | refine.add_headers())
        
        logger.info(f"✅ Created AttachmentCollection with {len(attachment_files)} files for {len(papers_to_include)} papers")
        return processed_collection
    else:
        logger.warning("No attachments available - creating empty collection")
        return AttachmentCollection([])

def to_llmdata(collection: PaperCollection, 
               query_template: str = "Analyze these research papers: {query}") -> 'LLMMessage':
    """
    Convert PaperCollection directly to LLMMessage for TidyLLM workflows
    
    Args:
        collection: PaperCollection to convert
        query_template: Template for LLM query (with {query} placeholder)
        
    Returns:
        LLMMessage ready for chat() verb
        
    Usage:
        insights = (papers("attention mechanisms") 
                   | discover.arxiv(3)
                   | to_llmdata("Summarize key innovations in: {query}")
                   | chat(claude()))
    """
    
    if not LLMDATA_AVAILABLE:
        logger.error("LLMData not available - cannot create LLMMessage")
        return None
    
    # Generate query text
    query_text = query_template.format(query=collection.query)
    
    # Convert to attachments
    attachments = as_attachments(collection)
    
    if attachments and hasattr(llmdata, 'llm_message'):
        # Create LLMMessage with attachments
        message = llmdata.llm_message(query_text)
        
        # Add attachment data to message
        if attachments.text_content:
            message.content += "\n\n" + attachments.text_content
        
        # Add image attachments if any
        for image in attachments.images:
            if not hasattr(message, 'image_attachments'):
                message.image_attachments = []
            message.image_attachments.append(image)
        
        logger.info(f"✅ Created LLMMessage for {len(collection.papers)} papers")
        return message
    else:
        logger.error("Failed to create LLMMessage - check LLMData installation")
        return None

def save_for_llm_analysis(collection: PaperCollection, 
                         output_dir: str = "./paper_analysis",
                         format: str = "markdown") -> Dict[str, str]:
    """
    Save paper collection in LLM-friendly formats
    
    Args:
        collection: PaperCollection to save
        output_dir: Directory to save files
        format: Output format ("markdown", "json", "text")
        
    Returns:
        Dict with saved file paths
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    if format == "markdown":
        # Create comprehensive markdown file
        md_content = []
        md_content.append(f"# Research Analysis: {collection.query}")
        md_content.append(f"Generated: {collection.metadata.get('created', 'Unknown')}")
        md_content.append(f"Total Papers: {len(collection.papers)}")
        md_content.append("")
        
        # Collection statistics
        if collection.stats:
            md_content.append("## Collection Statistics")
            for key, value in collection.stats.items():
                md_content.append(f"- **{key.replace('_', ' ').title()}**: {value}")
            md_content.append("")
        
        # Analysis results
        if collection.analysis_results:
            md_content.append("## Analysis Results")
            for analysis_type, results in collection.analysis_results.items():
                md_content.append(f"### {analysis_type.replace('_', ' ').title()}")
                if isinstance(results, dict):
                    for key, value in results.items():
                        if isinstance(value, (int, float, str)):
                            md_content.append(f"- **{key}**: {value}")
                md_content.append("")
        
        # Individual papers
        md_content.append("## Papers")
        for i, paper in enumerate(collection.papers, 1):
            md_content.append(f"### {i}. {paper.title}")
            md_content.append(f"**Authors**: {', '.join(paper.authors)}")
            
            if paper.published_date:
                md_content.append(f"**Published**: {paper.published_date}")
            
            if paper.categories:
                md_content.append(f"**Categories**: {', '.join(paper.categories)}")
            
            if paper.abstract:
                md_content.append(f"**Abstract**: {paper.abstract}")
            
            md_content.append("")
        
        # Save markdown file
        md_file = output_path / f"{collection.query.replace(' ', '_')}_analysis.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_content))
        saved_files['markdown'] = str(md_file)
    
    elif format == "json":
        # Save as JSON
        json_file = output_path / f"{collection.query.replace(' ', '_')}_data.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(collection.to_dict(), f, indent=2, ensure_ascii=False)
        saved_files['json'] = str(json_file)
    
    elif format == "text":
        # Save as plain text
        text_content = []
        text_content.append(f"Research Analysis: {collection.query}")
        text_content.append("=" * 60)
        
        for i, paper in enumerate(collection.papers, 1):
            text_content.append(f"\nPaper {i}: {paper.title}")
            text_content.append(f"Authors: {', '.join(paper.authors)}")
            
            if paper.abstract:
                text_content.append(f"Abstract: {paper.abstract}")
            
            text_content.append("-" * 40)
        
        txt_file = output_path / f"{collection.query.replace(' ', '_')}_papers.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(text_content))
        saved_files['text'] = str(txt_file)
    
    logger.info(f"✅ Saved paper collection in {format} format to {output_dir}")
    return saved_files

# Extension methods for PaperCollection (monkey patching)
def _add_attachment_methods():
    """Add attachment methods to PaperCollection class"""
    
    if LLMDATA_AVAILABLE:
        # Add methods that return functions for pipeline usage
        PaperCollection.as_attachments = lambda self, **kwargs: lambda _: as_attachments(self, **kwargs)
        PaperCollection.to_llmdata = lambda self, **kwargs: lambda _: to_llmdata(self, **kwargs)
        PaperCollection.save_for_llm = lambda self, **kwargs: lambda _: save_for_llm_analysis(self, **kwargs)
        
        logger.info("✅ Added attachment methods to PaperCollection")

# Initialize attachment methods
_add_attachment_methods()

__all__ = [
    'as_attachments',
    'to_llmdata', 
    'save_for_llm_analysis',
    'LLMDATA_AVAILABLE'
]