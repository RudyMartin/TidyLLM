#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TidyLLM-Papers Core Classes

Core data structures for paper processing following TidyLLM patterns.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class Paper:
    """
    Individual research paper with metadata and content
    
    Represents a single research paper with all associated metadata,
    content, and processing state.
    """
    
    def __init__(self, 
                 title: str,
                 authors: List[str] = None,
                 arxiv_id: str = None,
                 pdf_url: str = None,
                 abstract: str = None,
                 published_date: str = None,
                 categories: List[str] = None,
                 **metadata):
        
        self.title = title
        self.authors = authors or []
        self.arxiv_id = arxiv_id
        self.pdf_url = pdf_url
        self.abstract = abstract
        self.published_date = published_date
        self.categories = categories or []
        self.metadata = metadata
        
        # Content fields (populated by analysis)
        self.content = ""
        self.citations = []
        self.references = []
        self.images = []
        self.tables = []
        
        # Processing state
        self.downloaded = False
        self.analyzed = False
        self.local_path = None
        
        # Additional metadata
        self.discovered_date = datetime.now().isoformat()
        self.processing_notes = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert paper to dictionary representation"""
        return {
            'title': self.title,
            'authors': self.authors,
            'arxiv_id': self.arxiv_id,
            'pdf_url': self.pdf_url,
            'abstract': self.abstract,
            'published_date': self.published_date,
            'categories': self.categories,
            'metadata': self.metadata,
            'content': self.content,
            'citations': self.citations,
            'references': self.references,
            'downloaded': self.downloaded,
            'analyzed': self.analyzed,
            'local_path': self.local_path,
            'discovered_date': self.discovered_date,
            'processing_notes': self.processing_notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Paper':
        """Create paper from dictionary representation"""
        paper = cls(
            title=data.get('title', ''),
            authors=data.get('authors', []),
            arxiv_id=data.get('arxiv_id'),
            pdf_url=data.get('pdf_url'),
            abstract=data.get('abstract'),
            published_date=data.get('published_date'),
            categories=data.get('categories', []),
            **data.get('metadata', {})
        )
        
        # Restore processing state
        paper.content = data.get('content', '')
        paper.citations = data.get('citations', [])
        paper.references = data.get('references', [])
        paper.downloaded = data.get('downloaded', False)
        paper.analyzed = data.get('analyzed', False)
        paper.local_path = data.get('local_path')
        paper.discovered_date = data.get('discovered_date', datetime.now().isoformat())
        paper.processing_notes = data.get('processing_notes', [])
        
        return paper
    
    def add_processing_note(self, note: str):
        """Add processing note with timestamp"""
        timestamped_note = f"[{datetime.now().isoformat()}] {note}"
        self.processing_notes.append(timestamped_note)
    
    def __str__(self):
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += " et al."
        
        return f"{self.title} ({authors_str}) - {self.published_date or 'Unknown date'}"
    
    def __repr__(self):
        return f"Paper(title='{self.title[:50]}...', authors={len(self.authors)}, arxiv_id='{self.arxiv_id}')"

class PaperCollection:
    """
    Collection of research papers with TidyLLM-style pipeline processing
    
    Similar to AttachmentCollection in llmdata.attachments_enhanced,
    but specialized for research paper workflows.
    """
    
    def __init__(self, query: str = None, papers: List[Paper] = None):
        self.query = query or ""
        self.papers = papers or []
        self.metadata = {
            'created': datetime.now().isoformat(),
            'query': self.query,
            'total_papers': len(self.papers)
        }
        
        # Processing results
        self.analysis_results = {}
        self.citation_analysis = {}
        self.attachments_ready = False
        
        # Statistics
        self.stats = {
            'downloaded': 0,
            'analyzed': 0,
            'total_citations': 0,
            'unique_authors': 0
        }
        
        self._update_stats()
    
    def __or__(self, operation):
        """Pipeline operator for TidyLLM-style processing"""
        return operation(self)
    
    def __len__(self):
        return len(self.papers)
    
    def __iter__(self):
        return iter(self.papers)
    
    def __getitem__(self, index):
        return self.papers[index]
    
    def add_paper(self, paper: Paper):
        """Add paper to collection"""
        self.papers.append(paper)
        self.metadata['total_papers'] = len(self.papers)
        self._update_stats()
    
    def add_papers(self, papers: List[Paper]):
        """Add multiple papers to collection"""
        self.papers.extend(papers)
        self.metadata['total_papers'] = len(self.papers)
        self._update_stats()
    
    def _update_stats(self):
        """Update collection statistics"""
        self.stats['downloaded'] = sum(1 for p in self.papers if p.downloaded)
        self.stats['analyzed'] = sum(1 for p in self.papers if p.analyzed)
        self.stats['total_citations'] = sum(len(p.citations) for p in self.papers)
        
        # Count unique authors
        all_authors = set()
        for paper in self.papers:
            all_authors.update(paper.authors)
        self.stats['unique_authors'] = len(all_authors)
    
    def filter_by_category(self, category: str) -> 'PaperCollection':
        """Filter papers by category"""
        filtered_papers = [p for p in self.papers if category in p.categories]
        filtered_collection = PaperCollection(
            query=f"{self.query} [filtered by {category}]",
            papers=filtered_papers
        )
        return filtered_collection
    
    def filter_by_date_range(self, start_date: str = None, end_date: str = None) -> 'PaperCollection':
        """Filter papers by publication date range"""
        filtered_papers = []
        for paper in self.papers:
            if not paper.published_date:
                continue
            
            paper_date = paper.published_date
            include = True
            
            if start_date and paper_date < start_date:
                include = False
            if end_date and paper_date > end_date:
                include = False
                
            if include:
                filtered_papers.append(paper)
        
        filtered_collection = PaperCollection(
            query=f"{self.query} [filtered by date]",
            papers=filtered_papers
        )
        return filtered_collection
    
    def top_papers(self, n: int = 10) -> 'PaperCollection':
        """Get top N papers (by relevance or date)"""
        top_papers = self.papers[:n]
        filtered_collection = PaperCollection(
            query=f"{self.query} [top {n}]",
            papers=top_papers
        )
        return filtered_collection
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert collection to dictionary representation"""
        return {
            'query': self.query,
            'metadata': self.metadata,
            'stats': self.stats,
            'papers': [paper.to_dict() for paper in self.papers],
            'analysis_results': self.analysis_results,
            'citation_analysis': self.citation_analysis,
            'attachments_ready': self.attachments_ready
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PaperCollection':
        """Create collection from dictionary representation"""
        papers = [Paper.from_dict(paper_data) for paper_data in data.get('papers', [])]
        collection = cls(query=data.get('query', ''), papers=papers)
        
        collection.metadata.update(data.get('metadata', {}))
        collection.stats.update(data.get('stats', {}))
        collection.analysis_results = data.get('analysis_results', {})
        collection.citation_analysis = data.get('citation_analysis', {})
        collection.attachments_ready = data.get('attachments_ready', False)
        
        return collection
    
    def save_to_file(self, filepath: str):
        """Save collection to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Saved paper collection to {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'PaperCollection':
        """Load collection from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded paper collection from {filepath}")
        return cls.from_dict(data)
    
    def summary(self) -> str:
        """Generate summary of collection"""
        summary_lines = [
            f"📄 Paper Collection: {self.query}",
            f"📊 Total Papers: {len(self.papers)}",
            f"⬇️  Downloaded: {self.stats['downloaded']}/{len(self.papers)}",
            f"🔍 Analyzed: {self.stats['analyzed']}/{len(self.papers)}",
            f"📖 Total Citations: {self.stats['total_citations']}",
            f"👥 Unique Authors: {self.stats['unique_authors']}",
            f"📅 Created: {self.metadata['created']}"
        ]
        
        if self.papers:
            summary_lines.append("\n📑 Recent Papers:")
            for i, paper in enumerate(self.papers[:5]):
                summary_lines.append(f"  {i+1}. {paper}")
        
        return "\n".join(summary_lines)
    
    def __str__(self):
        return self.summary()
    
    def __repr__(self):
        return f"PaperCollection(query='{self.query}', papers={len(self.papers)})"

# Factory function following TidyLLM patterns
def papers(query: str) -> PaperCollection:
    """
    Create paper collection with TidyLLM-style interface
    
    Usage:
        research = (papers("machine learning attention mechanisms")
                   | discover.arxiv(limit=10)
                   | analyze.content()
                   | cite.extract_references())
    
    Args:
        query: Search query for paper discovery
        
    Returns:
        PaperCollection ready for pipeline processing
    """
    return PaperCollection(query=query)

__all__ = [
    'Paper',
    'PaperCollection', 
    'papers'
]