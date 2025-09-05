#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TidyLLM-Papers Discovery Verbs

Research paper discovery operations following TidyLLM verb patterns.
Provides arxiv(), scholar(), and other discovery methods.
"""

import logging
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import time

from .core import Paper, PaperCollection

logger = logging.getLogger(__name__)

# ArXiv integration
try:
    import arxiv
    ARXIV_AVAILABLE = True
except ImportError:
    ARXIV_AVAILABLE = False
    logger.warning("ArXiv package not available - ArXiv discovery disabled")

class DiscoveryOperations:
    """TidyLLM-style discovery operations"""
    
    @staticmethod
    def arxiv(limit: int = 10, 
              sort_by: str = "relevance", 
              sort_order: str = "descending",
              max_results: int = None,
              categories: List[str] = None) -> Callable[[PaperCollection], PaperCollection]:
        """
        Discover papers from ArXiv following TidyLLM patterns
        
        Args:
            limit: Maximum number of papers to retrieve
            sort_by: "relevance", "lastUpdatedDate", "submittedDate"
            sort_order: "ascending" or "descending"
            max_results: Legacy parameter for compatibility
            categories: Filter by ArXiv categories (e.g., ["cs.AI", "cs.LG"])
            
        Returns:
            Function that processes PaperCollection through ArXiv discovery
            
        Usage:
            papers("attention mechanisms") | discover.arxiv(limit=5)
        """
        
        # Handle legacy parameter
        if max_results:
            limit = max_results
        
        def _arxiv_discovery(collection: PaperCollection) -> PaperCollection:
            if not ARXIV_AVAILABLE:
                logger.error("ArXiv package not available - install with: pip install arxiv")
                return collection
            
            try:
                logger.info(f"🔍 Searching ArXiv for: '{collection.query}' (limit: {limit})")
                
                # Build query with category filters if specified
                query = collection.query
                if categories:
                    category_filter = " OR ".join([f"cat:{cat}" for cat in categories])
                    query = f"({query}) AND ({category_filter})"
                
                # Map sort parameters to ArXiv API
                sort_criterion_map = {
                    "relevance": arxiv.SortCriterion.Relevance,
                    "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
                    "submittedDate": arxiv.SortCriterion.SubmittedDate
                }
                
                sort_order_map = {
                    "ascending": arxiv.SortOrder.Ascending,
                    "descending": arxiv.SortOrder.Descending
                }
                
                # Create ArXiv search
                search = arxiv.Search(
                    query=query,
                    max_results=limit * 2,  # Get extra results to filter
                    sort_by=sort_criterion_map.get(sort_by, arxiv.SortCriterion.Relevance),
                    sort_order=sort_order_map.get(sort_order, arxiv.SortOrder.Descending)
                )
                
                # Process results
                papers_found = []
                for result in search.results():
                    if len(papers_found) >= limit:
                        break
                    
                    paper = Paper(
                        title=result.title,
                        authors=[author.name for author in result.authors],
                        arxiv_id=result.entry_id.split('/')[-1],
                        pdf_url=result.pdf_url,
                        abstract=result.summary,
                        published_date=result.published.strftime("%Y-%m-%d"),
                        categories=result.categories
                    )
                    
                    # Add ArXiv-specific metadata
                    paper.metadata.update({
                        'source': 'arxiv',
                        'updated_date': result.updated.strftime("%Y-%m-%d"),
                        'journal_ref': result.journal_ref,
                        'doi': result.doi,
                        'primary_category': result.primary_category,
                        'comment': result.comment,
                        'links': [link.href for link in result.links]
                    })
                    
                    paper.add_processing_note(f"Discovered via ArXiv search: {collection.query}")
                    papers_found.append(paper)
                    
                    # Rate limiting - be nice to ArXiv
                    time.sleep(0.1)
                
                # Add papers to collection
                collection.add_papers(papers_found)
                
                logger.info(f"✅ Found {len(papers_found)} papers from ArXiv")
                collection.metadata['discovery_source'] = 'arxiv'
                collection.metadata['discovery_params'] = {
                    'limit': limit,
                    'sort_by': sort_by,
                    'sort_order': sort_order,
                    'categories': categories
                }
                
                return collection
                
            except Exception as e:
                logger.error(f"ArXiv discovery failed: {e}")
                collection.add_paper(Paper(
                    title=f"ArXiv Discovery Error: {str(e)}",
                    authors=["System"],
                    abstract=f"Failed to search ArXiv for query: {collection.query}"
                ))
                return collection
        
        return _arxiv_discovery
    
    @staticmethod
    def recent(days: int = 30) -> Callable[[PaperCollection], PaperCollection]:
        """
        Discover recent papers from the last N days
        
        Args:
            days: Number of days to look back
            
        Usage:
            papers("machine learning") | discover.recent(days=7)
        """
        
        def _recent_discovery(collection: PaperCollection) -> PaperCollection:
            # Use ArXiv with date-based query modification
            from datetime import datetime, timedelta
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Modify query to include date filter
            date_query = f"{collection.query} AND submittedDate:[{start_date.strftime('%Y%m%d')}* TO {end_date.strftime('%Y%m%d')}*]"
            
            # Create temporary collection with date-filtered query
            temp_collection = PaperCollection(query=date_query)
            
            # Use ArXiv discovery with date sorting
            arxiv_discovery = DiscoveryOperations.arxiv(
                limit=50,  # Get more results since we're filtering by date
                sort_by="submittedDate",
                sort_order="descending"
            )
            
            # Apply discovery
            temp_collection = arxiv_discovery(temp_collection)
            
            # Filter results to ensure they're within date range
            recent_papers = []
            for paper in temp_collection.papers:
                if paper.published_date:
                    paper_date = datetime.strptime(paper.published_date, "%Y-%m-%d")
                    if paper_date >= start_date:
                        paper.add_processing_note(f"Recent paper (last {days} days)")
                        recent_papers.append(paper)
            
            collection.add_papers(recent_papers)
            collection.metadata['discovery_filter'] = f'recent_{days}_days'
            
            logger.info(f"✅ Found {len(recent_papers)} recent papers (last {days} days)")
            return collection
        
        return _recent_discovery
    
    @staticmethod
    def by_category(categories: List[str], limit: int = 10) -> Callable[[PaperCollection], PaperCollection]:
        """
        Discover papers by specific ArXiv categories
        
        Args:
            categories: List of ArXiv categories (e.g., ["cs.AI", "cs.LG", "cs.CV"])
            limit: Maximum papers per category
            
        Usage:
            papers("neural networks") | discover.by_category(["cs.AI", "cs.LG"])
        """
        
        def _category_discovery(collection: PaperCollection) -> PaperCollection:
            all_papers = []
            
            for category in categories:
                logger.info(f"🔍 Searching category: {category}")
                
                # Create category-specific query
                category_query = f"cat:{category}"
                if collection.query:
                    category_query = f"({collection.query}) AND cat:{category}"
                
                temp_collection = PaperCollection(query=category_query)
                
                # Use ArXiv discovery for this category
                arxiv_discovery = DiscoveryOperations.arxiv(
                    limit=limit,
                    sort_by="relevance"
                )
                
                temp_collection = arxiv_discovery(temp_collection)
                
                # Tag papers with category
                for paper in temp_collection.papers:
                    paper.metadata['primary_discovery_category'] = category
                    paper.add_processing_note(f"Found in category: {category}")
                
                all_papers.extend(temp_collection.papers)
                
                # Rate limiting between categories
                time.sleep(0.5)
            
            collection.add_papers(all_papers)
            collection.metadata['discovery_categories'] = categories
            
            logger.info(f"✅ Found {len(all_papers)} papers across {len(categories)} categories")
            return collection
        
        return _category_discovery
    
    @staticmethod
    def sample(n: int = 5) -> Callable[[PaperCollection], PaperCollection]:
        """
        Get a random sample of papers for testing/exploration
        
        Args:
            n: Number of sample papers to retrieve
            
        Usage:
            papers("deep learning") | discover.sample(3)
        """
        
        def _sample_discovery(collection: PaperCollection) -> PaperCollection:
            # Use ArXiv with broader search to get variety
            arxiv_discovery = DiscoveryOperations.arxiv(
                limit=n * 3,  # Get more papers to sample from
                sort_by="lastUpdatedDate"  # Mix of recent and older
            )
            
            temp_collection = arxiv_discovery(PaperCollection(query=collection.query))
            
            # Sample papers if we have enough
            if len(temp_collection.papers) > n:
                import random
                sampled_papers = random.sample(temp_collection.papers, n)
            else:
                sampled_papers = temp_collection.papers
            
            # Mark as samples
            for paper in sampled_papers:
                paper.add_processing_note("Selected as sample paper")
            
            collection.add_papers(sampled_papers)
            collection.metadata['discovery_type'] = 'sample'
            collection.metadata['sample_size'] = len(sampled_papers)
            
            logger.info(f"✅ Sampled {len(sampled_papers)} papers")
            return collection
        
        return _sample_discovery

# Create singleton instance for TidyLLM-style access
discover = DiscoveryOperations()

__all__ = [
    'discover',
    'DiscoveryOperations'
]