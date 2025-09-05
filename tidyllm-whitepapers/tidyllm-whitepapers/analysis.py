#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TidyLLM-Papers Analysis Verbs

Content analysis operations following TidyLLM verb patterns.
Provides content(), abstracts(), download(), and other analysis methods.
"""

import logging
import re
import requests
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from pathlib import Path
import time

from .core import Paper, PaperCollection

logger = logging.getLogger(__name__)

# Integration with existing PDF processing infrastructure
try:
    import sys
    backend_path = Path(__file__).parent.parent / 'src'
    if str(backend_path) not in sys.path:
        sys.path.insert(0, str(backend_path))
    
    from backend.mcp.workers.pdf_processing_worker import PDFProcessingWorker
    EXISTING_PDF_INFRASTRUCTURE = True
except ImportError:
    EXISTING_PDF_INFRASTRUCTURE = False
    logger.info("⚠️ Existing PDF infrastructure not available - using basic processing")

# Basic PDF processing fallback
try:
    import PyPDF2
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

class AnalysisOperations:
    """TidyLLM-style analysis operations"""
    
    @staticmethod
    def abstracts() -> Callable[[PaperCollection], PaperCollection]:
        """
        Analyze paper abstracts for key themes and concepts
        
        Returns:
            Function that processes PaperCollection abstract analysis
            
        Usage:
            papers("neural networks") | discover.arxiv(5) | analyze.abstracts()
        """
        
        def _abstract_analysis(collection: PaperCollection) -> PaperCollection:
            logger.info("📖 Analyzing paper abstracts...")
            
            abstract_analysis = {
                'total_papers': len(collection.papers),
                'papers_with_abstracts': 0,
                'common_keywords': {},
                'themes': [],
                'average_length': 0
            }
            
            all_abstracts = []
            keyword_counts = {}
            
            for paper in collection.papers:
                if paper.abstract:
                    abstract_analysis['papers_with_abstracts'] += 1
                    all_abstracts.append(paper.abstract)
                    
                    # Extract keywords (simple approach)
                    keywords = _extract_keywords_from_text(paper.abstract)
                    for keyword in keywords:
                        keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
                    
                    paper.add_processing_note("Abstract analyzed for keywords")
            
            # Calculate statistics
            if all_abstracts:
                abstract_analysis['average_length'] = sum(len(abstract) for abstract in all_abstracts) / len(all_abstracts)
                
                # Get most common keywords
                sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
                abstract_analysis['common_keywords'] = dict(sorted_keywords[:20])
                
                # Identify themes based on keyword clustering
                abstract_analysis['themes'] = _identify_themes(sorted_keywords[:50])
            
            collection.analysis_results['abstracts'] = abstract_analysis
            
            logger.info(f"✅ Analyzed {abstract_analysis['papers_with_abstracts']} abstracts")
            logger.info(f"🔤 Found {len(abstract_analysis['common_keywords'])} common keywords")
            
            return collection
        
        return _abstract_analysis
    
    @staticmethod
    def download(download_dir: str = "./papers", 
                 max_papers: int = None,
                 skip_existing: bool = True) -> Callable[[PaperCollection], PaperCollection]:
        """
        Download PDF files for papers in collection
        
        Args:
            download_dir: Directory to save PDF files
            max_papers: Maximum number of papers to download (None for all)
            skip_existing: Skip papers that are already downloaded
            
        Usage:
            papers("attention") | discover.arxiv(3) | analyze.download("./my_papers")
        """
        
        def _download_analysis(collection: PaperCollection) -> PaperCollection:
            download_path = Path(download_dir)
            download_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"📥 Downloading papers to: {download_path}")
            
            downloaded_count = 0
            skipped_count = 0
            failed_count = 0
            
            papers_to_download = collection.papers[:max_papers] if max_papers else collection.papers
            
            for i, paper in enumerate(papers_to_download, 1):
                if not paper.pdf_url:
                    logger.warning(f"No PDF URL for paper: {paper.title[:50]}...")
                    failed_count += 1
                    continue
                
                # Generate filename
                safe_title = _sanitize_filename(paper.title)
                if paper.arxiv_id:
                    filename = f"{paper.arxiv_id}_{safe_title[:50]}.pdf"
                else:
                    filename = f"{safe_title[:50]}.pdf"
                
                filepath = download_path / filename
                
                # Skip if already exists and skip_existing is True
                if skip_existing and filepath.exists():
                    logger.info(f"⏭️  Skipping existing: {filename}")
                    paper.local_path = str(filepath)
                    paper.downloaded = True
                    skipped_count += 1
                    continue
                
                try:
                    logger.info(f"📥 Downloading ({i}/{len(papers_to_download)}): {paper.title[:50]}...")
                    
                    # Download with proper headers
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (compatible; TidyLLM-Papers/1.0; +research@tidyllm.org)'
                    }
                    
                    response = requests.get(paper.pdf_url, headers=headers, stream=True, timeout=60)
                    response.raise_for_status()
                    
                    # Save file
                    with open(filepath, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    # Update paper record
                    paper.local_path = str(filepath)
                    paper.downloaded = True
                    paper.add_processing_note(f"Downloaded to: {filename}")
                    
                    downloaded_count += 1
                    
                    # Rate limiting - be respectful
                    time.sleep(2)
                    
                except Exception as e:
                    logger.error(f"❌ Download failed for {paper.title[:50]}: {e}")
                    paper.add_processing_note(f"Download failed: {str(e)}")
                    failed_count += 1
            
            # Update collection statistics
            collection._update_stats()
            collection.analysis_results['download'] = {
                'total_attempted': len(papers_to_download),
                'downloaded': downloaded_count,
                'skipped': skipped_count,
                'failed': failed_count,
                'download_directory': str(download_path)
            }
            
            logger.info(f"✅ Download complete: {downloaded_count} downloaded, {skipped_count} skipped, {failed_count} failed")
            
            return collection
        
        return _download_analysis
    
    @staticmethod
    def content(extract_images: bool = True,
                extract_tables: bool = True,
                max_content_length: int = 50000) -> Callable[[PaperCollection], PaperCollection]:
        """
        Extract full content from downloaded PDF papers
        
        Args:
            extract_images: Extract images from PDFs
            extract_tables: Extract tables from PDFs
            max_content_length: Maximum content length per paper
            
        Usage:
            papers("ml") | discover.arxiv(2) | analyze.download() | analyze.content()
        """
        
        def _content_analysis(collection: PaperCollection) -> PaperCollection:
            logger.info("📄 Extracting content from downloaded papers...")
            
            processed_count = 0
            failed_count = 0
            
            for paper in collection.papers:
                if not paper.downloaded or not paper.local_path:
                    continue
                
                if not Path(paper.local_path).exists():
                    logger.warning(f"PDF file not found: {paper.local_path}")
                    continue
                
                try:
                    # Try existing PDF infrastructure first
                    if EXISTING_PDF_INFRASTRUCTURE:
                        content_data = _extract_with_existing_infrastructure(
                            paper.local_path, extract_images, extract_tables
                        )
                    else:
                        content_data = _extract_with_fallback_method(
                            paper.local_path, extract_images, extract_tables
                        )
                    
                    if content_data:
                        # Update paper with extracted content
                        paper.content = content_data.get('text', '')[:max_content_length]
                        paper.images = content_data.get('images', [])
                        paper.tables = content_data.get('tables', [])
                        paper.analyzed = True
                        
                        paper.add_processing_note(f"Content extracted: {len(paper.content)} chars, {len(paper.images)} images, {len(paper.tables)} tables")
                        processed_count += 1
                    
                except Exception as e:
                    logger.error(f"❌ Content extraction failed for {paper.title[:50]}: {e}")
                    paper.add_processing_note(f"Content extraction failed: {str(e)}")
                    failed_count += 1
            
            # Update collection statistics
            collection._update_stats()
            collection.analysis_results['content_extraction'] = {
                'total_papers': len(collection.papers),
                'processed': processed_count,
                'failed': failed_count,
                'extract_images': extract_images,
                'extract_tables': extract_tables
            }
            
            logger.info(f"✅ Content extraction complete: {processed_count} processed, {failed_count} failed")
            
            return collection
        
        return _content_analysis
    
    @staticmethod
    def metadata() -> Callable[[PaperCollection], PaperCollection]:
        """
        Analyze metadata patterns across paper collection
        
        Usage:
            papers("ai") | discover.arxiv(10) | analyze.metadata()
        """
        
        def _metadata_analysis(collection: PaperCollection) -> PaperCollection:
            logger.info("📊 Analyzing collection metadata...")
            
            metadata_analysis = {
                'publication_years': {},
                'categories': {},
                'author_counts': {},
                'most_productive_authors': {},
                'venue_analysis': {},
                'collaboration_patterns': {}
            }
            
            all_authors = []
            author_paper_counts = {}
            
            for paper in collection.papers:
                # Publication year analysis
                if paper.published_date:
                    year = paper.published_date[:4]
                    metadata_analysis['publication_years'][year] = metadata_analysis['publication_years'].get(year, 0) + 1
                
                # Category analysis
                for category in paper.categories:
                    metadata_analysis['categories'][category] = metadata_analysis['categories'].get(category, 0) + 1
                
                # Author analysis
                author_count = len(paper.authors)
                metadata_analysis['author_counts'][str(author_count)] = metadata_analysis['author_counts'].get(str(author_count), 0) + 1
                
                # Track individual authors
                for author in paper.authors:
                    all_authors.append(author)
                    author_paper_counts[author] = author_paper_counts.get(author, 0) + 1
            
            # Most productive authors
            sorted_authors = sorted(author_paper_counts.items(), key=lambda x: x[1], reverse=True)
            metadata_analysis['most_productive_authors'] = dict(sorted_authors[:10])
            
            # Collaboration patterns
            single_author_papers = sum(1 for paper in collection.papers if len(paper.authors) == 1)
            multi_author_papers = len(collection.papers) - single_author_papers
            
            metadata_analysis['collaboration_patterns'] = {
                'single_author_papers': single_author_papers,
                'multi_author_papers': multi_author_papers,
                'collaboration_ratio': multi_author_papers / len(collection.papers) if collection.papers else 0,
                'average_authors_per_paper': len(all_authors) / len(collection.papers) if collection.papers else 0
            }
            
            collection.analysis_results['metadata'] = metadata_analysis
            
            logger.info(f"✅ Metadata analysis complete for {len(collection.papers)} papers")
            
            return collection
        
        return _metadata_analysis

# Helper functions
def _extract_keywords_from_text(text: str, min_length: int = 3) -> List[str]:
    """Extract keywords from text using simple NLP techniques"""
    # Remove common stop words and extract meaningful terms
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
    
    # Simple keyword extraction
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    keywords = []
    
    for word in words:
        if len(word) >= min_length and word not in stop_words:
            keywords.append(word)
    
    return keywords

def _identify_themes(keyword_pairs: List[tuple], min_frequency: int = 2) -> List[str]:
    """Identify themes from keyword frequency pairs"""
    themes = []
    
    # Group related keywords into themes
    ai_ml_terms = ['machine', 'learning', 'neural', 'network', 'deep', 'artificial', 'intelligence', 'model', 'training', 'algorithm']
    vision_terms = ['image', 'vision', 'visual', 'computer', 'detection', 'recognition', 'segmentation']
    nlp_terms = ['language', 'text', 'natural', 'processing', 'translation', 'generation', 'understanding']
    
    for theme_name, theme_terms in [
        ('AI/Machine Learning', ai_ml_terms),
        ('Computer Vision', vision_terms), 
        ('Natural Language Processing', nlp_terms)
    ]:
        theme_count = sum(count for keyword, count in keyword_pairs if any(term in keyword for term in theme_terms))
        if theme_count >= min_frequency:
            themes.append(f"{theme_name} ({theme_count} mentions)")
    
    return themes

def _sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file system usage"""
    # Remove or replace problematic characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = re.sub(r'[^\w\s-]', '', filename)
    filename = re.sub(r'[-\s]+', '-', filename)
    return filename.strip('-_')

def _extract_with_existing_infrastructure(filepath: str, extract_images: bool, extract_tables: bool) -> Dict[str, Any]:
    """Extract content using existing PDF processing infrastructure"""
    try:
        worker = PDFProcessingWorker()
        
        # Create mock MCP message for compatibility
        from backend.core.mcp_message import MCPMessage
        mock_message = MCPMessage(
            content=f"Extract content from {filepath}",
            attachments=[filepath],
            metadata={
                'extract_images': extract_images,
                'extract_tables': extract_tables
            }
        )
        
        result = worker.process_task(mock_message)
        
        if result.get('success'):
            return {
                'text': result.get('text', ''),
                'images': result.get('images', []),
                'tables': result.get('tables', [])
            }
    except Exception as e:
        logger.warning(f"Existing infrastructure extraction failed: {e}")
        return None

def _extract_with_fallback_method(filepath: str, extract_images: bool, extract_tables: bool) -> Dict[str, Any]:
    """Fallback content extraction using basic PyPDF2"""
    if not PYPDF_AVAILABLE:
        logger.error("PyPDF2 not available for fallback extraction")
        return None
    
    try:
        with open(filepath, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        
        return {
            'text': text,
            'images': [],  # Basic extraction doesn't support images
            'tables': []   # Basic extraction doesn't support tables
        }
        
    except Exception as e:
        logger.error(f"Fallback extraction failed: {e}")
        return None

# Create singleton instance for TidyLLM-style access
analyze = AnalysisOperations()

__all__ = [
    'analyze',
    'AnalysisOperations'
]