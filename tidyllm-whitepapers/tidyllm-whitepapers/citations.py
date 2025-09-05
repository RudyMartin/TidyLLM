#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TidyLLM-Papers Citation Verbs

Citation extraction and formatting operations following TidyLLM verb patterns.
Provides extract_references(), format_bibtex(), and other citation methods.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime

from .core import Paper, PaperCollection

logger = logging.getLogger(__name__)

class CitationOperations:
    """TidyLLM-style citation operations"""
    
    @staticmethod
    def extract_references() -> Callable[[PaperCollection], PaperCollection]:
        """
        Extract references and citations from paper content
        
        Returns:
            Function that processes PaperCollection for citation extraction
            
        Usage:
            papers("ml") | discover.arxiv(3) | analyze.content() | cite.extract_references()
        """
        
        def _extract_citations(collection: PaperCollection) -> PaperCollection:
            logger.info("📚 Extracting references and citations...")
            
            citation_analysis = {
                'total_papers': len(collection.papers),
                'papers_with_references': 0,
                'total_references': 0,
                'reference_patterns': {},
                'common_venues': {},
                'citation_years': {},
                'most_cited_papers': []
            }
            
            all_references = []
            venue_counts = {}
            
            for paper in collection.papers:
                if not paper.content:
                    continue
                
                # Extract references from content
                references = _extract_references_from_content(paper.content)
                
                if references:
                    paper.references = references
                    citation_analysis['papers_with_references'] += 1
                    citation_analysis['total_references'] += len(references)
                    all_references.extend(references)
                    
                    # Analyze reference patterns
                    for ref in references:
                        ref_type = _classify_reference(ref)
                        citation_analysis['reference_patterns'][ref_type] = citation_analysis['reference_patterns'].get(ref_type, 0) + 1
                        
                        # Extract venue information
                        venue = _extract_venue_from_reference(ref)
                        if venue:
                            venue_counts[venue] = venue_counts.get(venue, 0) + 1
                        
                        # Extract year
                        year = _extract_year_from_reference(ref)
                        if year:
                            citation_analysis['citation_years'][year] = citation_analysis['citation_years'].get(year, 0) + 1
                    
                    paper.add_processing_note(f"Extracted {len(references)} references")
                
                # Also extract in-text citations
                citations = _extract_inline_citations(paper.content)
                if citations:
                    paper.citations = citations
                    paper.add_processing_note(f"Extracted {len(citations)} in-text citations")
            
            # Sort venues by frequency
            sorted_venues = sorted(venue_counts.items(), key=lambda x: x[1], reverse=True)
            citation_analysis['common_venues'] = dict(sorted_venues[:15])
            
            # Find most cited papers (by reference frequency)
            reference_titles = {}
            for ref in all_references:
                title = _extract_title_from_reference(ref)
                if title:
                    reference_titles[title] = reference_titles.get(title, 0) + 1
            
            most_cited = sorted(reference_titles.items(), key=lambda x: x[1], reverse=True)
            citation_analysis['most_cited_papers'] = most_cited[:10]
            
            collection.citation_analysis = citation_analysis
            
            logger.info(f"✅ Citation extraction complete: {citation_analysis['total_references']} references from {citation_analysis['papers_with_references']} papers")
            
            return collection
        
        return _extract_citations
    
    @staticmethod
    def format_bibtex(filename: str = None) -> Callable[[PaperCollection], PaperCollection]:
        """
        Format papers as BibTeX entries
        
        Args:
            filename: Optional file to save BibTeX entries
            
        Usage:
            papers("ai") | discover.arxiv(5) | cite.format_bibtex("my_bibliography.bib")
        """
        
        def _format_bibtex(collection: PaperCollection) -> PaperCollection:
            logger.info("📋 Formatting papers as BibTeX entries...")
            
            bibtex_entries = []
            
            for i, paper in enumerate(collection.papers):
                # Generate BibTeX key
                if paper.authors:
                    first_author = paper.authors[0].split()[-1]  # Last name
                    year = paper.published_date[:4] if paper.published_date else "unknown"
                    bibtex_key = f"{first_author.lower()}{year}{paper.title[:10].replace(' ', '').lower()}"
                else:
                    bibtex_key = f"paper{i+1}"
                
                # Format authors
                authors_str = " and ".join(paper.authors) if paper.authors else "Anonymous"
                
                # Determine entry type
                if any(cat.startswith('cs.') for cat in paper.categories):
                    entry_type = "inproceedings"  # Most CS papers are conference papers
                else:
                    entry_type = "article"
                
                # Build BibTeX entry
                bibtex_entry = f"@{entry_type}{{{bibtex_key},\n"
                bibtex_entry += f"  title={{{paper.title}}},\n"
                bibtex_entry += f"  author={{{authors_str}}},\n"
                
                if paper.published_date:
                    bibtex_entry += f"  year={{{paper.published_date[:4]}}},\n"
                
                if paper.arxiv_id:
                    bibtex_entry += f"  eprint={{{paper.arxiv_id}}},\n"
                    bibtex_entry += f"  archivePrefix={{arXiv}},\n"
                
                if paper.categories:
                    primary_cat = paper.categories[0] if paper.categories else ""
                    bibtex_entry += f"  primaryClass={{{primary_cat}}},\n"
                
                if paper.abstract:
                    # Clean abstract for BibTeX
                    clean_abstract = paper.abstract.replace('\n', ' ').replace('{', '').replace('}', '')
                    bibtex_entry += f"  abstract={{{clean_abstract}}},\n"
                
                if paper.pdf_url:
                    bibtex_entry += f"  url={{{paper.pdf_url}}},\n"
                
                # Add custom fields
                bibtex_entry += f"  note={{Retrieved from ArXiv}},\n"
                bibtex_entry += "}\n\n"
                
                bibtex_entries.append(bibtex_entry)
            
            # Combine all entries
            full_bibtex = "".join(bibtex_entries)
            
            # Save to file if specified
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(full_bibtex)
                logger.info(f"💾 BibTeX entries saved to: {filename}")
            
            # Store in collection
            collection.analysis_results['bibtex'] = {
                'entries_count': len(bibtex_entries),
                'filename': filename,
                'content': full_bibtex
            }
            
            logger.info(f"✅ Generated {len(bibtex_entries)} BibTeX entries")
            
            return collection
        
        return _format_bibtex
    
    @staticmethod
    def network_analysis() -> Callable[[PaperCollection], PaperCollection]:
        """
        Analyze citation networks and author collaboration patterns
        
        Usage:
            papers("networks") | discover.arxiv(10) | cite.extract_references() | cite.network_analysis()
        """
        
        def _network_analysis(collection: PaperCollection) -> PaperCollection:
            logger.info("🕸️  Analyzing citation networks...")
            
            network_analysis = {
                'author_collaborations': {},
                'citation_clusters': [],
                'influential_papers': [],
                'research_groups': [],
                'cross_references': 0
            }
            
            # Build author collaboration network
            collaborations = {}
            for paper in collection.papers:
                if len(paper.authors) > 1:
                    # Create collaboration edges
                    for i, author1 in enumerate(paper.authors):
                        for author2 in paper.authors[i+1:]:
                            pair = tuple(sorted([author1, author2]))
                            collaborations[pair] = collaborations.get(pair, 0) + 1
            
            # Sort by collaboration frequency
            sorted_collaborations = sorted(collaborations.items(), key=lambda x: x[1], reverse=True)
            network_analysis['author_collaborations'] = dict(sorted_collaborations[:20])
            
            # Find research groups (authors who collaborate frequently)
            research_groups = []
            for (author1, author2), count in sorted_collaborations[:10]:
                if count >= 2:  # At least 2 collaborations
                    research_groups.append({
                        'members': [author1, author2],
                        'collaboration_count': count
                    })
            network_analysis['research_groups'] = research_groups
            
            # Analyze cross-references within collection
            paper_titles = {paper.title.lower(): paper for paper in collection.papers}
            cross_refs = 0
            
            for paper in collection.papers:
                if paper.references:
                    for ref in paper.references:
                        ref_title = _extract_title_from_reference(ref)
                        if ref_title and ref_title.lower() in paper_titles:
                            cross_refs += 1
            
            network_analysis['cross_references'] = cross_refs
            
            # Identify potentially influential papers (many references, well-cited authors)
            influential_scores = {}
            for paper in collection.papers:
                score = 0
                score += len(paper.references) * 0.1  # Papers with many references
                score += len(paper.authors) * 0.05    # Collaborative papers
                
                # Bonus for well-known venues in categories
                if any('nips' in cat.lower() or 'icml' in cat.lower() or 'iclr' in cat.lower() 
                       for cat in paper.categories):
                    score += 1
                
                influential_scores[paper.title] = score
            
            # Sort by influence score
            sorted_influential = sorted(influential_scores.items(), key=lambda x: x[1], reverse=True)
            network_analysis['influential_papers'] = sorted_influential[:10]
            
            collection.analysis_results['citation_network'] = network_analysis
            
            logger.info(f"✅ Network analysis complete: {len(network_analysis['author_collaborations'])} collaborations, {cross_refs} cross-references")
            
            return collection
        
        return _network_analysis
    
    @staticmethod
    def export_references(format: str = "apa", filename: str = None) -> Callable[[PaperCollection], PaperCollection]:
        """
        Export references in various citation formats
        
        Args:
            format: Citation format ("apa", "mla", "chicago", "ieee")
            filename: Optional file to save formatted references
            
        Usage:
            papers("ai") | discover.arxiv(3) | cite.export_references("apa", "references.txt")
        """
        
        def _export_references(collection: PaperCollection) -> PaperCollection:
            logger.info(f"📄 Exporting references in {format.upper()} format...")
            
            formatted_references = []
            
            for paper in collection.papers:
                if format.lower() == "apa":
                    ref = _format_apa_reference(paper)
                elif format.lower() == "mla":
                    ref = _format_mla_reference(paper)
                elif format.lower() == "chicago":
                    ref = _format_chicago_reference(paper)
                elif format.lower() == "ieee":
                    ref = _format_ieee_reference(paper)
                else:
                    ref = _format_apa_reference(paper)  # Default to APA
                
                formatted_references.append(ref)
            
            # Combine all references
            full_reference_list = "\n\n".join(formatted_references)
            
            # Save to file if specified
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(full_reference_list)
                logger.info(f"💾 References saved to: {filename}")
            
            # Store in collection
            collection.analysis_results['formatted_references'] = {
                'format': format,
                'count': len(formatted_references),
                'filename': filename,
                'content': full_reference_list
            }
            
            logger.info(f"✅ Exported {len(formatted_references)} references in {format.upper()} format")
            
            return collection
        
        return _export_references

# Helper functions for citation extraction and formatting
def _extract_references_from_content(content: str) -> List[str]:
    """Extract reference list from paper content"""
    references = []
    
    # Look for references section
    ref_patterns = [
        r'references\s*\n(.*?)(?=\n[A-Z][^.]*\n|\n\s*$)',
        r'bibliography\s*\n(.*?)(?=\n[A-Z][^.]*\n|\n\s*$)',
        r'works cited\s*\n(.*?)(?=\n[A-Z][^.]*\n|\n\s*$)'
    ]
    
    for pattern in ref_patterns:
        matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
        for match in matches:
            ref_section = match.group(1)
            # Split into individual references
            ref_lines = re.split(r'\n(?=[A-Z][^.]*[A-Z])', ref_section)
            references.extend([ref.strip() for ref in ref_lines if ref.strip()])
    
    return references

def _extract_inline_citations(content: str) -> List[str]:
    """Extract in-text citations from content"""
    citations = []
    
    # Common citation patterns
    citation_patterns = [
        r'\([A-Z][a-z]+(?:\s+et\s+al\.)?(?:,\s*\d{4})*\)',  # (Author, 2020)
        r'\[[A-Z][a-z]+(?:\s+et\s+al\.)?(?:,\s*\d{4})*\]',  # [Author, 2020]
        r'\([A-Z][a-z]+\s+and\s+[A-Z][a-z]+,\s*\d{4}\)'     # (Author and Author, 2020)
    ]
    
    for pattern in citation_patterns:
        matches = re.findall(pattern, content)
        citations.extend(matches)
    
    return list(set(citations))  # Remove duplicates

def _classify_reference(reference: str) -> str:
    """Classify reference type"""
    ref_lower = reference.lower()
    
    if 'arxiv' in ref_lower:
        return 'arxiv_preprint'
    elif any(venue in ref_lower for venue in ['conference', 'proceedings', 'workshop']):
        return 'conference'
    elif any(venue in ref_lower for venue in ['journal', 'transactions', 'letters']):
        return 'journal'
    elif 'thesis' in ref_lower or 'dissertation' in ref_lower:
        return 'thesis'
    elif any(venue in ref_lower for venue in ['book', 'chapter']):
        return 'book'
    elif 'url' in ref_lower or 'http' in ref_lower:
        return 'web_resource'
    else:
        return 'other'

def _extract_venue_from_reference(reference: str) -> Optional[str]:
    """Extract venue name from reference"""
    # Simple venue extraction patterns
    venue_patterns = [
        r'In\s+([^,]+(?:Conference|Workshop|Symposium|Meeting))',
        r'([A-Z][A-Za-z\s]+(?:Journal|Transactions|Letters|Review))',
        r'Proceedings\s+of\s+([^,]+)'
    ]
    
    for pattern in venue_patterns:
        match = re.search(pattern, reference, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return None

def _extract_year_from_reference(reference: str) -> Optional[str]:
    """Extract publication year from reference"""
    year_match = re.search(r'\b(19|20)\d{2}\b', reference)
    return year_match.group(0) if year_match else None

def _extract_title_from_reference(reference: str) -> Optional[str]:
    """Extract title from reference"""
    # Simple title extraction (between quotes or after author names)
    title_patterns = [
        r'"([^"]+)"',
        r"'([^']+)'",
        r'[A-Z][a-z]+(?:\s+et\s+al\.)?(?:,\s*\d{4})?[.,]\s*([A-Z][^.]+)\.'
    ]
    
    for pattern in title_patterns:
        match = re.search(pattern, reference)
        if match:
            return match.group(1).strip()
    
    return None

def _format_apa_reference(paper: Paper) -> str:
    """Format paper as APA reference"""
    authors = ", ".join(paper.authors) if paper.authors else "Anonymous"
    year = f"({paper.published_date[:4]})" if paper.published_date else "(n.d.)"
    title = paper.title
    
    ref = f"{authors} {year}. {title}."
    
    if paper.arxiv_id:
        ref += f" arXiv preprint arXiv:{paper.arxiv_id}."
    
    return ref

def _format_mla_reference(paper: Paper) -> str:
    """Format paper as MLA reference"""
    if paper.authors:
        first_author = paper.authors[0]
        if len(paper.authors) > 1:
            authors = f"{first_author}, et al."
        else:
            authors = first_author
    else:
        authors = "Anonymous"
    
    title = f'"{paper.title}"'
    year = paper.published_date[:4] if paper.published_date else "n.d."
    
    ref = f"{authors}. {title} {year}."
    
    if paper.pdf_url:
        ref += f" Web. {datetime.now().strftime('%d %b %Y')}."
    
    return ref

def _format_chicago_reference(paper: Paper) -> str:
    """Format paper as Chicago reference"""
    authors = ", ".join(paper.authors) if paper.authors else "Anonymous"
    title = f'"{paper.title}"'
    year = paper.published_date[:4] if paper.published_date else "n.d."
    
    ref = f"{authors}. {title} ({year})."
    
    if paper.arxiv_id:
        ref += f" arXiv preprint arXiv:{paper.arxiv_id}."
    
    return ref

def _format_ieee_reference(paper: Paper) -> str:
    """Format paper as IEEE reference"""
    authors = ", ".join(paper.authors) if paper.authors else "Anonymous"
    title = f'"{paper.title},"'
    year = paper.published_date[:4] if paper.published_date else "n.d."
    
    ref = f"{authors}, {title} {year}."
    
    if paper.arxiv_id:
        ref += f" arXiv:{paper.arxiv_id}."
    
    return ref

# Create singleton instance for TidyLLM-style access
cite = CitationOperations()

__all__ = [
    'cite',
    'CitationOperations'
]