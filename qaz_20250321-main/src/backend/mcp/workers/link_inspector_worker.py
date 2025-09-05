#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Link Inspector Worker

Specialized worker for extracting, validating, and analyzing links from documents.
This enables systematic link validation, broken link detection, and link quality assessment.
"""

import logging
import re
import json
import requests
import urllib.parse
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logger = logging.getLogger(__name__)

@dataclass
class Link:
    """Represents a single link found in a document"""
    link_id: str
    raw_text: str
    url: str
    link_text: str
    link_type: str  # 'http', 'https', 'ftp', 'mailto', 'internal', 'file'
    context: str  # Surrounding text for context
    page_number: Optional[int]
    section: Optional[str]
    confidence_score: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'link_id': self.link_id,
            'raw_text': self.raw_text,
            'url': self.url,
            'link_text': self.link_text,
            'link_type': self.link_type,
            'context': self.context,
            'page_number': self.page_number,
            'section': self.section,
            'confidence_score': self.confidence_score,
            'metadata': self.metadata
        }

@dataclass
class LinkValidationResult:
    """Represents the validation result for a single link"""
    link_id: str
    url: str
    is_valid: bool
    status_code: Optional[int]
    response_time: Optional[float]
    error_message: Optional[str]
    redirect_url: Optional[str]
    content_type: Optional[str]
    validation_timestamp: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'link_id': self.link_id,
            'url': self.url,
            'is_valid': self.is_valid,
            'status_code': self.status_code,
            'response_time': self.response_time,
            'error_message': self.error_message,
            'redirect_url': self.redirect_url,
            'content_type': self.content_type,
            'validation_timestamp': self.validation_timestamp,
            'metadata': self.metadata
        }

@dataclass
class LinkAnalysisStructure:
    """Represents the complete link analysis structure"""
    document_id: str
    document_title: str
    total_links: int
    valid_links: int
    broken_links: int
    links: List[Link]
    validation_results: List[LinkValidationResult]
    extraction_method: str
    confidence_score: float
    link_sections: Dict[str, List[str]]  # section -> list of link_ids
    metadata: Dict[str, Any]
    created_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'document_id': self.document_id,
            'document_title': self.document_title,
            'total_links': self.total_links,
            'valid_links': self.valid_links,
            'broken_links': self.broken_links,
            'links': [link.to_dict() for link in self.links],
            'validation_results': [result.to_dict() for result in self.validation_results],
            'extraction_method': self.extraction_method,
            'confidence_score': self.confidence_score,
            'link_sections': self.link_sections,
            'metadata': self.metadata,
            'created_at': self.created_at
        }

class LinkInspectorWorker:
    """Specialized worker for extracting and validating links from documents"""
    
    def __init__(self, max_workers: int = 10, timeout: int = 10):
        self.available_methods = self._check_available_methods()
        self.link_cache = {}
        self.patterns = self._initialize_patterns()
        self.max_workers = max_workers
        self.timeout = timeout
        self.session = self._create_session()
    
    def _check_available_methods(self) -> Dict[str, bool]:
        """Check which link extraction methods are available"""
        methods = {}
        
        # Check for requests (for link validation)
        try:
            import requests
            methods['requests'] = True
        except ImportError:
            methods['requests'] = False
        
        # Check for beautifulsoup (for HTML parsing)
        try:
            import bs4
            methods['beautifulsoup'] = True
        except ImportError:
            methods['beautifulsoup'] = False
        
        # Check for pdfplumber (for PDF link extraction)
        try:
            import pdfplumber
            methods['pdfplumber'] = True
        except ImportError:
            methods['pdfplumber'] = False
        
        # Check for fitz (PyMuPDF)
        try:
            import fitz
            methods['fitz'] = True
        except ImportError:
            methods['fitz'] = False
        
        logger.info(f"Available link extraction methods: {methods}")
        return methods
    
    def _create_session(self) -> Optional[requests.Session]:
        """Create a requests session for link validation"""
        if not self.available_methods.get('requests'):
            return None
        
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; LinkInspector/1.0)'
        })
        return session
    
    def _initialize_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize regex patterns for link detection"""
        patterns = {
            # HTTP/HTTPS URLs
            'http_url': re.compile(
                r'https?://[^\s<>"{}|\\^`\[\]]+',
                re.IGNORECASE
            ),
            
            # URLs with common TLDs
            'url_with_tld': re.compile(
                r'https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s<>"{}|\\^`\[\]]*)?',
                re.IGNORECASE
            ),
            
            # Email addresses
            'email': re.compile(
                r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
                re.IGNORECASE
            ),
            
            # Internal references (like #section, page 5, etc.)
            'internal_ref': re.compile(
                r'(?:page|section|chapter|figure|table)\s+\d+',
                re.IGNORECASE
            ),
            
            # File paths
            'file_path': re.compile(
                r'(?:file://|ftp://)[^\s<>"{}|\\^`\[\]]+',
                re.IGNORECASE
            ),
            
            # DOI references
            'doi': re.compile(
                r'doi:?\s*10\.\d{4,}/[^\s]+',
                re.IGNORECASE
            ),
            
            # arXiv references
            'arxiv': re.compile(
                r'arxiv\.org/(?:abs|pdf)/[^\s]+',
                re.IGNORECASE
            )
        }
        
        return patterns
    
    def extract_links_from_text(self, text: str, document_id: str, page_number: Optional[int] = None, section: Optional[str] = None) -> List[Link]:
        """Extract links from plain text"""
        links = []
        
        # Extract HTTP/HTTPS URLs
        for match in self.patterns['http_url'].finditer(text):
            url = match.group()
            link_id = self._generate_link_id(document_id, url, match.start())
            
            # Get context (surrounding text)
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end].strip()
            
            # Extract link text (try to find descriptive text)
            link_text = self._extract_link_text(text, match.start(), match.end())
            
            link = Link(
                link_id=link_id,
                raw_text=match.group(),
                url=url,
                link_text=link_text,
                link_type=self._determine_link_type(url),
                context=context,
                page_number=page_number,
                section=section,
                confidence_score=0.9,
                metadata={'extraction_method': 'regex', 'position': match.start()}
            )
            links.append(link)
        
        # Extract email addresses
        for match in self.patterns['email'].finditer(text):
            email = match.group()
            link_id = self._generate_link_id(document_id, email, match.start())
            
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            context = text[start:end].strip()
            
            link = Link(
                link_id=link_id,
                raw_text=match.group(),
                url=f"mailto:{email}",
                link_text=email,
                link_type='mailto',
                context=context,
                page_number=page_number,
                section=section,
                confidence_score=0.95,
                metadata={'extraction_method': 'regex', 'position': match.start()}
            )
            links.append(link)
        
        # Extract DOIs
        for match in self.patterns['doi'].finditer(text):
            doi = match.group()
            link_id = self._generate_link_id(document_id, doi, match.start())
            
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            context = text[start:end].strip()
            
            # Convert DOI to URL
            doi_url = doi.replace('doi:', 'https://doi.org/')
            
            link = Link(
                link_id=link_id,
                raw_text=match.group(),
                url=doi_url,
                link_text=doi,
                link_type='doi',
                context=context,
                page_number=page_number,
                section=section,
                confidence_score=0.9,
                metadata={'extraction_method': 'regex', 'position': match.start()}
            )
            links.append(link)
        
        return links
    
    def extract_links_from_pdf(self, pdf_path: str, document_id: str) -> List[Link]:
        """Extract links from PDF document"""
        links = []
        
        if self.available_methods.get('pdfplumber'):
            links.extend(self._extract_links_with_pdfplumber(pdf_path, document_id))
        elif self.available_methods.get('fitz'):
            links.extend(self._extract_links_with_fitz(pdf_path, document_id))
        
        return links
    
    def _extract_links_with_pdfplumber(self, pdf_path: str, document_id: str) -> List[Link]:
        """Extract links using pdfplumber"""
        links = []
        
        try:
            import pdfplumber
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract text
                    text = page.extract_text() or ""
                    
                    # Extract links from text
                    page_links = self.extract_links_from_text(
                        text, document_id, page_num, f"page_{page_num}"
                    )
                    links.extend(page_links)
                    
                    # Extract links from annotations (if available)
                    if hasattr(page, 'annots') and page.annots:
                        for annot in page.annots:
                            if annot.get('subtype') == 'Link':
                                link_url = annot.get('A', {}).get('URI')
                                if link_url:
                                    link_id = self._generate_link_id(document_id, link_url, 0)
                                    
                                    link = Link(
                                        link_id=link_id,
                                        raw_text=link_url,
                                        url=link_url,
                                        link_text=link_url,
                                        link_type=self._determine_link_type(link_url),
                                        context=f"PDF annotation on page {page_num}",
                                        page_number=page_num,
                                        section=f"page_{page_num}",
                                        confidence_score=0.95,
                                        metadata={'extraction_method': 'pdf_annotation', 'page': page_num}
                                    )
                                    links.append(link)
        
        except Exception as e:
            logger.error(f"Error extracting links with pdfplumber: {e}")
        
        return links
    
    def _extract_links_with_fitz(self, pdf_path: str, document_id: str) -> List[Link]:
        """Extract links using PyMuPDF (fitz)"""
        links = []
        
        try:
            import fitz
            
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text
                text = page.get_text()
                
                # Extract links from text
                page_links = self.extract_links_from_text(
                    text, document_id, page_num + 1, f"page_{page_num + 1}"
                )
                links.extend(page_links)
                
                # Extract links from annotations
                annots = page.get_annotations()
                for annot in annots:
                    if annot.type[0] == 8:  # Link annotation
                        link_url = annot.get_uri()
                        if link_url:
                            link_id = self._generate_link_id(document_id, link_url, 0)
                            
                            link = Link(
                                link_id=link_id,
                                raw_text=link_url,
                                url=link_url,
                                link_text=link_url,
                                link_type=self._determine_link_type(link_url),
                                context=f"PDF annotation on page {page_num + 1}",
                                page_number=page_num + 1,
                                section=f"page_{page_num + 1}",
                                confidence_score=0.95,
                                metadata={'extraction_method': 'pdf_annotation', 'page': page_num + 1}
                            )
                            links.append(link)
            
            doc.close()
        
        except Exception as e:
            logger.error(f"Error extracting links with fitz: {e}")
        
        return links
    
    def validate_links(self, links: List[Link], validate_external: bool = True) -> List[LinkValidationResult]:
        """Validate a list of links"""
        validation_results = []
        
        if not validate_external:
            # Skip external validation, mark all as valid
            for link in links:
                result = LinkValidationResult(
                    link_id=link.link_id,
                    url=link.url,
                    is_valid=True,
                    status_code=None,
                    response_time=None,
                    error_message=None,
                    redirect_url=None,
                    content_type=None,
                    validation_timestamp=datetime.now().isoformat(),
                    metadata={'validation_skipped': True}
                )
                validation_results.append(result)
            return validation_results
        
        # Validate links in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_link = {
                executor.submit(self._validate_single_link, link): link 
                for link in links
            }
            
            for future in as_completed(future_to_link):
                link = future_to_link[future]
                try:
                    result = future.result()
                    validation_results.append(result)
                except Exception as e:
                    logger.error(f"Error validating link {link.url}: {e}")
                    result = LinkValidationResult(
                        link_id=link.link_id,
                        url=link.url,
                        is_valid=False,
                        status_code=None,
                        response_time=None,
                        error_message=str(e),
                        redirect_url=None,
                        content_type=None,
                        validation_timestamp=datetime.now().isoformat(),
                        metadata={'validation_error': True}
                    )
                    validation_results.append(result)
        
        return validation_results
    
    def _validate_single_link(self, link: Link) -> LinkValidationResult:
        """Validate a single link"""
        start_time = time.time()
        
        try:
            # Skip validation for certain link types
            if link.link_type in ['mailto', 'internal', 'file']:
                return LinkValidationResult(
                    link_id=link.link_id,
                    url=link.url,
                    is_valid=True,
                    status_code=None,
                    response_time=time.time() - start_time,
                    error_message=None,
                    redirect_url=None,
                    content_type=None,
                    validation_timestamp=datetime.now().isoformat(),
                    metadata={'link_type': link.link_type, 'validation_skipped': True}
                )
            
            # Validate external links
            if self.session:
                response = self.session.head(
                    link.url, 
                    timeout=self.timeout,
                    allow_redirects=True
                )
                
                return LinkValidationResult(
                    link_id=link.link_id,
                    url=link.url,
                    is_valid=response.status_code < 400,
                    status_code=response.status_code,
                    response_time=time.time() - start_time,
                    error_message=None if response.status_code < 400 else f"HTTP {response.status_code}",
                    redirect_url=response.url if response.url != link.url else None,
                    content_type=response.headers.get('content-type'),
                    validation_timestamp=datetime.now().isoformat(),
                    metadata={'link_type': link.link_type}
                )
            else:
                return LinkValidationResult(
                    link_id=link.link_id,
                    url=link.url,
                    is_valid=False,
                    status_code=None,
                    response_time=time.time() - start_time,
                    error_message="No HTTP client available",
                    redirect_url=None,
                    content_type=None,
                    validation_timestamp=datetime.now().isoformat(),
                    metadata={'link_type': link.link_type, 'no_client': True}
                )
        
        except requests.exceptions.RequestException as e:
            return LinkValidationResult(
                link_id=link.link_id,
                url=link.url,
                is_valid=False,
                status_code=None,
                response_time=time.time() - start_time,
                error_message=str(e),
                redirect_url=None,
                content_type=None,
                validation_timestamp=datetime.now().isoformat(),
                metadata={'link_type': link.link_type, 'exception': type(e).__name__}
            )
        except Exception as e:
            return LinkValidationResult(
                link_id=link.link_id,
                url=link.url,
                is_valid=False,
                status_code=None,
                response_time=time.time() - start_time,
                error_message=str(e),
                redirect_url=None,
                content_type=None,
                validation_timestamp=datetime.now().isoformat(),
                metadata={'link_type': link.link_type, 'exception': type(e).__name__}
            )
    
    def analyze_document_links(self, document_path: str, document_id: str, document_title: str = "", validate_links: bool = True) -> LinkAnalysisStructure:
        """Analyze links in a document"""
        
        # Check cache
        cache_key = f"{document_id}_{hash(document_path)}"
        if cache_key in self.link_cache:
            logger.info(f"Using cached link analysis for {document_id}")
            return self.link_cache[cache_key]
        
        logger.info(f"Starting link analysis for document: {document_path}")
        
        # Extract links based on file type
        if document_path.lower().endswith('.pdf'):
            links = self.extract_links_from_pdf(document_path, document_id)
        else:
            # For text files, read and extract links
            try:
                with open(document_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                links = self.extract_links_from_text(text, document_id)
            except Exception as e:
                logger.error(f"Error reading document {document_path}: {e}")
                links = []
        
        # Validate links
        validation_results = self.validate_links(links, validate_links) if links else []
        
        # Count valid and broken links
        valid_links = sum(1 for result in validation_results if result.is_valid)
        broken_links = len(validation_results) - valid_links
        
        # Group links by section
        link_sections = {}
        for link in links:
            section = link.section or "unknown"
            if section not in link_sections:
                link_sections[section] = []
            link_sections[section].append(link.link_id)
        
        # Create analysis structure
        analysis = LinkAnalysisStructure(
            document_id=document_id,
            document_title=document_title,
            total_links=len(links),
            valid_links=valid_links,
            broken_links=broken_links,
            links=links,
            validation_results=validation_results,
            extraction_method="link_inspector_worker",
            confidence_score=self._calculate_confidence_score(links, validation_results),
            link_sections=link_sections,
            metadata={
                'file_path': document_path,
                'extraction_timestamp': datetime.now().isoformat(),
                'validation_enabled': validate_links
            },
            created_at=datetime.now().isoformat()
        )
        
        # Cache the result
        self.link_cache[cache_key] = analysis
        
        logger.info(f"Link analysis completed for {document_id}: {len(links)} links found, {valid_links} valid, {broken_links} broken")
        
        return analysis
    
    def _generate_link_id(self, document_id: str, url: str, position: int) -> str:
        """Generate a unique link ID"""
        content = f"{document_id}_{url}_{position}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _extract_link_text(self, text: str, start: int, end: int) -> str:
        """Extract descriptive text for a link"""
        # Look for text before the link (common pattern: "see <link>")
        before_text = text[max(0, start-20):start].strip()
        after_text = text[end:min(len(text), end+20)].strip()
        
        # Try to find meaningful text
        if before_text and not before_text.endswith('http'):
            return before_text.split()[-1] if before_text.split() else ""
        elif after_text and not after_text.startswith('http'):
            return after_text.split()[0] if after_text.split() else ""
        else:
            return ""
    
    def _determine_link_type(self, url: str) -> str:
        """Determine the type of link"""
        url_lower = url.lower()
        
        if url_lower.startswith('mailto:'):
            return 'mailto'
        elif url_lower.startswith('file://') or url_lower.startswith('ftp://'):
            return 'file'
        elif url_lower.startswith('https://'):
            return 'https'
        elif url_lower.startswith('http://'):
            return 'http'
        elif url_lower.startswith('doi.org') or 'doi:' in url_lower:
            return 'doi'
        elif 'arxiv.org' in url_lower:
            return 'arxiv'
        else:
            return 'unknown'
    
    def _calculate_confidence_score(self, links: List[Link], validation_results: List[LinkValidationResult]) -> float:
        """Calculate overall confidence score for the analysis"""
        if not links:
            return 0.0
        
        # Base confidence on extraction quality
        avg_extraction_confidence = sum(link.confidence_score for link in links) / len(links)
        
        # Adjust based on validation results
        if validation_results:
            validation_success_rate = sum(1 for r in validation_results if r.is_valid) / len(validation_results)
            return (avg_extraction_confidence + validation_success_rate) / 2
        else:
            return avg_extraction_confidence
    
    def get_broken_links_report(self, analysis: LinkAnalysisStructure) -> Dict[str, Any]:
        """Generate a report of broken links"""
        broken_links = [
            result for result in analysis.validation_results 
            if not result.is_valid
        ]
        
        return {
            'document_id': analysis.document_id,
            'document_title': analysis.document_title,
            'total_links': analysis.total_links,
            'broken_links_count': len(broken_links),
            'broken_links_percentage': (len(broken_links) / analysis.total_links * 100) if analysis.total_links > 0 else 0,
            'broken_links': [
                {
                    'url': result.url,
                    'error_message': result.error_message,
                    'link_text': next((link.link_text for link in analysis.links if link.link_id == result.link_id), ''),
                    'context': next((link.context for link in analysis.links if link.link_id == result.link_id), ''),
                    'page_number': next((link.page_number for link in analysis.links if link.link_id == result.link_id), None),
                    'section': next((link.section for link in analysis.links if link.link_id == result.link_id), None)
                }
                for result in broken_links
            ],
            'validation_timestamp': datetime.now().isoformat()
        }
    
    def save_analysis_to_json(self, analysis: LinkAnalysisStructure, output_path: str) -> bool:
        """Save link analysis to JSON file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Link analysis saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving link analysis to {output_path}: {e}")
            return False
    
    def load_analysis_from_json(self, json_path: str) -> Optional[LinkAnalysisStructure]:
        """Load link analysis from JSON file"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Reconstruct the analysis structure
            links = [Link(**link_data) for link_data in data.get('links', [])]
            validation_results = [LinkValidationResult(**result_data) for result_data in data.get('validation_results', [])]
            
            analysis = LinkAnalysisStructure(
                document_id=data['document_id'],
                document_title=data['document_title'],
                total_links=data['total_links'],
                valid_links=data['valid_links'],
                broken_links=data['broken_links'],
                links=links,
                validation_results=validation_results,
                extraction_method=data['extraction_method'],
                confidence_score=data['confidence_score'],
                link_sections=data['link_sections'],
                metadata=data['metadata'],
                created_at=data['created_at']
            )
            
            return analysis
        except Exception as e:
            logger.error(f"Error loading link analysis from {json_path}: {e}")
            return None
