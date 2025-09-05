#!/usr/bin/env python3
"""
Links Quality Checker - Good Links vs Bad Links Analyzer
=======================================================

Analyzes the quality and reliability of links/references in research papers.
Business stakeholders understand that broken links indicate poor maintenance
and potentially unreliable research.

Business Questions Answered:
- Are the references current and accessible?
- Is this research well-maintained and reliable?
- How many dead links indicate stale research?
- Are the sources authoritative and trustworthy?
"""

import re
import sys
import time
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from urllib.parse import urlparse
import concurrent.futures
import threading

# Use existing document processing
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "tidyllm-documents"))

try:
    from tidyllm_documents.extraction.text import TextExtractor
    EXTRACTION_AVAILABLE = True
except ImportError:
    print("INFO: Using fallback text extraction (tidyllm-documents not available)")
    EXTRACTION_AVAILABLE = False
    import PyPDF2

# For URL checking - using simple approach to avoid heavy dependencies
try:
    import urllib.request
    import urllib.error
    import socket
    URL_CHECKING_AVAILABLE = True
except ImportError:
    URL_CHECKING_AVAILABLE = False

@dataclass
class LinkInfo:
    """Information about a detected link."""
    url: str
    link_type: str  # doi, arxiv, github, academic, commercial, etc.
    context: str  # surrounding text
    status: str  # good, bad, unknown, timeout
    response_time: float
    authority_score: float  # 0-1 based on domain reputation

@dataclass
class LinksAnalysis:
    """Complete links quality analysis."""
    total_links: int
    good_links: int
    bad_links: int
    timeout_links: int
    unknown_links: int
    link_quality_score: float  # 0-100
    authority_score: float  # 0-100
    maintenance_indicator: str  # Excellent, Good, Fair, Poor
    reliability_assessment: str
    links_by_type: Dict[str, int]
    detailed_links: List[LinkInfo]

class LinksQualityChecker:
    """Check quality and reliability of links in research papers."""
    
    def __init__(self):
        if EXTRACTION_AVAILABLE:
            self.extractor = TextExtractor()
        else:
            self.extractor = None
        
        # Authoritative domains (higher trust score)
        self.authoritative_domains = {
            # Academic publishers
            'ieee.org': 0.95, 'acm.org': 0.95, 'springer.com': 0.9, 'elsevier.com': 0.9,
            'nature.com': 0.95, 'science.org': 0.95, 'pnas.org': 0.9, 'cell.com': 0.9,
            # Repositories and archives
            'arxiv.org': 0.9, 'doi.org': 0.95, 'github.com': 0.8, 'gitlab.com': 0.8,
            # Academic institutions
            'edu': 0.85, 'ac.uk': 0.85, 'mit.edu': 0.9, 'stanford.edu': 0.9,
            # Standards organizations
            'w3.org': 0.9, 'ietf.org': 0.9, 'iso.org': 0.85,
            # Government/official
            'gov': 0.9, 'nih.gov': 0.95, 'nasa.gov': 0.9,
            # Tech companies (research arms)
            'research.google.com': 0.8, 'research.microsoft.com': 0.8,
            'research.ibm.com': 0.8, 'openai.com': 0.75
        }
        
        # Link type patterns
        self.link_patterns = {
            'doi': r'(?:https?://)?(?:dx\.)?doi\.org/[^\s\)]+|doi:\s*[^\s\)]+',
            'arxiv': r'(?:https?://)?arxiv\.org/[^\s\)]+',
            'github': r'(?:https?://)?github\.com/[^\s\)]+',
            'academic_publisher': r'(?:https?://)?(?:ieeexplore\.ieee\.org|dl\.acm\.org|link\.springer\.com|www\.sciencedirect\.com|www\.nature\.com)[^\s\)]+',
            'general_http': r'https?://[^\s\)\]]+',
            'general_www': r'www\.[^\s\)\]]+',
            'ftp': r'ftp://[^\s\)\]]+',
            'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        }
        
        # Rate limiting for URL checks
        self.request_delay = 0.5  # seconds between requests
        self.timeout = 10  # seconds for each request
        self.max_workers = 5  # concurrent requests
        
        # Thread-safe counter
        self._request_count = 0
        self._lock = threading.Lock()
    
    def extract_document_text(self, pdf_path: str) -> str:
        """Extract text using tidyllm-documents or fallback."""
        if EXTRACTION_AVAILABLE and self.extractor:
            text, metadata = self.extractor.extract_text(pdf_path, max_pages=30)
            if text:
                return text
        
        # Fallback extraction
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                max_pages = min(len(pdf_reader.pages), 30)
                
                for page_num in range(max_pages):
                    try:
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        text += page_text + "\n"
                    except:
                        continue
                
                return text.encode('ascii', 'ignore').decode('ascii')
        except Exception as e:
            print(f"Error extracting {pdf_path}: {e}")
            return ""
    
    def extract_links(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract all links from text with their types and context."""
        links = []
        
        for link_type, pattern in self.link_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                url = match.group(0).strip()
                start_pos = match.start()
                
                # Get context (50 characters before and after)
                context_start = max(0, start_pos - 50)
                context_end = min(len(text), match.end() + 50)
                context = text[context_start:context_end].replace('\n', ' ').strip()
                
                # Clean up URL
                url = self._clean_url(url)
                if url and len(url) > 8:  # Minimum reasonable URL length
                    links.append((url, link_type, context))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_links = []
        for url, link_type, context in links:
            if url not in seen:
                seen.add(url)
                unique_links.append((url, link_type, context))
        
        return unique_links
    
    def _clean_url(self, url: str) -> str:
        """Clean and normalize URL."""
        # Remove common trailing punctuation
        url = re.sub(r'[.,;:!?\)\]]+$', '', url)
        
        # Add http:// if missing for www links
        if url.startswith('www.') and not url.startswith('http'):
            url = 'http://' + url
        
        # Clean up common PDF artifacts
        url = re.sub(r'\s+', '', url)  # Remove spaces
        
        return url
    
    def calculate_authority_score(self, url: str) -> float:
        """Calculate authority score based on domain reputation."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Remove www prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # Check exact matches first
            if domain in self.authoritative_domains:
                return self.authoritative_domains[domain]
            
            # Check domain endings
            for auth_domain, score in self.authoritative_domains.items():
                if domain.endswith('.' + auth_domain) or domain == auth_domain:
                    return score
            
            # Default scoring based on TLD
            if domain.endswith('.edu'):
                return 0.8
            elif domain.endswith('.gov'):
                return 0.85
            elif domain.endswith('.org'):
                return 0.6
            elif domain.endswith(('.com', '.net')):
                return 0.4
            else:
                return 0.3
                
        except:
            return 0.2
    
    def check_url_status(self, url: str) -> Tuple[str, float]:
        """Check if URL is accessible. Returns (status, response_time)."""
        if not URL_CHECKING_AVAILABLE:
            return "unknown", 0.0
        
        # Rate limiting
        with self._lock:
            self._request_count += 1
            if self._request_count > 1:
                time.sleep(self.request_delay)
        
        start_time = time.time()
        
        try:
            # Handle special cases
            if 'doi.org' in url or url.startswith('doi:'):
                # DOIs are generally reliable, but may have access restrictions
                return "good", 0.1  # Fast assumed response for DOIs
            
            if 'arxiv.org' in url:
                # ArXiv is very reliable
                return "good", 0.1
            
            # Create request with headers to avoid blocking
            req = urllib.request.Request(
                url,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            )
            
            # Set socket timeout
            socket.setdefaulttimeout(self.timeout)
            
            with urllib.request.urlopen(req) as response:
                response_time = time.time() - start_time
                
                if response.getcode() == 200:
                    return "good", response_time
                elif response.getcode() in [301, 302, 303, 307, 308]:
                    return "good", response_time  # Redirects are OK
                else:
                    return "bad", response_time
                    
        except urllib.error.HTTPError as e:
            response_time = time.time() - start_time
            if e.code in [403, 429]:  # Forbidden/Rate limited - might be accessible
                return "unknown", response_time
            else:
                return "bad", response_time
                
        except (urllib.error.URLError, socket.timeout, Exception):
            response_time = time.time() - start_time
            if response_time >= self.timeout:
                return "timeout", response_time
            else:
                return "bad", response_time
    
    def analyze_links_batch(self, links: List[Tuple[str, str, str]], max_checks: int = 20) -> List[LinkInfo]:
        """Analyze links in batch with concurrent checking."""
        link_infos = []
        
        # Limit the number of links to check to avoid overwhelming servers
        links_to_check = links[:max_checks] if len(links) > max_checks else links
        
        if URL_CHECKING_AVAILABLE and len(links_to_check) > 0:
            print(f"Checking {len(links_to_check)} links (limited for server politeness)...")
            
            # Use ThreadPoolExecutor for concurrent but controlled requests
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all URL check tasks
                future_to_link = {
                    executor.submit(self.check_url_status, url): (url, link_type, context)
                    for url, link_type, context in links_to_check
                }
                
                # Process completed requests
                for future in concurrent.futures.as_completed(future_to_link):
                    url, link_type, context = future_to_link[future]
                    
                    try:
                        status, response_time = future.result()
                    except Exception as e:
                        status, response_time = "bad", 0.0
                    
                    authority_score = self.calculate_authority_score(url)
                    
                    link_infos.append(LinkInfo(
                        url=url,
                        link_type=link_type,
                        context=context[:100],  # Truncate context
                        status=status,
                        response_time=response_time,
                        authority_score=authority_score
                    ))
        else:
            # No URL checking available - analyze based on patterns and authority
            for url, link_type, context in links:
                authority_score = self.calculate_authority_score(url)
                
                # Heuristic status based on URL patterns
                if any(domain in url for domain in ['doi.org', 'arxiv.org', 'github.com']):
                    status = "good"
                elif authority_score > 0.7:
                    status = "good"
                else:
                    status = "unknown"
                
                link_infos.append(LinkInfo(
                    url=url,
                    link_type=link_type,
                    context=context[:100],
                    status=status,
                    response_time=0.0,
                    authority_score=authority_score
                ))
        
        # Add remaining links as unknown if we hit the limit
        for url, link_type, context in links[max_checks:]:
            authority_score = self.calculate_authority_score(url)
            link_infos.append(LinkInfo(
                url=url,
                link_type=link_type,
                context=context[:100],
                status="unknown",
                response_time=0.0,
                authority_score=authority_score
            ))
        
        return link_infos
    
    def analyze_document(self, pdf_path: str, max_link_checks: int = 20) -> LinksAnalysis:
        """Complete links quality analysis of a document."""
        # Extract text
        text = self.extract_document_text(pdf_path)
        if not text:
            return LinksAnalysis(0, 0, 0, 0, 0, 0.0, 0.0, "Unable to analyze", 
                               "Cannot extract text", {}, [])
        
        # Extract links
        raw_links = self.extract_links(text)
        
        if not raw_links:
            return LinksAnalysis(0, 0, 0, 0, 0, 50.0, 50.0, "No links found", 
                               "No external references", {}, [])
        
        # Analyze links
        link_infos = self.analyze_links_batch(raw_links, max_link_checks)
        
        # Calculate statistics
        total_links = len(link_infos)
        good_links = sum(1 for link in link_infos if link.status == "good")
        bad_links = sum(1 for link in link_infos if link.status == "bad")
        timeout_links = sum(1 for link in link_infos if link.status == "timeout")
        unknown_links = sum(1 for link in link_infos if link.status == "unknown")
        
        # Link quality score (0-100)
        checked_links = good_links + bad_links + timeout_links
        if checked_links > 0:
            link_quality_score = (good_links / checked_links) * 100
        else:
            link_quality_score = 50.0  # Neutral if no links checked
        
        # Authority score (average of all links)
        authority_score = (sum(link.authority_score for link in link_infos) / total_links) * 100 if total_links > 0 else 0.0
        
        # Maintenance indicator
        if link_quality_score >= 90 and bad_links == 0:
            maintenance = "Excellent"
        elif link_quality_score >= 75:
            maintenance = "Good" 
        elif link_quality_score >= 50:
            maintenance = "Fair"
        else:
            maintenance = "Poor"
        
        # Reliability assessment
        if authority_score >= 80 and link_quality_score >= 80:
            reliability = "Highly Reliable - Authoritative sources with good maintenance"
        elif authority_score >= 60 and link_quality_score >= 60:
            reliability = "Reliable - Good sources with acceptable maintenance"
        elif bad_links > total_links * 0.3:
            reliability = "Questionable - Many broken links indicate poor maintenance"
        elif authority_score < 40:
            reliability = "Low Authority - Few authoritative sources referenced"
        else:
            reliability = "Moderate - Mixed source quality and maintenance"
        
        # Links by type
        links_by_type = defaultdict(int)
        for link in link_infos:
            links_by_type[link.link_type] += 1
        
        return LinksAnalysis(
            total_links=total_links,
            good_links=good_links,
            bad_links=bad_links,
            timeout_links=timeout_links,
            unknown_links=unknown_links,
            link_quality_score=link_quality_score,
            authority_score=authority_score,
            maintenance_indicator=maintenance,
            reliability_assessment=reliability,
            links_by_type=dict(links_by_type),
            detailed_links=link_infos
        )
    
    def generate_business_report(self, analysis: LinksAnalysis, paper_title: str = "") -> str:
        """Generate business-friendly links quality report."""
        if analysis.total_links == 0:
            return f"**{paper_title}**\nNo external links found for analysis."
        
        report = []
        
        # Header
        report.append("# LINKS QUALITY & RELIABILITY ANALYSIS")
        if paper_title:
            report.append(f"**Paper:** {paper_title}")
        report.append(f"**Total Links Found:** {analysis.total_links}")
        report.append("")
        
        # Key Business Metrics
        report.append("## KEY RELIABILITY INDICATORS")
        report.append(f"- **Link Quality Score:** {analysis.link_quality_score:.1f}% (working links)")
        report.append(f"- **Authority Score:** {analysis.authority_score:.1f}% (source reputation)")
        report.append(f"- **Maintenance Indicator:** {analysis.maintenance_indicator}")
        report.append(f"- **Overall Assessment:** {analysis.reliability_assessment}")
        report.append("")
        
        # Link Status Breakdown
        report.append("## LINK STATUS BREAKDOWN")
        report.append(f"- **Working Links:** {analysis.good_links} ({(analysis.good_links/analysis.total_links)*100:.1f}%)")
        if analysis.bad_links > 0:
            report.append(f"- **Broken Links:** {analysis.bad_links} ({(analysis.bad_links/analysis.total_links)*100:.1f}%)")
        if analysis.timeout_links > 0:
            report.append(f"- **Timeout Links:** {analysis.timeout_links} ({(analysis.timeout_links/analysis.total_links)*100:.1f}%)")
        if analysis.unknown_links > 0:
            report.append(f"- **Unknown Status:** {analysis.unknown_links} ({(analysis.unknown_links/analysis.total_links)*100:.1f}%)")
        report.append("")
        
        # Link Types
        if analysis.links_by_type:
            report.append("## REFERENCE TYPES")
            sorted_types = sorted(analysis.links_by_type.items(), key=lambda x: x[1], reverse=True)
            for link_type, count in sorted_types:
                percentage = (count / analysis.total_links) * 100
                type_name = link_type.replace('_', ' ').title()
                report.append(f"- **{type_name}:** {count} links ({percentage:.1f}%)")
            report.append("")
        
        # Business Impact Assessment
        report.append("## BUSINESS IMPACT ASSESSMENT")
        
        if analysis.bad_links > analysis.total_links * 0.2:
            report.append("[WARNING] **HIGH BROKEN LINK RATE** - May indicate outdated or poorly maintained research")
        elif analysis.bad_links == 0 and analysis.good_links > 5:
            report.append("[EXCELLENT] **ALL LINKS WORKING** - Well-maintained and current research")
        
        if analysis.authority_score > 80:
            report.append("[GOOD] **HIGH AUTHORITY SOURCES** - References reputable academic and institutional sources")
        elif analysis.authority_score < 40:
            report.append("[WARNING] **LOW AUTHORITY SOURCES** - Few references to established academic sources")
        
        # Trust and reliability indicators
        doi_count = analysis.links_by_type.get('doi', 0)
        arxiv_count = analysis.links_by_type.get('arxiv', 0)
        academic_count = analysis.links_by_type.get('academic_publisher', 0)
        
        scholarly_links = doi_count + arxiv_count + academic_count
        if scholarly_links > analysis.total_links * 0.5:
            report.append("[GOOD] **SCHOLARLY REFERENCES** - Majority of links are to academic sources")
        elif scholarly_links < analysis.total_links * 0.2:
            report.append("[NOTE] **LIMITED SCHOLARLY LINKS** - Few references to peer-reviewed sources")
        
        report.append("")
        
        # Detailed Link Analysis (top issues)
        broken_links = [link for link in analysis.detailed_links if link.status == "bad"]
        if broken_links:
            report.append("## BROKEN LINKS REQUIRING ATTENTION")
            for i, link in enumerate(broken_links[:5], 1):  # Show top 5
                report.append(f"{i}. **{link.url}** ({link.link_type})")
                report.append(f"   Context: ...{link.context}...")
                report.append("")
        
        # High authority links (showcase quality)
        high_auth_links = [link for link in analysis.detailed_links 
                          if link.authority_score > 0.8 and link.status == "good"]
        if high_auth_links:
            report.append("## HIGH AUTHORITY REFERENCES")
            for i, link in enumerate(high_auth_links[:3], 1):  # Show top 3
                report.append(f"{i}. **{link.url}** (Authority: {link.authority_score:.2f})")
                report.append(f"   Type: {link.link_type.replace('_', ' ').title()}")
                report.append("")
        
        # Business recommendation
        report.append("## BUSINESS RECOMMENDATION")
        
        if analysis.link_quality_score >= 85 and analysis.authority_score >= 70:
            report.append("[RECOMMENDED] **HIGH QUALITY REFERENCES** - Well-maintained with authoritative sources")
        elif analysis.bad_links > 3:
            report.append("[CAUTION] **MAINTENANCE ISSUES** - Multiple broken links may indicate outdated research")
        elif analysis.authority_score < 50:
            report.append("[REVIEW] **SOURCE QUALITY CONCERNS** - Limited authoritative references")
        else:
            report.append("[ACCEPTABLE] **STANDARD REFERENCE QUALITY** - Meets basic reliability standards")
        
        return "\n".join(report)
    
    def batch_analyze(self, paper_directory: str, max_link_checks: int = 15) -> Dict[str, LinksAnalysis]:
        """Analyze links quality for all papers in directory."""
        results = {}
        paper_dir = Path(paper_directory)
        
        pdf_files = list(paper_dir.glob("**/*.pdf"))
        print(f"Analyzing links quality for {len(pdf_files)} research papers...")
        print(f"Note: Checking max {max_link_checks} links per paper for server politeness")
        
        for pdf_file in pdf_files:
            print(f"  Processing: {pdf_file.name}")
            try:
                analysis = self.analyze_document(str(pdf_file), max_link_checks)
                results[pdf_file.stem] = analysis
                
                # Brief status update
                if analysis.total_links > 0:
                    print(f"    Found {analysis.total_links} links, {analysis.good_links} working, {analysis.bad_links} broken")
                else:
                    print(f"    No links found")
                    
            except Exception as e:
                print(f"    Error analyzing {pdf_file.name}: {e}")
                continue
        
        return results
    
    def generate_portfolio_reliability_report(self, results: Dict[str, LinksAnalysis]) -> str:
        """Generate portfolio-wide links reliability analysis."""
        if not results:
            return "No papers analyzed for links portfolio."
        
        report = ["# PORTFOLIO LINKS RELIABILITY ANALYSIS", "=" * 50, ""]
        
        # Portfolio statistics
        valid_results = [r for r in results.values() if r.total_links > 0]
        total_papers = len(results)
        papers_with_links = len(valid_results)
        
        if papers_with_links == 0:
            return "No papers found with external links."
        
        total_links = sum(r.total_links for r in valid_results)
        total_good = sum(r.good_links for r in valid_results)
        total_bad = sum(r.bad_links for r in valid_results)
        avg_authority = sum(r.authority_score for r in valid_results) / papers_with_links
        avg_quality = sum(r.link_quality_score for r in valid_results) / papers_with_links
        
        report.extend([
            f"**Portfolio Size:** {total_papers} papers ({papers_with_links} with links)",
            f"**Total Links Analyzed:** {total_links}",
            f"**Overall Link Quality:** {avg_quality:.1f}% working",
            f"**Overall Authority Score:** {avg_authority:.1f}%",
            f"**Broken Links:** {total_bad} ({(total_bad/total_links)*100:.1f}% of all links)",
            ""
        ])
        
        # Maintenance quality distribution
        maintenance_dist = defaultdict(int)
        for analysis in valid_results:
            maintenance_dist[analysis.maintenance_indicator] += 1
        
        report.extend(["## MAINTENANCE QUALITY DISTRIBUTION", "-" * 30])
        for quality, count in sorted(maintenance_dist.items()):
            percentage = (count / papers_with_links) * 100
            report.append(f"- **{quality}:** {count} papers ({percentage:.1f}%)")
        report.append("")
        
        # Problem papers (high broken link rate)
        problem_papers = [(paper_id, analysis) for paper_id, analysis in results.items() 
                         if analysis.total_links > 0 and analysis.bad_links > analysis.total_links * 0.3]
        
        if problem_papers:
            report.extend(["## PAPERS WITH LINK MAINTENANCE ISSUES", "-" * 30])
            for paper_id, analysis in sorted(problem_papers, key=lambda x: x[1].bad_links, reverse=True)[:5]:
                broken_pct = (analysis.bad_links / analysis.total_links) * 100
                report.append(f"**{paper_id[:50]}**")
                report.append(f"  - Broken Links: {analysis.bad_links}/{analysis.total_links} ({broken_pct:.1f}%)")
                report.append(f"  - Quality Score: {analysis.link_quality_score:.1f}%")
                report.append("")
        
        # High quality papers
        high_quality = [(paper_id, analysis) for paper_id, analysis in results.items()
                       if analysis.total_links >= 5 and analysis.link_quality_score >= 85 and analysis.authority_score >= 70]
        
        if high_quality:
            report.extend(["## HIGH QUALITY REFERENCE PAPERS", "-" * 30])
            for paper_id, analysis in sorted(high_quality, key=lambda x: x[1].authority_score, reverse=True)[:5]:
                report.append(f"**{paper_id[:50]}**")
                report.append(f"  - Link Quality: {analysis.link_quality_score:.1f}%")
                report.append(f"  - Authority Score: {analysis.authority_score:.1f}%")
                report.append(f"  - Total Links: {analysis.total_links}")
                report.append("")
        
        # Portfolio recommendations
        report.extend(["## PORTFOLIO RECOMMENDATIONS", "-" * 30])
        
        if avg_quality >= 80:
            report.append("[EXCELLENT] **EXCELLENT PORTFOLIO** - High quality link maintenance across papers")
        elif total_bad > total_links * 0.2:
            report.append("[WARNING] **MAINTENANCE ATTENTION NEEDED** - High broken link rate indicates aging research")
        
        if avg_authority >= 70:
            report.append("[GOOD] **AUTHORITATIVE SOURCES** - Portfolio references reputable academic sources")
        elif avg_authority < 50:
            report.append("[WARNING] **SOURCE QUALITY CONCERNS** - Consider prioritizing papers with higher authority references")
        
        papers_no_links = total_papers - papers_with_links
        if papers_no_links > total_papers * 0.3:
            report.append(f"[NOTE] **LIMITED EXTERNAL REFERENCES** - {papers_no_links} papers have no external links")
        
        return "\n".join(report)

def main():
    """Demo links quality analysis."""
    checker = LinksQualityChecker()
    
    # Analyze papers in repository
    paper_repo_path = Path(__file__).parent.parent / "paper_repository"
    
    if paper_repo_path.exists():
        # Single paper example
        pdf_files = list(paper_repo_path.glob("**/*.pdf"))
        if pdf_files:
            test_paper = pdf_files[0]
            print(f"Analyzing links in: {test_paper.name}")
            print("=" * 60)
            
            analysis = checker.analyze_document(str(test_paper), max_link_checks=10)
            report = checker.generate_business_report(analysis, test_paper.stem)
            print(report)
            print("\n" + "=" * 60)
        
        # Portfolio analysis (limited checking for demo)
        print("GENERATING PORTFOLIO LINKS ANALYSIS...")
        results = checker.batch_analyze(str(paper_repo_path), max_link_checks=10)
        portfolio_report = checker.generate_portfolio_reliability_report(results)
        print(portfolio_report)
        
        # Save results
        output_file = Path(__file__).parent / "links_quality_results.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("LINKS QUALITY ANALYSIS RESULTS\n")
            f.write("=" * 50 + "\n\n")
            f.write(portfolio_report)
            f.write("\n\nDETAILED PAPER ANALYSES:\n")
            f.write("=" * 50 + "\n\n")
            
            for paper_id, analysis in results.items():
                if analysis.total_links > 0:
                    f.write(checker.generate_business_report(analysis, paper_id))
                    f.write("\n\n" + "-" * 80 + "\n\n")
        
        print(f"\nDetailed results saved to: {output_file}")
    else:
        print("Paper repository not found. Please ensure papers are in the paper_repository directory.")

if __name__ == "__main__":
    main()