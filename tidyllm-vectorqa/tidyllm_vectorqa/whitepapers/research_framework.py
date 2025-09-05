"""
Research Framework - Y="+", C="-" Mathematical Model
===================================================

Implementation of the mathematical decomposition framework for Context Engineering:
Y = "+" (Relevant Content) vs C = "-" (Context Collapse)

Where C = R + S + N:
- R: Relevant systematic items
- S: Superfluous marginally systematic content  
- N: True noise and errors

This module provides scoring and analysis functions for research papers.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import re
import math

@dataclass
class DecompositionScore:
    """Represents R+S+N decomposition of content"""
    relevant: float      # R: Core systematic content (0.0-1.0)
    superfluous: float   # S: Marginally systematic (0.0-1.0)  
    noise: float         # N: True noise/errors (0.0-1.0)
    
    @property
    def total(self) -> float:
        """Total should sum to 1.0"""
        return self.relevant + self.superfluous + self.noise
    
    @property
    def y_score(self) -> float:
        """Y score - overall relevance metric"""
        return self.relevant + (0.5 * self.superfluous)
    
    def normalize(self) -> 'DecompositionScore':
        """Ensure components sum to 1.0"""
        total = self.total
        if total == 0:
            return DecompositionScore(0.33, 0.33, 0.34)
        
        return DecompositionScore(
            relevant=self.relevant / total,
            superfluous=self.superfluous / total,
            noise=self.noise / total
        )

@dataclass 
class PaperAnalysis:
    """Complete analysis of a research paper"""
    title: str
    authors: List[str]
    arxiv_id: Optional[str]
    abstract: str
    decomposition: DecompositionScore
    context_relevance: str  # "High", "Medium", "Low"
    key_concepts: List[str]
    methodology_score: float
    
    @property
    def overall_score(self) -> float:
        """Combined scoring metric"""
        return (self.decomposition.y_score * 0.7) + (self.methodology_score * 0.3)

class ResearchFramework:
    """Core framework for Y="+", C="-" analysis"""
    
    # Mathematical decomposition keywords
    RELEVANT_KEYWORDS = [
        'mathematical decomposition', 'signal separation', 'noise reduction',
        'systematic analysis', 'orthogonal projection', 'residual analysis',
        'variance decomposition', 'systematic risk', 'signal processing',
        'mathematical model', 'systematic error', 'noise filtering'
    ]
    
    SUPERFLUOUS_KEYWORDS = [
        'background information', 'literature review', 'historical context',
        'future work', 'acknowledgments', 'related work', 'introduction',
        'general discussion', 'broader implications', 'survey'
    ]
    
    NOISE_KEYWORDS = [
        'formatting error', 'typo', 'reference error', 'unclear text',
        'incomplete sentence', 'broken equation', 'missing data',
        'unrelated content', 'advertising', 'boilerplate'
    ]
    
    def __init__(self):
        self.analysis_cache = {}
    
    def analyze_paper_content(self, title: str, abstract: str, content: str = "") -> DecompositionScore:
        """
        Analyze paper content and return R+S+N decomposition
        
        Args:
            title: Paper title
            abstract: Paper abstract
            content: Full paper content (optional)
            
        Returns:
            DecompositionScore with R, S, N components
        """
        # Combine available text
        full_text = f"{title} {abstract} {content}".lower()
        
        # Count keyword matches
        relevant_matches = sum(1 for kw in self.RELEVANT_KEYWORDS if kw in full_text)
        superfluous_matches = sum(1 for kw in self.SUPERFLUOUS_KEYWORDS if kw in full_text)
        noise_matches = sum(1 for kw in self.NOISE_KEYWORDS if kw in full_text)
        
        # Base scoring using keyword analysis
        base_r = 0.4 + (relevant_matches * 0.08)  # Start at 40%, add up to 32% more
        base_s = 0.2 + (superfluous_matches * 0.05)  # Start at 20%, add up to 20% more  
        base_n = 0.1 + (noise_matches * 0.03)  # Start at 10%, add up to 12% more
        
        # Apply content-specific bonuses/penalties
        math_terms = ['mathematical', 'equation', 'formula', 'theorem', 'proof', 'decomposition']
        math_count = sum(1 for term in math_terms if term in full_text)
        
        if math_count >= 3:
            base_r += 0.15  # High math content boosts relevance
            base_s -= 0.05  # Reduces superfluous content
        elif math_count >= 1:
            base_r += 0.08
            
        # Context Engineering terms boost
        context_terms = ['context', 'engineering', 'systematic', 'signal', 'noise']
        context_count = sum(1 for term in context_terms if term in full_text)
        
        if context_count >= 2:
            base_r += 0.12
            base_n -= 0.03
            
        # Quality indicators
        quality_terms = ['robust', 'validated', 'comprehensive', 'rigorous']
        quality_count = sum(1 for term in quality_terms if term in full_text)
        
        if quality_count >= 1:
            base_r += 0.05
            base_n -= 0.02
            
        # Complexity penalty (too much jargon can add noise)
        complex_terms = ['heterogeneous', 'multidimensional', 'nonlinear', 'stochastic']
        complexity = sum(1 for term in complex_terms if term in full_text)
        
        if complexity >= 2:
            base_s += 0.08
            base_n += 0.05
        
        # Ensure realistic bounds
        r_score = max(0.25, min(0.85, base_r))
        s_score = max(0.10, min(0.45, base_s))
        n_score = max(0.05, min(0.35, base_n))
        
        # Create and normalize the score
        score = DecompositionScore(r_score, s_score, n_score).normalize()
        return score
    
    def calculate_y_score(self, decomposition: DecompositionScore) -> float:
        """
        Calculate Y score using the Y="+" framework
        
        Y represents relevant content that contributes positively to understanding
        """
        return decomposition.y_score
    
    def calculate_context_collapse_risk(self, decomposition: DecompositionScore) -> Tuple[float, str]:
        """
        Calculate Context Collapse risk using C="-" framework
        
        Returns:
            (risk_score, risk_level) where risk_level is "Low", "Medium", "High"
        """
        # Context collapse increases with superfluous content and noise
        c_score = decomposition.superfluous + (1.5 * decomposition.noise)
        
        if c_score < 0.3:
            return c_score, "Low"
        elif c_score < 0.6:
            return c_score, "Medium" 
        else:
            return c_score, "High"
    
    def analyze_mathematical_content(self, text: str) -> Dict[str, float]:
        """
        Analyze mathematical content quality in research papers
        
        Returns scores for different mathematical aspects
        """
        text_lower = text.lower()
        
        scores = {
            'equation_density': len(re.findall(r'equation|formula|\$.*?\$', text)) / max(len(text.split()), 1),
            'proof_rigor': len(re.findall(r'proof|theorem|lemma|corollary', text_lower)) / max(len(text.split()), 1),
            'model_complexity': len(re.findall(r'model|framework|system|approach', text_lower)) / max(len(text.split()), 1),
            'validation_strength': len(re.findall(r'validation|verification|test|experiment', text_lower)) / max(len(text.split()), 1)
        }
        
        return {k: min(1.0, v * 100) for k, v in scores.items()}
    
    def generate_paper_recommendations(self, analysis: PaperAnalysis) -> List[str]:
        """Generate actionable recommendations based on paper analysis"""
        recommendations = []
        
        decomp = analysis.decomposition
        
        if decomp.relevant > 0.8:
            recommendations.append("✅ High relevance - Excellent for Context Engineering research")
        elif decomp.relevant > 0.6:
            recommendations.append("🔵 Good relevance - Useful supporting material")
        else:
            recommendations.append("⚠️ Low relevance - Consider secondary source")
        
        if decomp.noise > 0.3:
            recommendations.append("🔧 High noise content - May need preprocessing")
        
        if decomp.superfluous > 0.4:
            recommendations.append("✂️ High superfluous content - Focus on core sections")
        
        y_score = decomp.y_score
        if y_score > 0.9:
            recommendations.append("🎯 Core paper - Essential for framework validation")
        elif y_score > 0.7:
            recommendations.append("📖 Supporting paper - Good for literature review")
        
        context_risk, risk_level = self.calculate_context_collapse_risk(decomp)
        if risk_level == "High":
            recommendations.append("⚡ High context collapse risk - Use with caution")
        elif risk_level == "Medium":
            recommendations.append("⚠️ Medium context collapse risk - Validate key claims")
        
        return recommendations

# Demo data for the Streamlit app
def get_demo_papers() -> List[PaperAnalysis]:
    """Return demo papers with realistic varied decomposition scores"""
    framework = ResearchFramework()
    
    papers = [
        {
            "title": "Mathematical Decomposition of Signal and Noise in Context Engineering Systems",
            "authors": ["Dr. Sarah Chen", "Prof. Michael Torres", "Dr. Elena Kozlov"],
            "arxiv_id": "2024.12001",
            "abstract": "This paper presents a comprehensive mathematical framework for decomposing signal and noise in context engineering systems. We develop systematic methods for identifying relevant systematic content, superfluous marginally systematic information, and true noise components. Our robust mathematical approach validates the Y=R+S+N decomposition theory with rigorous proof and comprehensive validation across multiple domains."
        },
        {
            "title": "Survey of Background Literature in Information Theory Applications", 
            "authors": ["John Smith", "Mary Johnson"],
            "arxiv_id": "2024.11045", 
            "abstract": "This comprehensive survey provides extensive background information and literature review of information theory applications. The paper includes general discussion of historical context, related work across multiple disciplines, and broader implications for future research directions. While containing some relevant mathematical concepts, much content serves as introduction and acknowledgments."
        },
        {
            "title": "Multidimensional Stochastic Analysis with Heterogeneous Nonlinear Components",
            "authors": ["Complex Research Collective"],
            "arxiv_id": "2024.10234",
            "abstract": "Complex multidimensional analysis of heterogeneous stochastic systems with nonlinear dynamics. The methodology involves unclear mathematical formulations and ambiguous theoretical frameworks. Some equations contain formatting errors and incomplete derivations, making validation difficult."
        },
        {
            "title": "Context Collapse Prevention Through Systematic Decomposition", 
            "authors": ["Dr. Alex Rivera", "Prof. Lisa Wang"],
            "arxiv_id": "2024.09876",
            "abstract": "Systematic approach to preventing context collapse in AI systems through mathematical decomposition. The paper presents validated methods for context engineering with robust signal separation techniques. Mathematical proofs demonstrate comprehensive noise reduction capabilities in systematic frameworks."
        }
    ]
    
    analyses = []
    for paper in papers:
        decomp = framework.analyze_paper_content(paper["title"], paper["abstract"])
        
        analysis = PaperAnalysis(
            title=paper["title"],
            authors=paper["authors"],
            arxiv_id=paper["arxiv_id"],
            abstract=paper["abstract"],
            decomposition=decomp,
            context_relevance="High" if decomp.y_score > 0.8 else "Medium" if decomp.y_score > 0.6 else "Low",
            key_concepts=["mathematical decomposition", "signal separation", "noise reduction"],
            methodology_score=0.85 if decomp.y_score > 0.8 else 0.65 if decomp.y_score > 0.6 else 0.45
        )
        analyses.append(analysis)
    
    return analyses

# Context Collapse Analysis (from dbreuning.com)
def analyze_context_collapse_types(text: str) -> Dict[str, float]:
    """
    Analyze the four types of context collapse from dbreuning.com:
    1. Context Poisoning - misinformation absorption
    2. Context Distraction - information overload  
    3. Context Confusion - non-relevant data
    4. Context Clash - conflicting sources
    """
    text_lower = text.lower()
    
    # Context Poisoning indicators
    poisoning_terms = ['misinformation', 'bias', 'incorrect', 'false', 'misleading', 'error']
    poisoning_score = sum(1 for term in poisoning_terms if term in text_lower) / len(poisoning_terms)
    
    # Context Distraction indicators  
    distraction_terms = ['overwhelming', 'too much', 'information overload', 'complexity', 'extensive']
    distraction_score = sum(1 for term in distraction_terms if term in text_lower) / len(distraction_terms)
    
    # Context Confusion indicators
    confusion_terms = ['unclear', 'ambiguous', 'confusing', 'non-relevant', 'unrelated', 'mixed']
    confusion_score = sum(1 for term in confusion_terms if term in text_lower) / len(confusion_terms)
    
    # Context Clash indicators
    clash_terms = ['conflicting', 'contradictory', 'inconsistent', 'opposing', 'disagreement', 'conflict']
    clash_score = sum(1 for term in clash_terms if term in text_lower) / len(clash_terms)
    
    return {
        'poisoning': min(1.0, poisoning_score),
        'distraction': min(1.0, distraction_score), 
        'confusion': min(1.0, confusion_score),
        'clash': min(1.0, clash_score)
    }

# Academic Paper Structure Analysis
def extract_table_of_contents(text: str) -> List[Dict[str, str]]:
    """
    Extract table of contents from academic paper text.
    
    Args:
        text: Full paper text
        
    Returns:
        List of dictionaries with section headers and page numbers
    """
    toc_entries = []
    text_lines = text.split('\n')
    
    # Common academic paper section patterns
    section_patterns = [
        r'^(\d+\.?\s+)([A-Z][^.]*?)\.?\s*\.+\s*(\d+)$',  # 1. Introduction ........ 5
        r'^([A-Z][A-Z\s]+)\s*\.+\s*(\d+)$',              # INTRODUCTION ........ 5  
        r'^(\d+\.?\d*\.?\s+)([A-Z][^.]*?)\.?\s*(\d+)$',  # 1.1 Background .... 7
        r'^([A-Z][a-z\s]+)\s+(\d+)$',                    # Introduction 5
        r'^(\d+\.?\s+)([A-Z][a-z\s]+)$',                 # 1. Introduction
    ]
    
    for line in text_lines:
        line = line.strip()
        if not line:
            continue
            
        for pattern in section_patterns:
            match = re.match(pattern, line)
            if match:
                if len(match.groups()) == 3:
                    section_num, title, page = match.groups()
                    toc_entries.append({
                        'section_number': section_num.strip(),
                        'title': title.strip(),
                        'page': page.strip()
                    })
                elif len(match.groups()) == 2:
                    if match.groups()[1].isdigit():  # Title and page
                        title, page = match.groups()
                        toc_entries.append({
                            'section_number': '',
                            'title': title.strip(),
                            'page': page.strip()
                        })
                    else:  # Section number and title
                        section_num, title = match.groups()
                        toc_entries.append({
                            'section_number': section_num.strip(),
                            'title': title.strip(),
                            'page': ''
                        })
                break
    
    # If no formal TOC found, extract section headers from content
    if not toc_entries:
        section_headers = [
            r'^#+\s+(.+)$',                    # Markdown headers
            r'^\d+\.?\s+([A-Z][^.]*?)\.?$',    # 1. Introduction
            r'^([A-Z][A-Z\s]{3,})$',           # ABSTRACT, INTRODUCTION
            r'^([A-Z][a-z\s]{3,}):?\s*$',      # Introduction:
        ]
        
        for line_num, line in enumerate(text_lines):
            line = line.strip()
            for pattern in section_headers:
                match = re.match(pattern, line)
                if match:
                    title = match.group(1).strip()
                    # Estimate page number based on line position
                    estimated_page = max(1, line_num // 50)  # ~50 lines per page
                    
                    toc_entries.append({
                        'section_number': '',
                        'title': title,
                        'page': str(estimated_page)
                    })
                    break
    
    return toc_entries

def extract_bibliography(text: str) -> List[Dict[str, str]]:
    """
    Extract bibliography/references from academic paper text.
    
    Args:
        text: Full paper text
        
    Returns:
        List of dictionaries with citation information
    """
    references = []
    text_lines = text.split('\n')
    
    # Find references section
    ref_section_start = -1
    ref_patterns = [
        r'^references?\s*$',
        r'^bibliography\s*$', 
        r'^works\s+cited\s*$',
        r'^literature\s+cited\s*$'
    ]
    
    for i, line in enumerate(text_lines):
        line_lower = line.strip().lower()
        for pattern in ref_patterns:
            if re.match(pattern, line_lower):
                ref_section_start = i
                break
        if ref_section_start != -1:
            break
    
    if ref_section_start == -1:
        return []  # No references section found
    
    # Extract references from references section
    citation_patterns = [
        r'^\[(\d+)\]\s+(.+)$',                           # [1] Author, Title, Journal
        r'^(\d+\.)\s+(.+)$',                             # 1. Author, Title, Journal
        r'^([A-Z][a-z]+,?\s+[A-Z]\.?\s*(?:[A-Z]\.?\s*)*)\s*\((\d{4})\)\.?\s*(.+)$',  # Author (2024). Title
        r'^(.+?)\s*\((\d{4})\)\.?\s*(.+)$',              # General author (year) pattern
    ]
    
    ref_number = 1
    for line in text_lines[ref_section_start+1:]:
        line = line.strip()
        if not line:
            continue
            
        # Stop if we hit another major section
        if re.match(r'^[A-Z][A-Z\s]+$', line) and len(line) > 3:
            break
            
        citation_found = False
        for pattern in citation_patterns:
            match = re.match(pattern, line)
            if match:
                groups = match.groups()
                
                if len(groups) == 2:  # Number and full citation
                    ref_num, full_citation = groups
                    references.append({
                        'number': ref_num.rstrip('.'),
                        'full_citation': full_citation.strip(),
                        'author': '',
                        'year': '',
                        'title': '',
                        'journal': ''
                    })
                elif len(groups) == 3:  # Author, year, title/journal
                    author, year, title_journal = groups
                    references.append({
                        'number': str(ref_number),
                        'full_citation': line,
                        'author': author.strip(),
                        'year': year.strip(),
                        'title': title_journal.strip()[:100],
                        'journal': ''
                    })
                elif len(groups) == 4:  # Full structured citation
                    ref_num, author, year, title_journal = groups
                    references.append({
                        'number': ref_num.rstrip('.'),
                        'full_citation': line,
                        'author': author.strip(),
                        'year': year.strip(),
                        'title': title_journal.strip()[:100],
                        'journal': ''
                    })
                
                citation_found = True
                ref_number += 1
                break
        
        # If no pattern matched but line looks like a reference
        if not citation_found and len(line) > 20 and ('.' in line or ',' in line):
            references.append({
                'number': str(ref_number),
                'full_citation': line,
                'author': '',
                'year': '',
                'title': line[:100],
                'journal': ''
            })
            ref_number += 1
    
    return references