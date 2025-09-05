#!/usr/bin/env python3
"""
Research Framework for tidyllm-compliance
========================================

Y="+", C="-" Mathematical Model for Research Paper Analysis

Implementation of the mathematical decomposition framework for Context Engineering:
Y = "+" (Relevant Content) vs C = "-" (Context Collapse)

Where C = R + S + N:
- R: Relevant systematic items
- S: Superfluous marginally systematic content  
- N: True noise and errors

This module provides scoring and analysis functions for research papers
referenced in regulatory and compliance documents.

Integrated with tidyllm-compliance for:
- Analyzing research papers mentioned in risk management documents
- Extracting key insights from academic sources supporting compliance decisions
- Reference drilling and citation analysis for regulatory research
- Quality assessment of research backing regulatory guidance

Part of tidyllm-compliance: Professional regulatory compliance platform
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import re
import math
from datetime import datetime

# Import compliance validation if available
try:
    from ..sop_conflict_analysis.yrsn_analyzer import YRSNNoiseAnalyzer
except ImportError:
    YRSNNoiseAnalyzer = None

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
class ResearchPaper:
    """Represents a research paper with analysis metadata"""
    title: str
    authors: List[str]
    abstract: str = ""
    content: str = ""
    year: Optional[int] = None
    doi: Optional[str] = None
    references: List[str] = None
    decomposition_score: Optional[DecompositionScore] = None
    regulatory_relevance_score: float = 0.0
    source: str = "unknown"
    
    def __post_init__(self):
        if self.references is None:
            self.references = []

class ResearchFramework:
    """
    Research paper analysis framework for regulatory compliance.
    
    Features:
    - Y="+", C="-" mathematical decomposition analysis
    - Integration with YRSN noise analysis for quality assessment
    - Regulatory relevance scoring for compliance applications
    - Reference and citation extraction for deep-dive analysis
    - Content quality assessment for regulatory research backing
    """
    
    def __init__(self, enable_yrsn_validation: bool = True):
        # Initialize YRSN analyzer if available
        self.yrsn_analyzer = YRSNNoiseAnalyzer() if YRSNNoiseAnalyzer and enable_yrsn_validation else None
        
        # Regulatory relevance keywords for scoring
        self.regulatory_keywords = {
            'high_relevance': [
                'regulatory', 'compliance', 'risk management', 'model validation',
                'supervisory guidance', 'federal reserve', 'basel', 'sarbanes oxley',
                'governance', 'audit', 'internal controls', 'stress testing'
            ],
            'medium_relevance': [
                'financial', 'banking', 'credit risk', 'market risk', 'operational risk',
                'quantitative', 'statistical', 'modeling', 'validation', 'monitoring'
            ],
            'domain_specific': [
                'machine learning', 'artificial intelligence', 'data science',
                'algorithm', 'prediction', 'classification', 'regression'
            ]
        }
        
        print(f"[RESEARCH_FRAMEWORK] Initialized research analysis framework")
        if self.yrsn_analyzer:
            print(f"[INTEGRATION] YRSN validation enabled for quality assessment")
    
    def analyze_research_paper(self, paper: ResearchPaper, 
                             regulatory_context: Optional[str] = None) -> Dict[str, any]:
        """
        Comprehensive analysis of research paper for regulatory compliance.
        
        Args:
            paper: Research paper to analyze
            regulatory_context: Specific regulatory context for relevance scoring
            
        Returns:
            Complete analysis results with decomposition and compliance metrics
        """
        print(f"\n[RESEARCH_ANALYSIS] Analyzing: {paper.title[:60]}...")
        
        analysis_result = {
            'paper_metadata': {
                'title': paper.title,
                'authors': paper.authors,
                'year': paper.year,
                'doi': paper.doi,
                'source': paper.source
            },
            'analysis_timestamp': datetime.now().isoformat(),
            'regulatory_context': regulatory_context
        }
        
        # Step 1: Y="+", C="-" Decomposition Analysis
        if paper.content:
            decomposition = self._analyze_content_decomposition(paper.content)
            analysis_result['decomposition_analysis'] = {
                'relevant_score': decomposition.relevant,
                'superfluous_score': decomposition.superfluous,
                'noise_score': decomposition.noise,
                'y_score': decomposition.y_score,
                'total_normalized': decomposition.total
            }
            paper.decomposition_score = decomposition
        
        # Step 2: Regulatory Relevance Scoring
        regulatory_score = self._calculate_regulatory_relevance(paper, regulatory_context)
        analysis_result['regulatory_relevance'] = {
            'overall_score': regulatory_score,
            'relevance_category': self._categorize_relevance(regulatory_score),
            'keyword_analysis': self._analyze_regulatory_keywords(paper.content + " " + paper.abstract)
        }
        paper.regulatory_relevance_score = regulatory_score
        
        # Step 3: YRSN Quality Validation (if available)
        if self.yrsn_analyzer and paper.abstract:
            try:
                yrsn_result = self.yrsn_analyzer.validate_sop_response(
                    paper.abstract,
                    f"Research paper quality assessment: {paper.title}"
                )
                analysis_result['yrsn_validation'] = {
                    'noise_percentage': yrsn_result['yrsn_metrics']['noise_percentage'],
                    'compliance_status': yrsn_result['compliance_status'],
                    'actionable_indicators': yrsn_result['yrsn_metrics']['actionable_indicators'],
                    'noise_indicators': yrsn_result['yrsn_metrics']['noise_indicators']
                }
            except Exception as e:
                print(f"[WARNING] YRSN validation failed for {paper.title}: {e}")
        
        # Step 4: Reference and Citation Analysis
        if paper.content:
            reference_analysis = self._analyze_references_and_citations(paper.content)
            analysis_result['reference_analysis'] = reference_analysis
        
        # Step 5: Section Analysis (where authors invested effort)
        if paper.content:
            section_analysis = self._analyze_paper_sections(paper.content)
            analysis_result['section_analysis'] = section_analysis
        
        # Step 6: Overall Assessment
        analysis_result['overall_assessment'] = self._generate_overall_assessment(analysis_result)
        
        print(f"[COMPLETE] Analysis complete for {paper.title}")
        print(f"Y-Score: {analysis_result.get('decomposition_analysis', {}).get('y_score', 'N/A')}")
        print(f"Regulatory Relevance: {analysis_result['regulatory_relevance']['relevance_category']}")
        
        return analysis_result
    
    def _analyze_content_decomposition(self, content: str) -> DecompositionScore:
        """
        Analyze content using Y="+", C="-" decomposition framework.
        
        Args:
            content: Full text content of the research paper
            
        Returns:
            DecompositionScore with R+S+N analysis
        """
        # Relevant content indicators
        relevant_patterns = [
            r'\b(result|conclusion|finding|evidence|data|analysis|study|research)\b',
            r'\b(significant|p\s*[<>=]\s*0\.\d+|correlation|regression|model)\b',
            r'\b(methodology|approach|framework|algorithm|technique)\b'
        ]
        
        # Superfluous content indicators
        superfluous_patterns = [
            r'\b(introduction|background|literature review|related work)\b',
            r'\b(furthermore|moreover|additionally|in addition|however)\b',
            r'\b(well-known|established|traditional|conventional)\b'
        ]
        
        # Noise indicators
        noise_patterns = [
            r'\b(unclear|ambiguous|possibly|might|could be|seems)\b',
            r'\b(etc\.|and so on|among others|various|multiple)\b',
            r'\b(TODO|FIXME|placeholder|draft|incomplete)\b'
        ]
        
        content_lower = content.lower()
        total_words = len(content.split())
        
        if total_words == 0:
            return DecompositionScore(0.33, 0.33, 0.34)
        
        # Count pattern matches
        relevant_matches = sum(len(re.findall(pattern, content_lower)) for pattern in relevant_patterns)
        superfluous_matches = sum(len(re.findall(pattern, content_lower)) for pattern in superfluous_patterns)
        noise_matches = sum(len(re.findall(pattern, content_lower)) for pattern in noise_patterns)
        
        # Calculate scores with normalization
        total_matches = relevant_matches + superfluous_matches + noise_matches
        
        if total_matches == 0:
            # Default distribution for content without clear indicators
            return DecompositionScore(0.6, 0.25, 0.15)
        
        relevant_score = relevant_matches / total_matches
        superfluous_score = superfluous_matches / total_matches
        noise_score = noise_matches / total_matches
        
        # Ensure minimum relevant content for research papers
        if relevant_score < 0.3:
            relevant_score = 0.3
            remaining = 0.7
            superfluous_score = (superfluous_score / (superfluous_score + noise_score)) * remaining
            noise_score = remaining - superfluous_score
        
        return DecompositionScore(relevant_score, superfluous_score, noise_score)
    
    def _calculate_regulatory_relevance(self, paper: ResearchPaper, 
                                      regulatory_context: Optional[str] = None) -> float:
        """
        Calculate regulatory relevance score for the research paper.
        
        Args:
            paper: Research paper to score
            regulatory_context: Specific regulatory context for weighting
            
        Returns:
            Regulatory relevance score (0.0 to 1.0)
        """
        content_to_analyze = f"{paper.title} {paper.abstract} {paper.content[:1000]}"
        content_lower = content_to_analyze.lower()
        
        # Score based on keyword categories
        high_matches = sum(1 for keyword in self.regulatory_keywords['high_relevance'] 
                          if keyword in content_lower)
        medium_matches = sum(1 for keyword in self.regulatory_keywords['medium_relevance'] 
                           if keyword in content_lower)
        domain_matches = sum(1 for keyword in self.regulatory_keywords['domain_specific'] 
                           if keyword in content_lower)
        
        # Weighted scoring
        base_score = (
            (high_matches * 0.8) + 
            (medium_matches * 0.5) + 
            (domain_matches * 0.3)
        ) / 10  # Normalize to 0-1 range
        
        # Context-specific boosting
        if regulatory_context:
            context_keywords = regulatory_context.lower().split()
            context_boost = sum(0.1 for keyword in context_keywords if keyword in content_lower)
            base_score += min(context_boost, 0.3)  # Cap context boost at 0.3
        
        # Author credibility boost (if from known regulatory/academic institutions)
        institution_boost = 0.0
        for author in paper.authors:
            author_lower = author.lower()
            if any(inst in author_lower for inst in ['federal reserve', 'treasury', 'occ', 'fdic', 'university']):
                institution_boost += 0.05
        
        base_score += min(institution_boost, 0.2)  # Cap institution boost at 0.2
        
        return min(1.0, base_score)  # Ensure score doesn't exceed 1.0
    
    def _categorize_relevance(self, score: float) -> str:
        """Categorize regulatory relevance score."""
        if score >= 0.8:
            return "HIGHLY_RELEVANT"
        elif score >= 0.6:
            return "MODERATELY_RELEVANT"
        elif score >= 0.4:
            return "SOMEWHAT_RELEVANT"
        else:
            return "LIMITED_RELEVANCE"
    
    def _analyze_regulatory_keywords(self, text: str) -> Dict[str, int]:
        """Analyze presence of regulatory keywords in text."""
        text_lower = text.lower()
        
        return {
            'high_relevance_count': sum(1 for keyword in self.regulatory_keywords['high_relevance'] 
                                      if keyword in text_lower),
            'medium_relevance_count': sum(1 for keyword in self.regulatory_keywords['medium_relevance'] 
                                        if keyword in text_lower),
            'domain_specific_count': sum(1 for keyword in self.regulatory_keywords['domain_specific'] 
                                       if keyword in text_lower)
        }
    
    def _analyze_references_and_citations(self, content: str) -> Dict[str, any]:
        """Analyze references and citations in the paper."""
        # Extract reference patterns
        reference_patterns = [
            r'\[\d+\]',  # [1], [2], etc.
            r'\(\d{4}\)',  # (2020), (2021), etc.
            r'et al\.?,?\s*\(?\d{4}\)?',  # et al. 2020
            r'doi:\s*10\.\d+/[^\s]+',  # DOI references
        ]
        
        references_found = []
        for pattern in reference_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            references_found.extend(matches)
        
        # Extract bibliography if present
        bibliography_match = re.search(r'(references|bibliography)\s*:?\s*\n(.+?)\n\n', 
                                      content, re.IGNORECASE | re.DOTALL)
        bibliography = bibliography_match.group(2) if bibliography_match else ""
        
        return {
            'total_references_found': len(references_found),
            'reference_samples': references_found[:10],  # First 10 references
            'has_bibliography_section': bool(bibliography),
            'bibliography_length': len(bibliography.split('\n')) if bibliography else 0
        }
    
    def _analyze_paper_sections(self, content: str) -> Dict[str, any]:
        """Analyze paper sections to identify where authors invested effort."""
        # Common academic paper sections
        section_patterns = {
            'abstract': r'abstract\s*:?\s*\n(.+?)\n\n',
            'introduction': r'introduction\s*:?\s*\n(.+?)\n\n',
            'methodology': r'(methodology|methods?)\s*:?\s*\n(.+?)\n\n',
            'results': r'results\s*:?\s*\n(.+?)\n\n',
            'discussion': r'discussion\s*:?\s*\n(.+?)\n\n',
            'conclusion': r'conclusions?\s*:?\s*\n(.+?)\n\n'
        }
        
        section_analysis = {}
        total_content_length = len(content)
        
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                section_content = match.group(1) if len(match.groups()) == 1 else match.group(2)
                section_length = len(section_content)
                
                section_analysis[section_name] = {
                    'length': section_length,
                    'percentage_of_total': (section_length / total_content_length) * 100 if total_content_length > 0 else 0,
                    'word_count': len(section_content.split())
                }
            else:
                section_analysis[section_name] = {
                    'length': 0,
                    'percentage_of_total': 0,
                    'word_count': 0
                }
        
        # Identify where most effort was invested
        max_section = max(section_analysis.items(), key=lambda x: x[1]['length'])
        
        return {
            'section_breakdown': section_analysis,
            'primary_focus_section': max_section[0],
            'primary_focus_percentage': max_section[1]['percentage_of_total'],
            'total_sections_found': sum(1 for section in section_analysis.values() if section['length'] > 0)
        }
    
    def _generate_overall_assessment(self, analysis_result: Dict[str, any]) -> Dict[str, any]:
        """Generate overall assessment of the research paper."""
        assessment = {
            'quality_rating': 'unknown',
            'regulatory_utility': 'unknown',
            'research_value_score': 0.0,
            'recommendations': []
        }
        
        # Quality rating based on Y-score and YRSN analysis
        decomp = analysis_result.get('decomposition_analysis', {})
        y_score = decomp.get('y_score', 0.5)
        
        yrsn = analysis_result.get('yrsn_validation', {})
        yrsn_noise = yrsn.get('noise_percentage', 50) if yrsn else 50
        
        # Combined quality assessment
        quality_score = (y_score * 0.6) + ((100 - yrsn_noise) / 100 * 0.4)
        
        if quality_score >= 0.8:
            assessment['quality_rating'] = 'EXCELLENT'
        elif quality_score >= 0.65:
            assessment['quality_rating'] = 'GOOD'
        elif quality_score >= 0.5:
            assessment['quality_rating'] = 'ACCEPTABLE'
        else:
            assessment['quality_rating'] = 'NEEDS_REVIEW'
        
        # Regulatory utility assessment
        reg_relevance = analysis_result.get('regulatory_relevance', {})
        relevance_category = reg_relevance.get('relevance_category', 'LIMITED_RELEVANCE')
        
        if relevance_category == 'HIGHLY_RELEVANT':
            assessment['regulatory_utility'] = 'HIGH_VALUE'
        elif relevance_category == 'MODERATELY_RELEVANT':
            assessment['regulatory_utility'] = 'MODERATE_VALUE'
        else:
            assessment['regulatory_utility'] = 'SUPPLEMENTARY'
        
        # Research value score (0-100)
        research_value = (
            (quality_score * 40) + 
            (reg_relevance.get('overall_score', 0) * 35) +
            (min(analysis_result.get('reference_analysis', {}).get('total_references_found', 0) / 50, 1.0) * 25)
        )
        assessment['research_value_score'] = min(100, research_value * 100)
        
        # Generate recommendations
        recommendations = []
        
        if assessment['quality_rating'] in ['NEEDS_REVIEW', 'ACCEPTABLE']:
            recommendations.append("Consider supplementing with higher-quality research sources")
        
        if assessment['regulatory_utility'] == 'SUPPLEMENTARY':
            recommendations.append("Use as supporting evidence only - not primary regulatory guidance")
        
        if analysis_result.get('reference_analysis', {}).get('total_references_found', 0) < 10:
            recommendations.append("Limited references - verify claims with additional sources")
        
        if not recommendations:
            recommendations.append("Research appears suitable for regulatory compliance analysis")
        
        assessment['recommendations'] = recommendations
        
        return assessment

# Utility functions for compatibility
def get_demo_papers() -> List[ResearchPaper]:
    """Get demo research papers for testing."""
    return [
        ResearchPaper(
            title="Model Risk Management in Financial Services",
            authors=["Federal Reserve Board"],
            abstract="Comprehensive guidance on model risk management practices for financial institutions...",
            year=2011,
            source="regulatory_guidance"
        ),
        ResearchPaper(
            title="Machine Learning Applications in Credit Risk Assessment",
            authors=["Academic Research Team"],
            abstract="This paper examines the application of machine learning techniques to credit risk modeling...",
            year=2020,
            source="academic_research"
        )
    ]

# Example usage and demonstration
def demo_research_framework():
    """Demonstrate research framework functionality."""
    framework = ResearchFramework(enable_yrsn_validation=True)
    
    # Get demo papers
    papers = get_demo_papers()
    
    print("\nResearch Framework Demo")
    print("=" * 40)
    
    for i, paper in enumerate(papers, 1):
        print(f"\n[PAPER {i}] Analyzing: {paper.title}")
        
        # Add some content for analysis
        paper.content = f"""This research presents a comprehensive analysis of {paper.title.lower()}.
        
        The methodology involves statistical analysis and empirical validation.
        Results show significant correlations with p < 0.05.
        
        The study concludes with actionable recommendations for implementation.
        References include 25 academic sources and regulatory guidance documents.
        """
        
        try:
            analysis = framework.analyze_research_paper(paper, "model_validation")
            
            print(f"Quality Rating: {analysis['overall_assessment']['quality_rating']}")
            print(f"Regulatory Utility: {analysis['overall_assessment']['regulatory_utility']}")
            print(f"Research Value Score: {analysis['overall_assessment']['research_value_score']:.1f}/100")
            
            if analysis.get('decomposition_analysis'):
                decomp = analysis['decomposition_analysis']
                print(f"Y-Score: {decomp['y_score']:.3f} (Relevant: {decomp['relevant_score']:.3f})")
            
        except Exception as e:
            print(f"Analysis failed: {e}")

if __name__ == "__main__":
    demo_research_framework()