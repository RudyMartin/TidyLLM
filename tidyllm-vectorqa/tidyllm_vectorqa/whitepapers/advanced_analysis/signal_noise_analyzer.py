#!/usr/bin/env python3
"""
Signal-to-Noise Ratio Calculator - Content Quality vs Fluff Analyzer
====================================================================

Measures the density of meaningful content vs filler text, giving business stakeholders
a clear efficiency metric for research quality and author focus.

Business Questions Answered:
- How much substance vs fluff is in this research?
- Are authors being concise and focused?
- Is this paper worth the time investment to read?
- Which papers have the highest information density?
"""

import re
import sys
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict, Counter

# Use existing document processing from tidyllm-documents
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "tidyllm-documents"))

try:
    from tidyllm_documents.extraction.text import TextExtractor
    EXTRACTION_AVAILABLE = True
except ImportError:
    print("INFO: Using fallback text extraction (tidyllm-documents not available)")
    EXTRACTION_AVAILABLE = False
    import PyPDF2

@dataclass
class SignalMetrics:
    """Signal (valuable content) metrics."""
    technical_terms: int
    quantitative_data: int
    novel_concepts: int
    citations: int
    equations_formulas: int
    algorithmic_content: int
    total_signal_score: float

@dataclass
class NoiseMetrics:
    """Noise (filler content) metrics."""
    filler_phrases: int
    redundant_expressions: int
    weak_qualifiers: int
    generic_statements: int
    excessive_hedging: int
    bloated_descriptions: int
    total_noise_score: float

@dataclass
class ContentAnalysis:
    """Complete signal-to-noise analysis."""
    signal_metrics: SignalMetrics
    noise_metrics: NoiseMetrics
    signal_to_noise_ratio: float
    content_efficiency: float  # 0-100%
    information_density: float  # Signal per 100 words
    quality_assessment: str
    total_words: int
    substantive_sentences: int
    filler_sentences: int

class SignalNoiseAnalyzer:
    """Analyze content quality vs fluff - Business Efficiency Style."""
    
    def __init__(self):
        if EXTRACTION_AVAILABLE:
            self.extractor = TextExtractor()
        else:
            self.extractor = None
        
        # SIGNAL indicators (valuable content)
        self.technical_terms = {
            'algorithm', 'methodology', 'implementation', 'optimization', 'convergence',
            'computational', 'mathematical', 'statistical', 'empirical', 'experimental',
            'framework', 'architecture', 'model', 'system', 'analysis', 'evaluation',
            'performance', 'accuracy', 'precision', 'recall', 'validation', 'verification',
            'theorem', 'proof', 'equation', 'formula', 'calculation', 'measurement',
            'dataset', 'training', 'testing', 'benchmark', 'baseline', 'comparison',
            'correlation', 'regression', 'classification', 'clustering', 'prediction',
            'neural', 'network', 'machine learning', 'artificial intelligence',
            'deep learning', 'transformer', 'attention', 'embedding', 'vector'
        }
        
        self.quantitative_indicators = [
            r'\d+%', r'\d+\.\d+', r'±\d+', r'p\s*[<>=]\s*\d+\.?\d*',
            r'n\s*=\s*\d+', r'\d+\s*±\s*\d+', r'r\s*=\s*\d+\.?\d*',
            r'accuracy\s*[:=]\s*\d+\.?\d*', r'precision\s*[:=]\s*\d+\.?\d*',
            r'\d+\s*ms\b', r'\d+\s*seconds?\b', r'\d+\s*GB\b', r'\d+\s*MB\b'
        ]
        
        self.novelty_indicators = [
            'novel', 'new', 'innovative', 'proposed', 'introduce', 'present',
            'develop', 'design', 'create', 'establish', 'demonstrate',
            'first time', 'breakthrough', 'advancement', 'improvement'
        ]
        
        # NOISE indicators (filler content)
        self.filler_phrases = {
            'it is important to note', 'it should be noted that', 'it is worth mentioning',
            'as mentioned earlier', 'as previously discussed', 'as we will see',
            'in this paper we', 'in this work we', 'in this study we',
            'the purpose of this paper', 'the goal of this work', 'the aim of this study',
            'it is well known that', 'it is clear that', 'it is obvious that',
            'needless to say', 'obviously', 'clearly', 'undoubtedly',
            'furthermore', 'moreover', 'in addition', 'additionally', 'also',
            'on the other hand', 'however', 'nevertheless', 'nonetheless',
            'in conclusion', 'to conclude', 'in summary', 'to summarize'
        }
        
        self.weak_qualifiers = {
            'somewhat', 'rather', 'quite', 'fairly', 'relatively', 'reasonably',
            'possibly', 'probably', 'likely', 'potentially', 'presumably',
            'apparently', 'seemingly', 'supposedly', 'allegedly',
            'to some extent', 'in some cases', 'under certain conditions',
            'might be', 'could be', 'may be', 'appears to be', 'seems to be'
        }
        
        self.redundant_expressions = [
            r'\b(\w+)\s+and\s+\1\b',  # "fast and fast"
            r'\b(very|extremely|highly|quite|rather)\s+(very|extremely|highly|quite|rather)\b',
            r'\bthe\s+fact\s+that\b', r'\bit\s+is\s+the\s+case\s+that\b',
            r'\bin\s+order\s+to\b', r'\bfor\s+the\s+purpose\s+of\b',
            r'\bdue\s+to\s+the\s+fact\s+that\b', r'\bowing\s+to\s+the\s+fact\s+that\b'
        ]
        
        self.generic_statements = [
            r'\bmany\s+researchers\s+have\b', r'\bnumerous\s+studies\s+have\b',
            r'\bit\s+has\s+been\s+shown\s+that\b', r'\bstudies\s+have\s+shown\b',
            r'\bresearch\s+has\s+demonstrated\b', r'\bit\s+is\s+widely\s+accepted\b',
            r'\bthere\s+is\s+growing\s+interest\b', r'\bthere\s+has\s+been\s+increasing\b'
        ]
    
    def extract_document_text(self, pdf_path: str) -> str:
        """Extract text using tidyllm-documents or fallback."""
        if EXTRACTION_AVAILABLE and self.extractor:
            text, metadata = self.extractor.extract_text(pdf_path, max_pages=20)
            if text:
                return text
        
        # Fallback extraction
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                max_pages = min(len(pdf_reader.pages), 20)
                
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
    
    def calculate_signal_metrics(self, text: str) -> SignalMetrics:
        """Calculate signal (valuable content) metrics."""
        text_lower = text.lower()
        
        # Technical terms count
        technical_count = sum(text_lower.count(term) for term in self.technical_terms)
        
        # Quantitative data count
        quantitative_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) 
                                for pattern in self.quantitative_indicators)
        
        # Novel concepts count
        novel_count = sum(text_lower.count(term) for term in self.novelty_indicators)
        
        # Citations count (rough estimate)
        citations = len(re.findall(r'\[\d+\]|\(\w+\s+et\s+al\.?,?\s+\d{4}\)', text))
        
        # Equations and formulas
        equations = len(re.findall(r'equation\s*\(\d+\)|formula\s*\(\d+\)|\\[a-z]+\{|\$.*?\$', 
                                  text, re.IGNORECASE))
        
        # Algorithmic content
        algorithmic = len(re.findall(r'\b(algorithm|procedure|method|step)\s+\d+', 
                                    text, re.IGNORECASE))
        
        # Calculate total signal score
        signal_score = (
            technical_count * 1.0 +
            quantitative_count * 2.0 +  # Higher weight for data
            novel_count * 1.5 +
            citations * 1.0 +
            equations * 2.5 +  # Highest weight for mathematical content
            algorithmic * 2.0
        )
        
        return SignalMetrics(
            technical_terms=technical_count,
            quantitative_data=quantitative_count,
            novel_concepts=novel_count,
            citations=citations,
            equations_formulas=equations,
            algorithmic_content=algorithmic,
            total_signal_score=signal_score
        )
    
    def calculate_noise_metrics(self, text: str) -> NoiseMetrics:
        """Calculate noise (filler content) metrics."""
        text_lower = text.lower()
        
        # Filler phrases count
        filler_count = sum(text_lower.count(phrase) for phrase in self.filler_phrases)
        
        # Weak qualifiers count
        qualifier_count = sum(text_lower.count(qualifier) for qualifier in self.weak_qualifiers)
        
        # Redundant expressions
        redundant_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) 
                             for pattern in self.redundant_expressions)
        
        # Generic statements
        generic_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) 
                           for pattern in self.generic_statements)
        
        # Excessive hedging (multiple qualifiers in sentence)
        sentences = re.split(r'[.!?]+', text)
        hedging_count = 0
        for sentence in sentences:
            qualifier_in_sentence = sum(qualifier in sentence.lower() 
                                      for qualifier in self.weak_qualifiers)
            if qualifier_in_sentence > 2:  # More than 2 qualifiers per sentence
                hedging_count += 1
        
        # Bloated descriptions (very long sentences with low information)
        bloated_count = 0
        for sentence in sentences:
            if len(sentence.split()) > 30:  # Long sentences
                # Check if they contain technical content
                technical_in_sentence = sum(term in sentence.lower() 
                                          for term in self.technical_terms)
                if technical_in_sentence < 2:  # Low technical content
                    bloated_count += 1
        
        # Calculate total noise score
        noise_score = (
            filler_count * 1.0 +
            qualifier_count * 0.5 +
            redundant_count * 1.5 +
            generic_count * 1.0 +
            hedging_count * 2.0 +
            bloated_count * 1.5
        )
        
        return NoiseMetrics(
            filler_phrases=filler_count,
            redundant_expressions=redundant_count,
            weak_qualifiers=qualifier_count,
            generic_statements=generic_count,
            excessive_hedging=hedging_count,
            bloated_descriptions=bloated_count,
            total_noise_score=noise_score
        )
    
    def analyze_sentence_quality(self, text: str) -> Tuple[int, int]:
        """Count substantive vs filler sentences."""
        sentences = re.split(r'[.!?]+', text)
        substantive = 0
        filler = 0
        
        for sentence in sentences:
            if len(sentence.split()) < 5:  # Too short
                continue
            
            sentence_lower = sentence.lower()
            
            # Count signals in sentence
            technical_signals = sum(term in sentence_lower for term in self.technical_terms)
            quantitative_signals = sum(bool(re.search(pattern, sentence, re.IGNORECASE)) 
                                     for pattern in self.quantitative_indicators)
            
            # Count noise in sentence
            filler_noise = sum(phrase in sentence_lower for phrase in self.filler_phrases)
            qualifier_noise = sum(qualifier in sentence_lower for qualifier in self.weak_qualifiers)
            
            total_signals = technical_signals + quantitative_signals
            total_noise = filler_noise + qualifier_noise
            
            if total_signals > total_noise:
                substantive += 1
            elif total_noise > total_signals:
                filler += 1
            # Equal or no clear dominance = neutral (not counted)
        
        return substantive, filler
    
    def analyze_document(self, pdf_path: str) -> ContentAnalysis:
        """Complete signal-to-noise analysis."""
        # Extract text
        text = self.extract_document_text(pdf_path)
        if not text:
            return ContentAnalysis(
                SignalMetrics(0, 0, 0, 0, 0, 0, 0.0),
                NoiseMetrics(0, 0, 0, 0, 0, 0, 0.0),
                0.0, 0.0, 0.0, "Unable to analyze", 0, 0, 0
            )
        
        total_words = len(text.split())
        if total_words == 0:
            return ContentAnalysis(
                SignalMetrics(0, 0, 0, 0, 0, 0, 0.0),
                NoiseMetrics(0, 0, 0, 0, 0, 0, 0.0),
                0.0, 0.0, 0.0, "No content found", 0, 0, 0
            )
        
        # Calculate metrics
        signal_metrics = self.calculate_signal_metrics(text)
        noise_metrics = self.calculate_noise_metrics(text)
        substantive_sentences, filler_sentences = self.analyze_sentence_quality(text)
        
        # Calculate ratios
        if noise_metrics.total_noise_score > 0:
            signal_to_noise = signal_metrics.total_signal_score / noise_metrics.total_noise_score
        else:
            signal_to_noise = signal_metrics.total_signal_score  # All signal, no noise
        
        # Content efficiency (signal / total content indicators)
        total_indicators = signal_metrics.total_signal_score + noise_metrics.total_noise_score
        if total_indicators > 0:
            content_efficiency = (signal_metrics.total_signal_score / total_indicators) * 100
        else:
            content_efficiency = 50.0  # Neutral
        
        # Information density (signal per 100 words)
        information_density = (signal_metrics.total_signal_score / total_words) * 100
        
        # Quality assessment
        if signal_to_noise >= 3.0:
            quality = "High Signal - Information Dense"
        elif signal_to_noise >= 2.0:
            quality = "Good Signal - Well Focused"
        elif signal_to_noise >= 1.0:
            quality = "Balanced - Moderate Focus"
        elif signal_to_noise >= 0.5:
            quality = "High Noise - Excessive Filler"
        else:
            quality = "Very High Noise - Bloated Content"
        
        return ContentAnalysis(
            signal_metrics=signal_metrics,
            noise_metrics=noise_metrics,
            signal_to_noise_ratio=signal_to_noise,
            content_efficiency=content_efficiency,
            information_density=information_density,
            quality_assessment=quality,
            total_words=total_words,
            substantive_sentences=substantive_sentences,
            filler_sentences=filler_sentences
        )
    
    def generate_business_report(self, analysis: ContentAnalysis, paper_title: str = "") -> str:
        """Generate business-friendly efficiency report."""
        if analysis.total_words == 0:
            return f"**{paper_title}**\nUnable to analyze content quality."
        
        report = []
        
        # Header
        report.append("# CONTENT EFFICIENCY ANALYSIS")
        if paper_title:
            report.append(f"**Paper:** {paper_title}")
        report.append(f"**Document Size:** {analysis.total_words:,} words")
        report.append("")
        
        # Key Business Metrics
        report.append("## KEY EFFICIENCY METRICS")
        report.append(f"- **Signal-to-Noise Ratio:** {analysis.signal_to_noise_ratio:.2f}")
        report.append(f"- **Content Efficiency:** {analysis.content_efficiency:.1f}%")
        report.append(f"- **Information Density:** {analysis.information_density:.2f} signals per 100 words")
        report.append(f"- **Quality Assessment:** {analysis.quality_assessment}")
        report.append("")
        
        # Reading Efficiency
        reading_time = analysis.total_words / 250  # ~250 words per minute
        valuable_time = (analysis.signal_metrics.total_signal_score / 
                        (analysis.signal_metrics.total_signal_score + analysis.noise_metrics.total_noise_score) 
                        * reading_time if (analysis.signal_metrics.total_signal_score + analysis.noise_metrics.total_noise_score) > 0 else reading_time / 2)
        
        report.append("## READING EFFICIENCY")
        report.append(f"- **Estimated Reading Time:** {reading_time:.1f} minutes")
        report.append(f"- **Valuable Content Time:** {valuable_time:.1f} minutes ({(valuable_time/reading_time)*100:.0f}%)")
        report.append(f"- **Time Investment ROI:** {'High' if analysis.content_efficiency > 70 else 'Medium' if analysis.content_efficiency > 50 else 'Low'}")
        report.append("")
        
        # Signal Breakdown
        report.append("## VALUABLE CONTENT ANALYSIS")
        s = analysis.signal_metrics
        report.append(f"- **Technical Terms:** {s.technical_terms}")
        report.append(f"- **Quantitative Data:** {s.quantitative_data}")
        report.append(f"- **Novel Concepts:** {s.novel_concepts}")
        report.append(f"- **Citations:** {s.citations}")
        report.append(f"- **Equations/Formulas:** {s.equations_formulas}")
        report.append(f"- **Algorithmic Content:** {s.algorithmic_content}")
        report.append(f"- **Total Signal Score:** {s.total_signal_score:.1f}")
        report.append("")
        
        # Noise Breakdown
        report.append("## FILLER CONTENT ANALYSIS")
        n = analysis.noise_metrics
        report.append(f"- **Filler Phrases:** {n.filler_phrases}")
        report.append(f"- **Weak Qualifiers:** {n.weak_qualifiers}")
        report.append(f"- **Redundant Expressions:** {n.redundant_expressions}")
        report.append(f"- **Generic Statements:** {n.generic_statements}")
        report.append(f"- **Excessive Hedging:** {n.excessive_hedging}")
        report.append(f"- **Bloated Descriptions:** {n.bloated_descriptions}")
        report.append(f"- **Total Noise Score:** {n.total_noise_score:.1f}")
        report.append("")
        
        # Sentence Quality
        total_classified = analysis.substantive_sentences + analysis.filler_sentences
        if total_classified > 0:
            substantive_pct = (analysis.substantive_sentences / total_classified) * 100
            report.append("## SENTENCE QUALITY")
            report.append(f"- **Substantive Sentences:** {analysis.substantive_sentences} ({substantive_pct:.1f}%)")
            report.append(f"- **Filler Sentences:** {analysis.filler_sentences} ({100-substantive_pct:.1f}%)")
            report.append("")
        
        # Business Recommendations
        report.append("## BUSINESS RECOMMENDATION")
        
        if analysis.signal_to_noise_ratio >= 2.5:
            report.append("[EXCELLENT] **HIGH VALUE READ** - Dense with valuable information, excellent time investment")
        elif analysis.signal_to_noise_ratio >= 1.5:
            report.append("[GOOD] **SOLID CONTENT** - Good balance of substance to filler, recommended read")
        elif analysis.signal_to_noise_ratio >= 1.0:
            report.append("[OK] **MODERATE VALUE** - Balanced content, worth reading but expect some filler")
        elif analysis.signal_to_noise_ratio >= 0.5:
            report.append("[WARNING] **HIGH FILLER** - Substantial bloat, consider skimming for key points")
        else:
            report.append("[POOR] **EXCESSIVE FILLER** - Very low information density, may not be worth full read")
        
        if analysis.information_density < 1.0:
            report.append("- **Low Information Density** - Consider executive summary or abstract only")
        elif analysis.information_density > 3.0:
            report.append("- **High Information Density** - Technical paper requiring careful reading")
        
        return "\n".join(report)
    
    def batch_analyze(self, paper_directory: str) -> Dict[str, ContentAnalysis]:
        """Analyze all papers for content efficiency."""
        results = {}
        paper_dir = Path(paper_directory)
        
        pdf_files = list(paper_dir.glob("**/*.pdf"))
        print(f"Analyzing content efficiency for {len(pdf_files)} research papers...")
        
        for pdf_file in pdf_files:
            print(f"  Processing: {pdf_file.name}")
            try:
                analysis = self.analyze_document(str(pdf_file))
                results[pdf_file.stem] = analysis
            except Exception as e:
                print(f"    Error analyzing {pdf_file.name}: {e}")
                continue
        
        return results
    
    def generate_efficiency_portfolio(self, results: Dict[str, ContentAnalysis]) -> str:
        """Generate portfolio efficiency analysis."""
        if not results:
            return "No papers analyzed for efficiency portfolio."
        
        report = ["# CONTENT EFFICIENCY PORTFOLIO", "=" * 50, ""]
        
        # Portfolio statistics
        valid_results = [r for r in results.values() if r.total_words > 0]
        total_papers = len(valid_results)
        
        if total_papers == 0:
            return "No valid analyses found."
        
        avg_snr = sum(r.signal_to_noise_ratio for r in valid_results) / total_papers
        avg_efficiency = sum(r.content_efficiency for r in valid_results) / total_papers
        avg_density = sum(r.information_density for r in valid_results) / total_papers
        
        report.extend([
            f"**Portfolio Size:** {total_papers} research papers",
            f"**Average Signal-to-Noise Ratio:** {avg_snr:.2f}",
            f"**Average Content Efficiency:** {avg_efficiency:.1f}%",
            f"**Average Information Density:** {avg_density:.2f}",
            ""
        ])
        
        # Quality distribution
        quality_dist = defaultdict(int)
        for analysis in valid_results:
            if analysis.signal_to_noise_ratio >= 2.5:
                quality_dist["Excellent"] += 1
            elif analysis.signal_to_noise_ratio >= 1.5:
                quality_dist["Good"] += 1
            elif analysis.signal_to_noise_ratio >= 1.0:
                quality_dist["Moderate"] += 1
            else:
                quality_dist["Poor"] += 1
        
        report.extend(["## QUALITY DISTRIBUTION", "-" * 30])
        for quality, count in quality_dist.items():
            percentage = (count / total_papers) * 100
            report.append(f"- **{quality}:** {count} papers ({percentage:.1f}%)")
        report.append("")
        
        # Efficiency rankings
        top_efficient = sorted(results.items(), key=lambda x: x[1].signal_to_noise_ratio, reverse=True)[:5]
        report.extend(["## TOP EFFICIENT PAPERS", "-" * 30])
        for paper_id, analysis in top_efficient:
            if analysis.total_words > 0:
                report.append(f"**{paper_id[:50]}**")
                report.append(f"  - Signal-to-Noise: {analysis.signal_to_noise_ratio:.2f}")
                report.append(f"  - Efficiency: {analysis.content_efficiency:.1f}%")
                report.append(f"  - Quality: {analysis.quality_assessment}")
                report.append("")
        
        # Time investment analysis
        total_words = sum(r.total_words for r in valid_results)
        total_reading_time = total_words / 250  # minutes
        efficient_papers = sum(1 for r in valid_results if r.content_efficiency > 60)
        
        report.extend(["## TIME INVESTMENT ANALYSIS", "-" * 30])
        report.append(f"- **Total Reading Time:** {total_reading_time:.0f} minutes ({total_reading_time/60:.1f} hours)")
        report.append(f"- **High Efficiency Papers:** {efficient_papers}/{total_papers} ({(efficient_papers/total_papers)*100:.1f}%)")
        report.append(f"- **Portfolio ROI:** {'High' if avg_efficiency > 60 else 'Medium' if avg_efficiency > 45 else 'Low'}")
        
        return "\n".join(report)

def main():
    """Demo signal-to-noise analysis."""
    analyzer = SignalNoiseAnalyzer()
    
    # Analyze papers in repository
    paper_repo_path = Path(__file__).parent.parent / "paper_repository"
    
    if paper_repo_path.exists():
        # Single paper example
        pdf_files = list(paper_repo_path.glob("**/*.pdf"))
        if pdf_files:
            test_paper = pdf_files[0]
            print(f"Analyzing: {test_paper.name}")
            print("=" * 60)
            
            analysis = analyzer.analyze_document(str(test_paper))
            report = analyzer.generate_business_report(analysis, test_paper.stem)
            print(report)
            print("\n" + "=" * 60)
        
        # Portfolio analysis
        print("GENERATING EFFICIENCY PORTFOLIO...")
        results = analyzer.batch_analyze(str(paper_repo_path))
        portfolio_report = analyzer.generate_efficiency_portfolio(results)
        print(portfolio_report)
        
        # Save results
        output_file = Path(__file__).parent / "signal_noise_results.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("SIGNAL-TO-NOISE ANALYSIS RESULTS\n")
            f.write("=" * 50 + "\n\n")
            f.write(portfolio_report)
            f.write("\n\nDETAILED PAPER ANALYSES:\n")
            f.write("=" * 50 + "\n\n")
            
            for paper_id, analysis in results.items():
                if analysis.total_words > 0:
                    f.write(analyzer.generate_business_report(analysis, paper_id))
                    f.write("\n\n" + "-" * 80 + "\n\n")
        
        print(f"\nDetailed results saved to: {output_file}")
    else:
        print("Paper repository not found. Please ensure papers are in the paper_repository directory.")

if __name__ == "__main__":
    main()