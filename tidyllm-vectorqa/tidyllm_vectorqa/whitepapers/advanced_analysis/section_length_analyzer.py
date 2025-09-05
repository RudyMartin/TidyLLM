#!/usr/bin/env python3
"""
Section Length Analysis - Business Dashboard for Research Investment
===================================================================

Shows where authors spent their time and effort, giving business stakeholders
a quick understanding of research focus and thoroughness.

Business Questions Answered:
- Where did researchers invest most effort?
- Is this a thorough study or quick analysis?
- How much attention was paid to methodology vs results?
- Is the validation comprehensive enough?
"""

import re
import sys
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

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
class SectionMetrics:
    """Metrics for a document section."""
    name: str
    word_count: int
    page_estimate: float
    effort_percentage: float
    content_type: str  # methodology, results, background, etc.
    complexity_score: float

@dataclass
class DocumentAnalysis:
    """Complete section analysis for business dashboard."""
    sections: List[SectionMetrics]
    total_words: int
    research_focus: str  # Primary focus area
    effort_distribution: Dict[str, float]  # By content type
    thoroughness_score: float  # 0-100
    validation_investment: float  # Percentage on validation
    methodology_depth: float  # Percentage on methodology

class SectionLengthAnalyzer:
    """Analyze where authors invested their effort - Business Dashboard Style."""
    
    def __init__(self):
        if EXTRACTION_AVAILABLE:
            self.extractor = TextExtractor()
        else:
            self.extractor = None
            
        # Business-friendly section classifications
        self.section_types = {
            'background': ['introduction', 'background', 'related work', 'literature', 'survey', 'overview'],
            'methodology': ['method', 'approach', 'technique', 'algorithm', 'implementation', 'design', 'model'],
            'results': ['result', 'finding', 'outcome', 'performance', 'evaluation', 'analysis'],
            'validation': ['validation', 'experiment', 'test', 'proof', 'verification', 'case study'],
            'discussion': ['discussion', 'interpretation', 'implication', 'limitation', 'analysis'],
            'conclusion': ['conclusion', 'summary', 'future work', 'recommendation', 'next steps']
        }
        
        # Technical complexity indicators
        self.complexity_indicators = [
            'algorithm', 'equation', 'formula', 'mathematical', 'theorem', 'proof',
            'implementation', 'architecture', 'framework', 'system', 'model',
            'optimization', 'convergence', 'complexity', 'computational'
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
    
    def identify_sections(self, text: str) -> List[Tuple[str, str, int]]:
        """Identify document sections with business-friendly names."""
        sections = []
        
        # Split text into potential sections
        # Look for section headers (numbered or titled)
        section_patterns = [
            r'\n\s*(\d+\.?\d*)\s+([A-Z][^.\n]{10,80})\n',  # "1. Introduction"
            r'\n\s*(Abstract|Introduction|Methodology?|Methods?|Results?|Discussion|Conclusion|References)\b[^\n]{0,50}\n',
            r'\n\s*([A-Z][A-Z\s]{5,30})\n(?=[A-Z])',  # ALL CAPS headers
            r'\n\s*([A-Z][a-z\s]{10,50})\n(?=[A-Z])'   # Title Case headers
        ]
        
        found_sections = []
        for pattern in section_patterns:
            matches = list(re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE))
            for match in matches:
                start_pos = match.start()
                if len(match.groups()) >= 2:
                    section_name = match.group(2).strip()
                else:
                    section_name = match.group(1).strip()
                
                found_sections.append((start_pos, section_name))
        
        # Sort by position and create sections
        found_sections.sort(key=lambda x: x[0])
        
        if not found_sections:
            # Fallback: treat whole document as single section
            return [("Full Document", text, len(text.split()))]
        
        # Create sections with content
        for i, (pos, name) in enumerate(found_sections):
            # Get content from this section to next (or end)
            start_text_pos = pos
            if i < len(found_sections) - 1:
                end_text_pos = found_sections[i + 1][0]
                content = text[start_text_pos:end_text_pos]
            else:
                content = text[start_text_pos:]
            
            word_count = len(content.split())
            if word_count > 10:  # Only include substantial sections
                sections.append((name, content, word_count))
        
        return sections
    
    def classify_section_type(self, section_name: str, content: str) -> str:
        """Classify section type for business understanding."""
        name_lower = section_name.lower()
        content_lower = content.lower()
        
        # Direct name matching first
        for section_type, keywords in self.section_types.items():
            if any(keyword in name_lower for keyword in keywords):
                return section_type
        
        # Content-based classification
        type_scores = defaultdict(int)
        content_words = content_lower.split()[:200]  # First 200 words
        
        for section_type, keywords in self.section_types.items():
            for keyword in keywords:
                type_scores[section_type] += content_lower.count(keyword)
        
        if type_scores:
            return max(type_scores, key=type_scores.get)
        
        return 'other'
    
    def calculate_complexity_score(self, content: str) -> float:
        """Calculate technical complexity (0-1 scale)."""
        content_lower = content.lower()
        complexity_count = sum(content_lower.count(indicator) for indicator in self.complexity_indicators)
        word_count = len(content.split())
        
        if word_count == 0:
            return 0.0
        
        # Normalize: complexity terms per 100 words
        complexity_ratio = (complexity_count / word_count) * 100
        return min(complexity_ratio / 10.0, 1.0)  # Cap at 1.0
    
    def analyze_document(self, pdf_path: str) -> DocumentAnalysis:
        """Complete section analysis for business dashboard."""
        # Extract text
        text = self.extract_document_text(pdf_path)
        if not text:
            return DocumentAnalysis([], 0, "Unable to analyze", {}, 0.0, 0.0, 0.0)
        
        # Identify sections
        raw_sections = self.identify_sections(text)
        total_words = sum(word_count for _, _, word_count in raw_sections)
        
        if total_words == 0:
            return DocumentAnalysis([], 0, "No content found", {}, 0.0, 0.0, 0.0)
        
        # Create section metrics
        sections = []
        effort_distribution = defaultdict(float)
        
        for section_name, content, word_count in raw_sections:
            content_type = self.classify_section_type(section_name, content)
            effort_percentage = (word_count / total_words) * 100
            complexity_score = self.calculate_complexity_score(content)
            page_estimate = word_count / 250  # ~250 words per page
            
            sections.append(SectionMetrics(
                name=section_name,
                word_count=word_count,
                page_estimate=page_estimate,
                effort_percentage=effort_percentage,
                content_type=content_type,
                complexity_score=complexity_score
            ))
            
            effort_distribution[content_type] += effort_percentage
        
        # Business insights
        research_focus = max(effort_distribution, key=effort_distribution.get) if effort_distribution else "unclear"
        
        # Thoroughness score (based on section variety and depth)
        section_types_count = len(set(s.content_type for s in sections))
        thoroughness_score = min((section_types_count * 15) + (total_words / 100), 100)
        
        # Investment percentages
        validation_investment = effort_distribution.get('validation', 0) + effort_distribution.get('results', 0) * 0.5
        methodology_depth = effort_distribution.get('methodology', 0)
        
        return DocumentAnalysis(
            sections=sections,
            total_words=total_words,
            research_focus=research_focus.title(),
            effort_distribution=dict(effort_distribution),
            thoroughness_score=thoroughness_score,
            validation_investment=validation_investment,
            methodology_depth=methodology_depth
        )
    
    def generate_business_report(self, analysis: DocumentAnalysis, paper_title: str = "") -> str:
        """Generate business-friendly analysis report."""
        if not analysis.sections:
            return f"**{paper_title}**\nUnable to analyze document structure."
        
        report = []
        
        # Header
        report.append(f"# RESEARCH INVESTMENT ANALYSIS")
        if paper_title:
            report.append(f"**Paper:** {paper_title}")
        report.append(f"**Document Size:** {analysis.total_words:,} words (~{analysis.total_words/250:.1f} pages)")
        report.append("")
        
        # Key Insights (Business Summary)
        report.append("## KEY BUSINESS INSIGHTS")
        report.append(f"- **Primary Research Focus:** {analysis.research_focus}")
        report.append(f"- **Study Thoroughness:** {analysis.thoroughness_score:.0f}% (0-100 scale)")
        report.append(f"- **Methodology Investment:** {analysis.methodology_depth:.1f}% of effort")
        report.append(f"- **Validation Investment:** {analysis.validation_investment:.1f}% of effort")
        report.append("")
        
        # Effort Distribution (Business Dashboard)
        report.append("## EFFORT ALLOCATION")
        effort_items = sorted(analysis.effort_distribution.items(), key=lambda x: x[1], reverse=True)
        for effort_type, percentage in effort_items:
            if percentage > 5:  # Only show significant allocations
                report.append(f"- **{effort_type.title()}:** {percentage:.1f}%")
        report.append("")
        
        # Detailed Section Breakdown
        report.append("## DETAILED SECTION ANALYSIS")
        report.append("| Section | Word Count | Effort % | Pages | Complexity | Type |")
        report.append("|---------|------------|----------|-------|------------|------|")
        
        for section in sorted(analysis.sections, key=lambda x: x.word_count, reverse=True):
            complexity_bar = "*" * int(section.complexity_score * 5)  # 0-5 stars
            report.append(f"| {section.name[:25]} | {section.word_count:,} | {section.effort_percentage:.1f}% | {section.page_estimate:.1f} | {complexity_bar} | {section.content_type} |")
        report.append("")
        
        # Business Recommendations
        report.append("## BUSINESS ASSESSMENT")
        
        if analysis.methodology_depth < 15:
            report.append("[WARNING] **LIMITED METHODOLOGY** - Low investment in explaining methods may indicate rushed research")
        elif analysis.methodology_depth > 40:
            report.append("[GOOD] **THOROUGH METHODOLOGY** - Substantial investment in explaining approach shows rigor")
        
        if analysis.validation_investment < 20:
            report.append("[WARNING] **WEAK VALIDATION** - Limited evidence/testing may indicate preliminary findings")
        elif analysis.validation_investment > 30:
            report.append("[GOOD] **STRONG VALIDATION** - Substantial testing and results analysis shows reliability")
        
        if analysis.thoroughness_score < 40:
            report.append("[NOTE] **BRIEF STUDY** - Limited scope, suitable for quick insights but not comprehensive analysis")
        elif analysis.thoroughness_score > 70:
            report.append("[EXCELLENT] **COMPREHENSIVE STUDY** - Extensive analysis with multiple perspectives and thorough coverage")
        
        return "\n".join(report)
    
    def batch_analyze(self, paper_directory: str) -> Dict[str, DocumentAnalysis]:
        """Analyze all papers for business dashboard."""
        results = {}
        paper_dir = Path(paper_directory)
        
        pdf_files = list(paper_dir.glob("**/*.pdf"))
        print(f"Analyzing section investment for {len(pdf_files)} research papers...")
        
        for pdf_file in pdf_files:
            print(f"  Processing: {pdf_file.name}")
            try:
                analysis = self.analyze_document(str(pdf_file))
                results[pdf_file.stem] = analysis
            except Exception as e:
                print(f"    Error analyzing {pdf_file.name}: {e}")
                continue
        
        return results
    
    def generate_portfolio_report(self, results: Dict[str, DocumentAnalysis]) -> str:
        """Generate business portfolio analysis across all papers."""
        if not results:
            return "No papers analyzed for portfolio report."
        
        report = ["# RESEARCH PORTFOLIO ANALYSIS", "=" * 50, ""]
        
        # Portfolio statistics
        total_papers = len(results)
        avg_thoroughness = sum(r.thoroughness_score for r in results.values()) / total_papers
        avg_words = sum(r.total_words for r in results.values()) / total_papers
        
        report.extend([
            f"**Portfolio Size:** {total_papers} research papers",
            f"**Average Study Size:** {avg_words:,.0f} words ({avg_words/250:.1f} pages)",
            f"**Average Thoroughness:** {avg_thoroughness:.1f}% (0-100 scale)",
            ""
        ])
        
        # Focus area distribution
        focus_areas = defaultdict(int)
        for analysis in results.values():
            focus_areas[analysis.research_focus] += 1
        
        report.extend(["## RESEARCH FOCUS DISTRIBUTION", "-" * 30])
        for focus, count in sorted(focus_areas.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_papers) * 100
            report.append(f"- **{focus}:** {count} papers ({percentage:.1f}%)")
        report.append("")
        
        # Investment patterns
        report.extend(["## INVESTMENT PATTERNS", "-" * 30])
        
        strong_methodology = sum(1 for r in results.values() if r.methodology_depth > 30)
        strong_validation = sum(1 for r in results.values() if r.validation_investment > 25)
        comprehensive_studies = sum(1 for r in results.values() if r.thoroughness_score > 70)
        
        report.extend([
            f"- **High Methodology Investment:** {strong_methodology}/{total_papers} papers ({(strong_methodology/total_papers)*100:.1f}%)",
            f"- **Strong Validation:** {strong_validation}/{total_papers} papers ({(strong_validation/total_papers)*100:.1f}%)",
            f"- **Comprehensive Studies:** {comprehensive_studies}/{total_papers} papers ({(comprehensive_studies/total_papers)*100:.1f}%)",
            ""
        ])
        
        # Top performers
        top_thorough = sorted(results.items(), key=lambda x: x[1].thoroughness_score, reverse=True)[:3]
        report.extend(["## TOP THOROUGH STUDIES", "-" * 30])
        for paper_id, analysis in top_thorough:
            report.append(f"**{paper_id[:50]}**")
            report.append(f"  - Thoroughness: {analysis.thoroughness_score:.1f}%")
            report.append(f"  - Focus: {analysis.research_focus}")
            report.append(f"  - Size: {analysis.total_words:,} words")
            report.append("")
        
        return "\n".join(report)

def main():
    """Demo section length analysis."""
    analyzer = SectionLengthAnalyzer()
    
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
        print("GENERATING PORTFOLIO ANALYSIS...")
        results = analyzer.batch_analyze(str(paper_repo_path))
        portfolio_report = analyzer.generate_portfolio_report(results)
        print(portfolio_report)
        
        # Save results
        output_file = Path(__file__).parent / "section_analysis_results.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("SECTION LENGTH ANALYSIS RESULTS\n")
            f.write("=" * 50 + "\n\n")
            f.write(portfolio_report)
            f.write("\n\nDETAILED PAPER ANALYSES:\n")
            f.write("=" * 50 + "\n\n")
            
            for paper_id, analysis in results.items():
                f.write(analyzer.generate_business_report(analysis, paper_id))
                f.write("\n\n" + "-" * 80 + "\n\n")
        
        print(f"\nDetailed results saved to: {output_file}")
    else:
        print("Paper repository not found. Please ensure papers are in the paper_repository directory.")

if __name__ == "__main__":
    main()