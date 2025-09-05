#!/usr/bin/env python3
"""
TOC Analysis Engine - Extract Table of Contents and Map Development Scope
=========================================================================

Analyzes paper structure through TOC patterns to quickly assess:
- Development scope and scale
- Research methodology approach
- Paper organization quality
- Content distribution patterns
"""

import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import PyPDF2

@dataclass
class TOCEntry:
    """Represents a table of contents entry."""
    level: int
    title: str
    page: Optional[int]
    section_number: Optional[str]
    content_type: str  # methodology, results, discussion, etc.

@dataclass
class TOCAnalysis:
    """Complete TOC analysis results."""
    entries: List[TOCEntry]
    scope_indicators: Dict[str, int]
    scale_assessment: str
    methodology_depth: float
    organization_quality: float
    content_distribution: Dict[str, float]

class TOCAnalyzer:
    """Extract and analyze table of contents from research papers."""
    
    def __init__(self):
        # Common TOC patterns in research papers
        self.toc_patterns = [
            r'^(\d+(?:\.\d+)*)\s+(.+?)(?:\s+(\d+))?\s*$',  # 1.2 Title 45
            r'^([A-Z]+)\.\s+(.+?)(?:\s+(\d+))?\s*$',        # A. Title 45
            r'^([ivxlcdm]+)\.\s+(.+?)(?:\s+(\d+))?\s*$',    # ii. Title 45 (roman numerals)
            r'^([•·-])\s+(.+?)(?:\s+(\d+))?\s*$',          # • Title 45
            r'^(.+?)\.{2,}(\d+)$',                         # Title....45
        ]
        
        # Section type classifications
        self.section_types = {
            'methodology': ['method', 'approach', 'procedure', 'technique', 'algorithm', 'implementation', 'design'],
            'results': ['result', 'finding', 'outcome', 'evaluation', 'performance', 'experiment', 'test'],
            'discussion': ['discussion', 'analysis', 'interpretation', 'implication', 'limitation'],
            'background': ['introduction', 'background', 'related work', 'literature review', 'survey'],
            'conclusion': ['conclusion', 'summary', 'future work', 'recommendation'],
            'validation': ['validation', 'verification', 'proof', 'demonstration', 'case study']
        }
        
        # Scope indicators
        self.scope_indicators = {
            'theoretical': ['theory', 'theoretical', 'mathematical', 'formal', 'proof'],
            'empirical': ['empirical', 'experimental', 'study', 'analysis', 'evaluation'],
            'implementation': ['implementation', 'system', 'framework', 'tool', 'platform'],
            'survey': ['survey', 'review', 'comparison', 'taxonomy', 'classification'],
            'novel': ['novel', 'new', 'proposed', 'innovative', 'original']
        }
    
    def extract_pdf_text_structured(self, pdf_path: str) -> Tuple[str, List[str]]:
        """Extract PDF text with page separation for TOC detection."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract first few pages where TOC usually appears
                pages = []
                full_text = ""
                max_pages = min(len(pdf_reader.pages), 5)  # Check first 5 pages
                
                for page_num in range(max_pages):
                    try:
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        # Clean and normalize - handle Unicode
                        page_text = page_text.replace('\n', ' ').replace('\t', ' ')
                        page_text = page_text.encode('ascii', 'ignore').decode('ascii')
                        page_text = ' '.join(page_text.split())
                        pages.append(page_text)
                        full_text += page_text + " "
                    except:
                        continue
                
                return full_text, pages
        except Exception as e:
            print(f"Error extracting PDF {pdf_path}: {e}")
            return "", []
    
    def detect_toc_section(self, text: str, pages: List[str]) -> Optional[str]:
        """Detect the table of contents section in the text."""
        toc_keywords = ['table of contents', 'contents', 'outline']
        
        # Look for explicit TOC markers
        for page_text in pages:
            lower_text = page_text.lower()
            for keyword in toc_keywords:
                if keyword in lower_text:
                    # Extract text after the TOC marker
                    start_idx = lower_text.find(keyword)
                    # Get substantial portion after TOC marker
                    toc_text = page_text[start_idx:start_idx+2000]
                    return toc_text
        
        # Enhanced fallback: Look for paper structure patterns
        for page_text in pages:
            # Look for common research paper section patterns
            section_patterns = [
                r'\b(introduction|abstract|methodology|results|conclusion|references)\b',
                r'\b\d+\.\s+(introduction|method|results|discussion|conclusion)',
                r'\b(I\.|II\.|III\.|IV\.|V\.)\s+[A-Z]',  # Roman numerals
                r'\b\d+(?:\.\d+)*\s+[A-Z][a-z]+',  # Numbered sections
            ]
            
            matches = 0
            for pattern in section_patterns:
                pattern_matches = re.findall(pattern, page_text, re.IGNORECASE)
                matches += len(pattern_matches)
            
            # If we find several section-like patterns, likely contains TOC or is structured
            if matches >= 4:
                return page_text[:2000]  # Return substantial portion
        
        # Final fallback: analyze first page for structure
        if pages:
            first_page = pages[0]
            # Look for any numbered structure
            if re.search(r'\b\d+\.\s+\w+', first_page):
                return first_page
        
        return None
    
    def parse_toc_entries(self, toc_text: str) -> List[TOCEntry]:
        """Parse TOC text into structured entries."""
        entries = []
        
        # Enhanced parsing - look for section headers throughout the text
        # Split on various delimiters and look for section patterns
        text_segments = re.split(r'[.\n\r]', toc_text)
        
        section_patterns = [
            # Standard numbered sections
            r'^(\d+(?:\.\d+)*)\s+([A-Z][^.]*?)(?:\s+(\d+))?$',
            # Text-based sections (Introduction, Methodology, etc.)
            r'^(Introduction|Abstract|Methodology|Methods?|Results?|Discussion|Conclusion|References|Related Work|Background|Experiments?|Evaluation|Analysis|Future Work)\b(.*)$',
            # Numbered with text
            r'^(\d+)\.\s+([A-Z][^.]*?)(?:\s+(\d+))?$',
        ]
        
        for segment in text_segments:
            segment = segment.strip()
            if len(segment) < 5:  # Skip very short segments
                continue
            
            # Try patterns
            for pattern in section_patterns:
                match = re.match(pattern, segment, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    
                    if len(groups) >= 1:
                        if groups[0].isdigit() or '.' in str(groups[0]):
                            # Numbered section
                            section_number = groups[0]
                            title = groups[1].strip() if len(groups) > 1 and groups[1] else segment
                        else:
                            # Named section
                            section_number = None
                            title = groups[0].strip()
                        
                        page = int(groups[2]) if len(groups) > 2 and groups[2] and str(groups[2]).isdigit() else None
                        
                        # Determine level and type
                        level = self._determine_section_level(section_number) if section_number else 1
                        content_type = self._classify_section_type(title)
                        
                        # Avoid duplicates
                        if not any(e.title.lower() == title.lower() for e in entries):
                            entries.append(TOCEntry(
                                level=level,
                                title=title,
                                page=page,
                                section_number=section_number,
                                content_type=content_type
                            ))
                        break
        
        # If still no entries, look for common academic section words
        if not entries:
            academic_sections = ['abstract', 'introduction', 'method', 'results', 'conclusion', 'references']
            text_lower = toc_text.lower()
            
            for section in academic_sections:
                if section in text_lower:
                    content_type = self._classify_section_type(section)
                    entries.append(TOCEntry(
                        level=1,
                        title=section.capitalize(),
                        page=None,
                        section_number=None,
                        content_type=content_type
                    ))
        
        return entries
    
    def _determine_section_level(self, section_number: Optional[str]) -> int:
        """Determine hierarchical level of section."""
        if not section_number:
            return 1
        
        # Count dots for decimal numbering (1.2.3 = level 3)
        if '.' in section_number:
            return len(section_number.split('.'))
        
        # Roman numerals are usually level 1
        if re.match(r'^[ivxlcdm]+$', section_number.lower()):
            return 1
        
        # Single letter/number = level 1
        return 1
    
    def _classify_section_type(self, title: str) -> str:
        """Classify section based on title content."""
        title_lower = title.lower()
        
        for section_type, keywords in self.section_types.items():
            if any(keyword in title_lower for keyword in keywords):
                return section_type
        
        return 'other'
    
    def analyze_scope_indicators(self, entries: List[TOCEntry]) -> Dict[str, int]:
        """Analyze scope indicators from TOC entries."""
        scope_counts = defaultdict(int)
        
        for entry in entries:
            title_lower = entry.title.lower()
            for scope_type, keywords in self.scope_indicators.items():
                if any(keyword in title_lower for keyword in keywords):
                    scope_counts[scope_type] += 1
        
        return dict(scope_counts)
    
    def assess_scale(self, entries: List[TOCEntry]) -> str:
        """Assess research scale based on TOC structure."""
        total_sections = len(entries)
        methodology_sections = sum(1 for e in entries if e.content_type == 'methodology')
        validation_sections = sum(1 for e in entries if e.content_type == 'validation')
        
        # Scale assessment logic
        if total_sections >= 15 and methodology_sections >= 3:
            return "Large-scale comprehensive study"
        elif total_sections >= 8 and methodology_sections >= 2:
            return "Medium-scale focused research"
        elif total_sections >= 5:
            return "Compact targeted study"
        else:
            return "Brief communication or position paper"
    
    def calculate_methodology_depth(self, entries: List[TOCEntry]) -> float:
        """Calculate methodology depth score (0-1)."""
        if not entries:
            return 0.0
        
        methodology_entries = [e for e in entries if e.content_type == 'methodology']
        total_entries = len(entries)
        
        # Base score from proportion
        proportion_score = len(methodology_entries) / total_entries
        
        # Bonus for hierarchical methodology (sub-sections)
        hierarchy_bonus = 0.0
        for entry in methodology_entries:
            if entry.level > 1:  # Has sub-sections
                hierarchy_bonus += 0.1
        
        return min(proportion_score + hierarchy_bonus, 1.0)
    
    def assess_organization_quality(self, entries: List[TOCEntry]) -> float:
        """Assess organization quality based on TOC structure (0-1)."""
        if not entries:
            return 0.0
        
        score = 0.0
        
        # Logical flow bonus
        expected_order = ['background', 'methodology', 'results', 'discussion', 'conclusion']
        entry_types = [e.content_type for e in entries if e.content_type in expected_order]
        
        if len(entry_types) >= 3:
            # Check if sections appear in logical order
            order_score = 0.0
            for i, section_type in enumerate(entry_types):
                if section_type in expected_order:
                    expected_pos = expected_order.index(section_type)
                    # Bonus for correct positioning
                    if i <= expected_pos + 1:  # Allow some flexibility
                        order_score += 0.2
            score += min(order_score, 0.6)
        
        # Hierarchy consistency bonus
        levels = [e.level for e in entries]
        if len(set(levels)) > 1:  # Has hierarchy
            score += 0.2
        
        # Balanced sections bonus
        content_distribution = defaultdict(int)
        for entry in entries:
            content_distribution[entry.content_type] += 1
        
        if len(content_distribution) >= 4:  # Covers multiple content types
            score += 0.2
        
        return min(score, 1.0)
    
    def calculate_content_distribution(self, entries: List[TOCEntry]) -> Dict[str, float]:
        """Calculate content distribution percentages."""
        if not entries:
            return {}
        
        content_counts = defaultdict(int)
        for entry in entries:
            content_counts[entry.content_type] += 1
        
        total = len(entries)
        return {content_type: count/total for content_type, count in content_counts.items()}
    
    def analyze_paper(self, pdf_path: str) -> TOCAnalysis:
        """Complete TOC analysis of a research paper."""
        # Extract text
        full_text, pages = self.extract_pdf_text_structured(pdf_path)
        
        if not pages:
            return TOCAnalysis([], {}, "Unable to analyze", 0.0, 0.0, {})
        
        # Detect TOC section
        toc_text = self.detect_toc_section(full_text, pages)
        
        if not toc_text:
            # Fallback: analyze first page structure
            entries = self.parse_toc_entries(pages[0]) if pages else []
        else:
            entries = self.parse_toc_entries(toc_text)
        
        # Perform analyses
        scope_indicators = self.analyze_scope_indicators(entries)
        scale_assessment = self.assess_scale(entries)
        methodology_depth = self.calculate_methodology_depth(entries)
        organization_quality = self.assess_organization_quality(entries)
        content_distribution = self.calculate_content_distribution(entries)
        
        return TOCAnalysis(
            entries=entries,
            scope_indicators=scope_indicators,
            scale_assessment=scale_assessment,
            methodology_depth=methodology_depth,
            organization_quality=organization_quality,
            content_distribution=content_distribution
        )
    
    def batch_analyze(self, paper_directory: str) -> Dict[str, TOCAnalysis]:
        """Analyze all papers in a directory."""
        results = {}
        paper_dir = Path(paper_directory)
        
        # Find all PDF files
        pdf_files = list(paper_dir.glob("**/*.pdf"))
        
        print(f"Analyzing TOC structure for {len(pdf_files)} papers...")
        
        for pdf_file in pdf_files:
            print(f"  Processing: {pdf_file.name}")
            try:
                analysis = self.analyze_paper(str(pdf_file))
                results[pdf_file.stem] = analysis
            except Exception as e:
                print(f"    Error analyzing {pdf_file.name}: {e}")
                continue
        
        return results
    
    def generate_analysis_report(self, results: Dict[str, TOCAnalysis]) -> str:
        """Generate comprehensive analysis report."""
        if not results:
            return "No papers analyzed."
        
        report = ["TOC ANALYSIS REPORT", "=" * 50, ""]
        
        # Summary statistics
        total_papers = len(results)
        avg_methodology_depth = sum(r.methodology_depth for r in results.values()) / total_papers
        avg_organization_quality = sum(r.organization_quality for r in results.values()) / total_papers
        
        report.extend([
            f"Papers Analyzed: {total_papers}",
            f"Average Methodology Depth: {avg_methodology_depth:.2f}",
            f"Average Organization Quality: {avg_organization_quality:.2f}",
            ""
        ])
        
        # Scale distribution
        scale_counts = defaultdict(int)
        for analysis in results.values():
            scale_counts[analysis.scale_assessment] += 1
        
        report.extend(["SCALE DISTRIBUTION:", "-" * 20])
        for scale, count in sorted(scale_counts.items()):
            percentage = (count / total_papers) * 100
            report.append(f"  {scale}: {count} papers ({percentage:.1f}%)")
        report.append("")
        
        # Scope indicators aggregation
        all_scope_indicators = defaultdict(int)
        for analysis in results.values():
            for scope, count in analysis.scope_indicators.items():
                all_scope_indicators[scope] += count
        
        if all_scope_indicators:
            report.extend(["SCOPE INDICATORS:", "-" * 20])
            for scope, count in sorted(all_scope_indicators.items(), key=lambda x: x[1], reverse=True):
                report.append(f"  {scope.capitalize()}: {count} mentions")
            report.append("")
        
        # Top organized papers
        organized_papers = sorted(results.items(), key=lambda x: x[1].organization_quality, reverse=True)[:5]
        report.extend(["TOP ORGANIZED PAPERS:", "-" * 20])
        for paper_id, analysis in organized_papers:
            report.append(f"  {paper_id}: {analysis.organization_quality:.2f} quality score")
            report.append(f"    Scale: {analysis.scale_assessment}")
            report.append(f"    Sections: {len(analysis.entries)}")
        
        return "\n".join(report)

def main():
    """Demo TOC analysis."""
    analyzer = TOCAnalyzer()
    
    # Analyze papers in repository
    paper_repo_path = Path(__file__).parent.parent / "paper_repository"
    
    if paper_repo_path.exists():
        results = analyzer.batch_analyze(str(paper_repo_path))
        report = analyzer.generate_analysis_report(results)
        print(report)
        
        # Save detailed results
        output_file = Path(__file__).parent / "toc_analysis_results.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
            f.write("\n\nDETAILED ANALYSIS:\n" + "=" * 50 + "\n")
            
            for paper_id, analysis in results.items():
                f.write(f"\nPAPER: {paper_id}\n")
                f.write(f"Scale: {analysis.scale_assessment}\n")
                f.write(f"Methodology Depth: {analysis.methodology_depth:.2f}\n")
                f.write(f"Organization Quality: {analysis.organization_quality:.2f}\n")
                f.write("TOC Entries:\n")
                for entry in analysis.entries:
                    f.write(f"  L{entry.level}: {entry.title} ({entry.content_type})\n")
                f.write("\n")
        
        print(f"\nDetailed results saved to: {output_file}")
    else:
        print("Paper repository not found. Please ensure papers are in the paper_repository directory.")

if __name__ == "__main__":
    main()