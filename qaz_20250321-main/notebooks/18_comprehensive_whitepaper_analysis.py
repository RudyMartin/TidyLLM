#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Whitepaper Analysis with Real Substance

This script analyzes real academic whitepapers to extract:
- Table of Contents (TOC)
- Captions (figures, tables, images)
- Links and references
- Bibliography
- Section summaries

Usage:
    python3 notebooks/18_comprehensive_whitepaper_analysis.py

Requirements:
    pip install pandas numpy matplotlib seaborn plotly pdfplumber
"""

import os
import sys
import json
import time
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pdfplumber

# Add src to path for backend imports
sys.path.insert(0, '../src')

# Set up plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ComprehensiveWhitepaperAnalysis:
    """Comprehensive analysis of real academic whitepapers."""
    
    def __init__(self):
        """Initialize the analysis with real whitepapers."""
        self.demo_output_dir = Path("../demo_output")
        self.archive_dir = Path("../archive/archive/academic_research")
        self.results = {}
        
        print("🔬 Comprehensive Whitepaper Analysis with Real Substance")
        print("=" * 60)
        
    def get_whitepapers(self) -> List[Path]:
        """Get list of real academic whitepapers."""
        whitepapers = []
        
        # Get from demo_output (AI/ML papers)
        if self.demo_output_dir.exists():
            demo_pdfs = list(self.demo_output_dir.glob("*.pdf"))
            whitepapers.extend(demo_pdfs)
        
        # Get from archive (academic research)
        if self.archive_dir.exists():
            archive_pdfs = list(self.archive_dir.glob("*.pdf"))
            whitepapers.extend(archive_pdfs)
        
        print(f"📚 Found {len(whitepapers)} academic whitepapers:")
        
        for paper in whitepapers:
            size_mb = paper.stat().st_size / (1024 * 1024)
            print(f"   • {paper.name} ({size_mb:.1f} MB)")
        
        return whitepapers
    
    def extract_comprehensive_content(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract comprehensive content from PDF including TOC, captions, links, bibliography."""
        print(f"\n📄 **Analyzing: {pdf_path.name}**")
        print("-" * 50)
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Extract all text
                all_text = ""
                page_texts = []
                
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        all_text += f"\n--- Page {i+1} ---\n{text}\n"
                        page_texts.append(text)
                
                # Extract document info
                doc_info = {
                    'filename': pdf_path.name,
                    'page_count': len(pdf.pages),
                    'file_size_mb': pdf_path.stat().st_size / (1024 * 1024),
                    'total_text_length': len(all_text)
                }
                
                # Extract title from first page
                title = self._extract_title(page_texts[0] if page_texts else "")
                doc_info['title'] = title
                
                print(f"   📖 Title: {title}")
                print(f"   📄 Pages: {doc_info['page_count']}")
                print(f"   📊 Text: {doc_info['total_text_length']} characters")
                
                # Comprehensive analysis
                toc = self._extract_toc(all_text)
                captions = self._extract_captions(all_text)
                links = self._extract_links(all_text)
                bibliography = self._extract_bibliography(all_text)
                sections = self._extract_sections(all_text)
                
                return {
                    'document_info': doc_info,
                    'table_of_contents': toc,
                    'captions': captions,
                    'links': links,
                    'bibliography': bibliography,
                    'sections': sections,
                    'raw_content': all_text
                }
                
        except Exception as e:
            print(f"❌ Error analyzing {pdf_path.name}: {e}")
            return {
                'document_info': {'filename': pdf_path.name, 'error': str(e)},
                'table_of_contents': {},
                'captions': [],
                'links': [],
                'bibliography': [],
                'sections': {},
                'raw_content': ""
            }
    
    def _extract_title(self, first_page_text: str) -> str:
        """Extract document title from first page."""
        lines = first_page_text.split('\n')
        
        # Look for title patterns
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if line and len(line) > 10 and len(line) < 200:
                # Skip common non-title lines
                if not any(skip in line.lower() for skip in ['abstract', 'introduction', 'page', 'doi:', 'arxiv:']):
                    return line
        
        return "Unknown Title"
    
    def _extract_toc(self, content: str) -> Dict[str, str]:
        """Extract comprehensive table of contents."""
        print("   📋 Extracting Table of Contents...")
        
        toc = {}
        lines = content.split('\n')
        
        # Multiple TOC patterns
        toc_patterns = [
            # Pattern 1: Numbered sections (1. Introduction, 2. Methods, etc.)
            r'^(\d+\.)\s*(.+)$',
            # Pattern 2: Roman numerals (I. Introduction, II. Methods, etc.)
            r'^([IVX]+\.)\s*(.+)$',
            # Pattern 3: Lettered sections (A. Introduction, B. Methods, etc.)
            r'^([A-Z]\.)\s*(.+)$',
            # Pattern 4: Decimal sections (1.1 Introduction, 1.2 Methods, etc.)
            r'^(\d+\.\d+)\s*(.+)$',
            # Pattern 5: All caps titles
            r'^([A-Z\s]{5,50})$'
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            for pattern in toc_patterns:
                match = re.match(pattern, line)
                if match:
                    section_id = match.group(1).strip()
                    section_title = match.group(2).strip() if len(match.groups()) > 1 else line
                    
                    # Filter out very short or very long titles
                    if 3 < len(section_title) < 100:
                        toc[section_id] = section_title
                        break
        
        print(f"   Found {len(toc)} TOC sections")
        return toc
    
    def _extract_captions(self, content: str) -> List[Dict[str, str]]:
        """Extract figure, table, and image captions."""
        print("   🖼️ Extracting Captions...")
        
        captions = []
        
        # Caption patterns
        caption_patterns = [
            # Figure captions
            r'Figure\s+(\d+[\.:]?\d*)[\.:]?\s*(.+)',
            r'Fig\.\s+(\d+[\.:]?\d*)[\.:]?\s*(.+)',
            # Table captions
            r'Table\s+(\d+[\.:]?\d*)[\.:]?\s*(.+)',
            r'Tab\.\s+(\d+[\.:]?\d*)[\.:]?\s*(.+)',
            # Image captions
            r'Image\s+(\d+[\.:]?\d*)[\.:]?\s*(.+)',
            r'Img\.\s+(\d+[\.:]?\d*)[\.:]?\s*(.+)',
            # Algorithm captions
            r'Algorithm\s+(\d+[\.:]?\d*)[\.:]?\s*(.+)',
            r'Alg\.\s+(\d+[\.:]?\d*)[\.:]?\s*(.+)'
        ]
        
        lines = content.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            for pattern in caption_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    caption_type = "Figure" if "figure" in pattern.lower() else \
                                 "Table" if "table" in pattern.lower() else \
                                 "Image" if "image" in pattern.lower() else \
                                 "Algorithm" if "algorithm" in pattern.lower() else "Other"
                    
                    captions.append({
                        'type': caption_type,
                        'number': match.group(1),
                        'text': match.group(2).strip(),
                        'line_number': i + 1,
                        'full_line': line
                    })
                    break
        
        print(f"   Found {len(captions)} captions")
        return captions
    
    def _extract_links(self, content: str) -> List[Dict[str, str]]:
        """Extract links, URLs, and references."""
        print("   🔗 Extracting Links and References...")
        
        links = []
        
        # URL patterns
        url_patterns = [
            r'https?://[^\s]+',
            r'www\.[^\s]+',
            r'doi\.org/[^\s]+',
            r'arxiv\.org/[^\s]+',
            r'github\.com/[^\s]+'
        ]
        
        # Reference patterns
        ref_patterns = [
            r'\[(\d+)\]\s*(.+)',
            r'\(([A-Za-z]+\s+et\s+al\.?\s*,\s*\d{4})\)',
            r'([A-Za-z]+\s+et\s+al\.?\s*,\s*\d{4})',
            r'([A-Z][a-z]+\s+&\s+[A-Z][a-z]+,\s*\d{4})'
        ]
        
        lines = content.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Extract URLs
            for pattern in url_patterns:
                matches = re.findall(pattern, line)
                for match in matches:
                    links.append({
                        'type': 'URL',
                        'content': match,
                        'line_number': i + 1,
                        'context': line[:100] + "..." if len(line) > 100 else line
                    })
            
            # Extract references
            for pattern in ref_patterns:
                matches = re.findall(pattern, line)
                for match in matches:
                    links.append({
                        'type': 'Reference',
                        'content': match,
                        'line_number': i + 1,
                        'context': line[:100] + "..." if len(line) > 100 else line
                    })
        
        print(f"   Found {len(links)} links/references")
        return links
    
    def _extract_bibliography(self, content: str) -> List[Dict[str, str]]:
        """Extract bibliography/references section."""
        print("   📚 Extracting Bibliography...")
        
        bibliography = []
        
        # Find bibliography section
        bib_patterns = [
            r'References?',
            r'Bibliography',
            r'Works\s+Cited',
            r'Literature\s+Cited'
        ]
        
        lines = content.split('\n')
        in_bib_section = False
        bib_start = -1
        
        # Find bibliography section start
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            for pattern in bib_patterns:
                if re.search(pattern, line_lower):
                    in_bib_section = True
                    bib_start = i
                    break
            if in_bib_section:
                break
        
        if in_bib_section:
            # Extract bibliography entries
            bib_lines = lines[bib_start:]
            
            for line in bib_lines:
                line = line.strip()
                if not line:
                    continue
                
                # Skip section headers
                if any(header in line.lower() for header in ['references', 'bibliography', 'works cited']):
                    continue
                
                # Look for bibliography entry patterns
                if re.match(r'^\d+\.\s', line) or \
                   re.match(r'^\[[A-Za-z]+\]', line) or \
                   re.match(r'^[A-Z][a-z]+,\s*[A-Z]\.', line) or \
                   re.match(r'^[A-Z][a-z]+\s+et\s+al\.', line):
                    
                    bibliography.append({
                        'entry': line,
                        'line_number': bib_start + len(bibliography) + 1
                    })
        
        print(f"   Found {len(bibliography)} bibliography entries")
        return bibliography
    
    def _extract_sections(self, content: str) -> Dict[str, Dict[str, Any]]:
        """Extract sections with summaries."""
        print("   📑 Extracting Sections with Summaries...")
        
        sections = {}
        lines = content.split('\n')
        
        # Find section boundaries
        current_section = None
        current_content = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a section header
            if re.match(r'^\d+\.\s', line) or \
               re.match(r'^[IVX]+\.\s', line) or \
               re.match(r'^[A-Z]\.\s', line) or \
               re.match(r'^\d+\.\d+\s', line):
                
                # Save previous section
                if current_section and current_content:
                    sections[current_section] = {
                        'content': '\n'.join(current_content),
                        'summary': self._generate_section_summary('\n'.join(current_content)),
                        'line_count': len(current_content),
                        'word_count': len(' '.join(current_content).split())
                    }
                
                # Start new section
                current_section = line
                current_content = []
            elif current_section:
                current_content.append(line)
        
        # Save last section
        if current_section and current_content:
            sections[current_section] = {
                'content': '\n'.join(current_content),
                'summary': self._generate_section_summary('\n'.join(current_content)),
                'line_count': len(current_content),
                'word_count': len(' '.join(current_content).split())
            }
        
        print(f"   Found {len(sections)} sections")
        return sections
    
    def _generate_section_summary(self, content: str) -> str:
        """Generate a brief summary of section content."""
        if len(content) < 100:
            return content
        
        # Extract key sentences (first few sentences)
        sentences = re.split(r'[.!?]+', content)
        key_sentences = [s.strip() for s in sentences if len(s.strip()) > 20][:3]
        
        summary = '. '.join(key_sentences)
        if len(summary) > 200:
            summary = summary[:200] + "..."
        
        return summary
    
    def analyze_whitepapers(self):
        """Analyze multiple whitepapers comprehensively."""
        whitepapers = self.get_whitepapers()
        
        if not whitepapers:
            print("❌ No whitepapers found for analysis")
            return
        
        # Select representative whitepapers
        selected_papers = whitepapers[:5]  # First 5 papers
        
        print(f"\n🔬 Running comprehensive analysis on {len(selected_papers)} whitepapers...")
        
        for paper_path in selected_papers:
            start_time = time.time()
            
            result = self.extract_comprehensive_content(paper_path)
            
            end_time = time.time()
            duration = end_time - start_time
            
            if 'error' not in result['document_info']:
                # Store results
                self.results[paper_path.stem] = {
                    'result': result,
                    'duration': duration,
                    'paper_path': str(paper_path)
                }
                
                # Display summary
                self._display_paper_summary(paper_path.name, result, duration)
            else:
                print(f"❌ Analysis failed for {paper_path.name}: {result['document_info']['error']}")
    
    def _display_paper_summary(self, filename: str, result: Dict, duration: float):
        """Display summary of paper analysis."""
        print(f"\n📊 **Analysis Summary: {filename}**")
        print(f"   ⏱️ Duration: {duration:.2f} seconds")
        print(f"   📋 TOC Sections: {len(result['table_of_contents'])}")
        print(f"   🖼️ Captions: {len(result['captions'])}")
        print(f"   🔗 Links/References: {len(result['links'])}")
        print(f"   📚 Bibliography: {len(result['bibliography'])}")
        print(f"   📑 Sections: {len(result['sections'])}")
        
        # Show top TOC sections
        toc = result['table_of_contents']
        if toc:
            top_sections = list(toc.items())[:5]
            print(f"   📋 Top Sections: {', '.join([f'{k}: {v[:30]}...' for k, v in top_sections])}")
    
    def generate_markdown_report(self) -> str:
        """Generate comprehensive markdown report."""
        if not self.results:
            return "# ❌ No Analysis Results Available"
        
        report = []
        report.append("# 🔬 Comprehensive Whitepaper Analysis Report")
        report.append("")
        report.append("*Real Academic Papers with Substantial Content Analysis*")
        report.append("")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Papers Analyzed:** {len(self.results)}")
        report.append("")
        
        # Executive Summary
        report.append("## 📋 Executive Summary")
        report.append("")
        
        total_toc = sum(len(data['result']['table_of_contents']) for data in self.results.values())
        total_captions = sum(len(data['result']['captions']) for data in self.results.values())
        total_links = sum(len(data['result']['links']) for data in self.results.values())
        total_bib = sum(len(data['result']['bibliography']) for data in self.results.values())
        total_sections = sum(len(data['result']['sections']) for data in self.results.values())
        
        report.append(f"- **Total Papers Analyzed:** {len(self.results)}")
        report.append(f"- **Total TOC Sections:** {total_toc}")
        report.append(f"- **Total Captions:** {total_captions}")
        report.append(f"- **Total Links/References:** {total_links}")
        report.append(f"- **Total Bibliography Entries:** {total_bib}")
        report.append(f"- **Total Content Sections:** {total_sections}")
        report.append("")
        
        # Detailed Analysis for Each Paper
        for paper_name, data in self.results.items():
            result = data['result']
            
            report.append(f"## 📄 {result['document_info']['title']}")
            report.append("")
            report.append(f"**File:** `{result['document_info']['filename']}`")
            report.append(f"**Pages:** {result['document_info']['page_count']}")
            report.append(f"**Size:** {result['document_info']['file_size_mb']:.1f} MB")
            report.append(f"**Analysis Time:** {data['duration']:.2f} seconds")
            report.append("")
            
            # Table of Contents
            if result['table_of_contents']:
                report.append("### 📋 Table of Contents")
                report.append("")
                for section_id, title in result['table_of_contents'].items():
                    report.append(f"- **{section_id}** {title}")
                report.append("")
            
            # Captions
            if result['captions']:
                report.append("### 🖼️ Captions")
                report.append("")
                for caption in result['captions'][:10]:  # Show first 10
                    report.append(f"- **{caption['type']} {caption['number']}:** {caption['text']}")
                if len(result['captions']) > 10:
                    report.append(f"- *... and {len(result['captions']) - 10} more captions*")
                report.append("")
            
            # Links and References
            if result['links']:
                report.append("### 🔗 Links and References")
                report.append("")
                for link in result['links'][:10]:  # Show first 10
                    report.append(f"- **{link['type']}:** {link['content']}")
                if len(result['links']) > 10:
                    report.append(f"- *... and {len(result['links']) - 10} more links*")
                report.append("")
            
            # Bibliography
            if result['bibliography']:
                report.append("### 📚 Bibliography")
                report.append("")
                for entry in result['bibliography'][:10]:  # Show first 10
                    report.append(f"- {entry['entry']}")
                if len(result['bibliography']) > 10:
                    report.append(f"- *... and {len(result['bibliography']) - 10} more entries*")
                report.append("")
            
            # Section Summaries
            if result['sections']:
                report.append("### 📑 Section Summaries")
                report.append("")
                for section_id, section_data in list(result['sections'].items())[:5]:  # Show first 5
                    report.append(f"#### {section_id}")
                    report.append(f"**Summary:** {section_data['summary']}")
                    report.append(f"**Word Count:** {section_data['word_count']}")
                    report.append("")
                if len(result['sections']) > 5:
                    report.append(f"*... and {len(result['sections']) - 5} more sections*")
                report.append("")
            
            report.append("---")
            report.append("")
        
        # Comparative Analysis
        report.append("## 📊 Comparative Analysis")
        report.append("")
        
        # Create comparison table
        comparison_data = []
        for paper_name, data in self.results.items():
            result = data['result']
            comparison_data.append({
                'Paper': result['document_info']['title'][:50] + "..." if len(result['document_info']['title']) > 50 else result['document_info']['title'],
                'Pages': result['document_info']['page_count'],
                'TOC Sections': len(result['table_of_contents']),
                'Captions': len(result['captions']),
                'Links': len(result['links']),
                'Bibliography': len(result['bibliography']),
                'Sections': len(result['sections']),
                'Analysis Time (s)': f"{data['duration']:.2f}"
            })
        
        # Add table header
        report.append("| Paper | Pages | TOC | Captions | Links | Bibliography | Sections | Time (s) |")
        report.append("|-------|-------|-----|----------|-------|--------------|----------|----------|")
        
        # Add table rows
        for row in comparison_data:
            report.append(f"| {row['Paper']} | {row['Pages']} | {row['TOC Sections']} | {row['Captions']} | {row['Links']} | {row['Bibliography']} | {row['Sections']} | {row['Analysis Time (s)']} |")
        
        report.append("")
        
        # Key Findings
        report.append("## 🔍 Key Findings")
        report.append("")
        
        # Find papers with most content
        most_toc = max(self.results.items(), key=lambda x: len(x[1]['result']['table_of_contents']))
        most_captions = max(self.results.items(), key=lambda x: len(x[1]['result']['captions']))
        most_links = max(self.results.items(), key=lambda x: len(x[1]['result']['links']))
        most_bib = max(self.results.items(), key=lambda x: len(x[1]['result']['bibliography']))
        
        report.append(f"- **Most Structured:** {most_toc[1]['result']['document_info']['title']} ({len(most_toc[1]['result']['table_of_contents'])} TOC sections)")
        report.append(f"- **Most Visual Content:** {most_captions[1]['result']['document_info']['title']} ({len(most_captions[1]['result']['captions'])} captions)")
        report.append(f"- **Most Referenced:** {most_links[1]['result']['document_info']['title']} ({len(most_links[1]['result']['links'])} links/references)")
        report.append(f"- **Most Comprehensive Bibliography:** {most_bib[1]['result']['document_info']['title']} ({len(most_bib[1]['result']['bibliography'])} entries)")
        report.append("")
        
        # Discussion Points
        report.append("## 💭 Discussion Points")
        report.append("")
        report.append("### Content Quality Assessment")
        report.append("- **Structure:** Papers with comprehensive TOC demonstrate better organization")
        report.append("- **Visual Elements:** Caption analysis reveals use of figures, tables, and diagrams")
        report.append("- **Research Rigor:** Bibliography depth indicates thorough literature review")
        report.append("- **Connectivity:** Link analysis shows integration with external resources")
        report.append("")
        
        report.append("### Technical Implementation")
        report.append("- **PDF Processing:** Successfully extracted structured content from academic PDFs")
        report.append("- **Pattern Recognition:** Identified multiple TOC, caption, and reference patterns")
        report.append("- **Content Analysis:** Generated meaningful section summaries")
        report.append("- **Performance:** Average analysis time of {:.2f} seconds per paper".format(
            sum(data['duration'] for data in self.results.values()) / len(self.results)
        ))
        report.append("")
        
        report.append("### Recommendations")
        report.append("- **Enhanced Caption Detection:** Implement more sophisticated figure/table recognition")
        report.append("- **Reference Parsing:** Add structured bibliography parsing (BibTeX, etc.)")
        report.append("- **Content Classification:** Categorize sections by type (methodology, results, etc.)")
        report.append("- **Cross-Reference Analysis:** Track internal and external citation patterns")
        report.append("")
        
        return "\n".join(report)
    
    def save_report(self, filename: str = "comprehensive_whitepaper_analysis.md"):
        """Save the markdown report to file."""
        report_content = self.generate_markdown_report()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\n📄 Report saved: {filename}")
        return filename
    
    def run_full_analysis(self):
        """Run the complete comprehensive analysis."""
        print("🔬 **Comprehensive Whitepaper Analysis**")
        print("=" * 60)
        
        # 1. Analyze whitepapers
        self.analyze_whitepapers()
        
        # 2. Generate and save report
        if self.results:
            report_file = self.save_report()
            print(f"\n✅ Analysis complete! Report saved as: {report_file}")
            
            # Show preview
            print(f"\n📋 **Report Preview**")
            print("-" * 30)
            report_content = self.generate_markdown_report()
            lines = report_content.split('\n')
            for line in lines[:20]:
                print(line)
            print("...")
        else:
            print("❌ No results to report")


def main():
    """Main analysis execution."""
    analyzer = ComprehensiveWhitepaperAnalysis()
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
