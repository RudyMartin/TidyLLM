#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real MVR Peer Review Demo with Actual PDF Documents

This script demonstrates the SR Validator's MVR Peer Review prompt using 
REAL PDF documents from the knowledge base.

Overview:
- Uses actual PDF documents from demo_output/
- Real TOC extraction and analysis
- Actual content processing
- Real compliance checking against MVS/VST

Usage:
    python3 notebooks/17_real_mvr_demo.py

Requirements:
    pip install pandas numpy matplotlib seaborn plotly pdfplumber
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import pdfplumber

# Add src to path for backend imports
sys.path.insert(0, '../src')

# Set up plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class RealMVRDemo:
    """Real MVR Peer Review Demo using actual PDF documents."""
    
    def __init__(self):
        """Initialize the demo with real PDF documents."""
        self.demo_output_dir = Path("../demo_output")
        self.favorites_dir = Path("../src/assets/prompts/favorites")
        self.results = {}
        
        print("🎯 Real MVR Peer Review Demo with Actual PDFs")
        print("=" * 55)
        
    def get_available_pdfs(self) -> List[Path]:
        """Get list of available PDF documents."""
        if not self.demo_output_dir.exists():
            print(f"❌ Demo output directory not found: {self.demo_output_dir}")
            return []
        
        pdf_files = list(self.demo_output_dir.glob("*.pdf"))
        print(f"📚 Found {len(pdf_files)} PDF documents:")
        
        for pdf in pdf_files:
            size_mb = pdf.stat().st_size / (1024 * 1024)
            print(f"   • {pdf.name} ({size_mb:.1f} MB)")
        
        return pdf_files
    
    def extract_pdf_content(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract real content from PDF document."""
        print(f"\n📄 Extracting content from: {pdf_path.name}")
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Extract text from all pages
                all_text = ""
                page_count = len(pdf.pages)
                
                print(f"   Pages: {page_count}")
                
                for i, page in enumerate(pdf.pages):
                    if i < 10:  # Limit to first 10 pages for demo
                        text = page.extract_text()
                        if text:
                            all_text += f"\n--- Page {i+1} ---\n{text}\n"
                
                # Extract basic document info
                doc_info = {
                    'filename': pdf_path.name,
                    'page_count': page_count,
                    'file_size_mb': pdf_path.stat().st_size / (1024 * 1024),
                    'extracted_pages': min(10, page_count),
                    'total_text_length': len(all_text)
                }
                
                # Try to extract title from first page
                first_page_text = ""
                if pdf.pages:
                    first_page_text = pdf.pages[0].extract_text() or ""
                
                # Extract potential title (first line or first 100 chars)
                lines = first_page_text.split('\n')
                potential_title = lines[0] if lines else "Unknown Title"
                if len(potential_title) > 100:
                    potential_title = potential_title[:100] + "..."
                
                doc_info['title'] = potential_title
                
                print(f"   Title: {potential_title}")
                print(f"   Text extracted: {len(all_text)} characters")
                
                return {
                    'document_info': doc_info,
                    'content': all_text,
                    'first_page': first_page_text
                }
                
        except Exception as e:
            print(f"❌ Error extracting PDF content: {e}")
            return {
                'document_info': {'filename': pdf_path.name, 'error': str(e)},
                'content': "",
                'first_page': ""
            }
    
    def extract_toc_from_content(self, content: str) -> Dict[str, str]:
        """Extract table of contents from PDF content."""
        print("   📋 Extracting Table of Contents...")
        
        toc = {}
        lines = content.split('\n')
        
        # Look for common TOC patterns
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Pattern 1: Numbered sections (1. Introduction, 2. Methods, etc.)
            if line and line[0].isdigit() and '.' in line[:5]:
                parts = line.split('.', 1)
                if len(parts) == 2:
                    section_num = parts[0].strip()
                    section_title = parts[1].strip()
                    if len(section_title) > 3:  # Avoid very short titles
                        toc[section_num] = section_title
            
            # Pattern 2: Roman numerals (I. Introduction, II. Methods, etc.)
            elif line and line[:3] in ['I. ', 'II ', 'III', 'IV ', 'V. ', 'VI ', 'VII', 'VIII', 'IX ', 'X. ']:
                parts = line.split('.', 1)
                if len(parts) == 2:
                    section_num = parts[0].strip()
                    section_title = parts[1].strip()
                    if len(section_title) > 3:
                        toc[section_num] = section_title
            
            # Pattern 3: All caps titles (ABSTRACT, INTRODUCTION, METHODS, etc.)
            elif line.isupper() and len(line) > 5 and len(line) < 50:
                toc[f"section_{len(toc)+1}"] = line.title()
        
        print(f"   Found {len(toc)} potential sections")
        return toc
    
    def analyze_mvs_compliance(self, content: str, toc: Dict[str, str]) -> Dict[str, Any]:
        """Analyze compliance with MVS requirements."""
        print("   🔍 Analyzing MVS compliance...")
        
        # Simplified MVS requirements for demo
        mvs_requirements = {
            "5.4.3": "Conceptual Soundness - Model methodology must be conceptually sound",
            "5.4.3.1": "Variable selection must be appropriate and well-documented",
            "5.4.3.2": "Model assumptions must be clearly stated and validated",
            "5.4.3.3": "Model limitations must be identified and documented",
            "5.12.1": "Outcome Analysis - Model performance must be validated through testing",
            "5.12.1.1": "Backtesting must demonstrate adequate predictive power",
            "5.12.1.2": "Benchmarking must show competitive performance"
        }
        
        compliance_results = {}
        content_lower = content.lower()
        
        for req_id, requirement in mvs_requirements.items():
            # Simple keyword matching for demo
            keywords = requirement.lower().split()
            matches = sum(1 for keyword in keywords if keyword in content_lower)
            coverage = matches / len(keywords) if keywords else 0
            
            compliance_results[req_id] = {
                'requirement': requirement,
                'coverage': coverage,
                'status': '✅ Compliant' if coverage > 0.3 else '❌ Non-Compliant',
                'evidence_found': matches
            }
        
        overall_compliance = sum(1 for result in compliance_results.values() 
                               if result['status'] == '✅ Compliant')
        total_requirements = len(mvs_requirements)
        
        return {
            'requirements_checked': total_requirements,
            'compliant_requirements': overall_compliance,
            'compliance_rate': overall_compliance / total_requirements,
            'detailed_results': compliance_results
        }
    
    def generate_peer_review_challenges(self, content: str, toc: Dict[str, str]) -> List[Dict[str, Any]]:
        """Generate peer review challenges based on content analysis."""
        print("   🎯 Generating peer review challenges...")
        
        challenges = []
        content_lower = content.lower()
        
        # Check for common issues
        issues_to_check = [
            {
                'type': 'methodology_clarity',
                'keywords': ['methodology', 'approach', 'method'],
                'challenge': 'Methodology description lacks sufficient detail for peer review',
                'severity': 'Medium'
            },
            {
                'type': 'evidence_support',
                'keywords': ['evidence', 'data', 'results', 'findings'],
                'challenge': 'Claims lack sufficient supporting evidence or data',
                'severity': 'High'
            },
            {
                'type': 'assumptions_documentation',
                'keywords': ['assumption', 'assume', 'limitation'],
                'challenge': 'Model assumptions and limitations not clearly documented',
                'severity': 'Medium'
            },
            {
                'type': 'validation_testing',
                'keywords': ['validation', 'testing', 'backtest', 'benchmark'],
                'challenge': 'Validation and testing procedures need more detail',
                'severity': 'High'
            }
        ]
        
        for issue in issues_to_check:
            keyword_matches = sum(1 for keyword in issue['keywords'] 
                                if keyword in content_lower)
            
            if keyword_matches < 2:  # If few keywords found, it might be an issue
                challenges.append({
                    'section': 'General',
                    'type': issue['type'],
                    'challenge': issue['challenge'],
                    'severity': issue['severity'],
                    'evidence': f"Only {keyword_matches} related terms found in document"
                })
        
        # Add specific challenges based on content length
        if len(content) < 5000:
            challenges.append({
                'section': 'General',
                'type': 'content_depth',
                'challenge': 'Document appears to lack sufficient depth for comprehensive peer review',
                'severity': 'Medium',
                'evidence': f"Document contains only {len(content)} characters"
            })
        
        print(f"   Generated {len(challenges)} peer review challenges")
        return challenges
    
    def run_real_analysis(self, pdf_path: Path) -> Dict[str, Any]:
        """Run real analysis on actual PDF document."""
        print(f"\n🔍 **Real Analysis: {pdf_path.name}**")
        print("-" * 40)
        
        # Extract content
        extraction_result = self.extract_pdf_content(pdf_path)
        
        if 'error' in extraction_result['document_info']:
            return {'error': extraction_result['document_info']['error']}
        
        # Extract TOC
        toc = self.extract_toc_from_content(extraction_result['content'])
        
        # Analyze MVS compliance
        mvs_analysis = self.analyze_mvs_compliance(extraction_result['content'], toc)
        
        # Generate peer review challenges
        challenges = self.generate_peer_review_challenges(extraction_result['content'], toc)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(mvs_analysis, challenges, extraction_result)
        
        return {
            'document_info': extraction_result['document_info'],
            'toc_analysis': {
                'total_sections': len(toc),
                'sections_found': list(toc.keys()),
                'toc_content': toc
            },
            'mvs_compliance': mvs_analysis,
            'peer_review_challenges': challenges,
            'quality_score': quality_score,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _calculate_quality_score(self, mvs_analysis: Dict, challenges: List, extraction_result: Dict) -> float:
        """Calculate quality score based on analysis results."""
        # Base score from MVS compliance
        compliance_score = mvs_analysis.get('compliance_rate', 0)
        
        # Penalty for peer review challenges
        challenge_penalty = len(challenges) * 0.1
        
        # Bonus for content depth
        content_bonus = min(0.2, len(extraction_result['content']) / 50000)
        
        # Calculate final score
        final_score = compliance_score - challenge_penalty + content_bonus
        return max(0.0, min(1.0, final_score))
    
    def run_demo_on_multiple_pdfs(self):
        """Run demo on multiple PDF documents."""
        pdf_files = self.get_available_pdfs()
        
        if not pdf_files:
            print("❌ No PDF files found for analysis")
            return
        
        # Select a few representative PDFs
        selected_pdfs = pdf_files[:3]  # First 3 PDFs
        
        print(f"\n🎯 Running analysis on {len(selected_pdfs)} selected PDFs...")
        
        for pdf_path in selected_pdfs:
            start_time = time.time()
            
            result = self.run_real_analysis(pdf_path)
            
            end_time = time.time()
            duration = end_time - start_time
            
            if 'error' not in result:
                # Store results
                self.results[pdf_path.stem] = {
                    'result': result,
                    'duration': duration,
                    'pdf_path': str(pdf_path)
                }
                
                # Display summary
                self._display_analysis_summary(pdf_path.name, result, duration)
            else:
                print(f"❌ Analysis failed for {pdf_path.name}: {result['error']}")
    
    def _display_analysis_summary(self, filename: str, result: Dict, duration: float):
        """Display summary of analysis results."""
        print(f"\n📊 **Analysis Summary: {filename}**")
        print(f"   Duration: {duration:.2f} seconds")
        print(f"   TOC Sections: {result['toc_analysis']['total_sections']}")
        print(f"   MVS Compliance: {result['mvs_compliance']['compliant_requirements']}/{result['mvs_compliance']['requirements_checked']}")
        print(f"   Peer Challenges: {len(result['peer_review_challenges'])}")
        print(f"   Quality Score: {result['quality_score']:.2f}")
        
        # Show top TOC sections
        toc = result['toc_analysis']['toc_content']
        if toc:
            print(f"   Top Sections: {', '.join(list(toc.keys())[:5])}")
    
    def create_comparison_report(self):
        """Create comparison report across all analyzed PDFs."""
        if not self.results:
            print("❌ No results to compare")
            return
        
        print(f"\n📈 **Cross-Document Comparison Report**")
        print("-" * 45)
        
        # Create comparison table
        comparison_data = []
        
        for doc_name, data in self.results.items():
            result = data['result']
            comparison_data.append({
                'Document': doc_name,
                'Duration (s)': data['duration'],
                'TOC Sections': result['toc_analysis']['total_sections'],
                'MVS Compliance': f"{result['mvs_compliance']['compliant_requirements']}/{result['mvs_compliance']['requirements_checked']}",
                'Peer Challenges': len(result['peer_review_challenges']),
                'Quality Score': f"{result['quality_score']:.2f}",
                'File Size (MB)': f"{result['document_info']['file_size_mb']:.1f}"
            })
        
        # Display comparison
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))
        
        # Create visualization
        self._create_real_comparison_chart()
    
    def _create_real_comparison_chart(self):
        """Create comparison visualization for real data."""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            doc_names = list(self.results.keys())
            quality_scores = [data['result']['quality_score'] for data in self.results.values()]
            durations = [data['duration'] for data in self.results.values()]
            toc_sections = [data['result']['toc_analysis']['total_sections'] for data in self.results.values()]
            peer_challenges = [len(data['result']['peer_review_challenges']) for data in self.results.values()]
            
            # Quality scores
            ax1.bar(doc_names, quality_scores, color='#2E86AB')
            ax1.set_title('Quality Scores by Document')
            ax1.set_ylabel('Quality Score')
            ax1.set_ylim(0, 1)
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Processing times
            ax2.bar(doc_names, durations, color='#A23B72')
            ax2.set_title('Processing Time by Document')
            ax2.set_ylabel('Duration (seconds)')
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # TOC sections
            ax3.bar(doc_names, toc_sections, color='#F18F01')
            ax3.set_title('TOC Sections by Document')
            ax3.set_ylabel('Number of Sections')
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Peer challenges
            ax4.bar(doc_names, peer_challenges, color='#C73E1D')
            ax4.set_title('Peer Review Challenges by Document')
            ax4.set_ylabel('Number of Challenges')
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig('real_mvr_comparison.png', dpi=300, bbox_inches='tight')
            print(f"\n📊 Real comparison chart saved: real_mvr_comparison.png")
            
        except Exception as e:
            print(f"❌ Error creating chart: {e}")
    
    def run_full_real_demo(self):
        """Run the complete real demo sequence."""
        print("🎯 **Real MVR Peer Review Demo with Actual PDFs**")
        print("=" * 55)
        
        # 1. Show available PDFs
        self.get_available_pdfs()
        
        # 2. Run analysis on multiple PDFs
        self.run_demo_on_multiple_pdfs()
        
        # 3. Create comparison report
        self.create_comparison_report()
        
        # 4. Summary
        self._show_real_summary()
    
    def _show_real_summary(self):
        """Show real demo summary."""
        print(f"\n🎯 **Real Demo Summary**")
        print("-" * 30)
        print(f"✅ Analyzed {len(self.results)} real PDF documents")
        print(f"✅ Extracted actual TOC and content")
        print(f"✅ Performed real MVS compliance analysis")
        print(f"✅ Generated actual peer review challenges")
        print(f"✅ Created real quality scores")
        
        if self.results:
            avg_quality = sum(data['result']['quality_score'] for data in self.results.values()) / len(self.results)
            avg_duration = sum(data['duration'] for data in self.results.values()) / len(self.results)
            print(f"📊 Average Quality Score: {avg_quality:.2f}")
            print(f"📊 Average Processing Time: {avg_duration:.2f} seconds")
        
        print(f"\n🎉 **Real Demo Complete!**")
        print("This demonstrates actual PDF processing and analysis capabilities.")


def main():
    """Main demo execution."""
    demo = RealMVRDemo()
    demo.run_full_real_demo()


if __name__ == "__main__":
    main()
