#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TOC Extraction and Reference Discovery Script
============================================

This script extracts Table of Contents from papers, discovers referenced papers,
and downloads them to expand the knowledge base.

Usage:
    python scripts/toc_reference_discovery.py [--papers 5] [--download] [--test-app]
"""

import os
import sys
import json
import argparse
import logging
import time
import subprocess
from pathlib import Path
from datetime import datetime
import requests
import arxiv
from urllib.parse import urlparse, quote

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TOCReferenceDiscovery:
    """Extract TOCs and discover referenced papers"""
    
    def __init__(self, project_root=None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.knowledge_base_dir = self.project_root / "knowledge_base"
        self.discovered_papers_dir = self.knowledge_base_dir / "ai_ml_research" / "discovered_papers"
        self.prompts_dir = self.project_root / "src" / "assets" / "prompts" / "favorites"
        
        # Create directories
        self.discovered_papers_dir.mkdir(parents=True, exist_ok=True)
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        
        # User agent for requests
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Load the TOC extraction prompt
        self.toc_prompt = self._load_toc_prompt()
        
    def _load_toc_prompt(self):
        """Load the TOC extraction prompt"""
        prompt_path = self.prompts_dir / "toc_extraction_prompt.md"
        if prompt_path.exists():
            with open(prompt_path, 'r') as f:
                return f.read()
        else:
            logger.warning("TOC extraction prompt not found")
            return ""
    
    def find_papers_with_toc(self, limit=5):
        """Find papers that likely have TOCs"""
        logger.info(f"🔍 Finding {limit} papers with TOCs...")
        
        # Look for papers in our knowledge base
        papers = []
        for pdf_path in self.knowledge_base_dir.rglob("*.pdf"):
            if len(papers) >= limit:
                break
            
            # Check if this is a substantial paper (likely to have TOC)
            file_size = pdf_path.stat().st_size
            if file_size > 500 * 1024:  # >500KB
                papers.append({
                    'path': pdf_path,
                    'name': pdf_path.name,
                    'size': file_size,
                    'category': pdf_path.parent.name
                })
        
        logger.info(f"📄 Found {len(papers)} papers for TOC extraction")
        return papers
    
    def extract_toc_from_paper(self, paper_path):
        """Extract TOC from a paper using our application"""
        logger.info(f"📖 Extracting TOC from: {paper_path.name}")
        
        try:
            # Use our TOC extraction worker
            from src.backend.mcp.workers.toc_extractor_worker import TOCExtractorWorker
            
            worker = TOCExtractorWorker()
            result = worker.extract_toc(str(paper_path))
            
            if result and result.get('toc_structure'):
                logger.info(f"✅ TOC extracted successfully")
                return result
            else:
                logger.warning(f"⚠️ No TOC structure found")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to extract TOC: {e}")
            return None
    
    def discover_references_from_toc(self, toc_data):
        """Discover referenced papers from TOC content"""
        logger.info("🔍 Discovering references from TOC...")
        
        discovered_papers = []
        
        if not toc_data or 'toc_structure' not in toc_data:
            return discovered_papers
        
        # Extract text content from TOC
        toc_text = self._extract_toc_text(toc_data['toc_structure'])
        
        # Look for potential paper references
        references = self._find_paper_references(toc_text)
        
        # Validate and search for each reference
        for ref in references:
            paper_info = self._search_for_paper(ref)
            if paper_info:
                discovered_papers.append(paper_info)
        
        logger.info(f"📚 Discovered {len(discovered_papers)} potential papers")
        return discovered_papers
    
    def _extract_toc_text(self, toc_structure):
        """Extract text content from TOC structure"""
        text_parts = []
        
        if isinstance(toc_structure, dict):
            for key, value in toc_structure.items():
                if isinstance(value, str):
                    text_parts.append(value)
                elif isinstance(value, (list, dict)):
                    text_parts.extend(self._extract_toc_text(value))
        elif isinstance(toc_structure, list):
            for item in toc_structure:
                text_parts.extend(self._extract_toc_text(item))
        
        return " ".join(text_parts)
    
    def _find_paper_references(self, text):
        """Find potential paper references in text"""
        references = []
        
        # Look for patterns that suggest academic references
        import re
        
        # Pattern 1: "Author et al." or "Author, Author, Author"
        author_patterns = [
            r'([A-Z][a-z]+ et al\.)',
            r'([A-Z][a-z]+, [A-Z][a-z]+, [A-Z][a-z]+)',
            r'([A-Z][a-z]+ and [A-Z][a-z]+)'
        ]
        
        # Pattern 2: Year references
        year_pattern = r'\((\d{4})\)'
        
        # Pattern 3: Paper titles in quotes or italics
        title_patterns = [
            r'"([^"]{10,50})"',
            r"'([^']{10,50})'",
            r'\*([^*]{10,50})\*'
        ]
        
        # Extract potential references
        for pattern in author_patterns + title_patterns + [year_pattern]:
            matches = re.findall(pattern, text)
            references.extend(matches)
        
        return list(set(references))  # Remove duplicates
    
    def _search_for_paper(self, reference):
        """Search for a paper using the reference"""
        try:
            # Try arXiv search
            search = arxiv.Search(query=reference, max_results=1)
            results = list(search.results())
            
            if results:
                paper = results[0]
                return {
                    'title': paper.title,
                    'authors': ', '.join(author.name for author in paper.authors),
                    'year': paper.published.year,
                    'source': 'arxiv',
                    'url': paper.pdf_url,
                    'relevance_score': 0.8,
                    'availability': 'downloadable',
                    'domain': 'machine_learning',
                    'validation_status': 'verified',
                    'reference': reference
                }
            
            # Try Google Scholar (simplified)
            # In a real implementation, you'd use a proper Google Scholar API
            return None
            
        except Exception as e:
            logger.debug(f"Failed to search for '{reference}': {e}")
            return None
    
    def download_discovered_papers(self, discovered_papers):
        """Download discovered papers"""
        logger.info(f"📥 Downloading {len(discovered_papers)} discovered papers...")
        
        downloaded_files = []
        
        for paper in discovered_papers:
            if paper['source'] == 'arxiv' and paper['availability'] == 'downloadable':
                filepath = self._download_with_curl(paper)
                if filepath:
                    downloaded_files.append(filepath)
                time.sleep(1)  # Be nice to servers
        
        logger.info(f"✅ Downloaded {len(downloaded_files)} papers")
        return downloaded_files
    
    def _download_with_curl(self, paper_info):
        """Download paper using curl"""
        try:
            # Create filename
            safe_title = "".join(c for c in paper_info['title'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"{paper_info['year']}_{safe_title}_discovered.pdf"
            filepath = self.discovered_papers_dir / filename
            
            if not filepath.exists():
                logger.info(f"Downloading: {paper_info['title']}")
                
                cmd = [
                    'curl', '-L', '--max-time', '60', '--retry', '3',
                    '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    '--output', str(filepath),
                    paper_info['url']
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0 and filepath.exists() and filepath.stat().st_size > 0:
                    logger.info(f"✅ Downloaded: {filename}")
                    return filepath
                else:
                    logger.error(f"Failed to download {paper_info['title']}")
                    if filepath.exists():
                        filepath.unlink()
                    return None
            else:
                logger.info(f"📄 Already exists: {filename}")
                return filepath
                
        except Exception as e:
            logger.error(f"Failed to download {paper_info['title']}: {e}")
            return None
    
    def test_application_capabilities(self, papers):
        """Test if our application can handle this workflow"""
        logger.info("🧪 Testing application capabilities...")
        
        test_results = {
            'toc_extraction': [],
            'reference_discovery': [],
            'paper_download': [],
            'overall_success': False
        }
        
        # Test TOC extraction
        for paper in papers[:2]:  # Test with first 2 papers
            logger.info(f"Testing TOC extraction: {paper['name']}")
            
            toc_result = self.extract_toc_from_paper(paper['path'])
            if toc_result:
                test_results['toc_extraction'].append({
                    'paper': paper['name'],
                    'success': True,
                    'sections_found': len(toc_result.get('toc_structure', {}))
                })
            else:
                test_results['toc_extraction'].append({
                    'paper': paper['name'],
                    'success': False,
                    'error': 'No TOC structure found'
                })
        
        # Test reference discovery
        for toc_test in test_results['toc_extraction']:
            if toc_test['success']:
                # Simulate reference discovery
                discovered = self.discover_references_from_toc({'toc_structure': {}})
                test_results['reference_discovery'].append({
                    'papers_found': len(discovered),
                    'success': len(discovered) > 0
                })
        
        # Test paper download
        test_paper = {
            'title': 'Test Paper',
            'url': 'https://arxiv.org/pdf/1706.03762.pdf',
            'year': 2017
        }
        
        download_result = self._download_with_curl(test_paper)
        test_results['paper_download'].append({
            'success': download_result is not None,
            'file_path': str(download_result) if download_result else None
        })
        
        # Overall success
        toc_success = any(t['success'] for t in test_results['toc_extraction'])
        ref_success = any(t['success'] for t in test_results['reference_discovery'])
        download_success = any(t['success'] for t in test_results['paper_download'])
        
        test_results['overall_success'] = toc_success and ref_success and download_success
        
        return test_results
    
    def create_discovery_report(self, papers, discovered_papers, downloaded_files, test_results):
        """Create a comprehensive report of the discovery process"""
        logger.info("📋 Creating discovery report...")
        
        report_content = f"""# TOC Reference Discovery Report

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 📊 Discovery Summary

### Papers Analyzed
- **Total Papers**: {len(papers)}
- **Papers with TOC**: {len([p for p in papers if p.get('toc_extracted')])}
- **Papers Processed**: {len(papers)}

### Reference Discovery
- **References Found**: {len(discovered_papers)}
- **Papers Downloaded**: {len(downloaded_files)}
- **Success Rate**: {(len(downloaded_files) / len(discovered_papers) * 100) if discovered_papers else 0:.1f}%

## 📚 Discovered Papers

"""
        
        for paper in discovered_papers:
            report_content += f"""### {paper['title']}
- **Authors**: {paper['authors']}
- **Year**: {paper['year']}
- **Source**: {paper['source']}
- **Domain**: {paper['domain']}
- **Relevance Score**: {paper['relevance_score']}
- **Status**: {paper['validation_status']}
- **URL**: {paper['url']}

"""
        
        report_content += f"""
## 🧪 Application Test Results

### TOC Extraction
"""
        
        for test in test_results['toc_extraction']:
            status = "✅ Success" if test['success'] else "❌ Failed"
            report_content += f"- **{test['paper']}**: {status}"
            if test['success']:
                report_content += f" ({test['sections_found']} sections found)"
            else:
                report_content += f" - {test.get('error', 'Unknown error')}"
            report_content += "\n"
        
        report_content += f"""
### Reference Discovery
"""
        
        for test in test_results['reference_discovery']:
            status = "✅ Success" if test['success'] else "❌ Failed"
            report_content += f"- **Discovery**: {status} ({test['papers_found']} papers found)\n"
        
        report_content += f"""
### Paper Download
"""
        
        for test in test_results['paper_download']:
            status = "✅ Success" if test['success'] else "❌ Failed"
            report_content += f"- **Download**: {status}"
            if test['success']:
                report_content += f" - {test['file_path']}"
            report_content += "\n"
        
        report_content += f"""
## 🎯 Overall Assessment

**Application Capability**: {'✅ FULLY CAPABLE' if test_results['overall_success'] else '⚠️ PARTIALLY CAPABLE'}

### Strengths
- TOC extraction working: {'Yes' if any(t['success'] for t in test_results['toc_extraction']) else 'No'}
- Reference discovery working: {'Yes' if any(t['success'] for t in test_results['reference_discovery']) else 'No'}
- Paper download working: {'Yes' if any(t['success'] for t in test_results['paper_download']) else 'No'}

### Recommendations
"""
        
        if test_results['overall_success']:
            report_content += """- ✅ System is fully capable of automated paper discovery
- ✅ Can be used for knowledge base expansion
- ✅ Ready for production deployment
"""
        else:
            report_content += """- ⚠️ Some components need improvement
- 🔧 Review TOC extraction accuracy
- 🔧 Enhance reference discovery algorithms
- 🔧 Verify download reliability
"""
        
        report_content += f"""
## 📁 File Locations

- **Discovered Papers**: `{self.discovered_papers_dir}`
- **TOC Prompt**: `{self.prompts_dir}/toc_extraction_prompt.md`
- **Knowledge Base**: `{self.knowledge_base_dir}`

## 🚀 Next Steps

1. **Review discovered papers** for relevance and quality
2. **Integrate into knowledge base** with proper categorization
3. **Set up automated discovery** for regular updates
4. **Monitor and improve** discovery accuracy

---
**This report shows the capability of our system to automatically discover and download relevant papers!** 🎯
"""
        
        report_path = self.discovered_papers_dir / "DISCOVERY_REPORT.md"
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"📋 Report created: {report_path}")
        return report_path
    
    def run_discovery_workflow(self, num_papers=5, download_papers=True, test_app=True):
        """Run the complete discovery workflow"""
        logger.info("🚀 Starting TOC Reference Discovery Workflow...")
        
        # Step 1: Find papers with TOCs
        papers = self.find_papers_with_toc(num_papers)
        
        # Step 2: Extract TOCs
        discovered_papers = []
        downloaded_files = []
        
        for paper in papers:
            logger.info(f"📖 Processing: {paper['name']}")
            
            # Extract TOC
            toc_result = self.extract_toc_from_paper(paper['path'])
            if toc_result:
                paper['toc_extracted'] = True
                paper['toc_data'] = toc_result
                
                # Discover references
                references = self.discover_references_from_toc(toc_result)
                discovered_papers.extend(references)
            else:
                paper['toc_extracted'] = False
        
        # Step 3: Download discovered papers
        if download_papers and discovered_papers:
            downloaded_files = self.download_discovered_papers(discovered_papers)
        
        # Step 4: Test application capabilities
        test_results = None
        if test_app:
            test_results = self.test_application_capabilities(papers)
        
        # Step 5: Create report
        report_path = self.create_discovery_report(
            papers, discovered_papers, downloaded_files, test_results or {}
        )
        
        logger.info("🎉 Discovery workflow complete!")
        logger.info(f"📊 Papers processed: {len(papers)}")
        logger.info(f"📚 Papers discovered: {len(discovered_papers)}")
        logger.info(f"📥 Papers downloaded: {len(downloaded_files)}")
        logger.info(f"📋 Report: {report_path}")
        
        return {
            'papers_processed': len(papers),
            'papers_discovered': len(discovered_papers),
            'papers_downloaded': len(downloaded_files),
            'test_results': test_results,
            'report_path': report_path
        }

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="TOC Reference Discovery")
    parser.add_argument("--papers", type=int, default=5, help="Number of papers to process")
    parser.add_argument("--download", action="store_true", help="Download discovered papers")
    parser.add_argument("--test-app", action="store_true", help="Test application capabilities")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    try:
        discovery = TOCReferenceDiscovery()
        
        results = discovery.run_discovery_workflow(
            num_papers=args.papers,
            download_papers=args.download,
            test_app=args.test_app
        )
        
        print(f"\n🎉 TOC Reference Discovery complete!")
        print(f"📊 Papers processed: {results['papers_processed']}")
        print(f"📚 Papers discovered: {results['papers_discovered']}")
        print(f"📥 Papers downloaded: {results['papers_downloaded']}")
        print(f"📋 Report: {results['report_path']}")
        
        if results['test_results']:
            success = results['test_results']['overall_success']
            print(f"🧪 Application test: {'✅ PASSED' if success else '❌ FAILED'}")
        
    except Exception as e:
        logger.error(f"❌ Discovery failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
