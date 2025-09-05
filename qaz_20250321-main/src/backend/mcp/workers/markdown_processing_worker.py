#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Markdown Processing Worker

Dedicated worker for markdown file processing with basic parsing.
This worker handles markdown files with structure analysis.
"""

import logging
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class MarkdownProcessingWorker:
    """Dedicated worker for markdown file processing"""
    
    def __init__(self):
        self.worker_id = f"markdown_worker_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.processing_stats = {
            'files_processed': 0,
            'successful_parses': 0,
            'failed_parses': 0,
            'total_sections': 0,
            'total_links': 0
        }
        
        logger.info("Markdown Processing Worker initialized")
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process markdown file with analysis"""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"Markdown file not found: {file_path}")
            
            # Validate file extension
            if not file_path.suffix.lower() in ['.md', '.markdown']:
                return {
                    'success': False,
                    'error': 'Not a markdown file',
                    'file_path': str(file_path),
                    'processing_time': datetime.now().isoformat(),
                    'worker_id': self.worker_id
                }
            
            # Read markdown file
            content = ""
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Try with different encoding
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
            
            # Clean text content using light_bleach if needed
            cleaning_result = self._clean_text_content(content)
            cleaned_content = cleaning_result.get('cleaned_text', content)
            
            # Analyze markdown content (use cleaned content)
            analysis_result = self._analyze_markdown_content(cleaned_content)
            
            self.processing_stats['successful_parses'] += 1
            self.processing_stats['files_processed'] += 1
            self.processing_stats['total_sections'] += analysis_result['section_count']
            self.processing_stats['total_links'] += analysis_result['link_count']
            
            logger.info(f"✅ Successfully parsed markdown file: {file_path} ({analysis_result['section_count']} sections)")
            
            return {
                'success': True,
                'data': {
                    'content': content,
                    'analysis': analysis_result
                },
                'file_path': str(file_path),
                'processing_time': datetime.now().isoformat(),
                'worker_id': self.worker_id,
                'file_size': file_path.stat().st_size
            }
            
        except Exception as e:
            self.processing_stats['failed_parses'] += 1
            self.processing_stats['files_processed'] += 1
            
            logger.error(f"❌ Failed to process markdown file {file_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'file_path': str(file_path),
                'processing_time': datetime.now().isoformat(),
                'worker_id': self.worker_id
            }
    
    def _clean_text_content(self, content: str) -> Dict[str, Any]:
        """Clean text content using TextCleaningWorker if needed"""
        try:
            # Import and initialize the TextCleaningWorker
            from text_cleaning_worker import TextCleaningWorker
            
            worker = TextCleaningWorker()
            
            # First check if cleaning is needed
            analysis = worker.validate_cleaning_needed(content)
            
            if not analysis['needs_cleaning']:
                logger.info("✅ Markdown content is clean - no bleaching needed")
                return {
                    'success': True,
                    'cleaned_text': content,
                    'cleaning_applied': False,
                    'analysis': analysis
                }
            
            # Clean the text if needed
            logger.info(f"🧹 Cleaning markdown content (HTML: {analysis['html_detected']}, JS: {analysis['js_detected']})")
            result = worker.clean_text(content, remove_fenced_js=True)
            
            if result['success']:
                logger.info(f"✅ Markdown cleaned successfully (removed {result['characters_removed']} characters)")
                return {
                    'success': True,
                    'cleaned_text': result['cleaned_text'],
                    'cleaning_applied': True,
                    'analysis': analysis,
                    'cleaning_stats': result['cleaning_summary']
                }
            else:
                logger.warning(f"⚠️ Markdown cleaning failed: {result['error']}")
                return {
                    'success': False,
                    'cleaned_text': content,  # Return original content
                    'cleaning_applied': False,
                    'error': result['error'],
                    'analysis': analysis
                }
                
        except Exception as e:
            logger.error(f"❌ Error in markdown cleaning: {e}")
            return {
                'success': False,
                'cleaned_text': content,  # Return original content
                'cleaning_applied': False,
                'error': str(e)
            }
    
    def _analyze_markdown_content(self, content: str) -> Dict[str, Any]:
        """Analyze markdown content and provide insights"""
        analysis = {
            'character_count': len(content),
            'line_count': 0,
            'section_count': 0,
            'heading_count': 0,
            'link_count': 0,
            'code_block_count': 0,
            'list_count': 0,
            'table_count': 0,
            'image_count': 0,
            'structure': [],
            'headings': [],
            'links': [],
            'content_type': 'unknown'
        }
        
        try:
            lines = content.split('\n')
            analysis['line_count'] = len(lines)
            
            # Extract headings
            headings = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)
            analysis['heading_count'] = len(headings)
            analysis['headings'] = [{'level': len(h[0]), 'text': h[1].strip()} for h in headings]
            
            # Extract links
            links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
            analysis['link_count'] = len(links)
            analysis['links'] = [{'text': link[0], 'url': link[1]} for link in links]
            
            # Count code blocks
            code_blocks = re.findall(r'```[\s\S]*?```', content)
            analysis['code_block_count'] = len(code_blocks)
            
            # Count lists
            list_items = re.findall(r'^[\s]*[-*+]\s+', content, re.MULTILINE)
            analysis['list_count'] = len(list_items)
            
            # Count tables
            table_rows = re.findall(r'^\|.*\|$', content, re.MULTILINE)
            analysis['table_count'] = len([row for row in table_rows if '---' not in row])
            
            # Count images
            images = re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', content)
            analysis['image_count'] = len(images)
            
            # Build structure
            analysis['structure'] = self._build_markdown_structure(content)
            
            # Determine content type
            analysis['content_type'] = self._detect_markdown_content_type(content)
            
        except Exception as e:
            analysis['error'] = str(e)
        
        return analysis
    
    def _build_markdown_structure(self, content: str) -> List[Dict[str, Any]]:
        """Build a hierarchical structure of the markdown document"""
        structure = []
        lines = content.split('\n')
        current_section = None
        
        for line in lines:
            # Check for headings
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if heading_match:
                level = len(heading_match.group(1))
                text = heading_match.group(2).strip()
                
                section = {
                    'type': 'heading',
                    'level': level,
                    'text': text,
                    'content': []
                }
                
                if level == 1:
                    structure.append(section)
                    current_section = section
                else:
                    if current_section:
                        current_section['content'].append(section)
            
            # Add content to current section
            elif current_section and line.strip():
                current_section['content'].append({
                    'type': 'content',
                    'text': line.strip()
                })
        
        return structure
    
    def _detect_markdown_content_type(self, content: str) -> str:
        """Detect the type of markdown content"""
        content_lower = content.lower()
        
        # Check for different content types
        if any(word in content_lower for word in ['readme', 'getting started', 'installation']):
            return 'readme'
        elif any(word in content_lower for word in ['api', 'reference', 'documentation']):
            return 'documentation'
        elif any(word in content_lower for word in ['changelog', 'version', 'release']):
            return 'changelog'
        elif any(word in content_lower for word in ['tutorial', 'guide', 'how to']):
            return 'tutorial'
        elif any(word in content_lower for word in ['report', 'analysis', 'summary']):
            return 'report'
        else:
            return 'general_markdown'
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            'worker_id': self.worker_id,
            'stats': self.processing_stats.copy(),
            'timestamp': datetime.now().isoformat()
        }
    
    def validate_markdown_file(self, file_path: str) -> Dict[str, Any]:
        """Validate markdown file without processing"""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return {
                    'valid': False,
                    'error': 'File not found',
                    'file_path': str(file_path)
                }
            
            # Check file extension
            if not file_path.suffix.lower() in ['.md', '.markdown']:
                return {
                    'valid': False,
                    'error': 'Not a markdown file',
                    'file_path': str(file_path)
                }
            
            # Check if file is readable
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline()
                
                return {
                    'valid': True,
                    'file_path': str(file_path),
                    'file_size': file_path.stat().st_size,
                    'first_line': first_line.strip()[:50] + '...' if len(first_line) > 50 else first_line.strip()
                }
                
            except UnicodeDecodeError:
                # Try with different encoding
                with open(file_path, 'r', encoding='latin-1') as f:
                    first_line = f.readline()
                
                return {
                    'valid': True,
                    'file_path': str(file_path),
                    'file_size': file_path.stat().st_size,
                    'first_line': first_line.strip()[:50] + '...' if len(first_line) > 50 else first_line.strip(),
                    'encoding': 'latin-1'
                }
                
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'file_path': str(file_path)
            }


# Convenience function for easy usage
def load_markdown_file(file_path: str) -> Dict[str, Any]:
    """Convenience function to load markdown file"""
    worker = MarkdownProcessingWorker()
    result = worker.process_document(file_path)
    
    if result['success']:
        return result['data']
    else:
        raise ValueError(f"Failed to load markdown file: {result['error']}")


if __name__ == "__main__":
    # Test the worker
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        worker = MarkdownProcessingWorker()
        result = worker.process_document(file_path)
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python markdown_processing_worker.py <markdown_file_path>")
