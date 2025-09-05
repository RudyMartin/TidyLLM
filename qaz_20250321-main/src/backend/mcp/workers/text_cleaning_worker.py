#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text Cleaning Worker

Dedicated worker for cleaning text content by removing HTML, JavaScript, and other unwanted elements.
This worker implements the light_bleach functionality for safe text processing.
"""

import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from html import unescape
from html.parser import HTMLParser

logger = logging.getLogger(__name__)


class HTMLStripper(HTMLParser):
    """Drop all tags and keep text data."""
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.parts = []
    
    def handle_data(self, data): 
        self.parts.append(data)
    
    def handle_entityref(self, name): 
        self.parts.append(f"&{name};")
    
    def handle_charref(self, name): 
        self.parts.append(f"&#{name};")
    
    def get_data(self): 
        return "".join(self.parts)


class TextCleaningWorker:
    """Dedicated worker for cleaning text content"""
    
    def __init__(self):
        self.worker_id = f"text_cleaning_worker_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.processing_stats = {
            'files_processed': 0,
            'successful_cleans': 0,
            'failed_cleans': 0,
            'html_removed': 0,
            'js_removed': 0,
            'total_characters_removed': 0
        }
        
        # Compile regex patterns for efficiency
        self.script_style_re = re.compile(r"(?is)<(script|style)\b[^>]*>.*?</\1\s*>")
        self.tag_re = re.compile(r"(?s)<[^>]+>")  # any HTML tag
        self.fenced_js_re = re.compile(r"(?is)```(?:js|javascript)\s.*?```")
        self.event_handler_attr_re = re.compile(r"""\s(on\w+)\s*=\s*(['"]).*?\2""", re.I | re.S)
        self.js_url_attr_re = re.compile(r"""\s(href|src)\s*=\s*(['"])\s*javascript:[^'"]*\2""", re.I)
        
        logger.info("Text Cleaning Worker initialized")
    
    def clean_text(self, text: str, remove_fenced_js: bool = False) -> Dict[str, Any]:
        """Clean text content by removing HTML, JavaScript, and other unwanted elements"""
        try:
            original_length = len(text)
            original_text = text
            
            # Track what was removed
            removed_elements = {
                'html_tags': 0,
                'script_blocks': 0,
                'style_blocks': 0,
                'event_handlers': 0,
                'js_urls': 0,
                'fenced_js': 0
            }
            
            # 1) Remove script/style blocks entirely
            text, script_style_count = self.script_style_re.subn("", text)
            removed_elements['script_blocks'] += script_style_count
            
            # 2) Remove event handlers and javascript: URLs from remaining tags
            text, event_handler_count = self.event_handler_attr_re.subn("", text)
            removed_elements['event_handlers'] += event_handler_count
            
            text, js_url_count = self.js_url_attr_re.subn("", text)
            removed_elements['js_urls'] += js_url_count
            
            # 3) Remove all HTML tags
            text, tag_count = self.tag_re.subn("", text)
            removed_elements['html_tags'] += tag_count
            
            # 4) Remove fenced JS code blocks (markdown)
            if remove_fenced_js:
                text, fenced_js_count = self.fenced_js_re.subn("", text)
                removed_elements['fenced_js'] += fenced_js_count
            
            # 5) Unescape HTML entities (&amp; -> &)
            text = unescape(text)
            
            # 6) Use HTMLStripper in case input is malformed HTML
            stripper = HTMLStripper()
            stripper.feed(text)
            stripper.close()
            text = stripper.get_data()
            
            # 7) Normalize whitespace
            text = re.sub(r"[ \t]+\n", "\n", text)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            
            # Calculate statistics
            final_length = len(text)
            characters_removed = original_length - final_length
            
            # Update processing stats
            self.processing_stats['successful_cleans'] += 1
            self.processing_stats['files_processed'] += 1
            self.processing_stats['total_characters_removed'] += characters_removed
            
            if removed_elements['html_tags'] > 0:
                self.processing_stats['html_removed'] += 1
            
            if (removed_elements['script_blocks'] > 0 or 
                removed_elements['event_handlers'] > 0 or 
                removed_elements['js_urls'] > 0 or 
                removed_elements['fenced_js'] > 0):
                self.processing_stats['js_removed'] += 1
            
            logger.info(f"✅ Successfully cleaned text (removed {characters_removed} characters)")
            
            return {
                'success': True,
                'cleaned_text': text,
                'original_length': original_length,
                'final_length': final_length,
                'characters_removed': characters_removed,
                'removed_elements': removed_elements,
                'cleaning_summary': {
                    'html_tags_removed': removed_elements['html_tags'],
                    'script_blocks_removed': removed_elements['script_blocks'],
                    'style_blocks_removed': removed_elements['style_blocks'],
                    'event_handlers_removed': removed_elements['event_handlers'],
                    'js_urls_removed': removed_elements['js_urls'],
                    'fenced_js_removed': removed_elements['fenced_js'],
                    'total_elements_removed': sum(removed_elements.values())
                },
                'processing_time': datetime.now().isoformat(),
                'worker_id': self.worker_id
            }
            
        except Exception as e:
            self.processing_stats['failed_cleans'] += 1
            self.processing_stats['files_processed'] += 1
            
            logger.error(f"❌ Failed to clean text: {e}")
            return {
                'success': False,
                'error': str(e),
                'original_text': text,
                'processing_time': datetime.now().isoformat(),
                'worker_id': self.worker_id
            }
    
    def clean_file(self, file_path: str, remove_fenced_js: bool = False, inplace: bool = False) -> Dict[str, Any]:
        """Clean a text file and optionally save the cleaned version"""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Read the file
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                original_text = f.read()
            
            # Clean the text
            result = self.clean_text(original_text, remove_fenced_js)
            
            if result['success'] and inplace:
                # Save cleaned text back to the same file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(result['cleaned_text'])
                result['saved_inplace'] = True
                result['file_path'] = str(file_path)
                logger.info(f"✅ Saved cleaned text inplace: {file_path}")
            elif result['success']:
                # Save to a new file with .clean.txt extension
                clean_file_path = file_path.with_suffix(file_path.suffix + '.clean.txt')
                with open(clean_file_path, 'w', encoding='utf-8') as f:
                    f.write(result['cleaned_text'])
                result['clean_file_path'] = str(clean_file_path)
                result['file_path'] = str(file_path)
                logger.info(f"✅ Saved cleaned text to: {clean_file_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to clean file {file_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'file_path': str(file_path),
                'processing_time': datetime.now().isoformat(),
                'worker_id': self.worker_id
            }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            'worker_id': self.worker_id,
            'stats': self.processing_stats.copy(),
            'timestamp': datetime.now().isoformat()
        }
    
    def validate_cleaning_needed(self, text: str) -> Dict[str, Any]:
        """Check if text needs cleaning by detecting HTML/JS patterns"""
        analysis = {
            'needs_cleaning': False,
            'html_detected': False,
            'js_detected': False,
            'script_tags': 0,
            'style_tags': 0,
            'html_tags': 0,
            'event_handlers': 0,
            'js_urls': 0,
            'fenced_js': 0
        }
        
        # Check for various patterns
        analysis['script_tags'] = len(self.script_style_re.findall(text))
        analysis['html_tags'] = len(self.tag_re.findall(text))
        analysis['event_handlers'] = len(self.event_handler_attr_re.findall(text))
        analysis['js_urls'] = len(self.js_url_attr_re.findall(text))
        analysis['fenced_js'] = len(self.fenced_js_re.findall(text))
        
        # Determine if cleaning is needed
        analysis['html_detected'] = analysis['html_tags'] > 0
        analysis['js_detected'] = (analysis['script_tags'] > 0 or 
                                 analysis['event_handlers'] > 0 or 
                                 analysis['js_urls'] > 0 or 
                                 analysis['fenced_js'] > 0)
        
        analysis['needs_cleaning'] = analysis['html_detected'] or analysis['js_detected']
        
        return analysis


# Convenience function for easy usage
def clean_text(text: str, remove_fenced_js: bool = False) -> str:
    """Convenience function to clean text"""
    worker = TextCleaningWorker()
    result = worker.clean_text(text, remove_fenced_js)
    
    if result['success']:
        return result['cleaned_text']
    else:
        raise ValueError(f"Failed to clean text: {result['error']}")


if __name__ == "__main__":
    # Test the worker
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        worker = TextCleaningWorker()
        result = worker.clean_file(file_path, remove_fenced_js=True)
        print(f"Cleaning result: {result['success']}")
        if result['success']:
            print(f"Characters removed: {result['characters_removed']}")
            print(f"Elements removed: {result['cleaning_summary']}")
    else:
        print("Usage: python text_cleaning_worker.py <file_path>")
