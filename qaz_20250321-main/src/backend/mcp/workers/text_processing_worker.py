#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text Processing Worker

Dedicated worker for text file processing with analysis capabilities.
This worker handles plain text files with content analysis and validation.
"""

import logging
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class TextProcessingWorker:
    """Dedicated worker for text file processing"""
    
    def __init__(self):
        self.worker_id = f"text_worker_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.processing_stats = {
            'files_processed': 0,
            'successful_parses': 0,
            'failed_parses': 0,
            'total_characters': 0,
            'total_words': 0,
            'total_lines': 0
        }
        
        logger.info("Text Processing Worker initialized")
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process text file with analysis"""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"Text file not found: {file_path}")
            
            # Validate file extension
            if not file_path.suffix.lower() == '.txt':
                return {
                    'success': False,
                    'error': 'Not a text file',
                    'file_path': str(file_path),
                    'processing_time': datetime.now().isoformat(),
                    'worker_id': self.worker_id
                }
            
            # Read text file
            content = ""
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Try with different encoding
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
            
            # Clean text content using light_bleach
            cleaning_result = self._clean_text_content(content)
            cleaned_content = cleaning_result.get('cleaned_text', content)
            
            # Analyze text content (use cleaned content)
            analysis_result = self._analyze_text_content(cleaned_content)
            
            self.processing_stats['successful_parses'] += 1
            self.processing_stats['files_processed'] += 1
            self.processing_stats['total_characters'] += len(content)
            self.processing_stats['total_words'] += analysis_result['word_count']
            self.processing_stats['total_lines'] += analysis_result['line_count']
            
            logger.info(f"✅ Successfully parsed text file: {file_path} ({analysis_result['word_count']} words)")
            
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
            
            logger.error(f"❌ Failed to process text file {file_path}: {e}")
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
                logger.info("✅ Text content is clean - no bleaching needed")
                return {
                    'success': True,
                    'cleaned_text': content,
                    'cleaning_applied': False,
                    'analysis': analysis
                }
            
            # Clean the text if needed
            logger.info(f"🧹 Cleaning text content (HTML: {analysis['html_detected']}, JS: {analysis['js_detected']})")
            result = worker.clean_text(content, remove_fenced_js=True)
            
            if result['success']:
                logger.info(f"✅ Text cleaned successfully (removed {result['characters_removed']} characters)")
                return {
                    'success': True,
                    'cleaned_text': result['cleaned_text'],
                    'cleaning_applied': True,
                    'analysis': analysis,
                    'cleaning_stats': result['cleaning_summary']
                }
            else:
                logger.warning(f"⚠️ Text cleaning failed: {result['error']}")
                return {
                    'success': False,
                    'cleaned_text': content,  # Return original content
                    'cleaning_applied': False,
                    'error': result['error'],
                    'analysis': analysis
                }
                
        except Exception as e:
            logger.error(f"❌ Error in text cleaning: {e}")
            return {
                'success': False,
                'cleaned_text': content,  # Return original content
                'cleaning_applied': False,
                'error': str(e)
            }
    
    def _analyze_text_content(self, content: str) -> Dict[str, Any]:
        """Analyze text content and provide insights"""
        analysis = {
            'character_count': len(content),
            'word_count': 0,
            'line_count': 0,
            'sentence_count': 0,
            'paragraph_count': 0,
            'average_word_length': 0,
            'average_sentence_length': 0,
            'unique_words': 0,
            'most_common_words': [],
            'language_indicators': {},
            'content_type': 'unknown'
        }
        
        try:
            # Basic counts
            lines = content.split('\n')
            analysis['line_count'] = len(lines)
            
            # Word analysis
            words = re.findall(r'\b\w+\b', content.lower())
            analysis['word_count'] = len(words)
            
            if words:
                analysis['average_word_length'] = sum(len(word) for word in words) / len(words)
                analysis['unique_words'] = len(set(words))
                
                # Most common words
                word_freq = {}
                for word in words:
                    if len(word) > 2:  # Skip very short words
                        word_freq[word] = word_freq.get(word, 0) + 1
                
                analysis['most_common_words'] = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Sentence analysis
            sentences = re.split(r'[.!?]+', content)
            analysis['sentence_count'] = len([s for s in sentences if s.strip()])
            
            if analysis['sentence_count'] > 0:
                analysis['average_sentence_length'] = analysis['word_count'] / analysis['sentence_count']
            
            # Paragraph analysis
            paragraphs = [p for p in content.split('\n\n') if p.strip()]
            analysis['paragraph_count'] = len(paragraphs)
            
            # Language indicators
            analysis['language_indicators'] = self._detect_language_indicators(content)
            
            # Content type detection
            analysis['content_type'] = self._detect_content_type(content)
            
        except Exception as e:
            analysis['error'] = str(e)
        
        return analysis
    
    def _detect_language_indicators(self, content: str) -> Dict[str, Any]:
        """Detect language indicators in text"""
        indicators = {
            'english_indicators': 0,
            'technical_indicators': 0,
            'code_indicators': 0,
            'likely_language': 'unknown'
        }
        
        # English indicators
        english_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        indicators['english_indicators'] = sum(1 for word in english_words if word in content.lower())
        
        # Technical indicators
        technical_words = ['algorithm', 'function', 'method', 'class', 'object', 'variable', 'parameter', 'return']
        indicators['technical_indicators'] = sum(1 for word in technical_words if word in content.lower())
        
        # Code indicators
        code_patterns = ['def ', 'class ', 'import ', 'from ', 'if __name__', 'function', 'var ', 'const ']
        indicators['code_indicators'] = sum(1 for pattern in code_patterns if pattern in content)
        
        # Determine likely language
        if indicators['code_indicators'] > 2:
            indicators['likely_language'] = 'code'
        elif indicators['technical_indicators'] > 3:
            indicators['likely_language'] = 'technical'
        elif indicators['english_indicators'] > 5:
            indicators['likely_language'] = 'english'
        
        return indicators
    
    def _detect_content_type(self, content: str) -> str:
        """Detect the type of content"""
        content_lower = content.lower()
        
        # Check for different content types
        if any(word in content_lower for word in ['log', 'error', 'warning', 'debug', 'info']):
            return 'log_file'
        elif any(word in content_lower for word in ['config', 'setting', 'parameter', 'option']):
            return 'configuration'
        elif any(word in content_lower for word in ['readme', 'documentation', 'guide', 'manual']):
            return 'documentation'
        elif any(word in content_lower for word in ['data', 'record', 'entry', 'row']):
            return 'data_file'
        elif any(word in content_lower for word in ['report', 'analysis', 'summary', 'conclusion']):
            return 'report'
        else:
            return 'general_text'
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            'worker_id': self.worker_id,
            'stats': self.processing_stats.copy(),
            'timestamp': datetime.now().isoformat()
        }
    
    def validate_text_file(self, file_path: str) -> Dict[str, Any]:
        """Validate text file without processing"""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return {
                    'valid': False,
                    'error': 'File not found',
                    'file_path': str(file_path)
                }
            
            # Check file extension
            if not file_path.suffix.lower() == '.txt':
                return {
                    'valid': False,
                    'error': 'Not a text file',
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
def load_text_file(file_path: str) -> Dict[str, Any]:
    """Convenience function to load text file"""
    worker = TextProcessingWorker()
    result = worker.process_document(file_path)
    
    if result['success']:
        return result['data']
    else:
        raise ValueError(f"Failed to load text file: {result['error']}")


if __name__ == "__main__":
    # Test the worker
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        worker = TextProcessingWorker()
        result = worker.process_document(file_path)
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python text_processing_worker.py <text_file_path>")
