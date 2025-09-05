#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple QA Orchestrator

Basic QA orchestrator with minimal dependencies and core functionality.
This is the foundation level for progressive complexity architecture.
"""

import logging
import uuid
import json
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from .qa_orchestrator import QAOrchestrator

logger = logging.getLogger(__name__)


class SimpleQAOrchestrator(QAOrchestrator):
    """Basic QA orchestrator with minimal dependencies"""
    
    def __init__(self):
        super().__init__()
        self.session_id = str(uuid.uuid4())
        self.quality_metrics = {}
        
        logger.info(f"Simple QA Orchestrator initialized with session ID: {self.session_id}")
    
    def process_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Basic document processing with simple quality checks"""
        
        start_time = datetime.now()
        
        try:
            logger.info("Starting simple document processing...")
            
            # Handle None document
            if document is None:
                raise ValueError("Document cannot be None")
            
            # Extract basic document information
            document_info = self._extract_document_info(document)
            
            # Perform basic quality checks
            quality_checks = self._perform_basic_quality_checks(document)
            
            # Calculate simple quality score
            quality_score = self._calculate_simple_quality_score(quality_checks)
            
            # Generate basic report
            report = self._generate_basic_report(document_info, quality_checks, quality_score)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                'document': document,
                'document_info': document_info,
                'quality_checks': quality_checks,
                'quality_score': quality_score,
                'report': report,
                'processing_time_ms': processing_time,
                'session_id': self.session_id,
                'orchestrator_type': 'simple',
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error in simple document processing: {e}")
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                'document': document,
                'document_info': {},
                'quality_checks': {},
                'quality_score': 0.0,
                'report': {'error': str(e)},
                'processing_time_ms': processing_time,
                'session_id': self.session_id,
                'orchestrator_type': 'simple',
                'status': 'error',
                'error': str(e)
            }
    
    def _extract_document_info(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Extract basic document information"""
        
        content = document.get('content', '')
        metadata = document.get('metadata', {})
        
        # Basic text analysis
        word_count = len(content.split()) if content else 0
        character_count = len(content) if content else 0
        line_count = len(content.split('\n')) if content else 0
        
        # Basic structure detection
        has_title = bool(re.search(r'^#+\s+', content, re.MULTILINE)) if content else False
        has_sections = bool(re.search(r'^##+\s+', content, re.MULTILINE)) if content else False
        has_lists = bool(re.search(r'^[\-\*]\s+', content, re.MULTILINE)) if content else False
        has_numbers = bool(re.search(r'\d+', content)) if content else False
        
        # File information
        file_path = document.get('file_path') or document.get('path', '')
        file_extension = Path(file_path).suffix.lower() if file_path else ''
        
        return {
            'word_count': word_count,
            'character_count': character_count,
            'line_count': line_count,
            'has_title': has_title,
            'has_sections': has_sections,
            'has_lists': has_lists,
            'has_numbers': has_numbers,
            'file_extension': file_extension,
            'file_path': file_path,
            'document_type': metadata.get('type', 'unknown'),
            'title': metadata.get('title', ''),
            'author': metadata.get('author', ''),
            'created_date': metadata.get('created_date', ''),
            'modified_date': metadata.get('modified_date', '')
        }
    
    def _perform_basic_quality_checks(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Perform basic quality checks on document"""
        
        content = document.get('content', '')
        document_info = self._extract_document_info(document)
        
        checks = {
            'content_length': {
                'check': 'Content has sufficient length',
                'passed': document_info['word_count'] >= 50,
                'value': document_info['word_count'],
                'threshold': 50,
                'weight': 0.2
            },
            'structure_presence': {
                'check': 'Document has basic structure',
                'passed': document_info['has_title'] or document_info['has_sections'],
                'value': {
                    'has_title': document_info['has_title'],
                    'has_sections': document_info['has_sections']
                },
                'weight': 0.3
            },
            'readability': {
                'check': 'Content is readable',
                'passed': self._check_readability(content),
                'value': self._calculate_readability_score(content),
                'threshold': 0.3,
                'weight': 0.2
            },
            'format_consistency': {
                'check': 'Format is consistent',
                'passed': self._check_format_consistency(content),
                'value': self._calculate_format_consistency_score(content),
                'threshold': 0.5,
                'weight': 0.15
            },
            'content_quality': {
                'check': 'Content quality indicators',
                'passed': self._check_content_quality(content),
                'value': self._calculate_content_quality_score(content),
                'threshold': 0.4,
                'weight': 0.15
            }
        }
        
        return checks
    
    def _check_readability(self, content: str) -> bool:
        """Check if content is readable"""
        if not content:
            return False
        
        # Simple readability check based on sentence length and word complexity
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        if not sentences:
            return False
        
        # Check average sentence length
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        
        # Check for very long sentences (hard to read)
        long_sentences = sum(1 for s in sentences if len(s.split()) > 50)
        long_sentence_ratio = long_sentences / len(sentences)
        
        # Check for very short sentences (might be fragmented)
        short_sentences = sum(1 for s in sentences if len(s.split()) < 3)
        short_sentence_ratio = short_sentences / len(sentences)
        
        # Pass if reasonable sentence lengths
        return (avg_sentence_length >= 5 and 
                avg_sentence_length <= 40 and 
                long_sentence_ratio < 0.3 and 
                short_sentence_ratio < 0.4)
    
    def _calculate_readability_score(self, content: str) -> float:
        """Calculate readability score (0-1)"""
        if not content:
            return 0.0
        
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        if not sentences:
            return 0.0
        
        # Calculate various readability metrics
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        
        # Normalize sentence length (optimal range: 10-20 words)
        sentence_length_score = 1.0 - abs(avg_sentence_length - 15) / 15
        sentence_length_score = max(0.0, min(1.0, sentence_length_score))
        
        # Check for variety in sentence structure
        unique_sentence_lengths = len(set(len(s.split()) for s in sentences))
        variety_score = min(1.0, unique_sentence_lengths / len(sentences))
        
        # Combine scores
        return (sentence_length_score * 0.7 + variety_score * 0.3)
    
    def _check_format_consistency(self, content: str) -> bool:
        """Check if document format is consistent"""
        if not content:
            return False
        
        lines = content.split('\n')
        if len(lines) < 3:
            return True  # Too short to check consistency
        
        # Check for consistent indentation patterns
        indentation_patterns = []
        for line in lines:
            if line.strip():
                indent_level = len(line) - len(line.lstrip())
                indentation_patterns.append(indent_level)
        
        if len(indentation_patterns) < 2:
            return True
        
        # Check if indentation is mostly consistent
        avg_indent = sum(indentation_patterns) / len(indentation_patterns)
        indent_variance = sum((i - avg_indent) ** 2 for i in indentation_patterns) / len(indentation_patterns)
        
        return indent_variance < 2.0  # Low variance indicates consistency
    
    def _calculate_format_consistency_score(self, content: str) -> float:
        """Calculate format consistency score (0-1)"""
        if not content:
            return 0.0
        
        lines = content.split('\n')
        if len(lines) < 3:
            return 1.0
        
        # Check indentation consistency
        indentation_patterns = []
        for line in lines:
            if line.strip():
                indent_level = len(line) - len(line.lstrip())
                indentation_patterns.append(indent_level)
        
        if len(indentation_patterns) < 2:
            return 1.0
        
        # Calculate consistency score
        avg_indent = sum(indentation_patterns) / len(indentation_patterns)
        if avg_indent == 0:
            return 1.0  # No indentation is consistent
        
        indent_variance = sum((i - avg_indent) ** 2 for i in indentation_patterns) / len(indentation_patterns)
        consistency_score = max(0.0, 1.0 - (indent_variance / 10.0))
        
        return consistency_score
    
    def _check_content_quality(self, content: str) -> bool:
        """Check basic content quality indicators"""
        if not content:
            return False
        
        # Check for common quality issues
        issues = 0
        
        # Check for excessive whitespace
        if content.count('  ') > len(content) * 0.1:  # More than 10% double spaces
            issues += 1
        
        # Check for excessive line breaks
        if content.count('\n\n\n') > len(content.split('\n')) * 0.1:  # More than 10% triple line breaks
            issues += 1
        
        # Check for mixed case issues (all caps or all lowercase)
        words = content.split()
        if words:
            all_caps_words = sum(1 for word in words if word.isupper() and len(word) > 2)
            all_lower_words = sum(1 for word in words if word.islower() and len(word) > 2)
            
            if all_caps_words > len(words) * 0.2:  # More than 20% all caps
                issues += 1
            if all_lower_words > len(words) * 0.9:  # More than 90% all lowercase
                issues += 1
        
        return issues <= 1  # Pass if 1 or fewer issues
    
    def _calculate_content_quality_score(self, content: str) -> float:
        """Calculate content quality score (0-1)"""
        if not content:
            return 0.0
        
        score = 1.0
        issues = 0
        
        # Check for excessive whitespace
        double_space_ratio = content.count('  ') / len(content)
        if double_space_ratio > 0.05:
            score -= double_space_ratio * 2
            issues += 1
        
        # Check for excessive line breaks
        triple_line_break_ratio = content.count('\n\n\n') / len(content.split('\n'))
        if triple_line_break_ratio > 0.05:
            score -= triple_line_break_ratio * 2
            issues += 1
        
        # Check for mixed case issues
        words = content.split()
        if words:
            all_caps_ratio = sum(1 for word in words if word.isupper() and len(word) > 2) / len(words)
            all_lower_ratio = sum(1 for word in words if word.islower() and len(word) > 2) / len(words)
            
            if all_caps_ratio > 0.15:
                score -= all_caps_ratio
                issues += 1
            if all_lower_ratio > 0.8:
                score -= all_lower_ratio * 0.5
                issues += 1
        
        # Penalize for multiple issues
        if issues > 1:
            score *= (0.9 ** (issues - 1))
        
        return max(0.0, min(1.0, score))
    
    def _calculate_simple_quality_score(self, quality_checks: Dict[str, Any]) -> float:
        """Calculate overall quality score from basic checks"""
        
        if not quality_checks:
            return 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        for check_name, check_data in quality_checks.items():
            weight = check_data.get('weight', 0.0)
            total_weight += weight
            
            if check_data.get('passed', False):
                # If check passed, use the calculated value or default to 1.0
                value = check_data.get('value', 1.0)
                if isinstance(value, (int, float)):
                    # Normalize values to 0-1 range
                    if check_name == 'content_length':
                        # Normalize word count (50+ words = 1.0, 0 words = 0.0)
                        score = min(1.0, value / 50.0)
                    else:
                        score = min(1.0, value)
                else:
                    score = 1.0
            else:
                # If check failed, use the calculated value or default to 0.0
                value = check_data.get('value', 0.0)
                if isinstance(value, (int, float)):
                    # Normalize values to 0-1 range
                    if check_name == 'content_length':
                        # Normalize word count (50+ words = 1.0, 0 words = 0.0)
                        score = min(1.0, value / 50.0)
                    else:
                        score = max(0.0, value)
                else:
                    score = 0.0
            
            total_score += score * weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _generate_basic_report(self, document_info: Dict[str, Any], quality_checks: Dict[str, Any], quality_score: float) -> Dict[str, Any]:
        """Generate basic quality report"""
        
        # Count passed/failed checks
        passed_checks = sum(1 for check in quality_checks.values() if check.get('passed', False))
        total_checks = len(quality_checks)
        
        # Identify issues
        issues = []
        recommendations = []
        
        for check_name, check_data in quality_checks.items():
            if not check_data.get('passed', False):
                issues.append(check_data.get('check', check_name))
                
                # Generate recommendations based on check type
                if check_name == 'content_length':
                    recommendations.append("Add more content to meet minimum length requirements")
                elif check_name == 'structure_presence':
                    recommendations.append("Add a title or section headers to improve document structure")
                elif check_name == 'readability':
                    recommendations.append("Review sentence lengths and structure for better readability")
                elif check_name == 'format_consistency':
                    recommendations.append("Ensure consistent formatting throughout the document")
                elif check_name == 'content_quality':
                    recommendations.append("Review content for formatting and style consistency")
        
        # Determine overall status
        if quality_score >= 0.8:
            status = "excellent"
        elif quality_score >= 0.6:
            status = "good"
        elif quality_score >= 0.4:
            status = "fair"
        else:
            status = "needs_improvement"
        
        return {
            'quality_score': quality_score,
            'status': status,
            'passed_checks': passed_checks,
            'total_checks': total_checks,
            'pass_rate': passed_checks / total_checks if total_checks > 0 else 0.0,
            'issues': issues,
            'recommendations': recommendations,
            'document_info': document_info,
            'quality_checks': quality_checks,
            'generated_at': datetime.now().isoformat(),
            'orchestrator_type': 'simple'
        }
    
    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get quality metrics for the session"""
        return {
            'session_id': self.session_id,
            'orchestrator_type': 'simple',
            'quality_metrics': self.quality_metrics,
            'session_start': getattr(self, 'session_start', datetime.now().isoformat())
        }
    
    def save_report(self, report: Dict[str, Any], output_path: str) -> bool:
        """Save quality report to file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"Simple QA report saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving simple QA report to {output_path}: {e}")
            return False
