#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone SimpleQAOrchestrator Demo

This script demonstrates the core SimpleQAOrchestrator functionality
without complex dependencies. It shows the MCP foundation capabilities
in a self-contained manner.
"""

import sys
import os
import json
import time
import uuid
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


class StandaloneSimpleQAOrchestrator:
    """Standalone version of SimpleQAOrchestrator for demonstration"""
    
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.quality_metrics = {}
        print(f"SimpleQAOrchestrator initialized with session ID: {self.session_id}")
    
    def process_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Basic document processing with simple quality checks"""
        
        start_time = datetime.now()
        
        try:
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


class StandaloneSimpleQADemo:
    """Standalone demo of SimpleQAOrchestrator MCP capabilities"""
    
    def __init__(self):
        self.orchestrator = StandaloneSimpleQAOrchestrator()
        self.demo_results = []
        
    def run_comprehensive_demo(self):
        """Run comprehensive demonstration of all SimpleQAOrchestrator features"""
        print("SimpleQAOrchestrator - MCP Foundation Demo")
        print("=" * 60)
        print(f"Session ID: {self.orchestrator.session_id}")
        print(f"Demo Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Test documents with varying quality levels
        test_documents = [
            {
                'name': 'Excellent Document',
                'content': """
# Comprehensive Technical Documentation

This document demonstrates excellent quality with proper structure, formatting, and content organization.

## Executive Summary

The document provides a comprehensive overview of technical concepts with clear explanations and proper formatting.

## Technical Architecture

### System Components

The system consists of several key components:

1. **Data Processing Layer**: Handles input validation and transformation
2. **Business Logic Layer**: Implements core business rules and algorithms
3. **Presentation Layer**: Manages user interface and interaction

### Performance Metrics

| Metric | Value | Status |
|--------|-------|--------|
| Response Time | < 100ms | Excellent |
| Throughput | 1000 req/s | Excellent |
| Error Rate | < 0.1% | Excellent |

## Implementation Details

The implementation follows best practices for software development:

- **Modular Design**: Components are loosely coupled and highly cohesive
- **Error Handling**: Comprehensive error handling and recovery mechanisms
- **Documentation**: Clear and comprehensive documentation
- **Testing**: Extensive unit and integration testing

## Conclusion

This document demonstrates the quality standards expected in professional technical documentation.
                """,
                'metadata': {
                    'title': 'Comprehensive Technical Documentation',
                    'author': 'Technical Writer',
                    'type': 'technical_documentation',
                    'created_date': '2024-01-15',
                    'modified_date': '2024-01-20'
                },
                'expected_score': 0.85
            },
            {
                'name': 'Good Document',
                'content': """
# Sample Document

This is a well-structured document with proper formatting. It contains multiple sentences that are easy to read and understand.

## Section 1

This section has good content with proper sentence structure. The sentences are neither too long nor too short, making them easy to read.

- Item 1: This is a list item
- Item 2: Another list item
- Item 3: Third list item

## Section 2

The document maintains consistent formatting throughout. Each section follows the same pattern and structure.

### Subsection

This subsection provides additional details with proper formatting and structure.
                """,
                'metadata': {
                    'title': 'Sample Document',
                    'author': 'Test Author',
                    'type': 'documentation'
                },
                'expected_score': 0.75
            },
            {
                'name': 'Poor Document',
                'content': """
this document has poor formatting and structure it lacks proper capitalization and punctuation making it very difficult to read the sentences are extremely long and run on without proper breaks which makes comprehension challenging for readers who need clear structure and formatting to understand the content properly there are no sections or subsections to organize the information and the overall quality is quite low this document would benefit significantly from proper editing and formatting improvements
                """,
                'metadata': {
                    'title': 'Poor Document',
                    'author': 'Author',
                    'type': 'basic'
                },
                'expected_score': 0.30
            },
            {
                'name': 'Empty Document',
                'content': '',
                'metadata': {
                    'title': 'Empty Document',
                    'author': 'Author',
                    'type': 'empty'
                },
                'expected_score': 0.10
            }
        ]
        
        print("Processing Test Documents")
        print("-" * 40)
        
        for i, doc in enumerate(test_documents, 1):
            print(f"\n{i}. Processing: {doc['name']}")
            print("-" * 30)
            
            document = {
                'content': doc['content'],
                'metadata': doc['metadata'],
                'file_path': f'/path/to/{doc["name"].lower().replace(" ", "_")}.md'
            }
            
            # Process document
            start_time = time.time()
            result = self.orchestrator.process_document(document)
            processing_time = (time.time() - start_time) * 1000
            
            # Store results
            demo_result = {
                'document_name': doc['name'],
                'result': result,
                'processing_time': processing_time,
                'expected_score': doc['expected_score']
            }
            self.demo_results.append(demo_result)
            
            # Display results
            self._display_processing_results(result, processing_time, doc['expected_score'])
        
        # Generate comprehensive report
        self._generate_comprehensive_report()
        
        # Save results
        self._save_demo_results()
        
        print("\nDemo completed successfully!")
        print(f"Results saved to: {Path(__file__).parent / 'standalone_demo_results.json'}")
    
    def _display_processing_results(self, result, processing_time, expected_score):
        """Display processing results for a document"""
        if result['status'] == 'success':
            report = result['report']
            quality_score = report['quality_score']
            
            print(f"Quality Score: {quality_score:.3f} (Expected: {expected_score:.3f})")
            print(f"Status: {report['status']}")
            print(f"Processing Time: {processing_time:.2f}ms")
            print(f"Pass Rate: {report['pass_rate']:.1%} ({report['passed_checks']}/{report['total_checks']})")
            
            # Show document info
            doc_info = result['document_info']
            print(f"Word Count: {doc_info['word_count']}")
            print(f"Character Count: {doc_info['character_count']}")
            print(f"Has Title: {'Yes' if doc_info['has_title'] else 'No'}")
            print(f"Has Sections: {'Yes' if doc_info['has_sections'] else 'No'}")
            print(f"Has Lists: {'Yes' if doc_info['has_lists'] else 'No'}")
            
            # Show issues and recommendations
            if report['issues']:
                print(f"Issues: {len(report['issues'])}")
                for issue in report['issues'][:2]:  # Show first 2 issues
                    print(f"   - {issue}")
            
            if report['recommendations']:
                print(f"Recommendations: {len(report['recommendations'])}")
                for rec in report['recommendations'][:2]:  # Show first 2 recommendations
                    print(f"   - {rec}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
    
    def _generate_comprehensive_report(self):
        """Generate comprehensive demo report"""
        print("\nComprehensive Demo Report")
        print("=" * 50)
        
        # Calculate statistics
        successful_results = [r for r in self.demo_results if r['result']['status'] == 'success']
        failed_results = [r for r in self.demo_results if r['result']['status'] == 'error']
        
        print(f"Total Documents Processed: {len(self.demo_results)}")
        print(f"Successful: {len(successful_results)}")
        print(f"Failed: {len(failed_results)}")
        print(f"Success Rate: {len(successful_results)/len(self.demo_results):.1%}")
        
        if successful_results:
            # Quality score analysis
            scores = [r['result']['report']['quality_score'] for r in successful_results]
            avg_score = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)
            
            print(f"\nQuality Score Analysis:")
            print(f"   Average Score: {avg_score:.3f}")
            print(f"   Minimum Score: {min_score:.3f}")
            print(f"   Maximum Score: {max_score:.3f}")
            
            # Processing time analysis
            times = [r['processing_time'] for r in successful_results]
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            print(f"\nProcessing Time Analysis:")
            print(f"   Average Time: {avg_time:.2f}ms")
            print(f"   Minimum Time: {min_time:.2f}ms")
            print(f"   Maximum Time: {max_time:.2f}ms")
            
            # Score accuracy analysis
            accuracy_scores = []
            for r in successful_results:
                actual_score = r['result']['report']['quality_score']
                expected_score = r['expected_score']
                accuracy = 1.0 - abs(actual_score - expected_score)
                accuracy_scores.append(accuracy)
            
            avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
            print(f"\nScore Accuracy Analysis:")
            print(f"   Average Accuracy: {avg_accuracy:.1%}")
        
        # MCP Compliance Check
        print(f"\nMCP Compliance Check:")
        mcp_fields = ['session_id', 'orchestrator_type', 'status', 'processing_time_ms']
        compliance_checks = []
        
        for r in self.demo_results:
            result = r['result']
            checks = [field in result for field in mcp_fields]
            compliance_checks.extend(checks)
        
        compliance_rate = sum(compliance_checks) / len(compliance_checks)
        print(f"   MCP Field Compliance: {compliance_rate:.1%}")
        print(f"   Session ID Consistency: {'Yes' if self._check_session_consistency() else 'No'}")
    
    def _check_session_consistency(self):
        """Check if session ID is consistent across all results"""
        session_ids = [r['result']['session_id'] for r in self.demo_results]
        return len(set(session_ids)) == 1 and session_ids[0] == self.orchestrator.session_id
    
    def _save_demo_results(self):
        """Save demo results to JSON file"""
        output_data = {
            'demo_info': {
                'session_id': self.orchestrator.session_id,
                'demo_timestamp': datetime.now().isoformat(),
                'orchestrator_type': 'simple',
                'total_documents': len(self.demo_results)
            },
            'results': []
        }
        
        for result in self.demo_results:
            output_data['results'].append({
                'document_name': result['document_name'],
                'processing_time': result['processing_time'],
                'expected_score': result['expected_score'],
                'result': result['result']
            })
        
        output_file = Path(__file__).parent / 'standalone_demo_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)


def main():
    """Main function to run the demo"""
    demo = StandaloneSimpleQADemo()
    demo.run_comprehensive_demo()


if __name__ == '__main__':
    main()
