#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EnhancedQAOrchestrator Demo - MCP Enhanced Level Demonstration

This script demonstrates the EnhancedQAOrchestrator capabilities as the 
intermediate level in the MCP (Model Context Protocol) progressive complexity architecture.

Features Demonstrated:
- Enhanced Document Processing Pipeline
- Document Inspection (TOC, Bibliography, Links, Structure)
- Caption Analysis and Quality Assessment
- Database Integration and Quality Prediction
- Enhanced Quality Score Calculation
- Progressive Complexity Inheritance
- MCP Enhanced Level Compliance
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


class MockDocumentInspectorCoordinator:
    """Mock document inspector coordinator for standalone demo"""
    
    def inspect_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Mock document inspection"""
        content = document.get('content', '')
        
        # TOC Analysis
        toc_analysis = self._analyze_toc(content)
        
        # Bibliography Analysis
        bibliography_analysis = self._analyze_bibliography(content)
        
        # Link Analysis
        link_analysis = self._analyze_links(content)
        
        # Structure Analysis
        structure_analysis = self._analyze_structure(content)
        
        # Calculate overall score
        overall_score = (toc_analysis['score'] + bibliography_analysis['score'] + 
                        link_analysis['score'] + structure_analysis['score']) / 4.0
        
        return {
            'toc_analysis': toc_analysis,
            'bibliography_analysis': bibliography_analysis,
            'link_analysis': link_analysis,
            'structure_analysis': structure_analysis,
            'overall_score': overall_score,
            'issues': self._identify_issues(toc_analysis, bibliography_analysis, link_analysis, structure_analysis),
            'recommendations': self._generate_recommendations(toc_analysis, bibliography_analysis, link_analysis, structure_analysis)
        }
    
    def _analyze_toc(self, content: str) -> Dict[str, Any]:
        """Analyze table of contents"""
        headings = re.findall(r'^#{1,6}\s+(.+)$', content, re.MULTILINE)
        
        if not headings:
            return {'score': 0.0, 'headings': [], 'depth': 0, 'issues': ['No headings found']}
        
        # Calculate TOC score based on structure
        depth = max(len(re.match(r'^#+', line).group()) for line in re.findall(r'^#{1,6}\s+', content, re.MULTILINE))
        score = min(1.0, len(headings) / 10.0 + depth / 6.0)
        
        return {
            'score': score,
            'headings': headings,
            'depth': depth,
            'issues': [] if score > 0.5 else ['Insufficient heading structure']
        }
    
    def _analyze_bibliography(self, content: str) -> Dict[str, Any]:
        """Analyze bibliography/references"""
        # Look for common bibliography patterns
        bib_patterns = [
            r'\d+\.\s+[A-Z][a-z]+,\s+[A-Z]\.\s*\([0-9]{4}\)',  # Author, A. (Year)
            r'\[[0-9]+\]',  # [1], [2], etc.
            r'References?',  # References section
            r'Bibliography',  # Bibliography section
        ]
        
        matches = sum(len(re.findall(pattern, content, re.IGNORECASE)) for pattern in bib_patterns)
        score = min(1.0, matches / 3.0)
        
        return {
            'score': score,
            'references_found': matches,
            'issues': [] if score > 0.3 else ['No bibliography or references found']
        }
    
    def _analyze_links(self, content: str) -> Dict[str, Any]:
        """Analyze links in document"""
        # Find URLs and links
        urls = re.findall(r'https?://[^\s]+', content)
        markdown_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
        
        total_links = len(urls) + len(markdown_links)
        score = min(1.0, total_links / 5.0)
        
        return {
            'score': score,
            'urls': urls,
            'markdown_links': markdown_links,
            'total_links': total_links,
            'issues': [] if score > 0.2 else ['No links found']
        }
    
    def _analyze_structure(self, content: str) -> Dict[str, Any]:
        """Analyze document structure"""
        lines = content.split('\n')
        
        # Count structural elements
        headings = len(re.findall(r'^#{1,6}\s+', content, re.MULTILINE))
        lists = len(re.findall(r'^[\-\*]\s+', content, re.MULTILINE))
        paragraphs = len([line for line in lines if line.strip() and not line.startswith('#') and not line.startswith('-') and not line.startswith('*')])
        
        # Calculate structure score
        structure_score = min(1.0, (headings + lists + paragraphs) / 20.0)
        
        return {
            'score': structure_score,
            'headings': headings,
            'lists': lists,
            'paragraphs': paragraphs,
            'issues': [] if structure_score > 0.3 else ['Poor document structure']
        }
    
    def _identify_issues(self, toc, bib, links, structure) -> List[str]:
        """Identify overall issues"""
        issues = []
        if toc['score'] < 0.5:
            issues.append('Insufficient table of contents')
        if bib['score'] < 0.3:
            issues.append('Missing bibliography or references')
        if links['score'] < 0.2:
            issues.append('No external links or references')
        if structure['score'] < 0.3:
            issues.append('Poor document structure')
        return issues
    
    def _generate_recommendations(self, toc, bib, links, structure) -> List[str]:
        """Generate recommendations"""
        recommendations = []
        if toc['score'] < 0.5:
            recommendations.append('Add more headings to improve document structure')
        if bib['score'] < 0.3:
            recommendations.append('Include a bibliography or references section')
        if links['score'] < 0.2:
            recommendations.append('Add relevant external links and references')
        if structure['score'] < 0.3:
            recommendations.append('Improve overall document structure and organization')
        return recommendations


class MockCaptionInspectorCoordinator:
    """Mock caption inspector coordinator for standalone demo"""
    
    def analyze_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Mock caption analysis"""
        content = document.get('content', '')
        
        # Find captions
        captions = self._find_captions(content)
        
        # Analyze caption quality
        caption_quality = self._analyze_caption_quality(captions)
        
        return {
            'total_captions': len(captions),
            'captions_with_numbers': len([c for c in captions if re.search(r'\d+', c)]),
            'captions_with_issues': len([c for c in captions if len(c) < 10 or len(c) > 100]),
            'quality_score': caption_quality['score'],
            'caption_details': captions,
            'recommendations': caption_quality['recommendations']
        }
    
    def _find_captions(self, content: str) -> List[str]:
        """Find captions in content"""
        # Look for common caption patterns
        caption_patterns = [
            r'Figure\s+\d+[:\s]+([^.\n]+)',
            r'Table\s+\d+[:\s]+([^.\n]+)',
            r'**([^**]+)**\s*\([Ff]igure\s+\d+\)',
            r'**([^**]+)**\s*\([Tt]able\s+\d+\)',
        ]
        
        captions = []
        for pattern in caption_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            captions.extend(matches)
        
        return captions
    
    def _analyze_caption_quality(self, captions: List[str]) -> Dict[str, Any]:
        """Analyze caption quality"""
        if not captions:
            return {
                'score': 0.0,
                'recommendations': ['Add captions to figures and tables']
            }
        
        # Calculate quality score
        good_captions = 0
        for caption in captions:
            if 10 <= len(caption) <= 100 and re.search(r'\d+', caption):
                good_captions += 1
        
        score = good_captions / len(captions) if captions else 0.0
        
        recommendations = []
        if score < 0.7:
            recommendations.append('Improve caption quality and numbering')
        if len(captions) < 2:
            recommendations.append('Add more captions to figures and tables')
        
        return {
            'score': score,
            'recommendations': recommendations
        }


class MockDatabaseQualityAnalyzer:
    """Mock database quality analyzer for standalone demo"""
    
    def predict_quality(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Mock quality prediction"""
        content = document.get('content', '')
        metadata = document.get('metadata', {})
        
        # Extract features
        features = self.extract_document_features(document)
        
        # Mock prediction based on features
        predicted_score = self._calculate_predicted_score(features)
        likely_issues = self._identify_likely_issues(features)
        confidence = self._calculate_confidence(features)
        
        return {
            'predicted_score': predicted_score,
            'likely_issues': likely_issues,
            'confidence': confidence,
            'pattern_count': 5,  # Mock pattern count
            'doc_features': features
        }
    
    def extract_document_features(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Extract document features"""
        content = document.get('content', '')
        metadata = document.get('metadata', {})
        
        return {
            'type': metadata.get('type', 'unknown'),
            'length': len(content),
            'has_tables': self.detect_tables(content),
            'has_numbers': self.detect_numerical_content(content),
            'complexity_score': self.calculate_complexity(content),
            'word_count': len(content.split()),
            'sentence_count': len([s for s in content.split('.') if s.strip()])
        }
    
    def detect_tables(self, content: str) -> bool:
        """Detect tables in content"""
        table_patterns = [
            r'\|\s*[^|]+\s*\|',  # Markdown tables
            r'<table',           # HTML tables
            r'\t+',              # Tab-separated
            r',\s*,',            # CSV-like
        ]
        return any(re.search(pattern, content) for pattern in table_patterns)
    
    def detect_numerical_content(self, content: str) -> bool:
        """Detect numerical content"""
        number_patterns = [
            r'\d+\.\d+',         # Decimals
            r'\d+%',             # Percentages
            r'\$\d+',            # Currency
            r'\d{4}-\d{2}-\d{2}' # Dates
        ]
        return any(re.search(pattern, content) for pattern in number_patterns)
    
    def calculate_complexity(self, content: str) -> float:
        """Calculate complexity score"""
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        if not sentences:
            return 0.0
        
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        unique_words = len(set(content.lower().split()))
        total_words = len(content.split())
        
        if total_words == 0:
            return 0.0
        
        complexity = min(1.0, (avg_sentence_length / 20.0 + unique_words / total_words) / 2.0)
        return complexity
    
    def _calculate_predicted_score(self, features: Dict[str, Any]) -> float:
        """Calculate predicted quality score"""
        score = 0.5  # Base score
        
        # Adjust based on features
        if features.get('length', 0) > 500:
            score += 0.1
        if features.get('has_tables'):
            score += 0.1
        if features.get('has_numbers'):
            score += 0.1
        if features.get('complexity_score', 0) > 0.5:
            score += 0.1
        if features.get('word_count', 0) > 100:
            score += 0.1
        
        return min(1.0, score)
    
    def _identify_likely_issues(self, features: Dict[str, Any]) -> List[str]:
        """Identify likely issues"""
        issues = []
        if features.get('length', 0) < 200:
            issues.append('Document too short')
        if not features.get('has_tables'):
            issues.append('No tables or structured data')
        if features.get('complexity_score', 0) < 0.3:
            issues.append('Document too simple')
        return issues
    
    def _calculate_confidence(self, features: Dict[str, Any]) -> float:
        """Calculate prediction confidence"""
        confidence = 0.5  # Base confidence
        
        # More features = higher confidence
        feature_count = sum([
            bool(features.get('type')),
            bool(features.get('length', 0) > 100),
            bool(features.get('word_count', 0) > 10),
            features.get('has_tables'),
            features.get('has_numbers')
        ])
        
        confidence += feature_count * 0.1
        return min(1.0, confidence)


class StandaloneEnhancedQAOrchestrator:
    """Standalone version of EnhancedQAOrchestrator for demonstration"""
    
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.quality_metrics = {}
        
        # Mock coordinators
        self.document_inspector = MockDocumentInspectorCoordinator()
        self.caption_inspector = MockCaptionInspectorCoordinator()
        self.quality_analyzer = MockDatabaseQualityAnalyzer()
        
        # Mock database availability
        self.db_available = True
        
        print(f"EnhancedQAOrchestrator initialized with session ID: {self.session_id}")
    
    def process_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced document processing with inspections and database integration"""
        
        start_time = datetime.now()
        
        try:
            # Handle None document
            if document is None:
                raise ValueError("Document cannot be None")
            
            # Get Simple QA results first (mock)
            simple_results = self._get_simple_results(document)
            
            # Perform enhanced inspections
            document_inspection = self._perform_document_inspection(document)
            caption_analysis = self._perform_caption_analysis(document)
            
            # Get quality prediction from database
            quality_prediction = self._predict_quality(document)
            
            # Calculate enhanced quality score
            enhanced_quality_score = self._calculate_enhanced_quality_score(
                simple_results['quality_score'],
                document_inspection,
                caption_analysis,
                quality_prediction
            )
            
            # Generate enhanced report
            enhanced_report = self._generate_enhanced_report(
                simple_results,
                document_inspection,
                caption_analysis,
                quality_prediction
            )
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                'document': document,
                'simple_results': simple_results,
                'document_inspection': document_inspection,
                'caption_analysis': caption_analysis,
                'quality_prediction': quality_prediction,
                'enhanced_quality_score': enhanced_quality_score,
                'enhanced_report': enhanced_report,
                'processing_time_ms': processing_time,
                'session_id': self.session_id,
                'orchestrator_type': 'enhanced',
                'db_available': self.db_available,
                'status': 'success'
            }
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                'document': document,
                'simple_results': {},
                'document_inspection': {},
                'caption_analysis': {},
                'quality_prediction': {},
                'enhanced_quality_score': 0.0,
                'enhanced_report': {'error': str(e)},
                'processing_time_ms': processing_time,
                'session_id': self.session_id,
                'orchestrator_type': 'enhanced',
                'db_available': self.db_available,
                'status': 'error',
                'error': str(e)
            }
    
    def _get_simple_results(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Mock simple QA results"""
        content = document.get('content', '')
        
        # Basic quality score calculation
        word_count = len(content.split())
        has_title = bool(re.search(r'^#+\s+', content, re.MULTILINE))
        has_sections = bool(re.search(r'^##+\s+', content, re.MULTILINE))
        
        quality_score = 0.5  # Base score
        if word_count > 50:
            quality_score += 0.2
        if has_title:
            quality_score += 0.15
        if has_sections:
            quality_score += 0.15
        
        return {
            'quality_score': min(1.0, quality_score),
            'status': 'success',
            'session_id': self.session_id,
            'orchestrator_type': 'simple'
        }
    
    def _perform_document_inspection(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive document inspection"""
        try:
            inspection_result = self.document_inspector.inspect_document(document)
            
            return {
                'toc_analysis': inspection_result.get('toc_analysis', {}),
                'bibliography_analysis': inspection_result.get('bibliography_analysis', {}),
                'link_analysis': inspection_result.get('link_analysis', {}),
                'structure_analysis': inspection_result.get('structure_analysis', {}),
                'inspection_score': inspection_result.get('overall_score', 0.0),
                'issues_found': inspection_result.get('issues', []),
                'recommendations': inspection_result.get('recommendations', [])
            }
            
        except Exception as e:
            return {
                'toc_analysis': {},
                'bibliography_analysis': {},
                'link_analysis': {},
                'structure_analysis': {},
                'inspection_score': 0.0,
                'issues_found': [f"Inspection error: {str(e)}"],
                'recommendations': ['Check document format and accessibility']
            }
    
    def _perform_caption_analysis(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Perform caption analysis"""
        try:
            caption_result = self.caption_inspector.analyze_document(document)
            
            return {
                'total_captions': caption_result.get('total_captions', 0),
                'captions_with_numbers': caption_result.get('captions_with_numbers', 0),
                'captions_with_issues': caption_result.get('captions_with_issues', 0),
                'caption_quality_score': caption_result.get('quality_score', 0.0),
                'caption_details': caption_result.get('caption_details', []),
                'caption_recommendations': caption_result.get('recommendations', [])
            }
            
        except Exception as e:
            return {
                'total_captions': 0,
                'captions_with_numbers': 0,
                'captions_with_issues': 0,
                'caption_quality_score': 0.0,
                'caption_details': [],
                'caption_recommendations': [f"Caption analysis error: {str(e)}"]
            }
    
    def _predict_quality(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Predict quality using database patterns"""
        try:
            return self.quality_analyzer.predict_quality(document)
        except Exception as e:
            return {
                'predicted_score': 0.5,
                'likely_issues': [f"Prediction error: {str(e)}"],
                'confidence': 0.0,
                'pattern_count': 0,
                'doc_features': {}
            }
    
    def _calculate_enhanced_quality_score(self, simple_score: float, document_inspection: Dict[str, Any], 
                                        caption_analysis: Dict[str, Any], quality_prediction: Dict[str, Any]) -> float:
        """Calculate enhanced quality score"""
        
        # Base score from simple QA
        enhanced_score = simple_score * 0.4
        
        # Add document inspection score
        inspection_score = document_inspection.get('inspection_score', 0.0)
        enhanced_score += inspection_score * 0.3
        
        # Add caption analysis score
        caption_score = caption_analysis.get('caption_quality_score', 0.0)
        enhanced_score += caption_score * 0.2
        
        # Add database prediction score
        predicted_score = quality_prediction.get('predicted_score', 0.5)
        confidence = quality_prediction.get('confidence', 0.0)
        enhanced_score += predicted_score * confidence * 0.1
        
        return min(1.0, enhanced_score)
    
    def _generate_enhanced_report(self, simple_results: Dict[str, Any], document_inspection: Dict[str, Any],
                                caption_analysis: Dict[str, Any], quality_prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Generate enhanced quality report"""
        
        enhanced_score = self._calculate_enhanced_quality_score(
            simple_results.get('quality_score', 0.0),
            document_inspection,
            caption_analysis,
            quality_prediction
        )
        
        # Determine status
        if enhanced_score >= 0.8:
            status = "excellent"
        elif enhanced_score >= 0.6:
            status = "good"
        elif enhanced_score >= 0.4:
            status = "fair"
        else:
            status = "needs_improvement"
        
        # Collect all issues and recommendations
        all_issues = []
        all_recommendations = []
        
        # From document inspection
        all_issues.extend(document_inspection.get('issues_found', []))
        all_recommendations.extend(document_inspection.get('recommendations', []))
        
        # From caption analysis
        all_recommendations.extend(caption_analysis.get('caption_recommendations', []))
        
        # From quality prediction
        all_issues.extend(quality_prediction.get('likely_issues', []))
        
        return {
            'enhanced_quality_score': enhanced_score,
            'status': status,
            'simple_qa_score': simple_results.get('quality_score', 0.0),
            'inspection_score': document_inspection.get('inspection_score', 0.0),
            'caption_score': caption_analysis.get('caption_quality_score', 0.0),
            'prediction_score': quality_prediction.get('predicted_score', 0.0),
            'prediction_confidence': quality_prediction.get('confidence', 0.0),
            'total_issues': len(all_issues),
            'total_recommendations': len(all_recommendations),
            'issues': all_issues,
            'recommendations': all_recommendations,
            'generated_at': datetime.now().isoformat(),
            'orchestrator_type': 'enhanced'
        }


class EnhancedQADemo:
    """Enhanced QA Orchestrator demo"""
    
    def __init__(self):
        self.orchestrator = StandaloneEnhancedQAOrchestrator()
        self.demo_results = []
        
    def run_comprehensive_demo(self):
        """Run comprehensive demonstration of EnhancedQAOrchestrator features"""
        print("EnhancedQAOrchestrator - MCP Enhanced Level Demo")
        print("=" * 60)
        print(f"Session ID: {self.orchestrator.session_id}")
        print(f"Demo Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Test documents with varying complexity levels
        test_documents = [
            {
                'name': 'Enhanced Technical Document',
                'content': """
# Advanced Technical Documentation

This document demonstrates enhanced features with comprehensive structure and analysis.

## Table of Contents

1. Introduction
2. Technical Architecture
3. Implementation Details
4. Performance Analysis
5. Conclusion

## Introduction

The enhanced document includes proper formatting, structure, and organization with comprehensive analysis capabilities.

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

### Figure 1: System Architecture Diagram
This figure shows the complete system architecture with proper caption and numbering.

### Table 1: Component Analysis
| Component | Complexity | Performance | Status |
|-----------|------------|-------------|--------|
| Data Layer | High | Excellent | ✅ |
| Logic Layer | Medium | Good | ✅ |
| UI Layer | Low | Excellent | ✅ |

## Performance Analysis

The system demonstrates excellent performance characteristics:

- **Response Time**: Average 50ms
- **Throughput**: 1500 requests/second
- **Error Rate**: 0.05%
- **Availability**: 99.9%

## Conclusion

This document demonstrates the enhanced quality standards expected in professional technical documentation.

## References

1. Smith, J. (2023). Advanced Systems Architecture. Technical Journal, 15(3), 45-67.
2. Doe, A. (2023). Performance Optimization Techniques. Systems Engineering, 8(2), 23-41.
3. Johnson, B. (2023). Quality Assurance in Technical Documentation. Documentation Quarterly, 12(1), 78-92.

For more information, visit: https://example.com/technical-docs
                """,
                'metadata': {
                    'title': 'Advanced Technical Documentation',
                    'author': 'Technical Writer',
                    'type': 'technical_documentation',
                    'created_date': '2024-01-15',
                    'modified_date': '2024-01-20'
                },
                'expected_score': 0.85
            },
            {
                'name': 'Good Enhanced Document',
                'content': """
# Sample Enhanced Document

This is a well-structured document with enhanced features and proper formatting.

## Introduction

This section provides a comprehensive overview of the enhanced capabilities.

## Technical Details

### Figure 1: Sample Chart
This figure demonstrates proper caption formatting with numbering.

### Table 1: Sample Data
| Category | Value | Status |
|----------|-------|--------|
| Feature A | 85% | Good |
| Feature B | 92% | Excellent |

## Analysis

The document includes:
- Proper structure and formatting
- Tables and figures with captions
- Clear organization
- Technical content

## References

1. Author, A. (2023). Sample Reference.
2. Writer, B. (2023). Another Reference.

Visit: https://example.com for more details.
                """,
                'metadata': {
                    'title': 'Sample Enhanced Document',
                    'author': 'Test Author',
                    'type': 'documentation'
                },
                'expected_score': 0.75
            },
            {
                'name': 'Basic Enhanced Document',
                'content': """
# Basic Enhanced Document

This document has some enhanced features but could be improved.

## Main Section

This section provides basic information with some structure.

### Subsection

- Item 1: Basic list item
- Item 2: Another item
- Item 3: Third item

The document includes basic formatting but lacks comprehensive features.
                """,
                'metadata': {
                    'title': 'Basic Enhanced Document',
                    'author': 'Author',
                    'type': 'basic'
                },
                'expected_score': 0.60
            },
            {
                'name': 'Poor Enhanced Document',
                'content': """
this document has poor enhanced features and lacks proper structure it has no tables figures or proper formatting making it difficult to read and understand the content lacks technical depth and proper organization which makes it unsuitable for enhanced analysis
                """,
                'metadata': {
                    'title': 'Poor Enhanced Document',
                    'author': 'Author',
                    'type': 'basic'
                },
                'expected_score': 0.30
            }
        ]
        
        print("Processing Enhanced Test Documents")
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
        
        print("\nEnhanced QA Demo completed successfully!")
        print(f"Results saved to: {Path(__file__).parent / 'enhanced_demo_results.json'}")
    
    def _display_processing_results(self, result, processing_time, expected_score):
        """Display processing results for a document"""
        if result['status'] == 'success':
            enhanced_report = result['enhanced_report']
            enhanced_score = enhanced_report['enhanced_quality_score']
            
            print(f"Enhanced Quality Score: {enhanced_score:.3f} (Expected: {expected_score:.3f})")
            print(f"Status: {enhanced_report['status']}")
            print(f"Processing Time: {processing_time:.2f}ms")
            
            # Show component scores
            print(f"Simple QA Score: {enhanced_report['simple_qa_score']:.3f}")
            print(f"Inspection Score: {enhanced_report['inspection_score']:.3f}")
            print(f"Caption Score: {enhanced_report['caption_score']:.3f}")
            print(f"Prediction Score: {enhanced_report['prediction_score']:.3f}")
            print(f"Prediction Confidence: {enhanced_report['prediction_confidence']:.3f}")
            
            # Show document inspection details
            doc_inspection = result['document_inspection']
            print(f"TOC Analysis Score: {doc_inspection['toc_analysis'].get('score', 0.0):.3f}")
            print(f"Bibliography Score: {doc_inspection['bibliography_analysis'].get('score', 0.0):.3f}")
            print(f"Link Analysis Score: {doc_inspection['link_analysis'].get('score', 0.0):.3f}")
            print(f"Structure Analysis Score: {doc_inspection['structure_analysis'].get('score', 0.0):.3f}")
            
            # Show caption analysis details
            caption_analysis = result['caption_analysis']
            print(f"Total Captions: {caption_analysis['total_captions']}")
            print(f"Captions with Numbers: {caption_analysis['captions_with_numbers']}")
            print(f"Captions with Issues: {caption_analysis['captions_with_issues']}")
            
            # Show issues and recommendations
            if enhanced_report['issues']:
                print(f"Issues: {len(enhanced_report['issues'])}")
                for issue in enhanced_report['issues'][:3]:  # Show first 3 issues
                    print(f"   - {issue}")
            
            if enhanced_report['recommendations']:
                print(f"Recommendations: {len(enhanced_report['recommendations'])}")
                for rec in enhanced_report['recommendations'][:3]:  # Show first 3 recommendations
                    print(f"   - {rec}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
    
    def _generate_comprehensive_report(self):
        """Generate comprehensive demo report"""
        print("\nEnhanced QA Comprehensive Demo Report")
        print("=" * 50)
        
        # Calculate statistics
        successful_results = [r for r in self.demo_results if r['result']['status'] == 'success']
        failed_results = [r for r in self.demo_results if r['result']['status'] == 'error']
        
        print(f"Total Documents Processed: {len(self.demo_results)}")
        print(f"Successful: {len(successful_results)}")
        print(f"Failed: {len(failed_results)}")
        print(f"Success Rate: {len(successful_results)/len(self.demo_results):.1%}")
        
        if successful_results:
            # Enhanced quality score analysis
            scores = [r['result']['enhanced_report']['enhanced_quality_score'] for r in successful_results]
            avg_score = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)
            
            print(f"\nEnhanced Quality Score Analysis:")
            print(f"   Average Score: {avg_score:.3f}")
            print(f"   Minimum Score: {min_score:.3f}")
            print(f"   Maximum Score: {max_score:.3f}")
            
            # Component score analysis
            simple_scores = [r['result']['enhanced_report']['simple_qa_score'] for r in successful_results]
            inspection_scores = [r['result']['enhanced_report']['inspection_score'] for r in successful_results]
            caption_scores = [r['result']['enhanced_report']['caption_score'] for r in successful_results]
            prediction_scores = [r['result']['enhanced_report']['prediction_score'] for r in successful_results]
            
            print(f"\nComponent Score Analysis:")
            print(f"   Average Simple QA Score: {sum(simple_scores)/len(simple_scores):.3f}")
            print(f"   Average Inspection Score: {sum(inspection_scores)/len(inspection_scores):.3f}")
            print(f"   Average Caption Score: {sum(caption_scores)/len(caption_scores):.3f}")
            print(f"   Average Prediction Score: {sum(prediction_scores)/len(prediction_scores):.3f}")
            
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
                actual_score = r['result']['enhanced_report']['enhanced_quality_score']
                expected_score = r['expected_score']
                accuracy = 1.0 - abs(actual_score - expected_score)
                accuracy_scores.append(accuracy)
            
            avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
            print(f"\nScore Accuracy Analysis:")
            print(f"   Average Accuracy: {avg_accuracy:.1%}")
        
        # MCP Compliance Check
        print(f"\nMCP Enhanced Level Compliance Check:")
        mcp_fields = ['session_id', 'orchestrator_type', 'status', 'processing_time_ms', 'db_available']
        compliance_checks = []
        
        for r in self.demo_results:
            result = r['result']
            checks = [field in result for field in mcp_fields]
            compliance_checks.extend(checks)
        
        compliance_rate = sum(compliance_checks) / len(compliance_checks)
        print(f"   MCP Field Compliance: {compliance_rate:.1%}")
        print(f"   Session ID Consistency: {'Yes' if self._check_session_consistency() else 'No'}")
        print(f"   Database Integration: {'Yes' if all(r['result']['db_available'] for r in successful_results) else 'No'}")
    
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
                'orchestrator_type': 'enhanced',
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
        
        output_file = Path(__file__).parent / 'enhanced_demo_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)


def main():
    """Main function to run the demo"""
    demo = EnhancedQADemo()
    demo.run_comprehensive_demo()


if __name__ == '__main__':
    main()
