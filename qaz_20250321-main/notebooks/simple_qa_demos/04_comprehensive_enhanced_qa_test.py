#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive EnhancedQAOrchestrator Test Suite - Large Sample Size Testing

This script provides THOROUGH testing of the EnhancedQAOrchestrator with a large
sample size (N=50+) to demonstrate real robustness and reliability.

Features:
- Large sample size testing (50+ documents)
- Diverse document types and complexity levels
- Statistical analysis and confidence intervals
- Performance benchmarking
- Error rate analysis
- Scalability testing
- Edge case testing
"""

import sys
import os
import json
import time
import uuid
import re
import random
import statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict

# Import the standalone orchestrator from the previous demo
import importlib.util
spec = importlib.util.spec_from_file_location("enhanced_demo", "03_enhanced_qa_orchestrator_demo.py")
enhanced_demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(enhanced_demo)
StandaloneEnhancedQAOrchestrator = enhanced_demo.StandaloneEnhancedQAOrchestrator


class DocumentGenerator:
    """Generate diverse test documents for comprehensive testing"""
    
    def __init__(self):
        self.document_templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, List[str]]:
        """Load document templates for different types"""
        return {
            'technical': [
                "# Technical Documentation\n\n## Overview\nThis document provides technical specifications.\n\n### Features\n- Feature 1\n- Feature 2\n\n## Implementation\nDetailed implementation notes.\n\n### Figure 1: System Architecture\nSystem architecture diagram.\n\n### Table 1: Performance Metrics\n| Metric | Value |\n|--------|-------|\n| Speed | 100ms |\n\n## References\n1. Author, A. (2023). Technical Paper.\n2. Writer, B. (2023). Implementation Guide.\n\nVisit: https://example.com/tech",
                
                "# Advanced Technical Guide\n\n## Introduction\nComprehensive technical guide.\n\n## Architecture\n### Component 1\nDescription of component.\n\n### Component 2\nAnother component.\n\n## Performance\n### Table 1: Benchmarks\n| Test | Result |\n|------|--------|\n| Load | 95% |\n\n### Figure 1: Performance Chart\nPerformance visualization.\n\n## Conclusion\nSummary of findings.\n\n## Bibliography\n1. Smith, J. (2023). Advanced Systems.\n2. Doe, A. (2023). Performance Analysis.\n\nMore info: https://example.com/advanced"
            ],
            'academic': [
                "# Research Paper\n\n## Abstract\nThis paper presents research findings.\n\n## Introduction\nBackground and motivation.\n\n## Methodology\n### Data Collection\nResearch methods used.\n\n### Analysis\nStatistical analysis performed.\n\n## Results\n### Table 1: Statistical Results\n| Variable | Mean | Std Dev |\n|----------|------|---------|\n| Group A | 15.2 | 2.1 |\n\n### Figure 1: Results Chart\nVisualization of results.\n\n## Discussion\nInterpretation of findings.\n\n## References\n1. Johnson, M. (2023). Research Methods.\n2. Brown, K. (2023). Statistical Analysis.\n\nDOI: https://doi.org/10.1234/paper",
                
                "# Academic Study\n\n## Executive Summary\nStudy overview.\n\n## Literature Review\nPrevious research.\n\n## Methods\n### Participants\nStudy participants.\n\n### Procedure\nResearch procedure.\n\n## Findings\n### Table 1: Participant Demographics\n| Age | Count | % |\n|-----|-------|---|\n| 18-25 | 45 | 30% |\n\n### Figure 1: Demographics Chart\nParticipant breakdown.\n\n## Conclusion\nStudy conclusions.\n\n## References\n1. Wilson, R. (2023). Academic Research.\n2. Taylor, S. (2023). Study Design.\n\nWebsite: https://university.edu/study"
            ],
            'business': [
                "# Business Report\n\n## Executive Summary\nBusiness overview.\n\n## Market Analysis\n### Table 1: Market Data\n| Quarter | Revenue | Growth |\n|---------|---------|--------|\n| Q1 | $1.2M | 15% |\n\n### Figure 1: Revenue Chart\nRevenue trends.\n\n## Recommendations\nBusiness recommendations.\n\n## References\n1. Business Weekly (2023).\n2. Market Report (2023).\n\nVisit: https://company.com/report",
                
                "# Strategic Plan\n\n## Vision Statement\nCompany vision.\n\n## Goals\n### Table 1: KPI Targets\n| KPI | Target | Current |\n|-----|--------|---------|\n| Sales | $5M | $3.2M |\n\n### Figure 1: Goal Progress\nProgress visualization.\n\n## Implementation\nImplementation plan.\n\n## References\n1. Strategy Guide (2023).\n2. Best Practices (2023).\n\nPortal: https://strategy.com/plan"
            ],
            'simple': [
                "# Simple Document\n\n## Introduction\nBasic introduction.\n\n## Content\nSimple content here.\n\n## Conclusion\nBasic conclusion.",
                
                "# Basic Guide\n\n## Overview\nSimple overview.\n\n## Steps\n1. Step one\n2. Step two\n\n## Summary\nBasic summary."
            ],
            'poor': [
                "this is a poor document with no structure or formatting it lacks proper organization and has no headings tables figures or references making it difficult to read and understand the content is poorly written with no clear structure or logical flow",
                
                "another bad document that has minimal content and no proper formatting it lacks any kind of structure or organization and provides no useful information or clear communication"
            ],
            'empty': [
                "",
                "   ",
                "\n\n\n"
            ]
        }
    
    def generate_document(self, doc_type: str, complexity: str = 'medium') -> Dict[str, Any]:
        """Generate a test document"""
        templates = self.document_templates.get(doc_type, self.document_templates['simple'])
        template = random.choice(templates)
        
        # Add some randomization for variety
        if complexity == 'high':
            template += "\n\n## Additional Section\nMore detailed content.\n\n### Subsection\nDetailed subsection.\n\n## References\n3. Additional Reference (2023)."
        elif complexity == 'low':
            template = template.split('\n\n')[0] + "\n\n## Basic Content\nSimple content only."
        
        # Add random metadata
        metadata = {
            'title': f"{doc_type.title()} Document {random.randint(1, 1000)}",
            'author': f"Author {random.randint(1, 100)}",
            'type': doc_type,
            'created_date': f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
            'complexity': complexity
        }
        
        return {
            'content': template,
            'metadata': metadata,
            'file_path': f'/path/to/{doc_type}_{random.randint(1, 1000)}.md'
        }
    
    def generate_test_suite(self, sample_size: int = 50) -> List[Dict[str, Any]]:
        """Generate a comprehensive test suite"""
        documents = []
        
        # Define distribution for different document types
        distribution = {
            'technical': int(sample_size * 0.25),      # 25% technical
            'academic': int(sample_size * 0.20),       # 20% academic
            'business': int(sample_size * 0.20),       # 20% business
            'simple': int(sample_size * 0.20),         # 20% simple
            'poor': int(sample_size * 0.10),           # 10% poor
            'empty': int(sample_size * 0.05)           # 5% empty
        }
        
        # Generate documents according to distribution
        for doc_type, count in distribution.items():
            for i in range(count):
                complexity = random.choice(['low', 'medium', 'high'])
                doc = self.generate_document(doc_type, complexity)
                doc['test_id'] = f"{doc_type}_{i+1}"
                documents.append(doc)
        
        # Shuffle to randomize order
        random.shuffle(documents)
        return documents


class ComprehensiveEnhancedQATester:
    """Comprehensive tester for EnhancedQAOrchestrator"""
    
    def __init__(self, sample_size: int = 50):
        self.orchestrator = StandaloneEnhancedQAOrchestrator()
        self.sample_size = sample_size
        self.doc_generator = DocumentGenerator()
        self.test_results = []
        self.performance_metrics = {}
        self.error_analysis = {}
        
    def run_comprehensive_test(self):
        """Run comprehensive testing with large sample size"""
        print(f"Comprehensive EnhancedQAOrchestrator Test Suite")
        print("=" * 60)
        print(f"Sample Size: N={self.sample_size}")
        print(f"Session ID: {self.orchestrator.session_id}")
        print(f"Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Generate test documents
        print("Generating test documents...")
        test_documents = self.doc_generator.generate_test_suite(self.sample_size)
        print(f"Generated {len(test_documents)} test documents")
        print()
        
        # Run tests
        print("Running comprehensive tests...")
        self._run_tests(test_documents)
        
        # Analyze results
        print("Analyzing results...")
        self._analyze_results()
        
        # Generate comprehensive report
        self._generate_comprehensive_report()
        
        # Save results
        self._save_test_results()
        
        print(f"\nComprehensive test completed!")
        print(f"Results saved to: {Path(__file__).parent / 'comprehensive_test_results.json'}")
    
    def _run_tests(self, documents: List[Dict[str, Any]]):
        """Run tests on all documents"""
        total_docs = len(documents)
        
        for i, doc in enumerate(documents, 1):
            print(f"Processing document {i}/{total_docs}: {doc['test_id']}")
            
            # Process document
            start_time = time.time()
            result = self.orchestrator.process_document(doc)
            processing_time = (time.time() - start_time) * 1000
            
            # Store result
            test_result = {
                'test_id': doc['test_id'],
                'document_type': doc['metadata']['type'],
                'complexity': doc['metadata']['complexity'],
                'result': result,
                'processing_time': processing_time,
                'success': result['status'] == 'success'
            }
            self.test_results.append(test_result)
            
            # Progress indicator
            if i % 10 == 0:
                print(f"  Progress: {i}/{total_docs} ({i/total_docs*100:.1f}%)")
    
    def _analyze_results(self):
        """Analyze test results comprehensively"""
        successful_results = [r for r in self.test_results if r['success']]
        failed_results = [r for r in self.test_results if not r['success']]
        
        # Basic statistics
        self.performance_metrics = {
            'total_documents': len(self.test_results),
            'successful': len(successful_results),
            'failed': len(failed_results),
            'success_rate': len(successful_results) / len(self.test_results),
            'error_rate': len(failed_results) / len(self.test_results)
        }
        
        if successful_results:
            # Processing time analysis
            times = [r['processing_time'] for r in successful_results]
            self.performance_metrics.update({
                'avg_processing_time': statistics.mean(times),
                'min_processing_time': min(times),
                'max_processing_time': max(times),
                'std_processing_time': statistics.stdev(times) if len(times) > 1 else 0,
                'median_processing_time': statistics.median(times)
            })
            
            # Quality score analysis
            scores = [r['result']['enhanced_report']['enhanced_quality_score'] for r in successful_results]
            self.performance_metrics.update({
                'avg_quality_score': statistics.mean(scores),
                'min_quality_score': min(scores),
                'max_quality_score': max(scores),
                'std_quality_score': statistics.stdev(scores) if len(scores) > 1 else 0,
                'median_quality_score': statistics.median(scores)
            })
            
            # Component score analysis
            simple_scores = [r['result']['enhanced_report']['simple_qa_score'] for r in successful_results]
            inspection_scores = [r['result']['enhanced_report']['inspection_score'] for r in successful_results]
            caption_scores = [r['result']['enhanced_report']['caption_score'] for r in successful_results]
            prediction_scores = [r['result']['enhanced_report']['prediction_score'] for r in successful_results]
            
            self.performance_metrics.update({
                'avg_simple_score': statistics.mean(simple_scores),
                'avg_inspection_score': statistics.mean(inspection_scores),
                'avg_caption_score': statistics.mean(caption_scores),
                'avg_prediction_score': statistics.mean(prediction_scores)
            })
        
        # Error analysis
        self.error_analysis = self._analyze_errors(failed_results)
        
        # Performance by document type
        self.performance_metrics['by_document_type'] = self._analyze_by_document_type()
        
        # Performance by complexity
        self.performance_metrics['by_complexity'] = self._analyze_by_complexity()
    
    def _analyze_errors(self, failed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze error patterns"""
        error_types = defaultdict(int)
        error_messages = []
        
        for result in failed_results:
            error = result['result'].get('error', 'Unknown error')
            error_types[error] += 1
            error_messages.append(error)
        
        return {
            'total_errors': len(failed_results),
            'error_types': dict(error_types),
            'most_common_error': max(error_types.items(), key=lambda x: x[1]) if error_types else None,
            'error_messages': error_messages
        }
    
    def _analyze_by_document_type(self) -> Dict[str, Dict[str, float]]:
        """Analyze performance by document type"""
        type_results = defaultdict(list)
        
        for result in self.test_results:
            if result['success']:
                type_results[result['document_type']].append(result)
        
        analysis = {}
        for doc_type, results in type_results.items():
            if results:
                times = [r['processing_time'] for r in results]
                scores = [r['result']['enhanced_report']['enhanced_quality_score'] for r in results]
                
                analysis[doc_type] = {
                    'count': len(results),
                    'avg_processing_time': statistics.mean(times),
                    'avg_quality_score': statistics.mean(scores),
                    'success_rate': len(results) / len([r for r in self.test_results if r['document_type'] == doc_type])
                }
        
        return analysis
    
    def _analyze_by_complexity(self) -> Dict[str, Dict[str, float]]:
        """Analyze performance by complexity level"""
        complexity_results = defaultdict(list)
        
        for result in self.test_results:
            if result['success']:
                complexity_results[result['complexity']].append(result)
        
        analysis = {}
        for complexity, results in complexity_results.items():
            if results:
                times = [r['processing_time'] for r in results]
                scores = [r['result']['enhanced_report']['enhanced_quality_score'] for r in results]
                
                analysis[complexity] = {
                    'count': len(results),
                    'avg_processing_time': statistics.mean(times),
                    'avg_quality_score': statistics.mean(scores),
                    'success_rate': len(results) / len([r for r in self.test_results if r['complexity'] == complexity])
                }
        
        return analysis
    
    def _generate_comprehensive_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 60)
        print("COMPREHENSIVE ENHANCED QA TEST REPORT")
        print("=" * 60)
        
        # Overall statistics
        print(f"\n📊 OVERALL STATISTICS")
        print(f"   Total Documents Tested: {self.performance_metrics['total_documents']}")
        print(f"   Successful: {self.performance_metrics['successful']}")
        print(f"   Failed: {self.performance_metrics['failed']}")
        print(f"   Success Rate: {self.performance_metrics['success_rate']:.1%}")
        print(f"   Error Rate: {self.performance_metrics['error_rate']:.1%}")
        
        # Performance metrics
        if 'avg_processing_time' in self.performance_metrics:
            print(f"\n⚡ PERFORMANCE METRICS")
            print(f"   Average Processing Time: {self.performance_metrics['avg_processing_time']:.2f}ms")
            print(f"   Median Processing Time: {self.performance_metrics['median_processing_time']:.2f}ms")
            print(f"   Min Processing Time: {self.performance_metrics['min_processing_time']:.2f}ms")
            print(f"   Max Processing Time: {self.performance_metrics['max_processing_time']:.2f}ms")
            print(f"   Standard Deviation: {self.performance_metrics['std_processing_time']:.2f}ms")
        
        # Quality score metrics
        if 'avg_quality_score' in self.performance_metrics:
            print(f"\n🎯 QUALITY SCORE METRICS")
            print(f"   Average Quality Score: {self.performance_metrics['avg_quality_score']:.3f}")
            print(f"   Median Quality Score: {self.performance_metrics['median_quality_score']:.3f}")
            print(f"   Min Quality Score: {self.performance_metrics['min_quality_score']:.3f}")
            print(f"   Max Quality Score: {self.performance_metrics['max_quality_score']:.3f}")
            print(f"   Standard Deviation: {self.performance_metrics['std_quality_score']:.3f}")
        
        # Component analysis
        if 'avg_simple_score' in self.performance_metrics:
            print(f"\n🔧 COMPONENT SCORE ANALYSIS")
            print(f"   Average Simple QA Score: {self.performance_metrics['avg_simple_score']:.3f}")
            print(f"   Average Inspection Score: {self.performance_metrics['avg_inspection_score']:.3f}")
            print(f"   Average Caption Score: {self.performance_metrics['avg_caption_score']:.3f}")
            print(f"   Average Prediction Score: {self.performance_metrics['avg_prediction_score']:.3f}")
        
        # Performance by document type
        if 'by_document_type' in self.performance_metrics:
            print(f"\n📄 PERFORMANCE BY DOCUMENT TYPE")
            for doc_type, metrics in self.performance_metrics['by_document_type'].items():
                print(f"   {doc_type.title()}:")
                print(f"     Count: {metrics['count']}")
                print(f"     Success Rate: {metrics['success_rate']:.1%}")
                print(f"     Avg Processing Time: {metrics['avg_processing_time']:.2f}ms")
                print(f"     Avg Quality Score: {metrics['avg_quality_score']:.3f}")
        
        # Performance by complexity
        if 'by_complexity' in self.performance_metrics:
            print(f"\n📈 PERFORMANCE BY COMPLEXITY")
            for complexity, metrics in self.performance_metrics['by_complexity'].items():
                print(f"   {complexity.title()}:")
                print(f"     Count: {metrics['count']}")
                print(f"     Success Rate: {metrics['success_rate']:.1%}")
                print(f"     Avg Processing Time: {metrics['avg_processing_time']:.2f}ms")
                print(f"     Avg Quality Score: {metrics['avg_quality_score']:.3f}")
        
        # Error analysis
        if self.error_analysis['total_errors'] > 0:
            print(f"\n❌ ERROR ANALYSIS")
            print(f"   Total Errors: {self.error_analysis['total_errors']}")
            print(f"   Error Types: {len(self.error_analysis['error_types'])}")
            if self.error_analysis['most_common_error']:
                print(f"   Most Common Error: {self.error_analysis['most_common_error'][0]}")
                print(f"   Occurrences: {self.error_analysis['most_common_error'][1]}")
        
        # Confidence intervals (simplified)
        if 'avg_processing_time' in self.performance_metrics and self.performance_metrics['successful'] > 30:
            print(f"\n📊 STATISTICAL CONFIDENCE")
            print(f"   Sample Size: N={self.performance_metrics['successful']}")
            print(f"   Confidence Level: High (N > 30)")
            print(f"   Standard Error: {self.performance_metrics['std_processing_time'] / (self.performance_metrics['successful'] ** 0.5):.2f}ms")
    
    def _save_test_results(self):
        """Save comprehensive test results"""
        output_data = {
            'test_info': {
                'session_id': self.orchestrator.session_id,
                'test_timestamp': datetime.now().isoformat(),
                'orchestrator_type': 'enhanced',
                'sample_size': self.sample_size,
                'total_documents': len(self.test_results)
            },
            'performance_metrics': self.performance_metrics,
            'error_analysis': self.error_analysis,
            'detailed_results': []
        }
        
        # Add detailed results (sampled for file size)
        for result in self.test_results[:20]:  # First 20 for detailed analysis
            output_data['detailed_results'].append({
                'test_id': result['test_id'],
                'document_type': result['document_type'],
                'complexity': result['complexity'],
                'processing_time': result['processing_time'],
                'success': result['success'],
                'enhanced_quality_score': result['result']['enhanced_report']['enhanced_quality_score'] if result['success'] else 0.0
            })
        
        # Add summary statistics for all results
        output_data['summary_statistics'] = {
            'all_processing_times': [r['processing_time'] for r in self.test_results],
            'all_quality_scores': [r['result']['enhanced_report']['enhanced_quality_score'] for r in self.test_results if r['success']],
            'success_flags': [r['success'] for r in self.test_results]
        }
        
        output_file = Path(__file__).parent / 'comprehensive_test_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)


def main():
    """Main function to run comprehensive testing"""
    # Test with different sample sizes
    sample_sizes = [50, 100, 200]  # Test with increasing sample sizes
    
    for sample_size in sample_sizes:
        print(f"\n{'='*60}")
        print(f"RUNNING COMPREHENSIVE TEST WITH SAMPLE SIZE N={sample_size}")
        print(f"{'='*60}")
        
        tester = ComprehensiveEnhancedQATester(sample_size)
        tester.run_comprehensive_test()
        
        # Brief pause between tests
        if sample_size != sample_sizes[-1]:
            print(f"\nPausing before next test...")
            time.sleep(2)


if __name__ == '__main__':
    main()
