#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SimpleQAOrchestrator Demo - MCP Foundation Demonstration

This script demonstrates the SimpleQAOrchestrator capabilities as the foundation
of the MCP (Model Context Protocol) progressive complexity architecture.

Features Demonstrated:
- MCP Document Processing Pipeline
- Quality Assessment Engine
- Session Management
- Error Handling
- Performance Monitoring
- Report Generation
- MCP Compliance
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from backend.mcp.orchestrators.simple_qa_orchestrator import SimpleQAOrchestrator


class SimpleQADemo:
    """Comprehensive demo of SimpleQAOrchestrator MCP capabilities"""
    
    def __init__(self):
        self.orchestrator = SimpleQAOrchestrator()
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
| Response Time | < 100ms | ✅ Excellent |
| Throughput | 1000 req/s | ✅ Excellent |
| Error Rate | < 0.1% | ✅ Excellent |

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
                'name': 'Fair Document',
                'content': """
# Basic Document

This document has some structure but could be improved. It contains basic content with minimal formatting.

The sentences are generally readable but could benefit from better organization. There are some formatting inconsistencies that should be addressed.

## Main Section

This section provides basic information but lacks detailed structure. The content is present but could be better organized.

- Basic list item
- Another item
- Third item

The document ends with a simple conclusion.
                """,
                'metadata': {
                    'title': 'Basic Document',
                    'author': 'Author',
                    'type': 'basic'
                },
                'expected_score': 0.60
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
        print(f"Results saved to: {Path(__file__).parent / 'demo_results.json'}")
    
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
        
        output_file = Path(__file__).parent / 'demo_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    def run_quick_demo(self):
        """Run a quick demonstration with a single document"""
        print("Quick SimpleQAOrchestrator Demo")
        print("=" * 40)
        
        test_document = {
            'content': """
# Quick Test Document

This is a quick test to demonstrate SimpleQAOrchestrator functionality.

## Features Tested

- Document processing pipeline
- Quality assessment engine
- Session management
- Error handling
- Performance monitoring

## Conclusion

The demo shows basic functionality working correctly.
            """,
            'metadata': {
                'title': 'Quick Test Document',
                'author': 'Demo User',
                'type': 'test'
            },
            'file_path': '/path/to/quick_test.md'
        }
        
        print(f"Session ID: {self.orchestrator.session_id}")
        print(f"Processing test document...")
        
        result = self.orchestrator.process_document(test_document)
        
        if result['status'] == 'success':
            report = result['report']
            print(f"Quality Score: {report['quality_score']:.3f}")
            print(f"Status: {report['status']}")
            print(f"Processing Time: {result['processing_time_ms']:.2f}ms")
            print(f"Pass Rate: {report['pass_rate']:.1%}")
            print("Demo completed successfully!")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")


def main():
    """Main function to run the demo"""
    demo = SimpleQADemo()
    
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        demo.run_quick_demo()
    else:
        demo.run_comprehensive_demo()


if __name__ == '__main__':
    main()
