#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real Documents EnhancedQAOrchestrator Test - Authentic Evidence

This script tests the EnhancedQAOrchestrator against REAL documents from our
repository, providing authentic evidence of performance with actual content.

Features:
- Real document testing (PDFs, markdown files, documentation)
- Authentic content from knowledge base and docs
- Real-world performance metrics
- Actual document complexity and structure
- Authentic evidence of system capabilities
"""

import sys
import os
import json
import time
import uuid
import re
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict

# Import the standalone orchestrator
import importlib.util
spec = importlib.util.spec_from_file_location("enhanced_demo", "03_enhanced_qa_orchestrator_demo.py")
enhanced_demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(enhanced_demo)
StandaloneEnhancedQAOrchestrator = enhanced_demo.StandaloneEnhancedQAOrchestrator


class RealDocumentLoader:
    """Load real documents from our repository"""
    
    def __init__(self):
        self.repo_root = Path(__file__).parent.parent.parent
        self.documents = []
        
    def load_real_documents(self) -> List[Dict[str, Any]]:
        """Load real documents from various sources in our repository"""
        documents = []
        
        # 1. Load markdown documentation files
        docs_files = self._load_markdown_files()
        documents.extend(docs_files)
        
        # 2. Load notebook files (Python scripts with markdown)
        notebook_files = self._load_notebook_files()
        documents.extend(notebook_files)
        
        # 3. Load README and documentation files
        readme_files = self._load_readme_files()
        documents.extend(readme_files)
        
        # 4. Load analysis reports
        report_files = self._load_report_files()
        documents.extend(report_files)
        
        # 5. Load configuration and setup files
        config_files = self._load_config_files()
        documents.extend(config_files)
        
        print(f"Loaded {len(documents)} real documents from repository")
        return documents
    
    def _load_markdown_files(self) -> List[Dict[str, Any]]:
        """Load markdown files from docs directory"""
        docs_dir = self.repo_root / "docs"
        markdown_files = []
        
        # Find all markdown files
        for md_file in docs_dir.rglob("*.md"):
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if len(content.strip()) > 50:  # Only include substantial files
                    relative_path = md_file.relative_to(self.repo_root)
                    
                    markdown_files.append({
                        'content': content,
                        'metadata': {
                            'title': md_file.stem.replace('_', ' ').title(),
                            'author': 'System Documentation',
                            'type': 'documentation',
                            'file_path': str(relative_path),
                            'file_size': len(content),
                            'source': 'docs'
                        },
                        'test_id': f"docs_{md_file.stem}"
                    })
            except Exception as e:
                print(f"Error loading {md_file}: {e}")
        
        return markdown_files
    
    def _load_notebook_files(self) -> List[Dict[str, Any]]:
        """Load Python notebook files"""
        notebooks_dir = self.repo_root / "notebooks"
        notebook_files = []
        
        # Find Python files that look like notebooks/demos
        for py_file in notebooks_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Only include substantial files with documentation
                if len(content.strip()) > 200 and ('"""' in content or "'''" in content):
                    relative_path = py_file.relative_to(self.repo_root)
                    
                    # Extract docstring for content
                    docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
                    if docstring_match:
                        doc_content = docstring_match.group(1)
                    else:
                        doc_content = f"# {py_file.stem}\n\nPython script: {py_file.name}\n\n{content[:500]}..."
                    
                    notebook_files.append({
                        'content': doc_content,
                        'metadata': {
                            'title': py_file.stem.replace('_', ' ').title(),
                            'author': 'System Demo',
                            'type': 'notebook',
                            'file_path': str(relative_path),
                            'file_size': len(content),
                            'source': 'notebooks'
                        },
                        'test_id': f"notebook_{py_file.stem}"
                    })
            except Exception as e:
                print(f"Error loading {py_file}: {e}")
        
        return notebook_files
    
    def _load_readme_files(self) -> List[Dict[str, Any]]:
        """Load README and important documentation files"""
        readme_files = []
        
        # Find README files
        for readme_file in self.repo_root.rglob("README*.md"):
            try:
                with open(readme_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if len(content.strip()) > 100:
                    relative_path = readme_file.relative_to(self.repo_root)
                    
                    readme_files.append({
                        'content': content,
                        'metadata': {
                            'title': readme_file.stem.replace('_', ' ').title(),
                            'author': 'Project Documentation',
                            'type': 'readme',
                            'file_path': str(relative_path),
                            'file_size': len(content),
                            'source': 'root'
                        },
                        'test_id': f"readme_{readme_file.stem}"
                    })
            except Exception as e:
                print(f"Error loading {readme_file}: {e}")
        
        return readme_files
    
    def _load_report_files(self) -> List[Dict[str, Any]]:
        """Load analysis and report files"""
        report_files = []
        
        # Find report files in various locations
        report_patterns = [
            "**/*_REPORT.md",
            "**/*_ANALYSIS.md", 
            "**/*_SUMMARY.md",
            "**/*_EVIDENCE.md"
        ]
        
        for pattern in report_patterns:
            for report_file in self.repo_root.glob(pattern):
                try:
                    with open(report_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if len(content.strip()) > 200:
                        relative_path = report_file.relative_to(self.repo_root)
                        
                        report_files.append({
                            'content': content,
                            'metadata': {
                                'title': report_file.stem.replace('_', ' ').title(),
                                'author': 'System Analysis',
                                'type': 'report',
                                'file_path': str(relative_path),
                                'file_size': len(content),
                                'source': 'analysis'
                            },
                            'test_id': f"report_{report_file.stem}"
                        })
                except Exception as e:
                    print(f"Error loading {report_file}: {e}")
        
        return report_files
    
    def _load_config_files(self) -> List[Dict[str, Any]]:
        """Load configuration and setup files"""
        config_files = []
        
        # Find configuration files
        config_patterns = [
            "**/requirements*.txt",
            "**/*.yaml",
            "**/*.yml",
            "**/pyproject.toml",
            "**/setup.py"
        ]
        
        for pattern in config_patterns:
            for config_file in self.repo_root.glob(pattern):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if len(content.strip()) > 50:
                        relative_path = config_file.relative_to(self.repo_root)
                        
                        # Create markdown-like content for config files
                        config_content = f"# {config_file.name}\n\n## Configuration\n\n```\n{content}\n```"
                        
                        config_files.append({
                            'content': config_content,
                            'metadata': {
                                'title': config_file.name,
                                'author': 'System Configuration',
                                'type': 'configuration',
                                'file_path': str(relative_path),
                                'file_size': len(content),
                                'source': 'config'
                            },
                            'test_id': f"config_{config_file.stem}"
                        })
                except Exception as e:
                    print(f"Error loading {config_file}: {e}")
        
        return config_files


class RealDocumentEnhancedQATester:
    """Test EnhancedQAOrchestrator with real documents"""
    
    def __init__(self):
        self.orchestrator = StandaloneEnhancedQAOrchestrator()
        self.doc_loader = RealDocumentLoader()
        self.test_results = []
        self.performance_metrics = {}
        self.error_analysis = {}
        
    def run_real_document_test(self):
        """Run comprehensive testing with real documents"""
        print(f"Real Documents EnhancedQAOrchestrator Test")
        print("=" * 60)
        print(f"Session ID: {self.orchestrator.session_id}")
        print(f"Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Load real documents
        print("Loading real documents from repository...")
        real_documents = self.doc_loader.load_real_documents()
        
        if not real_documents:
            print("No real documents found!")
            return
        
        print(f"Loaded {len(real_documents)} real documents")
        print()
        
        # Run tests
        print("Running tests with real documents...")
        self._run_tests(real_documents)
        
        # Analyze results
        print("Analyzing results...")
        self._analyze_results()
        
        # Generate comprehensive report
        self._generate_comprehensive_report()
        
        # Save results
        self._save_test_results()
        
        print(f"\nReal document test completed!")
        print(f"Results saved to: {Path(__file__).parent / 'real_documents_test_results.json'}")
    
    def _run_tests(self, documents: List[Dict[str, Any]]):
        """Run tests on all real documents"""
        total_docs = len(documents)
        
        for i, doc in enumerate(documents, 1):
            print(f"Processing document {i}/{total_docs}: {doc['test_id']}")
            print(f"  File: {doc['metadata']['file_path']}")
            print(f"  Type: {doc['metadata']['type']}")
            print(f"  Size: {doc['metadata']['file_size']} chars")
            
            # Process document
            start_time = time.time()
            result = self.orchestrator.process_document(doc)
            processing_time = (time.time() - start_time) * 1000
            
            # Store result
            test_result = {
                'test_id': doc['test_id'],
                'document_type': doc['metadata']['type'],
                'file_path': doc['metadata']['file_path'],
                'file_size': doc['metadata']['file_size'],
                'source': doc['metadata']['source'],
                'result': result,
                'processing_time': processing_time,
                'success': result['status'] == 'success'
            }
            self.test_results.append(test_result)
            
            # Show quick result
            if result['status'] == 'success':
                enhanced_score = result['enhanced_report']['enhanced_quality_score']
                print(f"  Result: ✅ Success (Score: {enhanced_score:.3f}, Time: {processing_time:.2f}ms)")
            else:
                print(f"  Result: ❌ Failed ({result.get('error', 'Unknown error')})")
            
            print()
            
            # Progress indicator
            if i % 10 == 0:
                print(f"Progress: {i}/{total_docs} ({i/total_docs*100:.1f}%)")
                print()
    
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
                'avg_processing_time': sum(times) / len(times),
                'min_processing_time': min(times),
                'max_processing_time': max(times),
                'median_processing_time': sorted(times)[len(times)//2]
            })
            
            # Quality score analysis
            scores = [r['result']['enhanced_report']['enhanced_quality_score'] for r in successful_results]
            self.performance_metrics.update({
                'avg_quality_score': sum(scores) / len(scores),
                'min_quality_score': min(scores),
                'max_quality_score': max(scores),
                'median_quality_score': sorted(scores)[len(scores)//2]
            })
            
            # Component score analysis
            simple_scores = [r['result']['enhanced_report']['simple_qa_score'] for r in successful_results]
            inspection_scores = [r['result']['enhanced_report']['inspection_score'] for r in successful_results]
            caption_scores = [r['result']['enhanced_report']['caption_score'] for r in successful_results]
            prediction_scores = [r['result']['enhanced_report']['prediction_score'] for r in successful_results]
            
            self.performance_metrics.update({
                'avg_simple_score': sum(simple_scores) / len(simple_scores),
                'avg_inspection_score': sum(inspection_scores) / len(inspection_scores),
                'avg_caption_score': sum(caption_scores) / len(caption_scores),
                'avg_prediction_score': sum(prediction_scores) / len(prediction_scores)
            })
        
        # Error analysis
        self.error_analysis = self._analyze_errors(failed_results)
        
        # Performance by document type
        self.performance_metrics['by_document_type'] = self._analyze_by_document_type()
        
        # Performance by source
        self.performance_metrics['by_source'] = self._analyze_by_source()
        
        # Performance by file size
        self.performance_metrics['by_file_size'] = self._analyze_by_file_size()
    
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
                    'avg_processing_time': sum(times) / len(times),
                    'avg_quality_score': sum(scores) / len(scores),
                    'success_rate': len(results) / len([r for r in self.test_results if r['document_type'] == doc_type])
                }
        
        return analysis
    
    def _analyze_by_source(self) -> Dict[str, Dict[str, float]]:
        """Analyze performance by document source"""
        source_results = defaultdict(list)
        
        for result in self.test_results:
            if result['success']:
                source_results[result['source']].append(result)
        
        analysis = {}
        for source, results in source_results.items():
            if results:
                times = [r['processing_time'] for r in results]
                scores = [r['result']['enhanced_report']['enhanced_quality_score'] for r in results]
                
                analysis[source] = {
                    'count': len(results),
                    'avg_processing_time': sum(times) / len(times),
                    'avg_quality_score': sum(scores) / len(scores),
                    'success_rate': len(results) / len([r for r in self.test_results if r['source'] == source])
                }
        
        return analysis
    
    def _analyze_by_file_size(self) -> Dict[str, Dict[str, float]]:
        """Analyze performance by file size"""
        size_results = defaultdict(list)
        
        for result in self.test_results:
            if result['success']:
                size = result['file_size']
                if size < 1000:
                    size_category = 'small'
                elif size < 5000:
                    size_category = 'medium'
                elif size < 20000:
                    size_category = 'large'
                else:
                    size_category = 'very_large'
                
                size_results[size_category].append(result)
        
        analysis = {}
        for size_cat, results in size_results.items():
            if results:
                times = [r['processing_time'] for r in results]
                scores = [r['result']['enhanced_report']['enhanced_quality_score'] for r in results]
                
                analysis[size_cat] = {
                    'count': len(results),
                    'avg_processing_time': sum(times) / len(times),
                    'avg_quality_score': sum(scores) / len(scores),
                    'avg_file_size': sum(r['file_size'] for r in results) / len(results)
                }
        
        return analysis
    
    def _generate_comprehensive_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 60)
        print("REAL DOCUMENTS ENHANCED QA TEST REPORT")
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
        
        # Quality score metrics
        if 'avg_quality_score' in self.performance_metrics:
            print(f"\n🎯 QUALITY SCORE METRICS")
            print(f"   Average Quality Score: {self.performance_metrics['avg_quality_score']:.3f}")
            print(f"   Median Quality Score: {self.performance_metrics['median_quality_score']:.3f}")
            print(f"   Min Quality Score: {self.performance_metrics['min_quality_score']:.3f}")
            print(f"   Max Quality Score: {self.performance_metrics['max_quality_score']:.3f}")
        
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
        
        # Performance by source
        if 'by_source' in self.performance_metrics:
            print(f"\n📁 PERFORMANCE BY SOURCE")
            for source, metrics in self.performance_metrics['by_source'].items():
                print(f"   {source.title()}:")
                print(f"     Count: {metrics['count']}")
                print(f"     Success Rate: {metrics['success_rate']:.1%}")
                print(f"     Avg Processing Time: {metrics['avg_processing_time']:.2f}ms")
                print(f"     Avg Quality Score: {metrics['avg_quality_score']:.3f}")
        
        # Performance by file size
        if 'by_file_size' in self.performance_metrics:
            print(f"\n📏 PERFORMANCE BY FILE SIZE")
            for size_cat, metrics in self.performance_metrics['by_file_size'].items():
                print(f"   {size_cat.replace('_', ' ').title()}:")
                print(f"     Count: {metrics['count']}")
                print(f"     Avg File Size: {metrics['avg_file_size']:.0f} chars")
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
        
        # Authentic evidence summary
        print(f"\n🎯 AUTHENTIC EVIDENCE SUMMARY")
        print(f"   Real Documents Tested: {self.performance_metrics['total_documents']}")
        print(f"   Document Sources: {len(self.performance_metrics['by_source'])}")
        print(f"   Document Types: {len(self.performance_metrics['by_document_type'])}")
        print(f"   File Size Range: {self.performance_metrics['min_processing_time']:.2f}ms - {self.performance_metrics['max_processing_time']:.2f}ms")
        print(f"   Quality Score Range: {self.performance_metrics['min_quality_score']:.3f} - {self.performance_metrics['max_quality_score']:.3f}")
    
    def _save_test_results(self):
        """Save comprehensive test results"""
        output_data = {
            'test_info': {
                'session_id': self.orchestrator.session_id,
                'test_timestamp': datetime.now().isoformat(),
                'orchestrator_type': 'enhanced',
                'test_type': 'real_documents',
                'total_documents': len(self.test_results)
            },
            'performance_metrics': self.performance_metrics,
            'error_analysis': self.error_analysis,
            'detailed_results': []
        }
        
        # Add detailed results (first 20 for detailed analysis)
        for result in self.test_results[:20]:
            output_data['detailed_results'].append({
                'test_id': result['test_id'],
                'document_type': result['document_type'],
                'file_path': result['file_path'],
                'file_size': result['file_size'],
                'source': result['source'],
                'processing_time': result['processing_time'],
                'success': result['success'],
                'enhanced_quality_score': result['result']['enhanced_report']['enhanced_quality_score'] if result['success'] else 0.0
            })
        
        # Add summary statistics for all results
        output_data['summary_statistics'] = {
            'all_processing_times': [r['processing_time'] for r in self.test_results],
            'all_quality_scores': [r['result']['enhanced_report']['enhanced_quality_score'] for r in self.test_results if r['success']],
            'all_file_sizes': [r['file_size'] for r in self.test_results],
            'success_flags': [r['success'] for r in self.test_results]
        }
        
        output_file = Path(__file__).parent / 'real_documents_test_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)


def main():
    """Main function to run real document testing"""
    tester = RealDocumentEnhancedQATester()
    tester.run_real_document_test()


if __name__ == '__main__':
    main()
