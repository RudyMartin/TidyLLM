#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Unified File Classification Worker

Demonstrates the unified FileClassificationWorker functionality with all three modes:
- Simple: Basic extension and content analysis
- Enhanced: Multi-dimensional classification
- Advanced: Enhanced + AI/ML capabilities
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from backend.mcp.workers.file_classification_worker import FileClassificationWorker, ClassificationMode
import json


def test_unified_file_classification_worker():
    """Test the unified FileClassificationWorker with all three modes"""
    
    print("🚀 Testing Unified FileClassificationWorker")
    print("=" * 60)
    
    # Test content for all modes
    test_content = """
    Model Development Document
    
    This document outlines the development methodology for our risk model.
    The model architecture includes multiple layers and data sources.
    Implementation details are provided in the following sections.
    
    Development Methodology:
    - Data preprocessing and feature engineering
    - Model selection and validation
    - Performance evaluation metrics
    
    Review ID: REV00001
    
    This document contains confidential information and is subject to SOX compliance.
    The content is for internal use only and should not be shared externally.
    """
    
    # Test 1: Simple Mode
    print("\n📋 Test 1: Simple Mode")
    print("-" * 40)
    
    simple_worker = FileClassificationWorker(ClassificationMode.SIMPLE)
    print(f"Worker Mode: {simple_worker.mode.value}")
    
    simple_result = simple_worker.classify_file("model_dev_doc.pdf", test_content)
    print(f"Classification: {simple_result.get('type', 'Unknown')}")
    print(f"Category: {simple_result.get('category', 'Unknown')}")
    print(f"Confidence: {simple_result.get('confidence', 0.0):.2f}")
    
    # Test 2: Enhanced Mode
    print("\n📋 Test 2: Enhanced Mode")
    print("-" * 40)
    
    enhanced_worker = FileClassificationWorker(ClassificationMode.ENHANCED)
    print(f"Worker Mode: {enhanced_worker.mode.value}")
    
    enhanced_result = enhanced_worker.classify_file_enhanced({
        'filename': 'model_dev_doc.pdf',
        'content': test_content,
        'file_size': 1024 * 1024,  # 1MB
        'metadata': {
            'author': 'John Doe',
            'department': 'Engineering',
            'creation_date': '2024-01-15',
            'version': '1.0'
        },
        'context': {
            'project': 'Risk Model Development',
            'audience': 'internal',
            'workflow_stage': 'final'
        }
    })
    
    print(f"Primary Classification: {enhanced_result.get('primary_classification', 'Unknown')}")
    print(f"Overall Confidence: {enhanced_result.get('overall_confidence', 0.0):.2f}")
    print(f"Classification Method: {enhanced_result.get('classification_method', 'Unknown')}")
    
    # Show dimensions
    dimensions = enhanced_result.get('dimensions', {})
    print(f"\n📊 Classification Dimensions:")
    for dimension, data in dimensions.items():
        print(f"  {dimension}: {data.get('confidence_score', 0.0):.2f} confidence")
    
    # Show recommendations
    recommendations = enhanced_result.get('recommendations', [])
    if recommendations:
        print(f"\n💡 Recommendations:")
        for rec in recommendations:
            print(f"  - {rec}")
    
    # Test 3: Advanced Mode
    print("\n📋 Test 3: Advanced Mode")
    print("-" * 40)
    
    advanced_worker = FileClassificationWorker(ClassificationMode.ADVANCED)
    print(f"Worker Mode: {advanced_worker.mode.value}")
    
    advanced_result = advanced_worker.classify_file_enhanced({
        'filename': 'model_dev_doc.pdf',
        'content': test_content,
        'file_size': 1024 * 1024,
        'metadata': {
            'author': 'John Doe',
            'department': 'Engineering',
            'creation_date': '2024-01-15',
            'version': '1.0'
        },
        'context': {
            'project': 'Risk Model Development',
            'audience': 'internal',
            'workflow_stage': 'final'
        }
    })
    
    # Add advanced features
    advanced_features = advanced_worker._apply_advanced_features({}, advanced_result)
    advanced_result['advanced_features'] = advanced_features
    
    print(f"Primary Classification: {advanced_result.get('primary_classification', 'Unknown')}")
    print(f"Overall Confidence: {advanced_result.get('overall_confidence', 0.0):.2f}")
    
    # Show advanced features
    ai_analysis = advanced_features.get('ai_analysis', {})
    print(f"\n🤖 AI Analysis:")
    print(f"  Sentiment Score: {ai_analysis.get('sentiment_score', 0.0):.2f}")
    print(f"  Complexity Level: {ai_analysis.get('complexity_level', 'Unknown')}")
    print(f"  Writing Style: {ai_analysis.get('writing_style', 'Unknown')}")
    
    ml_predictions = advanced_features.get('ml_predictions', {})
    print(f"\n📈 ML Predictions:")
    print(f"  Quality Score: {ml_predictions.get('quality_score', 0.0):.2f}")
    print(f"  Completion Probability: {ml_predictions.get('completion_probability', 0.0):.2f}")
    print(f"  Risk Level: {ml_predictions.get('risk_level', 'Unknown')}")
    
    # Test 4: Mode Comparison
    print("\n📋 Test 4: Mode Comparison")
    print("-" * 40)
    
    test_files = [
        {
            'filename': 'financial_report.xlsx',
            'content': 'Annual Financial Report with SOX compliance requirements and confidential financial data.',
            'file_size': 2048 * 1024,
            'metadata': {'author': 'Finance Team'},
            'context': {'project': 'Annual Reporting'}
        },
        {
            'filename': 'research_paper.pdf',
            'content': 'Abstract: This research presents novel findings in sparse representation theory. Introduction, Methodology, Conclusion, References.',
            'file_size': 512 * 1024,
            'metadata': {'author': 'Research Team'},
            'context': {'project': 'Academic Research'}
        },
        {
            'filename': 'validation_scope.yaml',
            'content': 'Validation Scoping Template: Stakeholders, Timeline, Resources, Validation Framework, Review ID: REV00002',
            'file_size': 256 * 1024,
            'metadata': {'author': 'Compliance Team'},
            'context': {'project': 'Model Validation'}
        }
    ]
    
    print("Comparing classification results across modes:")
    print(f"{'File':<20} {'Simple':<15} {'Enhanced':<15} {'Advanced':<15}")
    print("-" * 70)
    
    for file_data in test_files:
        filename = file_data['filename']
        content = file_data['content']
        
        # Simple classification
        simple_result = simple_worker.classify_file(filename, content)
        simple_type = simple_result.get('type', 'Unknown')
        
        # Enhanced classification
        enhanced_result = enhanced_worker.classify_file_enhanced(file_data)
        enhanced_type = enhanced_result.get('primary_classification', 'Unknown')
        
        # Advanced classification
        advanced_result = advanced_worker.classify_file_enhanced(file_data)
        advanced_type = advanced_result.get('primary_classification', 'Unknown')
        
        print(f"{filename:<20} {simple_type:<15} {enhanced_type:<15} {advanced_type:<15}")
    
    # Test 5: Validation Comparison
    print("\n📋 Test 5: Validation Comparison")
    print("-" * 40)
    
    # Test simple validation
    simple_validation = simple_worker.validate_file({
        'filename': 'test.pdf',
        'content': test_content,
        'file_size': 1024 * 1024,
        'classification': simple_result
    })
    
    # Test enhanced validation
    enhanced_validation = enhanced_worker.validate_file_enhanced({
        'filename': 'test.pdf',
        'content': test_content,
        'file_size': 1024 * 1024,
        'classification': enhanced_result
    })
    
    print(f"Simple Validation - Valid: {simple_validation.get('valid', False)}")
    print(f"Enhanced Validation - Valid: {enhanced_validation.get('valid', False)}")
    
    enhanced_checks = enhanced_validation.get('enhanced_checks', {})
    if enhanced_checks:
        print(f"Enhanced Checks: {len(enhanced_checks)} additional validations")
        for check_type, check_data in enhanced_checks.items():
            print(f"  {check_type}: {check_data.get('recommendation', 'No recommendation')}")
    
    # Test 6: Performance Metrics
    print("\n📋 Test 6: Performance Metrics")
    print("-" * 40)
    
    for mode in [ClassificationMode.SIMPLE, ClassificationMode.ENHANCED, ClassificationMode.ADVANCED]:
        worker = FileClassificationWorker(mode)
        metrics = worker.get_performance_metrics()
        capabilities = worker.get_supported_file_types()
        
        print(f"\n{mode.value.upper()} Mode:")
        print(f"  Worker: {metrics['worker_name']}")
        print(f"  Type: {metrics['worker_type']}")
        print(f"  Supported File Types: {capabilities['total_types']}")
        print(f"  Classification Keywords: {len(capabilities['classification_keywords'])}")
        print(f"  Review ID Patterns: {len(capabilities['review_id_patterns'])}")
    
    print("\n✅ Unified FileClassificationWorker testing completed!")
    print("\n🎯 Key Benefits of Unified Worker:")
    print("  ✅ Single codebase - no duplication")
    print("  ✅ Configurable behavior via mode parameter")
    print("  ✅ Progressive complexity (Simple → Enhanced → Advanced)")
    print("  ✅ Consistent interface across all modes")
    print("  ✅ Easy maintenance and updates")
    print("  ✅ Follows MCP atomicity principles")


if __name__ == "__main__":
    test_unified_file_classification_worker()


