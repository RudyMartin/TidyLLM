#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test File Classification Worker

Demonstrates the FileClassificationWorker functionality with various file types
and content analysis.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from backend.mcp.workers.file_classification_worker import FileClassificationWorker
import json


def test_file_classification_worker():
    """Test the FileClassificationWorker with various scenarios"""
    
    print("🚀 Testing FileClassificationWorker")
    print("=" * 50)
    
    # Initialize worker
    worker = FileClassificationWorker()
    
    # Test 1: Model Development Document
    print("\n📋 Test 1: Model Development Document")
    print("-" * 30)
    
    mdd_content = """
    Model Development Document
    
    This document outlines the development methodology for our risk model.
    The model architecture includes multiple layers and data sources.
    Implementation details are provided in the following sections.
    
    Development Methodology:
    - Data preprocessing and feature engineering
    - Model selection and validation
    - Performance evaluation metrics
    
    Review ID: REV00001
    """
    
    result = worker.classify_file("model_dev_doc.pdf", mdd_content)
    print(f"Classification: {result['category']}")
    print(f"Method: {result['classification_method']}")
    print(f"Confidence: {result['confidence_score']:.2f}")
    print(f"Review ID: {worker.extract_review_id(mdd_content)}")
    
    # Test 2: Whitepaper
    print("\n📋 Test 2: Whitepaper")
    print("-" * 30)
    
    whitepaper_content = """
    Abstract
    
    This research paper presents a novel approach to sparse representation
    in computer vision applications. We introduce a new methodology that
    improves upon existing techniques.
    
    Introduction
    
    Sparse representation has become increasingly important in modern
    computer vision systems. Our study focuses on...
    
    Methodology
    
    We conducted extensive experiments using...
    
    Conclusion
    
    Our results demonstrate significant improvements...
    
    References
    
    1. Wright et al. (2009) - Sparse Representation for Computer Vision
    2. Olshausen & Field (1997) - Emergence of Simple-Cell Receptive Fields
    
    Review ID: REV00002
    """
    
    result = worker.classify_file("sparse_representation_paper.pdf", whitepaper_content)
    print(f"Classification: {result['category']}")
    print(f"Method: {result['classification_method']}")
    print(f"Confidence: {result['confidence_score']:.2f}")
    print(f"Review ID: {worker.extract_review_id(whitepaper_content)}")
    
    # Test 3: Validation Scoping Template
    print("\n📋 Test 3: Validation Scoping Template")
    print("-" * 30)
    
    vst_content = """
    Validation Scoping Template
    
    This document defines the scope and framework for model validation.
    
    Validation Scope:
    - Model performance assessment
    - Risk evaluation framework
    - Compliance verification
    
    Stakeholders:
    - Model Development Team
    - Risk Management
    - Compliance Officers
    
    Timeline:
    - Phase 1: Initial assessment (2 weeks)
    - Phase 2: Detailed validation (4 weeks)
    - Phase 3: Final review (1 week)
    
    Resources:
    - Validation team: 3 FTE
    - Tools: Automated validation framework
    - Budget: $50,000
    
    Review ID: REV00003
    """
    
    result = worker.classify_file("validation_scope.yaml", vst_content)
    print(f"Classification: {result['category']}")
    print(f"Method: {result['classification_method']}")
    print(f"Confidence: {result['confidence_score']:.2f}")
    print(f"Review ID: {worker.extract_review_id(vst_content)}")
    
    # Test 4: Annual Model Review
    print("\n📋 Test 4: Annual Model Review")
    print("-" * 30)
    
    amr_content = """
    Annual Model Review Report
    
    This comprehensive review covers the performance and risk assessment
    of our portfolio of models for the fiscal year 2024.
    
    Review Period: January 2024 - December 2024
    
    Model Performance Summary:
    - Model A: 95% accuracy, low risk
    - Model B: 87% accuracy, medium risk
    - Model C: 92% accuracy, low risk
    
    Risk Assessment:
    - Overall portfolio risk: Medium
    - Key risk factors identified
    - Mitigation strategies implemented
    
    Recommendations:
    - Retrain Model B with additional data
    - Implement enhanced monitoring for Model C
    - Continue current strategy for Model A
    
    Review ID: REV00004
    """
    
    result = worker.classify_file("annual_review_2024.xlsx", amr_content)
    print(f"Classification: {result['category']}")
    print(f"Method: {result['classification_method']}")
    print(f"Confidence: {result['confidence_score']:.2f}")
    print(f"Review ID: {worker.extract_review_id(amr_content)}")
    
    # Test 5: Unknown file type
    print("\n📋 Test 5: Unknown File Type")
    print("-" * 30)
    
    unknown_content = """
    Some random content that doesn't match any specific document type.
    This should be classified as unclassified.
    """
    
    result = worker.classify_file("unknown_file.txt", unknown_content)
    print(f"Classification: {result['category']}")
    print(f"Method: {result['classification_method']}")
    print(f"Confidence: {result['confidence_score']:.2f}")
    print(f"Review ID: {worker.extract_review_id(unknown_content)}")
    
    # Test 6: Batch classification
    print("\n📋 Test 6: Batch Classification")
    print("-" * 30)
    
    batch_files = [
        {
            'filename': 'model_doc.pdf',
            'content': mdd_content,
            'file_size': 1024 * 1024
        },
        {
            'filename': 'research_paper.pdf',
            'content': whitepaper_content,
            'file_size': 2048 * 1024
        },
        {
            'filename': 'validation_scope.yaml',
            'content': vst_content,
            'file_size': 512 * 1024
        }
    ]
    
    batch_result = worker._batch_classify_task({'files': batch_files})
    print(f"Total files processed: {batch_result['total_files']}")
    print(f"Review ID groups: {len(batch_result['review_id_groups'])}")
    
    for review_id, files in batch_result['review_id_groups'].items():
        print(f"  Review ID {review_id}: {len(files)} files")
        for file_info in files:
            print(f"    - {file_info['filename']} ({file_info['category']})")
    
    # Test 7: File validation
    print("\n📋 Test 7: File Validation")
    print("-" * 30)
    
    validation_result = worker.validate_file({
        'filename': 'test_document.pdf',
        'content': mdd_content,
        'file_size': 1024 * 1024,  # 1MB
        'classification': result
    })
    
    print(f"Valid: {validation_result['valid']}")
    print(f"File size: {validation_result['file_size_mb']:.2f}MB")
    print(f"Extension: {validation_result['file_extension']}")
    print(f"Review ID: {validation_result['review_id']}")
    
    if validation_result['errors']:
        print("Errors:")
        for error in validation_result['errors']:
            print(f"  - {error}")
    
    if validation_result['warnings']:
        print("Warnings:")
        for warning in validation_result['warnings']:
            print(f"  - {warning}")
    
    # Test 8: Performance metrics
    print("\n📋 Test 8: Performance Metrics")
    print("-" * 30)
    
    metrics = worker.get_performance_metrics()
    print(f"Worker: {metrics['worker_name']}")
    print(f"Type: {metrics['worker_type']}")
    print(f"Total tasks: {metrics['performance_metrics']['total_tasks']}")
    print(f"Successful tasks: {metrics['performance_metrics']['successful_tasks']}")
    print(f"Failed tasks: {metrics['performance_metrics']['failed_tasks']}")
    
    # Test 9: Supported file types
    print("\n📋 Test 9: Supported File Types")
    print("-" * 30)
    
    supported_types = worker.get_supported_file_types()
    print(f"Total file types: {supported_types['total_types']}")
    print(f"Classification keywords: {len(supported_types['classification_keywords'])}")
    print(f"Review ID patterns: {len(supported_types['review_id_patterns'])}")
    
    print("\nFile types:")
    for doc_type, config in supported_types['file_types'].items():
        print(f"  - {doc_type}: {config.get('category', 'Unknown')}")
    
    print("\n✅ FileClassificationWorker testing completed!")


if __name__ == "__main__":
    test_file_classification_worker()
