#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Image Processing Worker

Demonstrates the unified ImageProcessingWorker functionality with all three modes:
- Simple: Basic image extraction and metadata
- Enhanced: OCR, image classification, quality assessment
- Advanced: AI-powered analysis, object detection, semantic understanding
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from backend.mcp.workers.image_processing_worker import ImageProcessingWorker, ImageProcessingMode
import json


def test_image_processing_worker():
    """Test the unified ImageProcessingWorker with all three modes"""
    
    print("🚀 Testing ImageProcessingWorker")
    print("=" * 60)
    
    # Test 1: Simple Mode
    print("\n📋 Test 1: Simple Mode")
    print("-" * 40)
    
    simple_worker = ImageProcessingWorker(ImageProcessingMode.SIMPLE)
    print(f"Worker Mode: {simple_worker.mode.value}")
    
    # Test with mock data (no actual PDF file)
    simple_result = simple_worker._extract_images_simple({
        'file_path': 'test_document.pdf',
        'page_numbers': [0, 1]
    })
    
    print(f"Success: {simple_result.get('success', False)}")
    print(f"Extraction Method: {simple_result.get('extraction_method', 'Unknown')}")
    print(f"Confidence Score: {simple_result.get('confidence_score', 0.0):.2f}")
    print(f"Total Images: {simple_result.get('total_images', 0)}")
    print(f"Methods Tried: {simple_result.get('methods_tried', [])}")
    
    # Show available methods
    methods = simple_worker.get_supported_methods()
    available_methods = methods.get('available_methods', {})
    print(f"Available Methods:")
    for method, available in available_methods.items():
        status = "✅" if available else "❌"
        print(f"  {status} {method}")
    
    # Test 2: Enhanced Mode
    print("\n📋 Test 2: Enhanced Mode")
    print("-" * 40)
    
    enhanced_worker = ImageProcessingWorker(ImageProcessingMode.ENHANCED)
    print(f"Worker Mode: {enhanced_worker.mode.value}")
    
    enhanced_result = enhanced_worker._extract_images_enhanced({
        'file_path': 'test_document.pdf',
        'page_numbers': [0, 1]
    })
    
    print(f"Success: {enhanced_result.get('success', False)}")
    print(f"Extraction Method: {enhanced_result.get('extraction_method', 'Unknown')}")
    print(f"Confidence Score: {enhanced_result.get('confidence_score', 0.0):.2f}")
    print(f"Total Images: {enhanced_result.get('total_images', 0)}")
    
    # Show enhanced features (if images were found)
    if enhanced_result.get('images'):
        print(f"Enhanced features applied:")
        for i, image in enumerate(enhanced_result['images'][:2]):  # Show first 2 images
            metadata = image.get('metadata', {})
            print(f"  Image {i+1}:")
            print(f"    Enhanced Processing: {metadata.get('enhanced_processing', False)}")
            print(f"    OCR Available: {metadata.get('ocr_available', False)}")
            print(f"    Classification Applied: {metadata.get('classification_applied', False)}")
            
            # Show classification if available
            classification = image.get('classification', {})
            if classification:
                print(f"    Image Type: {classification.get('image_type', 'Unknown')}")
                print(f"    Size Category: {classification.get('size_category', 'Unknown')}")
                print(f"    Aspect Ratio: {classification.get('aspect_ratio', 0.0):.2f}")
            
            # Show quality metrics if available
            quality_metrics = image.get('quality_metrics', {})
            if quality_metrics:
                print(f"    Overall Quality: {quality_metrics.get('overall_quality', 0.0):.2f}")
                print(f"    Resolution Score: {quality_metrics.get('resolution_score', 0.0):.2f}")
    
    # Test 3: Advanced Mode
    print("\n📋 Test 3: Advanced Mode")
    print("-" * 40)
    
    advanced_worker = ImageProcessingWorker(ImageProcessingMode.ADVANCED)
    print(f"Worker Mode: {advanced_worker.mode.value}")
    
    advanced_result = advanced_worker._extract_images_advanced({
        'file_path': 'test_document.pdf',
        'page_numbers': [0, 1]
    })
    
    print(f"Success: {advanced_result.get('success', False)}")
    print(f"Extraction Method: {advanced_result.get('extraction_method', 'Unknown')}")
    print(f"Confidence Score: {advanced_result.get('confidence_score', 0.0):.2f}")
    print(f"Total Images: {advanced_result.get('total_images', 0)}")
    
    # Show advanced features
    advanced_features = advanced_result.get('advanced_features', {})
    if advanced_features:
        print(f"\n🤖 AI Analysis:")
        ai_analysis = advanced_features.get('ai_analysis', {})
        
        object_detection = ai_analysis.get('object_detection', {})
        print(f"  Object Detection:")
        print(f"    Objects Detected: {object_detection.get('objects_detected', 0)}")
        print(f"    Primary Objects: {', '.join(object_detection.get('primary_objects', []))}")
        print(f"    Object Confidence: {object_detection.get('object_confidence', 0.0):.2f}")
        
        image_segmentation = ai_analysis.get('image_segmentation', {})
        print(f"  Image Segmentation:")
        print(f"    Segments Detected: {image_segmentation.get('segments_detected', 0)}")
        print(f"    Segment Types: {', '.join(image_segmentation.get('segment_types', []))}")
        print(f"    Segmentation Confidence: {image_segmentation.get('segmentation_confidence', 0.0):.2f}")
        
        content_analysis = ai_analysis.get('content_analysis', {})
        print(f"  Content Analysis:")
        print(f"    Content Type: {content_analysis.get('content_type', 'Unknown')}")
        print(f"    Text Content: {content_analysis.get('text_content', False)}")
        print(f"    Visual Content: {content_analysis.get('visual_content', False)}")
        print(f"    Tables/Charts: {content_analysis.get('tables_charts', False)}")
        print(f"    Readability Score: {content_analysis.get('readability_score', 0.0):.2f}")
        
        semantic_understanding = ai_analysis.get('semantic_understanding', {})
        print(f"  Semantic Understanding:")
        print(f"    Semantic Topics: {', '.join(semantic_understanding.get('semantic_topics', []))}")
        print(f"    Document Context: {semantic_understanding.get('document_context', 'Unknown')}")
        print(f"    Semantic Confidence: {semantic_understanding.get('semantic_confidence', 0.0):.2f}")
        print(f"    Key Concepts: {', '.join(semantic_understanding.get('key_concepts', []))}")
        
        print(f"\n📈 ML Predictions:")
        ml_predictions = advanced_features.get('ml_predictions', {})
        print(f"  Image Quality Score: {ml_predictions.get('image_quality_score', 0.0):.2f}")
        print(f"  Content Relevance: {ml_predictions.get('content_relevance', 0.0):.2f}")
        print(f"  Processing Complexity: {ml_predictions.get('processing_complexity', 0.0):.2f}")
        
        recommended_actions = ml_predictions.get('recommended_actions', [])
        if recommended_actions:
            print(f"  Recommended Actions:")
            for action in recommended_actions:
                print(f"    - {action}")
        
        print(f"\n📊 Advanced Metrics:")
        advanced_metrics = advanced_features.get('advanced_metrics', {})
        print(f"  Feature Extraction Score: {advanced_metrics.get('feature_extraction_score', 0.0):.2f}")
        print(f"  Semantic Accuracy: {advanced_metrics.get('semantic_accuracy', 0.0):.2f}")
        print(f"  Processing Efficiency: {advanced_metrics.get('processing_efficiency', 0.0):.2f}")
        print(f"  Content Coverage: {advanced_metrics.get('content_coverage', 0.0):.2f}")
    
    # Test 4: Mode Comparison
    print("\n📋 Test 4: Mode Comparison")
    print("-" * 40)
    
    test_scenarios = [
        {
            'name': 'Document with Charts',
            'file_path': 'financial_report.pdf',
            'page_numbers': [0, 1, 2]
        },
        {
            'name': 'Technical Manual',
            'file_path': 'technical_manual.pdf',
            'page_numbers': [0, 1]
        },
        {
            'name': 'Simple Document',
            'file_path': 'simple_doc.pdf',
            'page_numbers': [0]
        }
    ]
    
    print("Comparing image processing across modes:")
    print(f"{'Document':<20} {'Simple':<15} {'Enhanced':<15} {'Advanced':<15}")
    print("-" * 70)
    
    for scenario in test_scenarios:
        name = scenario['name']
        file_path = scenario['file_path']
        page_numbers = scenario['page_numbers']
        
        # Simple extraction
        simple_result = simple_worker._extract_images_simple({
            'file_path': file_path,
            'page_numbers': page_numbers
        })
        simple_images = simple_result.get('total_images', 0)
        
        # Enhanced extraction
        enhanced_result = enhanced_worker._extract_images_enhanced({
            'file_path': file_path,
            'page_numbers': page_numbers
        })
        enhanced_images = enhanced_result.get('total_images', 0)
        
        # Advanced extraction
        advanced_result = advanced_worker._extract_images_advanced({
            'file_path': file_path,
            'page_numbers': page_numbers
        })
        advanced_images = advanced_result.get('total_images', 0)
        
        print(f"{name:<20} {simple_images:<15} {enhanced_images:<15} {advanced_images:<15}")
    
    # Test 5: Image Analysis (Advanced Mode Only)
    print("\n📋 Test 5: Image Analysis")
    print("-" * 40)
    
    analysis_result = advanced_worker._analyze_images_task({
        'file_path': 'test_document.pdf',
        'page_numbers': [0, 1]
    })
    
    if analysis_result.get('success', False):
        analysis = analysis_result.get('analysis_result', {})
        print(f"Image Quality: {analysis.get('image_quality', 'Unknown')}")
        print(f"Content Analysis Score: {analysis.get('content_analysis_score', 0.0):.2f}")
        
        improvements = analysis.get('recommended_improvements', [])
        if improvements:
            print(f"Recommended Improvements:")
            for improvement in improvements:
                print(f"  - {improvement}")
        
        patterns = analysis.get('analysis_patterns', [])
        if patterns:
            print(f"Analysis Patterns: {', '.join(patterns)}")
    else:
        print(f"Analysis failed: {analysis_result.get('error', 'Unknown error')}")
    
    # Test 6: Performance Metrics
    print("\n📋 Test 6: Performance Metrics")
    print("-" * 40)
    
    for mode in [ImageProcessingMode.SIMPLE, ImageProcessingMode.ENHANCED, ImageProcessingMode.ADVANCED]:
        worker = ImageProcessingWorker(mode)
        metrics = worker.get_performance_metrics()
        methods = worker.get_supported_methods()
        
        print(f"\n{mode.value.upper()} Mode:")
        print(f"  Worker: {metrics['worker_name']}")
        print(f"  Type: {metrics['worker_type']}")
        print(f"  Mode: {metrics['mode']}")
        print(f"  Cache Size: {metrics['cache_size']}")
        print(f"  Available Methods: {methods['available_methods']}")
        
        # Show specific method availability
        available_methods = methods.get('available_methods', {})
        print(f"  Method Details:")
        for method, available in available_methods.items():
            status = "Available" if available else "Not Available"
            print(f"    - {method}: {status}")
    
    # Test 7: OCR Simulation (if tesseract available)
    print("\n📋 Test 7: OCR Capabilities")
    print("-" * 40)
    
    methods = enhanced_worker.get_supported_methods()
    available_methods = methods.get('available_methods', {})
    
    if available_methods.get('tesseract', False):
        print("✅ Tesseract OCR is available")
        print("  - Text extraction from images")
        print("  - Confidence scoring")
        print("  - Word and line counting")
    else:
        print("❌ Tesseract OCR is not available")
        print("  - Install with: pip install pytesseract")
        print("  - Also requires system Tesseract installation")
    
    if available_methods.get('opencv', False):
        print("✅ OpenCV is available")
        print("  - Image processing and analysis")
        print("  - Object detection capabilities")
    else:
        print("❌ OpenCV is not available")
        print("  - Install with: pip install opencv-python")
    
    if available_methods.get('tensorflow', False):
        print("✅ TensorFlow is available")
        print("  - Deep learning models")
        print("  - Advanced image analysis")
    else:
        print("❌ TensorFlow is not available")
        print("  - Install with: pip install tensorflow")
    
    if available_methods.get('pytorch', False):
        print("✅ PyTorch is available")
        print("  - Deep learning models")
        print("  - Advanced image analysis")
    else:
        print("❌ PyTorch is not available")
        print("  - Install with: pip install torch")
    
    print("\n✅ ImageProcessingWorker testing completed!")
    print("\n🎯 Key Benefits of Unified Image Processing Worker:")
    print("  ✅ Progressive complexity: Simple → Enhanced → Advanced")
    print("  ✅ OCR and image classification in enhanced mode")
    print("  ✅ AI-powered analysis in advanced mode")
    print("  ✅ Object detection and semantic understanding")
    print("  ✅ Image quality assessment and recommendations")
    print("  ✅ Consistent interface across all modes")


if __name__ == "__main__":
    test_image_processing_worker()


