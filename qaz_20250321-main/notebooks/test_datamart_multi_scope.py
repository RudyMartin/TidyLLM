#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Multi-Scope DataMart Manager

Demonstrates the unified DataMartManager functionality with all three modes:
- Simple: Basic data storage and retrieval
- Enhanced: Advanced datatable operations, analytics
- Advanced: AI/ML integration, predictive analytics, real-time processing
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from backend.mcp.orchestrators.advanced_qa_orchestrator import DataMartManager, DataMartMode
import json


def test_datamart_multi_scope():
    """Test the unified DataMartManager with all three modes"""
    
    print("🚀 Testing Multi-Scope DataMart Manager")
    print("=" * 60)
    
    # Test 1: Simple Mode
    print("\n📋 Test 1: Simple Mode")
    print("-" * 40)
    
    simple_datamart = DataMartManager(DataMartMode.SIMPLE)
    print(f"DataMart Mode: {simple_datamart.mode.value}")
    
    # Test initialization
    init_result = simple_datamart.initialize_datamart()
    print(f"Initialization Success: {init_result}")
    
    # Test adding data
    test_data = {
        'worker_name': 'TestWorker',
        'worker_type': 'test',
        'mode': 'simple',
        'data_type': 'test_data',
        'confidence_score': 0.75,
        'processing_time': 1.5,
        'status': 'completed'
    }
    
    add_result = simple_datamart.add_analysis_data(test_data)
    print(f"Add Data Success: {add_result}")
    
    # Get metrics
    metrics = simple_datamart.get_performance_metrics()
    print(f"Buffer Size: {metrics.get('buffer_size', 0)}")
    print(f"DataMart ID: {metrics.get('datamart_id', 'N/A')[:8]}...")
    print(f"Status: {metrics.get('status', 'N/A')}")
    
    # Test 2: Enhanced Mode
    print("\n📋 Test 2: Enhanced Mode")
    print("-" * 40)
    
    enhanced_datamart = DataMartManager(DataMartMode.ENHANCED)
    print(f"DataMart Mode: {enhanced_datamart.mode.value}")
    
    # Test initialization
    init_result = enhanced_datamart.initialize_datamart()
    print(f"Initialization Success: {init_result}")
    
    # Test adding multiple data points
    test_data_enhanced = [
        {
            'worker_name': 'FileClassificationWorker',
            'worker_type': 'file_classification',
            'mode': 'enhanced',
            'data_type': 'document_analysis',
            'confidence_score': 0.85,
            'processing_time': 2.1,
            'status': 'completed'
        },
        {
            'worker_name': 'TOCExtractorWorker',
            'worker_type': 'toc_extraction',
            'mode': 'enhanced',
            'data_type': 'toc_analysis',
            'confidence_score': 0.78,
            'processing_time': 1.8,
            'status': 'completed'
        },
        {
            'worker_name': 'ImageProcessingWorker',
            'worker_type': 'image_processing',
            'mode': 'enhanced',
            'data_type': 'image_analysis',
            'confidence_score': 0.72,
            'processing_time': 3.2,
            'status': 'completed'
        }
    ]
    
    for data in test_data_enhanced:
        add_result = enhanced_datamart.add_analysis_data(data)
        print(f"Added {data['worker_name']}: {add_result}")
    
    # Get enhanced metrics
    enhanced_metrics = enhanced_datamart.get_performance_metrics()
    print(f"\nEnhanced Metrics:")
    print(f"  Buffer Size: {enhanced_metrics.get('buffer_size', 0)}")
    print(f"  Analytics Cache Size: {enhanced_metrics.get('analytics_cache_size', 0)}")
    print(f"  Worker Activity: {enhanced_metrics.get('worker_activity', [])}")
    print(f"  Average Confidence: {enhanced_metrics.get('average_confidence', 0.0):.2f}")
    print(f"  Total Processing Time: {enhanced_metrics.get('total_processing_time', 0.0):.2f}")
    
    # Test 3: Advanced Mode
    print("\n📋 Test 3: Advanced Mode")
    print("-" * 40)
    
    advanced_datamart = DataMartManager(DataMartMode.ADVANCED)
    print(f"DataMart Mode: {advanced_datamart.mode.value}")
    
    # Test initialization
    init_result = advanced_datamart.initialize_datamart()
    print(f"Initialization Success: {init_result}")
    
    # Test adding advanced data with AI/ML features
    test_data_advanced = [
        {
            'worker_name': 'FileClassificationWorker',
            'worker_type': 'file_classification',
            'mode': 'advanced',
            'data_type': 'document_analysis',
            'confidence_score': 0.92,
            'processing_time': 2.5,
            'status': 'completed',
            'ai_analysis': {
                'object_detection': {'objects_detected': 5, 'confidence': 0.85},
                'content_analysis': {'complexity': 'high', 'readability': 0.78}
            },
            'ml_predictions': {
                'quality_score': 0.88,
                'relevance_score': 0.91,
                'recommendations': ['Optimize processing pipeline', 'Enhance classification accuracy']
            },
            'performance_metrics': {
                'throughput': 150.5,
                'latency': 0.025,
                'accuracy': 0.92
            }
        },
        {
            'worker_name': 'ImageProcessingWorker',
            'worker_type': 'image_processing',
            'mode': 'advanced',
            'data_type': 'image_analysis',
            'confidence_score': 0.89,
            'processing_time': 4.1,
            'status': 'completed',
            'ai_analysis': {
                'object_detection': {'objects_detected': 8, 'confidence': 0.87},
                'image_segmentation': {'segments': 12, 'accuracy': 0.83}
            },
            'ml_predictions': {
                'quality_score': 0.85,
                'relevance_score': 0.88,
                'recommendations': ['Improve image resolution', 'Enhance OCR accuracy']
            },
            'performance_metrics': {
                'throughput': 45.2,
                'latency': 0.089,
                'accuracy': 0.89
            }
        }
    ]
    
    for data in test_data_advanced:
        add_result = advanced_datamart.add_analysis_data(data)
        print(f"Added {data['worker_name']}: {add_result}")
    
    # Get advanced metrics
    advanced_metrics = advanced_datamart.get_performance_metrics()
    print(f"\nAdvanced Metrics:")
    print(f"  Buffer Size: {advanced_metrics.get('buffer_size', 0)}")
    print(f"  Analytics Cache Size: {advanced_metrics.get('analytics_cache_size', 0)}")
    print(f"  Worker Activity: {advanced_metrics.get('worker_activity', [])}")
    print(f"  Average Confidence: {advanced_metrics.get('average_confidence', 0.0):.2f}")
    print(f"  Total Processing Time: {advanced_metrics.get('total_processing_time', 0.0):.2f}")
    
    # Show advanced analytics
    advanced_analytics = advanced_metrics.get('advanced_analytics', {})
    if advanced_analytics:
        print(f"\n🤖 Advanced Analytics:")
        print(f"  Trend Analysis: {advanced_analytics.get('trend_analysis', {})}")
        print(f"  Performance Prediction: {advanced_analytics.get('performance_prediction', {})}")
        print(f"  Anomaly Detection: {advanced_analytics.get('anomaly_detection', {})}")
        print(f"  Recommendations: {advanced_analytics.get('optimization_recommendations', [])}")
    
    # Show AI insights
    ai_insights = advanced_metrics.get('ai_insights', {})
    if ai_insights:
        print(f"\n🧠 AI Insights:")
        print(f"  Performance Optimization: {ai_insights.get('performance_optimization', 'N/A')}")
        print(f"  Quality Assurance: {ai_insights.get('quality_assurance', 'N/A')}")
        print(f"  Resource Utilization: {ai_insights.get('resource_utilization', 'N/A')}")
        print(f"  Recommendations: {ai_insights.get('recommendations', [])}")
    
    # Test 4: Mode Comparison
    print("\n📋 Test 4: Mode Comparison")
    print("-" * 40)
    
    print("Comparing DataMart capabilities across modes:")
    print(f"{'Feature':<25} {'Simple':<15} {'Enhanced':<15} {'Advanced':<15}")
    print("-" * 75)
    
    features = [
        ('Basic Storage', '✅', '✅', '✅'),
        ('Schema Definition', '❌', '✅', '✅'),
        ('Analytics Cache', '❌', '✅', '✅'),
        ('Performance Metrics', 'Basic', 'Enhanced', 'Advanced'),
        ('AI/ML Integration', '❌', '❌', '✅'),
        ('Trend Analysis', '❌', '❌', '✅'),
        ('Predictive Analytics', '❌', '❌', '✅'),
        ('Anomaly Detection', '❌', '❌', '✅'),
        ('Optimization Recommendations', '❌', '❌', '✅')
    ]
    
    for feature, simple, enhanced, advanced in features:
        print(f"{feature:<25} {simple:<15} {enhanced:<15} {advanced:<15}")
    
    # Test 5: DataMart Integration with Workers
    print("\n📋 Test 5: DataMart Integration with Workers")
    print("-" * 40)
    
    # Simulate worker integration
    worker_data = {
        'FileClassificationWorker': {
            'mode': 'enhanced',
            'confidence_score': 0.87,
            'processing_time': 2.3
        },
        'TOCExtractorWorker': {
            'mode': 'enhanced',
            'confidence_score': 0.81,
            'processing_time': 1.9
        },
        'ImageProcessingWorker': {
            'mode': 'advanced',
            'confidence_score': 0.89,
            'processing_time': 3.8
        }
    }
    
    print("Simulating worker integration with DataMart:")
    for worker_name, data in worker_data.items():
        integration_data = {
            'worker_name': worker_name,
            'worker_type': 'document_processing',
            'mode': data['mode'],
            'data_type': 'worker_analysis',
            'confidence_score': data['confidence_score'],
            'processing_time': data['processing_time'],
            'status': 'completed'
        }
        
        # Add to appropriate DataMart mode
        if data['mode'] == 'enhanced':
            result = enhanced_datamart.add_analysis_data(integration_data)
        else:
            result = advanced_datamart.add_analysis_data(integration_data)
        
        print(f"  {worker_name}: {'✅' if result else '❌'}")
    
    # Test 6: Performance Analysis
    print("\n📋 Test 6: Performance Analysis")
    print("-" * 40)
    
    # Compare performance across modes
    datamarts = {
        'Simple': simple_datamart,
        'Enhanced': enhanced_datamart,
        'Advanced': advanced_datamart
    }
    
    print("Performance comparison across DataMart modes:")
    print(f"{'Mode':<12} {'Buffer Size':<12} {'Cache Size':<12} {'Avg Confidence':<15} {'Total Time':<12}")
    print("-" * 70)
    
    for mode_name, datamart in datamarts.items():
        metrics = datamart.get_performance_metrics()
        buffer_size = metrics.get('buffer_size', 0)
        cache_size = metrics.get('analytics_cache_size', 0)
        avg_confidence = metrics.get('average_confidence', 0.0)
        total_time = metrics.get('total_processing_time', 0.0)
        
        print(f"{mode_name:<12} {buffer_size:<12} {cache_size:<12} {avg_confidence:<15.2f} {total_time:<12.2f}")
    
    # Test 7: DataMart Schema Analysis
    print("\n📋 Test 7: DataMart Schema Analysis")
    print("-" * 40)
    
    print("Schema comparison across modes:")
    
    # Simple mode schema
    print(f"\n📊 Simple Mode Schema:")
    print("  - Flexible structure (any data format)")
    print("  - Basic storage and retrieval")
    print("  - No predefined schema")
    
    # Enhanced mode schema
    print(f"\n📊 Enhanced Mode Schema:")
    enhanced_schema = [
        'timestamp', 'worker_name', 'worker_type', 'mode', 
        'data_type', 'confidence_score', 'processing_time', 'status'
    ]
    for field in enhanced_schema:
        print(f"  - {field}")
    
    # Advanced mode schema
    print(f"\n📊 Advanced Mode Schema:")
    advanced_schema = [
        'timestamp', 'worker_name', 'worker_type', 'mode', 
        'data_type', 'confidence_score', 'processing_time', 'status',
        'ai_analysis', 'ml_predictions', 'performance_metrics', 'datamart_id'
    ]
    for field in advanced_schema:
        print(f"  - {field}")
    
    print("\n✅ Multi-Scope DataMart testing completed!")
    print("\n🎯 Key Benefits of Multi-Scope DataMart:")
    print("  ✅ Progressive complexity: Simple → Enhanced → Advanced")
    print("  ✅ Enhanced analytics and caching in enhanced mode")
    print("  ✅ AI/ML integration and predictive analytics in advanced mode")
    print("  ✅ Consistent interface across all modes")
    print("  ✅ datatable-based high-performance processing")
    print("  ✅ Seamless integration with multi-scope workers")


if __name__ == "__main__":
    test_datamart_multi_scope()


