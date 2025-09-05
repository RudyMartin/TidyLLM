#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test TOC Extractor Worker

Demonstrates the unified TOCExtractorWorker functionality with all three modes:
- Simple: Basic regex-based TOC extraction
- Enhanced: PDF-aware extraction with page numbers, hierarchy detection
- Advanced: AI-powered structure analysis, semantic section grouping
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from backend.mcp.workers.toc_extractor_worker import TOCExtractorWorker, TOCMode
import json


def test_toc_extractor_worker():
    """Test the unified TOCExtractorWorker with all three modes"""
    
    print("🚀 Testing TOCExtractorWorker")
    print("=" * 60)
    
    # Test content for all modes
    test_content = """
    Table of Contents
    
    1. Introduction 1
    1.1 Background 2
    1.2 Objectives 3
    1.3 Scope and Limitations 4
    2. Literature Review 5
    2.1 Previous Research 6
    2.2 Theoretical Framework 8
    2.3 Research Gaps 10
    3. Methodology 12
    3.1 Research Design 13
    3.2 Data Collection Methods 15
    3.3 Analysis Techniques 17
    4. Results and Analysis 20
    4.1 Descriptive Statistics 21
    4.2 Hypothesis Testing 23
    4.3 Model Validation 25
    5. Discussion 28
    5.1 Interpretation of Results 29
    5.2 Implications for Practice 31
    5.3 Limitations and Future Research 33
    6. Conclusion 35
    6.1 Summary of Findings 36
    6.2 Recommendations 38
    References 40
    Appendices 45
    A. Survey Instrument 46
    B. Statistical Tables 48
    C. Additional Analysis 50
    """
    
    # Test 1: Simple Mode
    print("\n📋 Test 1: Simple Mode")
    print("-" * 40)
    
    simple_worker = TOCExtractorWorker(TOCMode.SIMPLE)
    print(f"Worker Mode: {simple_worker.mode.value}")
    
    simple_result = simple_worker._extract_toc_simple({
        'content': test_content,
        'filename': 'test_document.txt'
    })
    
    print(f"Success: {simple_result.get('success', False)}")
    print(f"Extraction Method: {simple_result.get('extraction_method', 'Unknown')}")
    print(f"Confidence Score: {simple_result.get('confidence_score', 0.0):.2f}")
    print(f"Total Entries: {simple_result.get('total_entries', 0)}")
    
    # Show first few entries
    toc_structure = simple_result.get('toc_structure', {})
    entries = toc_structure.get('entries', [])
    print(f"First 3 entries:")
    for i, entry in enumerate(entries[:3]):
        print(f"  {i+1}. {entry.get('title', 'Unknown')} (Level: {entry.get('level', 0)})")
    
    # Test 2: Enhanced Mode
    print("\n📋 Test 2: Enhanced Mode")
    print("-" * 40)
    
    enhanced_worker = TOCExtractorWorker(TOCMode.ENHANCED)
    print(f"Worker Mode: {enhanced_worker.mode.value}")
    
    enhanced_result = enhanced_worker._extract_toc_enhanced({
        'content': test_content,
        'filename': 'test_document.txt'
    })
    
    print(f"Success: {enhanced_result.get('success', False)}")
    print(f"Extraction Method: {enhanced_result.get('extraction_method', 'Unknown')}")
    print(f"Confidence Score: {enhanced_result.get('confidence_score', 0.0):.2f}")
    print(f"Total Entries: {enhanced_result.get('total_entries', 0)}")
    
    # Show enhanced features
    toc_structure = enhanced_result.get('toc_structure', {})
    entries = toc_structure.get('entries', [])
    print(f"Enhanced entries with metadata:")
    for i, entry in enumerate(entries[:3]):
        metadata = entry.get('metadata', {})
        print(f"  {i+1}. {entry.get('title', 'Unknown')}")
        print(f"     Level: {entry.get('level', 0)}, Page: {entry.get('page_number', 'N/A')}")
        print(f"     Category: {metadata.get('semantic_category', 'Unknown')}")
        print(f"     Importance: {metadata.get('importance_score', 0.0):.2f}")
    
    # Test 3: Advanced Mode
    print("\n📋 Test 3: Advanced Mode")
    print("-" * 40)
    
    advanced_worker = TOCExtractorWorker(TOCMode.ADVANCED)
    print(f"Worker Mode: {advanced_worker.mode.value}")
    
    advanced_result = advanced_worker._extract_toc_advanced({
        'content': test_content,
        'filename': 'test_document.txt'
    })
    
    print(f"Success: {advanced_result.get('success', False)}")
    print(f"Extraction Method: {advanced_result.get('extraction_method', 'Unknown')}")
    print(f"Confidence Score: {advanced_result.get('confidence_score', 0.0):.2f}")
    print(f"Total Entries: {advanced_result.get('total_entries', 0)}")
    
    # Show advanced features
    advanced_features = advanced_result.get('advanced_features', {})
    print(f"\n🤖 AI Analysis:")
    ai_analysis = advanced_features.get('ai_analysis', {})
    
    semantic_clustering = ai_analysis.get('semantic_clustering', {})
    clusters = semantic_clustering.get('clusters', [])
    print(f"  Semantic Clusters ({len(clusters)}):")
    for cluster in clusters:
        print(f"    - {cluster.get('name', 'Unknown')}: {', '.join(cluster.get('sections', []))}")
    
    topic_modeling = ai_analysis.get('topic_modeling', {})
    topics = topic_modeling.get('topics', [])
    print(f"  Topic Modeling ({len(topics)} topics):")
    for topic in topics:
        print(f"    - {topic.get('topic', 'Unknown')}: {topic.get('weight', 0.0):.2f}")
    
    section_importance = ai_analysis.get('section_importance', {})
    critical_sections = section_importance.get('critical_sections', [])
    print(f"  Critical Sections: {', '.join(critical_sections)}")
    
    print(f"\n📈 ML Predictions:")
    ml_predictions = advanced_features.get('ml_predictions', {})
    print(f"  Completeness Score: {ml_predictions.get('completeness_score', 0.0):.2f}")
    print(f"  Structure Quality: {ml_predictions.get('structure_quality', 0.0):.2f}")
    
    missing_sections = ml_predictions.get('missing_sections', [])
    if missing_sections:
        print(f"  Missing Sections: {', '.join(missing_sections)}")
    
    recommended_improvements = ml_predictions.get('recommended_improvements', [])
    if recommended_improvements:
        print(f"  Recommended Improvements:")
        for improvement in recommended_improvements:
            print(f"    - {improvement}")
    
    print(f"\n📊 Advanced Metrics:")
    advanced_metrics = advanced_features.get('advanced_metrics', {})
    print(f"  Hierarchy Depth: {advanced_metrics.get('hierarchy_depth', 0)}")
    print(f"  Section Balance: {advanced_metrics.get('section_balance', 0.0):.2f}")
    print(f"  Navigational Complexity: {advanced_metrics.get('navigational_complexity', 0.0):.2f}")
    print(f"  Content Coverage: {advanced_metrics.get('content_coverage', 0.0):.2f}")
    
    # Test 4: Mode Comparison
    print("\n📋 Test 4: Mode Comparison")
    print("-" * 40)
    
    test_documents = [
        {
            'name': 'Academic Paper',
            'content': """
            Table of Contents
            
            1. Abstract 1
            2. Introduction 2
            3. Literature Review 5
            4. Methodology 10
            5. Results 15
            6. Discussion 20
            7. Conclusion 25
            8. References 30
            """
        },
        {
            'name': 'Technical Report',
            'content': """
            Contents
            
            Executive Summary 1
            Chapter 1: Background 3
            Chapter 2: Technical Approach 8
            Chapter 3: Implementation 15
            Chapter 4: Results 22
            Chapter 5: Conclusions 28
            Appendix A: Data Tables 32
            Appendix B: Code Examples 35
            """
        },
        {
            'name': 'Simple Document',
            'content': """
            Document Outline
            
            Overview 1
            Details 3
            Summary 5
            """
        }
    ]
    
    print("Comparing TOC extraction across modes:")
    print(f"{'Document':<20} {'Simple':<15} {'Enhanced':<15} {'Advanced':<15}")
    print("-" * 70)
    
    for doc in test_documents:
        name = doc['name']
        content = doc['content']
        
        # Simple extraction
        simple_result = simple_worker._extract_toc_simple({
            'content': content,
            'filename': f'{name.lower().replace(" ", "_")}.txt'
        })
        simple_entries = simple_result.get('total_entries', 0)
        
        # Enhanced extraction
        enhanced_result = enhanced_worker._extract_toc_enhanced({
            'content': content,
            'filename': f'{name.lower().replace(" ", "_")}.txt'
        })
        enhanced_entries = enhanced_result.get('total_entries', 0)
        
        # Advanced extraction
        advanced_result = advanced_worker._extract_toc_advanced({
            'content': content,
            'filename': f'{name.lower().replace(" ", "_")}.txt'
        })
        advanced_entries = advanced_result.get('total_entries', 0)
        
        print(f"{name:<20} {simple_entries:<15} {enhanced_entries:<15} {advanced_entries:<15}")
    
    # Test 5: Document Structure Analysis (Advanced Mode Only)
    print("\n📋 Test 5: Document Structure Analysis")
    print("-" * 40)
    
    structure_result = advanced_worker._analyze_document_structure_task({
        'content': test_content,
        'filename': 'test_document.txt'
    })
    
    if structure_result.get('success', False):
        analysis = structure_result.get('analysis_result', {})
        print(f"Document Complexity: {analysis.get('document_complexity', 'Unknown')}")
        print(f"Structure Quality: {analysis.get('structure_quality', 0.0):.2f}")
        
        improvements = analysis.get('recommended_improvements', [])
        if improvements:
            print(f"Recommended Improvements:")
            for improvement in improvements:
                print(f"  - {improvement}")
        
        patterns = analysis.get('structural_patterns', [])
        if patterns:
            print(f"Structural Patterns: {', '.join(patterns)}")
    else:
        print(f"Analysis failed: {structure_result.get('error', 'Unknown error')}")
    
    # Test 6: Performance Metrics
    print("\n📋 Test 6: Performance Metrics")
    print("-" * 40)
    
    for mode in [TOCMode.SIMPLE, TOCMode.ENHANCED, TOCMode.ADVANCED]:
        worker = TOCExtractorWorker(mode)
        metrics = worker.get_performance_metrics()
        methods = worker.get_supported_methods()
        
        print(f"\n{mode.value.upper()} Mode:")
        print(f"  Worker: {metrics['worker_name']}")
        print(f"  Type: {metrics['worker_type']}")
        print(f"  Mode: {metrics['mode']}")
        print(f"  Cache Size: {metrics['cache_size']}")
        print(f"  Available Methods: {methods['available_methods']}")
        print(f"  Pattern Sets: {methods['patterns']}")
    
    print("\n✅ TOCExtractorWorker testing completed!")
    print("\n🎯 Key Benefits of Unified TOC Worker:")
    print("  ✅ Progressive complexity: Simple → Enhanced → Advanced")
    print("  ✅ PDF-aware extraction in enhanced mode")
    print("  ✅ AI-powered analysis in advanced mode")
    print("  ✅ Semantic clustering and topic modeling")
    print("  ✅ Document structure analysis")
    print("  ✅ Consistent interface across all modes")


if __name__ == "__main__":
    test_toc_extractor_worker()


