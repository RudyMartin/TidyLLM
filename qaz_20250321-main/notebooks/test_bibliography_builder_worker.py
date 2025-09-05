#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Bibliography Builder Worker

Demonstrates the unified BibliographyBuilderWorker functionality with all three modes:
- Simple: Basic regex-based citation extraction
- Enhanced: Multi-format citation parsing (APA, MLA, IEEE)
- Advanced: AI-powered analysis, citation network analysis, bibliometric analysis
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from backend.mcp.workers.bibliography_builder_worker import BibliographyBuilderWorker, BibliographyMode
import json


def test_bibliography_builder_worker():
    """Test the unified BibliographyBuilderWorker with all three modes"""
    
    print("🚀 Testing BibliographyBuilderWorker")
    print("=" * 60)
    
    # Test content for all modes
    test_content = """
    References
    
    1. Smith, J. (2020). Machine Learning Applications in Finance. Journal of Artificial Intelligence, 15(3), 245-260. DOI: 10.1234/ai.2020.001
    2. Johnson, A. (2019). Deep Learning Methods for Risk Assessment. Conference on Machine Learning, 45-52. arXiv: 1901.12345
    3. Brown, M. (2021). Statistical Analysis of Market Data. Statistics Journal, 28(4), 123-135.
    4. Wilson, R. (2018). Data Science Fundamentals. Data Science Review, 12(2), 78-89.
    5. Davis, L. (2022). Neural Networks in Trading. Neural Computing, 35(1), 15-28.
    6. Anderson, K. (2023). "Modern Portfolio Theory Revisited." Financial Analysis Quarterly, 45(2), 67-82.
    7. Taylor, S. (2021). Quantitative Methods in Risk Management. Risk Management Journal, 18(3), 156-170.
    8. Martinez, P. (2020). Blockchain Applications in Finance. Blockchain Technology Review, 8(1), 23-35.
    """
    
    # Test 1: Simple Mode
    print("\n📋 Test 1: Simple Mode")
    print("-" * 40)
    
    simple_worker = BibliographyBuilderWorker(BibliographyMode.SIMPLE)
    print(f"Worker Mode: {simple_worker.mode.value}")
    
    simple_result = simple_worker._extract_bibliography_simple({
        'content': test_content,
        'filename': 'test_document.txt'
    })
    
    print(f"Success: {simple_result.get('success', False)}")
    print(f"Extraction Method: {simple_result.get('extraction_method', 'Unknown')}")
    print(f"Confidence Score: {simple_result.get('confidence_score', 0.0):.2f}")
    print(f"Total Citations: {simple_result.get('total_citations', 0)}")
    
    # Show first few citations
    bibliography_structure = simple_result.get('bibliography_structure', {})
    citations = bibliography_structure.get('citations', [])
    print(f"First 3 citations:")
    for i, citation in enumerate(citations[:3]):
        print(f"  {i+1}. {citation.get('authors', ['Unknown'])[0]} ({citation.get('year', 'Unknown')})")
        print(f"     Title: {citation.get('title', 'Unknown')[:50]}...")
    
    # Test 2: Enhanced Mode
    print("\n📋 Test 2: Enhanced Mode")
    print("-" * 40)
    
    enhanced_worker = BibliographyBuilderWorker(BibliographyMode.ENHANCED)
    print(f"Worker Mode: {enhanced_worker.mode.value}")
    
    enhanced_result = enhanced_worker._extract_bibliography_enhanced({
        'content': test_content,
        'filename': 'test_document.txt'
    })
    
    print(f"Success: {enhanced_result.get('success', False)}")
    print(f"Extraction Method: {enhanced_result.get('extraction_method', 'Unknown')}")
    print(f"Confidence Score: {enhanced_result.get('confidence_score', 0.0):.2f}")
    print(f"Total Citations: {enhanced_result.get('total_citations', 0)}")
    
    # Show enhanced features
    bibliography_structure = enhanced_result.get('bibliography_structure', {})
    citations = bibliography_structure.get('citations', [])
    print(f"Enhanced citations with metadata:")
    for i, citation in enumerate(citations[:3]):
        metadata = citation.get('metadata', {})
        print(f"  {i+1}. {citation.get('authors', ['Unknown'])[0]} ({citation.get('year', 'Unknown')})")
        print(f"     Type: {citation.get('citation_type', 'Unknown')}")
        print(f"     Quality: {metadata.get('extraction_quality', 0.0):.2f}")
        print(f"     Completeness: {metadata.get('completeness_score', 0.0):.2f}")
        if citation.get('doi'):
            print(f"     DOI: {citation.get('doi')}")
    
    # Test 3: Advanced Mode
    print("\n📋 Test 3: Advanced Mode")
    print("-" * 40)
    
    advanced_worker = BibliographyBuilderWorker(BibliographyMode.ADVANCED)
    print(f"Worker Mode: {advanced_worker.mode.value}")
    
    advanced_result = advanced_worker._extract_bibliography_advanced({
        'content': test_content,
        'filename': 'test_document.txt'
    })
    
    print(f"Success: {advanced_result.get('success', False)}")
    print(f"Extraction Method: {advanced_result.get('extraction_method', 'Unknown')}")
    print(f"Confidence Score: {advanced_result.get('confidence_score', 0.0):.2f}")
    print(f"Total Citations: {advanced_result.get('total_citations', 0)}")
    
    # Show advanced features
    advanced_features = advanced_result.get('advanced_features', {})
    print(f"\n🤖 AI Analysis:")
    ai_analysis = advanced_features.get('ai_analysis', {})
    
    citation_network = ai_analysis.get('citation_network_analysis', {})
    print(f"  Citation Network Analysis:")
    print(f"    Network Density: {citation_network.get('network_density', 0.0):.2f}")
    print(f"    Central Authors: {', '.join(citation_network.get('central_authors', []))}")
    
    clusters = citation_network.get('citation_clusters', [])
    print(f"    Citation Clusters ({len(clusters)}):")
    for cluster in clusters:
        print(f"      - {cluster.get('name', 'Unknown')}: {cluster.get('citations', 0)} citations")
    
    topic_modeling = ai_analysis.get('topic_modeling', {})
    topics = topic_modeling.get('topics', [])
    print(f"  Topic Modeling ({len(topics)} topics):")
    for topic in topics:
        print(f"    - {topic.get('topic', 'Unknown')}: {topic.get('weight', 0.0):.2f} weight, {topic.get('citations', 0)} citations")
    
    impact_analysis = ai_analysis.get('impact_analysis', {})
    print(f"  Impact Analysis:")
    print(f"    High Impact: {impact_analysis.get('high_impact_citations', 0)}")
    print(f"    Medium Impact: {impact_analysis.get('medium_impact_citations', 0)}")
    print(f"    Low Impact: {impact_analysis.get('low_impact_citations', 0)}")
    print(f"    Impact Score: {impact_analysis.get('citation_impact_score', 0.0):.2f}")
    
    bibliometric_analysis = ai_analysis.get('bibliometric_analysis', {})
    print(f"  Bibliometric Analysis:")
    print(f"    Total Citations: {bibliometric_analysis.get('total_citations', 0)}")
    print(f"    Unique Authors: {bibliometric_analysis.get('unique_authors', 0)}")
    print(f"    Unique Journals: {bibliometric_analysis.get('unique_journals', 0)}")
    
    publication_years = bibliometric_analysis.get('publication_years', {})
    print(f"    Publication Years: {publication_years.get('range', 'Unknown')}")
    
    citation_types = bibliometric_analysis.get('citation_types', {})
    print(f"    Citation Types:")
    for citation_type, count in citation_types.items():
        print(f"      - {citation_type}: {count}")
    
    print(f"\n📈 ML Predictions:")
    ml_predictions = advanced_features.get('ml_predictions', {})
    print(f"  Citation Quality Score: {ml_predictions.get('citation_quality_score', 0.0):.2f}")
    print(f"  Impact Factor Estimate: {ml_predictions.get('impact_factor_estimate', 0.0):.2f}")
    print(f"  Relevance Score: {ml_predictions.get('relevance_score', 0.0):.2f}")
    
    recommended_citations = ml_predictions.get('recommended_citations', [])
    if recommended_citations:
        print(f"  Recommended Improvements:")
        for recommendation in recommended_citations:
            print(f"    - {recommendation}")
    
    print(f"\n📊 Advanced Metrics:")
    advanced_metrics = advanced_features.get('advanced_metrics', {})
    print(f"  Citation Diversity: {advanced_metrics.get('citation_diversity', 0.0):.2f}")
    print(f"  Temporal Distribution: {advanced_metrics.get('temporal_distribution', 0.0):.2f}")
    print(f"  Geographic Distribution: {advanced_metrics.get('geographic_distribution', 0.0):.2f}")
    print(f"  Institutional Diversity: {advanced_metrics.get('institutional_diversity', 0.0):.2f}")
    
    # Test 4: Mode Comparison
    print("\n📋 Test 4: Mode Comparison")
    print("-" * 40)
    
    test_documents = [
        {
            'name': 'Academic Paper',
            'content': """
            References
            
            1. Smith, J. (2020). Research Methods. Journal of Science, 10(2), 45-60.
            2. Johnson, A. (2019). Data Analysis. Statistics Review, 15(1), 23-35.
            3. Brown, M. (2021). Conclusions. Research Quarterly, 8(3), 67-78.
            """
        },
        {
            'name': 'Technical Report',
            'content': """
            Bibliography
            
            Anderson, K. (2022). Technical Implementation. Technical Report, 45-67.
            Taylor, S. (2021). System Design. Engineering Journal, 12(4), 89-102.
            Martinez, P. (2020). Performance Analysis. Systems Review, 6(2), 34-47.
            """
        },
        {
            'name': 'Simple Document',
            'content': """
            References
            
            Wilson, R. (2023). Overview. General Review, 5(1), 12-15.
            Davis, L. (2022). Summary. Summary Journal, 3(2), 8-10.
            """
        }
    ]
    
    print("Comparing bibliography extraction across modes:")
    print(f"{'Document':<20} {'Simple':<15} {'Enhanced':<15} {'Advanced':<15}")
    print("-" * 70)
    
    for doc in test_documents:
        name = doc['name']
        content = doc['content']
        
        # Simple extraction
        simple_result = simple_worker._extract_bibliography_simple({
            'content': content,
            'filename': f'{name.lower().replace(" ", "_")}.txt'
        })
        simple_citations = simple_result.get('total_citations', 0)
        
        # Enhanced extraction
        enhanced_result = enhanced_worker._extract_bibliography_enhanced({
            'content': content,
            'filename': f'{name.lower().replace(" ", "_")}.txt'
        })
        enhanced_citations = enhanced_result.get('total_citations', 0)
        
        # Advanced extraction
        advanced_result = advanced_worker._extract_bibliography_advanced({
            'content': content,
            'filename': f'{name.lower().replace(" ", "_")}.txt'
        })
        advanced_citations = advanced_result.get('total_citations', 0)
        
        print(f"{name:<20} {simple_citations:<15} {enhanced_citations:<15} {advanced_citations:<15}")
    
    # Test 5: Citation Analysis (Advanced Mode Only)
    print("\n📋 Test 5: Citation Analysis")
    print("-" * 40)
    
    analysis_result = advanced_worker._analyze_citations_task({
        'content': test_content,
        'filename': 'test_document.txt'
    })
    
    if analysis_result.get('success', False):
        analysis = analysis_result.get('analysis_result', {})
        print(f"Citation Quality: {analysis.get('citation_quality', 'Unknown')}")
        print(f"Coverage Score: {analysis.get('coverage_score', 0.0):.2f}")
        
        improvements = analysis.get('recommended_improvements', [])
        if improvements:
            print(f"Recommended Improvements:")
            for improvement in improvements:
                print(f"  - {improvement}")
        
        patterns = analysis.get('citation_patterns', [])
        if patterns:
            print(f"Citation Patterns: {', '.join(patterns)}")
    else:
        print(f"Analysis failed: {analysis_result.get('error', 'Unknown error')}")
    
    # Test 6: Performance Metrics
    print("\n📋 Test 6: Performance Metrics")
    print("-" * 40)
    
    for mode in [BibliographyMode.SIMPLE, BibliographyMode.ENHANCED, BibliographyMode.ADVANCED]:
        worker = BibliographyBuilderWorker(mode)
        metrics = worker.get_performance_metrics()
        methods = worker.get_supported_methods()
        
        print(f"\n{mode.value.upper()} Mode:")
        print(f"  Worker: {metrics['worker_name']}")
        print(f"  Type: {metrics['worker_type']}")
        print(f"  Mode: {metrics['mode']}")
        print(f"  Cache Size: {metrics['cache_size']}")
        print(f"  Available Methods: {methods['available_methods']}")
        print(f"  Pattern Sets: {methods['patterns']}")
    
    print("\n✅ BibliographyBuilderWorker testing completed!")
    print("\n🎯 Key Benefits of Unified Bibliography Worker:")
    print("  ✅ Progressive complexity: Simple → Enhanced → Advanced")
    print("  ✅ Multi-format citation parsing in enhanced mode")
    print("  ✅ AI-powered analysis in advanced mode")
    print("  ✅ Citation network analysis and bibliometric analysis")
    print("  ✅ Impact analysis and topic modeling")
    print("  ✅ Consistent interface across all modes")


if __name__ == "__main__":
    test_bibliography_builder_worker()


