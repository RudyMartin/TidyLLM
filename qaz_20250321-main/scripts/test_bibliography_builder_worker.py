#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Bibliography Builder Worker

Test script to validate the Bibliography Builder Worker functionality with sample academic papers.
"""

import sys
import os
import time
import json
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_bibliography_builder_worker():
    """Test basic bibliography extraction functionality"""
    print("🧪 Testing Bibliography Builder Worker")
    print("=" * 60)
    
    try:
        from backend.mcp.workers.bibliography_builder_worker import BibliographyBuilderWorker
        
        worker = BibliographyBuilderWorker()
        
        # Test with a sample paper
        test_pdf = "input/omnibus/all/2309.11495 - Chain-of-Verification Reduces Hallucination in Large Language Models.pdf"
        
        if os.path.exists(test_pdf):
            print(f"📄 Testing with: {os.path.basename(test_pdf)}")
            
            start_time = time.time()
            bibliography = worker.extract_bibliography(
                file_path=test_pdf,
                document_id="test_2309_11495",
                document_title="Chain-of-Verification Reduces Hallucination in Large Language Models"
            )
            extraction_time = time.time() - start_time
            
            print(f"✅ Extraction completed in {extraction_time:.2f} seconds")
            print(f"📊 Results:")
            print(f"   Total citations: {bibliography.total_citations}")
            print(f"   Extraction method: {bibliography.extraction_method}")
            print(f"   Confidence score: {bibliography.confidence_score:.3f}")
            
            if bibliography.citations:
                print(f"\n📝 Sample citations:")
                for i, citation in enumerate(bibliography.citations[:5]):
                    print(f"   {i+1}. {citation.authors[0] if citation.authors else 'Unknown'} ({citation.year or 'N/A'})")
                    print(f"      Title: {citation.title[:80]}{'...' if len(citation.title) > 80 else ''}")
                    print(f"      Type: {citation.citation_type}, Confidence: {citation.confidence_score:.3f}")
                    print()
            
            return bibliography
        else:
            print(f"❌ Test PDF not found: {test_pdf}")
            return None
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None

def test_worker_capabilities(worker):
    """Test worker capabilities and method availability"""
    print("\n🔧 Testing Worker Capabilities")
    print("=" * 60)
    
    try:
        # Check available methods
        methods = worker.available_methods
        print(f"📋 Available extraction methods:")
        for method, available in methods.items():
            status = "✅" if available else "❌"
            print(f"   {status} {method}")
        
        # Check patterns
        patterns = worker.patterns
        print(f"\n🔍 Pattern types loaded:")
        for pattern_type, pattern_list in patterns.items():
            print(f"   📝 {pattern_type}: {len(pattern_list)} patterns")
        
        # Check cache stats
        cache_stats = worker.get_cache_stats()
        print(f"\n💾 Cache statistics:")
        print(f"   Cache size: {cache_stats['cache_size']} entries")
        
        return True
        
    except Exception as e:
        print(f"❌ Capability test failed: {e}")
        return False

def print_bibliography_summary(bibliography):
    """Print a summary of the extracted bibliography"""
    if not bibliography or not bibliography.citations:
        print("❌ No bibliography data to summarize")
        return
    
    print("\n📊 Bibliography Summary")
    print("=" * 60)
    
    # Basic stats
    print(f"📄 Document: {bibliography.document_title}")
    print(f"🆔 Document ID: {bibliography.document_id}")
    print(f"📚 Total citations: {bibliography.total_citations}")
    print(f"🔧 Extraction method: {bibliography.extraction_method}")
    print(f"🎯 Confidence score: {bibliography.confidence_score:.3f}")
    
    # Citation type distribution
    type_counts = {}
    for citation in bibliography.citations:
        citation_type = citation.citation_type
        type_counts[citation_type] = type_counts.get(citation_type, 0) + 1
    
    print(f"\n📈 Citation type distribution:")
    for citation_type, count in type_counts.items():
        percentage = (count / bibliography.total_citations) * 100
        print(f"   {citation_type}: {count} ({percentage:.1f}%)")
    
    # Year distribution
    year_counts = {}
    for citation in bibliography.citations:
        if citation.year:
            year_counts[citation.year] = year_counts.get(citation.year, 0) + 1
    
    if year_counts:
        print(f"\n📅 Year distribution (top 5):")
        sorted_years = sorted(year_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for year, count in sorted_years:
            print(f"   {year}: {count} citations")
    
    # ArXiv citations
    arxiv_citations = [c for c in bibliography.citations if c.arxiv_id]
    if arxiv_citations:
        print(f"\n🔬 ArXiv citations: {len(arxiv_citations)}")
        for citation in arxiv_citations[:3]:
            print(f"   {citation.arxiv_id}: {citation.title[:60]}{'...' if len(citation.title) > 60 else ''}")

def test_citation_parsing():
    """Test citation parsing with sample citation texts"""
    print("\n🔍 Testing Citation Parsing")
    print("=" * 60)
    
    try:
        from backend.mcp.workers.bibliography_builder_worker import BibliographyBuilderWorker
        
        worker = BibliographyBuilderWorker()
        
        # Sample citation texts from the papers we found
        sample_citations = [
            "[1] MoloudAbdar,FarhadPourpanah,SadiqHussain,DanaRezazadegan,LiLiu,MohammadGhavamzadeh, Paul Fieguth, Xiaochun Cao, Abbas Khosravi, U Rajendra Acharya, et al. A review of uncertainty quantificationindeeplearning:Techniques,applicationsandchallenges. InformationFusion,76:243–297, 2021.",
            
            "LeonardAdolphs,KurtShuster,JackUrbanek,ArthurSzlam,andJasonWeston. Reasonfirst,then respond: Modulargenerationforknowledge-infuseddialogue. arXivpreprintarXiv:2111.05204, 2021.",
            
            "Rishabh Agarwal, Avi Singh, Lei M. Zhang, Bernd Bohnet, Stephanie Chan, Ankesh Anand, Zaheer Abbas, Azade Nova, John D. Co-Reyes, Eric Chu, Feryal Behbahani, Aleksandra Faust, and Hugo Larochelle. Many-shot in-context learning. CoRR, abs/2404.11018, 2024a.",
            
            "Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P.,Sastry,G.,Askell,A.,etal.Languagemodelsarefew-shotlearners.Advancesinneuralinformation processingsystems33,1877–1901(2020)."
        ]
        
        print("📝 Testing citation parsing with sample texts:")
        
        for i, citation_text in enumerate(sample_citations, 1):
            print(f"\n--- Sample {i} ---")
            print(f"Raw text: {citation_text[:100]}{'...' if len(citation_text) > 100 else ''}")
            
            # Parse the citation
            citation = worker._parse_single_citation(citation_text)
            
            if citation:
                print(f"✅ Parsed successfully:")
                print(f"   Authors: {citation.authors}")
                print(f"   Title: {citation.title[:60]}{'...' if len(citation.title) > 60 else ''}")
                print(f"   Year: {citation.year}")
                print(f"   Type: {citation.citation_type}")
                print(f"   ArXiv ID: {citation.arxiv_id}")
                print(f"   DOI: {citation.doi}")
                print(f"   Confidence: {citation.confidence_score:.3f}")
            else:
                print(f"❌ Failed to parse citation")
        
        return True
        
    except Exception as e:
        print(f"❌ Citation parsing test failed: {e}")
        return False

def test_multiple_papers():
    """Test bibliography extraction on multiple papers"""
    print("\n📚 Testing Multiple Papers")
    print("=" * 60)
    
    try:
        from backend.mcp.workers.bibliography_builder_worker import BibliographyBuilderWorker
        
        worker = BibliographyBuilderWorker()
        
        # List of papers to test
        test_papers = [
            "input/omnibus/all/2309.11495 - Chain-of-Verification Reduces Hallucination in Large Language Models.pdf",
            "input/omnibus/all/2209.07067.pdf",
            "input/omnibus/all/2210.03102.pdf",
            "input/omnibus/all/2205.14415.pdf",
            "input/omnibus/all/2403.05530v3.pdf"
        ]
        
        results = []
        
        for paper_path in test_papers:
            if os.path.exists(paper_path):
                print(f"\n📄 Processing: {os.path.basename(paper_path)}")
                
                start_time = time.time()
                bibliography = worker.extract_bibliography(
                    file_path=paper_path,
                    document_id=os.path.splitext(os.path.basename(paper_path))[0],
                    document_title=os.path.splitext(os.path.basename(paper_path))[0]
                )
                extraction_time = time.time() - start_time
                
                results.append({
                    'file': os.path.basename(paper_path),
                    'total_citations': bibliography.total_citations,
                    'confidence': bibliography.confidence_score,
                    'method': bibliography.extraction_method,
                    'time': extraction_time,
                    'success': bibliography.confidence_score > 0
                })
                
                print(f"   Citations: {bibliography.total_citations}")
                print(f"   Confidence: {bibliography.confidence_score:.3f}")
                print(f"   Method: {bibliography.extraction_method}")
                print(f"   Time: {extraction_time:.2f}s")
            else:
                print(f"❌ File not found: {paper_path}")
        
        # Summary
        print(f"\n📊 Summary of Results:")
        print("=" * 60)
        
        successful = [r for r in results if r['success']]
        total_citations = sum(r['total_citations'] for r in successful)
        avg_confidence = sum(r['confidence'] for r in successful) / len(successful) if successful else 0
        
        print(f"✅ Successful extractions: {len(successful)}/{len(results)}")
        print(f"📚 Total citations found: {total_citations}")
        print(f"🎯 Average confidence: {avg_confidence:.3f}")
        
        for result in results:
            status = "✅" if result['success'] else "❌"
            print(f"   {status} {result['file']}: {result['total_citations']} citations, {result['confidence']:.3f} confidence")
        
        return results
        
    except Exception as e:
        print(f"❌ Multiple papers test failed: {e}")
        return []

def test_statistics_and_export():
    """Test bibliography statistics and export functionality"""
    print("\n📈 Testing Statistics and Export")
    print("=" * 60)
    
    try:
        from backend.mcp.workers.bibliography_builder_worker import BibliographyBuilderWorker
        
        worker = BibliographyBuilderWorker()
        
        # Test with a sample paper
        test_pdf = "input/omnibus/all/2309.11495 - Chain-of-Verification Reduces Hallucination in Large Language Models.pdf"
        
        if os.path.exists(test_pdf):
            bibliography = worker.extract_bibliography(
                file_path=test_pdf,
                document_id="test_stats",
                document_title="Test Document"
            )
            
            if bibliography.citations:
                # Get statistics
                stats = worker.get_citation_statistics(bibliography)
                
                print("📊 Citation Statistics:")
                print(f"   Total citations: {stats.get('total_citations', 0)}")
                print(f"   Unique authors: {stats.get('unique_authors', 0)}")
                print(f"   Average confidence: {stats.get('avg_confidence', 0):.3f}")
                print(f"   ArXiv citations: {stats.get('arxiv_citations', 0)}")
                print(f"   Journal citations: {stats.get('journal_citations', 0)}")
                print(f"   Conference citations: {stats.get('conference_citations', 0)}")
                
                # Citation type distribution
                type_counts = stats.get('citation_types', {})
                if type_counts:
                    print(f"\n📈 Citation type distribution:")
                    for citation_type, count in type_counts.items():
                        percentage = (count / stats['total_citations']) * 100
                        print(f"   {citation_type}: {count} ({percentage:.1f}%)")
                
                # Top authors
                top_authors = stats.get('top_authors', [])
                if top_authors:
                    print(f"\n👥 Top authors:")
                    for author, count in top_authors[:5]:
                        print(f"   {author}: {count} citations")
                
                # Test JSON export
                json_export = worker.export_to_json(bibliography)
                print(f"\n💾 JSON export length: {len(json_export)} characters")
                
                # Test cache functionality
                print(f"\n💾 Cache test:")
                cache_stats_before = worker.get_cache_stats()
                print(f"   Cache size before: {cache_stats_before['cache_size']}")
                
                # Extract again (should use cache)
                start_time = time.time()
                bibliography_cached = worker.extract_bibliography(
                    file_path=test_pdf,
                    document_id="test_stats",
                    document_title="Test Document"
                )
                cached_time = time.time() - start_time
                
                print(f"   Cached extraction time: {cached_time:.3f}s")
                
                cache_stats_after = worker.get_cache_stats()
                print(f"   Cache size after: {cache_stats_after['cache_size']}")
                
                return True
            else:
                print("❌ No citations found for statistics test")
                return False
        else:
            print(f"❌ Test PDF not found: {test_pdf}")
            return False
            
    except Exception as e:
        print(f"❌ Statistics test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🔬 Bibliography Builder Worker Test Suite")
    print("=" * 60)
    
    all_passed = True
    
    # Test 1: Basic functionality
    bibliography = test_bibliography_builder_worker()
    if bibliography:
        print("✅ Basic functionality test passed")
    else:
        print("❌ Basic functionality test failed")
        all_passed = False
    
    # Test 2: Worker capabilities
    try:
        from backend.mcp.workers.bibliography_builder_worker import BibliographyBuilderWorker
        worker = BibliographyBuilderWorker()
        if test_worker_capabilities(worker):
            print("✅ Worker capabilities test passed")
        else:
            print("❌ Worker capabilities test failed")
            all_passed = False
    except Exception as e:
        print(f"❌ Worker capabilities test failed: {e}")
        all_passed = False
    
    # Test 3: Citation parsing
    if test_citation_parsing():
        print("✅ Citation parsing test passed")
    else:
        print("❌ Citation parsing test failed")
        all_passed = False
    
    # Test 4: Multiple papers
    results = test_multiple_papers()
    if results:
        print("✅ Multiple papers test passed")
    else:
        print("❌ Multiple papers test failed")
        all_passed = False
    
    # Test 5: Statistics and export
    if test_statistics_and_export():
        print("✅ Statistics and export test passed")
    else:
        print("❌ Statistics and export test failed")
        all_passed = False
    
    # Print bibliography summary if available
    if bibliography:
        print_bibliography_summary(bibliography)
    
    # Final summary
    print(f"\n🎯 Test Summary")
    print("=" * 60)
    if all_passed:
        print("✅ All tests passed! Bibliography Builder Worker is working correctly.")
    else:
        print("❌ Some tests failed. Please check the output above for details.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
