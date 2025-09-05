#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test TOC Extractor Worker

Test script to validate the TOC (Table of Contents) extractor worker functionality.
"""

import sys
import os
import time
import json
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_toc_extractor_worker():
    """Test the TOC extractor worker"""
    print("🔍 Testing TOC Extractor Worker")
    print("=" * 50)
    
    try:
        from backend.mcp.workers.toc_extractor_worker import TOCExtractorWorker
        
        # Initialize worker
        worker = TOCExtractorWorker()
        
        print(f"✅ Available methods: {list(worker.available_methods.keys())}")
        
        # Find a test PDF
        test_pdfs = [
            "data/input/reviews/Whitepaper-Model-Validation-Best-Practices-1.pdf",
            "input/omnibus/all/2309.11495 - Chain-of-Verification Reduces Hallucination in Large Language Models.pdf",
            "tests/test_data/sample_document.pdf"
        ]
        
        test_pdf = None
        for pdf_path in test_pdfs:
            if os.path.exists(pdf_path):
                test_pdf = pdf_path
                break
        
        if not test_pdf:
            print("⚠️  No test PDF found, testing capabilities only...")
            return test_worker_capabilities(worker)
        
        print(f"📄 Testing with: {test_pdf}")
        
        # Test TOC extraction
        start_time = time.time()
        toc_structure = worker.extract_toc(
            file_path=test_pdf,
            document_id="test_doc_001",
            document_title="Test Document"
        )
        processing_time = time.time() - start_time
        
        print(f"✅ TOC extraction completed in {processing_time:.3f} seconds")
        print(f"   Success: {toc_structure.confidence_score > 0}")
        print(f"   Method used: {toc_structure.extraction_method}")
        print(f"   Confidence score: {toc_structure.confidence_score}")
        print(f"   Total entries: {len(toc_structure.entries)}")
        print(f"   Total pages: {toc_structure.total_pages}")
        
        if toc_structure.entries:
            print("\n📊 TOC Structure:")
            print_toc_structure(toc_structure.entries, max_depth=3)
        
        # Test JSON export
        print("\n🧪 Testing JSON export...")
        json_data = worker.export_to_json(toc_structure)
        print(f"   JSON size: {len(json_data)} characters")
        
        # Test section navigation
        if toc_structure.entries:
            print("\n🧪 Testing section navigation...")
            test_section_navigation(worker, toc_structure)
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_worker_capabilities(worker):
    """Test worker capabilities without a PDF file"""
    print("🧪 Testing worker capabilities...")
    
    # Test method availability
    print(f"✅ Available methods: {worker.available_methods}")
    
    # Test cache functionality
    cache_stats = worker.get_cache_stats()
    print(f"✅ Cache stats: {cache_stats}")
    
    # Test with invalid input
    try:
        result = worker.extract_toc()
        print(f"❌ Should have raised ValueError for invalid input")
        return False
    except ValueError:
        print(f"✅ Invalid input handling: Correctly raised ValueError")
    
    return True

def print_toc_structure(entries, max_depth=3, current_depth=0):
    """Print TOC structure in a tree format"""
    if current_depth >= max_depth:
        return
    
    for entry in entries[:5]:  # Show first 5 entries at each level
        indent = "  " * current_depth
        page_info = f" (p.{entry.page_number})" if entry.page_number else ""
        print(f"{indent}• {entry.title}{page_info} [Level {entry.level}]")
        
        if entry.children and current_depth < max_depth - 1:
            print_toc_structure(entry.children, max_depth, current_depth + 1)

def test_section_navigation(worker, toc_structure):
    """Test section navigation functionality"""
    if not toc_structure.entries:
        print("   ⚠️  No entries to test navigation")
        return
    
    # Test getting section content
    first_entry = toc_structure.entries[0]
    section_content = worker.get_section_content(toc_structure, first_entry.section_id)
    
    if section_content:
        print(f"   ✅ Section content retrieval: {section_content['title']}")
    else:
        print(f"   ⚠️  Section content retrieval failed")
    
    # Test getting section range
    section_range = worker.get_section_range(toc_structure, first_entry.section_id)
    
    if section_range:
        print(f"   ✅ Section range: {section_range[0]} - {section_range[1]}")
    else:
        print(f"   ⚠️  Section range retrieval failed")

def test_toc_extraction_methods():
    """Test different TOC extraction methods"""
    print("\n🔄 Testing TOC Extraction Methods")
    print("=" * 50)
    
    try:
        from backend.mcp.workers.toc_extractor_worker import TOCExtractorWorker
        
        worker = TOCExtractorWorker()
        
        # Test each method individually
        methods = ['pdfplumber', 'fitz', 'pymupdf', 'pypdfium2']
        
        for method in methods:
            if worker.available_methods.get(method):
                print(f"✅ {method}: Available")
            else:
                print(f"❌ {method}: Not available")
        
        # Test method priority
        print("\n📋 Method Priority (in order of preference):")
        priority_methods = ['pdfplumber', 'fitz', 'pymupdf', 'pypdfium2']
        for i, method in enumerate(priority_methods, 1):
            status = "✅" if worker.available_methods.get(method) else "❌"
            print(f"   {i}. {method}: {status}")
        
        return True
        
    except Exception as e:
        print(f"❌ Method test failed: {e}")
        return False

def test_toc_patterns():
    """Test TOC pattern recognition"""
    print("\n🔍 Testing TOC Pattern Recognition")
    print("=" * 50)
    
    try:
        from backend.mcp.workers.toc_extractor_worker import TOCExtractorWorker
        
        worker = TOCExtractorWorker()
        
        # Test TOC header patterns
        test_headers = [
            "Table of Contents",
            "Contents",
            "Index",
            "Outline",
            "CHAPTER 1",
            "1.1 Introduction",
            "Appendix A"
        ]
        
        print("📋 Testing TOC header patterns:")
        for header in test_headers:
            is_toc = worker._is_toc_page(header)
            print(f"   '{header}': {'✅ TOC' if is_toc else '❌ Not TOC'}")
        
        # Test TOC entry patterns
        test_entries = [
            "Chapter 1. Introduction ................ 5",
            "1.1 Background ........................ 12",
            "Section 1: Overview ................... 3",
            "Appendix A. Data Tables .............. 45"
        ]
        
        print("\n📋 Testing TOC entry patterns:")
        for entry in test_entries:
            entries = worker._parse_toc_entries(entry, 1)
            if entries:
                print(f"   '{entry}': ✅ Parsed as '{entries[0].title}' (p.{entries[0].page_number})")
            else:
                print(f"   '{entry}': ❌ Not parsed")
        
        return True
        
    except Exception as e:
        print(f"❌ Pattern test failed: {e}")
        return False

def test_toc_hierarchy():
    """Test TOC hierarchy building"""
    print("\n🏗️  Testing TOC Hierarchy Building")
    print("=" * 50)
    
    try:
        from backend.mcp.workers.toc_extractor_worker import TOCExtractorWorker, TOCEntry
        
        worker = TOCExtractorWorker()
        
        # Create test entries
        test_entries = [
            TOCEntry("Chapter 1", 1, 1, "chapter_1", None, [], {}),
            TOCEntry("1.1 Introduction", 1, 2, "section_1_1", None, [], {}),
            TOCEntry("1.2 Background", 3, 2, "section_1_2", None, [], {}),
            TOCEntry("Chapter 2", 5, 1, "chapter_2", None, [], {}),
            TOCEntry("2.1 Methods", 5, 2, "section_2_1", None, [], {}),
        ]
        
        # Structure entries
        structured = worker._structure_toc_entries(test_entries)
        
        print(f"✅ Structured {len(test_entries)} entries into {len(structured)} root entries")
        
        for entry in structured:
            print(f"   • {entry.title} (Level {entry.level}, {len(entry.children)} children)")
            for child in entry.children:
                print(f"     - {child.title} (Level {child.level})")
        
        return True
        
    except Exception as e:
        print(f"❌ Hierarchy test failed: {e}")
        return False

def test_performance():
    """Test performance of TOC extraction"""
    print("\n⚡ Testing Performance")
    print("=" * 50)
    
    try:
        from backend.mcp.workers.toc_extractor_worker import TOCExtractorWorker
        
        worker = TOCExtractorWorker()
        
        # Test cache performance
        print("🧪 Testing cache performance...")
        
        # Find a test PDF
        test_pdfs = [
            "data/input/reviews/Whitepaper-Model-Validation-Best-Practices-1.pdf",
            "input/omnibus/all/2309.11495 - Chain-of-Verification Reduces Hallucination in Large Language Models.pdf",
            "tests/test_data/sample_document.pdf"
        ]
        
        test_pdf = None
        for pdf_path in test_pdfs:
            if os.path.exists(pdf_path):
                test_pdf = pdf_path
                break
        
        if test_pdf:
            # First call (should be slower)
            start_time = time.time()
            result1 = worker.extract_toc(test_pdf, document_id="perf_test_1")
            time1 = time.time() - start_time
            
            # Second call (should be faster due to cache)
            start_time = time.time()
            result2 = worker.extract_toc(test_pdf, document_id="perf_test_1")
            time2 = time.time() - start_time
            
            print(f"   First call: {time1:.3f} seconds")
            print(f"   Cached call: {time2:.3f} seconds")
            if time2 > 0:
                print(f"   Cache speedup: {time1/time2:.1f}x faster")
            else:
                print(f"   Cache speedup: Instant")
        else:
            print("   ⚠️  No test PDF available for performance test")
        
        # Test memory usage
        print("\n🧪 Testing memory management...")
        cache_stats = worker.get_cache_stats()
        print(f"   Cache size: {cache_stats['cache_size']} entries")
        
        # Clear cache
        worker.clear_cache()
        cache_stats_after = worker.get_cache_stats()
        print(f"   After clearing: {cache_stats_after['cache_size']} entries")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 TOC Extractor Worker Test Suite")
    print("=" * 60)
    
    # Track results
    results = {}
    
    # Test 1: Basic functionality
    results['basic_functionality'] = test_toc_extractor_worker()
    
    # Test 2: Method availability
    results['method_availability'] = test_toc_extraction_methods()
    
    # Test 3: Pattern recognition
    results['pattern_recognition'] = test_toc_patterns()
    
    # Test 4: Hierarchy building
    results['hierarchy_building'] = test_toc_hierarchy()
    
    # Test 5: Performance
    results['performance'] = test_performance()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TOC Extractor Worker Test Results")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All TOC extractor worker tests passed!")
        print("\n✅ Key Benefits:")
        print("   - Multiple extraction methods with automatic fallback")
        print("   - Hierarchical TOC structure building")
        print("   - Pattern-based TOC recognition")
        print("   - Section navigation and content retrieval")
        print("   - Caching for improved performance")
        print("   - JSON export for database storage")
    else:
        print("⚠️  Some TOC extractor worker tests failed. Check the output above for details.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
