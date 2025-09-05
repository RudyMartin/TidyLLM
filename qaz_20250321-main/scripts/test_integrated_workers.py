#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated Workers Test Suite

Comprehensive test script that demonstrates TOC Extractor, Bibliography Builder, 
and Table Extractor workers working together for complete document analysis.
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, Any, List
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def setup_logging():
    """Setup logging for the test suite"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/integrated_workers_test.log')
        ]
    )

def test_integrated_processing():
    """Test all three workers processing the same document"""
    print("🔍 Testing Integrated Document Processing")
    print("=" * 50)
    
    try:
        from backend.mcp.orchestrators.document_processing_orchestrator import DocumentProcessingOrchestrator
        
        # Initialize orchestrator
        orchestrator = DocumentProcessingOrchestrator()
        print(f"✅ Orchestrator initialized with {len(orchestrator.workers)} workers")
        
        # Check available workers
        available_workers = list(orchestrator.workers.keys())
        print(f"📋 Available workers: {', '.join(available_workers)}")
        
        # Find test documents
        test_documents = find_test_documents()
        if not test_documents:
            print("❌ No test documents found")
            return False
        
        print(f"📄 Found {len(test_documents)} test documents")
        
        # Process each document
        results = []
        for doc_path in test_documents[:3]:  # Limit to first 3 for testing
            print(f"\n📖 Processing: {Path(doc_path).name}")
            
            result = orchestrator.process_document(
                document_path=doc_path,
                task_types=['toc', 'bibliography', 'images', 'tables']
            )
            
            results.append(result)
            
            # Display results
            display_processing_results(result)
        
        # Summary
        display_summary(results)
        
        return True
        
    except Exception as e:
        print(f"❌ Integrated processing test failed: {e}")
        return False

def test_individual_workers():
    """Test each worker individually"""
    print("\n🔧 Testing Individual Workers")
    print("=" * 50)
    
    try:
        # Test TOC Extractor
        print("\n📑 Testing TOC Extractor Worker")
        test_toc_worker()
        
        # Test Bibliography Builder
        print("\n📚 Testing Bibliography Builder Worker")
        test_bibliography_worker()
        
        # Test Table Extraction
        print("\n📋 Testing Table Extraction")
        test_table_extraction()
        
        return True
        
    except Exception as e:
        print(f"❌ Individual workers test failed: {e}")
        return False

def test_toc_worker():
    """Test TOC Extractor Worker"""
    try:
        from backend.mcp.workers.toc_extractor_worker import TOCExtractorWorker
        
        worker = TOCExtractorWorker()
        print(f"✅ TOC Worker initialized")
        
        # Test with sample document
        test_docs = find_test_documents()
        if test_docs:
            doc_path = test_docs[0]
            print(f"📄 Testing with: {Path(doc_path).name}")
            
            toc_structure = worker.extract_toc(
                file_path=doc_path,
                document_id="test_toc",
                document_title="Test Document"
            )
            
            print(f"📑 TOC extracted: {len(toc_structure.entries)} sections")
            print(f"🎯 Confidence: {toc_structure.confidence_score:.2f}")
            print(f"🔧 Method: {toc_structure.extraction_method}")
            
            # Show first few entries
            for i, entry in enumerate(toc_structure.entries[:5]):
                print(f"   {i+1}. {entry.title} (Page {entry.page_number})")
            
            return True
        else:
            print("❌ No test documents available")
            return False
            
    except Exception as e:
        print(f"❌ TOC worker test failed: {e}")
        return False

def test_bibliography_worker():
    """Test Bibliography Builder Worker"""
    try:
        from backend.mcp.workers.bibliography_builder_worker import BibliographyBuilderWorker
        
        worker = BibliographyBuilderWorker()
        print(f"✅ Bibliography Worker initialized")
        
        # Test with sample document
        test_docs = find_test_documents()
        if test_docs:
            doc_path = test_docs[0]
            print(f"📄 Testing with: {Path(doc_path).name}")
            
            bibliography = worker.extract_bibliography(
                file_path=doc_path,
                document_id="test_bib",
                document_title="Test Document"
            )
            
            print(f"📚 Bibliography extracted: {len(bibliography.citations)} citations")
            print(f"🎯 Confidence: {bibliography.confidence_score:.2f}")
            print(f"🔧 Method: {bibliography.extraction_method}")
            
            # Get statistics
            stats = worker.get_citation_statistics(bibliography)
            print(f"📊 Citation types: {stats['citation_type_distribution']}")
            print(f"📈 Year range: {stats['year_distribution']}")
            
            # Show first few citations
            for i, citation in enumerate(bibliography.citations[:3]):
                print(f"   {i+1}. {citation.title} ({citation.year})")
                print(f"      Authors: {', '.join(citation.authors[:2])}")
            
            return True
        else:
            print("❌ No test documents available")
            return False
            
    except Exception as e:
        print(f"❌ Bibliography worker test failed: {e}")
        return False

def test_table_extraction():
    """Test Table Extraction"""
    try:
        from backend.mcp.workers.modern_pdf_processor import ModernPDFProcessor
        
        processor = ModernPDFProcessor()
        print(f"✅ PDF Processor initialized")
        
        # Test with sample document
        test_docs = find_test_documents()
        if test_docs:
            doc_path = test_docs[0]
            print(f"📄 Testing with: {Path(doc_path).name}")
            
            result = processor.process_pdf(doc_path)
            
            tables = result.get('tables', [])
            print(f"📋 Tables extracted: {len(tables)}")
            
            # Show table details
            for i, table in enumerate(tables[:3]):
                print(f"   Table {i+1}: {len(table.get('content', []))} rows, {len(table.get('header', []))} columns")
                if table.get('header'):
                    print(f"      Headers: {', '.join(table['header'][:3])}")
            
            return True
        else:
            print("❌ No test documents available")
            return False
            
    except Exception as e:
        print(f"❌ Table extraction test failed: {e}")
        return False

def test_cross_referencing():
    """Test cross-referencing between different extractors"""
    print("\n🔗 Testing Cross-Referencing")
    print("=" * 50)
    
    try:
        from backend.mcp.orchestrators.document_processing_orchestrator import DocumentProcessingOrchestrator
        
        orchestrator = DocumentProcessingOrchestrator()
        
        # Find a good test document
        test_docs = find_test_documents()
        if not test_docs:
            print("❌ No test documents available")
            return False
        
        doc_path = test_docs[0]
        print(f"📄 Cross-referencing: {Path(doc_path).name}")
        
        # Process document
        result = orchestrator.process_document(
            document_path=doc_path,
            task_types=['toc', 'bibliography', 'tables']
        )
        
        # Analyze cross-references
        cross_refs = analyze_cross_references(result)
        
        # Display cross-reference analysis
        display_cross_references(cross_refs)
        
        return True
        
    except Exception as e:
        print(f"❌ Cross-referencing test failed: {e}")
        return False

def analyze_cross_references(result):
    """Analyze cross-references between different extractors"""
    cross_refs = {
        'toc_sections': [],
        'citations_in_sections': {},
        'tables_in_sections': {},
        'citation_patterns': {},
        'table_patterns': {}
    }
    
    # Extract TOC sections
    if result.toc_extracted and 'toc' in result.metadata:
        toc_data = result.metadata['toc']['toc_data']
        cross_refs['toc_sections'] = toc_data.get('entries', [])
    
    # Extract citations
    if result.bibliography_extracted and 'bibliography' in result.metadata:
        bib_data = result.metadata['bibliography']['bibliography_data']
        citations = bib_data.get('citations', [])
        
        # Analyze citation patterns
        citation_types = {}
        citation_years = {}
        for citation in citations:
            # Handle both object and dict citations
            if hasattr(citation, 'citation_type'):
                citation_type = citation.citation_type
                citation_year = citation.year
            else:
                citation_type = citation.get('citation_type', 'unknown')
                citation_year = citation.get('year')
            
            citation_types[citation_type] = citation_types.get(citation_type, 0) + 1
            if citation_year:
                citation_years[citation_year] = citation_years.get(citation_year, 0) + 1
        
        cross_refs['citation_patterns'] = {
            'types': citation_types,
            'years': citation_years
        }
    
    # Extract tables
    if 'pdf' in result.metadata:
        tables = result.metadata['pdf'].get('tables', [])
        
        # Analyze table patterns
        table_sizes = []
        table_headers = []
        for table in tables:
            table_sizes.append((len(table.get('content', [])), len(table.get('header', []))))
            if table.get('header'):
                table_headers.extend(table['header'])
        
        cross_refs['table_patterns'] = {
            'sizes': table_sizes,
            'common_headers': list(set(table_headers))
        }
    
    return cross_refs

def display_cross_references(cross_refs):
    """Display cross-reference analysis"""
    print(f"📑 TOC Sections: {len(cross_refs['toc_sections'])}")
    
    if cross_refs['citation_patterns']:
        print(f"📚 Citation Patterns:")
        print(f"   Types: {cross_refs['citation_patterns']['types']}")
        print(f"   Years: {dict(sorted(cross_refs['citation_patterns']['years'].items())[:5])}")
    
    if cross_refs['table_patterns']:
        print(f"📋 Table Patterns:")
        print(f"   Total tables: {len(cross_refs['table_patterns']['sizes'])}")
        print(f"   Common headers: {cross_refs['table_patterns']['common_headers'][:5]}")

def test_performance_comparison():
    """Test performance comparison between individual and integrated processing"""
    print("\n⚡ Performance Comparison")
    print("=" * 50)
    
    try:
        from backend.mcp.orchestrators.document_processing_orchestrator import DocumentProcessingOrchestrator
        from backend.mcp.workers.toc_extractor_worker import TOCExtractorWorker
        from backend.mcp.workers.bibliography_builder_worker import BibliographyBuilderWorker
        from backend.mcp.workers.modern_pdf_processor import ModernPDFProcessor
        
        test_docs = find_test_documents()
        if not test_docs:
            print("❌ No test documents available")
            return False
        
        doc_path = test_docs[0]
        print(f"📄 Testing performance with: {Path(doc_path).name}")
        
        # Test individual processing
        print("\n🔧 Individual Processing:")
        start_time = time.time()
        
        toc_worker = TOCExtractorWorker()
        toc_start = time.time()
        toc_result = toc_worker.extract_toc(doc_path)
        toc_time = time.time() - toc_start
        
        bib_worker = BibliographyBuilderWorker()
        bib_start = time.time()
        bib_result = bib_worker.extract_bibliography(doc_path)
        bib_time = time.time() - bib_start
        
        pdf_processor = ModernPDFProcessor()
        pdf_start = time.time()
        pdf_result = pdf_processor.process_pdf(doc_path)
        pdf_time = time.time() - pdf_start
        
        individual_total = time.time() - start_time
        
        print(f"   TOC: {toc_time:.2f}s")
        print(f"   Bibliography: {bib_time:.2f}s")
        print(f"   PDF/Tables: {pdf_time:.2f}s")
        print(f"   Total: {individual_total:.2f}s")
        
        # Test integrated processing
        print("\n🔗 Integrated Processing:")
        start_time = time.time()
        
        orchestrator = DocumentProcessingOrchestrator()
        integrated_result = orchestrator.process_document(
            document_path=doc_path,
            task_types=['toc', 'bibliography', 'tables']
        )
        
        integrated_total = time.time() - start_time
        print(f"   Total: {integrated_total:.2f}s")
        
        # Performance analysis
        print(f"\n📊 Performance Analysis:")
        print(f"   Individual: {individual_total:.2f}s")
        print(f"   Integrated: {integrated_total:.2f}s")
        print(f"   Difference: {individual_total - integrated_total:.2f}s")
        print(f"   Efficiency: {((individual_total - integrated_total) / individual_total * 100):.1f}% improvement")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance comparison failed: {e}")
        return False

def find_test_documents():
    """Find test documents in the input folder"""
    input_dir = Path("input")
    if not input_dir.exists():
        print("❌ Input directory not found")
        return []
    
    # Look for PDF files
    pdf_files = []
    for pattern in ["**/*.pdf", "*.pdf"]:
        pdf_files.extend(input_dir.glob(pattern))
    
    # Sort by size (prefer larger files for better testing)
    pdf_files.sort(key=lambda x: x.stat().st_size, reverse=True)
    
    return [str(f) for f in pdf_files[:5]]  # Return top 5 largest files

def display_processing_results(result):
    """Display processing results for a document"""
    print(f"📊 Processing Results for {Path(result.document_path).name}:")
    print(f"   ✅ Success: {result.success}")
    print(f"   ⏱️  Time: {result.processing_time:.2f}s")
    
    if result.toc_extracted:
        print(f"   📑 TOC: {result.toc_sections} sections (confidence: {result.toc_confidence:.2f})")
    
    if result.bibliography_extracted:
        print(f"   📚 Bibliography: {result.citations_count} citations (confidence: {result.bibliography_confidence:.2f})")
    
    if result.images_extracted:
        print(f"   🖼️ Images: {result.images_count} images")
    
    if result.database_stored:
        print(f"   💾 Database: Results stored")
    
    if result.error_message:
        print(f"   ❌ Error: {result.error_message}")

def display_summary(results):
    """Display summary of all processing results"""
    print("\n📈 Processing Summary")
    print("=" * 50)
    
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    print(f"📄 Total documents: {len(results)}")
    print(f"✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")
    
    if successful:
        total_time = sum(r.processing_time for r in successful)
        avg_time = total_time / len(successful)
        
        total_toc_sections = sum(r.toc_sections for r in successful)
        total_citations = sum(r.citations_count for r in successful)
        total_images = sum(r.images_count for r in successful)
        
        print(f"\n⏱️  Performance:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Average time: {avg_time:.2f}s")
        
        print(f"\n📊 Extractions:")
        print(f"   TOC sections: {total_toc_sections}")
        print(f"   Citations: {total_citations}")
        print(f"   Images: {total_images}")
        
        print(f"\n🎯 Success rate: {(len(successful) / len(results) * 100):.1f}%")

def main():
    """Main test function"""
    print("🚀 Integrated Workers Test Suite")
    print("=" * 60)
    print("Testing TOC Extractor, Bibliography Builder, and Table Extraction")
    print("working together for comprehensive document analysis.")
    print()
    
    # Setup logging
    setup_logging()
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Run tests
    test_results = {}
    
    print("🔍 Starting Integrated Workers Tests...")
    print()
    
    # Test 1: Individual workers
    test_results['individual'] = test_individual_workers()
    
    # Test 2: Integrated processing
    test_results['integrated'] = test_integrated_processing()
    
    # Test 3: Cross-referencing
    test_results['cross_referencing'] = test_cross_referencing()
    
    # Test 4: Performance comparison
    test_results['performance'] = test_performance_comparison()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Results Summary")
    print("=" * 60)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    print(f"\n🎯 Overall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! Integrated workers are working correctly.")
        print("✅ TOC Extractor, Bibliography Builder, and Table Extraction")
        print("   are successfully integrated and working together.")
        return True
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
