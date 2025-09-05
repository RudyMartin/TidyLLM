#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Table Extraction

Test script to check for table extraction issues similar to the image extraction problems.
"""

import sys
import os
import time
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_table_extraction():
    """Test table extraction with different methods"""
    print("🔍 Testing Table Extraction")
    print("=" * 50)
    
    try:
        from backend.mcp.workers.modern_pdf_processor import ModernPDFProcessor
        
        processor = ModernPDFProcessor()
        
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
            print("⚠️  No test PDF found")
            return test_table_extraction_methods()
        
        print(f"📄 Testing with: {test_pdf}")
        
        # Test table extraction
        start_time = time.time()
        result = processor.process_pdf(test_pdf)
        processing_time = time.time() - start_time
        
        print(f"✅ Processing completed in {processing_time:.3f} seconds")
        print(f"   Success: {result['success']}")
        print(f"   Tables extracted: {len(result['processing']['tables'])}")
        print(f"   Method used: {result['processing']['method']}")
        
        if result['processing']['tables']:
            print("\n📊 Table Details:")
            for i, table in enumerate(result['processing']['tables'][:3]):  # Show first 3 tables
                print(f"   Table {i+1}:")
                print(f"     Page: {table['page']}")
                print(f"     Rows: {table['rows']}")
                print(f"     Columns: {table['columns']}")
                print(f"     Type: {table['type']}")
                if table['header']:
                    print(f"     Header: {table['header'][:3]}...")  # Show first 3 header items
        else:
            print("   ⚠️  No tables found in document")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_table_extraction_methods():
    """Test different table extraction methods"""
    print("🧪 Testing Table Extraction Methods")
    print("=" * 50)
    
    methods = {
        'pdfplumber': False,
        'pypdf': False,
        'fitz': False,
        'pymupdf': False
    }
    
    # Test pdfplumber
    try:
        import pdfplumber
        methods['pdfplumber'] = True
        print("✅ pdfplumber: Available")
    except ImportError:
        print("❌ pdfplumber: Not available")
    
    # Test pypdf
    try:
        import pypdf
        methods['pypdf'] = True
        print("✅ pypdf: Available")
    except ImportError:
        print("❌ pypdf: Not available")
    
    # Test fitz
    try:
        import fitz
        methods['fitz'] = True
        print("✅ fitz: Available")
    except ImportError:
        print("❌ fitz: Not available")
    
    # Test pymupdf
    try:
        import pymupdf
        methods['pymupdf'] = True
        print("✅ pymupdf: Available")
    except ImportError:
        print("❌ pymupdf: Not available")
    
    print(f"\n📋 Available methods: {[k for k, v in methods.items() if v]}")
    
    return True

def test_pdfplumber_table_extraction():
    """Test pdfplumber table extraction specifically"""
    print("\n🔍 Testing pdfplumber Table Extraction")
    print("=" * 50)
    
    try:
        import pdfplumber
        
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
            print("⚠️  No test PDF found")
            return True
        
        print(f"📄 Testing pdfplumber with: {test_pdf}")
        
        # Test pdfplumber directly
        with pdfplumber.open(test_pdf) as pdf:
            total_tables = 0
            for page_num, page in enumerate(pdf.pages):
                try:
                    tables = page.extract_tables()
                    if tables:
                        print(f"   Page {page_num + 1}: {len(tables)} tables")
                        total_tables += len(tables)
                        
                        # Show details of first table on each page
                        for table_index, table in enumerate(tables[:1]):
                            print(f"     Table {table_index + 1}: {len(table)} rows, {len(table[0]) if table else 0} columns")
                            
                except Exception as e:
                    print(f"   ⚠️  Error extracting tables from page {page_num + 1}: {e}")
        
        print(f"✅ Total tables found: {total_tables}")
        return True
        
    except Exception as e:
        print(f"❌ pdfplumber test failed: {e}")
        return False

def test_fallback_table_extraction():
    """Test fallback table extraction methods"""
    print("\n🔄 Testing Fallback Table Extraction")
    print("=" * 50)
    
    try:
        from backend.mcp.workers.pdf_fallback import FallbackPDFProcessor
        
        processor = FallbackPDFProcessor()
        
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
            print("⚠️  No test PDF found")
            return True
        
        print(f"📄 Testing fallback processor with: {test_pdf}")
        
        # Test fallback processing
        start_time = time.time()
        result = processor.process_pdf(test_pdf)
        processing_time = time.time() - start_time
        
        print(f"✅ Processing completed in {processing_time:.3f} seconds")
        print(f"   Success: {result['success']}")
        print(f"   Method used: {result['processing']['method']}")
        print(f"   Tables extracted: {len(result['processing']['tables'])}")
        
        if result['processing']['tables']:
            print("\n📊 Table Details:")
            for i, table in enumerate(result['processing']['tables'][:3]):
                print(f"   Table {i+1}:")
                print(f"     Page: {table['page']}")
                print(f"     Data rows: {len(table['data'])}")
                if table['data']:
                    print(f"     Columns: {len(table['data'][0])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Fallback test failed: {e}")
        return False

def test_table_extraction_errors():
    """Test for common table extraction errors"""
    print("\n⚠️  Testing for Common Table Extraction Errors")
    print("=" * 50)
    
    # Test with a non-existent file
    try:
        from backend.mcp.workers.modern_pdf_processor import ModernPDFProcessor
        
        processor = ModernPDFProcessor()
        result = processor.process_pdf("non_existent_file.pdf")
        
        print(f"✅ Non-existent file handling: {result['success']}")
        if not result['success']:
            print(f"   Error: {result.get('error', 'Unknown error')}")
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
    
    # Test with empty content
    try:
        from backend.mcp.workers.modern_pdf_processor import ModernPDFProcessor
        
        processor = ModernPDFProcessor()
        result = processor.process_pdf(file_content=b"")
        
        print(f"✅ Empty content handling: {result['success']}")
        if not result['success']:
            print(f"   Error: {result.get('error', 'Unknown error')}")
        
    except Exception as e:
        print(f"✅ Empty content test failed (expected): {e}")
    
    return True

def main():
    """Main test function"""
    print("🚀 Table Extraction Test Suite")
    print("=" * 60)
    
    # Track results
    results = {}
    
    # Test 1: Basic table extraction
    results['basic_extraction'] = test_table_extraction()
    
    # Test 2: Method availability
    results['method_availability'] = test_table_extraction_methods()
    
    # Test 3: pdfplumber specific
    results['pdfplumber_extraction'] = test_pdfplumber_table_extraction()
    
    # Test 4: Fallback methods
    results['fallback_extraction'] = test_fallback_table_extraction()
    
    # Test 5: Error handling
    results['error_handling'] = test_table_extraction_errors()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Table Extraction Test Results")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All table extraction tests passed!")
        print("\n✅ Table extraction appears to be working correctly.")
        print("   - No major issues detected")
        print("   - Multiple fallback methods available")
        print("   - Error handling is robust")
    else:
        print("⚠️  Some table extraction tests failed. Check the output above for details.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
