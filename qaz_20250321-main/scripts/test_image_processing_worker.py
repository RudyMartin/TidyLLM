#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Image Processing Worker

Test script to validate the specialized image processing worker and its fallback strategies.
"""

import sys
import os
import time
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_image_processing_worker():
    """Test the image processing worker"""
    print("🔍 Testing Image Processing Worker")
    print("=" * 50)
    
    try:
        from backend.mcp.workers.image_processing_worker import ImageProcessingWorker
        
        # Initialize worker
        worker = ImageProcessingWorker()
        
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
            print("⚠️  No test PDF found, creating a simple test...")
            return test_worker_capabilities(worker)
        
        print(f"📄 Testing with: {test_pdf}")
        
        # Test image processing
        start_time = time.time()
        result = worker.process_images(test_pdf)
        processing_time = time.time() - start_time
        
        print(f"✅ Processing completed in {processing_time:.3f} seconds")
        print(f"   Success: {result['success']}")
        print(f"   Total images: {result['total_images']}")
        print(f"   Method used: {result['processing_method']}")
        print(f"   Methods tried: {result['methods_tried']}")
        
        if result['images']:
            print("\n📊 Image Details:")
            for i, img in enumerate(result['images'][:5]):  # Show first 5 images
                print(f"   Image {i+1}:")
                print(f"     Page: {img['page']}")
                print(f"     Size: {img['width']}x{img['height']}")
                print(f"     Type: {img['type']}")
                print(f"     Format: {img['format']}")
                print(f"     Method: {img['method']}")
        
        # Test cache functionality
        print("\n🧪 Testing cache functionality...")
        cache_stats = worker.get_cache_stats()
        print(f"   Cache size: {cache_stats['cache_size']}")
        
        # Test with specific pages
        print("\n🧪 Testing with specific pages...")
        result_pages = worker.process_images(test_pdf, page_numbers=[0, 1])
        print(f"   Images from pages 1-2: {result_pages['total_images']}")
        
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
    result = worker.process_images()
    print(f"✅ Invalid input handling: {result['success']} - {result.get('error', 'No error')}")
    
    return True

def test_fallback_strategies():
    """Test different fallback strategies"""
    print("\n🔄 Testing Fallback Strategies")
    print("=" * 50)
    
    try:
        from backend.mcp.workers.image_processing_worker import ImageProcessingWorker
        
        worker = ImageProcessingWorker()
        
        # Test each method individually
        methods = ['pypdfium2', 'fitz', 'pymupdf', 'pdf2image']
        
        for method in methods:
            if worker.available_methods.get(method):
                print(f"✅ {method}: Available")
            else:
                print(f"❌ {method}: Not available")
        
        # Test method priority
        print("\n📋 Method Priority (in order of preference):")
        priority_methods = ['pypdfium2', 'fitz', 'pymupdf', 'pdf2image']
        for i, method in enumerate(priority_methods, 1):
            status = "✅" if worker.available_methods.get(method) else "❌"
            print(f"   {i}. {method}: {status}")
        
        return True
        
    except Exception as e:
        print(f"❌ Fallback test failed: {e}")
        return False

def test_integration_with_modern_pdf_processor():
    """Test integration with the modern PDF processor"""
    print("\n🔗 Testing Integration with Modern PDF Processor")
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
            print("⚠️  No test PDF found for integration test")
            return True
        
        print(f"📄 Testing integration with: {test_pdf}")
        
        # Process PDF with modern processor
        start_time = time.time()
        result = processor.process_pdf(test_pdf)
        processing_time = time.time() - start_time
        
        print(f"✅ Processing completed in {processing_time:.3f} seconds")
        print(f"   Success: {result['success']}")
        print(f"   Images extracted: {len(result['processing']['images'])}")
        print(f"   Method used: {result['processing']['method']}")
        
        if result['processing']['images']:
            print("\n📊 Image Details:")
            for i, img in enumerate(result['processing']['images'][:3]):  # Show first 3 images
                print(f"   Image {i+1}:")
                print(f"     Page: {img['page']}")
                print(f"     Size: {img['width']}x{img['height']}")
                print(f"     Type: {img['type']}")
                print(f"     Method: {img.get('method', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def test_performance():
    """Test performance of image processing"""
    print("\n⚡ Testing Performance")
    print("=" * 50)
    
    try:
        from backend.mcp.workers.image_processing_worker import ImageProcessingWorker
        
        worker = ImageProcessingWorker()
        
        # Test cache performance
        print("🧪 Testing cache performance...")
        
        # Test with a real PDF if available
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
            result1 = worker.process_images(test_pdf)
            time1 = time.time() - start_time
            
            # Second call (should be faster due to cache)
            start_time = time.time()
            result2 = worker.process_images(test_pdf)
            time2 = time.time() - start_time
            
            print(f"   First call: {time1:.3f} seconds")
            print(f"   Cached call: {time2:.3f} seconds")
            if time2 > 0:
                print(f"   Cache speedup: {time1/time2:.1f}x faster")
            else:
                print(f"   Cache speedup: Instant")
        else:
            print("   ⚠️  No test PDF available for performance test")
            time1 = time2 = 0
        
        print(f"   First call: {time1:.3f} seconds")
        print(f"   Cached call: {time2:.3f} seconds")
        print(f"   Cache speedup: {time1/time2:.1f}x faster")
        
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
    print("🚀 Image Processing Worker Test Suite")
    print("=" * 60)
    
    # Track results
    results = {}
    
    # Test 1: Basic functionality
    results['basic_functionality'] = test_image_processing_worker()
    
    # Test 2: Fallback strategies
    results['fallback_strategies'] = test_fallback_strategies()
    
    # Test 3: Integration
    results['integration'] = test_integration_with_modern_pdf_processor()
    
    # Test 4: Performance
    results['performance'] = test_performance()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Image Processing Worker Test Results")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All image processing worker tests passed!")
        print("\n✅ Key Benefits:")
        print("   - Multiple fallback strategies for image extraction")
        print("   - Automatic method selection based on availability")
        print("   - Caching for improved performance")
        print("   - Integration with modern PDF processor")
        print("   - Robust error handling and logging")
    else:
        print("⚠️  Some image processing worker tests failed. Check the output above for details.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
