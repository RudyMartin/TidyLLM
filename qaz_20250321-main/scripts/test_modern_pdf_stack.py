#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Test for Modern PDF Stack and Embeddings
==============================================

A simple script to test the modern PDF processing stack and embedding generation.
This is useful for quick validation during development and deployment.
"""

import sys
import os
import time
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_modern_pdf_libraries():
    """Test that all modern PDF libraries can be imported and used"""
    print("🔍 Testing modern PDF libraries...")
    
    results = {}
    
    # Test pdfplumber
    try:
        import pdfplumber
        print(f"✅ pdfplumber: {pdfplumber.__version__}")
        results['pdfplumber'] = True
    except ImportError as e:
        print(f"❌ pdfplumber: {e}")
        results['pdfplumber'] = False
    
    # Test pypdfium2
    try:
        import pypdfium2
        print("✅ pypdfium2: Available")
        results['pypdfium2'] = True
    except ImportError as e:
        print(f"❌ pypdfium2: {e}")
        results['pypdfium2'] = False
    
    # Test pypdf
    try:
        import pypdf
        print(f"✅ pypdf: {pypdf.__version__}")
        results['pypdf'] = True
    except ImportError as e:
        print(f"❌ pypdf: {e}")
        results['pypdf'] = False
    
    return results

def test_modern_pdf_processor():
    """Test the modern PDF processor"""
    print("\n🔍 Testing modern PDF processor...")
    
    try:
        from backend.mcp.workers.modern_pdf_processor import ModernPDFProcessor
        
        processor = ModernPDFProcessor()
        print("✅ ModernPDFProcessor: Imported successfully")
        
        # Check available methods
        methods = processor.available_methods
        print(f"✅ Available methods: {list(methods.keys())}")
        
        return True
        
    except ImportError as e:
        print(f"❌ ModernPDFProcessor: {e}")
        return False
    except Exception as e:
        print(f"❌ ModernPDFProcessor error: {e}")
        return False

def test_embedding_libraries():
    """Test embedding libraries"""
    print("\n🔍 Testing embedding libraries...")
    
    results = {}
    
    # Test sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ sentence-transformers: Available")
        results['sentence_transformers'] = True
    except ImportError as e:
        print(f"❌ sentence-transformers: {e}")
        results['sentence_transformers'] = False
    
    # Test numpy
    try:
        import numpy as np
        print(f"✅ numpy: {np.__version__}")
        results['numpy'] = True
    except ImportError as e:
        print(f"❌ numpy: {e}")
        results['numpy'] = False
    
    return results

def test_embedding_generation():
    """Test embedding generation"""
    print("\n🔍 Testing embedding generation...")
    
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        # Test texts
        test_texts = [
            "This is a test document for embedding generation.",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models require high-quality training data."
        ]
        
        # Load model using centralized EmbeddingHelper
        from backend.core.embedding_helper import EmbeddingHelper
        embedding_helper = EmbeddingHelper(target_dimensions=1024)
        print("✅ Model loaded successfully using centralized EmbeddingHelper")
        
        # Generate embeddings
        start_time = time.time()
        embeddings = []
        for text in test_texts:
            embedding, metadata = embedding_helper.generate_embedding(text, f"test_{len(embeddings)}")
            embeddings.append(embedding)
        embeddings = np.array(embeddings)
        generation_time = time.time() - start_time
        
        print(f"✅ Generated {len(embeddings)} embeddings")
        print(f"   - Dimension: {embeddings[0].shape[0]} (target: 1024)")
        print(f"   - Time: {generation_time:.3f} seconds")
        
        # Test similarity
        similarities = np.dot(embeddings, embeddings.T)
        print(f"✅ Similarity matrix: {similarities.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        return False

def test_pdf_processing_with_sample():
    """Test PDF processing with a sample document"""
    print("\n🔍 Testing PDF processing...")
    
    try:
        from backend.mcp.workers.modern_pdf_processor import ModernPDFProcessor
        
        processor = ModernPDFProcessor()
        
        # Look for a sample PDF in common locations
        sample_paths = [
            "data/input/reviews/Whitepaper-Model-Validation-Best-Practices-1.pdf",
            "input/omnibus/all/2309.11495 - Chain-of-Verification Reduces Hallucination in Large Language Models.pdf",
            "tests/test_data/sample_document.pdf"
        ]
        
        test_pdf = None
        for path in sample_paths:
            if os.path.exists(path):
                test_pdf = path
                break
        
        if not test_pdf:
            print("⚠️  No sample PDF found, skipping PDF processing test")
            return True
        
        print(f"📄 Testing with: {test_pdf}")
        
        # Process PDF
        start_time = time.time()
        result = processor.process_pdf(test_pdf)
        processing_time = time.time() - start_time
        
        if result['success']:
            processing = result['processing']
            print(f"✅ PDF processing successful")
            print(f"   - Pages: {processing['page_count']}")
            print(f"   - Text length: {len(processing['text_content'])}")
            print(f"   - Tables: {len(processing['tables'])}")
            print(f"   - Images: {len(processing['images'])}")
            print(f"   - Time: {processing_time:.3f} seconds")
            return True
        else:
            print(f"❌ PDF processing failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ PDF processing test failed: {e}")
        return False

def test_full_pipeline():
    """Test the complete pipeline from PDF to embeddings"""
    print("\n🔍 Testing full pipeline...")
    
    try:
        from backend.mcp.workers.modern_pdf_processor import ModernPDFProcessor
        from backend.core.embedding_helper import EmbeddingHelper
        import numpy as np
        
        # Find a sample PDF
        sample_paths = [
            "data/input/reviews/Whitepaper-Model-Validation-Best-Practices-1.pdf",
            "input/omnibus/all/2309.11495 - Chain-of-Verification Reduces Hallucination in Large Language Models.pdf",
            "tests/test_data/sample_document.pdf"
        ]
        
        test_pdf = None
        for path in sample_paths:
            if os.path.exists(path):
                test_pdf = path
                break
        
        if not test_pdf:
            print("⚠️  No sample PDF found, skipping full pipeline test")
            return True
        
        print(f"📄 Testing full pipeline with: {test_pdf}")
        
        # Step 1: Process PDF
        processor = ModernPDFProcessor()
        pdf_result = processor.process_pdf(test_pdf)
        
        if not pdf_result['success']:
            print(f"❌ PDF processing failed: {pdf_result.get('error', 'Unknown error')}")
            return False
        
        # Step 2: Extract text chunks
        text_content = pdf_result['processing']['text_content']
        
        # Simple chunking
        import re
        sentences = re.split(r'[.!?]+', text_content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        print(f"✅ Extracted {len(sentences)} sentences")
        
        # Step 3: Generate embeddings using centralized EmbeddingHelper
        embedding_helper = EmbeddingHelper(target_dimensions=1024)
        embeddings = []
        for i, sentence in enumerate(sentences[:5]):  # Limit to first 5 sentences
            embedding, metadata = embedding_helper.generate_embedding(sentence, f"sentence_{i}")
            embeddings.append(embedding)
        embeddings = np.array(embeddings)
        
        print(f"✅ Generated {len(embeddings)} embeddings (dimension: {embeddings[0].shape[0]})")
        
        # Step 4: Test similarity search
        query = "test document"
        query_embedding, query_metadata = embedding_helper.generate_embedding(query, "query")
        query_embedding = query_embedding.reshape(1, -1)
        
        similarities = np.dot(embeddings, query_embedding.T).flatten()
        most_similar_idx = np.argmax(similarities)
        
        print(f"✅ Similarity search completed")
        print(f"   - Query: '{query}'")
        print(f"   - Most similar score: {similarities[most_similar_idx]:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Modern PDF Stack and Embeddings Test")
    print("=" * 50)
    
    # Track results
    results = {}
    
    # Test 1: Modern PDF libraries
    results['pdf_libraries'] = test_modern_pdf_libraries()
    
    # Test 2: Modern PDF processor
    results['pdf_processor'] = test_modern_pdf_processor()
    
    # Test 3: Embedding libraries
    results['embedding_libraries'] = test_embedding_libraries()
    
    # Test 4: Embedding generation
    results['embedding_generation'] = test_embedding_generation()
    
    # Test 5: PDF processing
    results['pdf_processing'] = test_pdf_processing_with_sample()
    
    # Test 6: Full pipeline
    results['full_pipeline'] = test_full_pipeline()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    all_passed = True
    for test_name, result in results.items():
        if isinstance(result, dict):
            # For tests that return multiple results
            passed = all(result.values())
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{test_name}: {status}")
            if not passed:
                all_passed = False
        else:
            # For tests that return single boolean
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name}: {status}")
            if not result:
                all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Modern PDF stack is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
