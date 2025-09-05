#!/usr/bin/env python3
"""
Test Enhanced Embedding System with Model Tracking

This script tests the enhanced embedding helper with comprehensive
model tracking and dimension management.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from backend.core.embedding_helper import EmbeddingHelper, EmbeddingMetadata
import numpy as np

def test_enhanced_embedding_system():
    """Test the enhanced embedding system"""
    
    print("Testing Enhanced Embedding System")
    print("=" * 50)
    
    # Test 1: Initialize embedding helper
    print("\n1. Initializing Embedding Helper...")
    try:
        embedding_helper = EmbeddingHelper(target_dimensions=1024)
        print("✅ Embedding helper initialized successfully")
        
        # Get model info
        model_info = embedding_helper.get_model_info()
        print(f"   Model: {model_info['model_metadata']['model_name']}")
        print(f"   Original dimensions: {model_info['model_metadata']['original_dimensions']}")
        print(f"   Target dimensions: {model_info['model_metadata']['target_dimensions']}")
        print(f"   Needs padding: {model_info['model_metadata']['needs_padding']}")
        print(f"   Padding dimensions: {model_info['model_metadata']['padding_dimensions']}")
        
    except Exception as e:
        print(f"❌ Failed to initialize embedding helper: {e}")
        return
    
    # Test 2: Generate single embedding
    print("\n2. Testing Single Embedding Generation...")
    test_text = "This is a test document for embedding generation."
    
    try:
        embedding, metadata = embedding_helper.generate_embedding(test_text, "test_chunk_1")
        
        print(f"✅ Generated embedding successfully")
        print(f"   Embedding shape: {embedding.shape}")
        print(f"   Target dimensions: {len(embedding)}")
        print(f"   Model used: {metadata.model_name}")
        print(f"   Original dimensions: {metadata.original_dimensions}")
        print(f"   Padding method: {metadata.padding_method}")
        print(f"   Content hash: {metadata.content_hash[:16]}...")
        
        # Validate dimensions
        if len(embedding) == 1024:
            print("   ✅ Dimensions match target (1024)")
        else:
            print(f"   ❌ Dimension mismatch: {len(embedding)} != 1024")
            
    except Exception as e:
        print(f"❌ Failed to generate embedding: {e}")
    
    # Test 3: Generate batch embeddings
    print("\n3. Testing Batch Embedding Generation...")
    test_texts = [
        "First document for batch processing.",
        "Second document with different content.",
        "Third document to test multiple embeddings."
    ]
    content_ids = ["batch_1", "batch_2", "batch_3"]
    
    try:
        embeddings, metadata_list = embedding_helper.generate_batch_embeddings(test_texts, content_ids)
        
        print(f"✅ Generated {len(embeddings)} batch embeddings")
        
        for i, (embedding, metadata) in enumerate(zip(embeddings, metadata_list)):
            print(f"   Embedding {i+1}: {len(embedding)} dimensions ({metadata.padding_method})")
            
        # Check all have correct dimensions
        all_correct = all(len(emb) == 1024 for emb in embeddings)
        if all_correct:
            print("   ✅ All embeddings have correct dimensions")
        else:
            print("   ❌ Some embeddings have incorrect dimensions")
            
    except Exception as e:
        print(f"❌ Failed to generate batch embeddings: {e}")
    
    # Test 4: Validate embedding dimensions
    print("\n4. Testing Dimension Validation...")
    try:
        # Test valid embedding
        valid_embedding = np.random.random(1024)
        is_valid = embedding_helper.validate_embedding_dimensions(valid_embedding)
        print(f"   Valid embedding (1024): {'✅' if is_valid else '❌'}")
        
        # Test invalid embedding
        invalid_embedding = np.random.random(768)
        is_valid = embedding_helper.validate_embedding_dimensions(invalid_embedding)
        print(f"   Invalid embedding (768): {'❌' if not is_valid else '✅'}")
        
    except Exception as e:
        print(f"❌ Failed to validate dimensions: {e}")
    
    # Test 5: Test different text lengths
    print("\n5. Testing Different Text Lengths...")
    test_cases = [
        ("Short text", "Short"),
        ("Medium length text with more content to process", "Medium"),
        ("Very long text " * 50, "Long")
    ]
    
    for text, desc in test_cases:
        try:
            embedding, metadata = embedding_helper.generate_embedding(text, f"test_{desc.lower()}")
            print(f"   {desc} text: {len(embedding)} dimensions ({metadata.padding_method})")
        except Exception as e:
            print(f"   ❌ Failed with {desc} text: {e}")
    
    print("\n" + "=" * 50)
    print("Enhanced Embedding System Test Complete!")

if __name__ == "__main__":
    test_enhanced_embedding_system()
