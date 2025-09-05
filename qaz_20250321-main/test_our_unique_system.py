#!/usr/bin/env python3
"""
Test Our Unique Document Analysis System

This script demonstrates our custom parsing routines, embedding system,
and complete document analysis flow for the demo.
"""

import sys
import os
from pathlib import Path
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_1_our_document_processor():
    """Test our custom document processor"""
    print("\n" + "="*60)
    print("🎯 STEP 1: OUR UNIQUE DOCUMENT PROCESSOR")
    print("="*60)
    
    try:
        from backend.core.document_processor import DocumentProcessor
        
        # Initialize our custom processor
        processor = DocumentProcessor(input_dir="input/tests", output_dir="output/tests")
        print("✅ Custom DocumentProcessor initialized")
        
        # Scan for documents using our routine
        documents = processor.scan_input_directory()
        print(f"📁 Found {len(documents)} documents in input/tests")
        
        for doc in documents:
            print(f"   📄 {doc.name} ({doc.stat().st_size} bytes)")
        
        return processor, documents
        
    except Exception as e:
        print(f"❌ Document processor failed: {e}")
        return None, []

def test_2_our_extraction_helper():
    """Test our custom extraction helper"""
    print("\n" + "="*60)
    print("🎯 STEP 2: OUR UNIQUE EXTRACTION HELPER")
    print("="*60)
    
    try:
        from backend.core.extraction_helper import clean_text
        from backend.core.document_processor import DocumentProcessor
        
        # Initialize processor
        processor = DocumentProcessor(input_dir="input/tests", output_dir="output/tests")
        
        test_pdf = Path("input/tests/test_document.pdf")
        if test_pdf.exists():
            print(f"📄 Processing: {test_pdf.name}")
            
            # Use our custom extraction from DocumentProcessor
            text_content, metadata = processor.extract_text_from_pdf(test_pdf)
            print(f"✅ Raw extraction: {len(text_content)} characters")
            
            # Use our custom text cleaning
            cleaned_text = clean_text(text_content)
            print(f"✅ Cleaned text: {len(cleaned_text)} characters")
            
            # Show extraction metadata
            print(f"📊 Extraction method: {metadata.get('extraction_method', 'unknown')}")
            print(f"📊 Pages: {metadata.get('pages', 0)}")
            print(f"📊 File size: {metadata.get('file_size', 0)} bytes")
            
            # Show sample of cleaned text
            sample = cleaned_text[:300] + "..." if len(cleaned_text) > 300 else cleaned_text
            print(f"📝 Sample text: {sample}")
            
            return cleaned_text, metadata
        else:
            print("❌ Test document not found")
            return None, {}
            
    except Exception as e:
        print(f"❌ Extraction helper failed: {e}")
        return None, {}

def test_3_our_embedding_system():
    """Test our custom embedding system"""
    print("\n" + "="*60)
    print("🎯 STEP 3: OUR UNIQUE EMBEDDING SYSTEM")
    print("="*60)
    
    try:
        from backend.core.embedding_helper import EmbeddingHelper
        
        # Initialize our custom embedding system
        embedding_helper = EmbeddingHelper()
        print(f"✅ EmbeddingHelper initialized (dimension: {embedding_helper.get_dimension()})")
        
        # Test text for embedding
        test_texts = [
            "This is a sample document about machine learning.",
            "The document discusses neural networks and deep learning.",
            "Key topics include artificial intelligence and data science."
        ]
        
        print(f"🔢 Generating embeddings for {len(test_texts)} text segments...")
        embeddings = embedding_helper.generate_embeddings(test_texts)
        
        print(f"✅ Generated {len(embeddings)} embeddings")
        print(f"📊 Each embedding has {len(embeddings[0])} dimensions")
        
        return embedding_helper, embeddings
        
    except Exception as e:
        print(f"❌ Embedding system failed: {e}")
        return None, []

def test_4_our_favorites_prompt():
    """Test our favorites prompt with our custom system"""
    print("\n" + "="*60)
    print("🎯 STEP 4: OUR FAVORITES PROMPT INTEGRATION")
    print("="*60)
    
    try:
        from scripts.demo_favorites_prompt import FavoritesPromptDemo
        
        # Initialize our favorites prompt demo
        demo = FavoritesPromptDemo()
        print("✅ FavoritesPromptDemo initialized")
        
        # Run the demo with our custom processing
        print("🚀 Running favorites prompt demo...")
        results = demo.run_demo(1)
        
        print(f"📄 Source papers: {results['source_papers']}")
        print(f"📖 TOC sections: {results['toc_sections']}")
        print(f"🔍 References found: {results['references_found']}")
        print(f"✅ High-quality papers: {results['high_quality_papers']}")
        print(f"📥 Papers downloaded: {results['papers_downloaded']}")
        print(f"📋 Report: {results['report_path']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Favorites prompt failed: {e}")
        return None

def test_5_complete_demo_flow():
    """Test the complete demo flow"""
    print("\n" + "="*60)
    print("🎯 STEP 5: COMPLETE DEMO FLOW")
    print("="*60)
    
    print("🔄 Demonstrating complete document analysis workflow:")
    print("   1. Document Upload & Detection")
    print("   2. Custom Text Extraction & Cleaning")
    print("   3. Embedding Generation")
    print("   4. TOC Analysis & Reference Discovery")
    print("   5. Paper Filtering & Download")
    print("   6. Report Generation")
    
    # Run all steps
    processor, documents = test_1_our_document_processor()
    if processor and documents:
        text, metadata = test_2_our_extraction_helper()
        if text:
            embedding_helper, embeddings = test_3_our_embedding_system()
            if embedding_helper:
                results = test_4_our_favorites_prompt()
                if results:
                    print("\n🎉 COMPLETE DEMO FLOW SUCCESSFUL!")
                    print("✅ All components working together")
                    return True
    
    print("\n❌ Demo flow incomplete - some components failed")
    return False

def main():
    """Run the complete test of our unique system"""
    print("🚀 TESTING OUR UNIQUE DOCUMENT ANALYSIS SYSTEM")
    print("="*60)
    print("This demo showcases our custom:")
    print("• Document processing routines")
    print("• Text extraction and cleaning")
    print("• Embedding generation system")
    print("• Favorites prompt integration")
    print("• Complete analysis workflow")
    print("="*60)
    
    # Test complete flow
    success = test_5_complete_demo_flow()
    
    if success:
        print("\n" + "="*60)
        print("🎯 DEMO SUMMARY")
        print("="*60)
        print("✅ Document Processor: Working")
        print("✅ Extraction Helper: Working")
        print("✅ Embedding System: Working")
        print("✅ Favorites Prompt: Working")
        print("✅ Complete Flow: Working")
        print("\n🎉 Our unique system is ready for the Streamlit demo!")
    else:
        print("\n❌ Some components need attention")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
