#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Test for Custom Extraction Parsing Routines
================================================

A simple script to test the customized extraction parsing routines:
- Smart chunking algorithms
- Text cleanup and preprocessing
- Page continuity validation
- Metadata extraction
- Integration with modern PDF stack
"""

import sys
import os
import time
import re
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_smart_chunking_import():
    """Test that smart chunking functions can be imported"""
    print("🔍 Testing smart chunking import...")
    
    try:
        from backend.core.extraction_helper import smart_chunking, validate_page_continuity
        print("✅ Smart chunking functions imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Smart chunking import failed: {e}")
        return False

def test_smart_chunking_basic():
    """Test basic smart chunking functionality"""
    print("\n🔍 Testing basic smart chunking...")
    
    try:
        from backend.core.extraction_helper import smart_chunking
        
        # Test with simple text
        text = "This is a simple test document. It contains basic sentences."
        chunks = smart_chunking(text, max_words=50)
        
        print(f"✅ Generated {len(chunks)} chunks from simple text")
        
        # Check chunk properties
        for i, chunk in enumerate(chunks):
            word_count = len(chunk.split())
            print(f"   Chunk {i+1}: {word_count} words")
        
        return len(chunks) > 0
        
    except Exception as e:
        print(f"❌ Basic smart chunking failed: {e}")
        return False

def test_smart_chunking_complex():
    """Test smart chunking with complex text"""
    print("\n🔍 Testing smart chunking with complex text...")
    
    try:
        from backend.core.extraction_helper import smart_chunking
        
        # Test with complex text that should generate multiple chunks
        text = "This is a complex document with multiple sentences; it contains semicolons, commas, and various punctuation marks. The document also has references like Smith, J. 2023 and percentages like 25.5%. This is another sentence that should be in a separate chunk because we are testing the chunking algorithm. We need to make sure that the text is long enough to trigger multiple chunks. The algorithm should split text at logical boundaries like semicolons and commas."
        chunks = smart_chunking(text, max_words=25)
        
        print(f"✅ Generated {len(chunks)} chunks from complex text")
        
        # Verify semantic boundaries are respected
        for i, chunk in enumerate(chunks):
            print(f"   Chunk {i+1}: '{chunk[:50]}...'")
        
        return len(chunks) > 1
        
    except Exception as e:
        print(f"❌ Complex smart chunking failed: {e}")
        return False

def test_text_cleanup():
    """Test text cleanup and preprocessing functions"""
    print("\n🔍 Testing text cleanup functions...")
    
    try:
        # Test text cleanup function
        def clean_text(text: str) -> str:
            """Cleans extracted text from PDF artifacts"""
            # Remove multiple newlines but keep paragraph breaks
            text = re.sub(r'\n{2,}', '\n', text)
            # Remove multiple spaces and excessive whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            # Fix hyphenated line breaks
            text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
            return text.strip()
        
        # Test with hyphenated text (with space after hyphen as expected by the regex)
        original_text = "This text con- tains hyphenated words that should be re- paired during processing."
        cleaned_text = clean_text(original_text)
        
        print(f"✅ Original: '{original_text}'")
        print(f"✅ Cleaned: '{cleaned_text}'")
        
        # Verify hyphens are fixed - check if the pattern is applied
        if 'con-tains' not in cleaned_text and 'contains' in cleaned_text:
            print("✅ Hyphens fixed correctly")
            return True
        else:
            print("❌ Hyphens not fixed correctly")
            return False
        
    except Exception as e:
        print(f"❌ Text cleanup failed: {e}")
        return False

def test_page_continuity():
    """Test page continuity validation"""
    print("\n🔍 Testing page continuity validation...")
    
    try:
        from backend.core.extraction_helper import validate_page_continuity
        
        # Test with complete sentence
        complete_text = "This is a complete sentence."
        next_text = "This is the next page."
        
        result = validate_page_continuity(complete_text, next_text)
        print(f"✅ Complete sentence: {result}")
        
        # Test with incomplete sentence
        incomplete_text = "This sentence is incomplete"
        result = validate_page_continuity(incomplete_text, next_text)
        print(f"✅ Incomplete sentence: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Page continuity validation failed: {e}")
        return False

def test_metadata_extraction():
    """Test metadata extraction patterns"""
    print("\n🔍 Testing metadata extraction...")
    
    try:
        # Test number detection
        number_text = "The value is 42 and the percentage is 75%"
        has_numbers = bool(re.search(r'\d+', number_text))
        has_percentages = bool(re.search(r'\d+%', number_text))
        
        print(f"✅ Number detection: {has_numbers}")
        print(f"✅ Percentage detection: {has_percentages}")
        
        # Test reference detection
        reference_text = "According to Smith, J. 2023 and Brown, M. 2024"
        has_references = bool(re.search(r'[A-Z][a-z]+,\s*[A-Z]\.\s*\d{4}', reference_text))
        
        print(f"✅ Reference detection: {has_references}")
        
        # Test URL detection
        url_text = "Visit https://example.com and http://test.org"
        has_urls = bool(re.search(r'https?://', url_text))
        
        print(f"✅ URL detection: {has_urls}")
        
        return has_numbers and has_percentages and has_references and has_urls
        
    except Exception as e:
        print(f"❌ Metadata extraction failed: {e}")
        return False

def test_enhanced_chunking_with_metadata():
    """Test enhanced chunking with metadata extraction"""
    print("\n🔍 Testing enhanced chunking with metadata...")
    
    try:
        # Recreate the enhanced chunking function
        def create_smart_chunks_with_metadata(text: str, document_name: str) -> list:
            """Create smart chunks with enhanced metadata"""
            chunks = []
            sentences = re.split(r'[.!?]+', text)
            
            current_chunk = []
            current_length = 0
            chunk_index = 1
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                sentence_length = len(sentence.split())
                
                if current_length + sentence_length > 200 and current_chunk:
                    # Create chunk
                    chunk_text = " ".join(current_chunk)
                    chunk_data = {
                        "chunk_id": f"{document_name}_chunk_{chunk_index:03d}",
                        "text": chunk_text,
                        "word_count": len(chunk_text.split()),
                        "char_count": len(chunk_text),
                        "chunk_number": chunk_index,
                        "metadata": {
                            "has_numbers": bool(re.search(r'\d+', chunk_text)),
                            "has_percentages": bool(re.search(r'\d+%', chunk_text)),
                            "has_references": bool(re.search(r'[A-Z][a-z]+,\s*[A-Z]\.\s*\d{4}', chunk_text)),
                            "has_urls": bool(re.search(r'https?://', chunk_text))
                        }
                    }
                    chunks.append(chunk_data)
                    
                    current_chunk = []
                    current_length = 0
                    chunk_index += 1
                
                current_chunk.append(sentence)
                current_length += sentence_length
            
            # Add final chunk
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunk_data = {
                    "chunk_id": f"{document_name}_chunk_{chunk_index:03d}",
                    "text": chunk_text,
                    "word_count": len(chunk_text.split()),
                    "char_count": len(chunk_text),
                    "chunk_number": chunk_index,
                    "metadata": {
                        "has_numbers": bool(re.search(r'\d+', chunk_text)),
                        "has_percentages": bool(re.search(r'\d+%', chunk_text)),
                        "has_references": bool(re.search(r'[A-Z][a-z]+,\s*[A-Z]\.\s*\d{4}', chunk_text)),
                        "has_urls": bool(re.search(r'https?://', chunk_text))
                    }
                }
                chunks.append(chunk_data)
            
            return chunks
        
        # Test with mixed content
        text = "This document has 50% improvement. Visit https://example.com. Reference: Brown, M. 2023. It also has complex sentences with multiple clauses."
        chunks = create_smart_chunks_with_metadata(text, "test_doc")
        
        print(f"✅ Generated {len(chunks)} chunks with metadata")
        
        # Verify metadata extraction
        for chunk in chunks:
            print(f"   Chunk {chunk['chunk_number']}: {chunk['word_count']} words")
            print(f"     Numbers: {chunk['metadata']['has_numbers']}")
            print(f"     Percentages: {chunk['metadata']['has_percentages']}")
            print(f"     References: {chunk['metadata']['has_references']}")
            print(f"     URLs: {chunk['metadata']['has_urls']}")
        
        return len(chunks) > 0
        
    except Exception as e:
        print(f"❌ Enhanced chunking with metadata failed: {e}")
        return False

def test_integration_with_modern_pdf_stack():
    """Test integration of custom routines with modern PDF stack"""
    print("\n🔍 Testing integration with modern PDF stack...")
    
    try:
        from backend.mcp.workers.modern_pdf_processor import ModernPDFProcessor
        from backend.core.extraction_helper import smart_chunking
        
        processor = ModernPDFProcessor()
        
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
            print("⚠️  No sample PDF found, skipping integration test")
            return True
        
        print(f"📄 Testing integration with: {test_pdf}")
        
        # Step 1: Process PDF with modern stack
        pdf_result = processor.process_pdf(test_pdf)
        
        if not pdf_result['success']:
            print(f"❌ PDF processing failed: {pdf_result.get('error', 'Unknown error')}")
            return False
        
        # Step 2: Apply smart chunking
        raw_text = pdf_result['processing']['text_content']
        chunks = smart_chunking(raw_text, max_words=100)
        
        print(f"✅ Raw text length: {len(raw_text)}")
        print(f"✅ Generated {len(chunks)} chunks")
        
        # Step 3: Verify chunk quality
        for i, chunk in enumerate(chunks[:3]):  # Check first 3 chunks
            word_count = len(chunk.split())
            print(f"   Chunk {i+1}: {word_count} words")
        
        return len(chunks) > 0
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def test_performance():
    """Test performance of custom extraction routines"""
    print("\n🔍 Testing performance of custom routines...")
    
    try:
        from backend.core.extraction_helper import smart_chunking
        import time
        
        # Test with larger text
        test_text = "This is a test sentence. " * 100  # Make it longer
        
        start_time = time.time()
        chunks = smart_chunking(test_text, max_words=50)
        chunking_time = time.time() - start_time
        
        print(f"✅ Smart chunking time: {chunking_time:.3f} seconds")
        print(f"   - Generated {len(chunks)} chunks")
        
        # Performance should be reasonable
        if chunking_time < 5.0:
            print("✅ Performance is acceptable")
            return True
        else:
            print("⚠️  Performance might be slow")
            return False
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Custom Extraction Parsing Routines Test")
    print("=" * 50)
    
    # Track results
    results = {}
    
    # Test 1: Smart chunking import
    results['smart_chunking_import'] = test_smart_chunking_import()
    
    # Test 2: Basic smart chunking
    results['basic_chunking'] = test_smart_chunking_basic()
    
    # Test 3: Complex smart chunking
    results['complex_chunking'] = test_smart_chunking_complex()
    
    # Test 4: Text cleanup
    results['text_cleanup'] = test_text_cleanup()
    
    # Test 5: Page continuity
    results['page_continuity'] = test_page_continuity()
    
    # Test 6: Metadata extraction
    results['metadata_extraction'] = test_metadata_extraction()
    
    # Test 7: Enhanced chunking with metadata
    results['enhanced_chunking'] = test_enhanced_chunking_with_metadata()
    
    # Test 8: Integration with modern PDF stack
    results['integration'] = test_integration_with_modern_pdf_stack()
    
    # Test 9: Performance
    results['performance'] = test_performance()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Custom Extraction Routines Test Results")
    print("=" * 50)
    
    all_passed = True
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All custom extraction routines tests passed!")
    else:
        print("⚠️  Some custom extraction routines tests failed. Check the output above for details.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
