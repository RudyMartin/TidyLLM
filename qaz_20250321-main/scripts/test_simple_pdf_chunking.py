#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple PDF Chunking Demo

This script demonstrates PDF chunking using pymupdf without NLTK dependencies.
"""

import os
import sys
import re
from pathlib import Path
from typing import List, Dict

def clean_text(text: str) -> str:
    """Cleans extracted text from PDF artifacts"""
    # Remove multiple newlines but keep paragraph breaks
    text = re.sub(r'\n{2,}', '\n', text)
    # Remove multiple spaces and excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Fix hyphenated line breaks
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    return text.strip()

def simple_sentence_split(text: str) -> List[str]:
    """Simple sentence splitting without NLTK"""
    # Split on sentence endings
    sentences = re.split(r'[.!?]+', text)
    # Clean up sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def smart_chunking_simple(text: str, max_words: int = 200) -> List[str]:
    """Smart chunking without NLTK dependency"""
    sentences = simple_sentence_split(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())

        # If sentence is too long, split at logical points
        if sentence_length > max_words:
            sub_sentences = re.split(r'[;,:]|\b(and|but|or|which|that)\b', sentence)
            sub_sentences = [s.strip() for s in sub_sentences if s.strip()]
        else:
            sub_sentences = [sentence]

        for sub_sentence in sub_sentences:
            sub_length = len(sub_sentence.split())

            # Start new chunk if adding this exceeds max_words
            if current_length + sub_length > max_words and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0

            current_chunk.append(sub_sentence)
            current_length += sub_length

    # Add the last chunk if it has enough words
    if current_chunk and len(" ".join(current_chunk).split()) > 5:
        chunks.append(" ".join(current_chunk))

    return chunks

def chunk_text_into_segments(text: str, document_name: str, page_number: int, max_words: int = 200) -> List[Dict]:
    """Create chunks with metadata"""
    try:
        if not text.strip():
            return []

        chunks = []
        chunked_texts = smart_chunking_simple(text, max_words)

        for i, chunk_text in enumerate(chunked_texts):
            chunk_id = f"{document_name}_page_{page_number}_{i+1:03d}"
            chunks.append({
                "chunk_id": chunk_id, 
                "text": chunk_text,
                "metadata": {
                    "document_name": document_name,
                    "page_number": page_number,
                    "chunk_number": i + 1,
                    "total_chunks": len(chunked_texts),
                    "word_count": len(chunk_text.split())
                }
            })

        return chunks

    except Exception as e:
        print(f"❌ Error chunking text: {e}")
        return []

def test_simple_pdf_chunking():
    """Test simple PDF chunking with pymupdf"""
    
    print("📄 Simple PDF Chunking Demo")
    print("=" * 35)
    
    # Select a whitepaper to test
    whitepaper_path = "data/input/reviews/Whitepaper-Model-Validation-Best-Practices-1.pdf"
    
    if not os.path.exists(whitepaper_path):
        print("❌ Whitepaper not found: {}".format(whitepaper_path))
        return False
    
    print("📖 Processing: {}".format(whitepaper_path))
    
    try:
        # Import pymupdf
        import fitz  # pymupdf
        
        # Extract text from PDF
        print("\n🔍 Step 1: Extracting text from PDF using pymupdf...")
        
        doc = fitz.open(whitepaper_path)
        num_pages = len(doc)
        
        print("📄 Found {} pages".format(num_pages))
        
        all_chunks = []
        total_text_length = 0
        
        # Process first few pages for demo
        for page_num in range(min(3, num_pages)):
            page = doc[page_num]
            
            # Extract text
            text = page.get_text()
            text = clean_text(text)
            
            total_text_length += len(text)
            
            print(f"📄 Page {page_num + 1}: {len(text)} characters")
            
            # Create chunks for this page
            document_name = os.path.splitext(os.path.basename(whitepaper_path))[0]
            page_chunks = chunk_text_into_segments(
                text, 
                document_name, 
                page_num + 1, 
                max_words=200
            )
            
            all_chunks.extend(page_chunks)
            print(f"   ✂️ Created {len(page_chunks)} chunks")
        
        doc.close()
        
        print("\n✅ Text extraction and chunking completed")
        print("📊 Total text length: {} characters".format(total_text_length))
        print("📊 Total chunks created: {}".format(len(all_chunks)))
        
        # Show sample chunks
        print("\n📋 Sample Chunks:")
        print("=" * 25)
        
        for i, chunk in enumerate(all_chunks[:3], 1):
            print(f"\n🔹 Chunk {i}:")
            print(f"   ID: {chunk['chunk_id']}")
            print(f"   Page: {chunk['metadata']['page_number']}")
            print(f"   Words: {chunk['metadata']['word_count']}")
            print(f"   Content: {chunk['text'][:150]}...")
        
        # Show statistics
        if all_chunks:
            print("\n📊 Chunking Statistics:")
            print("=" * 25)
            
            word_counts = [chunk['metadata']['word_count'] for chunk in all_chunks]
            
            print(f"Total chunks: {len(all_chunks)}")
            print(f"Average words per chunk: {sum(word_counts) // len(word_counts)}")
            print(f"Largest chunk: {max(word_counts)} words")
            print(f"Smallest chunk: {min(word_counts)} words")
        
        return True
        
    except ImportError as e:
        print("❌ Missing dependency: {}".format(e))
        print("Please install: pip install pymupdf")
        return False
    except Exception as e:
        print("❌ Error during PDF processing: {}".format(e))
        return False

def main():
    """Main function"""
    
    print("🚀 Simple PDF Chunking Demo")
    print("=" * 25)
    
    success = test_simple_pdf_chunking()
    
    if success:
        print("\n🎉 PDF chunking demo completed!")
        print("\n📝 How chunking works:")
        print("1. ✅ PDF text extracted using pymupdf (modern library)")
        print("2. ✅ Text cleaned of PDF artifacts")
        print("3. ✅ Sentences split at punctuation (.!?)")
        print("4. ✅ Long sentences split at logical points (;, :)")
        print("5. ✅ Chunks created with ~200 words each")
        print("6. ✅ Rich metadata for each chunk")
        print("7. ✅ Ready for vector embedding and database storage")
    else:
        print("\n❌ PDF chunking demo failed!")

if __name__ == "__main__":
    main()
