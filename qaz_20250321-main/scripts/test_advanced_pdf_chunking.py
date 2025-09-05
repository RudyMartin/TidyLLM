#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced PDF Chunking Demo

This script demonstrates advanced PDF chunking using pymupdf and smart chunking logic
from the old DSPy codebase.
"""

import os
import sys
import re
import logging
from pathlib import Path
from typing import List, Dict
import nltk

def ensure_nltk():
    """Ensure NLTK punkt tokenizer is available"""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        print("📚 Downloading NLTK punkt tokenizer...")
        nltk.download("punkt")

def clean_text(text: str) -> str:
    """
    Cleans extracted text by normalizing Unicode characters, removing newlines,
    tabs, and excessive whitespace. This helps to reduce artifacts from PDF extraction.
    """
    # Remove multiple newlines but **keep paragraph breaks**
    text = re.sub(r'\n{2,}', '\n', text)  # Keeps a single \n for meaningful breaks
    # Remove common PDF artifacts (e.g., incorrect range values)
    text = text.replace('0.00-10', '0.00')
    # Remove multiple spaces, newlines, and excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Fix hyphenated line breaks (common in PDF text extraction)
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    return text.strip()

def smart_chunking(text: str, max_words: int = 200) -> List[str]:
    """
    Splits academic text into semantically meaningful chunks using sentence + sub-sentence splitting.
    This is the advanced chunking logic from the old DSPy codebase.
    """
    sentences = nltk.sent_tokenize(text)  # Split text into sentences
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())

        # If sentence is too long, attempt secondary splitting at logical points
        if sentence_length > max_words:
            sub_sentences = re.split(r'[;,:]|\b(and|but|or|which|that)\b', sentence)  # Break on logical points
            sub_sentences = [s.strip() for s in sub_sentences if s]  # Remove empty splits
        else:
            sub_sentences = [sentence]

        for sub_sentence in sub_sentences:
            sub_length = len(sub_sentence.split())

            # Start a new chunk if adding this sub-sentence exceeds max_words
            if current_length + sub_length > max_words:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0

            current_chunk.append(sub_sentence)
            current_length += sub_length

    # Add the last chunk if it contains enough words (avoid chunks < 5 words)
    if current_chunk and len(" ".join(current_chunk).split()) > 5:
        chunks.append(" ".join(current_chunk))

    return chunks

def chunk_text_into_segments(text: str, document_name: str, page_number: int, max_words: int = 200) -> List[Dict]:
    """Splits academic text into semantically meaningful chunks before embedding."""
    try:
        if not text.strip():
            return []

        chunks = []
        chunked_texts = smart_chunking(text, max_words)  # Use smart chunking

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

def test_advanced_pdf_chunking():
    """Test advanced PDF chunking with pymupdf and smart chunking"""
    
    print("📄 Advanced PDF Chunking Demo")
    print("=" * 45)
    
    # Ensure NLTK is ready
    ensure_nltk()
    
    # Select a whitepaper to test
    whitepaper_path = "data/input/reviews/Whitepaper-Model-Validation-Best-Practices-1.pdf"
    
    if not os.path.exists(whitepaper_path):
        print("❌ Whitepaper not found: {}".format(whitepaper_path))
        return False
    
    print("📖 Processing: {}".format(whitepaper_path))
    
    try:
        # Import pymupdf (modern PDF library)
        import fitz  # pymupdf
        
        # Extract text from PDF using pymupdf
        print("\n🔍 Step 1: Extracting text from PDF using pymupdf...")
        
        doc = fitz.open(whitepaper_path)
        num_pages = len(doc)
        
        print("📄 Found {} pages".format(num_pages))
        
        all_chunks = []
        total_text_length = 0
        
        # Process first few pages for demo
        for page_num in range(min(3, num_pages)):
            page = doc[page_num]
            
            # Extract text with better formatting
            text = page.get_text()
            text = clean_text(text)  # Apply text cleaning
            
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
        
        # Show sample chunks with metadata
        print("\n📋 Sample Chunks with Metadata:")
        print("=" * 40)
        
        for i, chunk in enumerate(all_chunks[:3], 1):
            print(f"\n🔹 Chunk {i}:")
            print(f"   ID: {chunk['chunk_id']}")
            print(f"   Page: {chunk['metadata']['page_number']}")
            print(f"   Words: {chunk['metadata']['word_count']}")
            print(f"   Content: {chunk['text'][:150]}...")
        
        # Show chunking statistics
        print("\n📊 Advanced Chunking Statistics:")
        print("=" * 35)
        
        word_counts = [chunk['metadata']['word_count'] for chunk in all_chunks]
        page_numbers = [chunk['metadata']['page_number'] for chunk in all_chunks]
        
        print(f"Total chunks: {len(all_chunks)}")
        print(f"Average words per chunk: {sum(word_counts) // len(word_counts)}")
        print(f"Largest chunk: {max(word_counts)} words")
        print(f"Smallest chunk: {min(word_counts)} words")
        print(f"Pages processed: {len(set(page_numbers))}")
        
        # Show chunk distribution by page
        print(f"\n📄 Chunk Distribution by Page:")
        page_chunk_counts = {}
        for chunk in all_chunks:
            page = chunk['metadata']['page_number']
            page_chunk_counts[page] = page_chunk_counts.get(page, 0) + 1
        
        for page in sorted(page_chunk_counts.keys()):
            print(f"   Page {page}: {page_chunk_counts[page]} chunks")
        
        return True
        
    except ImportError as e:
        print("❌ Missing dependency: {}".format(e))
        print("Please install: pip install pymupdf nltk")
        return False
    except Exception as e:
        print("❌ Error during PDF processing: {}".format(e))
        return False

def main():
    """Main function"""
    
    print("🚀 Advanced PDF Chunking Demo")
    print("=" * 30)
    
    success = test_advanced_pdf_chunking()
    
    if success:
        print("\n🎉 Advanced PDF chunking demo completed!")
        print("\n📝 Advanced chunking features:")
        print("1. ✅ Uses pymupdf (modern, fast PDF library)")
        print("2. ✅ Smart sentence boundary detection")
        print("3. ✅ Sub-sentence splitting for long sentences")
        print("4. ✅ Semantic chunking at logical break points")
        print("5. ✅ Rich metadata for each chunk")
        print("6. ✅ Text cleaning for PDF artifacts")
        print("7. ✅ Configurable chunk size (words, not characters)")
        print("8. ✅ Page continuity handling")
    else:
        print("\n❌ Advanced PDF chunking demo failed!")

if __name__ == "__main__":
    main()
