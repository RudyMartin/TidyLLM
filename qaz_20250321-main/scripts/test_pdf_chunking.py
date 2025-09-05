#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test PDF Chunking Process

This script demonstrates how PDFs are chunked for RAG processing.
"""

import os
import sys
from pathlib import Path

def test_pdf_chunking():
    """Test PDF chunking process"""
    
    print("📄 PDF Chunking Process Demo")
    print("=" * 40)
    
    # Select a whitepaper to test
    whitepaper_path = "data/input/reviews/Whitepaper-Model-Validation-Best-Practices-1.pdf"
    
    if not os.path.exists(whitepaper_path):
        print("❌ Whitepaper not found: {}".format(whitepaper_path))
        return False
    
    print("📖 Processing: {}".format(whitepaper_path))
    
    try:
        # Import PDF processing
        import PyPDF2
        
        # Extract text from PDF
        print("\n🔍 Step 1: Extracting text from PDF...")
        text_content = ""
        
        with open(whitepaper_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            
            print("📄 Found {} pages".format(num_pages))
            
            # Extract text from first few pages for demo
            for i in range(min(3, num_pages)):
                page = pdf_reader.pages[i]
                page_text = page.extract_text()
                text_content += f"\n--- Page {i+1} ---\n{page_text}"
        
        print("✅ Text extraction completed")
        print("📊 Total text length: {} characters".format(len(text_content)))
        
        # Demonstrate chunking
        print("\n✂️ Step 2: Creating chunks...")
        chunks = create_simple_chunks(text_content, chunk_size=500, overlap=50)
        
        print("✅ Created {} chunks".format(len(chunks)))
        
        # Show sample chunks
        print("\n📋 Sample Chunks:")
        print("=" * 30)
        
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"\n🔹 Chunk {i} (Length: {len(chunk)} chars):")
            print("-" * 20)
            print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
        
        # Show chunking statistics
        print("\n📊 Chunking Statistics:")
        print("=" * 25)
        chunk_lengths = [len(chunk) for chunk in chunks]
        print(f"Total chunks: {len(chunks)}")
        print(f"Average chunk size: {sum(chunk_lengths) // len(chunk_lengths)} characters")
        print(f"Largest chunk: {max(chunk_lengths)} characters")
        print(f"Smallest chunk: {min(chunk_lengths)} characters")
        
        return True
        
    except ImportError as e:
        print("❌ Missing dependency: {}".format(e))
        print("Please install: pip install PyPDF2")
        return False
    except Exception as e:
        print("❌ Error during PDF processing: {}".format(e))
        return False

def create_simple_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """Create simple overlapping text chunks"""
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings
            for i in range(end, max(start, end - 100), -1):
                if text[i] in '.!?':
                    end = i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks

def main():
    """Main function"""
    
    print("🚀 PDF Chunking Demo")
    print("=" * 25)
    
    success = test_pdf_chunking()
    
    if success:
        print("\n🎉 PDF chunking demo completed!")
        print("\n📝 How chunking works:")
        print("1. PDF text is extracted page by page")
        print("2. Text is split into overlapping chunks (~500 chars)")
        print("3. Chunks try to break at sentence boundaries")
        print("4. Each chunk gets metadata (source, position)")
        print("5. Chunks are stored in database for vector search")
    else:
        print("\n❌ PDF chunking demo failed!")

if __name__ == "__main__":
    main()
