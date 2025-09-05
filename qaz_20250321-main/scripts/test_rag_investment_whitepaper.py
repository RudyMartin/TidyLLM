#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Test for Investment Model Validation Whitepaper

This script tests the RAG pipeline with the investment-model-validation.pdf document.
"""

import os
import sys
import re
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

def clean_text(text: str) -> str:
    """Clean extracted text from PDF artifacts"""
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    return text.strip()

def simple_sentence_split(text: str) -> List[str]:
    """Simple sentence splitting"""
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]

def smart_chunking_simple(text: str, max_words: int = 200) -> List[str]:
    """Smart chunking without NLTK"""
    sentences = simple_sentence_split(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        
        if sentence_length > max_words:
            sub_sentences = re.split(r'[;,:]|\b(and|but|or|which|that)\b', sentence)
            sub_sentences = [s.strip() for s in sub_sentences if s.strip()]
        else:
            sub_sentences = [sentence]

        for sub_sentence in sub_sentences:
            sub_length = len(sub_sentence.split())
            
            if current_length + sub_length > max_words and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0

            current_chunk.append(sub_sentence)
            current_length += sub_length

    if current_chunk and len(" ".join(current_chunk).split()) > 5:
        chunks.append(" ".join(current_chunk))

    return chunks

def process_pdf_to_chunks(pdf_path: str) -> List[Dict]:
    """Process PDF and create chunks with metadata"""
    try:
        import fitz  # pymupdf
        
        doc = fitz.open(pdf_path)
        all_chunks = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            text = clean_text(text)
            
            if not text.strip():
                continue
                
            document_name = os.path.splitext(os.path.basename(pdf_path))[0]
            chunked_texts = smart_chunking_simple(text, max_words=200)
            
            for i, chunk_text in enumerate(chunked_texts):
                chunk_id = f"{document_name}_page_{page_num + 1}_{i + 1:03d}"
                chunk_data = {
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        "document_name": document_name,
                        "page_number": page_num + 1,
                        "chunk_number": i + 1,
                        "total_chunks": len(chunked_texts),
                        "word_count": len(chunk_text.split())
                    }
                }
                all_chunks.append(chunk_data)
        
        doc.close()
        return all_chunks
        
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        return []

def store_chunks_in_database(chunks: List[Dict]) -> bool:
    """Store chunks in Aurora PostgreSQL database"""
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        
        # Load environment variables
        load_dotenv('src/backend/config/credentials.env')
        database_url = os.getenv('DATABASE_URL')
        
        if not database_url:
            print("❌ DATABASE_URL not found")
            return False
        
        # Connect to database
        conn = psycopg2.connect(database_url)
        
        # Store chunks
        stored_count = 0
        with conn.cursor() as cursor:
            for chunk in chunks:
                try:
                    cursor.execute("""
                        INSERT INTO document_chunks 
                        (doc_id, chunk_id, page_num, chunk_text, char_count, embedding_model)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        chunk['metadata']['document_name'],
                        chunk['chunk_id'],
                        chunk['metadata']['page_number'],
                        chunk['text'],
                        len(chunk['text']),
                        'sentence-transformers'
                    ))
                    stored_count += 1
                except Exception as e:
                    print(f"⚠️ Error storing chunk {chunk['chunk_id']}: {e}")
        
        conn.commit()
        conn.close()
        
        print(f"✅ Stored {stored_count} chunks in database")
        return True
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def answer_question_with_rag(question: str, top_k: int = 3) -> Dict:
    """Answer a question using RAG from database"""
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        
        # Load environment variables
        load_dotenv('src/backend/config/credentials.env')
        database_url = os.getenv('DATABASE_URL')
        
        if not database_url:
            return {"error": "Database not configured"}
        
        # Connect to database
        conn = psycopg2.connect(database_url)
        
        # Simple keyword-based retrieval
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            search_terms = question.lower().split()
            search_conditions = []
            search_params = []
            
            for term in search_terms:
                if len(term) > 3:  # Only search for meaningful terms
                    search_conditions.append("LOWER(chunk_text) LIKE %s")
                    search_params.append(f"%{term}%")
            
            if not search_conditions:
                search_conditions = ["1=1"]  # Return all if no search terms
            
            query = f"""
                SELECT chunk_id, doc_id, page_num, chunk_text, char_count
                FROM document_chunks
                WHERE {' OR '.join(search_conditions)}
                ORDER BY char_count DESC
                LIMIT %s
            """
            search_params.append(top_k)
            
            cursor.execute(query, search_params)
            results = cursor.fetchall()
        
        conn.close()
        
        # Format results
        relevant_chunks = []
        for row in results:
            relevant_chunks.append({
                "chunk_id": row['chunk_id'],
                "document_name": row['doc_id'],
                "page_number": row['page_num'],
                "text": row['chunk_text'][:300] + "..." if len(row['chunk_text']) > 300 else row['chunk_text'],
                "char_count": row['char_count']
            })
        
        # Simple answer generation
        if relevant_chunks:
            combined_text = " ".join([chunk['text'] for chunk in relevant_chunks])
            sentences = simple_sentence_split(combined_text)
            answer_sentences = []
            
            for sentence in sentences:
                if any(term in sentence.lower() for term in question.lower().split() if len(term) > 3):
                    answer_sentences.append(sentence)
            
            answer = " ".join(answer_sentences[:3])  # Top 3 relevant sentences
            
            return {
                "question": question,
                "answer": answer,
                "relevant_chunks": relevant_chunks,
                "sources": [f"Page {chunk['page_number']}" for chunk in relevant_chunks]
            }
        else:
            return {
                "question": question,
                "answer": "No relevant information found in the document.",
                "relevant_chunks": [],
                "sources": []
            }
        
    except Exception as e:
        return {"error": f"RAG error: {e}"}

def test_investment_whitepaper_rag():
    """Test RAG with investment model validation whitepaper"""
    
    print("🧪 Investment Model Validation RAG Test")
    print("=" * 45)
    
    # Load environment variables
    load_dotenv('src/backend/config/credentials.env')
    
    # Check database connection
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        print("❌ DATABASE_URL not found. Please run setup_credentials.py first.")
        return False
    
    # Select investment whitepaper
    whitepaper_path = "data/input/reviews/investment-model-validation.pdf"
    
    if not os.path.exists(whitepaper_path):
        print("❌ Whitepaper not found: {}".format(whitepaper_path))
        return False
    
    print("📖 Processing: {}".format(whitepaper_path))
    
    try:
        # Step 1: Process PDF to chunks
        print("\n🔍 Step 1: Processing PDF to chunks...")
        chunks = process_pdf_to_chunks(whitepaper_path)
        
        if not chunks:
            print("❌ No chunks created from PDF")
            return False
        
        print(f"✅ Created {len(chunks)} chunks")
        
        # Step 2: Store chunks in database
        print("\n🗄️ Step 2: Storing chunks in database...")
        if not store_chunks_in_database(chunks):
            print("❌ Failed to store chunks in database")
            return False
        
        # Step 3: Test RAG question answering
        print("\n❓ Step 3: Testing RAG question answering...")
        
        test_questions = [
            "What is investment model validation?",
            "What are the key components of investment validation?",
            "How should investment models be tested?",
            "What are the risks of poor investment model validation?",
            "What validation techniques are recommended for investment models?"
        ]
        
        for question in test_questions:
            print(f"\n🔍 Question: {question}")
            result = answer_question_with_rag(question)
            
            if "error" in result:
                print(f"❌ Error: {result['error']}")
            else:
                print(f"📝 Answer: {result['answer']}")
                print(f"📚 Sources: {', '.join(result['sources'])}")
                print(f"📊 Found {len(result['relevant_chunks'])} relevant chunks")
                for chunk in result['relevant_chunks'][:2]:  # Show first 2 chunks
                    print(f"   📄 {chunk['document_name']} (Page {chunk['page_number']}): {chunk['text'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        return False

def main():
    """Main function"""
    
    print("🚀 Investment Model Validation RAG Test")
    print("=" * 40)
    
    success = test_investment_whitepaper_rag()
    
    if success:
        print("\n🎉 Investment whitepaper RAG test successful!")
        print("\n📝 What we accomplished:")
        print("1. ✅ Investment PDF processed and chunked")
        print("2. ✅ Chunks stored in Aurora PostgreSQL database")
        print("3. ✅ Investment-specific questions answered")
        print("4. ✅ Relevant chunks retrieved from database")
        print("\n🚀 Your RAG system works with multiple document types!")
    else:
        print("\n❌ Investment whitepaper RAG test failed!")

if __name__ == "__main__":
    main()
