#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test RAG with Whitepaper

This script tests the RAG QA Orchestrator with a whitepaper from the reviews folder.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

def test_rag_with_whitepaper():
    """Test RAG orchestrator with a whitepaper"""
    
    print("🧪 Testing RAG Orchestrator with Whitepaper")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv('src/backend/config/credentials.env')
    
    # Check if DATABASE_URL is available
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        print("❌ DATABASE_URL not found. Please run setup_credentials.py first.")
        return False
    
    # Select a whitepaper to test
    whitepaper_path = "data/input/reviews/Whitepaper-Model-Validation-Best-Practices-1.pdf"
    
    if not os.path.exists(whitepaper_path):
        print("❌ Whitepaper not found: {}".format(whitepaper_path))
        return False
    
    print("📄 Using whitepaper: {}".format(whitepaper_path))
    
    try:
        # Import the RAG orchestrator
        sys.path.append('src')
        from backend.mcp.orchestrators.rag_qa_orchestrator import RAGQAOrchestrator
        
        # Initialize the orchestrator
        print("🔧 Initializing RAG QA Orchestrator...")
        orchestrator = RAGQAOrchestrator()
        
        # Test document processing
        print("\n📖 Processing whitepaper...")
        result = orchestrator.process_document(
            document_path=whitepaper_path,
            user_query="What are the key best practices for model validation mentioned in this document?",
            research_depth="comprehensive"
        )
        
        print("\n✅ RAG processing completed!")
        print("\n📋 Results:")
        print("=" * 30)
        
        if hasattr(result, 'summary'):
            print("📝 Summary:")
            print(result.summary)
        
        if hasattr(result, 'key_findings'):
            print("\n🔍 Key Findings:")
            for i, finding in enumerate(result.key_findings, 1):
                print(f"{i}. {finding}")
        
        if hasattr(result, 'recommendations'):
            print("\n💡 Recommendations:")
            for i, rec in enumerate(result.recommendations, 1):
                print(f"{i}. {rec}")
        
        if hasattr(result, 'database_status'):
            print(f"\n🗄️ Database Status: {result.database_status}")
        
        return True
        
    except ImportError as e:
        print("❌ Import error: {}".format(e))
        print("Please ensure all dependencies are installed:")
        print("pip install dspy-ai sentence-transformers psycopg2-binary python-dotenv")
        return False
    except Exception as e:
        print("❌ Error during RAG processing: {}".format(e))
        return False

def main():
    """Main function"""
    
    print("🚀 RAG Whitepaper Test")
    print("=" * 30)
    
    success = test_rag_with_whitepaper()
    
    if success:
        print("\n🎉 RAG test completed successfully!")
        print("\n📝 Next steps:")
        print("1. Check the database for stored document chunks and embeddings")
        print("2. Try different research queries")
        print("3. Test with other whitepapers")
    else:
        print("\n❌ RAG test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
