#!/usr/bin/env python3
"""
Simple Build Test for Model Validation Domain RAG
=================================================
"""

import sys
from pathlib import Path

# Add parent path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

def main():
    print("Testing Knowledge Systems Architecture")
    print("=" * 50)
    
    # Test imports
    try:
        from knowledge_systems import KnowledgeInterface, get_knowledge_interface
        print("SUCCESS: Imports working")
    except Exception as e:
        print(f"ERROR: Import failed: {e}")
        return
    
    # Initialize system
    try:
        ki = get_knowledge_interface()
        print("SUCCESS: Knowledge Interface initialized")
    except Exception as e:
        print(f"ERROR: Initialization failed: {e}")
        return
    
    # Test connections
    try:
        connections = ki.test_connections()
        s3_ok = connections['s3_connection']['success']
        vector_ok = connections['vector_connection']['success']
        
        print(f"S3 Connection: {'OK' if s3_ok else 'FAILED'}")
        print(f"Vector DB: {'OK' if vector_ok else 'FAILED'}")
        
        if not s3_ok:
            print(f"  S3 Error: {connections['s3_connection']['error']}")
        if not vector_ok:
            print(f"  Vector Error: {connections['vector_connection']['error']}")
            
    except Exception as e:
        print(f"ERROR: Connection test failed: {e}")
    
    # Look for knowledge base
    kb_paths = [
        parent_dir / "knowledge_base",
        parent_dir / "tidyllm" / "knowledge_base"
    ]
    
    kb_path = None
    for path in kb_paths:
        if path.exists():
            kb_path = path
            break
    
    if kb_path:
        print(f"Found knowledge base: {kb_path}")
        pdf_count = len(list(kb_path.glob("*.pdf")))
        txt_count = len(list(kb_path.glob("*.txt")))
        print(f"  PDF files: {pdf_count}")
        print(f"  Text files: {txt_count}")
        
        if pdf_count + txt_count > 0:
            print("Knowledge base ready for processing")
        else:
            print("No documents found in knowledge base")
    else:
        print("Knowledge base not found")
    
    print("\nArchitecture test completed")

if __name__ == "__main__":
    main()