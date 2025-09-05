#!/usr/bin/env python3
"""
SOP Domain RAG Flow Creator
===========================

Creates a domain RAG flow specifically for SOP conflict resolution using the existing domain workflow system.
Processes all organized documentation in docs/2025-* folders.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Set AWS credentials
os.environ['AWS_ACCESS_KEY_ID'] = 'REMOVED_AWS_KEY'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'REMOVED_AWS_SECRET'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

def create_sop_domain_flow():
    """Create SOP domain RAG flow using existing system"""
    
    print("=" * 60)
    print("SOP DOMAIN RAG FLOW CREATION")
    print("Using existing TidyLLM domain workflow system")
    print("=" * 60)
    
    # Import existing domain workflow creator
    try:
        sys.path.insert(0, str(Path(__file__).parent / 'tidyllm'))
        from knowledge_systems.create_domain_workflow import DomainWorkflowCreator
        
        print("[OK] Imported existing DomainWorkflowCreator")
        
    except ImportError as e:
        print(f"[ERROR] Could not import domain workflow system: {e}")
        print("Falling back to direct implementation...")
        return create_sop_flow_direct()
    
    # Create SOP domain workflow
    try:
        creator = DomainWorkflowCreator()
        
        # Create SOP conflict resolution domain
        domain_name = "sop_conflict_resolution"
        input_folder = "docs"  # Our organized documentation
        
        print(f"[CREATE] Domain: {domain_name}")
        print(f"[CREATE] Input folder: {input_folder}")
        print(f"[CREATE] Processing {count_docs()} documentation files")
        
        # Use existing domain workflow system
        result = creator.create_domain_workflow(
            domain_name=domain_name,
            input_folder=input_folder,
            description="SOP conflict resolution for TidyLLM architectural documentation",
            file_patterns=["*.md", "*.txt", "*.rst"],
            test_queries=[
                "What is the official session management pattern?",
                "Which embedding system should be used?",
                "What are the conflicting architectural decisions?",
                "How should AWS S3 be accessed?",
                "What is the approved workflow system?"
            ]
        )
        
        print("[SUCCESS] SOP domain RAG flow created")
        print(f"S3 location: {result.get('s3_location', 'Unknown')}")
        print(f"Collection: {result.get('collection_name', 'Unknown')}")
        
        return result
        
    except Exception as e:
        print(f"[ERROR] Domain workflow creation failed: {e}")
        import traceback
        traceback.print_exc()
        return create_sop_flow_direct()

def count_docs():
    """Count documentation files in date folders"""
    docs_path = Path("docs")
    count = 0
    
    for date_folder in docs_path.glob("2025-*"):
        if date_folder.is_dir():
            count += len(list(date_folder.glob("*.md")))
            count += len(list(date_folder.glob("*.txt")))
            count += len(list(date_folder.glob("*.rst")))
    
    return count

def create_sop_flow_direct():
    """Direct implementation if domain workflow system isn't available"""
    
    print("\n[DIRECT] Creating SOP flow directly...")
    
    # Test backend connectivity
    print("[TEST] Testing backend connectivity...")
    
    try:
        import boto3
        s3 = boto3.client('s3')
        buckets = s3.list_buckets()
        print(f"[OK] S3 connection: {len(buckets['Buckets'])} buckets available")
        
    except Exception as e:
        print(f"[ERROR] S3 connection failed: {e}")
        return {"error": "S3 backend not available"}
    
    # Test embeddings backend
    try:
        # Try tidyllm-sentence first
        from tidyllm_sentence import TfidfVectorizer
        print("[OK] tidyllm-sentence embeddings available")
        embedding_backend = "tidyllm-sentence"
        
    except ImportError:
        try:
            # Fall back to basic TF-IDF
            from sklearn.feature_extraction.text import TfidfVectorizer
            print("[OK] sklearn TF-IDF available (fallback)")
            embedding_backend = "sklearn"
            
        except ImportError:
            print("[ERROR] No embedding backend available")
            return {"error": "No embedding backend"}
    
    # Create simple SOP processing flow
    docs_path = Path("docs")
    doc_count = count_docs()
    
    flow_config = {
        "domain": "sop_conflict_resolution",
        "input_path": str(docs_path),
        "document_count": doc_count,
        "s3_bucket": "nsc-mvp1",
        "s3_prefix": "sop_domain_rag/",
        "embedding_backend": embedding_backend,
        "conflict_queries": [
            "What is the official session management pattern for TidyLLM?",
            "Which embedding system should be used: tidyllm-sentence or tidyllm-vectorqa?",
            "Should we use UnifiedSessionManager or Gateway pattern?",
            "What are the conflicting architectural decisions?",
            "Which workflow system is approved: RAG2DAG, HeirOS, or YAML?",
            "How should AWS S3 be accessed in TidyLLM?",
            "What patterns are deprecated and should not be used?"
        ],
        "created_at": datetime.now().isoformat(),
        "status": "ready_for_processing"
    }
    
    print(f"[SUCCESS] Direct SOP flow configured")
    print(f"Documents: {doc_count} files in date folders")
    print(f"Backend: S3 + {embedding_backend}")
    print(f"Queries: {len(flow_config['conflict_queries'])} conflict detection queries")
    
    return flow_config

def test_conflict_queries(flow_config):
    """Test the conflict detection queries on the documentation"""
    
    print("\n" + "=" * 60)
    print("TESTING CONFLICT QUERIES")
    print("=" * 60)
    
    # Simple document search for conflict detection
    docs_path = Path("docs")
    
    for query in flow_config["conflict_queries"]:
        print(f"\n[QUERY] {query}")
        
        # Search for relevant documents
        relevant_docs = []
        
        for date_folder in docs_path.glob("2025-*"):
            if not date_folder.is_dir():
                continue
                
            for doc in date_folder.glob("*.md"):
                try:
                    with open(doc, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read().lower()
                        
                    # Simple keyword matching
                    keywords = extract_keywords(query)
                    if any(keyword in content for keyword in keywords):
                        relevant_docs.append({
                            'file': doc.name,
                            'date': date_folder.name,
                            'path': str(doc)
                        })
                        
                except Exception as e:
                    continue
        
        print(f"[RESULT] Found {len(relevant_docs)} relevant documents")
        for doc in relevant_docs[:3]:  # Show first 3
            print(f"  - {doc['file']} ({doc['date']})")
        
        if len(relevant_docs) > 3:
            print(f"  ... and {len(relevant_docs) - 3} more")
            
        # This is where the domain RAG would identify conflicts
        if len(relevant_docs) > 1:
            print(f"  [CONFLICT POTENTIAL] Multiple documents address this topic")

def extract_keywords(query):
    """Extract keywords from query for simple matching"""
    import re
    
    # Remove common words
    stop_words = {'what', 'is', 'the', 'should', 'be', 'used', 'how', 'for', 'which', 'or', 'and'}
    
    # Extract words
    words = re.findall(r'\b\w+\b', query.lower())
    keywords = [w for w in words if w not in stop_words and len(w) > 3]
    
    return keywords

def main():
    """Main execution"""
    
    # Create SOP domain flow
    flow_config = create_sop_domain_flow()
    
    if "error" in flow_config:
        print(f"[FAILED] SOP domain flow creation failed: {flow_config['error']}")
        return False
    
    # Test conflict queries
    test_conflict_queries(flow_config)
    
    print("\n" + "=" * 60)
    print("SOP DOMAIN RAG FLOW SUMMARY")
    print("=" * 60)
    print(f"Domain: {flow_config.get('domain', 'sop_conflict_resolution')}")
    print(f"Documents: {flow_config.get('document_count', 'Unknown')} files")
    print(f"Backend: {flow_config.get('embedding_backend', 'Unknown')}")
    print(f"Queries: {len(flow_config.get('conflict_queries', []))} conflict detection queries")
    print(f"Status: {flow_config.get('status', 'Unknown')}")
    
    print("\n[NEXT STEPS]")
    print("1. Run domain RAG processing on docs/2025-* folders")
    print("2. Execute conflict detection queries")
    print("3. Generate authoritative SOPs from conflict analysis")
    print("4. Validate resolutions against CRITICAL_DESIGN_DECISIONS.md")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)