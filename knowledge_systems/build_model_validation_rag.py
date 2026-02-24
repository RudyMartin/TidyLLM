#!/usr/bin/env python3
"""
Build Model Validation Domain RAG
=================================

Script to create the Model Validation domain RAG system from the knowledge base PDFs.
This demonstrates the consolidated knowledge systems architecture.
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add parent path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from knowledge_systems import KnowledgeInterface, get_knowledge_interface

def main():
    """Build Model Validation domain RAG"""
    print("Building Model Validation Domain RAG System")
    print("=" * 60)
    
    # Initialize knowledge interface
    print("Initializing Knowledge Interface...")
    try:
        ki = get_knowledge_interface()
        print("SUCCESS: Knowledge Interface initialized")
    except Exception as e:
        print(f"ERROR: Failed to initialize Knowledge Interface: {e}")
        return
    
    # Test system connections
    print("\nTesting system connections...")
    connections = ki.test_connections()
    
    print(f"S3 Connection: {'SUCCESS' if connections['s3_connection']['success'] else 'FAILED'}")
    if not connections['s3_connection']['success']:
        print(f"  Error: {connections['s3_connection']['error']}")
    
    print(f"Vector DB Connection: {'SUCCESS' if connections['vector_connection']['success'] else 'FAILED'}")
    if not connections['vector_connection']['success']:
        print(f"  Error: {connections['vector_connection']['error']}")
    
    # Locate knowledge base
    print("\nLocating knowledge base...")
    knowledge_base_paths = [
        Path(parent_dir) / "knowledge_base",
        Path(parent_dir) / "tidyllm" / "knowledge_base",
        Path(__file__).parent.parent.parent / "knowledge_base"
    ]
    
    knowledge_base_path = None
    for path in knowledge_base_paths:
        if path.exists():
            knowledge_base_path = path
            break
    
    if not knowledge_base_path:
        print("ERROR: Knowledge base directory not found!")
        print("   Searched locations:")
        for path in knowledge_base_paths:
            print(f"   - {path}")
        return
    
    print(f"SUCCESS: Found knowledge base at: {knowledge_base_path}")
    
    # Count documents
    pdf_files = list(knowledge_base_path.glob("*.pdf"))
    txt_files = list(knowledge_base_path.glob("*.txt"))
    md_files = list(knowledge_base_path.glob("*.md"))
    
    total_docs = len(pdf_files) + len(txt_files) + len(md_files)
    print(f"   PDF files: {len(pdf_files)}")
    print(f"   Text files: {len(txt_files)}")  
    print(f"   Markdown files: {len(md_files)}")
    print(f"   Total documents: {total_docs}")
    
    if total_docs == 0:
        print("ERROR: No documents found in knowledge base!")
        return
    
    # Create Model Validation domain RAG
    print(f"\nCreating Model Validation domain RAG...")
    print(f"   Processing {total_docs} documents...")
    
    try:
        result = ki.create_domain_rag(
            domain_name="model_validation",
            knowledge_base_path=knowledge_base_path,
            description="Model validation and risk management knowledge base from regulatory documents and best practices"
        )
        
        if result["success"]:
            print("SUCCESS: Model Validation domain RAG created successfully!")
            stats = result["stats"]
            print(f"   Documents processed: {stats.get('documents_processed', 0)}")
            print(f"   Total chunks: {stats.get('total_chunks', 0)}")
            print(f"   Processing errors: {stats.get('processing_errors', 0)}")
            print(f"   Last updated: {stats.get('last_updated', 'Unknown')}")
        else:
            print(f"ERROR: Failed to create domain RAG: {result['error']}")
            return
            
    except Exception as e:
        print(f"ERROR: Exception during domain RAG creation: {e}")
        return
    
    # Test the domain RAG with sample queries
    print("\nTesting domain RAG with sample queries...")
    
    test_queries = [
        "What are the key requirements for model validation under Basel III?",
        "How should model performance monitoring be conducted?", 
        "What documentation is required for model risk management?",
        "Explain the model development lifecycle stages"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n   Query {i}: {query}")
        try:
            response = ki.query(query, domain="model_validation")
            print(f"   ✅ Answer: {response.answer[:100]}...")
            print(f"   📊 Confidence: {response.confidence:.2f}")
            print(f"   📚 Sources: {len(response.sources)}")
            print(f"   ⏱️  Processing time: {response.processing_time:.2f}s")
        except Exception as e:
            print(f"   ❌ Query failed: {e}")
    
    # Generate Flow Agreement
    print("\n📋 Generating Flow Agreement...")
    try:
        flow_agreements = ki.get_flow_agreements()
        if "model_validation" in flow_agreements:
            flow_agreement = flow_agreements["model_validation"]
            print("✅ Flow Agreement generated")
            print(f"   🎯 Domain: {flow_agreement.get('domain')}")
            print(f"   📝 Description: {flow_agreement.get('description')}")
            print(f"   🔧 Operations: {len(flow_agreement.get('operations', []))}")
            print(f"   💬 Chat command: {flow_agreement.get('integration', {}).get('bracket_command')}")
        else:
            print("❌ Flow Agreement generation failed")
    except Exception as e:
        print(f"❌ Flow Agreement error: {e}")
    
    # MVR Analysis integration
    print("\n🔗 Setting up MVR Analysis integration...")
    try:
        mvr_integration = ki.setup_mvr_integration()
        if mvr_integration["success"]:
            print("✅ MVR Analysis integration ready")
            flow = mvr_integration["flow_agreement"]
            mvr_stages = flow.get("mvr_integration", {}).get("stages", [])
            print(f"   🔄 MVR Stages: {', '.join(mvr_stages)}")
        else:
            print(f"❌ MVR integration failed: {mvr_integration['error']}")
    except Exception as e:
        print(f"❌ MVR integration error: {e}")
    
    # Final system status
    print("\n📊 Final System Status")
    print("-" * 30)
    try:
        status = ki.get_system_status()
        print(f"Total domains: {status['summary']['total_domains']}")
        print(f"S3 available: {'✅' if status['summary']['s3_available'] else '❌'}")
        print(f"Vector DB available: {'✅' if status['summary']['vector_available'] else '❌'}")
        
        for domain, stats in status['domain_rags'].items():
            print(f"\nDomain '{domain}':")
            print(f"  📄 Documents: {stats.get('documents_processed', 0)}")
            print(f"  📊 Chunks: {stats.get('total_chunks', 0)}")
            print(f"  ❌ Errors: {stats.get('processing_errors', 0)}")
            
    except Exception as e:
        print(f"❌ Status error: {e}")
    
    print(f"\n🎉 Model Validation Domain RAG build completed!")
    print(f"   Use [model_validation_rag] in chat interface")
    print(f"   Integration ready for MVR Analysis workflow")

if __name__ == "__main__":
    main()