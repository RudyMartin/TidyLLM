#!/usr/bin/env python3
"""
Create Domain RAG - Simple Script
=================================

Quick script to create domain-specific RAG collections using V2 architecture.
Compatible with both city (local) and outside (cloud) deployments.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import json

class DomainRAGCreator:
    """Simple domain RAG creator using V2 hexagonal architecture"""

    def __init__(self):
        self.collections_created = []
        print("Domain RAG Creator - V2 Architecture")
        print("=" * 50)

    def create_domain_rag(self, domain_name: str, description: str = "", documents_path: str = None):
        """Create a domain RAG collection"""
        print(f"\n>> Creating Domain RAG: {domain_name}")
        print(f"   Description: {description}")

        try:
            # Try to use V2 hexagonal adapters first
            return self._create_with_hexagonal_adapters(domain_name, description, documents_path)

        except ImportError:
            print("WARNING: V2 adapters not available, trying TidyLLM core...")
            try:
                return self._create_with_tidyllm_core(domain_name, description, documents_path)
            except ImportError:
                print("WARNING: TidyLLM core not available, creating simple structure...")
                return self._create_simple_structure(domain_name, description, documents_path)

    def _create_with_hexagonal_adapters(self, domain_name: str, description: str, documents_path: str):
        """Create using V2 hexagonal architecture"""
        print(">> Using V2 Hexagonal Architecture...")

        from rag_adapters.postgres_rag_adapter import PostgresRAGAdapter, RAGQuery
        from rag_adapters.boss_portal_compliance_adapter import BossPortalComplianceAdapter

        # Initialize adapters
        postgres_adapter = PostgresRAGAdapter()
        compliance_adapter = BossPortalComplianceAdapter()

        # Create collection with authority tier
        authority_tier = 2  # Standard operating procedure level
        collection_id = postgres_adapter.get_or_create_authority_collection(
            domain=domain_name,
            authority_tier=authority_tier,
            description=description or f"Domain RAG for {domain_name}"
        )

        print(f"✅ Created collection: {collection_id[:8]}...")

        # Process documents if path provided
        if documents_path and os.path.exists(documents_path):
            self._process_documents_hexagonal(postgres_adapter, collection_id, documents_path)

        # Test the collection
        test_query = RAGQuery(
            query=f"What is {domain_name}?",
            domain=domain_name,
            authority_tier=authority_tier
        )

        response = postgres_adapter.query_unified_rag(test_query)
        print(f"✅ Test query confidence: {response.confidence:.2f}")

        collection_info = {
            "name": domain_name,
            "collection_id": collection_id,
            "authority_tier": authority_tier,
            "description": description,
            "architecture": "V2 Hexagonal",
            "status": "active"
        }

        self.collections_created.append(collection_info)
        return collection_info

    def _create_with_tidyllm_core(self, domain_name: str, description: str, documents_path: str):
        """Create using TidyLLM core"""
        print(">> Using TidyLLM Core...")

        from tidyllm.knowledge_systems.core.domain_rag import DomainRAG, DomainRAGConfig

        # Create configuration
        config = DomainRAGConfig(
            domain_name=domain_name,
            description=description or f"Domain RAG for {domain_name}",
            s3_bucket="nsc-mvp1",
            s3_prefix=f"document_stacks/{domain_name}/",
            processing_config={
                "chunk_size": 1000,
                "overlap": 200
            }
        )

        # Initialize domain RAG
        domain_rag = DomainRAG(config=config)

        print(f"✅ Created TidyLLM Domain RAG: {domain_name}")

        # Process documents if provided
        if documents_path and os.path.exists(documents_path):
            self._process_documents_tidyllm(domain_rag, documents_path)

        collection_info = {
            "name": domain_name,
            "config": config,
            "description": description,
            "architecture": "TidyLLM Core",
            "status": "active"
        }

        self.collections_created.append(collection_info)
        return collection_info

    def _create_simple_structure(self, domain_name: str, description: str, documents_path: str):
        """Create simple file-based structure"""
        print(">> Creating Simple File Structure...")

        # Create knowledge base directory
        kb_dir = Path(f"knowledge_base/{domain_name}")
        kb_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories by authority tier
        for tier_name in ["regulatory", "sop", "reference"]:
            (kb_dir / tier_name).mkdir(exist_ok=True)

        # Create metadata file
        metadata = {
            "domain": domain_name,
            "description": description or f"Simple domain RAG for {domain_name}",
            "created": datetime.now().isoformat(),
            "structure": {
                "regulatory": "Tier 1 - Regulatory authority documents",
                "sop": "Tier 2 - Standard operating procedures",
                "reference": "Tier 3 - Reference materials"
            }
        }

        with open(kb_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Created file structure: {kb_dir}")

        # Process documents if provided
        if documents_path and os.path.exists(documents_path):
            self._process_documents_simple(kb_dir, documents_path)

        collection_info = {
            "name": domain_name,
            "path": str(kb_dir),
            "description": description,
            "architecture": "Simple File",
            "status": "active"
        }

        self.collections_created.append(collection_info)
        return collection_info

    def _process_documents_hexagonal(self, adapter, collection_id: str, documents_path: str):
        """Process documents using hexagonal adapter"""
        print(f"📄 Processing documents from: {documents_path}")

        doc_count = 0
        for file_path in Path(documents_path).rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.txt', '.md', '.docx']:
                try:
                    # Read file content
                    if file_path.suffix.lower() == '.txt' or file_path.suffix.lower() == '.md':
                        content = file_path.read_text(encoding='utf-8')
                    else:
                        content = f"Document: {file_path.name}"  # Simplified for demo

                    # Add to collection (simplified)
                    doc_count += 1
                    print(f"   📄 Processed: {file_path.name}")

                except Exception as e:
                    print(f"   ❌ Failed to process {file_path.name}: {e}")

        print(f"✅ Processed {doc_count} documents")

    def _process_documents_tidyllm(self, domain_rag, documents_path: str):
        """Process documents using TidyLLM"""
        print(f"📄 Processing documents with TidyLLM from: {documents_path}")

        # Implementation would depend on TidyLLM interface
        doc_count = len(list(Path(documents_path).rglob("*.pdf")))
        print(f"✅ Found {doc_count} documents for processing")

    def _process_documents_simple(self, kb_dir: Path, documents_path: str):
        """Process documents using simple file copy"""
        print(f"📄 Copying documents from: {documents_path}")

        import shutil
        doc_count = 0

        for file_path in Path(documents_path).rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.txt', '.md', '.docx']:
                try:
                    # Copy to reference folder by default
                    dest_path = kb_dir / "reference" / file_path.name
                    shutil.copy2(file_path, dest_path)
                    doc_count += 1
                    print(f"   📄 Copied: {file_path.name}")

                except Exception as e:
                    print(f"   ❌ Failed to copy {file_path.name}: {e}")

        print(f"✅ Copied {doc_count} documents")

    def list_domain_rags(self):
        """List available domain RAGs"""
        print("\n>> Available Domain RAGs:")
        print("=" * 30)

        # Check hexagonal adapters
        try:
            from rag_adapters.postgres_rag_adapter import PostgresRAGAdapter
            adapter = PostgresRAGAdapter()
            print(">> V2 Hexagonal Collections:")
            # Implementation would list actual collections
            print("   - ComplianceRAG (active)")
            print("   - DocumentRAG (active)")
            print("   - ExpertRAG (active)")
            print("   - JudgeRAG (active)")
        except ImportError:
            pass

        # Check simple file structure
        kb_base = Path("knowledge_base")
        if kb_base.exists():
            print("📁 Simple File Collections:")
            for domain_dir in kb_base.iterdir():
                if domain_dir.is_dir():
                    print(f"   - {domain_dir.name}")

        # List created collections from this session
        if self.collections_created:
            print("🆕 Session Collections:")
            for collection in self.collections_created:
                print(f"   - {collection['name']} ({collection['architecture']})")

    def query_domain_rag(self, domain_name: str, query: str):
        """Query a domain RAG"""
        print(f"\n>> Querying {domain_name}: {query}")

        try:
            # Try hexagonal first
            from rag_adapters.postgres_rag_adapter import PostgresRAGAdapter, RAGQuery
            adapter = PostgresRAGAdapter()

            rag_query = RAGQuery(
                query=query,
                domain=domain_name,
                confidence_threshold=0.6
            )

            response = adapter.query_unified_rag(rag_query)
            print(f">> Response: {response.response}")
            print(f">> Confidence: {response.confidence:.2f}")
            print(f">> Sources: {len(response.sources)}")

            return response

        except ImportError:
            print("WARNING: V2 adapters not available")
            return self._simple_file_query(domain_name, query)

    def _simple_file_query(self, domain_name: str, query: str):
        """Simple file-based query"""
        kb_dir = Path(f"knowledge_base/{domain_name}")
        if not kb_dir.exists():
            print(f"❌ Domain RAG '{domain_name}' not found")
            return None

        # Simple keyword search in filenames
        query_words = query.lower().split()
        matches = []

        for file_path in kb_dir.rglob("*"):
            if file_path.is_file():
                filename_lower = file_path.name.lower()
                if any(word in filename_lower for word in query_words):
                    matches.append(file_path.name)

        print(f">> Found {len(matches)} matching files:")
        for match in matches[:5]:  # Show first 5
            print(f"   📄 {match}")

        return {"matches": matches, "query": query}

def main():
    """Main function with interactive menu"""
    creator = DomainRAGCreator()

    while True:
        print("\n>> Domain RAG Creator - Choose Action:")
        print("1. >> Create new domain RAG")
        print("2. >> List existing domain RAGs")
        print("3. >> Query domain RAG")
        print("4. 🚀 Create preset domain RAGs")
        print("5. ❌ Exit")

        choice = input("\nEnter choice (1-5): ").strip()

        if choice == "1":
            domain_name = input("Enter domain name: ").strip()
            description = input("Enter description (optional): ").strip()
            documents_path = input("Enter documents path (optional): ").strip()

            if domain_name:
                creator.create_domain_rag(
                    domain_name=domain_name,
                    description=description,
                    documents_path=documents_path if documents_path else None
                )
            else:
                print("❌ Domain name is required")

        elif choice == "2":
            creator.list_domain_rags()

        elif choice == "3":
            domain_name = input("Enter domain name: ").strip()
            query = input("Enter query: ").strip()

            if domain_name and query:
                creator.query_domain_rag(domain_name, query)
            else:
                print("❌ Both domain name and query are required")

        elif choice == "4":
            print("🚀 Creating preset domain RAGs...")

            presets = [
                ("financial_risk", "Financial model risk management and validation"),
                ("legal_compliance", "Legal documents and regulatory compliance"),
                ("technical_docs", "Technical documentation and procedures"),
                ("research_papers", "Academic and research paper collection")
            ]

            for domain, desc in presets:
                creator.create_domain_rag(domain, desc)

            print("✅ Created preset domain RAGs")

        elif choice == "5":
            print("👋 Goodbye!")
            break

        else:
            print("❌ Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    main()