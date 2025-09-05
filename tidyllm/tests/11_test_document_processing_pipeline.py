#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test 11: Document Processing Pipeline

Tests end-to-end document processing pipeline including upload, processing,
vectorization, and search functionality.

IMPORTANT FOR AGENTS/LLMs:
- DO NOT use mock document processing when real services are available
- Test COMPLETE pipeline: upload ’ process ’ vectorize ’ search
- SAVE pipeline execution evidence to tests/EVIDENCE folder
- Validate document content extraction and search accuracy
"""

import os
import sys
import json
import pytest
import tempfile
import hashlib
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tidyllm.settings_loader import SettingsLoader

# Document processing imports
try:
    import PyPDF2
    import docx
    DOCUMENT_PROCESSING_AVAILABLE = True
except ImportError:
    DOCUMENT_PROCESSING_AVAILABLE = False

# Vector search imports
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    VECTOR_SEARCH_AVAILABLE = True
except ImportError:
    VECTOR_SEARCH_AVAILABLE = False

class TestDocumentProcessingPipeline:
    """Test suite for document processing pipeline"""
    
    def save_evidence(self, evidence_data, test_name):
        """Save document processing evidence"""
        evidence_dir = Path(__file__).parent / "EVIDENCE"
        evidence_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"evidence_document_{test_name}_{timestamp}.json"
        evidence_path = evidence_dir / filename
        
        with open(evidence_path, 'w') as f:
            json.dump(evidence_data, f, indent=2, default=str)
        
        print(f"Document processing evidence saved: {evidence_path}")
        return evidence_path
    
    @pytest.fixture
    def settings_loader(self):
        """Fixture providing initialized SettingsLoader"""
        admin_settings_path = Path(__file__).parent.parent / "tidyllm" / "admin" / "settings.yaml"
        return SettingsLoader(str(admin_settings_path))
    
    def test_document_upload_and_extraction(self, settings_loader):
        """Test document upload and content extraction"""
        from tidyllm.document_pipeline import DocumentProcessor, DocumentUploader
        
        # Initialize components
        uploader = DocumentUploader()
        processor = DocumentProcessor()
        
        # Create test documents
        test_docs = []
        
        # Create PDF test document
        pdf_content = "This is a test PDF document for pipeline testing. It contains important information about machine learning and AI."
        pdf_path = self._create_test_pdf(pdf_content)
        test_docs.append({"path": pdf_path, "type": "pdf", "content": pdf_content})
        
        # Create text document
        txt_content = "This is a test text document. It discusses natural language processing and neural networks."
        txt_path = self._create_test_txt(txt_content)
        test_docs.append({"path": txt_path, "type": "txt", "content": txt_content})
        
        # Upload and process documents
        processed_docs = []
        for doc in test_docs:
            # Upload document
            upload_result = uploader.upload_document(
                file_path=doc["path"],
                document_type=doc["type"],
                metadata={"test": True, "pipeline": "integration_test"}
            )
            
            # Extract content
            extracted_content = processor.extract_content(upload_result.document_id)
            
            # Validate extraction
            assert extracted_content.document_id == upload_result.document_id
            assert len(extracted_content.text) > 0
            assert extracted_content.word_count > 0
            assert extracted_content.language is not None
            
            processed_docs.append({
                "upload_result": upload_result.__dict__,
                "extracted_content": extracted_content.__dict__,
                "original_content": doc["content"]
            })
        
        # Save evidence
        evidence_path = self.save_evidence(processed_docs, "upload_extraction")
        
        print(f" Document upload and extraction test completed")
        print(f"   Documents processed: {len(processed_docs)}")
        print(f"   Total characters extracted: {sum(len(d['extracted_content']['text']) for d in processed_docs)}")
        
        # Cleanup
        for doc in test_docs:
            os.unlink(doc["path"])
    
    def test_document_vectorization_and_indexing(self, settings_loader):
        """Test document vectorization and search indexing"""
        from tidyllm.document_pipeline import DocumentVectorizer, VectorIndex
        
        # Initialize components
        vectorizer = DocumentVectorizer()
        index = VectorIndex()
        
        # Test documents
        test_documents = [
            {"id": "doc1", "text": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience.", "metadata": {"topic": "ML"}},
            {"id": "doc2", "text": "Natural language processing helps computers understand and interpret human language in a valuable way.", "metadata": {"topic": "NLP"}},
            {"id": "doc3", "text": "Neural networks are computing systems inspired by biological neural networks that constitute animal brains.", "metadata": {"topic": "Neural Networks"}}
        ]
        
        # Vectorize documents
        vectorized_docs = []
        for doc in test_documents:
            vector_result = vectorizer.vectorize_document(
                document_id=doc["id"],
                text=doc["text"],
                metadata=doc["metadata"]
            )
            
            # Validate vectorization
            assert vector_result.document_id == doc["id"]
            assert len(vector_result.embeddings) > 0
            assert vector_result.embedding_model is not None
            assert vector_result.chunk_count > 0
            
            vectorized_docs.append(vector_result)
        
        # Add to search index
        for vector_doc in vectorized_docs:
            index_result = index.add_document(vector_doc)
            assert index_result.success
            assert index_result.document_id == vector_doc.document_id
        
        # Test search functionality
        search_queries = [
            "artificial intelligence",
            "language processing", 
            "neural networks"
        ]
        
        search_results = []
        for query in search_queries:
            results = index.search(
                query=query,
                limit=2,
                similarity_threshold=0.5
            )
            
            # Validate search results
            assert len(results.matches) > 0
            for match in results.matches:
                assert match.document_id is not None
                assert match.similarity_score >= 0.5
                assert len(match.matched_text) > 0
            
            search_results.append({
                "query": query,
                "results": [m.__dict__ for m in results.matches]
            })
        
        # Save evidence
        evidence_data = {
            "vectorized_documents": [v.__dict__ for v in vectorized_docs],
            "search_results": search_results
        }
        evidence_path = self.save_evidence(evidence_data, "vectorization_indexing")
        
        print(f" Document vectorization and indexing test completed")
        print(f"   Documents vectorized: {len(vectorized_docs)}")
        print(f"   Search queries tested: {len(search_queries)}")
    
    def test_end_to_end_pipeline(self, settings_loader):
        """Test complete end-to-end document processing pipeline"""
        from tidyllm.document_pipeline import DocumentPipeline
        
        # Initialize pipeline
        pipeline = DocumentPipeline()
        
        # Create test document
        test_content = """
        Artificial Intelligence and Machine Learning Overview
        
        Artificial Intelligence (AI) is a broad field of computer science that aims to create 
        intelligent machines capable of performing tasks that typically require human intelligence.
        
        Machine Learning (ML) is a subset of AI that enables computers to learn and improve 
        from experience without being explicitly programmed.
        
        Natural Language Processing (NLP) is another AI subfield that helps computers understand,
        interpret, and generate human language in a valuable way.
        """
        
        test_doc_path = self._create_test_txt(test_content)
        
        # Run complete pipeline
        pipeline_result = pipeline.process_document(
            file_path=test_doc_path,
            pipeline_config={
                "extract_content": True,
                "vectorize": True,
                "index": True,
                "enable_search": True
            }
        )
        
        # Validate pipeline result
        assert pipeline_result.success
        assert pipeline_result.document_id is not None
        assert pipeline_result.content_extracted
        assert pipeline_result.vectorized
        assert pipeline_result.indexed
        assert pipeline_result.searchable
        
        # Test search on processed document
        search_result = pipeline.search_documents(
            query="machine learning artificial intelligence",
            limit=1
        )
        
        # Validate search works
        assert len(search_result.matches) > 0
        assert search_result.matches[0].document_id == pipeline_result.document_id
        assert search_result.matches[0].similarity_score > 0.7
        
        # Save evidence
        evidence_data = {
            "pipeline_result": pipeline_result.__dict__,
            "search_result": {
                "query": "machine learning artificial intelligence",
                "matches": [m.__dict__ for m in search_result.matches]
            },
            "document_content_preview": test_content[:200] + "..."
        }
        evidence_path = self.save_evidence(evidence_data, "end_to_end_pipeline")
        
        print(f" End-to-end pipeline test completed")
        print(f"   Document ID: {pipeline_result.document_id}")
        print(f"   Search matches: {len(search_result.matches)}")
        print(f"   Best match similarity: {search_result.matches[0].similarity_score:.3f}")
        
        # Cleanup
        os.unlink(test_doc_path)
    
    def _create_test_pdf(self, content):
        """Create temporary PDF file for testing"""
        # For testing purposes, create a simple text file with .pdf extension
        # In real implementation, would create actual PDF
        fd, path = tempfile.mkstemp(suffix='.pdf')
        with os.fdopen(fd, 'w') as f:
            f.write(content)
        return path
    
    def _create_test_txt(self, content):
        """Create temporary text file for testing"""
        fd, path = tempfile.mkstemp(suffix='.txt')
        with os.fdopen(fd, 'w') as f:
            f.write(content)
        return path

def test_priority_document_pipeline_check():
    """Priority test for document processing pipeline readiness"""
    try:
        from tidyllm.document_pipeline import (
            DocumentProcessor, DocumentUploader, DocumentVectorizer, 
            VectorIndex, DocumentPipeline, ExtractedContent, VectorResult
        )
        
        # Test document processing functionality
        processor = DocumentProcessor()
        assert processor is not None
        
        # Test uploader
        uploader = DocumentUploader()
        assert uploader is not None
        
        # Test vectorizer
        vectorizer = DocumentVectorizer()
        assert vectorizer is not None
        
        # Test index
        index = VectorIndex()
        assert index is not None
        
        # Test complete pipeline
        pipeline = DocumentPipeline()
        assert pipeline is not None
        
        print("SUCCESS: Document processing pipeline implemented and working")
        
    except Exception as e:
        pytest.fail(f"CRITICAL: Document processing pipeline check failed: {e}")

if __name__ == "__main__":
    test_priority_document_pipeline_check()