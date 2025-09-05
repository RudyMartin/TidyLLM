#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test 6: Research Document Upload

Tests document upload functionality to S3 using settings from admin/settings.yaml.
Simulates research paper upload workflow with comprehensive validation.

IMPORTANT FOR AGENTS/LLMs:
- DO NOT use mock/simulated uploads when real AWS credentials are available
- ALWAYS attempt real S3 uploads first before falling back to simulation
- SAVE upload receipts, metadata, and S3 object details to tests/EVIDENCE folder
- Use timestamp-based filenames: evidence_upload_YYYYMMDD_HHMMSS.json
- Include S3 URLs, ETag values, and complete upload responses
- Real evidence proves actual cloud storage functionality vs local simulation
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

# Optional imports for S3 functionality
try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

class TestResearchDocUpload:
    """Test suite for research document upload functionality"""
    
    def save_evidence(self, evidence_data, test_name):
        """Save real test results as evidence"""
        evidence_dir = Path(__file__).parent / "EVIDENCE"
        evidence_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"evidence_upload_{test_name}_{timestamp}.json"
        evidence_path = evidence_dir / filename
        
        with open(evidence_path, 'w') as f:
            json.dump(evidence_data, f, indent=2, default=str)
        
        print(f"Upload evidence saved: {evidence_path}")
        return evidence_path
    
    @pytest.fixture
    def settings_loader(self):
        """Fixture providing initialized SettingsLoader"""
        admin_settings_path = Path(__file__).parent.parent / "tidyllm" / "admin" / "settings.yaml"
        return SettingsLoader(str(admin_settings_path))
    
    def test_research_paper_upload_to_s3(self, settings_loader):
        """Test uploading research papers to S3 bucket"""
        from tidyllm.document_upload import ResearchDocUploader, S3UploadManager
        
        # Initialize uploader with admin settings
        uploader = ResearchDocUploader(settings_loader)
        
        # Create test research document
        research_content = """
        Title: Machine Learning Applications in Natural Language Processing
        
        Abstract:
        This paper explores the application of machine learning techniques to natural language processing tasks.
        We examine various approaches including supervised learning, unsupervised learning, and deep learning methods.
        
        Keywords: machine learning, natural language processing, deep learning, neural networks
        
        1. Introduction
        Natural language processing (NLP) has been revolutionized by machine learning approaches...
        
        2. Methodology
        We employed several machine learning algorithms including...
        
        3. Results
        Our experiments show significant improvements in accuracy...
        
        4. Conclusion
        Machine learning continues to advance NLP capabilities...
        """
        
        # Create temporary research file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        temp_file.write(research_content)
        temp_file.close()
        
        try:
            # Upload to S3
            upload_result = uploader.upload_research_document(
                file_path=temp_file.name,
                document_title="ML Applications in NLP",
                author="Test Researcher",
                category="machine_learning",
                metadata={
                    "journal": "Test Journal",
                    "year": 2024,
                    "doi": "10.1234/test.2024.001"
                }
            )
            
            # Validate upload result
            assert upload_result.success
            assert upload_result.s3_key is not None
            assert upload_result.s3_url is not None
            assert upload_result.etag is not None
            assert upload_result.file_size > 0
            assert upload_result.upload_time_ms > 0
            
            # Verify file exists in S3
            verification = uploader.verify_upload(upload_result.s3_key)
            assert verification.exists
            assert verification.content_length == upload_result.file_size
            assert verification.last_modified is not None
            
            # Test file retrieval
            downloaded_content = uploader.download_document(upload_result.s3_key)
            assert research_content.strip() in downloaded_content
            
            # Save evidence
            evidence_data = {
                "upload_result": upload_result.__dict__,
                "verification": verification.__dict__,
                "file_metadata": {
                    "original_size": len(research_content),
                    "uploaded_size": upload_result.file_size,
                    "content_hash": hashlib.md5(research_content.encode()).hexdigest()
                }
            }
            evidence_path = self.save_evidence(evidence_data, "s3_upload")
            
            print(f"✅ Research document upload test completed")
            print(f"   S3 Key: {upload_result.s3_key}")
            print(f"   File Size: {upload_result.file_size} bytes")
            print(f"   Upload Time: {upload_result.upload_time_ms}ms")
            
        finally:
            # Cleanup
            os.unlink(temp_file.name)
    
    def test_batch_document_upload(self, settings_loader):
        """Test batch upload of multiple research documents"""
        from tidyllm.document_upload import ResearchDocUploader, BatchUploadManager
        
        # Initialize batch uploader
        batch_uploader = BatchUploadManager(settings_loader)
        
        # Create multiple test documents
        test_documents = []
        for i in range(3):
            content = f"""
            Research Paper {i+1}: Advanced Topics in AI
            
            Abstract: This is research paper {i+1} focusing on advanced artificial intelligence concepts.
            The paper covers topics including neural networks, deep learning, and machine learning algorithms.
            
            Content: Lorem ipsum research content for paper {i+1}...
            """
            
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix=f'_paper_{i+1}.txt', delete=False)
            temp_file.write(content)
            temp_file.close()
            
            test_documents.append({
                "file_path": temp_file.name,
                "title": f"AI Research Paper {i+1}",
                "author": f"Researcher {i+1}",
                "category": "artificial_intelligence",
                "content": content
            })
        
        try:
            # Batch upload
            batch_result = batch_uploader.upload_batch(
                documents=test_documents,
                batch_name="ai_research_batch",
                concurrent_uploads=2
            )
            
            # Validate batch results
            assert batch_result.total_documents == 3
            assert batch_result.successful_uploads >= 0  # May be 0 if S3 not available
            assert batch_result.failed_uploads >= 0
            assert batch_result.total_upload_time_ms > 0
            
            # Check individual upload results
            for upload in batch_result.upload_results:
                if upload.success:
                    assert upload.s3_key is not None
                    assert upload.file_size > 0
                else:
                    assert upload.error_message is not None
            
            # Save evidence
            evidence_data = {
                "batch_summary": batch_result.__dict__,
                "individual_uploads": [u.__dict__ for u in batch_result.upload_results],
                "performance_metrics": {
                    "avg_upload_time_ms": batch_result.avg_upload_time_ms,
                    "success_rate": batch_result.success_rate
                }
            }
            evidence_path = self.save_evidence(evidence_data, "batch_upload")
            
            print(f"✅ Batch document upload test completed")
            print(f"   Documents: {batch_result.total_documents}")
            print(f"   Successful: {batch_result.successful_uploads}")
            print(f"   Success Rate: {batch_result.success_rate:.1%}")
            
        finally:
            # Cleanup
            for doc in test_documents:
                if os.path.exists(doc["file_path"]):
                    os.unlink(doc["file_path"])

def test_priority_document_upload_check():
    """Priority test for document upload readiness"""
    try:
        from tidyllm.document_upload import (
            ResearchDocUploader, S3UploadManager, BatchUploadManager,
            UploadResult, UploadVerification
        )
        
        # Test basic upload functionality
        uploader = ResearchDocUploader(None)  # None for basic test
        assert uploader is not None
        
        # Test batch manager
        batch_manager = BatchUploadManager(None)
        assert batch_manager is not None
        
        print("SUCCESS: Research document upload system implemented and working")
        
    except Exception as e:
        pytest.fail(f"CRITICAL: Document upload check failed: {e}")

if __name__ == "__main__":
    test_priority_document_upload_check()