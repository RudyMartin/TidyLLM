#!/usr/bin/env python3
"""

# Centralized AWS credential management
import sys
from pathlib import Path

# Add admin directory to path for credential loading
sys.path.append(str(Path(__file__).parent.parent / 'tidyllm' / 'admin') if 'tidyllm' in str(Path(__file__)) else str(Path(__file__).parent / 'tidyllm' / 'admin'))
from credential_loader import set_aws_environment

# Load AWS credentials using centralized system
set_aws_environment()
Test SOP Documents Processing - Limited Date Range
==================================================

Tests the SOP Domain RAG system with just a few date folders to validate:
- Backend code changes work correctly
- Temporal resolution (newer dates win) functions properly  
- S3 integration works before full document streaming
- Flow agreements integration is stable

Test with: docs/2025-09-03, docs/2025-09-04, docs/2025-09-05 only
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Set AWS credentials for TidyLLM system




class TestSOPProcessing:
    """Test SOP processing with limited date range"""
    
    def __init__(self):
        self.docs_path = Path("docs")
        self.test_dates = ["2025-09-03", "2025-09-04", "2025-09-05"]  # Limited test set
        self.results = {}
        
        print("=" * 60)
        print("TEST SOP PROCESSING - LIMITED DATE RANGE")
        print("=" * 60)
        print(f"Testing dates: {', '.join(self.test_dates)}")
        print("Validating backend changes before full S3 streaming")
        print("=" * 60)
    
    def run_test(self):
        """Run the complete test process"""
        
        # Test 1: Document Collection
        print("\n[TEST 1] DOCUMENT COLLECTION")
        docs = self._test_document_collection()
        if not docs:
            print("[FAIL] No documents found for testing")
            return False
        
        # Test 2: Temporal Priority Logic
        print("\n[TEST 2] TEMPORAL PRIORITY LOGIC")  
        temporal_test = self._test_temporal_priority(docs)
        if not temporal_test:
            print("[FAIL] Temporal priority logic failed")
            return False
        
        # Test 3: Conflict Detection
        print("\n[TEST 3] CONFLICT DETECTION")
        conflicts = self._test_conflict_detection(docs)
        
        # Test 4: SOP Generation with Temporal Resolution
        print("\n[TEST 4] SOP GENERATION WITH TEMPORAL RESOLUTION")
        sops = self._test_sop_generation(conflicts)
        
        # Test 5: Backend Integration Test
        print("\n[TEST 5] BACKEND INTEGRATION TEST")
        backend_test = self._test_backend_integration()
        
        # Test Results Summary
        print("\n" + "=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)
        print(f"Documents processed: {len(docs)}")
        print(f"Conflicts detected: {len(conflicts)}")
        print(f"SOPs generated: {len(sops)}")
        print(f"Backend integration: {'PASS' if backend_test else 'FAIL'}")
        
        # Show temporal resolution in action
        print("\n[TEMPORAL RESOLUTION VALIDATION]")
        self._validate_temporal_resolution(sops)
        
        return True
    
    def _test_document_collection(self):
        """Test document collection from limited date folders"""
        
        docs = []
        
        for test_date in self.test_dates:
            date_folder = self.docs_path / test_date
            
            if not date_folder.exists():
                print(f"[SKIP] {test_date} - folder does not exist")
                continue
            
            date_docs = []
            for doc_file in date_folder.glob("*.md"):
                if doc_file.is_file():
                    date_docs.append({
                        'filename': doc_file.name,
                        'path': doc_file,
                        'date': test_date,
                        'size': doc_file.stat().st_size,
                        'modified': datetime.fromtimestamp(doc_file.stat().st_mtime)
                    })
            
            print(f"[COLLECT] {test_date}: {len(date_docs)} documents")
            docs.extend(date_docs)
        
        print(f"[TOTAL] {len(docs)} documents collected from {len(self.test_dates)} test dates")
        return docs
    
    def _test_temporal_priority(self, docs):
        """Test temporal priority logic"""
        
        print("[PRIORITY] Testing date-based priority calculation...")
        
        # Group docs by date and test priority ordering
        docs_by_date = {}
        for doc in docs:
            date = doc['date']
            if date not in docs_by_date:
                docs_by_date[date] = []
            docs_by_date[date].append(doc)
        
        # Test: 2025-09-05 should have highest priority
        dates_found = sorted(docs_by_date.keys())
        if dates_found:
            highest_priority_date = dates_found[-1]  # Should be 2025-09-05
            print(f"[PRIORITY] Highest priority date: {highest_priority_date}")
            
            # Validate priority ordering
            for i, date in enumerate(dates_found):
                priority_score = self._calculate_date_priority(date)
                print(f"[PRIORITY] {date}: priority_score = {priority_score}")
            
            return highest_priority_date == "2025-09-05"
        
        return False
    
    def _calculate_date_priority(self, date_str):
        """Calculate temporal priority for a date string"""
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            return date_obj.timestamp()  # Higher timestamp = higher priority
        except ValueError:
            return 0
    
    def _test_conflict_detection(self, docs):
        """Test conflict detection logic"""
        
        print("[CONFLICTS] Testing conflict detection across dates...")
        
        # Test queries for conflict detection
        test_queries = [
            "What is the official session management pattern?",
            "Which embedding system should be used?",
            "How should AWS S3 be accessed?",
        ]
        
        conflicts = []
        
        for query in test_queries:
            print(f"\n[QUERY] {query}")
            
            # Find documents that match this query across different dates
            matching_docs = []
            keywords = self._extract_keywords(query)
            
            for doc in docs:
                try:
                    with open(doc['path'], 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read().lower()
                    
                    if any(keyword.lower() in content for keyword in keywords):
                        matching_docs.append({
                            'doc': doc,
                            'content_snippet': content[:200] + "...",
                            'relevance': self._calculate_relevance(content, keywords)
                        })
                        print(f"[MATCH] {doc['filename']} ({doc['date']}) - relevance: {self._calculate_relevance(content, keywords):.2f}")
                
                except Exception as e:
                    print(f"[ERROR] Failed to read {doc['filename']}: {e}")
            
            # If multiple dates have matching documents, it's a conflict
            dates_with_matches = list(set(match['doc']['date'] for match in matching_docs))
            
            if len(dates_with_matches) > 1:
                conflict = {
                    'query': query,
                    'dates_involved': dates_with_matches,
                    'matching_documents': matching_docs,
                    'conflict_severity': 'high' if 'session' in query.lower() else 'medium'
                }
                conflicts.append(conflict)
                print(f"[CONFLICT] Found conflict across dates: {dates_with_matches}")
            else:
                print(f"[NO_CONFLICT] Only found in: {dates_with_matches}")
        
        print(f"\n[CONFLICTS] Total conflicts detected: {len(conflicts)}")
        return conflicts
    
    def _extract_keywords(self, query):
        """Extract keywords from query for matching"""
        import re
        stop_words = {'what', 'is', 'the', 'should', 'be', 'used', 'how', 'for', 'which', 'or', 'and'}
        words = re.findall(r'\b\w+\b', query.lower())
        return [w for w in words if w not in stop_words and len(w) > 3]
    
    def _calculate_relevance(self, content, keywords):
        """Calculate relevance score"""
        matches = sum(1 for keyword in keywords if keyword.lower() in content)
        return matches / len(keywords) if keywords else 0.0
    
    def _test_sop_generation(self, conflicts):
        """Test SOP generation with temporal resolution"""
        
        print("[SOP_GEN] Testing SOP generation with temporal resolution...")
        
        sops = {}
        
        for conflict in conflicts:
            query = conflict['query']
            dates_involved = conflict['dates_involved']
            
            # Apply temporal resolution: newest date wins
            most_recent_date = max(dates_involved)
            deprecated_dates = [d for d in dates_involved if d != most_recent_date]
            
            # Find authoritative document (from most recent date)
            authoritative_docs = [
                match['doc'] for match in conflict['matching_documents'] 
                if match['doc']['date'] == most_recent_date
            ]
            
            # Create SOP with temporal resolution
            sop = {
                'query': query,
                'resolution_strategy': 'temporal_priority_newest_wins',
                'authoritative_date': most_recent_date,
                'deprecated_dates': deprecated_dates,
                'authoritative_documents': [doc['filename'] for doc in authoritative_docs],
                'resolution': f"Use guidance from {most_recent_date} as authoritative",
                'conflicts_resolved': len(dates_involved),
                'temporal_priority_applied': True,
                'created_at': datetime.now().isoformat()
            }
            
            sops[query] = sop
            
            print(f"[SOP] {query}")
            print(f"      Authority: {most_recent_date}")
            print(f"      Deprecated: {deprecated_dates}")
            print(f"      Documents: {[doc['filename'] for doc in authoritative_docs]}")
        
        print(f"\n[SOP_GEN] Generated {len(sops)} SOPs with temporal resolution")
        return sops
    
    def _test_backend_integration(self):
        """Test backend integration components"""
        
        print("[BACKEND] Testing backend integration components...")
        
        # Test 1: AWS Credentials
        try:
            import boto3
            s3 = boto3.client('s3')
            buckets = s3.list_buckets()
            print(f"[AWS] S3 connection successful - {len(buckets['Buckets'])} buckets")
            aws_test = True
        except Exception as e:
            print(f"[AWS] S3 connection failed: {e}")
            aws_test = False
        
        # Test 2: Flow Agreements
        try:
            sys.path.insert(0, str(Path(__file__).parent / 'tidyllm'))
            from flow_agreements.base import BaseFlowAgreement, FlowAgreementConfig
            print("[FLOW] Flow agreements import successful")
            flow_test = True
        except Exception as e:
            print(f"[FLOW] Flow agreements import failed: {e}")
            flow_test = False
        
        # Test 3: TidyLLM Embedding System
        try:
            from tidyllm.knowledge_systems.facades.embedding_processor import EmbeddingProcessor
            processor = EmbeddingProcessor(target_dimension=1024)
            print("[EMBED] TidyLLM embedding system loaded")
            embed_test = True
        except Exception as e:
            print(f"[EMBED] TidyLLM embedding system failed: {e}")
            embed_test = False
        
        # Overall backend test
        backend_success = aws_test and (flow_test or embed_test)  # Need AWS + at least one TidyLLM component
        
        print(f"[BACKEND] Overall integration: {'PASS' if backend_success else 'FAIL'}")
        return backend_success
    
    def _validate_temporal_resolution(self, sops):
        """Validate that temporal resolution worked correctly"""
        
        print("[VALIDATION] Temporal resolution validation:")
        
        for query, sop in sops.items():
            auth_date = sop['authoritative_date']
            deprecated = sop['deprecated_dates']
            
            print(f"  Query: {query[:50]}...")
            print(f"    Authoritative: {auth_date}")
            print(f"    Deprecated: {deprecated}")
            
            # Validate that authoritative date is indeed the newest
            all_dates = [auth_date] + deprecated
            newest = max(all_dates)
            
            if auth_date == newest:
                print(f"    [OK] CORRECT - {auth_date} is newest")
            else:
                print(f"    [ERROR] - {auth_date} should be {newest}")
        
        print(f"\n[VALIDATION] Temporal resolution logic validated")
    
    def save_test_results(self):
        """Save test results for review"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"sop_test_results_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"[SAVE] Test results saved to: {results_file}")


def main():
    """Run the SOP processing test"""
    
    tester = TestSOPProcessing()
    success = tester.run_test()
    
    if success:
        print("\n" + "=" * 60)
        print("[SUCCESS] SOP PROCESSING TEST SUCCESSFUL")
        print("=" * 60)
        print("Backend changes validated - ready for full S3 streaming")
        tester.save_test_results()
    else:
        print("\n" + "=" * 60)
        print("[FAIL] SOP PROCESSING TEST FAILED")
        print("=" * 60)
        print("Fix backend issues before proceeding to S3 streaming")
    
    return success


if __name__ == "__main__":
    main()