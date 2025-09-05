#!/usr/bin/env python3
"""
Evidence Validation Script

IMPORTANT FOR AGENTS/LLMs:
This script helps validate that tests are producing REAL evidence
instead of mock/simulated results. Use this to verify authentic
system functionality vs fallback behavior.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path


def validate_evidence_directory():
    """Validate evidence directory and contents"""
    evidence_dir = Path(__file__).parent
    
    print("EVIDENCE DIRECTORY VALIDATION")
    print("=" * 50)
    
    # Check directory exists
    if not evidence_dir.exists():
        print("❌ ERROR: Evidence directory does not exist")
        return False
    
    # Find evidence files
    chat_files = list(evidence_dir.glob("evidence_chat_*.json"))
    upload_files = list(evidence_dir.glob("evidence_upload_*.json"))
    
    print(f"Evidence directory: {evidence_dir}")
    print(f"Chat evidence files: {len(chat_files)}")
    print(f"Upload evidence files: {len(upload_files)}")
    
    if len(chat_files) == 0 and len(upload_files) == 0:
        print("WARNING: No evidence files found")
        print("   This suggests tests are running in mock/simulation mode")
        print("   AGENTS should verify API credentials are available")
        return False
    
    # Validate evidence files
    real_evidence_count = 0
    mock_evidence_count = 0
    
    print("\nEVIDENCE VALIDATION RESULTS:")
    print("-" * 30)
    
    all_files = chat_files + upload_files
    for evidence_file in sorted(all_files):
        try:
            with open(evidence_file, 'r') as f:
                evidence = json.load(f)
            
            status = evidence.get('status', 'UNKNOWN')
            timestamp = evidence.get('timestamp', 'No timestamp')
            test_type = evidence.get('test_type', 'Unknown type')
            
            if 'SUCCESS_REAL' in status:
                status_icon = "REAL"
                real_evidence_count += 1
            elif 'MOCK' in status or 'SIMULATED' in status:
                status_icon = "MOCK"
                mock_evidence_count += 1
            else:
                status_icon = "UNKNOWN"
            
            print(f"   {status_icon} {evidence_file.name}")
            print(f"      Type: {test_type}")
            print(f"      Time: {timestamp}")
            print(f"      Status: {status}")
            
            # Additional validation for chat evidence
            if 'chat' in evidence_file.name:
                response = evidence.get('response', '')
                response_length = evidence.get('response_length', 0)
                
                if response_length > 50 and 'mock response' not in response.lower():
                    print(f"      Response: {response_length} chars (appears genuine)")
                else:
                    print(f"      Response: {response_length} chars (may be mock)")
            
            # Additional validation for upload evidence
            if 'upload' in evidence_file.name:
                s3_url = evidence.get('s3_url', '')
                etag = evidence.get('etag', '')
                
                if s3_url and etag:
                    print(f"      S3 URL: {s3_url}")
                    print(f"      ETag: {etag[:16]}...")
                else:
                    print(f"      S3 details incomplete (may be mock)")
            
            print()
            
        except Exception as e:
            print(f"   ERROR reading {evidence_file.name}: {e}")
    
    # Summary
    total_evidence = real_evidence_count + mock_evidence_count
    if total_evidence > 0:
        real_percentage = (real_evidence_count / total_evidence) * 100
        print(f"EVIDENCE SUMMARY:")
        print(f"   Real evidence: {real_evidence_count}/{total_evidence} ({real_percentage:.1f}%)")
        print(f"   Mock evidence: {mock_evidence_count}/{total_evidence}")
        
        if real_evidence_count > 0:
            print(f"SUCCESS: Found {real_evidence_count} real evidence files")
            print("   Tests are successfully connecting to actual services")
        else:
            print("WARNING: All evidence appears to be mock/simulated")
            print("   AGENTS should check API credentials and try real connections")
        
        return real_evidence_count > 0
    
    return False


def check_evidence_freshness():
    """Check if evidence is recent (within last 24 hours)"""
    evidence_dir = Path(__file__).parent
    recent_threshold = datetime.now() - timedelta(hours=24)
    
    print("\nEVIDENCE FRESHNESS CHECK:")
    print("-" * 30)
    
    evidence_files = list(evidence_dir.glob("evidence_*.json"))
    fresh_evidence = 0
    
    for evidence_file in evidence_files:
        try:
            with open(evidence_file, 'r') as f:
                evidence = json.load(f)
            
            timestamp_str = evidence.get('timestamp', '')
            if timestamp_str:
                # Parse ISO timestamp
                file_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                file_time = file_time.replace(tzinfo=None)  # Remove timezone for comparison
                
                if file_time > recent_threshold:
                    print(f"FRESH: {evidence_file.name} ({timestamp_str})")
                    fresh_evidence += 1
                else:
                    print(f"OLD: {evidence_file.name} ({timestamp_str})")
        
        except Exception as e:
            print(f"ERROR checking {evidence_file.name}: {e}")
    
    if fresh_evidence > 0:
        print(f"Found {fresh_evidence} fresh evidence files (< 24 hours)")
    else:
        print("No fresh evidence found - tests may not be running recently")


def agent_guidance():
    """Provide guidance for agents/LLMs"""
    print("\nGUIDANCE FOR AGENTS/LLMs:")
    print("=" * 50)
    print()
    print("IF YOU SEE MOSTLY MOCK/SIMULATED EVIDENCE:")
    print("1. Check for API keys in environment variables")
    print("2. Verify admin/settings.yaml has real credentials")
    print("3. Test connections manually before running full test suite")
    print("4. Look for 'Mock response' or 'simulated' in test outputs")
    print()
    print("WHEN REAL CREDENTIALS ARE AVAILABLE:")
    print("1. Prioritize real API calls over mock responses")
    print("2. Save complete request/response pairs as evidence")
    print("3. Include performance metrics and timing data")
    print("4. Delete old mock evidence when real evidence is generated")
    print()
    print("EVIDENCE QUALITY INDICATORS:")
    print("+ Status contains 'SUCCESS_REAL_API_CALL' or 'SUCCESS_REAL_S3_UPLOAD'")
    print("+ Response lengths > 50 chars with substantive content")
    print("+ S3 evidence includes ETags and valid URLs")
    print("+ Timestamps are recent (within hours, not days)")
    print("+ No 'mock', 'simulated', or 'fallback' in content")


if __name__ == "__main__":
    print("TidyLLM Evidence Validation")
    print("=" * 50)
    print("This script validates test evidence to ensure real")
    print("API calls are being made vs mock/simulated responses")
    print()
    
    has_real_evidence = validate_evidence_directory()
    check_evidence_freshness()
    agent_guidance()
    
    print("\n" + "=" * 50)
    if has_real_evidence:
        print("VALIDATION PASSED: Real evidence found")
        exit(0)
    else:
        print("VALIDATION WARNING: Limited or no real evidence")
        print("   Consider running tests with real credentials")
        exit(1)