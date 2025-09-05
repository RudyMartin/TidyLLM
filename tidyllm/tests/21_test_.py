#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test 8: Error Handling & Recovery

Tests system resilience under various failure scenarios.
Validates error handling, retry mechanisms, and recovery procedures.

IMPORTANT FOR AGENTS/LLMs:
- DO NOT simulate errors - test REAL failure scenarios when safe
- VALIDATE actual retry mechanisms, not mock implementations
- TEST graceful degradation under real service interruptions
- DOCUMENT recovery procedures and failure modes
"""

import os
import sys
import json
import pytest
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tidyllm.settings_loader import SettingsLoader

class TestErrorHandlingRecovery:
    """Test suite for error handling and recovery"""
    
    def save_evidence(self, evidence_data, test_name):
        """Save error handling evidence"""
        evidence_dir = Path(__file__).parent / "EVIDENCE"
        evidence_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"evidence_error_{test_name}_{timestamp}.json"
        evidence_path = evidence_dir / filename
        
        with open(evidence_path, 'w') as f:
            json.dump(evidence_data, f, indent=2, default=str)
        
        print(f"Error handling evidence saved: {evidence_path}")
        return evidence_path
    
    @pytest.fixture
    def settings_loader(self):
        """Fixture providing initialized SettingsLoader"""
        admin_settings_path = Path(__file__).parent.parent / "tidyllm" / "admin" / "settings.yaml"
        return SettingsLoader(str(admin_settings_path))
    
    def test_network_interruption_recovery(self, settings_loader):
        """Test recovery from network interruptions"""
        from tidyllm.error_handling import with_retry, NetworkError, RetryConfig
        
        # Test retry mechanism with simulated network failure
        retry_config = RetryConfig(max_attempts=3, base_delay=0.1)
        
        attempt_count = 0
        
        @with_retry(retry_config, (NetworkError,))
        def simulate_network_call():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise NetworkError("Simulated network failure")
            return "Success after retries"
        
        result = simulate_network_call()
        assert result == "Success after retries"
        assert attempt_count == 3
        print("✅ Network interruption recovery working")
        
    def test_api_rate_limiting_responses(self, settings_loader):
        """Test handling of API rate limiting"""
        from tidyllm.error_handling import handle_api_error, APIError, CircuitBreaker
        
        # Test circuit breaker pattern for API rate limiting
        circuit_breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1, expected_exception=APIError)
        
        call_count = 0
        
        @circuit_breaker
        def simulate_rate_limited_api():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise APIError("Rate limit exceeded")
            return "API call successful"
        
        # First two calls should fail and open circuit
        try:
            simulate_rate_limited_api()
            assert False, "Should have raised APIError"
        except APIError:
            pass
        
        try:
            simulate_rate_limited_api()
            assert False, "Should have raised APIError" 
        except APIError:
            pass
        
        # Third call should fail due to circuit breaker
        try:
            simulate_rate_limited_api()
            assert False, "Should have raised TidyLLMError for circuit breaker"
        except Exception as e:
            assert "Circuit breaker OPEN" in str(e)
        
        print("✅ API rate limiting and circuit breaker working")
        
    def test_database_connection_failures(self, settings_loader):
        """Test database connection failure recovery"""
        from tidyllm.error_handling import handle_database_error, DatabaseError, error_collector
        
        # Test database error handling with decorator
        connection_attempts = 0
        
        @handle_database_error
        def simulate_database_connection():
            nonlocal connection_attempts
            connection_attempts += 1
            if connection_attempts < 3:
                raise DatabaseError("Connection timeout")
            return "Database connected successfully"
        
        result = simulate_database_connection()
        assert result == "Database connected successfully"
        assert connection_attempts == 3
        
        # Check error collector recorded the failures
        error_summary = error_collector.get_error_summary()
        assert error_summary['total_errors'] > 0
        
        print("✅ Database connection failure recovery working")
        
    def test_s3_upload_retry_mechanisms(self, settings_loader):
        """Test S3 upload retry logic"""
        from tidyllm.error_handling import handle_storage_error, StorageError, safe_execute
        
        # Test S3 upload with retry logic
        upload_attempts = 0
        
        @handle_storage_error
        def simulate_s3_upload(filename, data):
            nonlocal upload_attempts
            upload_attempts += 1
            if upload_attempts < 3:
                raise StorageError("S3 upload timeout")
            return {"status": "success", "etag": "abc123", "attempts": upload_attempts}
        
        result = simulate_s3_upload("test.txt", "test data")
        assert result["status"] == "success"
        assert result["attempts"] == 3
        
        # Test safe_execute utility
        success_result, error = safe_execute(lambda: "safe operation")
        assert success_result == "safe operation"
        assert error is None
        
        failed_result, error = safe_execute(lambda: 1/0)
        assert failed_result is None
        assert isinstance(error, ZeroDivisionError)
        
        print("✅ S3 upload retry mechanisms working")
        
    def test_graceful_service_degradation(self, settings_loader):
        """Test graceful degradation when services are unavailable"""
        from tidyllm.error_handling import graceful_fallback, ServiceHealthChecker, TidyLLMError
        
        # Test graceful fallback mechanism
        def primary_service():
            raise TidyLLMError("Primary service unavailable", "SERVICE_DOWN")
        
        def fallback_service():
            return "Fallback service response"
        
        @graceful_fallback(fallback_service)
        def service_call():
            return primary_service()
        
        result = service_call()
        assert result == "Fallback service response"
        
        # Test service health checker
        health_checker = ServiceHealthChecker()
        health_status = health_checker.get_overall_health()
        
        assert 'overall_status' in health_status
        assert 'services' in health_status
        assert 'timestamp' in health_status
        
        print("✅ Graceful service degradation working")

def test_priority_error_handling_check():
    """Priority test for error handling readiness"""
    try:
        from tidyllm.error_handling import (
            TidyLLMError, NetworkError, DatabaseError, APIError, StorageError,
            with_retry, CircuitBreaker, ErrorCollector, ServiceHealthChecker
        )
        
        # Test basic error handling functionality
        error = TidyLLMError("Test error", "TEST")
        assert error.error_type == "TEST"
        assert error.message == "Test error"
        
        # Test error collector
        collector = ErrorCollector()
        collector.record_error(error)
        summary = collector.get_error_summary()
        assert summary['total_errors'] == 1
        
        print("SUCCESS: Error handling infrastructure implemented and working")
        
    except Exception as e:
        pytest.fail(f"CRITICAL: Error handling infrastructure check failed: {e}")

if __name__ == "__main__":
    test_priority_error_handling_check()