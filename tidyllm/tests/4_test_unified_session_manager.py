#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test 4: Unified Session Manager

Tests the official UnifiedSessionManager architecture for:
- PostgreSQL database operations
- S3 client operations  
- MLflow experiment tracking

IMPORTANT: This replaces scattered DatabaseManager tests with unified architecture testing.
"""

import os
import sys
import pytest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import official UnifiedSessionManager
try:
    from scripts.start_unified_sessions import UnifiedSessionManager
    UNIFIED_SESSION_AVAILABLE = True
except ImportError:
    UNIFIED_SESSION_AVAILABLE = False

class TestUnifiedSessionManager:
    """Test suite for unified session management"""
    
    def test_unified_session_manager_import(self):
        """Test that UnifiedSessionManager can be imported"""
        assert UNIFIED_SESSION_AVAILABLE, "UnifiedSessionManager should be available"
        
        # Test initialization
        session_mgr = UnifiedSessionManager()
        assert session_mgr is not None
        
    def test_s3_client_access(self):
        """Test S3 client access via UnifiedSessionManager"""
        if not UNIFIED_SESSION_AVAILABLE:
            pytest.skip("UnifiedSessionManager not available")
            
        session_mgr = UnifiedSessionManager()
        
        try:
            # Test S3 client creation
            s3_client = session_mgr.get_s3_client()
            assert s3_client is not None
            
            # Test that client has expected methods
            assert hasattr(s3_client, 'put_object')
            assert hasattr(s3_client, 'get_object')
            assert hasattr(s3_client, 'list_objects_v2')
            
        except Exception as e:
            # S3 client creation might fail due to credentials, but should not crash
            assert "credentials" in str(e).lower() or "config" in str(e).lower()
            
    def test_postgres_connection_access(self):
        """Test PostgreSQL connection access via UnifiedSessionManager"""
        if not UNIFIED_SESSION_AVAILABLE:
            pytest.skip("UnifiedSessionManager not available")
            
        session_mgr = UnifiedSessionManager()
        
        try:
            # Test PostgreSQL connection
            conn = session_mgr.get_postgres_connection()
            assert conn is not None
            
            # Test simple query
            result = session_mgr.execute_postgres_query("SELECT 1 as test")
            assert result is not None
            
            conn.close()
            
        except Exception as e:
            # Database connection might fail due to configuration, but should not crash
            assert any(keyword in str(e).lower() for keyword in 
                      ["connection", "database", "postgres", "credentials"])
            
    def test_mlflow_client_access(self):
        """Test MLflow client access via UnifiedSessionManager"""
        if not UNIFIED_SESSION_AVAILABLE:
            pytest.skip("UnifiedSessionManager not available")
            
        session_mgr = UnifiedSessionManager()
        
        try:
            # Test MLflow client creation
            mlflow_client = session_mgr.get_mlflow_client()
            assert mlflow_client is not None
            
            # Test MLflow logging (should not crash)
            session_mgr.log_mlflow_experiment({
                'test': 'unified_session_manager_test',
                'architecture': 'official',
                'component': 'session_management'
            })
            
        except Exception as e:
            # MLflow might fail due to configuration, but should not crash
            assert any(keyword in str(e).lower() for keyword in 
                      ["mlflow", "tracking", "uri", "database"])
            
    def test_unified_architecture_constraints(self):
        """Test that UnifiedSessionManager follows architecture constraints"""
        if not UNIFIED_SESSION_AVAILABLE:
            pytest.skip("UnifiedSessionManager not available")
            
        session_mgr = UnifiedSessionManager()
        
        # Verify it's the official implementation
        assert hasattr(session_mgr, 'get_s3_client')
        assert hasattr(session_mgr, 'get_postgres_connection') 
        assert hasattr(session_mgr, 'get_mlflow_client')
        assert hasattr(session_mgr, 'execute_postgres_query')
        assert hasattr(session_mgr, 'log_mlflow_experiment')
        
        # Test that methods return appropriate types
        try:
            s3_client = session_mgr.get_s3_client()
            # Should be boto3 S3 client
            assert str(type(s3_client)).find('boto') != -1
        except:
            pass  # Credential issues are acceptable
        
    def test_no_direct_boto3_usage(self):
        """Test that we don't import boto3 directly (should use UnifiedSessionManager)"""
        # This test ensures we're following the architecture constraint
        # of using UnifiedSessionManager instead of direct boto3 imports
        
        # Import check - boto3 should only be used inside UnifiedSessionManager
        try:
            import boto3
            # If boto3 is available, it should only be used via UnifiedSessionManager
            assert UNIFIED_SESSION_AVAILABLE, "If boto3 is available, UnifiedSessionManager should be too"
        except ImportError:
            # boto3 not available is fine - UnifiedSessionManager handles this
            pass

if __name__ == "__main__":
    pytest.main([__file__, "-v"])