"""
Shared database utilities for TidyLLM demos

MIGRATED TO USE OFFICIAL UNIFIEDSESSIONMANAGER
==============================================

This module now uses the official UnifiedSessionManager instead of 
creating direct psycopg2 connections, following TidyLLM architecture constraints.
"""
import streamlit as st
from typing import Dict, Any, Optional
import sys
from pathlib import Path

# Add project root to path for UnifiedSessionManager import
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import official UnifiedSessionManager (required architecture)
try:
    from scripts.start_unified_sessions import UnifiedSessionManager
    UNIFIED_SESSION_AVAILABLE = True
except ImportError:
    UNIFIED_SESSION_AVAILABLE = False
    st.error("UnifiedSessionManager not available. Check scripts/start_unified_sessions.py")

from .utils import load_settings

class DatabaseManager:
    """
    MIGRATED: Database Manager using UnifiedSessionManager
    
    No longer creates direct psycopg2 connections.
    All database operations go through official UnifiedSessionManager.
    """
    
    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        self.settings = settings or load_settings()
        
        # Use official UnifiedSessionManager instead of direct connections
        if UNIFIED_SESSION_AVAILABLE:
            self.session_mgr = UnifiedSessionManager()
            self.connected = True
            st.success("✅ Connected via UnifiedSessionManager (official architecture)")
        else:
            self.session_mgr = None
            self.connected = False
            st.error("❌ UnifiedSessionManager not available")
    
    def connect(self) -> bool:
        """Test database connection via UnifiedSessionManager"""
        if not UNIFIED_SESSION_AVAILABLE:
            st.error("UnifiedSessionManager not available")
            return False
        
        try:
            # Test connection through UnifiedSessionManager
            test_result = self.session_mgr.execute_postgres_query("SELECT 1")
            if test_result:
                self.connected = True
                st.success("✅ Database connection verified via UnifiedSessionManager")
                return True
            else:
                self.connected = False
                st.error("❌ Database connection test failed")
                return False
                
        except Exception as e:
            st.error(f"❌ Database connection failed: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """
        Connection cleanup handled by UnifiedSessionManager
        No manual disconnection needed.
        """
        st.info("ℹ️ Connection managed by UnifiedSessionManager - no manual disconnect needed")
        self.connected = False
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> Optional[list]:
        """Execute database query via UnifiedSessionManager"""
        
        if not self.connected or not self.session_mgr:
            st.error("❌ Database not connected via UnifiedSessionManager")
            return None
        
        try:
            # Execute query through official UnifiedSessionManager
            if params:
                results = self.session_mgr.execute_postgres_query(query, params)
            else:
                results = self.session_mgr.execute_postgres_query(query)
            
            # Log successful query to MLflow via UnifiedSessionManager
            self.session_mgr.log_mlflow_experiment({
                'operation': 'demo_database_query',
                'query_type': 'SELECT' if query.strip().upper().startswith('SELECT') else 'MODIFY',
                'has_parameters': params is not None,
                'success': True
            })
            
            return results
                
        except Exception as e:
            st.error(f"❌ Query execution failed: {e}")
            
            # Log failed query to MLflow
            try:
                self.session_mgr.log_mlflow_experiment({
                    'operation': 'demo_database_query',
                    'query_type': 'SELECT' if query.strip().upper().startswith('SELECT') else 'MODIFY',
                    'has_parameters': params is not None,
                    'success': False,
                    'error': str(e)
                })
            except:
                pass  # Don't fail on logging failure
            
            return None
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            result = self.execute_query("SELECT 1")
            return result is not None and len(result) > 0
        except Exception:
            return False
    
    def get_table_info(self, table_name: str) -> Optional[list]:
        """Get table structure information"""
        query = """
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_name = %s
        ORDER BY ordinal_position
        """
        return self.execute_query(query, (table_name,))
    
    def get_table_count(self, table_name: str) -> Optional[int]:
        """Get row count for a table"""
        query = f"SELECT COUNT(*) FROM {table_name}"
        result = self.execute_query(query)
        if result:
            return result[0][0]
        return None

def get_database_manager() -> DatabaseManager:
    """Get a database manager instance"""
    return DatabaseManager()


