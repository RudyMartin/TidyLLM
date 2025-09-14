"""
Database Validator for TidyLLM
=============================

Corporate-safe database connectivity validation with timeout protection.
"""

from typing import Dict, Any
from .base import BaseValidator

# LEGACY CODE - Commented out since we now use unified sessions system
# try:
#     import polars as pl
#     POLARS_AVAILABLE = True
# except ImportError:
#     POLARS_AVAILABLE = False


class DatabaseValidator(BaseValidator):
    """Corporate-safe database connectivity validator."""
    
    def validate_database_connectivity(self) -> Dict[str, Any]:
        """
        Validate database connectivity with corporate safety.
        
        Returns:
            Database validation results
        """
        
        # Detect corporate environment first
        env_info = self.detect_corporate_environment()
        
        if self.corporate_mode:
            return self.corporate_safe_result(
                'Database',
                'Corporate environment - database validation skipped to prevent potential network delays'
            )
        
        # Standard database validation logic would go here
        return self.run_with_timeout(self._test_database_connection, 'Database')
    
    def _test_database_connection(self) -> Dict[str, Any]:
        """Test actual database connection using unified sessions system."""
        
        try:
            # Use unified sessions system for database connection
            from tidyllm.infrastructure.session.unified import get_global_session_manager
            
            session_manager = get_global_session_manager()
            if not session_manager:
                return {
                    'status': 'error',
                    'message': 'Unified session manager not available'
                }
            
            # Get PostgreSQL connection from unified sessions
            postgres_conn = session_manager.get_postgres_connection()
            if not postgres_conn:
                return {
                    'status': 'error',
                    'message': 'PostgreSQL connection not available from unified sessions'
                }
            
            # Test the connection with a simple query
            with postgres_conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                
            return {
                'status': 'success',
                'message': 'Database connection successful via unified sessions',
                'connection_details': {
                    'connection_type': 'unified_sessions',
                    'test_result': result[0] if result else None
                }
            }
                
        except ImportError as e:
            return {
                'status': 'error',
                'message': f'Unified sessions system not available: {e}'
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Database connection failed: {e}'
            }