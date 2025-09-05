#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Centralized Database Connection Manager

This module provides a centralized way to manage database connections,
ensuring consistent connection handling, pooling, and configuration across the application.
"""

import os
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool

logger = logging.getLogger(__name__)


class DatabaseConnectionManager:
    """Centralized database connection manager"""
    
    def __init__(self, database_url: Optional[str] = None, pool_size: int = 5):
        self.database_url = database_url or os.getenv('DATABASE_URL')
        self.pool_size = pool_size
        self.connection_pool = None
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize the connection pool"""
        if not self.database_url:
            logger.warning("DATABASE_URL not found in environment. Database connections will not be available.")
            return
        
        try:
            self.connection_pool = SimpleConnectionPool(
                1, self.pool_size, self.database_url
            )
            logger.info(f"✅ Database connection pool initialized with {self.pool_size} connections")
            
            # Test the connection
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT version();")
                    version = cursor.fetchone()
                    logger.info(f"Connected to: {version[0]}")
                    
        except Exception as e:
            logger.error(f"Failed to initialize database connection pool: {e}")
            self.connection_pool = None
    
    def get_connection(self):
        """Get a connection from the pool"""
        if not self.connection_pool:
            raise RuntimeError("Database connection pool not initialized")
        
        return self.connection_pool.getconn()
    
    def return_connection(self, conn):
        """Return a connection to the pool"""
        if self.connection_pool:
            self.connection_pool.putconn(conn)
    
    @contextmanager
    def get_cursor(self, cursor_factory=None):
        """Get a cursor with automatic connection management"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor(cursor_factory=cursor_factory or RealDictCursor)
            yield cursor
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                self.return_connection(conn)
    
    def execute_query(self, query: str, params: tuple = None) -> Optional[list]:
        """Execute a query and return results"""
        try:
            with self.get_cursor() as cursor:
                cursor.execute(query, params)
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return None
    
    def execute_command(self, command: str, params: tuple = None) -> bool:
        """Execute a command (INSERT, UPDATE, DELETE)"""
        try:
            with self.get_cursor() as cursor:
                cursor.execute(command, params)
                return True
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return False
    
    def ensure_extension(self, extension_name: str) -> bool:
        """Ensure a PostgreSQL extension is available"""
        try:
            with self.get_cursor() as cursor:
                cursor.execute(f"CREATE EXTENSION IF NOT EXISTS {extension_name};")
                logger.info(f"✅ Extension '{extension_name}' ensured")
                return True
        except Exception as e:
            logger.error(f"Failed to ensure extension '{extension_name}': {e}")
            return False
    
    def test_connection(self) -> bool:
        """Test if the database connection is working"""
        try:
            with self.get_cursor() as cursor:
                cursor.execute("SELECT 1;")
                result = cursor.fetchone()
                return result is not None
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def close_pool(self):
        """Close the connection pool"""
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("Database connection pool closed")


# Global instance for easy access
_db_manager = None


def get_database_manager() -> DatabaseConnectionManager:
    """Get the global database connection manager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseConnectionManager()
    return _db_manager


def get_database_connection():
    """Get a database connection (for backward compatibility)"""
    manager = get_database_manager()
    return manager.get_connection()


@contextmanager
def get_database_cursor(cursor_factory=None):
    """Get a database cursor (for backward compatibility)"""
    manager = get_database_manager()
    with manager.get_cursor(cursor_factory) as cursor:
        yield cursor
