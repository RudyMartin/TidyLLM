#!/usr/bin/env python3
"""
TidyLLM Shared Connection Pool Manager
=====================================

Enterprise-grade PostgreSQL connection pooling for all TidyLLM components.
Solves the "one connection vs multiple connection strings" architectural problem.

This replaces per-component connection strategies with a centralized pool that:
- Manages all PostgreSQL connections from a single point
- Provides both direct connections (for adapters) and connection strings (for external tools)
- Monitors usage across all components
- Implements proper resource management and security

Usage:
    from tidyllm.infrastructure.connection_pool import get_global_pool

    # For V2 adapters (preferred)
    pool = get_global_pool()
    result = pool.execute_query("SELECT * FROM experiments", "v2_adapter")

    # For external tools like MLflow
    connection_string = pool.get_connection_string("mlflow")
"""

import os
import sys
import yaml
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from contextlib import contextmanager

class TidyLLMConnectionPool:
    """
    Shared PostgreSQL connection pool for all TidyLLM components.

    This is the single source of truth for database connections across:
    - Main TidyLLM system
    - MLflow tracking server
    - V2 RAG adapters
    - Streamlit applications
    - Background workers
    - API services
    """

    def __init__(self, pg_config: Dict[str, Any]):
        """
        Initialize the shared connection pool.

        Args:
            pg_config: PostgreSQL configuration from settings.yaml
        """
        self.pg_config = pg_config
        self.pool = None
        self.lock = threading.Lock()

        # Statistics and monitoring
        self.stats = {
            "pool_created": datetime.now(),
            "total_queries": 0,
            "active_connections": 0,
            "peak_connections": 0,
            "clients": {},  # client_name -> {queries: int, last_used: datetime}
            "errors": 0,
            "connection_string_requests": 0
        }

        # Pool configuration
        self.min_connections = 2
        self.max_connections = 20

        # Initialize the pool
        self._create_pool()

    def _create_pool(self):
        """Create the PostgreSQL connection pool."""
        try:
            # Use infrastructure delegate instead of direct psycopg2
            try:
                from .infra_delegate import get_infra_delegate
                self.infra = get_infra_delegate()
            except ImportError:
                # Running as script - use absolute import
                from packages.tidyllm.infrastructure.infra_delegate import get_infra_delegate
                self.infra = get_infra_delegate()

            # Import psycopg2 for direct connections
            import psycopg2
            from psycopg2 import pool

            # Create a real psycopg2 connection pool
            self.real_pool = psycopg2.pool.SimpleConnectionPool(
                self.min_connections,
                self.max_connections,
                host=self.pg_config.get('host', 'localhost'),
                port=self.pg_config.get('port', 5432),
                database=self.pg_config.get('database', 'postgres'),
                user=self.pg_config.get('username'),
                password=self.pg_config.get('password')
            )

            # Create a wrapper that uses the real pool
            class DelegatePool:
                def __init__(self, real_pool):
                    self.real_pool = real_pool

                def getconn(self):
                    return self.real_pool.getconn()

                def putconn(self, conn):
                    self.real_pool.putconn(conn)

            self.pool = DelegatePool(self.real_pool)
            print(f"[CONNECTION_POOL] Created real psycopg2 pool")

        except Exception as e:
            print(f"[CONNECTION_POOL] ERROR: Failed to create pool: {e}")
            raise

    @contextmanager
    def get_connection(self, client_name: str = "unknown"):
        """
        Get a connection from the pool (context manager).

        Args:
            client_name: Name of the component requesting the connection

        Yields:
            Database connection from infrastructure delegate
        """
        connection = None
        try:
            with self.lock:
                connection = self.pool.getconn()
                self.stats["active_connections"] += 1
                self.stats["peak_connections"] = max(
                    self.stats["peak_connections"],
                    self.stats["active_connections"]
                )

                # Track client usage
                if client_name not in self.stats["clients"]:
                    self.stats["clients"][client_name] = {"queries": 0, "last_used": datetime.now()}

            yield connection

        except Exception as e:
            self.stats["errors"] += 1
            print(f"[CONNECTION_POOL] ERROR in {client_name}: {e}")
            raise

        finally:
            if connection:
                with self.lock:
                    self.pool.putconn(connection)
                    self.stats["active_connections"] -= 1

    def execute_query(self, query: str, client_name: str = "unknown", params: Optional[Tuple] = None) -> List[Tuple]:
        """
        Execute a query through the connection pool.

        Args:
            query: SQL query to execute
            client_name: Name of the component executing the query
            params: Query parameters (optional)

        Returns:
            List[Tuple]: Query results
        """
        with self.get_connection(client_name) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)

            # Update statistics
            with self.lock:
                self.stats["total_queries"] += 1
                self.stats["clients"][client_name]["queries"] += 1
                self.stats["clients"][client_name]["last_used"] = datetime.now()

            # Return results for SELECT queries
            if query.strip().upper().startswith('SELECT'):
                return cursor.fetchall()
            else:
                conn.commit()
                return []

    def get_connection_string(self, client_name: str = "external_tool") -> str:
        """
        Get connection string for external tools (like MLflow).

        Note: External tools will create their own connections, but this ensures
        they use the same credentials and database as the shared pool.

        Args:
            client_name: Name of the external tool

        Returns:
            str: PostgreSQL connection string
        """
        with self.lock:
            self.stats["connection_string_requests"] += 1
            if client_name not in self.stats["clients"]:
                self.stats["clients"][client_name] = {
                    "queries": 0,
                    "last_used": datetime.now(),
                    "type": "external_connection_string"
                }

        # Build actual PostgreSQL connection string from config
        try:
            # Use the postgresql_primary credentials for external tools like MLflow
            host = self.pg_config.get('host', 'localhost')
            port = self.pg_config.get('port', 5432)
            database = self.pg_config.get('database', 'postgres')
            username = self.pg_config.get('username', 'postgres')
            password = self.pg_config.get('password', '')

            # Build PostgreSQL connection string
            connection_string = f"postgresql://{username}:{password}@{host}:{port}/{database}"

            print(f"[CONNECTION_POOL] Built connection string for {client_name}: {connection_string[:50]}...")
            return connection_string

        except Exception as e:
            print(f"[CONNECTION_POOL] Error building connection string: {e}")
            # Fallback to managed delegate if building fails
            return "managed-by-infrastructure-delegate"

    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        with self.lock:
            stats_copy = self.stats.copy()
            stats_copy["pool_config"] = {
                "min_connections": self.min_connections,
                "max_connections": self.max_connections,
                "current_active": self.stats["active_connections"]
            }
            return stats_copy

    def getconn(self):
        """
        Get a raw connection from the pool (psycopg2 interface).

        Returns:
            Raw database connection
        """
        return self.pool.getconn()

    def putconn(self, conn):
        """
        Return a connection to the pool (psycopg2 interface).

        Args:
            conn: Connection to return
        """
        self.pool.putconn(conn)

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the connection pool."""
        try:
            result = self.execute_query("SELECT 1", "health_check")
            return {
                "status": "healthy",
                "test_query": "passed",
                "pool_active": True,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "pool_active": False,
                "timestamp": datetime.now().isoformat()
            }

    def close_pool(self):
        """Close the connection pool."""
        if hasattr(self, 'real_pool') and self.real_pool:
            self.real_pool.closeall()
            print("[CONNECTION_POOL] Pool closed")


# Global connection pool instance
_global_pool: Optional[TidyLLMConnectionPool] = None
_pool_lock = threading.Lock()


def initialize_global_pool(pg_config: Optional[Dict[str, Any]] = None) -> TidyLLMConnectionPool:
    """
    Initialize the global connection pool.

    Args:
        pg_config: PostgreSQL configuration. If None, loads from settings.yaml

    Returns:
        TidyLLMConnectionPool: The global pool instance
    """
    global _global_pool

    with _pool_lock:
        if _global_pool is not None:
            return _global_pool

        if pg_config is None:
            pg_config = load_postgresql_config()

        _global_pool = TidyLLMConnectionPool(pg_config)
        print("[CONNECTION_POOL] Global pool initialized")
        return _global_pool


def get_global_pool() -> TidyLLMConnectionPool:
    """
    Get the global connection pool instance.

    Returns:
        TidyLLMConnectionPool: The global pool instance

    Raises:
        RuntimeError: If pool not initialized
    """
    global _global_pool

    if _global_pool is None:
        # Auto-initialize if not done
        return initialize_global_pool()

    return _global_pool


def load_postgresql_config() -> Dict[str, Any]:
    """
    Load PostgreSQL configuration from settings.yaml using SettingsLoader.

    Returns:
        Dict[str, Any]: PostgreSQL configuration
    """
    # Use SettingsLoader to get configuration - it already knows the root path!
    try:
        # Import SettingsLoader from infrastructure
        import sys
        from pathlib import Path

        # Get to qa-shipping root from SettingsLoader's perspective
        # Go up from packages/tidyllm/infrastructure to root
        qa_root = Path(__file__).parent.parent.parent.parent.resolve()
        if str(qa_root) not in sys.path:
            sys.path.insert(0, str(qa_root))

        from infrastructure.yaml_loader import SettingsLoader

        # Create loader and get database config
        loader = SettingsLoader()
        db_config = loader.get_database_config()

        # Convert to the format expected by connection pool
        pg_config = {
            'host': db_config['host'],
            'port': db_config['port'],
            'database': db_config['database'],
            'username': db_config['username'],
            'password': db_config['password']
        }

        print(f"[CONNECTION_POOL] Loaded PostgreSQL config via SettingsLoader from {loader.settings_path}")
        return pg_config

    except ImportError as e:
        print(f"[CONNECTION_POOL] Failed to import SettingsLoader: {e}")
        # Fallback to direct YAML loading if SettingsLoader not available
        settings_path = Path(__file__).parent.parent.parent.parent / "infrastructure" / "settings.yaml"
        if settings_path.exists():
            with open(settings_path, 'r') as f:
                config = yaml.safe_load(f)

            pg_config = config.get('credentials', {}).get('postgresql_primary', {})
            if not pg_config:
                pg_config = config.get('credentials', {}).get('postgresql', {})

            if not pg_config:
                raise ValueError(f"No PostgreSQL configuration found in {settings_path}")

            print(f"[CONNECTION_POOL] Loaded PostgreSQL config directly from {settings_path}")
            return pg_config

    raise FileNotFoundError("Could not load PostgreSQL configuration")


# Convenience functions for common operations
def execute_query(query: str, client_name: str = "direct_call", params: Optional[Tuple] = None) -> List[Tuple]:
    """Execute query using global pool."""
    pool = get_global_pool()
    return pool.execute_query(query, client_name, params)


def get_mlflow_connection_string() -> str:
    """Get connection string for MLflow."""
    pool = get_global_pool()
    return pool.get_connection_string("mlflow")


def get_pool_stats() -> Dict[str, Any]:
    """Get global pool statistics."""
    pool = get_global_pool()
    return pool.get_stats()


def pool_health_check() -> Dict[str, Any]:
    """Perform health check on global pool."""
    pool = get_global_pool()
    return pool.health_check()


if __name__ == "__main__":
    # Test the connection pool
    print("Testing TidyLLM Connection Pool")
    print("=" * 50)

    try:
        # Initialize pool
        pool = initialize_global_pool()

        # Test direct query
        print("\n1. Testing direct query execution...")
        result = pool.execute_query("SELECT version()", "test_client")
        print(f"PostgreSQL version query successful")

        # Test connection string for MLflow
        print("\n2. Testing MLflow connection string...")
        mlflow_uri = pool.get_connection_string("mlflow")
        print(f"MLflow URI generated: {mlflow_uri[:50]}...")

        # Test health check
        print("\n3. Testing health check...")
        health = pool.health_check()
        print(f"Health status: {health['status']}")

        # Show statistics
        print("\n4. Pool Statistics:")
        stats = pool.get_stats()
        print(f"Total queries: {stats['total_queries']}")
        print(f"Active clients: {len(stats['clients'])}")
        print(f"Peak connections: {stats['peak_connections']}")

        for client, info in stats['clients'].items():
            print(f"  {client}: {info['queries']} queries")

        print("\n[SUCCESS] Connection pool test completed successfully!")

    except Exception as e:
        print(f"\n[ERROR] Connection pool test failed: {e}")

    finally:
        if _global_pool:
            _global_pool.close_pool()