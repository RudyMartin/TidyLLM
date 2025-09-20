#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo Connection Manager

A lightweight connection manager for demo systems that can optionally connect to PostgreSQL
when available, but gracefully falls back to in-memory storage when not.
"""

import os
import json
import logging
import threading
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timedelta
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum

# Optional PostgreSQL support
try:
    # #future_fix: Convert to use enhanced service infrastructure
    import psycopg2
    from psycopg2.extras import RealDictCursor
    from psycopg2.pool import SimpleConnectionPool
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    print("⚠️ PostgreSQL not available - using in-memory storage only")

logger = logging.getLogger(__name__)


class ConnectionStatus(Enum):
    """Connection status enumeration"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class ConnectionConfig:
    """Database connection configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "demo_db"
    username: str = "demo_user"
    password: str = ""
    pool_size: int = 5
    timeout: int = 30
    ssl_mode: str = "prefer"


class DemoConnectionManager:
    """
    Lightweight connection manager for demo systems.
    
    Features:
    - Optional PostgreSQL connection with connection pooling
    - Graceful fallback to in-memory storage
    - Automatic connection management
    - Demo-specific data storage and retrieval
    """
    
    def __init__(self, config: Optional[ConnectionConfig] = None):
        self.config = config or self._load_config()
        self.connection_pool = None
        self.status = ConnectionStatus.DISCONNECTED
        self.in_memory_storage = {}
        self.lock = threading.Lock()
        
        # Initialize connection
        self._initialize_connection()
    
    def _load_config(self) -> ConnectionConfig:
        """Load configuration from environment or use defaults"""
        return ConnectionConfig(
    # #future_fix: Convert to use enhanced service infrastructure
            host=os.getenv('POSTGRES_HOST', os.getenv('DEMO_DB_HOST', 'localhost')),
    # #future_fix: Convert to use enhanced service infrastructure
            port=int(os.getenv('POSTGRES_PORT', os.getenv('DEMO_DB_PORT', '5432'))),
    # #future_fix: Convert to use enhanced service infrastructure
            database=os.getenv('POSTGRES_DB', os.getenv('DEMO_DB_NAME', 'demo_db')),
    # #future_fix: Convert to use enhanced service infrastructure
            username=os.getenv('POSTGRES_USER', os.getenv('DEMO_DB_USER', 'demo_user')),
    # #future_fix: Convert to use enhanced service infrastructure
            password=os.getenv('POSTGRES_PASSWORD', os.getenv('DEMO_DB_PASSWORD', '')),
    # #future_fix: Convert to use enhanced service infrastructure
            pool_size=int(os.getenv('POSTGRES_POOL_SIZE', os.getenv('DEMO_DB_POOL_SIZE', '5'))),
    # #future_fix: Convert to use enhanced service infrastructure
            timeout=int(os.getenv('POSTGRES_TIMEOUT', os.getenv('DEMO_DB_TIMEOUT', '30'))),
    # #future_fix: Convert to use enhanced service infrastructure
            ssl_mode=os.getenv('POSTGRES_SSL_MODE', os.getenv('DEMO_DB_SSL_MODE', 'require'))
        )
    
    def _initialize_connection(self):
        """Initialize database connection if available"""
        if not POSTGRES_AVAILABLE:
            logger.info("PostgreSQL not available - using in-memory storage")
            self.status = ConnectionStatus.DISCONNECTED
            return
        
        try:
            self.status = ConnectionStatus.CONNECTING
            
            # Build connection string
            connection_string = (
    # #future_fix: Convert to use enhanced service infrastructure
                f"postgresql://{self.config.username}:{self.config.password}"
                f"@{self.config.host}:{self.config.port}/{self.config.database}"
                f"?sslmode={self.config.ssl_mode}"
            )
            
            # Initialize connection pool
            self.connection_pool = SimpleConnectionPool(
                1, self.config.pool_size, connection_string
            )
            
            # Test connection
            if self.test_connection():
                self.status = ConnectionStatus.CONNECTED
                logger.info("✅ Connected to PostgreSQL database")
                
                # Initialize demo tables
                self._initialize_demo_tables()
            else:
                self.status = ConnectionStatus.ERROR
                logger.warning("Failed to connect to PostgreSQL - using in-memory storage")
                
        except Exception as e:
            self.status = ConnectionStatus.ERROR
            logger.warning(f"PostgreSQL connection failed: {e} - using in-memory storage")
    
    def _initialize_demo_tables(self):
        """Initialize demo-specific tables"""
        try:
            with self.get_cursor() as cursor:
                # Error tracking table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS demo_errors (
                        id SERIAL PRIMARY KEY,
                        error_id VARCHAR(255) UNIQUE NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        severity VARCHAR(50) NOT NULL,
                        error_type VARCHAR(100) NOT NULL,
                        error_message TEXT NOT NULL,
                        agent_name VARCHAR(100),
                        task_type VARCHAR(100),
                        model_used VARCHAR(100),
                        cost_usd DECIMAL(10,4),
                        response_time_ms INTEGER,
                        user_facing BOOLEAN DEFAULT FALSE,
                        context_data JSONB,
                        stack_trace TEXT,
                        alert_sent BOOLEAN DEFAULT FALSE
                    );
                """)
                
                # SPARSE command tracking table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS demo_sparse_commands (
                        id SERIAL PRIMARY KEY,
                        command_id VARCHAR(255) UNIQUE NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        command_type VARCHAR(100) NOT NULL,
                        command_text TEXT NOT NULL,
                        response_text TEXT,
                        processing_time_ms INTEGER,
                        success BOOLEAN DEFAULT TRUE,
                        error_message TEXT,
                        metadata JSONB
                    );
                """)
                
                # Protection system events table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS demo_protection_events (
                        id SERIAL PRIMARY KEY,
                        event_id VARCHAR(255) UNIQUE NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        input_text TEXT NOT NULL,
                        is_suspicious BOOLEAN NOT NULL,
                        action_taken VARCHAR(100) NOT NULL,
                        confidence_score DECIMAL(5,4),
                        risk_factors JSONB,
                        sanitized_output TEXT,
                        metadata JSONB
                    );
                """)
                
                logger.info("✅ Demo tables initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize demo tables: {e}")
    
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
        if self.status != ConnectionStatus.CONNECTED:
            raise RuntimeError("Database not connected")
        
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
    
    def test_connection(self) -> bool:
        """Test if the database connection is working"""
        if self.status != ConnectionStatus.CONNECTED:
            return False
        
        try:
            with self.get_cursor() as cursor:
                cursor.execute("SELECT 1;")
                result = cursor.fetchone()
                return result is not None
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def is_connected(self) -> bool:
        """Check if connected to database"""
        return self.status == ConnectionStatus.CONNECTED
    
    # In-memory storage methods (fallback)
    def store_in_memory(self, key: str, data: Any):
        """Store data in memory"""
        with self.lock:
            self.in_memory_storage[key] = {
                'data': data,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_from_memory(self, key: str) -> Optional[Any]:
        """Get data from memory"""
        with self.lock:
            if key in self.in_memory_storage:
                return self.in_memory_storage[key]['data']
            return None
    
    def list_memory_keys(self) -> List[str]:
        """List all keys in memory storage"""
        with self.lock:
            return list(self.in_memory_storage.keys())
    
    # Demo-specific storage methods
    def store_error(self, error_data: Dict[str, Any]) -> bool:
        """Store error data (database or memory)"""
        if self.is_connected():
            try:
                with self.get_cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO demo_errors (
                            error_id, severity, error_type, error_message, agent_name,
                            task_type, model_used, cost_usd, response_time_ms,
                            user_facing, context_data, stack_trace, alert_sent
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (error_id) DO UPDATE SET
                            timestamp = CURRENT_TIMESTAMP,
                            severity = EXCLUDED.severity,
                            error_message = EXCLUDED.error_message,
                            context_data = EXCLUDED.context_data
                    """, (
                        error_data.get('error_id'),
                        error_data.get('severity'),
                        error_data.get('error_type'),
                        error_data.get('error_message'),
                        error_data.get('agent_name'),
                        error_data.get('task_type'),
                        error_data.get('model_used'),
                        error_data.get('cost_usd'),
                        error_data.get('response_time_ms'),
                        error_data.get('user_facing', False),
                        json.dumps(error_data.get('context', {})),
                        error_data.get('stack_trace'),
                        error_data.get('alert_sent', False)
                    ))
                    return True
            except Exception as e:
                logger.error(f"Failed to store error in database: {e}")
                # Fall back to memory storage
        
        # Store in memory
        key = f"error_{error_data.get('error_id', datetime.now().isoformat())}"
        self.store_in_memory(key, error_data)
        return True
    
    def store_sparse_command(self, command_data: Dict[str, Any]) -> bool:
        """Store SPARSE command data (database or memory)"""
        if self.is_connected():
            try:
                with self.get_cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO demo_sparse_commands (
                            command_id, command_type, command_text, response_text,
                            processing_time_ms, success, error_message, metadata
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (command_id) DO UPDATE SET
                            timestamp = CURRENT_TIMESTAMP,
                            response_text = EXCLUDED.response_text,
                            processing_time_ms = EXCLUDED.processing_time_ms,
                            success = EXCLUDED.success
                    """, (
                        command_data.get('command_id'),
                        command_data.get('command_type'),
                        command_data.get('command_text'),
                        command_data.get('response_text'),
                        command_data.get('processing_time_ms'),
                        command_data.get('success', True),
                        command_data.get('error_message'),
                        json.dumps(command_data.get('metadata', {}))
                    ))
                    return True
            except Exception as e:
                logger.error(f"Failed to store SPARSE command in database: {e}")
                # Fall back to memory storage
        
        # Store in memory
        key = f"sparse_{command_data.get('command_id', datetime.now().isoformat())}"
        self.store_in_memory(key, command_data)
        return True
    
    def store_protection_event(self, event_data: Dict[str, Any]) -> bool:
        """Store protection event data (database or memory)"""
        if self.is_connected():
            try:
                with self.get_cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO demo_protection_events (
                            event_id, input_text, is_suspicious, action_taken,
                            confidence_score, risk_factors, sanitized_output, metadata
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (event_id) DO UPDATE SET
                            timestamp = CURRENT_TIMESTAMP,
                            action_taken = EXCLUDED.action_taken,
                            sanitized_output = EXCLUDED.sanitized_output
                    """, (
                        event_data.get('event_id'),
                        event_data.get('input_text'),
                        event_data.get('is_suspicious'),
                        event_data.get('action_taken'),
                        event_data.get('confidence_score'),
                        json.dumps(event_data.get('risk_factors', {})),
                        event_data.get('sanitized_output'),
                        json.dumps(event_data.get('metadata', {}))
                    ))
                    return True
            except Exception as e:
                logger.error(f"Failed to store protection event in database: {e}")
                # Fall back to memory storage
        
        # Store in memory
        key = f"protection_{event_data.get('event_id', datetime.now().isoformat())}"
        self.store_in_memory(key, event_data)
        return True
    
    def get_recent_errors(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent errors (database or memory)"""
        if self.is_connected():
            try:
                with self.get_cursor() as cursor:
                    cursor.execute("""
                        SELECT * FROM demo_errors 
                        WHERE timestamp >= NOW() - INTERVAL '%s hours'
                        ORDER BY timestamp DESC
                    """, (hours,))
                    return [dict(row) for row in cursor.fetchall()]
            except Exception as e:
                logger.error(f"Failed to get errors from database: {e}")
                # Fall back to memory storage
        
        # Get from memory
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_errors = []
        
        with self.lock:
            for key, value in self.in_memory_storage.items():
                if key.startswith('error_'):
                    timestamp = datetime.fromisoformat(value['timestamp'])
                    if timestamp >= cutoff_time:
                        recent_errors.append(value['data'])
        
        return sorted(recent_errors, key=lambda x: x.get('timestamp', ''), reverse=True)
    
    def get_recent_contract_commands(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent contract commands (database or memory)"""
        if self.connection and POSTGRES_AVAILABLE:
            try:
                with self.connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT * FROM contract_commands WHERE timestamp > NOW() - INTERVAL %s HOUR",
                        (hours,)
                    )
                    return [dict(zip([desc[0] for desc in cursor.description], row)) for row in cursor.fetchall()]
            except Exception as e:
                logger.error(f"Failed to get contract commands from database: {e}")
                return []
        
        # Fallback to memory storage
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_commands = []
        
        for key, value in self.memory_storage.items():
            if key.startswith('contract_cmd_'):
                try:
                    cmd_time = datetime.fromisoformat(value['timestamp'])
                    if cmd_time > cutoff_time:
                        recent_commands.append(value['data'])
                except (KeyError, ValueError):
                    continue
        
        return sorted(recent_commands, key=lambda x: x.get('timestamp', ''), reverse=True)
    
    def get_status(self) -> Dict[str, Any]:
        """Get connection manager status"""
        return {
            'status': self.status.value,
            'postgres_available': POSTGRES_AVAILABLE,
            'connected': self.is_connected(),
            'memory_storage_keys': len(self.list_memory_keys()),
            'config': {
                'host': self.config.host,
                'port': self.config.port,
                'database': self.config.database,
                'pool_size': self.config.pool_size
            }
        }
    
    def close(self):
        """Close the connection manager"""
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("Database connection pool closed")


# Global instance for easy access
_connection_manager = None


def get_connection_manager() -> DemoConnectionManager:
    """Get the global connection manager instance"""
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = DemoConnectionManager()
    return _connection_manager


if __name__ == "__main__":
    # Test the connection manager
    print("🔌 Testing Demo Connection Manager")
    
    manager = get_connection_manager()
    status = manager.get_status()
    
    print(f"Status: {status['status']}")
    print(f"PostgreSQL Available: {status['postgres_available']}")
    print(f"Connected: {status['connected']}")
    print(f"Memory Storage Keys: {status['memory_storage_keys']}")
    
    # Test storage
    test_error = {
        'error_id': 'test_123',
        'severity': 'INFO',
        'error_type': 'test_error',
        'error_message': 'This is a test error',
        'timestamp': datetime.now().isoformat()
    }
    
    success = manager.store_error(test_error)
    print(f"Error storage success: {success}")
    
    recent_errors = manager.get_recent_errors()
    print(f"Recent errors: {len(recent_errors)}")
    
    print("✅ Connection manager test completed")
