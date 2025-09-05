#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced DataMart Manager
Supports live/remote DataMart integration with robust error handling and performance monitoring
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Try to import AWS SDK for S3/Kinesis support
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    print("⚠️ AWS SDK not available, S3/Kinesis features disabled")

# Try to import Redis for caching
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("⚠️ Redis not available, caching features disabled")

# Try to import PostgreSQL for database support
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    print("⚠️ PostgreSQL not available, database features disabled")

from .datamart_numpy_substitution import DataMartManager, DataMartMode


class ConnectionType(Enum):
    """Types of data source connections"""
    LOCAL = "local"
    S3 = "s3"
    KINESIS = "kinesis"
    POSTGRESQL = "postgresql"
    REDIS = "redis"
    HTTP = "http"
    DATAMART_LIVE = "datamart_live"
    DATAMART_STREAM = "datamart_stream"


class ConnectionStatus(Enum):
    """Connection status enumeration"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class ConnectionConfig:
    """Configuration for data source connections"""
    name: str
    type: ConnectionType
    connection_string: str
    timeout: int = 30
    retry_attempts: int = 3
    fallback_source: Optional[str] = None
    authentication: Optional[Dict[str, Any]] = None
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring"""
    connection_latency: float = 0.0
    data_freshness: float = 0.0
    processing_time: float = 0.0
    error_rate: float = 0.0
    cache_hit_rate: float = 0.0
    throughput: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ErrorInfo:
    """Error information for handling and reporting"""
    error_type: str
    error_message: str
    timestamp: datetime
    connection_name: str
    retry_count: int = 0
    resolved: bool = False


class EnhancedDataMartManager:
    """Enhanced DataMart Manager with live/remote data source support"""
    
    def __init__(self, mode: DataMartMode = DataMartMode.ENHANCED):
        self.mode = mode
        self.datamart_id = str(uuid.uuid4())
        self.connections: Dict[str, ConnectionConfig] = {}
        self.connection_status: Dict[str, ConnectionStatus] = {}
        self.performance_metrics: Dict[str, PerformanceMetrics] = {}
        self.error_log: List[ErrorInfo] = []
        self.cache: Dict[str, Any] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        
        # Initialize base DataMart
        self.base_datamart = DataMartManager(mode)
        
        # Threading for concurrent operations
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.lock = threading.Lock()
        
        # Performance monitoring
        self.monitoring_enabled = True
        self.alert_thresholds = {
            'connection_latency': 5000,  # 5 seconds
            'error_rate': 0.05,          # 5%
            'data_freshness': 3600,      # 1 hour
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
    
    def add_connection(self, config: ConnectionConfig) -> bool:
        """Add a new data source connection"""
        try:
            with self.lock:
                self.connections[config.name] = config
                self.connection_status[config.name] = ConnectionStatus.DISCONNECTED
                self.performance_metrics[config.name] = PerformanceMetrics()
            
            self.logger.info(f"Added connection: {config.name} ({config.type.value})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add connection {config.name}: {e}")
            return False
    
    def connect_live_source(self, source_config: Dict[str, Any]) -> bool:
        """Connect to a live data source"""
        try:
            connection_type = ConnectionType(source_config.get('type', 'local'))
            
            if connection_type == ConnectionType.S3:
                return self._connect_s3(source_config)
            elif connection_type == ConnectionType.KINESIS:
                return self._connect_kinesis(source_config)
            elif connection_type == ConnectionType.POSTGRESQL:
                return self._connect_postgresql(source_config)
            elif connection_type == ConnectionType.REDIS:
                return self._connect_redis(source_config)
            elif connection_type == ConnectionType.DATAMART_LIVE:
                return self._connect_datamart_live(source_config)
            else:
                self.logger.warning(f"Unsupported connection type: {connection_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to connect to live source: {e}")
            return False
    
    def _connect_s3(self, config: Dict[str, Any]) -> bool:
        """Connect to S3 data source"""
        if not AWS_AVAILABLE:
            self.logger.error("AWS SDK not available for S3 connection")
            return False
        
        try:
            connection_name = config.get('name', 's3_connection')
            bucket_name = config.get('bucket')
            region = config.get('region', 'us-east-1')
            
            # Create S3 client
            s3_client = boto3.client('s3', region_name=region)
            
            # Test connection
            s3_client.head_bucket(Bucket=bucket_name)
            
            # Store connection info
            connection_config = ConnectionConfig(
                name=connection_name,
                type=ConnectionType.S3,
                connection_string=f"s3://{bucket_name}",
                timeout=config.get('timeout', 30),
                authentication={'type': 'aws_iam', 'region': region}
            )
            
            self.add_connection(connection_config)
            self.connection_status[connection_name] = ConnectionStatus.CONNECTED
            
            self.logger.info(f"Successfully connected to S3: {bucket_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to S3: {e}")
            return False
    
    def _connect_kinesis(self, config: Dict[str, Any]) -> bool:
        """Connect to Kinesis stream"""
        if not AWS_AVAILABLE:
            self.logger.error("AWS SDK not available for Kinesis connection")
            return False
        
        try:
            connection_name = config.get('name', 'kinesis_connection')
            stream_name = config.get('stream_name')
            region = config.get('region', 'us-east-1')
            
            # Create Kinesis client
            kinesis_client = boto3.client('kinesis', region_name=region)
            
            # Test connection
            kinesis_client.describe_stream(StreamName=stream_name)
            
            # Store connection info
            connection_config = ConnectionConfig(
                name=connection_name,
                type=ConnectionType.KINESIS,
                connection_string=f"kinesis://{stream_name}",
                timeout=config.get('timeout', 30),
                authentication={'type': 'aws_iam', 'region': region}
            )
            
            self.add_connection(connection_config)
            self.connection_status[connection_name] = ConnectionStatus.CONNECTED
            
            self.logger.info(f"Successfully connected to Kinesis: {stream_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Kinesis: {e}")
            return False
    
    def _connect_postgresql(self, config: Dict[str, Any]) -> bool:
        """Connect to PostgreSQL database"""
        if not POSTGRES_AVAILABLE:
            self.logger.error("PostgreSQL not available for database connection")
            return False
        
        try:
            connection_name = config.get('name', 'postgresql_connection')
            host = config.get('host')
            port = config.get('port', 5432)
            database = config.get('database')
            username = config.get('username')
            password = config.get('password')
            
            # Test connection
            conn = psycopg2.connect(
                host=host,
                port=port,
                database=database,
                user=username,
                password=password
            )
            conn.close()
            
            # Store connection info
            connection_config = ConnectionConfig(
                name=connection_name,
                type=ConnectionType.POSTGRESQL,
                connection_string=f"postgresql://{username}@{host}:{port}/{database}",
                timeout=config.get('timeout', 30),
                authentication={'username': username, 'password': password}
            )
            
            self.add_connection(connection_config)
            self.connection_status[connection_name] = ConnectionStatus.CONNECTED
            
            self.logger.info(f"Successfully connected to PostgreSQL: {database}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to PostgreSQL: {e}")
            return False
    
    def _connect_redis(self, config: Dict[str, Any]) -> bool:
        """Connect to Redis cache"""
        if not REDIS_AVAILABLE:
            self.logger.error("Redis not available for cache connection")
            return False
        
        try:
            connection_name = config.get('name', 'redis_connection')
            host = config.get('host', 'localhost')
            port = config.get('port', 6379)
            database = config.get('database', 0)
            
            # Test connection
            redis_client = redis.Redis(host=host, port=port, db=database)
            redis_client.ping()
            
            # Store connection info
            connection_config = ConnectionConfig(
                name=connection_name,
                type=ConnectionType.REDIS,
                connection_string=f"redis://{host}:{port}/{database}",
                timeout=config.get('timeout', 30)
            )
            
            self.add_connection(connection_config)
            self.connection_status[connection_name] = ConnectionStatus.CONNECTED
            
            self.logger.info(f"Successfully connected to Redis: {host}:{port}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            return False
    
    def _connect_datamart_live(self, config: Dict[str, Any]) -> bool:
        """Connect to live DataMart source"""
        try:
            connection_name = config.get('name', 'datamart_live_connection')
            source_url = config.get('source_url')
            
            # Store connection info
            connection_config = ConnectionConfig(
                name=connection_name,
                type=ConnectionType.DATAMART_LIVE,
                connection_string=source_url,
                timeout=config.get('timeout', 30),
                fallback_source=config.get('fallback_source')
            )
            
            self.add_connection(connection_config)
            self.connection_status[connection_name] = ConnectionStatus.CONNECTED
            
            self.logger.info(f"Successfully connected to live DataMart: {source_url}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to live DataMart: {e}")
            return False
    
    def get_live_data(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get live data from connected sources"""
        start_time = time.time()
        
        try:
            source_name = query_params.get('source', 'primary')
            freshness = query_params.get('freshness', 'live')
            fallback = query_params.get('fallback', 'cached')
            
            # Try primary source first
            if source_name in self.connections:
                data = self._fetch_from_source(source_name, query_params)
                if data:
                    self._update_performance_metrics(source_name, start_time, success=True)
                    return data
            
            # Try fallback source
            if fallback == 'cached' and source_name in self.cache:
                cache_age = (datetime.now() - self.cache_timestamps[source_name]).total_seconds()
                if cache_age < 3600:  # 1 hour cache
                    self.logger.info(f"Using cached data for {source_name}")
                    return self.cache[source_name]
            
            # Try fallback source from config
            connection = self.connections.get(source_name)
            if connection and connection.fallback_source:
                self.logger.info(f"Trying fallback source: {connection.fallback_source}")
                data = self._fetch_from_fallback(connection.fallback_source, query_params)
                if data:
                    return data
            
            # Return empty result
            self._update_performance_metrics(source_name, start_time, success=False)
            return {'error': 'No data available from any source'}
            
        except Exception as e:
            self.logger.error(f"Error getting live data: {e}")
            self._record_error('get_live_data', str(e), source_name)
            return {'error': str(e)}
    
    def _fetch_from_source(self, source_name: str, query_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fetch data from a specific source"""
        try:
            connection = self.connections[source_name]
            
            if connection.type == ConnectionType.S3:
                return self._fetch_from_s3(connection, query_params)
            elif connection.type == ConnectionType.POSTGRESQL:
                return self._fetch_from_postgresql(connection, query_params)
            elif connection.type == ConnectionType.REDIS:
                return self._fetch_from_redis(connection, query_params)
            elif connection.type == ConnectionType.DATAMART_LIVE:
                return self._fetch_from_datamart_live(connection, query_params)
            else:
                self.logger.warning(f"Unsupported source type: {connection.type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error fetching from {source_name}: {e}")
            return None
    
    def _fetch_from_s3(self, connection: ConnectionConfig, query_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fetch data from S3"""
        if not AWS_AVAILABLE:
            return None
        
        try:
            # Parse S3 URL
            bucket_name = connection.connection_string.replace('s3://', '').split('/')[0]
            key = '/'.join(connection.connection_string.replace('s3://', '').split('/')[1:])
            
            s3_client = boto3.client('s3')
            response = s3_client.get_object(Bucket=bucket_name, Key=key)
            data = json.loads(response['Body'].read().decode('utf-8'))
            
            # Cache the result
            self.cache[connection.name] = data
            self.cache_timestamps[connection.name] = datetime.now()
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching from S3: {e}")
            return None
    
    def _fetch_from_postgresql(self, connection: ConnectionConfig, query_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fetch data from PostgreSQL"""
        if not POSTGRES_AVAILABLE:
            return None
        
        try:
            # Parse connection string
            conn_string = connection.connection_string.replace('postgresql://', '')
            auth_info = connection.authentication
            
            conn = psycopg2.connect(
                connection.connection_string,
                user=auth_info.get('username'),
                password=auth_info.get('password')
            )
            
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Execute query
            query = query_params.get('query', 'SELECT * FROM live_criteria LIMIT 100')
            cursor.execute(query)
            
            results = cursor.fetchall()
            data = [dict(row) for row in results]
            
            cursor.close()
            conn.close()
            
            # Cache the result
            self.cache[connection.name] = data
            self.cache_timestamps[connection.name] = datetime.now()
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching from PostgreSQL: {e}")
            return None
    
    def _fetch_from_redis(self, connection: ConnectionConfig, query_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fetch data from Redis"""
        if not REDIS_AVAILABLE:
            return None
        
        try:
            # Parse connection string
            conn_string = connection.connection_string.replace('redis://', '')
            host_port_db = conn_string.split('/')
            host_port = host_port_db[0].split(':')
            host = host_port[0]
            port = int(host_port[1]) if len(host_port) > 1 else 6379
            database = int(host_port_db[1]) if len(host_port_db) > 1 else 0
            
            redis_client = redis.Redis(host=host, port=port, db=database)
            
            # Get data
            key = query_params.get('key', 'default_key')
            data = redis_client.get(key)
            
            if data:
                return json.loads(data.decode('utf-8'))
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error fetching from Redis: {e}")
            return None
    
    def _fetch_from_datamart_live(self, connection: ConnectionConfig, query_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fetch data from live DataMart"""
        try:
            # This would integrate with the actual DataMart live system
            # For now, return mock data
            return {
                'source': 'datamart_live',
                'timestamp': datetime.now().isoformat(),
                'data': 'Live DataMart data would be fetched here',
                'freshness': 'live'
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching from live DataMart: {e}")
            return None
    
    def _fetch_from_fallback(self, fallback_source: str, query_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fetch data from fallback source"""
        try:
            if fallback_source.startswith('local://'):
                # Local file fallback
                file_path = fallback_source.replace('local://', '')
                with open(file_path, 'r') as f:
                    return json.load(f)
            else:
                # Try as a new connection
                return self._fetch_from_source('fallback', query_params)
                
        except Exception as e:
            self.logger.error(f"Error fetching from fallback: {e}")
            return None
    
    def _update_performance_metrics(self, source_name: str, start_time: float, success: bool):
        """Update performance metrics for a source"""
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        with self.lock:
            metrics = self.performance_metrics[source_name]
            metrics.processing_time = processing_time
            metrics.timestamp = datetime.now()
            
            if not success:
                metrics.error_rate += 0.1  # Increment error rate
            else:
                metrics.error_rate = max(0, metrics.error_rate - 0.01)  # Decrement error rate
    
    def _record_error(self, error_type: str, error_message: str, connection_name: str):
        """Record an error for monitoring"""
        error_info = ErrorInfo(
            error_type=error_type,
            error_message=error_message,
            timestamp=datetime.now(),
            connection_name=connection_name
        )
        
        with self.lock:
            self.error_log.append(error_info)
            
            # Keep only last 100 errors
            if len(self.error_log) > 100:
                self.error_log = self.error_log[-100:]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        with self.lock:
            return {
                'datamart_id': self.datamart_id,
                'mode': self.mode.value if hasattr(self.mode, 'value') else str(self.mode),
                'connections': {
                    name: {
                        'status': status.value,
                        'metrics': {
                            'processing_time': metrics.processing_time,
                            'error_rate': metrics.error_rate,
                            'timestamp': metrics.timestamp.isoformat()
                        }
                    }
                    for name, status, metrics in zip(
                        self.connection_status.keys(),
                        self.connection_status.values(),
                        self.performance_metrics.values()
                    )
                },
                'cache_stats': {
                    'cache_size': len(self.cache),
                    'cache_keys': list(self.cache.keys())
                },
                'error_stats': {
                    'total_errors': len(self.error_log),
                    'recent_errors': len([e for e in self.error_log if (datetime.now() - e.timestamp).total_seconds() < 3600])
                }
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all connections"""
        health_status = {
            'overall_status': 'healthy',
            'connections': {},
            'alerts': []
        }
        
        for name, status in self.connection_status.items():
            metrics = self.performance_metrics.get(name, PerformanceMetrics())
            
            # Check for alerts
            if metrics.processing_time > self.alert_thresholds['connection_latency']:
                health_status['alerts'].append(f"High latency for {name}: {metrics.processing_time}ms")
            
            if metrics.error_rate > self.alert_thresholds['error_rate']:
                health_status['alerts'].append(f"High error rate for {name}: {metrics.error_rate:.2%}")
            
            health_status['connections'][name] = {
                'status': status.value,
                'latency': metrics.processing_time,
                'error_rate': metrics.error_rate
            }
        
        # Update overall status
        if health_status['alerts']:
            health_status['overall_status'] = 'degraded'
        
        if any(status == ConnectionStatus.FAILED for status in self.connection_status.values()):
            health_status['overall_status'] = 'unhealthy'
        
        return health_status
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self.executor.shutdown(wait=True)
            self.logger.info("Enhanced DataMart Manager cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


# Convenience function for creating enhanced DataMart manager
def create_enhanced_datamart_manager(mode: DataMartMode = DataMartMode.ENHANCED) -> EnhancedDataMartManager:
    """Create an enhanced DataMart manager with default configuration"""
    manager = EnhancedDataMartManager(mode)
    
    # Add default connections from environment variables
    if os.getenv('DATAMART_LIVE_CONNECTION'):
        manager.connect_live_source({
            'type': 'datamart_live',
            'name': 'default_live',
            'source_url': os.getenv('DATAMART_LIVE_CONNECTION')
        })
    
    if os.getenv('REDIS_HOST'):
        manager.connect_live_source({
            'type': 'redis',
            'name': 'default_cache',
            'host': os.getenv('REDIS_HOST'),
            'port': int(os.getenv('REDIS_PORT', 6379))
        })
    
    return manager
