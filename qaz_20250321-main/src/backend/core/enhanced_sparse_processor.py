#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced SPARSE Command Processor
Processes SPARSE commands with live/remote DataMart integration
"""

import os
import yaml
import json
import time
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
import threading
from pathlib import Path

from .enhanced_datamart_manager import EnhancedDataMartManager, ConnectionConfig, ConnectionType


class SparseCommandType(Enum):
    """Types of SPARSE commands"""
    LOCAL = "local"
    LIVE = "live"
    HYBRID = "hybrid"
    STREAMING = "streaming"


@dataclass
class SparseCommandResult:
    """Result of SPARSE command processing"""
    success: bool
    command_type: SparseCommandType
    data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0
    data_freshness: Optional[float] = None
    source_used: Optional[str] = None
    fallback_used: bool = False
    performance_metrics: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)


class EnhancedSparseProcessor:
    """Enhanced SPARSE command processor with live DataMart support"""
    
    def __init__(self, agreements_path: str = "sparse/enhanced_sparse_agreements.yaml"):
        # Setup logging first
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        self.agreements_path = Path(agreements_path)
        self.agreements = self._load_agreements()
        self.datamart_manager = EnhancedDataMartManager()
        self.processing_cache = {}
        self.performance_history = []
        
        # Initialize connections from agreements
        self._initialize_connections()
    
    def _load_agreements(self) -> Dict[str, Any]:
        """Load enhanced SPARSE agreements"""
        try:
            if self.agreements_path.exists():
                with open(self.agreements_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                self.logger.warning(f"Agreements file not found: {self.agreements_path}")
                return {}
        except Exception as e:
            self.logger.error(f"Error loading agreements: {e}")
            return {}
    
    def _initialize_connections(self):
        """Initialize connections from agreements configuration"""
        try:
            connection_config = self.agreements.get('connection_management', {})
            
            # Initialize S3 connections
            for s3_config in connection_config.get('s3_connections', []):
                self.datamart_manager.connect_live_source({
                    'type': 's3',
                    'name': s3_config['name'],
                    'bucket': s3_config['bucket'],
                    'region': s3_config['region']
                })
            
            # Initialize database connections
            for db_config in connection_config.get('database_connections', []):
                if db_config['type'] == 'postgresql':
                    self.datamart_manager.connect_live_source({
                        'type': 'postgresql',
                        'name': db_config['name'],
                        'host': db_config['host'],
                        'port': db_config['port'],
                        'database': db_config['database'],
                        'username': os.getenv(f"{db_config['name'].upper()}_USERNAME"),
                        'password': os.getenv(f"{db_config['name'].upper()}_PASSWORD")
                    })
                elif db_config['type'] == 'redis':
                    self.datamart_manager.connect_live_source({
                        'type': 'redis',
                        'name': db_config['name'],
                        'host': db_config['host'],
                        'port': db_config['port']
                    })
            
            # Initialize streaming connections
            for stream_config in connection_config.get('streaming_connections', []):
                self.datamart_manager.connect_live_source({
                    'type': 'kinesis',
                    'name': stream_config['name'],
                    'stream_name': stream_config['stream_name'],
                    'region': stream_config['region']
                })
                
        except Exception as e:
            self.logger.error(f"Error initializing connections: {e}")
    
    def process_sparse_command(self, command: str, context: Optional[Dict[str, Any]] = None) -> SparseCommandResult:
        """Process a SPARSE command with enhanced capabilities"""
        start_time = time.time()
        
        try:
            # Extract command from brackets
            if command.startswith('[') and command.endswith(']'):
                command_text = command[1:-1]
            else:
                command_text = command
            
            # Lookup command in agreements
            agreement = self._find_agreement(command_text)
            if not agreement:
                return SparseCommandResult(
                    success=False,
                    command_type=SparseCommandType.LOCAL,
                    error_message=f"Command not found in agreements: {command_text}"
                )
            
            # Determine command type
            command_type = self._determine_command_type(agreement)
            
            # Process based on type
            if command_type == SparseCommandType.LIVE:
                result = self._process_live_command(agreement, context)
            elif command_type == SparseCommandType.STREAMING:
                result = self._process_streaming_command(agreement, context)
            elif command_type == SparseCommandType.HYBRID:
                result = self._process_hybrid_command(agreement, context)
            else:
                result = self._process_local_command(agreement, context)
            
            # Update processing time
            result.processing_time = (time.time() - start_time) * 1000
            
            # Cache result
            self._cache_result(command_text, result)
            
            # Update performance history
            self._update_performance_history(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing SPARSE command: {e}")
            return SparseCommandResult(
                success=False,
                command_type=SparseCommandType.LOCAL,
                error_message=str(e),
                processing_time=(time.time() - start_time) * 1000
            )
    
    def _find_agreement(self, command_text: str) -> Optional[Dict[str, Any]]:
        """Find agreement for a command"""
        try:
            # Search in enhanced agreements
            enhanced_agreements = self.agreements.get('enhanced_agreements', {})
            
            for category, commands in enhanced_agreements.items():
                if command_text in commands:
                    return commands[command_text]
            
            # Search in legacy agreements (backward compatibility)
            legacy_agreements = self.agreements.get('agreements', {})
            
            for category, commands in legacy_agreements.items():
                if command_text in commands:
                    return commands[command_text]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding agreement: {e}")
            return None
    
    def _determine_command_type(self, agreement: Dict[str, Any]) -> SparseCommandType:
        """Determine the type of SPARSE command"""
        try:
            # Check if it has live data sources
            if 'data_sources' in agreement:
                data_sources = agreement['data_sources']
                
                # Check for streaming sources
                if any(source.get('type') == 'datamart_stream' for source in data_sources.values()):
                    return SparseCommandType.STREAMING
                
                # Check for live sources
                if any(source.get('type') in ['datamart_live', 's3', 'postgresql'] for source in data_sources.values()):
                    return SparseCommandType.LIVE
                
                # Check for hybrid (both local and live)
                if len(data_sources) > 1:
                    return SparseCommandType.HYBRID
            
            return SparseCommandType.LOCAL
            
        except Exception as e:
            self.logger.error(f"Error determining command type: {e}")
            return SparseCommandType.LOCAL
    
    def _process_live_command(self, agreement: Dict[str, Any], context: Optional[Dict[str, Any]]) -> SparseCommandResult:
        """Process a live SPARSE command"""
        try:
            data_sources = agreement.get('data_sources', {})
            parameters = agreement.get('parameters', [])
            
            # Get live data from primary source
            primary_source = data_sources.get('primary', {})
            if primary_source:
                query_params = self._build_query_params(parameters, context)
                live_data = self.datamart_manager.get_live_data({
                    'source': 'primary',
                    **query_params
                })
                
                if live_data and 'error' not in live_data:
                    return SparseCommandResult(
                        success=True,
                        command_type=SparseCommandType.LIVE,
                        data=live_data,
                        source_used='primary',
                        data_freshness=self._calculate_freshness(live_data)
                    )
            
            # Try fallback sources
            for source_name, source_config in data_sources.items():
                if source_name == 'primary':
                    continue
                
                fallback_data = self.datamart_manager.get_live_data({
                    'source': source_name,
                    'fallback': 'cached'
                })
                
                if fallback_data and 'error' not in fallback_data:
                    return SparseCommandResult(
                        success=True,
                        command_type=SparseCommandType.LIVE,
                        data=fallback_data,
                        source_used=source_name,
                        fallback_used=True,
                        data_freshness=self._calculate_freshness(fallback_data)
                    )
            
            return SparseCommandResult(
                success=False,
                command_type=SparseCommandType.LIVE,
                error_message="No live data available from any source"
            )
            
        except Exception as e:
            self.logger.error(f"Error processing live command: {e}")
            return SparseCommandResult(
                success=False,
                command_type=SparseCommandType.LIVE,
                error_message=str(e)
            )
    
    def _process_streaming_command(self, agreement: Dict[str, Any], context: Optional[Dict[str, Any]]) -> SparseCommandResult:
        """Process a streaming SPARSE command"""
        try:
            data_sources = agreement.get('data_sources', {})
            parameters = agreement.get('parameters', [])
            
            # Get streaming data
            streaming_source = data_sources.get('streaming', {})
            if streaming_source:
                # For now, simulate streaming data
                # In production, this would connect to Kinesis or similar
                streaming_data = {
                    'stream_type': 'kinesis',
                    'window_size': '5m',
                    'metrics': {
                        'throughput': 1500,
                        'latency': 250,
                        'errors': 0.02,
                        'quality_score': 0.95
                    },
                    'timestamp': datetime.now().isoformat()
                }
                
                return SparseCommandResult(
                    success=True,
                    command_type=SparseCommandType.STREAMING,
                    data=streaming_data,
                    source_used='streaming',
                    data_freshness=0.0  # Real-time
                )
            
            return SparseCommandResult(
                success=False,
                command_type=SparseCommandType.STREAMING,
                error_message="No streaming source configured"
            )
            
        except Exception as e:
            self.logger.error(f"Error processing streaming command: {e}")
            return SparseCommandResult(
                success=False,
                command_type=SparseCommandType.STREAMING,
                error_message=str(e)
            )
    
    def _process_hybrid_command(self, agreement: Dict[str, Any], context: Optional[Dict[str, Any]]) -> SparseCommandResult:
        """Process a hybrid SPARSE command (both local and live)"""
        try:
            # Try live first
            live_result = self._process_live_command(agreement, context)
            if live_result.success:
                return live_result
            
            # Fall back to local processing
            local_result = self._process_local_command(agreement, context)
            local_result.fallback_used = True
            return local_result
            
        except Exception as e:
            self.logger.error(f"Error processing hybrid command: {e}")
            return SparseCommandResult(
                success=False,
                command_type=SparseCommandType.HYBRID,
                error_message=str(e)
            )
    
    def _process_local_command(self, agreement: Dict[str, Any], context: Optional[Dict[str, Any]]) -> SparseCommandResult:
        """Process a local SPARSE command (legacy behavior)"""
        try:
            # This would implement the original local processing logic
            # For now, return a mock result
            local_data = {
                'source': 'local',
                'agreement': agreement.get('sparse_encoding'),
                'action': agreement.get('action'),
                'timestamp': datetime.now().isoformat()
            }
            
            return SparseCommandResult(
                success=True,
                command_type=SparseCommandType.LOCAL,
                data=local_data,
                source_used='local'
            )
            
        except Exception as e:
            self.logger.error(f"Error processing local command: {e}")
            return SparseCommandResult(
                success=False,
                command_type=SparseCommandType.LOCAL,
                error_message=str(e)
            )
    
    def _build_query_params(self, parameters: List[str], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Build query parameters from agreement parameters and context"""
        query_params = {}
        
        try:
            for param in parameters:
                if ':' in param:
                    key, value = param.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Handle environment variable substitution
                    if value.startswith('${') and value.endswith('}'):
                        env_var = value[2:-1]
                        value = os.getenv(env_var, value)
                    
                    query_params[key] = value
            
            # Add context parameters
            if context:
                query_params.update(context)
            
            return query_params
            
        except Exception as e:
            self.logger.error(f"Error building query parameters: {e}")
            return {}
    
    def _calculate_freshness(self, data: Dict[str, Any]) -> Optional[float]:
        """Calculate data freshness in seconds"""
        try:
            if 'timestamp' in data:
                timestamp_str = data['timestamp']
                if isinstance(timestamp_str, str):
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    return (datetime.now() - timestamp).total_seconds()
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error calculating freshness: {e}")
            return None
    
    def _cache_result(self, command: str, result: SparseCommandResult):
        """Cache command result"""
        try:
            cache_key = f"{command}_{result.command_type.value}"
            self.processing_cache[cache_key] = {
                'result': result,
                'timestamp': datetime.now()
            }
            
            # Keep only last 100 cached results
            if len(self.processing_cache) > 100:
                oldest_key = min(self.processing_cache.keys(), 
                               key=lambda k: self.processing_cache[k]['timestamp'])
                del self.processing_cache[oldest_key]
                
        except Exception as e:
            self.logger.error(f"Error caching result: {e}")
    
    def _update_performance_history(self, result: SparseCommandResult):
        """Update performance history"""
        try:
            performance_record = {
                'command_type': result.command_type.value,
                'processing_time': result.processing_time,
                'success': result.success,
                'timestamp': result.timestamp,
                'source_used': result.source_used,
                'fallback_used': result.fallback_used
            }
            
            self.performance_history.append(performance_record)
            
            # Keep only last 1000 performance records
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
                
        except Exception as e:
            self.logger.error(f"Error updating performance history: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all commands"""
        try:
            if not self.performance_history:
                return {'message': 'No performance data available'}
            
            # Calculate metrics by command type
            metrics = {}
            for command_type in SparseCommandType:
                type_records = [r for r in self.performance_history 
                              if r['command_type'] == command_type.value]
                
                if type_records:
                    processing_times = [r['processing_time'] for r in type_records]
                    success_rate = sum(1 for r in type_records if r['success']) / len(type_records)
                    
                    metrics[command_type.value] = {
                        'total_commands': len(type_records),
                        'success_rate': success_rate,
                        'avg_processing_time': sum(processing_times) / len(processing_times),
                        'max_processing_time': max(processing_times),
                        'min_processing_time': min(processing_times)
                    }
            
            # Add DataMart metrics
            datamart_metrics = self.datamart_manager.get_performance_metrics()
            metrics['datamart'] = datamart_metrics
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {e}")
            return {'error': str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the SPARSE processor"""
        try:
            health_status = {
                'processor_status': 'healthy',
                'agreements_loaded': len(self.agreements.get('enhanced_agreements', {})) > 0,
                'datamart_health': self.datamart_manager.health_check(),
                'cache_status': {
                    'cache_size': len(self.processing_cache),
                    'performance_history_size': len(self.performance_history)
                }
            }
            
            # Check for issues
            if not health_status['agreements_loaded']:
                health_status['processor_status'] = 'degraded'
                health_status['alerts'] = ['No agreements loaded']
            
            datamart_health = health_status['datamart_health']
            if datamart_health['overall_status'] != 'healthy':
                health_status['processor_status'] = 'degraded'
                if 'alerts' not in health_status:
                    health_status['alerts'] = []
                health_status['alerts'].extend(datamart_health.get('alerts', []))
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Error performing health check: {e}")
            return {
                'processor_status': 'unhealthy',
                'error': str(e)
            }
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self.datamart_manager.cleanup()
            self.logger.info("Enhanced SPARSE processor cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


# Convenience function for creating enhanced SPARSE processor
def create_enhanced_sparse_processor(agreements_path: Optional[str] = None) -> EnhancedSparseProcessor:
    """Create an enhanced SPARSE processor with default configuration"""
    if agreements_path is None:
        agreements_path = "sparse/enhanced_sparse_agreements.yaml"
    
    return EnhancedSparseProcessor(agreements_path)
