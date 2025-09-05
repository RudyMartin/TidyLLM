#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DataMart Service - Standalone Data Management Service

This service provides DataMart functionality as a shared infrastructure service,
eliminating circular dependencies by removing ownership from AdvancedQAOrchestrator.

CRITICAL FIX: This resolves the circular import:
datamart_numpy_substitution.py ↔ advanced_qa_orchestrator.py

The DataMart is now a service that can be used by any coordinator or worker
without creating ownership conflicts or circular dependencies.
"""

import logging
import uuid
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class DataMartMode(Enum):
    """DataMart operational modes"""
    SIMPLE = "simple"
    ENHANCED = "enhanced"  
    ADVANCED = "advanced"


class DataMartManager:
    """Configurable DataMart Manager with progressive complexity using datatable for high-performance data processing"""
    
    def __init__(self, mode: DataMartMode = DataMartMode.SIMPLE):
        self.mode = mode
        self.datamart_id = str(uuid.uuid4())
        self.buffer = None  # Will be initialized as datatable Frame
        self.performance_metrics = {}
        self.analytics_cache = {}  # For enhanced/advanced analytics
        logger.info(f"DataMart Service initialized in {mode.value} mode with ID: {self.datamart_id}")
    
    def initialize_datamart(self):
        """Initialize datatable buffer based on mode"""
        try:
            if self.mode == DataMartMode.SIMPLE:
                return self._initialize_simple()
            elif self.mode == DataMartMode.ENHANCED:
                return self._initialize_enhanced()
            elif self.mode == DataMartMode.ADVANCED:
                return self._initialize_advanced()
            else:
                raise ValueError(f"Unsupported DataMart mode: {self.mode}")
        except Exception as e:
            logger.error(f"Error initializing DataMart Service: {e}")
            return False
    
    def _initialize_simple(self) -> bool:
        """Initialize simple DataMart with basic storage"""
        try:
            import datatable as dt
            self.buffer = dt.Frame()  # Empty datatable
            logger.info("✅ Simple DataMart buffer initialized with datatable")
            return True
        except ImportError:
            logger.warning("⚠️ datatable not available, using fallback")
            self.buffer = []
            return False
    
    def _initialize_enhanced(self) -> bool:
        """Initialize enhanced DataMart with analytics capabilities"""
        try:
            import datatable as dt
            # Enhanced buffer with predefined schema
            self.buffer = dt.Frame({
                'timestamp': [],
                'worker_name': [],
                'worker_type': [],
                'mode': [],
                'data_type': [],
                'confidence_score': [],
                'processing_time': [],
                'success_status': [],
                'metadata': []
            })
            logger.info("✅ Enhanced DataMart buffer initialized with analytics schema")
            return True
        except ImportError:
            logger.warning("⚠️ datatable not available for enhanced mode, using fallback")
            return self._initialize_simple()
    
    def _initialize_advanced(self) -> bool:
        """Initialize advanced DataMart with full analytics and caching"""
        try:
            import datatable as dt
            # Advanced buffer with comprehensive schema
            self.buffer = dt.Frame({
                'timestamp': [],
                'worker_name': [],
                'worker_type': [],
                'coordinator': [],
                'mode': [],
                'data_type': [],
                'confidence_score': [],
                'processing_time': [],
                'success_status': [],
                'error_code': [],
                'retry_count': [],
                'quality_score': [],
                'complexity_score': [],
                'analytics_version': [],
                'metadata': []
            })
            
            # Initialize analytics cache
            self.analytics_cache = {
                'performance_trends': {},
                'quality_patterns': {},
                'error_analytics': {},
                'worker_efficiency': {}
            }
            
            logger.info("✅ Advanced DataMart buffer initialized with full analytics")
            return True
        except ImportError:
            logger.warning("⚠️ datatable not available for advanced mode, using enhanced fallback")
            return self._initialize_enhanced()
    
    def add_analysis_data(self, data: Dict[str, Any]) -> bool:
        """Add analysis data to DataMart"""
        try:
            if self.buffer is None:
                logger.error("DataMart buffer not initialized")
                return False
            
            # Prepare data based on mode
            processed_data = self._prepare_data_for_storage(data)
            
            if isinstance(self.buffer, list):
                # Fallback mode (no datatable)
                self.buffer.append(processed_data)
            else:
                # datatable mode
                import datatable as dt
                new_row = dt.Frame(processed_data)
                self.buffer = dt.rbind([self.buffer, new_row])
            
            # Update performance metrics
            self._update_performance_metrics(processed_data)
            
            logger.debug(f"Added analysis data to DataMart: {data.get('data_type', 'unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding data to DataMart: {e}")
            return False
    
    def _prepare_data_for_storage(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for storage based on DataMart mode"""
        
        base_data = {
            'timestamp': [datetime.now().isoformat()],
            'worker_name': [data.get('worker_name', 'unknown')],
            'worker_type': [data.get('worker_type', 'unknown')],
            'mode': [self.mode.value],
            'data_type': [data.get('data_type', 'unknown')],
            'confidence_score': [data.get('confidence_score', 0.0)],
            'processing_time': [data.get('processing_time', 0.0)],
            'success_status': [data.get('success', True)],
            'metadata': [json.dumps(data.get('metadata', {}))]
        }
        
        if self.mode in [DataMartMode.ENHANCED, DataMartMode.ADVANCED]:
            # Add enhanced fields
            pass
        
        if self.mode == DataMartMode.ADVANCED:
            # Add advanced fields
            base_data.update({
                'coordinator': [data.get('coordinator', 'unknown')],
                'error_code': [data.get('error_code', None)],
                'retry_count': [data.get('retry_count', 0)],
                'quality_score': [data.get('quality_score', 0.0)],
                'complexity_score': [data.get('complexity_score', 0.0)],
                'analytics_version': [data.get('analytics_version', '1.0')]
            })
        
        return base_data
    
    def _update_performance_metrics(self, data: Dict[str, Any]):
        """Update performance metrics"""
        worker_name = data.get('worker_name', ['unknown'])[0]
        
        if worker_name not in self.performance_metrics:
            self.performance_metrics[worker_name] = {
                'total_operations': 0,
                'successful_operations': 0,
                'total_processing_time': 0.0,
                'average_confidence': 0.0,
                'confidence_scores': []
            }
        
        metrics = self.performance_metrics[worker_name]
        metrics['total_operations'] += 1
        
        if data.get('success_status', [True])[0]:
            metrics['successful_operations'] += 1
        
        processing_time = data.get('processing_time', [0.0])[0]
        metrics['total_processing_time'] += processing_time
        
        confidence = data.get('confidence_score', [0.0])[0]
        metrics['confidence_scores'].append(confidence)
        metrics['average_confidence'] = sum(metrics['confidence_scores']) / len(metrics['confidence_scores'])
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        
        if isinstance(self.buffer, list):
            total_entries = len(self.buffer)
        else:
            try:
                total_entries = self.buffer.nrows if self.buffer is not None else 0
            except:
                total_entries = 0
        
        overall_metrics = {
            'datamart_id': self.datamart_id,
            'mode': self.mode.value,
            'total_entries': total_entries,
            'worker_metrics': self.performance_metrics,
            'buffer_type': 'datatable' if not isinstance(self.buffer, list) else 'fallback',
            'analytics_cache_size': len(self.analytics_cache) if self.analytics_cache else 0
        }
        
        return overall_metrics
    
    def query_data(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Query data from DataMart"""
        try:
            if self.buffer is None:
                return []
            
            if isinstance(self.buffer, list):
                # Fallback mode - simple filtering
                if not filters:
                    return self.buffer
                
                filtered_data = []
                for entry in self.buffer:
                    matches = True
                    for key, value in filters.items():
                        if key in entry and entry[key] != value:
                            matches = False
                            break
                    if matches:
                        filtered_data.append(entry)
                return filtered_data
            
            else:
                # datatable mode - more sophisticated querying
                import datatable as dt
                
                if not filters:
                    return self.buffer.to_list()
                
                # Apply filters (basic implementation)
                filtered_frame = self.buffer
                for column, value in filters.items():
                    if column in self.buffer.names:
                        filtered_frame = filtered_frame[dt.f[column] == value, :]
                
                return filtered_frame.to_list()
                
        except Exception as e:
            logger.error(f"Error querying DataMart: {e}")
            return []
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get advanced analytics (for ADVANCED mode)"""
        if self.mode != DataMartMode.ADVANCED:
            return {"error": "Analytics only available in ADVANCED mode"}
        
        try:
            # Calculate analytics from current data
            analytics = {
                'performance_trends': self._calculate_performance_trends(),
                'quality_patterns': self._calculate_quality_patterns(),
                'error_analytics': self._calculate_error_analytics(),
                'worker_efficiency': self._calculate_worker_efficiency()
            }
            
            # Update cache
            self.analytics_cache.update(analytics)
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error calculating analytics: {e}")
            return {"error": str(e)}
    
    def _calculate_performance_trends(self) -> Dict[str, Any]:
        """Calculate performance trends"""
        # Placeholder implementation
        return {
            "trend": "stable",
            "average_processing_time": sum(
                metrics.get('total_processing_time', 0) / max(metrics.get('total_operations', 1), 1)
                for metrics in self.performance_metrics.values()
            ) / max(len(self.performance_metrics), 1) if self.performance_metrics else 0
        }
    
    def _calculate_quality_patterns(self) -> Dict[str, Any]:
        """Calculate quality patterns"""
        # Placeholder implementation
        return {
            "average_quality": 0.8,
            "quality_distribution": {"high": 0.6, "medium": 0.3, "low": 0.1}
        }
    
    def _calculate_error_analytics(self) -> Dict[str, Any]:
        """Calculate error analytics"""
        # Placeholder implementation
        total_operations = sum(metrics.get('total_operations', 0) for metrics in self.performance_metrics.values())
        total_successful = sum(metrics.get('successful_operations', 0) for metrics in self.performance_metrics.values())
        
        return {
            "error_rate": (total_operations - total_successful) / max(total_operations, 1) if total_operations else 0,
            "common_errors": []
        }
    
    def _calculate_worker_efficiency(self) -> Dict[str, Any]:
        """Calculate worker efficiency metrics"""
        efficiency_metrics = {}
        
        for worker_name, metrics in self.performance_metrics.items():
            total_ops = metrics.get('total_operations', 1)
            successful_ops = metrics.get('successful_operations', 0)
            avg_time = metrics.get('total_processing_time', 0) / max(total_ops, 1)
            avg_confidence = metrics.get('average_confidence', 0.0)
            
            efficiency_metrics[worker_name] = {
                'success_rate': successful_ops / total_ops,
                'average_processing_time': avg_time,
                'average_confidence': avg_confidence,
                'efficiency_score': (successful_ops / total_ops) * avg_confidence / max(avg_time, 0.1)
            }
        
        return efficiency_metrics
    
    def reset_datamart(self):
        """Reset DataMart (clear all data)"""
        logger.info("Resetting DataMart Service...")
        
        self.performance_metrics.clear()
        self.analytics_cache.clear()
        
        if isinstance(self.buffer, list):
            self.buffer.clear()
        else:
            try:
                import datatable as dt
                # Reinitialize empty buffer with same schema
                if self.mode == DataMartMode.SIMPLE:
                    self.buffer = dt.Frame()
                else:
                    self.initialize_datamart()
            except:
                self.buffer = []
        
        logger.info("DataMart Service reset completed")


# Singleton pattern for shared DataMart service
_datamart_instance: Optional[DataMartManager] = None

def get_datamart_service(mode: DataMartMode = DataMartMode.ENHANCED) -> DataMartManager:
    """Get singleton DataMart service instance"""
    global _datamart_instance
    
    if _datamart_instance is None:
        _datamart_instance = DataMartManager(mode)
        _datamart_instance.initialize_datamart()
    
    return _datamart_instance

def reset_datamart_service():
    """Reset the singleton DataMart service"""
    global _datamart_instance
    
    if _datamart_instance:
        _datamart_instance.reset_datamart()
        _datamart_instance = None