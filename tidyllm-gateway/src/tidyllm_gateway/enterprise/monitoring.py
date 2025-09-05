#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enterprise Monitoring and Health Checking

Comprehensive monitoring system for gateway health, performance metrics,
and alerting for corporate IT operations.
"""

import logging
import time
import threading
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque


class HealthStatus(Enum):
    """Gateway health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    MAINTENANCE = "maintenance"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Health check configuration"""
    name: str
    check_function: Callable[[], Dict[str, Any]]
    interval_seconds: int = 60
    timeout_seconds: int = 10
    failure_threshold: int = 3
    recovery_threshold: int = 2
    enabled: bool = True


@dataclass
class MetricDefinition:
    """Metric definition and configuration"""
    name: str
    description: str
    metric_type: str  # counter, gauge, histogram, timer
    unit: str = ""
    labels: List[str] = field(default_factory=list)
    alert_thresholds: Dict[str, float] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert configuration and state"""
    id: str
    name: str
    severity: AlertSeverity
    condition: Callable[[Dict[str, Any]], bool]
    message_template: str
    cooldown_minutes: int = 15
    enabled: bool = True
    last_triggered: Optional[datetime] = None


class MetricsCollector:
    """
    Enterprise metrics collector for gateway monitoring
    
    Features:
    - Real-time performance metrics
    - Request/response tracking
    - Cost and usage analytics
    - Custom metric definitions
    - Time-series data storage
    - Alert threshold monitoring
    """
    
    def __init__(self, metrics_config: Dict[str, Any] = None):
        self.metrics_config = metrics_config or {}
        
        # Metric storage (in production, this would be a time-series database)
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Metric definitions
        self.metric_definitions: Dict[str, MetricDefinition] = {}
        
        # Time-series data (last 24 hours)
        self.time_series_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1440))  # 1 per minute
        
        # Collection thread
        self.collection_thread = None
        self.collection_running = False
        
        self.logger = logging.getLogger(f"{__name__}.MetricsCollector")
        
        # Initialize default metrics
        self._initialize_default_metrics()
        
        self.logger.info("📊 Metrics Collector initialized")
    
    def _initialize_default_metrics(self):
        """Initialize default gateway metrics"""
        
        default_metrics = [
            MetricDefinition(
                "gateway_requests_total",
                "Total number of gateway requests",
                "counter",
                "requests",
                ["gateway_type", "endpoint", "status"],
                {"rate_threshold": 1000}  # requests per minute
            ),
            MetricDefinition(
                "gateway_request_duration_seconds",
                "Gateway request duration",
                "histogram",
                "seconds",
                ["gateway_type", "endpoint"],
                {"p99_threshold": 5.0, "p95_threshold": 2.0}
            ),
            MetricDefinition(
                "gateway_active_connections",
                "Number of active gateway connections",
                "gauge",
                "connections",
                ["gateway_type"],
                {"max_threshold": 100}
            ),
            MetricDefinition(
                "gateway_cost_usd",
                "Gateway usage cost",
                "counter",
                "usd",
                ["gateway_type", "tenant_id"],
                {"daily_threshold": 1000.0}
            ),
            MetricDefinition(
                "gateway_tokens_consumed",
                "Total tokens consumed",
                "counter",
                "tokens",
                ["provider", "model"],
                {"hourly_threshold": 100000}
            ),
            MetricDefinition(
                "gateway_errors_total",
                "Total number of gateway errors",
                "counter",
                "errors",
                ["gateway_type", "error_type"],
                {"rate_threshold": 10}  # errors per minute
            ),
            MetricDefinition(
                "gateway_health_score",
                "Gateway health score",
                "gauge",
                "score",
                ["gateway_type"],
                {"min_threshold": 0.8}  # 80% minimum health
            )
        ]
        
        for metric_def in default_metrics:
            self.metric_definitions[metric_def.name] = metric_def
    
    def start_collection(self):
        """Start metrics collection thread"""
        if not self.collection_running:
            self.collection_running = True
            self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
            self.collection_thread.start()
            self.logger.info("📈 Metrics collection started")
    
    def stop_collection(self):
        """Stop metrics collection thread"""
        self.collection_running = False
        if self.collection_thread:
            self.collection_thread.join()
        self.logger.info("📉 Metrics collection stopped")
    
    def _collection_loop(self):
        """Main metrics collection loop"""
        while self.collection_running:
            try:
                # Collect time-series snapshots every minute
                timestamp = datetime.utcnow()
                
                for metric_name in self.metric_definitions:
                    current_value = self._get_current_metric_value(metric_name)
                    self.time_series_data[metric_name].append({
                        "timestamp": timestamp,
                        "value": current_value
                    })
                
                # Sleep until next minute boundary
                time.sleep(60 - datetime.utcnow().second)
                
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
                time.sleep(60)
    
    def increment_counter(self, name: str, value: int = 1, labels: Dict[str, str] = None):
        """Increment a counter metric"""
        key = self._build_metric_key(name, labels)
        self.counters[key] += value
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric value"""
        key = self._build_metric_key(name, labels)
        self.gauges[key] = value
    
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a value in a histogram"""
        key = self._build_metric_key(name, labels)
        self.histograms[key].append(value)
        
        # Keep only last 1000 values
        if len(self.histograms[key]) > 1000:
            self.histograms[key] = self.histograms[key][-1000:]
    
    def record_timer(self, name: str, duration_ms: float, labels: Dict[str, str] = None):
        """Record a timing measurement"""
        key = self._build_metric_key(name, labels)
        self.timers[key].append({
            "timestamp": datetime.utcnow(),
            "duration_ms": duration_ms
        })
    
    def _build_metric_key(self, name: str, labels: Dict[str, str] = None) -> str:
        """Build metric key with labels"""
        if not labels:
            return name
        
        label_str = ",".join([f"{k}={v}" for k, v in sorted(labels.items())])
        return f"{name}{{{label_str}}}"
    
    def _get_current_metric_value(self, metric_name: str) -> float:
        """Get current value for a metric"""
        
        metric_def = self.metric_definitions.get(metric_name)
        if not metric_def:
            return 0.0
        
        if metric_def.metric_type == "counter":
            # Sum all counters for this metric
            total = 0
            for key, value in self.counters.items():
                if key.startswith(metric_name):
                    total += value
            return total
        
        elif metric_def.metric_type == "gauge":
            # Get latest gauge value
            for key, value in self.gauges.items():
                if key.startswith(metric_name):
                    return value
            return 0.0
        
        elif metric_def.metric_type == "histogram":
            # Calculate average of histogram values
            values = []
            for key, hist_values in self.histograms.items():
                if key.startswith(metric_name):
                    values.extend(hist_values)
            
            return sum(values) / len(values) if values else 0.0
        
        return 0.0
    
    def get_metric_summary(self, metric_name: str) -> Dict[str, Any]:
        """Get comprehensive metric summary"""
        
        metric_def = self.metric_definitions.get(metric_name)
        if not metric_def:
            return {"error": "Metric not found"}
        
        summary = {
            "name": metric_name,
            "description": metric_def.description,
            "type": metric_def.metric_type,
            "unit": metric_def.unit,
            "current_value": self._get_current_metric_value(metric_name),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if metric_def.metric_type == "histogram":
            # Calculate percentiles for histogram
            all_values = []
            for key, values in self.histograms.items():
                if key.startswith(metric_name):
                    all_values.extend(values)
            
            if all_values:
                sorted_values = sorted(all_values)
                count = len(sorted_values)
                
                summary.update({
                    "count": count,
                    "min": min(sorted_values),
                    "max": max(sorted_values),
                    "p50": sorted_values[int(count * 0.5)],
                    "p90": sorted_values[int(count * 0.9)],
                    "p95": sorted_values[int(count * 0.95)],
                    "p99": sorted_values[int(count * 0.99)]
                })
        
        elif metric_def.metric_type == "timer":
            # Calculate timing statistics
            recent_timings = []
            cutoff_time = datetime.utcnow() - timedelta(hours=1)
            
            for key, timings in self.timers.items():
                if key.startswith(metric_name):
                    recent_timings.extend([
                        t["duration_ms"] for t in timings 
                        if t["timestamp"] > cutoff_time
                    ])
            
            if recent_timings:
                summary.update({
                    "count_last_hour": len(recent_timings),
                    "avg_duration_ms": sum(recent_timings) / len(recent_timings),
                    "min_duration_ms": min(recent_timings),
                    "max_duration_ms": max(recent_timings)
                })
        
        return summary
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        
        metrics = {}
        for metric_name in self.metric_definitions:
            metrics[metric_name] = self.get_metric_summary(metric_name)
        
        return {
            "metrics": metrics,
            "collection_active": self.collection_running,
            "timestamp": datetime.utcnow().isoformat()
        }


class HealthChecker:
    """
    Enterprise health checker for gateway monitoring
    
    Features:
    - Configurable health checks
    - Service dependency monitoring
    - Automatic recovery detection
    - Health status aggregation
    - Alert integration
    - Dashboard integration
    """
    
    def __init__(self, health_checks: List[HealthCheck] = None):
        self.health_checks = health_checks or []
        
        # Health check results
        self.check_results: Dict[str, Dict[str, Any]] = {}
        self.check_failures: Dict[str, int] = defaultdict(int)
        self.check_successes: Dict[str, int] = defaultdict(int)
        
        # Overall health status
        self.overall_status = HealthStatus.HEALTHY
        self.status_history: deque = deque(maxlen=100)
        
        # Health check thread
        self.health_thread = None
        self.health_running = False
        
        self.logger = logging.getLogger(f"{__name__}.HealthChecker")
        
        # Initialize default health checks
        self._initialize_default_checks()
        
        self.logger.info("🩺 Health Checker initialized")
    
    def _initialize_default_checks(self):
        """Initialize default health checks"""
        
        default_checks = [
            HealthCheck(
                "gateway_response_time",
                self._check_response_time,
                interval_seconds=30,
                failure_threshold=3
            ),
            HealthCheck(
                "database_connectivity",
                self._check_database_connectivity,
                interval_seconds=60,
                failure_threshold=2
            ),
            HealthCheck(
                "external_dependencies",
                self._check_external_dependencies,
                interval_seconds=120,
                failure_threshold=1
            ),
            HealthCheck(
                "memory_usage",
                self._check_memory_usage,
                interval_seconds=30,
                failure_threshold=3
            ),
            HealthCheck(
                "error_rate",
                self._check_error_rate,
                interval_seconds=60,
                failure_threshold=2
            )
        ]
        
        self.health_checks.extend(default_checks)
    
    def start_monitoring(self):
        """Start health monitoring thread"""
        if not self.health_running:
            self.health_running = True
            self.health_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.health_thread.start()
            self.logger.info("🚀 Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring thread"""
        self.health_running = False
        if self.health_thread:
            self.health_thread.join()
        self.logger.info("🛑 Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main health monitoring loop"""
        while self.health_running:
            try:
                # Run all enabled health checks
                for check in self.health_checks:
                    if check.enabled:
                        self._run_health_check(check)
                
                # Update overall health status
                self._update_overall_status()
                
                # Wait before next round of checks
                time.sleep(10)  # Check every 10 seconds, individual checks have their own intervals
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                time.sleep(30)
    
    def _run_health_check(self, check: HealthCheck):
        """Run a single health check"""
        
        # Check if it's time to run this check
        last_run = self.check_results.get(check.name, {}).get("last_run")
        if last_run:
            time_since_last = (datetime.utcnow() - last_run).total_seconds()
            if time_since_last < check.interval_seconds:
                return
        
        start_time = time.time()
        
        try:
            # Execute health check with timeout
            result = self._run_with_timeout(check.check_function, check.timeout_seconds)
            
            execution_time = (time.time() - start_time) * 1000  # ms
            
            # Process result
            if result.get("healthy", False):
                self.check_successes[check.name] += 1
                self.check_failures[check.name] = 0  # Reset failure count
                status = HealthStatus.HEALTHY
            else:
                self.check_failures[check.name] += 1
                status = HealthStatus.UNHEALTHY
            
            # Store result
            self.check_results[check.name] = {
                "status": status,
                "healthy": result.get("healthy", False),
                "message": result.get("message", ""),
                "details": result.get("details", {}),
                "execution_time_ms": execution_time,
                "last_run": datetime.utcnow(),
                "failure_count": self.check_failures[check.name],
                "success_count": self.check_successes[check.name]
            }
            
        except Exception as e:
            self.check_failures[check.name] += 1
            
            self.check_results[check.name] = {
                "status": HealthStatus.CRITICAL,
                "healthy": False,
                "message": f"Health check failed: {e}",
                "details": {},
                "execution_time_ms": (time.time() - start_time) * 1000,
                "last_run": datetime.utcnow(),
                "failure_count": self.check_failures[check.name],
                "success_count": self.check_successes[check.name]
            }
    
    def _run_with_timeout(self, func: Callable, timeout_seconds: int) -> Dict[str, Any]:
        """Run function with timeout"""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Health check timeout")
        
        # Set timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        
        try:
            result = func()
            signal.alarm(0)  # Cancel timeout
            return result
        except TimeoutError:
            return {"healthy": False, "message": "Health check timeout"}
        finally:
            signal.signal(signal.SIGALRM, old_handler)
    
    def _update_overall_status(self):
        """Update overall health status based on check results"""
        
        if not self.check_results:
            self.overall_status = HealthStatus.HEALTHY
            return
        
        # Count status levels
        status_counts = defaultdict(int)
        for result in self.check_results.values():
            status_counts[result["status"]] += 1
        
        # Determine overall status
        if status_counts[HealthStatus.CRITICAL] > 0:
            new_status = HealthStatus.CRITICAL
        elif status_counts[HealthStatus.UNHEALTHY] > 0:
            new_status = HealthStatus.UNHEALTHY
        elif status_counts[HealthStatus.DEGRADED] > 0:
            new_status = HealthStatus.DEGRADED
        else:
            new_status = HealthStatus.HEALTHY
        
        # Update status history
        if new_status != self.overall_status:
            self.status_history.append({
                "from_status": self.overall_status.value,
                "to_status": new_status.value,
                "timestamp": datetime.utcnow(),
                "reason": "Health check status change"
            })
            
            self.logger.info(f"Health status changed: {self.overall_status.value} → {new_status.value}")
        
        self.overall_status = new_status
    
    # Default health check implementations
    def _check_response_time(self) -> Dict[str, Any]:
        """Check gateway response time"""
        # This would test actual gateway endpoints
        return {"healthy": True, "message": "Response time within limits"}
    
    def _check_database_connectivity(self) -> Dict[str, Any]:
        """Check database connectivity"""
        # This would test database connections
        return {"healthy": True, "message": "Database connections healthy"}
    
    def _check_external_dependencies(self) -> Dict[str, Any]:
        """Check external service dependencies"""
        # This would test external service availability
        return {"healthy": True, "message": "External dependencies available"}
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check system memory usage"""
        import psutil
        
        memory_usage = psutil.virtual_memory().percent
        
        if memory_usage > 90:
            return {"healthy": False, "message": f"High memory usage: {memory_usage}%"}
        elif memory_usage > 80:
            return {"healthy": True, "message": f"Elevated memory usage: {memory_usage}%", "details": {"warning": True}}
        else:
            return {"healthy": True, "message": f"Memory usage normal: {memory_usage}%"}
    
    def _check_error_rate(self) -> Dict[str, Any]:
        """Check error rate"""
        # This would check actual error metrics
        return {"healthy": True, "message": "Error rate within acceptable limits"}
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary"""
        
        return {
            "overall_status": self.overall_status.value,
            "individual_checks": self.check_results,
            "summary": {
                "total_checks": len(self.health_checks),
                "passing_checks": sum(1 for r in self.check_results.values() if r.get("healthy")),
                "failing_checks": sum(1 for r in self.check_results.values() if not r.get("healthy"))
            },
            "monitoring_active": self.health_running,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def add_health_check(self, check: HealthCheck):
        """Add a new health check"""
        self.health_checks.append(check)
        self.logger.info(f"Added health check: {check.name}")
    
    def remove_health_check(self, check_name: str):
        """Remove a health check"""
        self.health_checks = [c for c in self.health_checks if c.name != check_name]
        self.check_results.pop(check_name, None)
        self.check_failures.pop(check_name, None)
        self.check_successes.pop(check_name, None)
        self.logger.info(f"Removed health check: {check_name}")