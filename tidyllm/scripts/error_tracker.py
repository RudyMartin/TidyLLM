"""
Intelligent Error Tracking & Alerting System - Standalone Version

Provides smart filtering and intelligent alerting for prompt-based pipeline failures.
Focuses on business-impacting issues rather than noise.

This is a standalone version that can be shipped and vetted independently.
"""

import time
import json
import uuid
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict
import logging
import threading
from enum import Enum

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

class AlertType(Enum):
    """Alert delivery types"""
    EMAIL = "email"
    SLACK = "slack"
    SMS = "sms"
    DASHBOARD = "dashboard"

@dataclass
class ErrorRecord:
    """Represents an error record with full context"""
    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    error_type: str
    error_message: str
    prompt_id: Optional[str] = None
    agent_name: Optional[str] = None
    task_type: Optional[str] = None
    model_used: Optional[str] = None
    context_data: Optional[Dict[str, Any]] = None
    stack_trace: Optional[str] = None
    cost_usd: Optional[float] = None
    response_time_ms: Optional[int] = None
    user_facing: bool = False
    alert_sent: bool = False
    alert_recipients: Optional[List[str]] = None

class AlertManager:
    """Manages alert delivery across different channels"""
    
    def __init__(self):
        self.alert_handlers = {
            AlertType.EMAIL: self._send_email_alert,
            AlertType.SLACK: self._send_slack_alert,
            AlertType.SMS: self._send_sms_alert,
            AlertType.DASHBOARD: self._send_dashboard_alert
        }
    
    def send_alert(self, alert_type: AlertType, recipients: List[str], 
                  message: str, error_id: str, severity: ErrorSeverity = ErrorSeverity.WARNING):
        """Send alert through specified channel"""
        try:
            handler = self.alert_handlers.get(alert_type)
            if handler:
                handler(recipients, message, error_id, severity)
                logger.info(f"Alert sent via {alert_type.value} for error {error_id}")
            else:
                logger.warning(f"Unknown alert type: {alert_type}")
        except Exception as e:
            logger.error(f"Failed to send alert via {alert_type.value}: {e}")
    
    def _send_email_alert(self, recipients: List[str], message: str, error_id: str, severity: ErrorSeverity):
        """Send email alert (placeholder implementation)"""
        logger.info(f"EMAIL ALERT [{severity.value.upper()}] to {recipients}: {message[:100]}...")
        # Implementation would integrate with email service
    
    def _send_slack_alert(self, recipients: List[str], message: str, error_id: str, severity: ErrorSeverity):
        """Send Slack alert (placeholder implementation)"""
        logger.info(f"SLACK ALERT [{severity.value.upper()}] to {recipients}: {message[:100]}...")
        # Implementation would integrate with Slack API
    
    def _send_sms_alert(self, recipients: List[str], message: str, error_id: str, severity: ErrorSeverity):
        """Send SMS alert (placeholder implementation)"""
        logger.info(f"SMS ALERT [{severity.value.upper()}] to {recipients}: {message[:100]}...")
        # Implementation would integrate with SMS service
    
    def _send_dashboard_alert(self, recipients: List[str], message: str, error_id: str, severity: ErrorSeverity):
        """Send dashboard alert (placeholder implementation)"""
        logger.info(f"DASHBOARD ALERT [{severity.value.upper()}] to {recipients}: {message[:100]}...")
        # Implementation would update real-time dashboard

class PromptPipelineErrorTracker:
    """Intelligent error tracking for prompt-based pipelines"""
    
    # Critical errors that require immediate attention
    CRITICAL_ERRORS = {
        'prompt_timeout': {
            'threshold': 30,  # seconds
            'impact': 'User experience blocked',
            'action': 'Immediate investigation',
            'alert_recipients': ['oncall@company.com', 'devops@company.com']
        },
        'llm_api_failure': {
            'threshold': 3,   # consecutive failures
            'impact': 'Core functionality broken',
            'action': 'Switch to backup model',
            'alert_recipients': ['oncall@company.com', 'ai-team@company.com']
        },
        'cost_exceeded': {
            'threshold': 100, # USD per hour
            'impact': 'Budget overrun',
            'action': 'Stop processing, alert admin',
            'alert_recipients': ['admin@company.com', 'finance@company.com']
        },
        'security_violation': {
            'threshold': 1,   # any occurrence
            'impact': 'Security breach',
            'action': 'Immediate lockdown',
            'alert_recipients': ['security@company.com', 'admin@company.com']
        },
        'dspy_wrapper_failure': {
            'threshold': 5,   # consecutive failures
            'impact': 'DSPy operations failing',
            'action': 'Check DSPy configuration and dependencies',
            'alert_recipients': ['ai-team@company.com', 'devops@company.com']
        }
    }
    
    # Warning errors that should be monitored
    WARNING_ERRORS = {
        'high_latency': {
            'threshold': 5,   # seconds
            'impact': 'Poor user experience',
            'action': 'Performance investigation',
            'alert_recipients': ['ai-team@company.com']
        },
        'high_error_rate': {
            'threshold': 10,  # % of requests
            'impact': 'System reliability',
            'action': 'Investigate error patterns',
            'alert_recipients': ['ai-team@company.com']
        },
        'cache_miss': {
            'threshold': 50,  # % of requests
            'impact': 'Increased costs and latency',
            'action': 'Optimize caching strategy',
            'alert_recipients': ['ai-team@company.com']
        },
        'polars_performance_degradation': {
            'threshold': 3,   # consecutive slow operations
            'impact': 'Data processing slowdown',
            'action': 'Check Polars configuration and data size',
            'alert_recipients': ['ai-team@company.com']
        }
    }
    
    def __init__(self, db_connection=None):
        # Use the new connection manager
        from .connection_manager import get_connection_manager
        self.connection_manager = get_connection_manager()
        self.alert_manager = AlertManager()
        self.error_patterns = self._load_error_patterns()
        self.error_counts = defaultdict(int)
        self.last_reset = datetime.now()
        self.error_history: List[ErrorRecord] = []
        self.lock = threading.Lock()
        
    def track_error(self, error_data: Dict[str, Any]):
        """Track an error with intelligent filtering and alerting"""
        
        with self.lock:
            # Determine severity based on error type and context
            severity = self._determine_severity(error_data)
            
            # Create error record
            error_id = str(uuid.uuid4())
            error_record = ErrorRecord(
                error_id=error_id,
                timestamp=datetime.now(),
                severity=severity,
                error_type=error_data['error_type'],
                error_message=error_data['error_message'],
                prompt_id=error_data.get('prompt_id'),
                agent_name=error_data.get('agent_name'),
                task_type=error_data.get('task_type'),
                model_used=error_data.get('model_used'),
                context_data=error_data.get('context', {}),
                stack_trace=error_data.get('stack_trace'),
                cost_usd=error_data.get('cost_usd'),
                response_time_ms=error_data.get('response_time_ms'),
                user_facing=error_data.get('user_facing', False)
            )
            
            # Store error in memory (and database if available)
            self._store_error(error_record)
            
            # Check for patterns and thresholds
            if self._should_alert(error_data, severity):
                self._send_alert(error_record)
            
            # Update error counts for pattern detection
            self._update_error_counts(error_data['error_type'])
    
    def _determine_severity(self, error_data: Dict[str, Any]) -> ErrorSeverity:
        """Intelligently determine error severity"""
        
        error_type = error_data['error_type']
        context = error_data.get('context', {})
        
        # Check critical conditions
        if error_type in self.CRITICAL_ERRORS:
            return ErrorSeverity.CRITICAL
        
        # Check warning conditions
        if error_type in self.WARNING_ERRORS:
            return ErrorSeverity.WARNING
        
        # Check for business impact
        if self._has_business_impact(error_data):
            return ErrorSeverity.WARNING
        
        # Default to info
        return ErrorSeverity.INFO
    
    def _has_business_impact(self, error_data: Dict[str, Any]) -> bool:
        """Check if error has business impact"""
        
        # High cost impact
        if error_data.get('cost_usd', 0) > 50:
            return True
        
        # User-facing error
        if error_data.get('user_facing', False):
            return True
        
        # Security-related
        if 'security' in error_data.get('error_message', '').lower():
            return True
        
        # Performance impact
        if error_data.get('response_time_ms', 0) > 10000:
            return True
        
        # DSPy-specific impact
        if 'dspy' in error_data.get('error_type', '').lower():
            return True
        
        return False
    
    def _should_alert(self, error_data: Dict[str, Any], severity: ErrorSeverity) -> bool:
        """Determine if alert should be sent"""
        
        # Always alert on critical errors
        if severity == ErrorSeverity.CRITICAL:
            return True
        
        # Check frequency thresholds
        error_type = error_data['error_type']
        if self._exceeds_frequency_threshold(error_type):
            return True
        
        # Check pattern matches
        if self._matches_alert_pattern(error_data):
            return True
        
        return False
    
    def _exceeds_frequency_threshold(self, error_type: str) -> bool:
        """Check if error frequency exceeds threshold"""
        
        # Reset counts if needed (every hour)
        if (datetime.now() - self.last_reset).seconds > 3600:
            self.error_counts.clear()
            self.last_reset = datetime.now()
        
        # Get threshold for this error type
        threshold = self.WARNING_ERRORS.get(error_type, {}).get('threshold', 5)
        
        # Check if exceeded
        return self.error_counts[error_type] >= threshold
    
    def _matches_alert_pattern(self, error_data: Dict[str, Any]) -> bool:
        """Check if error matches alert patterns"""
        
        for pattern in self.error_patterns:
            if self._matches_pattern(error_data, pattern):
                return True
        
        return False
    
    def _matches_pattern(self, error_data: Dict[str, Any], pattern: Dict[str, Any]) -> bool:
        """Check if error matches a specific pattern"""
        
        # Check error type match
        if pattern.get('error_type') and error_data['error_type'] != pattern['error_type']:
            return False
        
        # Check message content match
        if pattern.get('message_contains'):
            if pattern['message_contains'] not in error_data['error_message']:
                return False
        
        # Check context match
        if pattern.get('context_conditions'):
            context = error_data.get('context', {})
            for condition in pattern['context_conditions']:
                if not self._evaluate_condition(context, condition):
                    return False
        
        return True
    
    def _evaluate_condition(self, context: Dict[str, Any], condition: Dict[str, Any]) -> bool:
        """Evaluate a context condition"""
        
        field = condition.get('field')
        operator = condition.get('operator', 'equals')
        value = condition.get('value')
        
        if field not in context:
            return False
        
        context_value = context[field]
        
        if operator == 'equals':
            return context_value == value
        elif operator == 'contains':
            return value in str(context_value)
        elif operator == 'greater_than':
            return context_value > value
        elif operator == 'less_than':
            return context_value < value
        
        return False
    
    def _send_alert(self, error_record: ErrorRecord):
        """Send intelligent alert"""
        
        # Determine alert recipients based on severity
        recipients = self._get_alert_recipients(error_record.severity, error_record.error_type)
        
        # Create alert message
        message = self._create_alert_message(error_record)
        
        # Send alert
        self.alert_manager.send_alert(
            alert_type=AlertType.EMAIL,  # Default to email
            recipients=recipients,
            message=message,
            error_id=error_record.error_id,
            severity=error_record.severity
        )
        
        # Update error record
        error_record.alert_sent = True
        error_record.alert_recipients = recipients
    
    def _get_alert_recipients(self, severity: ErrorSeverity, error_type: str) -> List[str]:
        """Get alert recipients based on severity and error type"""
        
        if severity == ErrorSeverity.CRITICAL:
            # Get specific recipients for critical errors
            error_config = self.CRITICAL_ERRORS.get(error_type, {})
            return error_config.get('alert_recipients', ['oncall@company.com'])
        elif severity == ErrorSeverity.WARNING:
            # Get specific recipients for warning errors
            error_config = self.WARNING_ERRORS.get(error_type, {})
            return error_config.get('alert_recipients', ['ai-team@company.com'])
        else:
            # Info level - no alerts
            return []
    
    def _create_alert_message(self, error_record: ErrorRecord) -> str:
        """Create intelligent alert message"""
        
        template = """
🚨 DSPY WRAPPER ALERT: {severity}

Error Type: {error_type}
Time: {timestamp}
Agent: {agent_name}
Task: {task_type}

Impact: {impact}
Recommended Action: {action}

Error Details: {error_message}

Context: {context_summary}

View Details: {dashboard_url}
        """
        
        # Get impact and action from error configuration
        error_config = self.CRITICAL_ERRORS.get(error_record.error_type) or \
                      self.WARNING_ERRORS.get(error_record.error_type) or \
                      {'impact': 'Unknown', 'action': 'Investigate'}
        
        return template.format(
            severity=error_record.severity.value.upper(),
            error_type=error_record.error_type,
            timestamp=error_record.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            agent_name=error_record.agent_name or 'Unknown',
            task_type=error_record.task_type or 'Unknown',
            impact=error_config['impact'],
            action=error_config['action'],
            error_message=error_record.error_message[:200] + '...' if len(error_record.error_message) > 200 else error_record.error_message,
            context_summary=self._summarize_context(error_record.context_data),
            dashboard_url=f"https://dashboard.example.com/errors/{error_record.error_id}"
        )
    
    def _summarize_context(self, context_data: Optional[Dict[str, Any]]) -> str:
        """Summarize context data for alert message"""
        if not context_data:
            return "No context data"
        
        summary_parts = []
        for key, value in context_data.items():
            if isinstance(value, (str, int, float)):
                summary_parts.append(f"{key}: {value}")
            elif isinstance(value, dict):
                summary_parts.append(f"{key}: {len(value)} items")
        
        return "; ".join(summary_parts[:5])  # Limit to 5 items
    
    def _store_error(self, error_record: ErrorRecord):
        """Store error record"""
        self.error_history.append(error_record)
        
        # Keep only last 1000 errors in memory
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]
        
        # Store using connection manager (database or memory)
        try:
            error_data = {
                'error_id': error_record.error_id,
                'severity': error_record.severity.value,
                'error_type': error_record.error_type,
                'error_message': error_record.error_message,
                'agent_name': error_record.agent_name,
                'task_type': error_record.task_type,
                'model_used': error_record.model_used,
                'cost_usd': error_record.cost_usd,
                'response_time_ms': error_record.response_time_ms,
                'user_facing': error_record.user_facing,
                'context': error_record.context_data,
                'stack_trace': error_record.stack_trace,
                'alert_sent': False,
                'timestamp': error_record.timestamp.isoformat()
            }
            self.connection_manager.store_error(error_data)
        except Exception as e:
            logger.error(f"Failed to store error: {e}")
    
    def _store_in_database(self, error_record: ErrorRecord):
        """Store error record in database (deprecated - use connection manager)"""
        # This method is deprecated - use connection_manager.store_error() instead
        pass
    
    def _update_error_counts(self, error_type: str):
        """Update error counts for frequency tracking"""
        self.error_counts[error_type] += 1
    
    def _load_error_patterns(self) -> List[Dict[str, Any]]:
        """Load error patterns for intelligent matching"""
        return [
            {
                'name': 'dspy_timeout_pattern',
                'error_type': 'prompt_timeout',
                'message_contains': 'dspy',
                'context_conditions': [
                    {'field': 'response_time_ms', 'operator': 'greater_than', 'value': 30000}
                ]
            },
            {
                'name': 'polars_performance_pattern',
                'error_type': 'high_latency',
                'message_contains': 'polars',
                'context_conditions': [
                    {'field': 'data_size_mb', 'operator': 'greater_than', 'value': 100}
                ]
            }
        ]
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary for the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_errors = [
            error for error in self.error_history 
            if error.timestamp >= cutoff_time
        ]
        
        severity_counts = defaultdict(int)
        error_type_counts = defaultdict(int)
        
        for error in recent_errors:
            severity_counts[error.severity.value] += 1
            error_type_counts[error.error_type] += 1
        
        return {
            'total_errors': len(recent_errors),
            'severity_breakdown': dict(severity_counts),
            'error_type_breakdown': dict(error_type_counts),
            'time_period_hours': hours
        }
    
    def get_alert_history(self, hours: int = 24) -> List[ErrorRecord]:
        """Get history of errors that triggered alerts"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            error for error in self.error_history 
            if error.alert_sent and error.timestamp >= cutoff_time
        ]

# Convenience functions for DSPy wrapper integration
def create_error_tracker(db_connection=None) -> PromptPipelineErrorTracker:
    """Create an error tracker for DSPy wrapper operations"""
    # db_connection parameter is kept for backward compatibility but not used
    return PromptPipelineErrorTracker()

def track_dspy_error(error_type: str, error_message: str, **kwargs):
    """Track a DSPy-related error"""
    tracker = PromptPipelineErrorTracker()
    
    error_data = {
        'error_type': error_type,
        'error_message': error_message,
        'agent_name': kwargs.get('agent_name', 'dspy_wrapper'),
        'task_type': kwargs.get('task_type', 'dspy_operation'),
        'model_used': kwargs.get('model_used'),
        'prompt_id': kwargs.get('prompt_id'),
        'cost_usd': kwargs.get('cost_usd'),
        'response_time_ms': kwargs.get('response_time_ms'),
        'user_facing': kwargs.get('user_facing', False),
        'context': kwargs.get('context', {}),
        'stack_trace': kwargs.get('stack_trace')
    }
    
    tracker.track_error(error_data)

if __name__ == "__main__":
    # Test the error tracking system
    print("🎯 Testing Error Tracking System")
    
    tracker = create_error_tracker()
    
    # Test error tracking
    track_dspy_error(
        error_type='demo_test_error',
        error_message='This is a test error for demo purposes',
        agent_name='demo_team',
        task_type='testing',
        user_facing=False
    )
    
    # Get summary
    summary = tracker.get_error_summary(hours=1)
    print(f"Error summary: {summary}")
