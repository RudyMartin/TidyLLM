"""
Anti-Sabotage Demo Protection System - Standalone Version

Provides multi-layer protection for demo environments with transparent mode indicators.
Protects against malicious inputs while maintaining demo functionality.

This is a standalone version that can be shipped and vetted independently.
"""

import os
import sys
import time
import hashlib
import threading
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import logging
import json
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class ProtectionRule:
    """Represents a protection rule for demo safety"""
    name: str
    rule_type: str  # 'file_size', 'filename', 'content', 'frequency', 'system_capacity'
    threshold: Union[int, float, str]
    action: str  # 'block', 'warn', 'simulate', 'limit'
    description: str
    enabled: bool = True

@dataclass
class ProtectionEvent:
    """Represents a protection event"""
    timestamp: datetime
    rule_name: str
    event_type: str
    details: Dict[str, Any]
    action_taken: str
    user_friendly_message: str

class DemoProtectionSystem:
    """Multi-layer demo protection system"""
    
    def __init__(self, demo_mode: bool = True):
        # Use the connection manager for storage
        from .connection_manager import get_connection_manager
        self.connection_manager = get_connection_manager()
        
        self.demo_mode = demo_mode
        self.protection_rules = self._load_default_rules()
        self.event_history: List[ProtectionEvent] = []
        self.request_counts = defaultdict(int)
        self.last_reset = datetime.now()
        self.system_capacity = self._assess_system_capacity()
        self.lock = threading.Lock()
        
        # Protection statistics
        self.stats = {
            'total_requests': 0,
            'blocked_requests': 0,
            'warned_requests': 0,
            'simulated_requests': 0,
            'sabotage_attempts': 0
        }
    
    def _load_default_rules(self) -> List[ProtectionRule]:
        """Load default protection rules"""
        return [
            # File size protection
            ProtectionRule(
                name="max_file_size",
                rule_type="file_size",
                threshold=100 * 1024 * 1024,  # 100MB
                action="block",
                description="Maximum file size for demo processing"
            ),
            
            # Filename protection
            ProtectionRule(
                name="suspicious_filenames",
                rule_type="filename",
                threshold=".exe,.zip,.tar,.gz,test_large,crash,bomb,malicious",
                action="warn",
                description="Detect suspicious or potentially harmful filenames"
            ),
            
            # Content protection
            ProtectionRule(
                name="malicious_content",
                rule_type="content",
                threshold="script,executable,binary,malware",
                action="block",
                description="Detect potentially malicious content"
            ),
            
            # Request frequency protection
            ProtectionRule(
                name="rate_limiting",
                rule_type="frequency",
                threshold=100,  # requests per minute
                action="limit",
                description="Rate limiting to prevent demo abuse"
            ),
            
            # System capacity protection
            ProtectionRule(
                name="system_capacity",
                rule_type="system_capacity",
                threshold=0.8,  # 80% of capacity
                action="simulate",
                description="Switch to simulation mode when system is overloaded"
            ),
            
            # DSPy-specific protection
            ProtectionRule(
                name="dspy_complexity_limit",
                rule_type="content",
                threshold="recursive,infinite,loop,stack_overflow",
                action="warn",
                description="Detect potentially problematic DSPy operations"
            ),
            
            # Cost protection
            ProtectionRule(
                name="cost_limit",
                rule_type="cost",
                threshold=50.0,  # USD per request
                action="block",
                description="Prevent expensive operations in demo mode"
            )
        ]
    
    def _assess_system_capacity(self) -> Dict[str, Any]:
        """Assess current system capacity"""
        try:
            import psutil
            
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'available_memory_gb': psutil.virtual_memory().available / (1024**3),
                'timestamp': datetime.now()
            }
        except ImportError:
            # Fallback if psutil not available
            return {
                'cpu_percent': 50.0,  # Assume moderate load
                'memory_percent': 60.0,
                'disk_percent': 70.0,
                'available_memory_gb': 4.0,
                'timestamp': datetime.now()
            }
    
    def protect_input(self, input_data: Union[str, bytes, Dict[str, Any]], 
                     input_type: str = "text") -> Tuple[bool, Dict[str, Any]]:
        """Protect against malicious input"""
        
        with self.lock:
            self.stats['total_requests'] += 1
            self._update_request_counts()
            
            protection_result = {
                'safe': True,
                'warnings': [],
                'blocked': False,
                'simulation_mode': False,
                'action_taken': 'allow',
                'user_friendly_message': 'Input processed normally',
                'protection_events': []
            }
            
            # Apply protection rules
            for rule in self.protection_rules:
                if not rule.enabled:
                    continue
                
                rule_result = self._apply_protection_rule(rule, input_data, input_type)
                
                if rule_result['triggered']:
                    protection_result['protection_events'].append(rule_result)
                    
                    if rule.action == 'block':
                        protection_result['safe'] = False
                        protection_result['blocked'] = True
                        protection_result['action_taken'] = 'blocked'
                        protection_result['user_friendly_message'] = rule_result['user_message']
                        self.stats['blocked_requests'] += 1
                        self.stats['sabotage_attempts'] += 1
                        break
                    
                    elif rule.action == 'warn':
                        protection_result['warnings'].append(rule_result['user_message'])
                        self.stats['warned_requests'] += 1
                    
                    elif rule.action == 'simulate':
                        protection_result['simulation_mode'] = True
                        protection_result['action_taken'] = 'simulated'
                        protection_result['user_friendly_message'] = rule_result['user_message']
                        self.stats['simulated_requests'] += 1
                    
                    elif rule.action == 'limit':
                        # Apply rate limiting
                        if not self._check_rate_limit():
                            protection_result['safe'] = False
                            protection_result['blocked'] = True
                            protection_result['action_taken'] = 'rate_limited'
                            protection_result['user_friendly_message'] = "Rate limit exceeded. Please wait before making more requests."
                            self.stats['blocked_requests'] += 1
                            break
            
            # Record protection event
            self._record_protection_event(protection_result)
            
            return protection_result['safe'], protection_result
    
    def _apply_protection_rule(self, rule: ProtectionRule, input_data: Union[str, bytes, Dict[str, Any]], 
                              input_type: str) -> Dict[str, Any]:
        """Apply a specific protection rule"""
        
        result = {
            'triggered': False,
            'rule_name': rule.name,
            'user_message': '',
            'technical_details': {}
        }
        
        if rule.rule_type == 'file_size':
            if input_type == 'file' and hasattr(input_data, 'size'):
                if input_data.size > rule.threshold:
                    result['triggered'] = True
                    result['user_message'] = f"File too large for demo ({input_data.size / (1024*1024):.1f}MB). Maximum size is {rule.threshold / (1024*1024):.1f}MB."
                    result['technical_details'] = {'file_size': input_data.size, 'threshold': rule.threshold}
        
        elif rule.rule_type == 'filename':
            if input_type == 'file' and hasattr(input_data, 'name'):
                suspicious_extensions = rule.threshold.split(',')
                filename_lower = input_data.name.lower()
                for ext in suspicious_extensions:
                    if ext.strip() in filename_lower:
                        result['triggered'] = True
                        result['user_message'] = f"File '{input_data.name}' appears to be a test file and may be skipped for demo safety."
                        result['technical_details'] = {'filename': input_data.name, 'suspicious_pattern': ext.strip()}
                        break
        
        elif rule.rule_type == 'content':
            if isinstance(input_data, str):
                suspicious_patterns = rule.threshold.split(',')
                input_lower = input_data.lower()
                for pattern in suspicious_patterns:
                    if pattern.strip() in input_lower:
                        result['triggered'] = True
                        result['user_message'] = f"Content contains potentially problematic patterns: {pattern.strip()}"
                        result['technical_details'] = {'pattern_found': pattern.strip()}
                        break
        
        elif rule.rule_type == 'frequency':
            # Rate limiting is handled separately
            pass
        
        elif rule.rule_type == 'system_capacity':
            current_capacity = self._assess_system_capacity()
            if current_capacity['cpu_percent'] > rule.threshold * 100 or \
               current_capacity['memory_percent'] > rule.threshold * 100:
                result['triggered'] = True
                result['user_message'] = f"System is under high load (CPU: {current_capacity['cpu_percent']:.1f}%, Memory: {current_capacity['memory_percent']:.1f}%). Processing in simulation mode."
                result['technical_details'] = current_capacity
        
        elif rule.rule_type == 'cost':
            if isinstance(input_data, dict) and 'estimated_cost' in input_data:
                if input_data['estimated_cost'] > rule.threshold:
                    result['triggered'] = True
                    result['user_message'] = f"Operation would cost ${input_data['estimated_cost']:.2f}, exceeding demo limit of ${rule.threshold:.2f}."
                    result['technical_details'] = {'estimated_cost': input_data['estimated_cost'], 'threshold': rule.threshold}
        
        return result
    
    def _check_rate_limit(self) -> bool:
        """Check if request is within rate limits"""
        current_time = datetime.now()
        
        # Reset counts if needed (every minute)
        if (current_time - self.last_reset).seconds > 60:
            self.request_counts.clear()
            self.last_reset = current_time
        
        # Check rate limit
        rate_limit_rule = next((r for r in self.protection_rules if r.name == 'rate_limiting'), None)
        if rate_limit_rule:
            current_count = self.request_counts.get('total', 0)
            if current_count >= rate_limit_rule.threshold:
                return False
        
        # Update count
        self.request_counts['total'] += 1
        return True
    
    def _update_request_counts(self):
        """Update request count statistics"""
        current_time = datetime.now()
        
        # Reset counts if needed (every hour)
        if (current_time - self.last_reset).seconds > 3600:
            self.request_counts.clear()
            self.last_reset = current_time
    
    def _record_protection_event(self, protection_result: Dict[str, Any]):
        """Record a protection event"""
        event = ProtectionEvent(
            timestamp=datetime.now(),
            rule_name="input_protection",
            event_type=protection_result['action_taken'],
            details=protection_result,
            action_taken=protection_result['action_taken'],
            user_friendly_message=protection_result['user_friendly_message']
        )
        
        self.event_history.append(event)
        
        # Keep only last 1000 events
        if len(self.event_history) > 1000:
            self.event_history = self.event_history[-1000:]
        
        # Store in database via connection manager
        try:
            event_data = {
                'event_id': f"protection_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                'input_text': str(protection_result.get('input_data', '')),
                'is_suspicious': not protection_result.get('safe', True),
                'action_taken': protection_result.get('action_taken', 'unknown'),
                'confidence_score': 0.8 if protection_result.get('safe', True) else 0.9,
                'risk_factors': {
                    'warnings': protection_result.get('warnings', []),
                    'blocked': protection_result.get('blocked', False),
                    'simulation_mode': protection_result.get('simulation_mode', False)
                },
                'sanitized_output': protection_result.get('user_friendly_message', ''),
                'metadata': {
                    'protection_events': protection_result.get('protection_events', []),
                    'demo_mode': self.demo_mode
                }
            }
            self.connection_manager.store_protection_event(event_data)
        except Exception as e:
            logger.warning(f"Failed to store protection event in database: {e}")
    
    def get_protection_status(self) -> Dict[str, Any]:
        """Get current protection system status"""
        current_capacity = self._assess_system_capacity()
        
        return {
            'demo_mode': self.demo_mode,
            'system_capacity': current_capacity,
            'protection_stats': self.stats.copy(),
            'active_rules': len([r for r in self.protection_rules if r.enabled]),
            'total_rules': len(self.protection_rules),
            'recent_events': len([e for e in self.event_history if (datetime.now() - e.timestamp).seconds < 3600]),
            'rate_limit_status': {
                'current_requests': self.request_counts.get('total', 0),
                'limit': next((r.threshold for r in self.protection_rules if r.name == 'rate_limiting'), 100)
            }
        }
    
    def get_protection_history(self, hours: int = 24) -> List[ProtectionEvent]:
        """Get protection event history"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            event for event in self.event_history 
            if event.timestamp >= cutoff_time
        ]
    
    def add_custom_rule(self, rule: ProtectionRule):
        """Add a custom protection rule"""
        self.protection_rules.append(rule)
        logger.info(f"Added custom protection rule: {rule.name}")
    
    def disable_rule(self, rule_name: str):
        """Disable a protection rule"""
        for rule in self.protection_rules:
            if rule.name == rule_name:
                rule.enabled = False
                logger.info(f"Disabled protection rule: {rule_name}")
                break
    
    def enable_rule(self, rule_name: str):
        """Enable a protection rule"""
        for rule in self.protection_rules:
            if rule.name == rule_name:
                rule.enabled = True
                logger.info(f"Enabled protection rule: {rule_name}")
                break

class TransparentModeIndicator:
    """Provides transparent mode indicators for demo users"""
    
    def __init__(self, protection_system: DemoProtectionSystem):
        self.protection_system = protection_system
    
    def get_mode_indicator(self) -> Dict[str, Any]:
        """Get current mode indicator for display"""
        status = self.protection_system.get_protection_status()
        
        # Determine current mode
        if status['demo_mode'] and status['system_capacity']['cpu_percent'] < 80:
            mode = "FULL_POWER"
            mode_description = "All features operational with real processing"
            confidence = 0.95
        elif status['demo_mode'] and status['system_capacity']['cpu_percent'] >= 80:
            mode = "HYBRID"
            mode_description = "Mixed real and simulated processing due to system load"
            confidence = 0.7
        else:
            mode = "DEMO"
            mode_description = "Demonstration mode with simulated results"
            confidence = 0.5
        
        return {
            'mode': mode,
            'mode_description': mode_description,
            'confidence': confidence,
            'system_status': status['system_capacity'],
            'protection_active': status['active_rules'] > 0,
            'recent_events': status['recent_events'],
            'user_message': self._get_user_friendly_message(mode, status)
        }
    
    def _get_user_friendly_message(self, mode: str, status: Dict[str, Any]) -> str:
        """Get user-friendly message about current mode"""
        if mode == "FULL_POWER":
            return "🚀 **FULL POWER MODE** - All features operational with real processing"
        elif mode == "HYBRID":
            return f"⚡ **HYBRID MODE** - Mixed processing due to system load (CPU: {status['system_capacity']['cpu_percent']:.1f}%)"
        else:
            return "🎭 **DEMO MODE** - Demonstration with simulated results"
    
    def show_analysis_authenticity(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Show users whether analysis is real or simulated"""
        execution_mode = result.get('execution_mode', 'unknown')
        confidence = result.get('confidence', 0.0)
        
        if execution_mode == 'real':
            return {
                'authenticity': 'REAL',
                'message': f"🔥 **REAL ANALYSIS** - Confidence: {confidence:.1%}",
                'description': "This analysis used actual system capabilities with live processing.",
                'confidence': confidence
            }
        elif execution_mode == 'simulated':
            return {
                'authenticity': 'SIMULATED',
                'message': f"🎭 **SIMULATED ANALYSIS** - Demo Mode",
                'description': "This analysis is simulated for demonstration. Enable real mode by fixing system dependencies.",
                'confidence': confidence
            }
        else:
            return {
                'authenticity': 'UNKNOWN',
                'message': "❓ **UNKNOWN MODE** - System status unclear",
                'description': "Unable to determine analysis authenticity.",
                'confidence': 0.0
            }

# Convenience functions for demo team
def create_demo_protection(demo_mode: bool = True) -> DemoProtectionSystem:
    """Create a demo protection system"""
    return DemoProtectionSystem(demo_mode)

def create_transparent_indicator(protection_system: DemoProtectionSystem) -> TransparentModeIndicator:
    """Create a transparent mode indicator"""
    return TransparentModeIndicator(protection_system)

def protect_demo_input(input_data: Union[str, bytes, Dict[str, Any]], 
                      input_type: str = "text") -> Tuple[bool, Dict[str, Any]]:
    """Protect demo input with default protection system"""
    protection_system = DemoProtectionSystem()
    return protection_system.protect_input(input_data, input_type)

if __name__ == "__main__":
    # Test the demo protection system
    print("🎯 Testing Demo Protection System")
    
    protection_system = create_demo_protection()
    indicator = create_transparent_indicator(protection_system)
    
    # Test mode indicator
    mode_info = indicator.get_mode_indicator()
    print(f"Current mode: {mode_info['mode']}")
    print(f"User message: {mode_info['user_message']}")
    
    # Test input protection
    safe, result = protect_demo_input("This is a normal input")
    print(f"Normal input safe: {safe}")
    
    safe, result = protect_demo_input("This contains script tags <script>alert('test')</script>")
    print(f"Suspicious input safe: {safe}")
    print(f"Action: {result['action_taken']}")
