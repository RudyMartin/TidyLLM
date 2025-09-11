"""
Base Validator for TidyLLM
==========================

Provides corporate-safe validation with timeout protection and environment detection.
"""

import time
import threading
from typing import Dict, Any, Optional
from pathlib import Path
import sys

# Add admin directory for ReadTheRoom
sys.path.insert(0, str(Path(__file__).parent.parent / "admin"))

try:
    from read_the_room import read_the_room
    ROOM_READER_AVAILABLE = True
except ImportError:
    ROOM_READER_AVAILABLE = False


class BaseValidator:
    """Base validator with corporate-safe timeout and environment detection."""
    
    def __init__(self, timeout_seconds: float = 10.0):
        """
        Initialize base validator.
        
        Args:
            timeout_seconds: Maximum time for any validation operation
        """
        self.timeout = timeout_seconds
        self.room_reading = None
        self.corporate_mode = False
        
    def detect_corporate_environment(self) -> Dict[str, Any]:
        """Detect corporate environment using ReadTheRoom."""
        
        if ROOM_READER_AVAILABLE:
            try:
                self.room_reading = read_the_room(timeout_seconds=5.0)
                corporate_indicators = self.room_reading.get('corporate_indicators', {})
                self.corporate_mode = corporate_indicators.get('likely_corporate', False)
                
                # Enhanced corporate detection
                corporate_score = corporate_indicators.get('corporate_score', 0)
                if corporate_score >= 4:  # High confidence corporate environment
                    self.corporate_mode = True
                    
                return self.room_reading
                    
            except Exception as e:
                print(f"[BASE-VALIDATOR] Room reading failed: {e}")
                self.corporate_mode = True  # Assume corporate if room reading fails
        else:
            self.corporate_mode = True  # Assume corporate if no room reader
            
        return {'corporate_mode': self.corporate_mode}
    
    def run_with_timeout(self, test_func, service_name: str, timeout_seconds: Optional[float] = None) -> Dict[str, Any]:
        """
        Run a test function with timeout protection.
        
        Args:
            test_func: Function to run with timeout
            service_name: Name of service being tested
            timeout_seconds: Optional custom timeout
            
        Returns:
            Test results with timing and status
        """
        
        if timeout_seconds is None:
            timeout_seconds = self.timeout / 3
            
        result = {'status': 'timeout', 'message': f'{service_name} test timed out', 'latency': 0}
        
        def target():
            nonlocal result
            try:
                result = test_func()
            except Exception as e:
                result = {
                    'status': 'error',
                    'message': f'{service_name} test failed: {e}',
                    'latency': 0
                }
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        
        start_time = time.time()
        thread.start()
        thread.join(timeout_seconds)
        
        latency = (time.time() - start_time) * 1000
        
        if thread.is_alive():
            return {
                'status': 'timeout',
                'message': f'{service_name} test timed out after {timeout_seconds}s',
                'latency': latency
            }
        
        result['latency'] = latency
        return result
    
    def corporate_safe_result(self, service_name: str, message: str) -> Dict[str, Any]:
        """
        Return a corporate-safe result without making actual service calls.
        
        Args:
            service_name: Name of the service
            message: Corporate-safe message
            
        Returns:
            Corporate-safe result dict
        """
        return {
            'status': 'corporate_safe',
            'message': message,
            'latency': 0,
            'corporate_mode': True,
            'service': service_name
        }