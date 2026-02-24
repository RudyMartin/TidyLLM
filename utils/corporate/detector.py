"""
Corporate Environment Detector
==============================

Universal corporate environment detection for TidyLLM applications.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import time

# Import ReadTheRoom from admin
try:
    from ..admin.read_the_room import read_the_room
    ROOM_READER_AVAILABLE = True
except ImportError:
    ROOM_READER_AVAILABLE = False


class CorporateEnvironmentDetector:
    """
    Detects corporate environments and recommends connection strategies.
    
    This is a universal detector that can be used by any TidyLLM application
    to determine if it's running in a corporate environment.
    """
    
    def __init__(self, cache_duration: int = 300):
        """
        Initialize detector with optional caching.
        
        Args:
            cache_duration: How long to cache detection results (seconds)
        """
        self.cache_duration = cache_duration
        self._cached_result = None
        self._cache_timestamp = None
        
    def detect(self, timeout_seconds: float = 5.0, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Detect corporate environment characteristics.
        
        Args:
            timeout_seconds: Maximum time for detection
            force_refresh: Bypass cache and force new detection
            
        Returns:
            Detection results including corporate score and recommendations
        """
        # Check cache
        if not force_refresh and self._cached_result:
            if self._cache_timestamp and (time.time() - self._cache_timestamp) < self.cache_duration:
                return self._cached_result
        
        # Perform detection
        result = self._perform_detection(timeout_seconds)
        
        # Cache result
        self._cached_result = result
        self._cache_timestamp = time.time()
        
        return result
    
    def _perform_detection(self, timeout_seconds: float) -> Dict[str, Any]:
        """Perform actual environment detection."""
        
        result = {
            'timestamp': time.time(),
            'detection_method': 'unknown',
            'corporate_score': 0,
            'is_corporate': False,
            'confidence': 'low',
            'recommended_mode': 'standard',
            'details': {}
        }
        
        # Try ReadTheRoom first
        if ROOM_READER_AVAILABLE:
            try:
                room_analysis = read_the_room(timeout_seconds=timeout_seconds)
                
                # Extract corporate indicators
                corporate_indicators = room_analysis.get('corporate_indicators', {})
                corporate_score = corporate_indicators.get('corporate_score', 0)
                
                result.update({
                    'detection_method': 'read_the_room',
                    'corporate_score': corporate_score,
                    'is_corporate': corporate_score >= 4,
                    'confidence': self._determine_confidence(corporate_score),
                    'recommended_mode': 'corporate' if corporate_score >= 4 else 'standard',
                    'details': room_analysis
                })
                
            except Exception as e:
                # Fallback to basic detection
                result = self._basic_detection()
                result['detection_error'] = str(e)
        else:
            # Use basic detection
            result = self._basic_detection()
        
        return result
    
    def _basic_detection(self) -> Dict[str, Any]:
        """Basic corporate detection without ReadTheRoom."""
        
        corporate_score = 0
        indicators = []
        
        # Check for corporate proxy
        if os.environ.get('HTTP_PROXY') or os.environ.get('HTTPS_PROXY'):
            corporate_score += 2
            indicators.append('proxy_configured')
        
        # Check for corporate domains
        hostname = os.environ.get('COMPUTERNAME', '').lower()
        username = os.environ.get('USERNAME', '').lower()
        
        corporate_patterns = ['corp', 'enterprise', 'company', 'org']
        for pattern in corporate_patterns:
            if pattern in hostname or pattern in username:
                corporate_score += 1
                indicators.append(f'corporate_pattern_{pattern}')
        
        # Check for VPN indicators
        if os.path.exists('/etc/resolv.conf'):
            try:
                with open('/etc/resolv.conf', 'r') as f:
                    content = f.read().lower()
                    if 'vpn' in content or 'corporate' in content:
                        corporate_score += 1
                        indicators.append('vpn_detected')
            except:
                pass
        
        # Check for restricted paths
        restricted_paths = [
            'C:\\ProgramData\\Corporate',
            '/opt/corporate',
            '/usr/local/corporate'
        ]
        for path in restricted_paths:
            if os.path.exists(path):
                corporate_score += 1
                indicators.append('corporate_path_exists')
                break
        
        return {
            'timestamp': time.time(),
            'detection_method': 'basic',
            'corporate_score': corporate_score,
            'is_corporate': corporate_score >= 3,
            'confidence': self._determine_confidence(corporate_score),
            'recommended_mode': 'corporate' if corporate_score >= 3 else 'standard',
            'details': {
                'indicators': indicators,
                'basic_detection': True
            }
        }
    
    def _determine_confidence(self, score: int) -> str:
        """Determine confidence level based on corporate score."""
        if score >= 5:
            return 'high'
        elif score >= 3:
            return 'medium'
        else:
            return 'low'
    
    def get_recommendation(self) -> str:
        """Get recommended connection mode based on last detection."""
        if not self._cached_result:
            self.detect()
        
        if self._cached_result.get('is_corporate'):
            return 'corporate_safe'
        else:
            return 'standard'
    
    def get_visual_indicator(self) -> str:
        """Get visual indicator for UI display."""
        if not self._cached_result:
            self.detect()
        
        score = self._cached_result.get('corporate_score', 0)
        is_corporate = self._cached_result.get('is_corporate', False)
        
        if is_corporate:
            return f"[CORPORATE] Corporate Environment (Score: {score}/6)"
        else:
            return f"[STANDARD] Standard Environment (Score: {score}/6)"


def detect_corporate_environment(timeout_seconds: float = 5.0, 
                                cache: bool = True) -> Dict[str, Any]:
    """
    Convenience function for one-time corporate environment detection.
    
    Args:
        timeout_seconds: Maximum time for detection
        cache: Whether to use cached results
        
    Returns:
        Detection results
    """
    detector = CorporateEnvironmentDetector()
    return detector.detect(timeout_seconds=timeout_seconds, force_refresh=not cache)


def is_corporate_environment() -> bool:
    """
    Simple boolean check for corporate environment.
    
    Returns:
        True if corporate environment detected, False otherwise
    """
    result = detect_corporate_environment()
    return result.get('is_corporate', False)


def get_recommended_connection_mode() -> str:
    """
    Get recommended connection mode for current environment.
    
    Returns:
        'corporate_safe' or 'standard'
    """
    result = detect_corporate_environment()
    return result.get('recommended_mode', 'standard')