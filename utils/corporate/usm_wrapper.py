#!/usr/bin/env python3
"""
Corporate-Safe UnifiedSessionManager Wrapper
===========================================

This module provides a corporate-safe wrapper around UnifiedSessionManager that prevents
hanging in tight corporate permission environments. It uses the corporate_safe_validator
to determine the environment and initializes USM with appropriate safety measures.

Key Features:
- Uses CorporateSafeValidator to detect corporate environments
- Skips hanging service calls in corporate mode
- Provides fallback initialization paths
- Maintains compatibility with existing USM interface
"""

import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

# Import from same package
try:
    from .validator import CorporateSafeValidator
    VALIDATOR_AVAILABLE = True
except ImportError:
    VALIDATOR_AVAILABLE = False

try:
    # Try to import UnifiedSessionManager
    from scripts.infrastructure.start_unified_sessions import UnifiedSessionManager
    USM_AVAILABLE = True
except ImportError:
    USM_AVAILABLE = False

try:
    import boto3
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False


class CorporateSafeUSM:
    """Corporate-safe wrapper around UnifiedSessionManager"""
    
    def __init__(self, timeout_seconds: float = 15.0):
        """
        Initialize corporate-safe USM wrapper.
        
        Args:
            timeout_seconds: Maximum time to spend on initialization
        """
        self.timeout = timeout_seconds
        self.validator = None
        self.usm = None
        self.corporate_mode = False
        self.validation_results = None
        self.initialization_status = "not_initialized"
        
    def initialize_safely(self) -> Dict[str, Any]:
        """
        Initialize USM with corporate safety measures.
        
        Returns:
            Initialization results dict
        """
        print("[CORPORATE-USM] Starting corporate-safe USM initialization...")
        
        start_time = time.time()
        
        # Step 1: Validate environment safety
        validation_results = self._validate_environment()
        
        # Step 2: Initialize USM based on validation results
        if validation_results.get('overall_assessment') == 'ready':
            usm_results = self._initialize_standard_usm()
        elif validation_results.get('overall_assessment') == 'partial_ready':
            usm_results = self._initialize_limited_usm()
        else:
            usm_results = self._initialize_fallback_usm()
        
        total_time = (time.time() - start_time) * 1000
        
        return {
            'timestamp': time.time(),
            'total_time': total_time,
            'corporate_mode': self.corporate_mode,
            'validation_results': validation_results,
            'usm_initialization': usm_results,
            'overall_status': usm_results.get('status', 'unknown'),
            'usm_available': self.usm is not None
        }
    
    def _validate_environment(self) -> Dict[str, Any]:
        """Validate environment using CorporateSafeValidator"""
        
        if not VALIDATOR_AVAILABLE:
            print("[CORPORATE-USM] Validator not available - assuming corporate mode")
            self.corporate_mode = True
            return {
                'overall_assessment': 'not_ready',
                'corporate_mode': True,
                'validator_available': False
            }
        
        try:
            self.validator = CorporateSafeValidator(timeout_seconds=self.timeout / 2)
            self.validation_results = self.validator.run_full_validation_safe()
            
            self.corporate_mode = self.validation_results.get('corporate_mode', True)
            
            print(f"[CORPORATE-USM] Environment validation completed")
            print(f"   Corporate Mode: {self.corporate_mode}")
            print(f"   Assessment: {self.validation_results.get('overall_assessment', 'unknown')}")
            
            return self.validation_results
            
        except Exception as e:
            print(f"[CORPORATE-USM] Environment validation failed: {e}")
            self.corporate_mode = True
            return {
                'overall_assessment': 'validation_failed',
                'corporate_mode': True,
                'error': str(e)
            }
    
    def _initialize_standard_usm(self) -> Dict[str, Any]:
        """Initialize USM in standard mode with full functionality"""
        
        print("[CORPORATE-USM] Initializing USM in standard mode...")
        
        if not USM_AVAILABLE:
            return {
                'status': 'error',
                'message': 'UnifiedSessionManager not available',
                'mode': 'standard'
            }
        
        try:
            self.usm = UnifiedSessionManager()
            self.initialization_status = "standard"
            
            return {
                'status': 'success',
                'message': 'USM initialized in standard mode with full functionality',
                'mode': 'standard',
                'features': ['s3_client', 'bedrock_client', 'health_monitoring', 'connection_pooling']
            }
            
        except Exception as e:
            print(f"[CORPORATE-USM] Standard USM initialization failed: {e}")
            return self._initialize_fallback_usm()
    
    def _initialize_limited_usm(self) -> Dict[str, Any]:
        """Initialize USM in limited mode with reduced functionality"""
        
        print("[CORPORATE-USM] Initializing USM in limited mode...")
        
        if not USM_AVAILABLE:
            return self._initialize_fallback_usm()
        
        try:
            # Initialize USM but note that some operations may be limited
            self.usm = UnifiedSessionManager()
            self.initialization_status = "limited"
            
            return {
                'status': 'partial',
                'message': 'USM initialized in limited mode - some operations may be restricted',
                'mode': 'limited',
                'features': ['basic_s3_client', 'basic_bedrock_client'],
                'restrictions': ['no_list_buckets', 'no_list_models', 'limited_health_checks']
            }
            
        except Exception as e:
            print(f"[CORPORATE-USM] Limited USM initialization failed: {e}")
            return self._initialize_fallback_usm()
    
    def _initialize_fallback_usm(self) -> Dict[str, Any]:
        """Initialize fallback USM using direct boto3 clients"""
        
        print("[CORPORATE-USM] Initializing USM in fallback mode...")
        
        if not AWS_AVAILABLE:
            return {
                'status': 'error',
                'message': 'No AWS SDK available - cannot initialize any clients',
                'mode': 'unavailable'
            }
        
        try:
            # Create a minimal USM-like interface using direct boto3
            self.usm = self._create_fallback_usm()
            self.initialization_status = "fallback"
            
            return {
                'status': 'fallback',
                'message': 'USM initialized in fallback mode using direct boto3 clients',
                'mode': 'fallback',
                'features': ['direct_boto3_s3', 'direct_boto3_bedrock'],
                'limitations': ['no_connection_pooling', 'no_health_monitoring', 'basic_functionality_only']
            }
            
        except Exception as e:
            print(f"[CORPORATE-USM] Fallback USM initialization failed: {e}")
            self.initialization_status = "failed"
            return {
                'status': 'error',
                'message': f'All USM initialization methods failed: {e}',
                'mode': 'failed'
            }
    
    def _create_fallback_usm(self) -> 'FallbackUSM':
        """Create a minimal USM-like interface"""
        
        class FallbackUSM:
            """Minimal USM-like interface using direct boto3"""
            
            def __init__(self):
                self._s3_client = None
                self._bedrock_client = None
            
            def get_s3_client(self):
                if not self._s3_client:
                    self._s3_client = boto3.client('s3')
                return self._s3_client
            
            def get_bedrock_client(self):
                if not self._bedrock_client:
                    self._bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
                return self._bedrock_client
            
            def health_check(self):
                return {
                    's3': {'healthy': False, 'message': 'Corporate mode - health check disabled'},
                    'bedrock': {'healthy': False, 'message': 'Corporate mode - health check disabled'}
                }
        
        return FallbackUSM()
    
    def get_s3_client(self):
        """Get S3 client from initialized USM"""
        if self.usm and hasattr(self.usm, 'get_s3_client'):
            return self.usm.get_s3_client()
        return None
    
    def get_bedrock_client(self):
        """Get Bedrock client from initialized USM"""
        if self.usm and hasattr(self.usm, 'get_bedrock_client'):
            return self.usm.get_bedrock_client()
        return None
    
    def is_corporate_mode(self) -> bool:
        """Check if running in corporate mode"""
        return self.corporate_mode
    
    def get_initialization_status(self) -> str:
        """Get USM initialization status"""
        return self.initialization_status
    
    def get_validation_results(self) -> Optional[Dict[str, Any]]:
        """Get environment validation results"""
        return self.validation_results


def initialize_corporate_safe_usm(timeout_seconds: float = 15.0) -> CorporateSafeUSM:
    """
    Initialize UnifiedSessionManager with corporate safety measures.
    
    Args:
        timeout_seconds: Maximum time to spend on initialization
        
    Returns:
        CorporateSafeUSM instance
    """
    corporate_usm = CorporateSafeUSM(timeout_seconds)
    results = corporate_usm.initialize_safely()
    
    print(f"\n[CORPORATE-USM] Initialization Summary:")
    print(f"   Status: {results.get('overall_status')}")
    print(f"   Corporate Mode: {results.get('corporate_mode')}")
    print(f"   Total Time: {results.get('total_time', 0):.2f}ms")
    print(f"   USM Available: {results.get('usm_available')}")
    
    return corporate_usm


def main():
    """Demo corporate-safe USM initialization"""
    print("=" * 60)
    print("CORPORATE-SAFE USM INITIALIZATION")
    print("=" * 60)
    
    try:
        corporate_usm = initialize_corporate_safe_usm(timeout_seconds=20.0)
        
        # Test basic functionality
        print(f"\n[DEMO] Testing basic functionality...")
        s3_client = corporate_usm.get_s3_client()
        bedrock_client = corporate_usm.get_bedrock_client()
        
        print(f"   S3 Client: {'Available' if s3_client else 'Not Available'}")
        print(f"   Bedrock Client: {'Available' if bedrock_client else 'Not Available'}")
        print(f"   Corporate Mode: {corporate_usm.is_corporate_mode()}")
        print(f"   Initialization Status: {corporate_usm.get_initialization_status()}")
        
        print(f"\n[DEMO] Corporate-safe USM initialization completed successfully")
        
    except Exception as e:
        print(f"\n[ERROR] Corporate-safe USM demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()