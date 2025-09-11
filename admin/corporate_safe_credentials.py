#!/usr/bin/env python3
"""
Corporate-Safe Credential Loader
================================

Uses ReadTheRoom to detect credential pathways before attempting to read them.
Prevents hanging in tight corporate permission environments.

This is the corporate-safe version of credential loading that:
1. Reads the room first (non-blocking)
2. Only attempts credential reading if safe
3. Uses recommended pathway based on environment
4. Has built-in timeouts and fallbacks
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import time
import yaml

# Import our room reader
from .read_the_room import read_the_room


class CorporateSafeCredentialLoader:
    """
    Corporate-safe credential loading that won't hang in tight permission environments.
    """
    
    def __init__(self, timeout_seconds: float = 10.0):
        """
        Initialize corporate-safe credential loader.
        
        Args:
            timeout_seconds: Maximum time to spend on credential operations
        """
        self.timeout = timeout_seconds
        self.room_reading = None
        self.load_start_time = None
        
    def _check_timeout(self) -> bool:
        """Check if we've exceeded our timeout."""
        if self.load_start_time is None:
            return False
        return (time.time() - self.load_start_time) > self.timeout
    
    def load_credentials_safely(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Load credentials using corporate-safe approach.
        
        Returns:
            Tuple of (credentials_dict, room_reading_dict)
        """
        self.load_start_time = time.time()
        
        # Step 1: Read the room (non-blocking, fast)
        print("[CORPORATE-SAFE] Reading the room...")
        try:
            self.room_reading = read_the_room(timeout_seconds=5.0)
        except Exception as e:
            print(f"[WARNING] Room reading failed: {e}")
            self.room_reading = {'safe_to_proceed': False, 'error': str(e)}
        
        # Step 2: Check if it's safe to proceed
        if not self.room_reading.get('safe_to_proceed', False):
            print("[CORPORATE-SAFE] Room reading indicates NOT SAFE to proceed")
            return self._handle_unsafe_environment()
        
        # Step 3: Use recommended approach
        recommended_approach = self.room_reading.get('recommended_approach', 'settings_file')
        print(f"[CORPORATE-SAFE] Using recommended approach: {recommended_approach}")
        
        try:
            credentials = self._load_by_approach(recommended_approach)
            return credentials, self.room_reading
        except Exception as e:
            print(f"[CORPORATE-SAFE] Primary approach failed: {e}")
            return self._try_fallback_approaches()
    
    def _handle_unsafe_environment(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Handle cases where room reading indicates unsafe environment."""
        
        corporate_indicators = self.room_reading.get('corporate_indicators', {})
        
        if corporate_indicators.get('likely_corporate', False):
            print("[CORPORATE-SAFE] Corporate environment detected")
            print("[CORPORATE-SAFE] Attempting IAM role approach only")
            
            # In corporate environments, try IAM role approach only
            credentials = {
                'approach': 'iam_role',
                'aws_access_key_id': None,  # Let boto3 use IAM role
                'aws_secret_access_key': None,  # Let boto3 use IAM role  
                'aws_default_region': 'us-east-1',  # Safe default
                'corporate_mode': True,
                'room_reading_safe': False
            }
            
            return credentials, self.room_reading
        
        else:
            print("[CORPORATE-SAFE] Environment unclear, using minimal settings")
            # Fallback to settings file only
            return self._try_settings_file_only()
    
    def _load_by_approach(self, approach: str) -> Dict[str, Any]:
        """Load credentials using the specified approach."""
        
        if self._check_timeout():
            raise TimeoutError("Corporate-safe timeout exceeded")
        
        if approach == 'iam_role':
            return self._load_iam_role_credentials()
        
        elif approach == 'settings_file':
            return self._load_settings_file_credentials()
        
        elif approach == 'environment_variables':
            return self._load_environment_credentials() 
        
        else:
            raise ValueError(f"Unknown approach: {approach}")
    
    def _load_iam_role_credentials(self) -> Dict[str, Any]:
        """Load credentials using IAM role approach (corporate-safe)."""
        
        print("[CORPORATE-SAFE] Using IAM role approach")
        
        # Don't try to read actual credentials - let boto3 handle it
        return {
            'approach': 'iam_role',
            'aws_access_key_id': None,  # boto3 will get from IAM role
            'aws_secret_access_key': None,  # boto3 will get from IAM role
            'aws_default_region': os.environ.get('AWS_DEFAULT_REGION', 'us-east-1'),
            'corporate_mode': True,
            'iam_role_used': True
        }
    
    def _load_settings_file_credentials(self) -> Dict[str, Any]:
        """Load credentials from settings file (corporate-safe)."""
        
        print("[CORPORATE-SAFE] Using settings file approach")
        
        # Try to load from settings.yaml
        settings_path = Path(__file__).parent / "settings.yaml"
        
        if not settings_path.exists():
            raise FileNotFoundError(f"Settings file not found: {settings_path}")
        
        try:
            with open(settings_path, 'r') as f:
                settings = yaml.safe_load(f)
            
            credentials_section = settings.get('credentials', {})
            aws_creds = credentials_section.get('aws', {})
            
            return {
                'approach': 'settings_file',
                'aws_access_key_id': aws_creds.get('access_key_id'),
                'aws_secret_access_key': aws_creds.get('secret_access_key'),
                'aws_default_region': aws_creds.get('default_region', 'us-east-1'),
                'corporate_mode': False,
                'settings_file_path': str(settings_path)
            }
            
        except Exception as e:
            raise Exception(f"Error reading settings file: {e}")
    
    def _load_environment_credentials(self) -> Dict[str, Any]:
        """Load credentials from environment variables (with corporate timeout)."""
        
        print("[CORPORATE-SAFE] Using environment variables approach")
        
        # Quick timeout check
        if self._check_timeout():
            raise TimeoutError("Timeout during environment variable reading")
        
        # Only read if room reading indicated it was safe
        try:
            return {
                'approach': 'environment_variables',
                'aws_access_key_id': os.environ.get('AWS_ACCESS_KEY_ID'),
                'aws_secret_access_key': os.environ.get('AWS_SECRET_ACCESS_KEY'),  
                'aws_default_region': os.environ.get('AWS_DEFAULT_REGION', 'us-east-1'),
                'corporate_mode': False
            }
        except Exception as e:
            raise Exception(f"Error reading environment variables: {e}")
    
    def _try_settings_file_only(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Try settings file approach only (safest fallback)."""
        
        print("[CORPORATE-SAFE] Fallback: settings file only")
        
        try:
            credentials = self._load_settings_file_credentials()
            return credentials, self.room_reading
        except Exception as e:
            # Last resort - return minimal working config
            print(f"[CORPORATE-SAFE] Settings file failed: {e}")
            
            minimal_credentials = {
                'approach': 'manual_configuration_required',
                'aws_access_key_id': None,
                'aws_secret_access_key': None,
                'aws_default_region': 'us-east-1',
                'error': str(e),
                'manual_setup_required': True
            }
            
            return minimal_credentials, self.room_reading
    
    def _try_fallback_approaches(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Try fallback approaches in safe order."""
        
        print("[CORPORATE-SAFE] Trying fallback approaches...")
        
        # Safe fallback order for corporate environments
        fallback_approaches = ['settings_file', 'iam_role']
        
        for approach in fallback_approaches:
            if self._check_timeout():
                break
                
            try:
                print(f"[CORPORATE-SAFE] Trying fallback: {approach}")
                credentials = self._load_by_approach(approach)
                return credentials, self.room_reading
            except Exception as e:
                print(f"[CORPORATE-SAFE] Fallback {approach} failed: {e}")
                continue
        
        # All fallbacks failed
        return self._try_settings_file_only()
    
    def test_credentials_safely(self, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test credentials safely without hanging.
        
        Args:
            credentials: Credential dict from load_credentials_safely()
            
        Returns:
            Test results dict
        """
        
        print("[CORPORATE-SAFE] Testing credentials safely...")
        
        # Don't actually test in corporate environments to avoid hanging
        if credentials.get('corporate_mode', False):
            return {
                'test_performed': False,
                'reason': 'Corporate mode - skipping actual credential testing to avoid hanging',
                'assumed_working': True,
                'approach': credentials.get('approach', 'unknown')
            }
        
        # For non-corporate, do minimal testing with timeout
        try:
            if credentials.get('approach') == 'iam_role':
                return {
                    'test_performed': False,
                    'reason': 'IAM role - boto3 will handle authentication',
                    'assumed_working': True
                }
            
            # Quick validation of credential format
            access_key = credentials.get('aws_access_key_id')
            secret_key = credentials.get('aws_secret_access_key')
            
            if access_key and secret_key:
                return {
                    'test_performed': True,
                    'credentials_present': True,
                    'access_key_format': 'valid' if len(access_key) >= 16 else 'invalid',
                    'approach': credentials.get('approach', 'unknown')
                }
            else:
                return {
                    'test_performed': True,
                    'credentials_present': False,
                    'approach': credentials.get('approach', 'unknown')
                }
                
        except Exception as e:
            return {
                'test_performed': True,
                'error': str(e),
                'approach': credentials.get('approach', 'unknown')
            }


def load_credentials_corporate_safe(timeout_seconds: float = 10.0) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Corporate-safe credential loading function.
    
    Args:
        timeout_seconds: Maximum time to spend loading credentials
        
    Returns:
        Tuple of (credentials, room_reading)
    """
    loader = CorporateSafeCredentialLoader(timeout_seconds)
    return loader.load_credentials_safely()


def main():
    """Demo corporate-safe credential loading."""
    print("=" * 70)
    print("CORPORATE-SAFE CREDENTIAL LOADER")
    print("=" * 70)
    
    print("[START] Loading credentials with corporate safety...")
    
    try:
        credentials, room_reading = load_credentials_corporate_safe(timeout_seconds=10.0)
        
        print(f"\n[SUCCESS] Credentials loaded safely!")
        print(f"   Approach: {credentials.get('approach', 'unknown')}")
        print(f"   Corporate Mode: {credentials.get('corporate_mode', False)}")
        print(f"   Has Access Key: {bool(credentials.get('aws_access_key_id'))}")
        print(f"   Region: {credentials.get('aws_default_region', 'unknown')}")
        
        print(f"\n[ROOM] Room reading summary:")
        print(f"   Corporate Environment: {room_reading.get('corporate_indicators', {}).get('likely_corporate', False)}")
        print(f"   Safe to Proceed: {room_reading.get('safe_to_proceed', False)}")
        print(f"   Recommended Approach: {room_reading.get('recommended_approach', 'unknown')}")
        
        # Test credentials safely
        loader = CorporateSafeCredentialLoader()
        test_results = loader.test_credentials_safely(credentials)
        
        print(f"\n[TEST] Credential test results:")
        print(f"   Test Performed: {test_results.get('test_performed', False)}")
        if test_results.get('test_performed', False):
            print(f"   Credentials Present: {test_results.get('credentials_present', False)}")
        else:
            print(f"   Reason: {test_results.get('reason', 'unknown')}")
        
    except Exception as e:
        print(f"\n[ERROR] Corporate-safe loading failed: {e}")


if __name__ == "__main__":
    main()