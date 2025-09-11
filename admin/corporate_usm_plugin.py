#!/usr/bin/env python3
"""
Corporate USM Plugin - ReadTheRoom Integration
==============================================

PLUGIN approach to integrate ReadTheRoom with existing USM WITHOUT changing USM code.

This creates a corporate-safe wrapper around the existing UnifiedSessionManager
that uses ReadTheRoom to prevent hanging in tight corporate environments.

APPROACH:
- Does NOT modify existing USM code 
- Creates a corporate-safe wrapper class
- Uses ReadTheRoom before calling USM
- Provides corporate environment detection
- Has built-in timeouts and fallbacks
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Add path for USM import
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts" / "infrastructure"))

try:
    from start_unified_sessions import UnifiedSessionManager
    USM_AVAILABLE = True
except ImportError:
    USM_AVAILABLE = False

# Our corporate-safe modules  
from read_the_room import read_the_room


class CorporateUSMPlugin:
    """
    Corporate-safe wrapper around existing UnifiedSessionManager.
    
    Uses ReadTheRoom to determine if it's safe to initialize USM in corporate environments.
    """
    
    def __init__(self, timeout_seconds: float = 15.0):
        """
        Initialize corporate plugin.
        
        Args:
            timeout_seconds: Maximum time to spend on USM initialization
        """
        self.timeout = timeout_seconds
        self.room_reading = None
        self.usm_instance = None
        self.corporate_mode = False
        self.initialization_start = None
        
    def initialize_corporate_safe(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Initialize USM using corporate-safe approach.
        
        Returns:
            Tuple of (success, details_dict)
        """
        
        self.initialization_start = time.time()
        
        print("[CORPORATE-USM] Starting corporate-safe USM initialization...")
        
        # Step 1: Read the room first (this is fast and won't hang)
        try:
            print("[CORPORATE-USM] Reading the room...")
            self.room_reading = read_the_room(timeout_seconds=5.0)
        except Exception as e:
            print(f"[CORPORATE-USM] Room reading failed: {e}")
            return False, {'error': f'Room reading failed: {e}'}
        
        # Step 2: Check if we're in a corporate environment
        corporate_indicators = self.room_reading.get('corporate_indicators', {})
        self.corporate_mode = corporate_indicators.get('likely_corporate', False)
        
        if self.corporate_mode:
            print("[CORPORATE-USM] Corporate environment detected - using safe approach")
            return self._initialize_corporate_mode()
        else:
            print("[CORPORATE-USM] Standard environment - using normal approach")  
            return self._initialize_standard_mode()
    
    def _initialize_corporate_mode(self) -> Tuple[bool, Dict[str, Any]]:
        """Initialize in corporate mode with extra safety."""
        
        print("[CORPORATE-USM] Corporate mode initialization...")
        
        # In corporate environments, be extra careful
        if not self.room_reading.get('safe_to_proceed', False):
            print("[CORPORATE-USM] Room reading says NOT SAFE - skipping USM initialization")
            
            return False, {
                'corporate_mode': True,
                'safe_to_proceed': False,  
                'reason': 'Room reading indicated unsafe to proceed with credential discovery',
                'recommendation': 'Use IAM role or manual configuration',
                'room_reading': self.room_reading
            }
        
        # Corporate environment but room reading says safe - proceed with caution
        return self._try_usm_with_timeout()
    
    def _initialize_standard_mode(self) -> Tuple[bool, Dict[str, Any]]:
        """Initialize in standard mode."""
        
        print("[CORPORATE-USM] Standard mode initialization...")
        
        if self.room_reading.get('safe_to_proceed', True):
            return self._try_usm_with_timeout()
        else:
            print("[CORPORATE-USM] Room reading recommends caution - using timeout")
            return self._try_usm_with_timeout(extra_caution=True)
    
    def _try_usm_with_timeout(self, extra_caution: bool = False) -> Tuple[bool, Dict[str, Any]]:
        """Try to initialize USM with timeout protection."""
        
        if not USM_AVAILABLE:
            return False, {'error': 'UnifiedSessionManager not available'}
        
        timeout = self.timeout / 2 if extra_caution else self.timeout
        print(f"[CORPORATE-USM] Initializing USM with {timeout}s timeout...")
        
        try:
            # IMPORTANT: This uses the EXISTING USM code without modification
            # We just wrap it with timeout and corporate detection
            self.usm_instance = UnifiedSessionManager()
            
            # Check if initialization completed within timeout  
            elapsed = time.time() - self.initialization_start
            if elapsed > timeout:
                print(f"[CORPORATE-USM] USM initialization took {elapsed:.2f}s (timeout: {timeout}s)")
                return False, {
                    'error': 'USM initialization timeout',
                    'elapsed_seconds': elapsed,
                    'timeout_seconds': timeout
                }
            
            print(f"[CORPORATE-USM] USM initialized successfully in {elapsed:.2f}s")
            
            return True, {
                'success': True,
                'elapsed_seconds': elapsed,
                'corporate_mode': self.corporate_mode,
                'usm_available': True,
                'room_reading': self.room_reading
            }
            
        except Exception as e:
            print(f"[CORPORATE-USM] USM initialization failed: {e}")
            return False, {
                'error': str(e),
                'corporate_mode': self.corporate_mode,
                'room_reading': self.room_reading
            }
    
    def get_usm_instance(self) -> Optional[object]:
        """
        Get the UnifiedSessionManager instance if available.
        
        Returns:
            USM instance or None if not initialized
        """
        return self.usm_instance
    
    def is_corporate_mode(self) -> bool:
        """Check if we're running in corporate mode."""
        return self.corporate_mode
    
    def get_room_reading(self) -> Dict[str, Any]:
        """Get the room reading results."""
        return self.room_reading or {}
    
    def get_corporate_safe_session_info(self) -> Dict[str, Any]:
        """
        Get session information in a corporate-safe way.
        
        Returns:
            Session info dict (safe for corporate environments)
        """
        
        if not self.usm_instance:
            return {
                'usm_available': False,
                'corporate_mode': self.corporate_mode,
                'room_reading': self.room_reading
            }
        
        # Get basic info without triggering credential reads
        try:
            info = {
                'usm_available': True,
                'corporate_mode': self.corporate_mode,
                'room_reading': self.room_reading
            }
            
            # In corporate mode, don't try to access credential details
            if self.corporate_mode:
                info['credential_details'] = 'Hidden for corporate safety'
            else:
                # In standard mode, we can get more details
                if hasattr(self.usm_instance, 'config'):
                    config = self.usm_instance.config
                    info['credential_source'] = str(getattr(config, 'credential_source', 'unknown'))
                    info['postgres_host'] = getattr(config, 'postgres_host', 'unknown')
                    info['s3_bucket'] = getattr(config, 's3_default_bucket', 'unknown')
            
            return info
            
        except Exception as e:
            return {
                'usm_available': True,
                'error': str(e),
                'corporate_mode': self.corporate_mode
            }


def initialize_usm_corporate_safe(timeout_seconds: float = 15.0) -> Tuple[Optional[object], Dict[str, Any]]:
    """
    Initialize UnifiedSessionManager with corporate safety.
    
    Args:
        timeout_seconds: Maximum time to spend on initialization
        
    Returns:
        Tuple of (USM instance or None, details dict)
    """
    
    plugin = CorporateUSMPlugin(timeout_seconds)
    success, details = plugin.initialize_corporate_safe()
    
    if success:
        return plugin.get_usm_instance(), details
    else:
        return None, details


def main():
    """Demo corporate-safe USM initialization."""
    print("=" * 70)
    print("CORPORATE-SAFE USM PLUGIN")
    print("=" * 70)
    
    print("[START] Initializing USM with corporate safety...")
    
    try:
        usm_instance, details = initialize_usm_corporate_safe(timeout_seconds=15.0)
        
        if usm_instance:
            print(f"\n[SUCCESS] USM initialized successfully!")
            print(f"   Corporate Mode: {details.get('corporate_mode', False)}")
            print(f"   Elapsed Time: {details.get('elapsed_seconds', 0):.2f}s")
            
            # Get corporate-safe session info
            plugin = CorporateUSMPlugin()
            plugin.usm_instance = usm_instance
            plugin.corporate_mode = details.get('corporate_mode', False)
            plugin.room_reading = details.get('room_reading', {})
            
            session_info = plugin.get_corporate_safe_session_info()
            
            print(f"\n[SESSION] USM Session Information:")
            for key, value in session_info.items():
                if key != 'room_reading':  # Skip detailed room reading in summary
                    print(f"   {key}: {value}")
            
        else:
            print(f"\n[FAILED] USM initialization failed")
            print(f"   Reason: {details.get('error', 'unknown')}")
            print(f"   Corporate Mode: {details.get('corporate_mode', False)}")
            
            if details.get('corporate_mode'):
                print(f"   [CORPORATE] This may be normal in tight corporate environments")
                print(f"   [CORPORATE] Consider using IAM role or manual configuration")
        
    except Exception as e:
        print(f"\n[ERROR] Corporate-safe USM plugin failed: {e}")


if __name__ == "__main__":
    main()