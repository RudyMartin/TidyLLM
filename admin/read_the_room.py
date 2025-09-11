#!/usr/bin/env python3
"""
ReadTheRoom - Corporate Environment Credential Pathway Detection
================================================================

Detects credential pathways WITHOUT reading actual credential values.
Prevents hanging in tight corporate permission environments.

This function checks for the EXISTENCE of credential sources without
trying to ACCESS them - crucial for corporate environments where reading
might hang or fail due to permissions.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import time


class ReadTheRoom:
    """
    Corporate-safe credential pathway detection.
    
    Detects what credential pathways are available without reading values
    to prevent hanging in tight permission environments.
    """
    
    def __init__(self, timeout_seconds: float = 5.0):
        """Initialize with corporate-safe timeout."""
        self.timeout = timeout_seconds
        self.room_state = {}
        self.start_time = time.time()
    
    def _check_timeout(self) -> bool:
        """Check if we've exceeded our corporate-safe timeout."""
        return (time.time() - self.start_time) > self.timeout
    
    def detect_credential_pathways(self) -> Dict[str, Any]:
        """
        Detect credential pathways without reading values.
        
        Returns:
            Dict with pathway availability (not values!)
        """
        pathways = {
            'timestamp': time.time(),
            'timeout_seconds': self.timeout,
            'environment_variables': self._detect_env_pathways(),
            'files': self._detect_file_pathways(),
            'aws_profile': self._detect_aws_profile_pathways(),
            'iam_role': self._detect_iam_role_pathways(),
            'kms_access': self._detect_kms_pathways(),
            'system_capabilities': self._detect_system_capabilities(),
            'corporate_indicators': self._detect_corporate_indicators(),
            'safe_to_proceed': False  # Will be set based on findings
        }
        
        # Determine if it's safe to proceed with credential reading
        pathways['safe_to_proceed'] = self._assess_safety(pathways)
        pathways['recommended_approach'] = self._recommend_approach(pathways)
        
        return pathways
    
    def _detect_env_pathways(self) -> Dict[str, Any]:
        """Detect environment variable pathways WITHOUT reading values."""
        if self._check_timeout():
            return {'status': 'timeout', 'pathways': []}
        
        # Check for EXISTENCE of env var names, not values
        expected_vars = [
            'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY', 
            'AWS_DEFAULT_REGION',
            'AWS_REGION',
            'AWS_PROFILE',
            'AWS_ROLE_ARN',
            'AWS_SESSION_TOKEN'
        ]
        
        pathways = []
        for var_name in expected_vars:
            try:
                # Check if env var EXISTS (but don't read the value)
                exists = var_name in os.environ
                pathways.append({
                    'name': var_name,
                    'exists': exists,
                    'read_attempted': False  # Corporate safety: never read values
                })
            except Exception as e:
                pathways.append({
                    'name': var_name,
                    'exists': False,
                    'error': str(e)
                })
        
        return {
            'status': 'complete',
            'total_vars_checked': len(expected_vars),
            'vars_present': sum(1 for p in pathways if p.get('exists', False)),
            'pathways': pathways
        }
    
    def _detect_file_pathways(self) -> Dict[str, Any]:
        """Detect credential file pathways WITHOUT reading contents."""
        if self._check_timeout():
            return {'status': 'timeout', 'pathways': []}
        
        # Check for file existence, not contents
        potential_files = [
            Path.home() / '.aws' / 'credentials',
            Path.home() / '.aws' / 'config',
            Path(__file__).parent / 'settings.yaml',
            Path(__file__).parent / 'proposed_settings.yaml',
            Path.cwd() / 'tidyllm_config.yaml',
            Path('/etc/tidyllm/config.yaml')  # Linux
        ]
        
        pathways = []
        for file_path in potential_files:
            try:
                exists = file_path.exists() if hasattr(file_path, 'exists') else os.path.exists(str(file_path))
                readable = os.access(str(file_path), os.R_OK) if exists else False
                
                pathways.append({
                    'path': str(file_path),
                    'exists': exists,
                    'readable': readable,
                    'size_bytes': file_path.stat().st_size if exists and hasattr(file_path, 'stat') else 0,
                    'content_read': False  # Corporate safety: never read contents
                })
            except Exception as e:
                pathways.append({
                    'path': str(file_path),
                    'exists': False,
                    'error': str(e)
                })
        
        return {
            'status': 'complete',
            'files_checked': len(potential_files),
            'files_exist': sum(1 for p in pathways if p.get('exists', False)),
            'files_readable': sum(1 for p in pathways if p.get('readable', False)),
            'pathways': pathways
        }
    
    def _detect_aws_profile_pathways(self) -> Dict[str, Any]:
        """Detect AWS profile pathways WITHOUT accessing AWS."""
        if self._check_timeout():
            return {'status': 'timeout'}
        
        try:
            # Check if boto3 is available (but don't use it)
            boto3_available = False
            try:
                import boto3
                boto3_available = True
            except ImportError:
                pass
            
            # Check AWS config directory structure
            aws_dir = Path.home() / '.aws'
            aws_dir_exists = aws_dir.exists()
            
            return {
                'status': 'complete',
                'boto3_available': boto3_available,
                'aws_directory_exists': aws_dir_exists,
                'profile_access_attempted': False,  # Corporate safety
                'recommendation': 'Use IAM role if in corporate environment'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _detect_iam_role_pathways(self) -> Dict[str, Any]:
        """Detect IAM role availability WITHOUT making AWS calls."""
        if self._check_timeout():
            return {'status': 'timeout'}
        
        # Corporate environments often use IAM roles
        indicators = {
            'ec2_metadata_accessible': False,  # Don't actually check
            'ecs_task_role_available': False,  # Don't actually check
            'lambda_execution_role': False,    # Don't actually check
            'recommendation': 'Safe pathway for corporate environments'
        }
        
        # Check for corporate environment indicators
        corporate_indicators = [
            os.environ.get('AWS_EXECUTION_ENV'),  # Lambda/ECS
            os.environ.get('AWS_LAMBDA_FUNCTION_NAME'),  # Lambda
            os.environ.get('ECS_CONTAINER_METADATA_URI'),  # ECS
        ]
        
        indicators['corporate_env_detected'] = any(indicator for indicator in corporate_indicators)
        indicators['aws_execution_env'] = bool(os.environ.get('AWS_EXECUTION_ENV'))
        
        return {
            'status': 'complete',
            'indicators': indicators,
            'aws_calls_attempted': False  # Corporate safety
        }
    
    def _detect_kms_pathways(self) -> Dict[str, Any]:
        """Detect KMS pathway availability WITHOUT making calls."""
        if self._check_timeout():
            return {'status': 'timeout'}
        
        return {
            'status': 'complete',
            'kms_available': 'unknown',  # Corporate safety: don't test
            'recommendation': 'KMS access usually available in corporate "room"',
            'kms_calls_attempted': False
        }
    
    def _detect_system_capabilities(self) -> Dict[str, Any]:
        """Detect system-level capabilities safely."""
        if self._check_timeout():
            return {'status': 'timeout'}
        
        return {
            'python_version': sys.version,
            'platform': sys.platform,
            'current_user': os.environ.get('USER', os.environ.get('USERNAME', 'unknown')),
            'current_directory': str(Path.cwd()),
            'path_separator': os.path.sep,
            'corporate_indicators': {
                'domain_joined': 'USERDOMAIN' in os.environ,
                'corporate_proxy': any(proxy in os.environ for proxy in ['HTTP_PROXY', 'HTTPS_PROXY', 'CORPORATE_PROXY']),
                'restricted_user': not os.access(str(Path.home()), os.W_OK)
            }
        }
    
    def _detect_corporate_indicators(self) -> Dict[str, Any]:
        """Detect corporate environment indicators."""
        indicators = {
            'domain_environment': 'USERDOMAIN' in os.environ,
            'proxy_environment': any(proxy in os.environ for proxy in ['HTTP_PROXY', 'HTTPS_PROXY']),
            'restricted_home': not os.access(str(Path.home()), os.W_OK),
            'aws_execution_env': bool(os.environ.get('AWS_EXECUTION_ENV')),
            'lambda_environment': bool(os.environ.get('AWS_LAMBDA_FUNCTION_NAME')),
            'ecs_environment': bool(os.environ.get('ECS_CONTAINER_METADATA_URI')),
        }
        
        corporate_score = sum(1 for indicator in indicators.values() if indicator)
        
        return {
            'indicators': indicators,
            'corporate_score': corporate_score,
            'likely_corporate': corporate_score >= 2,
            'recommended_approach': 'iam_role' if corporate_score >= 2 else 'environment_vars'
        }
    
    def _assess_safety(self, pathways: Dict[str, Any]) -> bool:
        """Assess if it's safe to proceed with credential reading."""
        
        # If likely corporate, be more cautious
        if pathways.get('corporate_indicators', {}).get('likely_corporate', False):
            # Corporate: prefer IAM roles, avoid env var reading
            return pathways.get('iam_role', {}).get('corporate_env_detected', False)
        
        # Non-corporate: more permissive
        env_vars_available = pathways.get('environment_variables', {}).get('vars_present', 0) > 2
        files_available = pathways.get('files', {}).get('files_readable', 0) > 0
        
        return env_vars_available or files_available
    
    def _recommend_approach(self, pathways: Dict[str, Any]) -> str:
        """Recommend safest credential approach based on room reading."""
        
        corporate_indicators = pathways.get('corporate_indicators', {})
        
        if corporate_indicators.get('likely_corporate', False):
            if corporate_indicators.get('indicators', {}).get('aws_execution_env'):
                return 'iam_role'  # Safest for corporate
            else:
                return 'settings_file'  # Safe fallback
        
        # Non-corporate environments
        env_vars = pathways.get('environment_variables', {}).get('vars_present', 0)
        if env_vars >= 3:
            return 'environment_variables'
        
        files = pathways.get('files', {}).get('files_readable', 0)
        if files > 0:
            return 'settings_file'
        
        return 'manual_configuration_required'


def read_the_room(timeout_seconds: float = 5.0) -> Dict[str, Any]:
    """
    Corporate-safe credential pathway detection.
    
    Args:
        timeout_seconds: Maximum time to spend detecting (corporate safety)
    
    Returns:
        Dict with pathway information and safety recommendations
    """
    reader = ReadTheRoom(timeout_seconds)
    return reader.detect_credential_pathways()


def main():
    """Demo ReadTheRoom functionality."""
    print("=" * 60)
    print("READ THE ROOM - Corporate Credential Pathway Detection")
    print("=" * 60)
    
    print("🔍 Reading the room (timeout: 5s)...")
    room_reading = read_the_room(timeout_seconds=5.0)
    
    print(f"\n📊 Room Reading Complete:")
    print(f"   Corporate Environment: {room_reading.get('corporate_indicators', {}).get('likely_corporate', False)}")
    print(f"   Safe to Proceed: {room_reading.get('safe_to_proceed', False)}")
    print(f"   Recommended Approach: {room_reading.get('recommended_approach', 'unknown')}")
    
    print(f"\n🌍 Environment Variables:")
    env_info = room_reading.get('environment_variables', {})
    print(f"   Variables Present: {env_info.get('vars_present', 0)}/{env_info.get('total_vars_checked', 0)}")
    
    print(f"\n📁 File Pathways:")
    file_info = room_reading.get('files', {})
    print(f"   Files Available: {file_info.get('files_exist', 0)}/{file_info.get('files_checked', 0)}")
    print(f"   Files Readable: {file_info.get('files_readable', 0)}")
    
    print(f"\n🏢 Corporate Indicators:")
    corp_info = room_reading.get('corporate_indicators', {})
    corp_score = corp_info.get('corporate_score', 0)
    print(f"   Corporate Score: {corp_score}/6")
    
    if corp_info.get('likely_corporate', False):
        print("   🚨 CORPORATE ENVIRONMENT DETECTED")
        print("   ✅ Using corporate-safe credential detection")
    else:
        print("   ℹ️ Standard environment detected")
    
    print(f"\n💡 Recommendation: {room_reading.get('recommended_approach', 'unknown')}")
    
    # Show full results in JSON for debugging
    print(f"\n🔧 Full Room Reading (for debugging):")
    print(json.dumps(room_reading, indent=2, default=str))


if __name__ == "__main__":
    main()