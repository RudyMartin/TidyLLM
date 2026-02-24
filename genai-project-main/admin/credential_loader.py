#!/usr/bin/env python3
"""
Centralized AWS Credential Loader
=================================

Single source of truth for AWS credential loading across all TidyLLM scripts.
Eliminates hardcoded credentials by providing a standardized loading mechanism.

Usage:
    from tidyllm.admin.credential_loader import load_aws_credentials
    
    # Load credentials (tries YAML first, then platform scripts)
    credentials = load_aws_credentials()
    
    # Set in environment
    import os
    for key, value in credentials.items():
        os.environ[key] = value
    
    # Or use directly with boto3
    import boto3
    session = boto3.Session(**{
        'aws_access_key_id': credentials.get('AWS_ACCESS_KEY_ID'),
        'aws_secret_access_key': credentials.get('AWS_SECRET_ACCESS_KEY'),
        'region_name': credentials.get('AWS_DEFAULT_REGION')
    })
"""

import os
import sys
import yaml
import platform
from pathlib import Path


class CredentialLoader:
    """Centralized AWS credential management for TidyLLM"""
    
    def __init__(self):
        # Admin directory (where settings.yaml and credential scripts live)
        self.admin_dir = Path(__file__).parent
        self.settings_file = self.admin_dir / "settings.yaml"
        
        # Platform-specific credential scripts
        if platform.system().lower() == 'windows':
            self.credential_script = self.admin_dir / "set_aws_env.bat"
            self.platform = "windows"
        else:
            self.credential_script = self.admin_dir / "set_aws_env.sh"
            self.platform = "linux"
    
    def load_credentials(self, verbose=False):
        """
        Load AWS credentials using priority order:
        1. YAML settings file
        2. Platform-specific credential script
        3. Environment variables (if already set)
        
        Returns:
            dict: AWS credentials ready for use
        """
        
        if verbose:
            print("[CREDENTIAL_LOADER] Loading AWS credentials...")
        
        # Method 1: Try YAML first
        credentials = self._load_from_yaml(verbose)
        if self._has_valid_credentials(credentials):
            if verbose:
                print("  [SUCCESS] Loaded from YAML")
            return credentials
        
        # Method 2: Try platform-specific script
        credentials = self._load_from_platform_script(verbose)
        if self._has_valid_credentials(credentials):
            if verbose:
                print("  [SUCCESS] Loaded from platform script")
            return credentials
        
        # Method 3: Check environment variables
        credentials = self._load_from_environment(verbose)
        if self._has_valid_credentials(credentials):
            if verbose:
                print("  [SUCCESS] Loaded from environment")
            return credentials
        
        # Method 4: Emergency fallback (not recommended)
        if verbose:
            print("  [WARNING] Using emergency fallback credentials")
        return self._emergency_fallback()
    
    def _load_from_yaml(self, verbose=False):
        """Load credentials from settings.yaml"""
        
        try:
            if not self.settings_file.exists():
                return {}
            
            if verbose:
                print(f"  [YAML] Reading: {self.settings_file}")
            
            with open(self.settings_file, 'r') as f:
                settings = yaml.safe_load(f)
            
            # Navigate YAML structure
            aws_config = settings.get('aws', {})
            bedrock_config = aws_config.get('bedrock', {})
            credentials_config = bedrock_config.get('credentials', {})
            
            # Extract region from multiple locations
            region = (
                credentials_config.get('region') or 
                bedrock_config.get('region') or 
                aws_config.get('region') or 
                settings.get('s3', {}).get('region') or
                'us-east-1'
            )
            
            credentials = {'AWS_DEFAULT_REGION': region}
            
            # Extract credentials if available
            for yaml_key, env_key in [
                ('access_key_id', 'AWS_ACCESS_KEY_ID'),
                ('secret_access_key', 'AWS_SECRET_ACCESS_KEY'),
                ('session_token', 'AWS_SESSION_TOKEN'),
                ('profile', 'AWS_PROFILE')
            ]:
                value = credentials_config.get(yaml_key)
                if value:
                    credentials[env_key] = value
            
            return credentials
            
        except Exception as e:
            if verbose:
                print(f"  [YAML] Failed to load: {e}")
            return {}
    
    def _load_from_platform_script(self, verbose=False):
        """Load credentials from platform-specific script"""
        
        try:
            if not self.credential_script.exists():
                return {}
            
            if verbose:
                print(f"  [SCRIPT] Reading: {self.credential_script}")
            
            credentials = {}
            
            with open(self.credential_script, 'r') as f:
                for line in f:
                    line = line.strip()
                    
                    # Windows batch format: set VAR=value
                    if self.platform == "windows" and line.startswith('set AWS_'):
                        parts = line[4:].split('=', 1)  # Remove 'set '
                        if len(parts) == 2:
                            key, value = parts
                            credentials[key] = value
                    
                    # Linux shell format: export VAR=value
                    elif self.platform == "linux" and line.startswith('export AWS_'):
                        parts = line[7:].split('=', 1)  # Remove 'export '
                        if len(parts) == 2:
                            key, value = parts
                            credentials[key] = value
            
            return credentials
            
        except Exception as e:
            if verbose:
                print(f"  [SCRIPT] Failed to load: {e}")
            return {}
    
    def _load_from_environment(self, verbose=False):
        """Load credentials from existing environment variables"""
        
        credentials = {}
        
        for env_var in ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 
                       'AWS_DEFAULT_REGION', 'AWS_SESSION_TOKEN', 'AWS_PROFILE']:
            value = os.environ.get(env_var)
            if value:
                credentials[env_var] = value
        
        if verbose and credentials:
            print(f"  [ENV] Found {len(credentials)} environment variables")
        
        return credentials
    
    def _has_valid_credentials(self, credentials):
        """Check if credentials dict has minimum required values"""
        
        # Must have either access keys or profile
        has_keys = ('AWS_ACCESS_KEY_ID' in credentials and 
                   'AWS_SECRET_ACCESS_KEY' in credentials)
        has_profile = 'AWS_PROFILE' in credentials
        
        return has_keys or has_profile
    
    def _emergency_fallback(self):
        """Emergency fallback - only use when all other methods fail"""
        
        return {
            # Credentials loaded by centralized system,
            # Credentials loaded by centralized system,
            # Credentials loaded by centralized system
        }
    
    def set_environment_variables(self, credentials=None):
        """Set credentials as environment variables"""
        
        if credentials is None:
            credentials = self.load_credentials()
        
        for key, value in credentials.items():
            os.environ[key] = value
        
        return len(credentials)
    
    def get_boto3_kwargs(self, credentials=None):
        """Get credentials formatted for boto3.Session()"""
        
        if credentials is None:
            credentials = self.load_credentials()
        
        boto3_kwargs = {}
        
        if 'AWS_ACCESS_KEY_ID' in credentials:
            boto3_kwargs['aws_access_key_id'] = credentials['AWS_ACCESS_KEY_ID']
        
        if 'AWS_SECRET_ACCESS_KEY' in credentials:
            boto3_kwargs['aws_secret_access_key'] = credentials['AWS_SECRET_ACCESS_KEY']
        
        if 'AWS_DEFAULT_REGION' in credentials:
            boto3_kwargs['region_name'] = credentials['AWS_DEFAULT_REGION']
        
        if 'AWS_SESSION_TOKEN' in credentials:
            boto3_kwargs['aws_session_token'] = credentials['AWS_SESSION_TOKEN']
        
        if 'AWS_PROFILE' in credentials:
            boto3_kwargs['profile_name'] = credentials['AWS_PROFILE']
        
        return boto3_kwargs
    
    def get_s3_configuration(self, environment=None):
        """
        Get S3 configuration for the specified environment.
        
        Args:
            environment: Environment name (development, staging, production)
            
        Returns:
            dict: S3 configuration with bucket, base prefixes, and path builder
        """
        try:
            # Load YAML settings
            if not self.settings_file.exists():
                return self._get_default_s3_config()
            
            with open(self.settings_file, 'r') as f:
                settings = yaml.safe_load(f)
            
            s3_config = settings.get('s3', {})
            
            # Get base configuration
            bucket = s3_config.get('bucket', 'nsc-mvp1')
            base_prefixes = s3_config.get('prefixes', {
                'knowledge_base': 'knowledge_base/',
                'mvr_analysis': 'mvr_analysis/',
                'pages': 'pages/',
                'embeddings': 'embeddings/',
                'metadata': 'metadata/',
                'temp': 'temp/'
            })
            
            # Apply environment-specific overrides
            prefix_override = ""
            if environment:
                env_config = s3_config.get('environments', {}).get(environment, {})
                bucket = env_config.get('bucket', bucket)
                prefix_override = env_config.get('prefix_override', '')
            
            def build_path(base_prefix, requirement):
                """Build complete S3 path: [env_override] + base_prefix + requirement"""
                base = base_prefixes.get(base_prefix, base_prefix + '/')
                if not base.endswith('/'):
                    base += '/'
                
                # Ensure requirement ends with '/' if provided
                if requirement and not requirement.endswith('/'):
                    requirement += '/'
                
                # Combine: prefix_override + base + requirement
                full_path = prefix_override + base + (requirement or '')
                return full_path
            
            return {
                'bucket': bucket,
                'environment': environment,
                'base_prefixes': base_prefixes,
                'prefix_override': prefix_override,
                'build_path': build_path
            }
            
        except Exception as e:
            print(f"[WARN] Failed to load S3 config from YAML: {e}")
            return self._get_default_s3_config()
    
    def _get_default_s3_config(self):
        """Get default S3 configuration when YAML is unavailable"""
        
        def build_path(base_prefix, requirement):
            """Simple path builder for default config"""
            base_prefixes = {
                'knowledge_base': 'knowledge_base/',
                'mvr_analysis': 'mvr_analysis/',
                'pages': 'pages/',
                'embeddings': 'embeddings/',
                'metadata': 'metadata/',
                'temp': 'temp/'
            }
            
            base = base_prefixes.get(base_prefix, base_prefix + '/')
            if not base.endswith('/'):
                base += '/'
            
            if requirement and not requirement.endswith('/'):
                requirement += '/'
                
            return base + (requirement or '')
        
        return {
            'bucket': 'nsc-mvp1',
            'environment': None,
            'base_prefixes': {
                'knowledge_base': 'knowledge_base/',
                'mvr_analysis': 'mvr_analysis/', 
                'pages': 'pages/',
                'embeddings': 'embeddings/',
                'metadata': 'metadata/',
                'temp': 'temp/'
            },
            'prefix_override': '',
            'build_path': build_path
        }


# Global instance for easy importing
_credential_loader = CredentialLoader()


def load_aws_credentials(verbose=False):
    """
    Load AWS credentials using the centralized system.
    
    Args:
        verbose: Print detailed loading information
        
    Returns:
        dict: AWS credentials ready for environment variables
    """
    return _credential_loader.load_credentials(verbose=verbose)


def set_aws_environment(verbose=False):
    """
    Load and set AWS credentials as environment variables.
    
    Args:
        verbose: Print detailed loading information
        
    Returns:
        int: Number of environment variables set
    """
    return _credential_loader.set_environment_variables()


def get_boto3_session(verbose=False):
    """
    Create boto3 session with loaded credentials.
    
    Args:
        verbose: Print detailed loading information
        
    Returns:
        boto3.Session: Configured session
    """
    import boto3
    
    boto3_kwargs = _credential_loader.get_boto3_kwargs()
    return boto3.Session(**boto3_kwargs)


def get_s3_config(environment=None):
    """
    Get S3 configuration (bucket, prefixes) for the specified environment.
    
    Args:
        environment: Environment name (development, staging, production). If None, uses default.
        
    Returns:
        dict: S3 configuration with bucket and prefix builder function
    """
    return _credential_loader.get_s3_configuration(environment)


def build_s3_path(base_prefix, requirement, environment=None):
    """
    Build complete S3 path using base prefix + requirement.
    
    Args:
        base_prefix: Base prefix name (knowledge_base, mvr_analysis, etc.)
        requirement: Specific requirement (checklist, raw, reports, etc.)
        environment: Environment name (optional)
        
    Returns:
        str: Complete S3 path (e.g., "dev/knowledge_base/checklist/")
        
    Examples:
        >>> build_s3_path("knowledge_base", "checklist")
        "knowledge_base/checklist/"
        
        >>> build_s3_path("mvr_analysis", "raw", "development") 
        "dev/mvr_analysis/raw/"
    """
    s3_config = get_s3_config(environment)
    return s3_config['build_path'](base_prefix, requirement)


def test_credentials():
    """Test credential loading and AWS connectivity"""
    
    print("=" * 60)
    print("CREDENTIAL LOADER TEST")
    print("=" * 60)
    
    # Test credential loading
    credentials = load_aws_credentials(verbose=True)
    print(f"\n[TEST] Loaded {len(credentials)} credential values:")
    
    for key in credentials.keys():
        if 'SECRET' in key:
            print(f"  {key}: ***masked***")
        else:
            print(f"  {key}: {credentials[key]}")
    
    # Test S3 configuration
    print(f"\n[TEST] S3 Configuration:")
    
    # Test default environment
    default_config = get_s3_config()
    print(f"  Default bucket: {default_config['bucket']}")
    print(f"  Environment: {default_config['environment']}")
    
    # Test path building
    print(f"\n[TEST] S3 Path Building:")
    test_paths = [
        ("knowledge_base", "checklist"),
        ("knowledge_base", "sop"),  
        ("mvr_analysis", "raw"),
        ("mvr_analysis", "reports"),
        ("embeddings", "tfidf")
    ]
    
    for base, requirement in test_paths:
        path = build_s3_path(base, requirement)
        print(f"  {base} + {requirement} = {path}")
    
    # Test environment-specific paths
    print(f"\n[TEST] Environment-Specific Paths:")
    for env in ["development", "staging", "production"]:
        try:
            path = build_s3_path("knowledge_base", "checklist", env)
            config = get_s3_config(env)
            print(f"  {env}: {config['bucket']}/{path}")
        except Exception as e:
            print(f"  {env}: [ERROR] {e}")
    
    # Test boto3 session
    try:
        session = get_boto3_session()
        s3 = session.client('s3')
        buckets = s3.list_buckets()['Buckets']
        print(f"\n[SUCCESS] S3 connectivity test passed - {len(buckets)} buckets accessible")
        return True
    except Exception as e:
        print(f"\n[ERROR] S3 connectivity test failed: {e}")
        return False


if __name__ == "__main__":
    test_credentials()