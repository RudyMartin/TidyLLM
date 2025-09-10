"""
Corporate Session Management Validator
=====================================

Handles the complexity of AWS connections in corporate environments where
the standard SessionManager won't work due to:
- Corporate proxies
- IAM role assumptions
- Custom SSL certificates
- Network restrictions
- Compliance requirements

This provides a dummy-proof way to establish AWS connectivity.
"""

import os
import sys
import boto3
import json
import logging
import socket
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class CorporateEnvironmentInfo:
    """Information about the corporate environment."""
    has_proxy: bool = False
    proxy_settings: Dict[str, str] = None
    ssl_verification: bool = True
    ca_bundle_path: Optional[str] = None
    network_restrictions: List[str] = None
    iam_role_available: bool = False
    current_aws_profile: Optional[str] = None


class CorporateSessionManager:
    """
    Corporate-friendly AWS session management.
    
    Handles the complexity of establishing AWS connections in corporate
    environments where standard approaches fail.
    """
    
    def __init__(self):
        self.environment_info = None
        self.session = None
        self.validated_services = {}
        
    def detect_corporate_environment(self) -> CorporateEnvironmentInfo:
        """
        Detect corporate environment settings.
        
        Returns information about proxy settings, SSL requirements,
        and network restrictions that might affect AWS connectivity.
        """
        info = CorporateEnvironmentInfo()
        
        # Check for proxy settings
        proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
        proxy_settings = {}
        for var in proxy_vars:
            if os.environ.get(var):
                proxy_settings[var.lower()] = os.environ[var]
                info.has_proxy = True
        
        info.proxy_settings = proxy_settings if proxy_settings else None
        
        # Check for custom CA bundle
        ca_vars = ['REQUESTS_CA_BUNDLE', 'CURL_CA_BUNDLE', 'SSL_CERT_FILE']
        for var in ca_vars:
            if os.environ.get(var):
                info.ca_bundle_path = os.environ[var]
                break
        
        # Check if running on EC2 (IAM role available)
        try:
            import requests
            # Quick check for EC2 metadata service (with short timeout)
            response = requests.get(
                'http://169.254.169.254/latest/meta-data/instance-id',
                timeout=2
            )
            if response.status_code == 200:
                info.iam_role_available = True
        except:
            info.iam_role_available = False
        
        # Check current AWS profile
        info.current_aws_profile = os.environ.get('AWS_PROFILE')
        
        self.environment_info = info
        return info
    
    def test_network_connectivity(self) -> Dict[str, bool]:
        """
        Test network connectivity to required AWS services.
        
        Returns a dict of service -> connectivity status.
        """
        required_endpoints = {
            'bedrock': ('bedrock-runtime.us-east-1.amazonaws.com', 443),
            's3': ('s3.amazonaws.com', 443),
            'sts': ('sts.amazonaws.com', 443),
        }
        
        connectivity = {}
        
        for service, (host, port) in required_endpoints.items():
            try:
                # Test socket connection with timeout
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10)  # 10 second timeout
                result = sock.connect_ex((host, port))
                sock.close()
                connectivity[service] = (result == 0)
            except Exception as e:
                logger.warning(f"Network test for {service} failed: {e}")
                connectivity[service] = False
        
        return connectivity
    
    def create_corporate_session(self, 
                                 access_key_id: Optional[str] = None,
                                 secret_access_key: Optional[str] = None,
                                 session_token: Optional[str] = None,
                                 region: str = 'us-east-1',
                                 profile: Optional[str] = None) -> boto3.Session:
        """
        Create AWS session with corporate environment considerations.
        
        This method tries multiple approaches to establish connectivity:
        1. Explicit credentials (if provided)
        2. Environment variables 
        3. IAM roles (if on EC2)
        4. AWS profiles
        5. Default credential chain
        """
        session_config = {
            'region_name': region
        }
        
        # Configure SSL/TLS settings for corporate environments
        if self.environment_info and self.environment_info.ca_bundle_path:
            os.environ['AWS_CA_BUNDLE'] = self.environment_info.ca_bundle_path
        
        # Method 1: Explicit credentials
        if access_key_id and secret_access_key:
            logger.info("Using explicit credentials")
            session_config.update({
                'aws_access_key_id': access_key_id,
                'aws_secret_access_key': secret_access_key,
            })
            if session_token:
                session_config['aws_session_token'] = session_token
        
        # Method 2: AWS Profile
        elif profile:
            logger.info(f"Using AWS profile: {profile}")
            session_config['profile_name'] = profile
        
        # Method 3: Environment variables (default)
        else:
            logger.info("Using environment variables or default credential chain")
        
        try:
            session = boto3.Session(**session_config)
            
            # Test the session by calling STS get-caller-identity
            sts = session.client('sts')
            identity = sts.get_caller_identity()
            
            logger.info(f"AWS session created successfully")
            logger.info(f"Account: {identity.get('Account')}")
            logger.info(f"User/Role: {identity.get('Arn')}")
            
            self.session = session
            return session
            
        except Exception as e:
            logger.error(f"Failed to create AWS session: {e}")
            raise
    
    def validate_bedrock_access(self) -> Tuple[bool, str]:
        """
        Test Bedrock service access.
        
        Returns (success, message).
        """
        if not self.session:
            return False, "No AWS session available"
        
        try:
            bedrock = self.session.client('bedrock')
            
            # Test by listing foundation models
            response = bedrock.list_foundation_models()
            model_count = len(response.get('modelSummaries', []))
            
            self.validated_services['bedrock'] = True
            return True, f"Bedrock access validated - {model_count} models available"
            
        except Exception as e:
            error_msg = str(e)
            
            # Provide helpful error messages for common corporate issues
            if 'AccessDenied' in error_msg:
                return False, "Access Denied - Check IAM permissions for bedrock:ListFoundationModels"
            elif 'UnauthorizedOperation' in error_msg:
                return False, "Unauthorized - Bedrock may not be enabled in this region"
            elif 'timeout' in error_msg.lower():
                return False, "Network timeout - Check corporate firewall/proxy settings"
            else:
                return False, f"Bedrock validation failed: {error_msg}"
    
    def validate_s3_access(self, bucket_name: Optional[str] = None) -> Tuple[bool, str]:
        """
        Test S3 service access.
        
        Returns (success, message).
        """
        if not self.session:
            return False, "No AWS session available"
        
        try:
            s3 = self.session.client('s3')
            
            if bucket_name:
                # Test specific bucket access
                try:
                    s3.head_bucket(Bucket=bucket_name)
                    # Try to list objects (limited to 1 for speed)
                    response = s3.list_objects_v2(Bucket=bucket_name, MaxKeys=1)
                    
                    self.validated_services['s3'] = True
                    return True, f"S3 bucket '{bucket_name}' access validated"
                    
                except s3.exceptions.NoSuchBucket:
                    return False, f"S3 bucket '{bucket_name}' does not exist"
                except Exception as e:
                    if 'AccessDenied' in str(e):
                        return False, f"Access denied to S3 bucket '{bucket_name}' - Check IAM permissions"
                    else:
                        raise
            else:
                # Test general S3 access by listing buckets
                response = s3.list_buckets()
                bucket_count = len(response.get('Buckets', []))
                
                self.validated_services['s3'] = True
                return True, f"S3 access validated - {bucket_count} buckets accessible"
                
        except Exception as e:
            error_msg = str(e)
            
            if 'AccessDenied' in error_msg:
                return False, "S3 Access Denied - Check IAM permissions for s3:ListBucket"
            elif 'timeout' in error_msg.lower():
                return False, "S3 Network timeout - Check corporate firewall/proxy settings"  
            else:
                return False, f"S3 validation failed: {error_msg}"
    
    def validate_postgresql_connection(self, 
                                       host: str,
                                       port: int,
                                       database: str, 
                                       username: str,
                                       password: str) -> Tuple[bool, str]:
        """
        Test PostgreSQL database connection.
        
        Returns (success, message).
        """
        try:
            import psycopg2
            
            # Test basic connectivity first
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10)
                result = sock.connect_ex((host, port))
                sock.close()
                
                if result != 0:
                    return False, f"Cannot reach PostgreSQL server {host}:{port} - Network connectivity issue"
            except Exception as e:
                return False, f"Network test to PostgreSQL failed: {e}"
            
            # Test database connection
            conn_string = f"host={host} port={port} dbname={database} user={username} password={password} connect_timeout=10"
            
            with psycopg2.connect(conn_string) as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT version();")
                    version = cursor.fetchone()[0]
                    
            self.validated_services['postgresql'] = True            
            return True, f"PostgreSQL connection validated - {version}"
            
        except ImportError:
            return False, "psycopg2 not installed - Run: pip install psycopg2-binary"
        except Exception as e:
            error_msg = str(e)
            
            if 'authentication failed' in error_msg.lower():
                return False, "PostgreSQL authentication failed - Check username/password"
            elif 'does not exist' in error_msg.lower():
                return False, f"PostgreSQL database '{database}' does not exist"
            elif 'timeout' in error_msg.lower():
                return False, "PostgreSQL connection timeout - Check network/firewall settings"
            else:
                return False, f"PostgreSQL validation failed: {error_msg}"
    
    def generate_environment_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive report of the corporate environment
        and validation results.
        """
        if not self.environment_info:
            self.detect_corporate_environment()
        
        connectivity = self.test_network_connectivity()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'environment': {
                'has_proxy': self.environment_info.has_proxy,
                'proxy_settings': self.environment_info.proxy_settings,
                'ca_bundle_path': self.environment_info.ca_bundle_path,
                'iam_role_available': self.environment_info.iam_role_available,
                'aws_profile': self.environment_info.current_aws_profile,
            },
            'network_connectivity': connectivity,
            'validated_services': self.validated_services.copy(),
            'aws_session_active': self.session is not None,
        }
        
        if self.session:
            try:
                sts = self.session.client('sts')
                identity = sts.get_caller_identity()
                report['aws_identity'] = {
                    'account': identity.get('Account'),
                    'user_arn': identity.get('Arn'),
                    'user_id': identity.get('UserId'),
                }
            except:
                pass
        
        return report


def validate_corporate_environment() -> Dict[str, Any]:
    """
    Convenience function to validate corporate environment.
    
    Returns a comprehensive validation report.
    """
    manager = CorporateSessionManager()
    manager.detect_corporate_environment()
    
    return manager.generate_environment_report()


def test_full_aws_stack(access_key_id: Optional[str] = None,
                        secret_access_key: Optional[str] = None,
                        session_token: Optional[str] = None,
                        region: str = 'us-east-1',
                        s3_bucket: Optional[str] = None,
                        postgres_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Test the full AWS stack required for TidyLLM in a corporate environment.
    
    This is the "dummy-proof" validation function.
    """
    manager = CorporateSessionManager()
    results = {
        'overall_success': False,
        'environment_detection': {},
        'aws_session': {},
        'service_validation': {},
        'recommendations': []
    }
    
    try:
        # Step 1: Detect environment
        env_info = manager.detect_corporate_environment()
        results['environment_detection'] = {
            'success': True,
            'has_proxy': env_info.has_proxy,
            'proxy_settings': env_info.proxy_settings,
            'iam_available': env_info.iam_role_available,
            'ca_bundle': env_info.ca_bundle_path,
        }
        
        # Step 2: Create AWS session
        try:
            session = manager.create_corporate_session(
                access_key_id=access_key_id,
                secret_access_key=secret_access_key,
                session_token=session_token,
                region=region
            )
            results['aws_session'] = {
                'success': True,
                'region': region,
                'message': 'AWS session created successfully'
            }
        except Exception as e:
            results['aws_session'] = {
                'success': False,
                'error': str(e),
                'message': 'Failed to create AWS session'
            }
            results['recommendations'].append("Check AWS credentials and network connectivity")
            return results
        
        # Step 3: Validate services
        service_results = {}
        
        # Test Bedrock
        bedrock_success, bedrock_msg = manager.validate_bedrock_access()
        service_results['bedrock'] = {
            'success': bedrock_success,
            'message': bedrock_msg
        }
        
        # Test S3
        s3_success, s3_msg = manager.validate_s3_access(s3_bucket)
        service_results['s3'] = {
            'success': s3_success, 
            'message': s3_msg
        }
        
        # Test PostgreSQL (if config provided)
        if postgres_config:
            pg_success, pg_msg = manager.validate_postgresql_connection(
                host=postgres_config['host'],
                port=postgres_config['port'], 
                database=postgres_config['database'],
                username=postgres_config['username'],
                password=postgres_config['password']
            )
            service_results['postgresql'] = {
                'success': pg_success,
                'message': pg_msg
            }
        
        results['service_validation'] = service_results
        
        # Determine overall success
        aws_ok = results['aws_session']['success']
        bedrock_ok = service_results['bedrock']['success']
        s3_ok = service_results['s3']['success']
        pg_ok = service_results.get('postgresql', {}).get('success', True)  # Optional
        
        results['overall_success'] = aws_ok and bedrock_ok and s3_ok and pg_ok
        
        # Generate recommendations
        if not bedrock_ok:
            results['recommendations'].append("Configure Bedrock permissions: bedrock:ListFoundationModels, bedrock:InvokeModel")
        if not s3_ok:
            results['recommendations'].append("Configure S3 permissions: s3:ListBucket, s3:GetObject, s3:PutObject")
        if postgres_config and not pg_ok:
            results['recommendations'].append("Check PostgreSQL connectivity and credentials")
        
        if env_info.has_proxy:
            results['recommendations'].append("Proxy detected - ensure AWS SDK proxy configuration is correct")
        
        return results
        
    except Exception as e:
        results['overall_success'] = False
        results['error'] = str(e)
        results['recommendations'].append("Contact system administrator for assistance")
        return results