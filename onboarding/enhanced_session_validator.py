"""
Enhanced Corporate Session Management Validator
===============================================

Universal connectivity validator that works with:
- Standard AWS credentials (access key/secret)
- SSO/SAML temporary credentials  
- IAM roles and assume role chains
- Corporate proxy environments
- PostgreSQL databases
- MLflow tracking servers
- Custom authentication flows

Based on the proven pre-flight testing framework but enhanced for
corporate environments with SSO and temporary credential management.
"""

import os
import sys
import boto3
import json
import logging
import socket
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
import subprocess
import time

logger = logging.getLogger(__name__)

@dataclass
class CredentialInfo:
    """Information about available credentials."""
    source: str  # 'environment', 'profile', 'sso', 'iam_role', 'assume_role'
    account_id: Optional[str] = None
    user_arn: Optional[str] = None
    expiration: Optional[datetime] = None
    region: Optional[str] = None
    profile_name: Optional[str] = None
    role_arn: Optional[str] = None
    session_token: Optional[str] = None

@dataclass
class ServiceValidationResult:
    """Result of service connectivity validation."""
    service: str
    success: bool
    message: str
    details: Dict[str, Any] = None
    latency_ms: Optional[float] = None
    warnings: List[str] = None

class EnhancedSessionValidator:
    """
    Enhanced corporate session validator with SSO and temporary credential support.
    
    This validator can handle multiple authentication scenarios:
    1. Direct AWS credentials (access key/secret)
    2. SSO temporary credentials with expiration
    3. IAM roles and assume role chains
    4. AWS profiles with different credential sources
    5. Corporate proxy configurations
    6. Database connections (PostgreSQL)
    7. MLflow tracking server connections
    """
    
    def __init__(self, region: str = 'us-east-1', profile: Optional[str] = None):
        self.region = region
        self.profile = profile
        self.session = None
        self.credential_info = None
        self.validation_results = []
        self.environment_info = {}
        
    def detect_corporate_environment(self) -> Dict[str, Any]:
        """
        Detect corporate environment settings including proxy, SSO, and network restrictions.
        """
        env_info = {
            'proxy_detected': False,
            'proxy_settings': {},
            'ca_bundle_path': None,
            'sso_configured': False,
            'iam_role_available': False,
            'aws_profiles': [],
            'network_restrictions': []
        }
        
        # Check for proxy settings
        proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY']
        for var in proxy_vars:
            if os.environ.get(var):
                env_info['proxy_detected'] = True
                env_info['proxy_settings'][var] = os.environ[var]
        
        # Check for custom CA bundle
        ca_vars = ['REQUESTS_CA_BUNDLE', 'CURL_CA_BUNDLE', 'SSL_CERT_FILE', 'AWS_CA_BUNDLE']
        for var in ca_vars:
            if os.environ.get(var):
                env_info['ca_bundle_path'] = os.environ[var]
                break
        
        # Check for SSO configuration
        sso_vars = ['AWS_SSO_START_URL', 'AWS_SSO_REGION', 'AWS_SSO_ROLE_NAME']
        if any(os.environ.get(var) for var in sso_vars):
            env_info['sso_configured'] = True
        
        # Check for IAM role (EC2 metadata service)
        try:
            import requests
            response = requests.get(
                'http://169.254.169.254/latest/meta-data/instance-id',
                timeout=2
            )
            if response.status_code == 200:
                env_info['iam_role_available'] = True
        except:
            env_info['iam_role_available'] = False
        
        # Check AWS profiles
        aws_config_file = Path.home() / '.aws' / 'config'
        aws_credentials_file = Path.home() / '.aws' / 'credentials'
        
        profiles = set()
        for config_file in [aws_config_file, aws_credentials_file]:
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        for line in f:
                            if line.strip().startswith('[') and line.strip().endswith(']'):
                                profile_name = line.strip()[1:-1]
                                if profile_name not in ['default']:
                                    profiles.add(profile_name.replace('profile ', ''))
                except:
                    pass
        
        env_info['aws_profiles'] = list(profiles)
        
        self.environment_info = env_info
        return env_info
    
    def discover_credentials(self) -> List[CredentialInfo]:
        """
        Discover all available credential sources in corporate environment.
        
        Returns list of credential sources in priority order.
        """
        credentials = []
        
        # 1. Check explicit environment variables (highest priority)
        if os.environ.get('AWS_ACCESS_KEY_ID') and os.environ.get('AWS_SECRET_ACCESS_KEY'):
            cred = CredentialInfo(
                source='environment',
                region=os.environ.get('AWS_DEFAULT_REGION', self.region),
                session_token=os.environ.get('AWS_SESSION_TOKEN')
            )
            credentials.append(cred)
        
        # 2. Check for SSO credentials via AWS CLI
        if self.environment_info.get('sso_configured'):
            try:
                result = subprocess.run(
                    ['aws', 'sts', 'get-caller-identity'],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    cred = CredentialInfo(source='sso', region=self.region)
                    credentials.append(cred)
            except:
                pass
        
        # 3. Check AWS profiles
        for profile_name in self.environment_info.get('aws_profiles', []):
            try:
                session = boto3.Session(profile_name=profile_name, region_name=self.region)
                sts = session.client('sts')
                identity = sts.get_caller_identity()
                
                cred = CredentialInfo(
                    source='profile',
                    profile_name=profile_name,
                    account_id=identity.get('Account'),
                    user_arn=identity.get('Arn'),
                    region=self.region
                )
                credentials.append(cred)
            except:
                pass
        
        # 4. Check IAM role (EC2/ECS/Lambda)
        if self.environment_info.get('iam_role_available'):
            try:
                session = boto3.Session(region_name=self.region)
                sts = session.client('sts')
                identity = sts.get_caller_identity()
                
                cred = CredentialInfo(
                    source='iam_role',
                    account_id=identity.get('Account'),
                    user_arn=identity.get('Arn'),
                    region=self.region
                )
                credentials.append(cred)
            except:
                pass
        
        return credentials
    
    def create_session_with_credentials(self, credential_info: CredentialInfo) -> boto3.Session:
        """
        Create AWS session using specific credential source.
        """
        session_config = {'region_name': self.region}
        
        if credential_info.source == 'environment':
            # Use environment variables
            pass  # boto3 will pick these up automatically
            
        elif credential_info.source == 'profile':
            session_config['profile_name'] = credential_info.profile_name
            
        elif credential_info.source == 'sso':
            # For SSO, we rely on AWS CLI to have done the authentication
            pass  # boto3 will use the cached SSO credentials
            
        # Configure SSL/TLS for corporate environments
        if self.environment_info.get('ca_bundle_path'):
            os.environ['AWS_CA_BUNDLE'] = self.environment_info['ca_bundle_path']
        
        session = boto3.Session(**session_config)
        
        # Test the session and get identity info
        sts = session.client('sts')
        identity = sts.get_caller_identity()
        
        # Update credential info with actual details
        credential_info.account_id = identity.get('Account')
        credential_info.user_arn = identity.get('Arn')
        
        # Check for temporary credentials (session token)
        if credential_info.session_token or 'assumed-role' in identity.get('Arn', ''):
            # For temporary credentials, try to determine expiration
            try:
                # This is a heuristic - temporary credentials usually expire within 1-12 hours
                credential_info.expiration = datetime.now() + timedelta(hours=1)
            except:
                pass
        
        self.session = session
        self.credential_info = credential_info
        
        return session
    
    def validate_aws_connectivity(self) -> ServiceValidationResult:
        """
        Test basic AWS connectivity and permissions.
        """
        start_time = time.time()
        
        if not self.session:
            return ServiceValidationResult(
                service='aws',
                success=False,
                message='No AWS session available'
            )
        
        try:
            # Test STS (basic AWS connectivity)
            sts = self.session.client('sts')
            identity = sts.get_caller_identity()
            
            latency = (time.time() - start_time) * 1000
            
            return ServiceValidationResult(
                service='aws',
                success=True,
                message=f"AWS connectivity verified - Account: {identity['Account']}",
                details={
                    'account_id': identity['Account'],
                    'user_arn': identity['Arn'],
                    'user_id': identity['UserId'],
                    'credential_source': self.credential_info.source if self.credential_info else 'unknown'
                },
                latency_ms=latency
            )
            
        except Exception as e:
            return ServiceValidationResult(
                service='aws',
                success=False,
                message=f"AWS connectivity failed: {str(e)}"
            )
    
    def validate_s3_connectivity(self, bucket_name: Optional[str] = None) -> ServiceValidationResult:
        """
        Test S3 connectivity and permissions.
        """
        start_time = time.time()
        
        try:
            s3 = self.session.client('s3')
            
            if bucket_name:
                # Test specific bucket
                s3.head_bucket(Bucket=bucket_name)
                s3.list_objects_v2(Bucket=bucket_name, MaxKeys=1)
                message = f"S3 bucket '{bucket_name}' access verified"
            else:
                # Test general S3 access
                response = s3.list_buckets()
                bucket_count = len(response.get('Buckets', []))
                message = f"S3 access verified - {bucket_count} buckets accessible"
            
            latency = (time.time() - start_time) * 1000
            
            return ServiceValidationResult(
                service='s3',
                success=True,
                message=message,
                latency_ms=latency
            )
            
        except Exception as e:
            error_msg = str(e)
            if 'AccessDenied' in error_msg:
                message = "S3 access denied - check IAM permissions for s3:ListBucket, s3:GetObject"
            elif 'timeout' in error_msg.lower():
                message = "S3 timeout - check corporate firewall/proxy settings"
            else:
                message = f"S3 validation failed: {error_msg}"
                
            return ServiceValidationResult(
                service='s3',
                success=False,
                message=message
            )
    
    def validate_bedrock_connectivity(self, test_model: bool = False) -> ServiceValidationResult:
        """
        Test Bedrock connectivity and model access.
        """
        start_time = time.time()
        
        try:
            # Test Bedrock service access
            bedrock = self.session.client('bedrock', region_name=self.region)
            
            # List available models
            response = bedrock.list_foundation_models()
            models = response.get('modelSummaries', [])
            
            # Filter for Claude models (most relevant for TidyLLM)
            claude_models = [
                model for model in models 
                if 'claude' in model.get('modelId', '').lower()
            ]
            
            details = {
                'total_models': len(models),
                'claude_models': len(claude_models),
                'available_models': [model['modelId'] for model in claude_models[:5]]  # First 5
            }
            
            warnings = []
            
            # Optional: Test model invocation (requires bedrock:InvokeModel permission)
            if test_model and claude_models:
                try:
                    bedrock_runtime = self.session.client('bedrock-runtime', region_name=self.region)
                    
                    test_payload = {
                        "prompt": "\n\nHuman: Hello\n\nAssistant:",
                        "max_tokens_to_sample": 10,
                        "temperature": 0.1
                    }
                    
                    response = bedrock_runtime.invoke_model(
                        body=json.dumps(test_payload),
                        modelId=claude_models[0]['modelId'],
                        accept='application/json',
                        contentType='application/json'
                    )
                    
                    details['model_test'] = 'success'
                    
                except Exception as model_e:
                    warnings.append(f"Model invocation test failed: {str(model_e)}")
                    details['model_test'] = 'failed'
            
            latency = (time.time() - start_time) * 1000
            
            message = f"Bedrock access verified - {len(models)} models available"
            if claude_models:
                message += f" ({len(claude_models)} Claude models)"
            
            return ServiceValidationResult(
                service='bedrock',
                success=True,
                message=message,
                details=details,
                latency_ms=latency,
                warnings=warnings
            )
            
        except Exception as e:
            error_msg = str(e)
            if 'AccessDenied' in error_msg:
                message = "Bedrock access denied - check IAM permissions for bedrock:ListFoundationModels"
            elif 'UnauthorizedOperation' in error_msg:
                message = f"Bedrock not available in region {self.region} or not enabled"
            else:
                message = f"Bedrock validation failed: {error_msg}"
                
            return ServiceValidationResult(
                service='bedrock',
                success=False,
                message=message
            )
    
    def validate_postgresql_connectivity(self, 
                                       host: str,
                                       port: int,
                                       database: str, 
                                       username: str,
                                       password: str,
                                       ssl_mode: str = 'require') -> ServiceValidationResult:
        """
        Test PostgreSQL database connectivity.
        """
        start_time = time.time()
        
        try:
            import psycopg2
            import psycopg2.extras
            
            # First test network connectivity
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10)
                result = sock.connect_ex((host, port))
                sock.close()
                
                if result != 0:
                    return ServiceValidationResult(
                        service='postgresql',
                        success=False,
                        message=f"Cannot reach PostgreSQL server {host}:{port} - network connectivity issue"
                    )
            except Exception as e:
                return ServiceValidationResult(
                    service='postgresql',
                    success=False,
                    message=f"Network test failed: {str(e)}"
                )
            
            # Test database connection
            conn_params = {
                'host': host,
                'port': port,
                'dbname': database,
                'user': username,
                'password': password,
                'sslmode': ssl_mode,
                'connect_timeout': 10
            }
            
            with psycopg2.connect(**conn_params) as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    # Get basic database info
                    cursor.execute('SELECT version(), current_database(), current_user, inet_server_addr(), inet_server_port()')
                    db_info = cursor.fetchone()
                    
                    # Get table count
                    cursor.execute("SELECT count(*) as table_count FROM information_schema.tables WHERE table_schema = 'public'")
                    table_info = cursor.fetchone()
                    
                    # Test if MLflow tables exist (common in TidyLLM setups)
                    cursor.execute("""
                        SELECT count(*) as mlflow_tables 
                        FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name LIKE '%experiment%' OR table_name LIKE '%run%' OR table_name LIKE '%metric%'
                    """)
                    mlflow_info = cursor.fetchone()
                    
            latency = (time.time() - start_time) * 1000
            
            details = {
                'version': db_info['version'].split(' ')[1] if db_info['version'] else 'unknown',
                'database': db_info['current_database'],
                'user': db_info['current_user'],
                'server_ip': db_info['inet_server_addr'],
                'server_port': db_info['inet_server_port'],
                'table_count': table_info['table_count'],
                'mlflow_tables': mlflow_info['mlflow_tables'],
                'ssl_mode': ssl_mode
            }
            
            warnings = []
            if table_info['table_count'] == 0:
                warnings.append("Database has no tables - may need initialization")
            if mlflow_info['mlflow_tables'] == 0:
                warnings.append("No MLflow tables detected - MLflow may not be configured")
            
            return ServiceValidationResult(
                service='postgresql',
                success=True,
                message=f"PostgreSQL connectivity verified - {db_info['version'].split()[0]} {db_info['version'].split()[1]}",
                details=details,
                latency_ms=latency,
                warnings=warnings
            )
            
        except ImportError:
            return ServiceValidationResult(
                service='postgresql',
                success=False,
                message="psycopg2 not installed - run: pip install psycopg2-binary"
            )
        except Exception as e:
            error_msg = str(e)
            
            if 'authentication failed' in error_msg.lower():
                message = "PostgreSQL authentication failed - check username/password"
            elif 'does not exist' in error_msg.lower():
                message = f"PostgreSQL database '{database}' does not exist"
            elif 'timeout' in error_msg.lower():
                message = "PostgreSQL connection timeout - check network/firewall settings"
            elif 'ssl' in error_msg.lower():
                message = f"PostgreSQL SSL error - check SSL configuration (current: {ssl_mode})"
            else:
                message = f"PostgreSQL validation failed: {error_msg}"
                
            return ServiceValidationResult(
                service='postgresql',
                success=False,
                message=message
            )
    
    def validate_mlflow_connectivity(self, tracking_uri: str) -> ServiceValidationResult:
        """
        Test MLflow tracking server connectivity.
        """
        start_time = time.time()
        
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
            
            # Set tracking URI
            mlflow.set_tracking_uri(tracking_uri)
            client = MlflowClient(tracking_uri)
            
            # Test basic operations
            experiments = client.search_experiments()
            
            # Try to get server info if available
            server_info = {}
            try:
                # This may not be available in all MLflow versions
                server_info = client.get_registry_uri()
            except:
                pass
            
            latency = (time.time() - start_time) * 1000
            
            details = {
                'tracking_uri': tracking_uri,
                'experiment_count': len(experiments),
                'experiments': [exp.name for exp in experiments[:5]],  # First 5
                'server_info': server_info
            }
            
            warnings = []
            if len(experiments) == 0:
                warnings.append("No experiments found - MLflow may be newly initialized")
            
            parsed_uri = urlparse(tracking_uri)
            if parsed_uri.scheme == 'http':
                warnings.append("Using HTTP (not HTTPS) - consider enabling SSL for production")
            
            return ServiceValidationResult(
                service='mlflow',
                success=True,
                message=f"MLflow connectivity verified - {len(experiments)} experiments found",
                details=details,
                latency_ms=latency,
                warnings=warnings
            )
            
        except ImportError:
            return ServiceValidationResult(
                service='mlflow',
                success=False,
                message="mlflow not installed - run: pip install mlflow"
            )
        except Exception as e:
            error_msg = str(e)
            
            if 'connection' in error_msg.lower():
                message = f"Cannot connect to MLflow server at {tracking_uri}"
            elif 'timeout' in error_msg.lower():
                message = f"MLflow server timeout at {tracking_uri}"
            elif 'unauthorized' in error_msg.lower():
                message = f"MLflow server authentication required at {tracking_uri}"
            else:
                message = f"MLflow validation failed: {error_msg}"
                
            return ServiceValidationResult(
                service='mlflow',
                success=False,
                message=message
            )
    
    def run_comprehensive_validation(self, 
                                   s3_bucket: Optional[str] = None,
                                   test_bedrock_models: bool = False,
                                   postgres_config: Optional[Dict[str, Any]] = None,
                                   mlflow_uri: Optional[str] = None) -> Dict[str, Any]:
        """
        Run comprehensive validation of all services in corporate environment.
        
        This is the main entry point for validating a complete TidyLLM stack.
        """
        validation_start = datetime.now()
        
        # Step 1: Detect corporate environment
        env_info = self.detect_corporate_environment()
        
        # Step 2: Discover available credentials
        available_credentials = self.discover_credentials()
        
        if not available_credentials:
            return {
                'overall_success': False,
                'error': 'No AWS credentials found',
                'environment_info': env_info,
                'recommendations': [
                    'Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables',
                    'Configure AWS CLI with "aws configure"',
                    'Set up AWS SSO with "aws sso login"',
                    'Ensure IAM role is attached if running on EC2/ECS'
                ]
            }
        
        # Step 3: Try to establish session with best available credentials
        session_established = False
        session_error = None
        
        for cred in available_credentials:
            try:
                self.create_session_with_credentials(cred)
                session_established = True
                break
            except Exception as e:
                session_error = str(e)
                continue
        
        if not session_established:
            return {
                'overall_success': False,
                'error': f'Failed to establish AWS session: {session_error}',
                'environment_info': env_info,
                'available_credentials': [cred.source for cred in available_credentials],
                'recommendations': [
                    'Check AWS credentials are valid and not expired',
                    'Verify network connectivity to AWS endpoints',
                    'Check corporate proxy configuration if applicable'
                ]
            }
        
        # Step 4: Run service validations
        validations = []
        
        # AWS basic connectivity
        aws_result = self.validate_aws_connectivity()
        validations.append(aws_result)
        
        # S3 validation
        s3_result = self.validate_s3_connectivity(s3_bucket)
        validations.append(s3_result)
        
        # Bedrock validation
        bedrock_result = self.validate_bedrock_connectivity(test_bedrock_models)
        validations.append(bedrock_result)
        
        # PostgreSQL validation (if configured)
        if postgres_config:
            pg_result = self.validate_postgresql_connectivity(**postgres_config)
            validations.append(pg_result)
        
        # MLflow validation (if configured)
        if mlflow_uri:
            mlflow_result = self.validate_mlflow_connectivity(mlflow_uri)
            validations.append(mlflow_result)
        
        # Step 5: Compile results
        successful_validations = [v for v in validations if v.success]
        failed_validations = [v for v in validations if not v.success]
        
        all_warnings = []
        for validation in validations:
            if validation.warnings:
                all_warnings.extend(validation.warnings)
        
        # Determine overall success
        critical_services = ['aws', 's3', 'bedrock']
        critical_failures = [v for v in failed_validations if v.service in critical_services]
        
        overall_success = len(critical_failures) == 0
        
        # Generate recommendations
        recommendations = []
        for validation in failed_validations:
            if 'permission' in validation.message.lower():
                recommendations.append(f"Fix {validation.service.upper()} IAM permissions")
            elif 'timeout' in validation.message.lower():
                recommendations.append(f"Check {validation.service.upper()} network connectivity")
            elif 'not installed' in validation.message.lower():
                recommendations.append(f"Install required Python package for {validation.service}")
        
        if env_info['proxy_detected']:
            recommendations.append("Verify AWS SDK proxy configuration for corporate environment")
        
        if self.credential_info and self.credential_info.expiration:
            time_until_expiry = self.credential_info.expiration - datetime.now()
            if time_until_expiry.total_seconds() < 3600:  # Less than 1 hour
                recommendations.append("AWS credentials expire soon - refresh temporary credentials")
        
        # Compile final result
        result = {
            'overall_success': overall_success,
            'validation_timestamp': validation_start.isoformat(),
            'validation_duration_ms': (datetime.now() - validation_start).total_seconds() * 1000,
            'environment_info': env_info,
            'credential_info': {
                'source': self.credential_info.source,
                'account_id': self.credential_info.account_id,
                'user_arn': self.credential_info.user_arn,
                'region': self.credential_info.region,
                'expires': self.credential_info.expiration.isoformat() if self.credential_info.expiration else None
            } if self.credential_info else None,
            'service_validations': {
                v.service: {
                    'success': v.success,
                    'message': v.message,
                    'details': v.details,
                    'latency_ms': v.latency_ms,
                    'warnings': v.warnings
                } for v in validations
            },
            'summary': {
                'total_services_tested': len(validations),
                'successful_services': len(successful_validations),
                'failed_services': len(failed_validations),
                'critical_failures': len(critical_failures)
            },
            'warnings': all_warnings,
            'recommendations': recommendations
        }
        
        return result

# Convenience functions for easy integration

def validate_corporate_aws_stack(region: str = 'us-east-1',
                                 s3_bucket: Optional[str] = None,
                                 postgres_host: Optional[str] = None,
                                 postgres_port: int = 5432,
                                 postgres_database: str = 'tidyllm',
                                 postgres_user: Optional[str] = None,
                                 postgres_password: Optional[str] = None,
                                 mlflow_uri: Optional[str] = None,
                                 test_bedrock_models: bool = False) -> Dict[str, Any]:
    """
    One-function validation for complete TidyLLM corporate stack.
    
    This is the main function to call from onboarding wizards.
    """
    validator = EnhancedSessionValidator(region=region)
    
    postgres_config = None
    if postgres_host and postgres_user and postgres_password:
        postgres_config = {
            'host': postgres_host,
            'port': postgres_port,
            'database': postgres_database,
            'username': postgres_user,
            'password': postgres_password,
            'ssl_mode': 'require'  # Corporate default
        }
    
    return validator.run_comprehensive_validation(
        s3_bucket=s3_bucket,
        test_bedrock_models=test_bedrock_models,
        postgres_config=postgres_config,
        mlflow_uri=mlflow_uri
    )

def quick_aws_validation(region: str = 'us-east-1') -> Dict[str, Any]:
    """
    Quick validation for just AWS services (for fast pre-flight checks).
    """
    return validate_corporate_aws_stack(region=region)

def print_validation_report(validation_result: Dict[str, Any]):
    """
    Print a nicely formatted validation report.
    """
    print("\n" + "="*70)
    print("TIDYLLM CORPORATE ENVIRONMENT VALIDATION REPORT")
    print("="*70)
    
    # Overall status
    status = "✅ PASS" if validation_result['overall_success'] else "❌ FAIL"
    print(f"Overall Status: {status}")
    print(f"Validation Time: {validation_result['validation_timestamp']}")
    print(f"Duration: {validation_result['validation_duration_ms']:.1f}ms")
    
    # Environment info
    env_info = validation_result.get('environment_info', {})
    if env_info:
        print("\n" + "-"*50)
        print("CORPORATE ENVIRONMENT")
        print("-"*50)
        if env_info.get('proxy_detected'):
            print("🌐 Corporate proxy detected")
        if env_info.get('sso_configured'):
            print("🔐 SSO configuration detected")
        if env_info.get('iam_role_available'):
            print("🏷️ IAM role available")
        if env_info.get('ca_bundle_path'):
            print(f"🔒 Custom CA bundle: {env_info['ca_bundle_path']}")
    
    # Credential info
    cred_info = validation_result.get('credential_info')
    if cred_info:
        print("\n" + "-"*50)
        print("AWS CREDENTIALS")
        print("-"*50)
        print(f"Source: {cred_info['source']}")
        print(f"Account: {cred_info['account_id']}")
        print(f"Region: {cred_info['region']}")
        if cred_info.get('expires'):
            print(f"Expires: {cred_info['expires']}")
    
    # Service validations
    services = validation_result.get('service_validations', {})
    if services:
        print("\n" + "-"*50)
        print("SERVICE VALIDATION RESULTS")
        print("-"*50)
        
        for service, result in services.items():
            status_icon = "✅" if result['success'] else "❌"
            latency = f" ({result['latency_ms']:.1f}ms)" if result.get('latency_ms') else ""
            print(f"{status_icon} {service.upper()}: {result['message']}{latency}")
            
            if result.get('warnings'):
                for warning in result['warnings']:
                    print(f"   ⚠️ {warning}")
    
    # Summary
    summary = validation_result.get('summary', {})
    print("\n" + "-"*50)
    print("SUMMARY")
    print("-"*50)
    print(f"Services Tested: {summary.get('total_services_tested', 0)}")
    print(f"Successful: {summary.get('successful_services', 0)}")
    print(f"Failed: {summary.get('failed_services', 0)}")
    
    # Warnings
    warnings = validation_result.get('warnings', [])
    if warnings:
        print("\n⚠️ WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
    
    # Recommendations
    recommendations = validation_result.get('recommendations', [])
    if recommendations:
        print("\n💡 RECOMMENDATIONS:")
        for rec in recommendations:
            print(f"  - {rec}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    """
    Example usage of the enhanced session validator.
    """
    print("TidyLLM Enhanced Corporate Environment Validator")
    print("=" * 60)
    
    # Run quick validation
    result = quick_aws_validation()
    
    # Print detailed report
    print_validation_report(result)
    
    # Exit with appropriate code
    sys.exit(0 if result['overall_success'] else 1)