"""
Configuration Generator for Corporate Deployments
================================================

Generates customized TidyLLM configuration files based on corporate
environment validation results and user preferences.
"""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

def create_template_config(user_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a customized settings.yaml based on user configuration.
    
    Args:
        user_config: User preferences from the onboarding wizard
    
    Returns:
        Complete configuration dictionary
    """
    
    # Start with base template
    config = {
        'environment': user_config.get('environment', 'corporate'),
        'organization': user_config.get('organization', 'Corporate'),
        'deployment_type': 'corporate',
        'generated_at': datetime.now().isoformat(),
    }
    
    # AWS Region
    aws_region = user_config.get('aws_region', 'us-east-1')
    
    # Database configuration (corporate defaults)
    config['postgres'] = {
        'host': 'YOUR_POSTGRES_HOST',
        'port': 5432,
        'db_name': 'tidyllm_production',
        'db_user': 'tidyllm_service',
        'db_password': 'USE_ENVIRONMENT_VARIABLE',  # TIDYLLM_DB_PASSWORD
        'ssl_mode': 'require',
        'connection_pool_size': 20,
        'max_retries': 5,
        'retry_delay': 2.0
    }
    
    # S3 configuration  
    config['s3'] = {
        'region': aws_region,
        'bucket': f"{user_config.get('organization', 'company').lower()}-tidyllm-storage",
        'prefix': 'tidyllm-data/',
        'connection_timeout': 60,
        'max_retries': 5
    }
    
    # AWS Bedrock configuration
    default_model = user_config.get('default_model', 'claude-3-sonnet')
    model_mapping = {
        'claude-3-sonnet': 'anthropic.claude-3-sonnet-20240229-v1:0',
        'claude-3-haiku': 'anthropic.claude-3-haiku-20240307-v1:0',
        'claude-3-opus': 'anthropic.claude-3-opus-20240229-v1:0',
        # Corporate aliases
        'corporate-llm': 'anthropic.claude-3-sonnet-20240229-v1:0',
        'quick-assist': 'anthropic.claude-3-haiku-20240307-v1:0',
    }
    
    config['aws'] = {
        'region': aws_region,
        'kms_key_id': 'YOUR_KMS_KEY_ARN',
        'bedrock': {
            'region': aws_region,
            'credentials': {
                'profile': None,
                'access_key_id': 'USE_ENVIRONMENT_VARIABLE',
                'secret_access_key': 'USE_ENVIRONMENT_VARIABLE',
                'session_token': 'USE_ENVIRONMENT_VARIABLE'
            },
            'default_model': model_mapping.get(default_model, model_mapping['claude-3-sonnet']),
            'model_mapping': model_mapping,
            'models': {
                'claude': {
                    model_mapping['claude-3-sonnet']: {
                        'max_tokens': 2048,
                        'temperature': 0.3,
                        'top_p': 0.9,
                        'stop_sequences': ['\n\nHuman:', '\n\nAssistant:']
                    },
                    model_mapping['claude-3-haiku']: {
                        'max_tokens': 2048,
                        'temperature': 0.3,
                        'top_p': 0.9,
                        'stop_sequences': ['\n\nHuman:', '\n\nAssistant:']
                    }
                }
            }
        }
    }
    
    # Security configuration (corporate requirements)
    security_config = user_config.get('security', {})
    config['security'] = {
        'data': {
            'encrypt_cache': security_config.get('encrypt_cache', True),
            'encrypt_logs': True,
            'mask_sensitive_data': True,
            'cache_retention_days': 7,
            'log_retention_days': 30
        },
        'access': {
            'require_auth': True,
            'auth_method': 'oauth2' if security_config.get('sso_enabled', True) else 'basic',
            'rate_limit': {
                'enabled': True,
                'requests_per_minute': 30,
                'requests_per_hour': 500
            }
        },
        'audit': {
            'enabled': security_config.get('audit_logging', True),
            'audit_all_requests': True,
            'audit_retention_days': 90,
            'audit_s3_bucket': f"{user_config.get('organization', 'company').lower()}-tidyllm-audit"
        }
    }
    
    # Logging configuration (corporate standards)
    config['logging'] = {
        'level': 'INFO',
        'format': 'json',
        'include_user_id': True,
        'include_session_id': True,
        'include_request_id': True,
        'handlers': {
            'file': {
                'enabled': True,
                'path': '/var/log/tidyllm/app.log',
                'max_size_mb': 100,
                'backup_count': 10
            },
            'syslog': {
                'enabled': True,
                'facility': 'local0',
                'address': ['YOUR_SYSLOG_SERVER', 514]
            },
            'cloudwatch': {
                'enabled': True,
                'log_group': 'tidyllm-corporate',
                'log_stream': 'application'
            }
        }
    }
    
    # Corporate network configuration
    config['network'] = {
        'proxy': {
            'enabled': False,  # Set to true if proxy detected
            'http_proxy': 'http://proxy.company.com:8080',
            'https_proxy': 'http://proxy.company.com:8080',
            'no_proxy': 'localhost,127.0.0.1,.company.com'
        },
        'timeouts': {
            'connect_timeout': 30,
            'read_timeout': 120,
            'total_timeout': 300
        },
        'ssl': {
            'verify': True,
            'ca_bundle': '/etc/ssl/certs/corporate-ca.pem'
        }
    }
    
    # Monitoring configuration (corporate requirements)
    config['monitoring'] = {
        'enabled': True,
        'health_check': {
            'enabled': True,
            'interval_seconds': 60,
            'endpoint': '/health'
        },
        'metrics': {
            'enabled': True,
            'export_to_prometheus': True,
            'prometheus_port': 9090
        },
        'alerts': {
            'enabled': True,
            'error_threshold': 10,
            'response_time_threshold': 5000,
            'notification_channels': [
                f"email:ops-team@{user_config.get('organization', 'company').lower()}.com",
                '#tidyllm-alerts'
            ]
        }
    }
    
    # Corporate integrations
    config['integrations'] = {
        'sso': {
            'enabled': security_config.get('sso_enabled', True),
            'provider': 'YOUR_SSO_PROVIDER',
            'client_id': 'YOUR_CLIENT_ID',
            'authority': 'YOUR_SSO_AUTHORITY'
        },
        'ldap': {
            'enabled': False,
            'server': 'ldap.company.com',
            'base_dn': f"ou=users,dc={user_config.get('organization', 'company').lower()},dc=com"
        }
    }
    
    # Add validation results if available
    if 'validation_results' in user_config:
        config['_deployment_validation'] = user_config['validation_results']
    
    # Add deployment notes
    config['deployment'] = {
        'notes': f"""
Corporate Deployment for {user_config.get('organization', 'Your Organization')}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

REQUIRED ENVIRONMENT VARIABLES:
- TIDYLLM_DB_PASSWORD: PostgreSQL database password
- AWS_ACCESS_KEY_ID: AWS access key (if not using IAM roles)
- AWS_SECRET_ACCESS_KEY: AWS secret key (if not using IAM roles)
- AWS_SESSION_TOKEN: AWS session token (if using temporary credentials)

NETWORK REQUIREMENTS:
- Outbound HTTPS to {aws_region} AWS endpoints
- Access to PostgreSQL database
- Access to S3 storage

SECURITY CHECKLIST:
- KMS encryption key configured
- Audit logging destination configured  
- SSL certificates installed
- Corporate CA certificates configured
""",
        'validation': {
            'required_env_vars': [
                'TIDYLLM_DB_PASSWORD',
                'AWS_ACCESS_KEY_ID',
                'AWS_SECRET_ACCESS_KEY'
            ],
            'required_network_access': [
                f'bedrock-runtime.{aws_region}.amazonaws.com:443',
                f's3.{aws_region}.amazonaws.com:443',
                'YOUR_POSTGRES_HOST:5432'
            ],
            'required_permissions': [
                'bedrock:InvokeModel',
                'bedrock:ListFoundationModels',
                's3:GetObject',
                's3:PutObject',
                's3:ListBucket',
                'kms:Encrypt',
                'kms:Decrypt'
            ]
        }
    }
    
    return config


def validate_aws_setup(access_key_id: Optional[str] = None,
                       secret_access_key: Optional[str] = None,
                       session_token: Optional[str] = None,
                       region: str = 'us-east-1') -> Dict[str, Any]:
    """
    Validate AWS setup for corporate deployment.
    
    Returns validation results and recommendations.
    """
    from .session_validator import test_full_aws_stack
    
    return test_full_aws_stack(
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        session_token=session_token,
        region=region
    )


def generate_dockerfile(config: Dict[str, Any]) -> str:
    """Generate Dockerfile for corporate deployment."""
    
    org_name = config.get('organization', 'company').lower()
    
    dockerfile = f"""# TidyLLM Corporate Deployment
# Organization: {config.get('organization', 'Corporate')}
# Generated: {datetime.now().strftime('%Y-%m-%d')}

FROM python:3.11-slim

# Corporate security requirements
RUN apt-get update && apt-get install -y \\
    ca-certificates \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy corporate CA certificates (if needed)
# COPY corporate-ca.crt /usr/local/share/ca-certificates/
# RUN update-ca-certificates

# Create application user
RUN groupadd -r tidyllm && useradd -r -g tidyllm tidyllm

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy configuration
COPY settings.yaml /app/

# Create log directory
RUN mkdir -p /var/log/tidyllm && chown tidyllm:tidyllm /var/log/tidyllm

# Switch to non-root user
USER tidyllm

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Start application
CMD ["uvicorn", "tidyllm.api:app", "--host", "0.0.0.0", "--port", "8000"]
"""
    
    return dockerfile


def generate_kubernetes_manifests(config: Dict[str, Any]) -> Dict[str, str]:
    """Generate Kubernetes deployment manifests."""
    
    org_name = config.get('organization', 'company').lower()
    
    # Deployment manifest
    deployment = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: tidyllm-corporate
  namespace: {org_name}
  labels:
    app: tidyllm
    organization: {org_name}
spec:
  replicas: 3
  selector:
    matchLabels:
      app: tidyllm
  template:
    metadata:
      labels:
        app: tidyllm
    spec:
      containers:
      - name: tidyllm
        image: {org_name}/tidyllm:latest
        ports:
        - containerPort: 8000
        env:
        - name: AWS_DEFAULT_REGION
          value: "{config.get('aws', {}).get('region', 'us-east-1')}"
        - name: TIDYLLM_DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: tidyllm-secrets
              key: db-password
        - name: AWS_ACCESS_KEY_ID
          valueFrom:
            secretKeyRef:
              name: tidyllm-secrets
              key: aws-access-key
        - name: AWS_SECRET_ACCESS_KEY
          valueFrom:
            secretKeyRef:
              name: tidyllm-secrets
              key: aws-secret-key
        volumeMounts:
        - name: config
          mountPath: /app/settings.yaml
          subPath: settings.yaml
        - name: logs
          mountPath: /var/log/tidyllm
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          limits:
            cpu: 1000m
            memory: 2Gi
          requests:
            cpu: 500m
            memory: 1Gi
      volumes:
      - name: config
        configMap:
          name: tidyllm-config
      - name: logs
        emptyDir: {{}}
---
apiVersion: v1
kind: Service
metadata:
  name: tidyllm-service
  namespace: {org_name}
spec:
  selector:
    app: tidyllm
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: tidyllm-config
  namespace: {org_name}
data:
  settings.yaml: |
{yaml.dump(config, default_flow_style=False, indent=4)}
"""
    
    # Service manifest
    service = f"""apiVersion: v1
kind: Service
metadata:
  name: tidyllm-service
  namespace: {org_name}
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
spec:
  type: LoadBalancer
  ports:
  - port: 443
    targetPort: 8000
    protocol: TCP
  selector:
    app: tidyllm
"""
    
    return {
        'deployment.yaml': deployment,
        'service.yaml': service
    }


def generate_docker_compose(config: Dict[str, Any]) -> str:
    """Generate Docker Compose file for corporate deployment."""
    
    org_name = config.get('organization', 'company').lower()
    aws_region = config.get('aws', {}).get('region', 'us-east-1')
    
    compose = f"""version: '3.8'

services:
  tidyllm:
    build: .
    image: {org_name}/tidyllm:latest
    container_name: tidyllm-corporate
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - AWS_DEFAULT_REGION={aws_region}
      - TIDYLLM_DB_PASSWORD=${{TIDYLLM_DB_PASSWORD}}
      - AWS_ACCESS_KEY_ID=${{AWS_ACCESS_KEY_ID}}
      - AWS_SECRET_ACCESS_KEY=${{AWS_SECRET_ACCESS_KEY}}
      - AWS_SESSION_TOKEN=${{AWS_SESSION_TOKEN}}
    volumes:
      - ./settings.yaml:/app/settings.yaml:ro
      - tidyllm-logs:/var/log/tidyllm
    networks:
      - tidyllm-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G

volumes:
  tidyllm-logs:

networks:
  tidyllm-network:
    driver: bridge
"""
    
    return compose