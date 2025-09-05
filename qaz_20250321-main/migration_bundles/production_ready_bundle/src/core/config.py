"""
Global Configuration Management for VectorQA Sage

This module provides centralized configuration management for the VectorQA Sage application,
including LLM provider settings, AWS configurations, embedding models, and system defaults.

The configuration is designed to be environment-aware and supports multiple LLM providers
with intelligent fallback mechanisms. It also manages AWS service configurations for
vector storage and embedding generation.

TODO - Add environment-specific configuration loading (dev/staging/prod)
TODO - Add configuration validation and schema enforcement
TODO - Add configuration hot-reloading capabilities
TODO - Add configuration encryption for sensitive values
"""

import boto3
import os
# Import helper for robust cross-environment imports

# Robust import setup
import sys
from pathlib import Path
_src_dir = Path(__file__).parent
while _src_dir.name != "src" and _src_dir.parent != _src_dir:
    _src_dir = _src_dir.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

try:
    from utils.import_helper import import_manager
    credential_manager = import_manager.safe_import(
        "backend.config.credential_manager", 
        ["config.credential_manager"]
    ).credential_manager
except ImportError:
    # Final fallback
    try:
        from config.credential_manager import credential_manager
    except ImportError:
        from config.credential_manager import credential_manager

def load_security_config():
    """Load security configuration from environment variables."""
    
    security_config = {
        "aws_only": os.getenv("AWS_ONLY_MODE", "true").lower() == "true",
        "allow_external_apis": os.getenv("ALLOW_EXTERNAL_APIS", "false").lower() == "true",
        "require_iam_roles": os.getenv("REQUIRE_IAM_ROLES", "true").lower() == "true",
        "audit_logging": os.getenv("AUDIT_LOGGING", "true").lower() == "true"
    }
    
    return security_config

CONFIG = {
    "bucket_name": "sagemaker-us-east-1-XXXXXXXXXXXX",
    "prefix": "dev",
    "num_training_samples": 1000,
    "region_name": "us-east-1",
    "json_folder": "json",
    "index_folder": "idx",
    "page_folder": "page", 
    "pdf_folder": "pdf", 
    "compiled_folder": "compiled_modules",
    "nlist": 512,
    "nprobe": 16,
    "default_model": "amazon.titan-embed-text-v2:0",
    "embedding_models": {
        "amazon.titan-embed-text-v1": 768,
        "amazon.titan-embed-text-v2:0": 1024,
        "cohere.embed-english-v3": 1024,
        "anthropic.claude-v2": 1536
    },
    
    # Security Configuration
    "security": load_security_config(),
    "security_mode": {
        "aws_only": True,  # Restrict to AWS services only
        "allow_external_apis": False,  # Disable external API calls
        "require_iam_roles": True,  # Require IAM role authentication
        "audit_logging": True,  # Enable comprehensive logging
        "data_encryption": True  # Enable encryption at rest/transit
    },
    
    # AWS-Only Model Configuration
    "aws_models": {
        "enabled": True,
        "fallback_to_external": False,  # Never fallback to external APIs
        "preferred_providers": ["bedrock", "sagemaker", "lambda"]
    },
    
    # AWS Bedrock Models for QA Use Case
    "bedrock_models": {
        # Embedding Models
        "embedding": {
            "primary": "amazon.titan-embed-text-v2:0",
            "fallback": "amazon.titan-embed-text-v1",
            "dimensions": 1024
        },
        
        # Document Analysis Models
        "document_analysis": {
            "primary": "anthropic.claude-3-sonnet-20240229-v1:0",
            "fallback": "amazon.titan-text-express-v1",
            "max_tokens": 4096,
            "temperature": 0.1
        },
        
        # Report Generation Models
        "report_generation": {
            "primary": "anthropic.claude-3-opus-20240229-v1:0",
            "fallback": "anthropic.claude-3-sonnet-20240229-v1:0",
            "max_tokens": 4096,
            "temperature": 0.2
        },
        
        # Quick Processing Models
        "quick_processing": {
            "primary": "anthropic.claude-3-haiku-20240307-v1:0",
            "fallback": "amazon.titan-text-lite-v1",
            "max_tokens": 2048,
            "temperature": 0.1
        }
    },
    
    # LLM Configuration (AWS-Only)
    "default_llm": "bedrock_claude_sonnet",  # Changed to AWS Bedrock as primary
    "fallback_order": ["bedrock_claude_sonnet", "bedrock_claude_haiku", "bedrock_titan_text", "bedrock_titan_lite"],  # AWS-only priority order
    "llm_models": {
        # AWS Bedrock Models Only
        "bedrock_claude_sonnet": {
            "provider": "bedrock",
            "model": "anthropic.claude-3-sonnet-20240229-v1:0",
            "max_tokens": 4096,
            "temperature": 0.1,
            "aws_only": True
        },
        "bedrock_claude_opus": {
            "provider": "bedrock", 
            "model": "anthropic.claude-3-opus-20240229-v1:0",
            "max_tokens": 4096,
            "temperature": 0.2,
            "aws_only": True
        },
        "bedrock_claude_haiku": {
            "provider": "bedrock",
            "model": "anthropic.claude-3-haiku-20240307-v1:0", 
            "max_tokens": 2048,
            "temperature": 0.1,
            "aws_only": True
        },
        "bedrock_titan_text": {
            "provider": "bedrock",
            "model": "amazon.titan-text-express-v1",
            "max_tokens": 8192,
            "temperature": 0.1,
            "aws_only": True
        },
        "bedrock_titan_lite": {
            "provider": "bedrock",
            "model": "amazon.titan-text-lite-v1",
            "max_tokens": 4096,
            "temperature": 0.1,
            "aws_only": True
        }
    }
}

MODEL_OPTIONS = {
    "titan_v1": {"id": "amazon.titan-embed-text-v1", "dimensions": 768},
    "titan_v2": {"id": "amazon.titan-embed-text-v2:0", "dimensions": 1024},
    "cohere": {"id": "cohere.embed-english-v3", "dimensions": None},
    "anthropic": {"id": "anthropic.claude-v2", "dimensions": None}
}

# Centralized AWS Clients with region_name
try:
    s3_client = boto3.client("s3", region_name=CONFIG["region_name"])
    bedrock_client = boto3.client("bedrock-runtime", region_name=CONFIG["region_name"])
except Exception as e:
    print(f"Warning: Could not initialize AWS clients: {e}")
    print("AWS functionality will be limited. Please check your AWS credentials.")
    s3_client = None
    bedrock_client = None