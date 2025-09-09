"""

# S3 Configuration Management
sys.path.append(str(Path(__file__).parent.parent / 'tidyllm' / 'admin') if 'tidyllm' in str(Path(__file__)) else str(Path(__file__).parent / 'tidyllm' / 'admin'))
from credential_loader import get_s3_config, build_s3_path

# Get S3 configuration (bucket and path builder)
s3_config = get_s3_config()  # Add environment parameter for dev/staging/prod

Workflow Configuration Management
================================

Provides consistent configuration loading with override capability.
- Loads defaults from settings.yaml
- Allows workflow-specific overrides
- Ensures consistency across all workflows
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class S3Config:
    """S3 configuration with override capability"""
    bucket: str
    region: str
    prefix: str = ""
    connection_timeout: int = 30
    max_retries: int = 3

@dataclass  
class WorkflowConfig:
    """Complete workflow configuration"""
    s3: S3Config
    postgres: Optional[Dict[str, Any]] = None
    aws: Optional[Dict[str, Any]] = None

def load_settings_yaml(settings_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Load settings from YAML file
    
    Args:
        settings_file: Path to settings file (uses default locations if None)
        
    Returns:
        Settings dictionary
    """
    if settings_file is None:
        # Try standard locations
        possible_paths = [
            Path(__file__).parent.parent.parent / "tidyllm" / "admin" / "settings.yaml",
            Path("tidyllm/admin/settings.yaml"),
            Path("settings.yaml")
        ]
        
        for path in possible_paths:
            if path.exists():
                settings_file = str(path)
                break
    
    if settings_file is None:
        logger.warning("No settings.yaml found, using defaults")
        return {}
    
    try:
        with open(settings_file, 'r') as f:
            settings = yaml.safe_load(f)
        
        logger.info(f"Loaded settings from {settings_file}")
        return settings
        
    except Exception as e:
        logger.error(f"Failed to load settings from {settings_file}: {e}")
        return {}

def create_workflow_config(
    s3_bucket: Optional[str] = None,
    s3_prefix: Optional[str] = None,
    s3_region: Optional[str] = None,
    settings_file: Optional[str] = None
) -> WorkflowConfig:
    """
    Create workflow configuration with override capability
    
    Args:
        s3_bucket: Override S3 bucket (uses settings.yaml default if None)
        s3_prefix: Override S3 prefix (uses settings.yaml default if None)  
        s3_region: Override S3 region (uses settings.yaml default if None)
        settings_file: Path to settings file
        
    Returns:
        WorkflowConfig with merged defaults and overrides
    """
    # Load base settings
    settings = load_settings_yaml(settings_file)
    
    # Extract S3 defaults from settings.yaml
    s3_settings = settings.get('s3', {})
    
    # Create S3 config with override capability
    s3_config = S3Config(
        bucket=s3_bucket or s3_settings.get('bucket', s3_config["bucket"]),
        region=s3_region or s3_settings.get('region', 'us-east-1'),
        prefix=s3_prefix or s3_settings.get('prefix', 'pages/'),
        connection_timeout=s3_settings.get('connection_timeout', 30),
        max_retries=s3_settings.get('max_retries', 3)
    )
    
    # Log configuration choices
    logger.info(f"Workflow S3 Config:")
    logger.info(f"  Bucket: {s3_config.bucket} {'(override)' if s3_bucket else '(from settings.yaml)'}")
    logger.info(f"  Prefix: {s3_config.prefix} {'(override)' if s3_prefix else '(from settings.yaml)'}")
    logger.info(f"  Region: {s3_config.region} {'(override)' if s3_region else '(from settings.yaml)'}")
    
    # Create complete workflow config
    workflow_config = WorkflowConfig(
        s3=s3_config,
        postgres=settings.get('postgres'),
        aws=settings.get('aws')
    )
    
    return workflow_config

def get_domain_specific_config(domain_name: str) -> Dict[str, Any]:
    """
    Get domain-specific configuration overrides
    
    Args:
        domain_name: Name of the domain
        
    Returns:
        Domain-specific config overrides
    """
    # Domain-specific configurations
    domain_configs = {
        # Research domains use research bucket
        "research": {
            "s3_bucket": s3_config["bucket"], 
            "s3_prefix": "research/"
        },
        
        # Model validation uses specialized structure  
        "model_validation": {
            "s3_bucket": s3_config["bucket"],
            "s3_prefix": "model_validation/"
        },
        
        # Legacy/special projects use different bucket
        "legacy_projects": {
            "s3_bucket": "dsai-2025-asu", 
            "s3_prefix": "workflows/"
        },
        
        # Development/testing domains
        "test_domains": {
            "s3_bucket": "dsai-2025-asu",
            "s3_prefix": "test_workflows/"
        }
    }
    
    # Check for domain-specific config
    if domain_name.startswith("test_") or domain_name.startswith("dev_"):
        return domain_configs.get("test_domains", {})
    
    # Check exact matches
    return domain_configs.get(domain_name, {})

def create_domain_workflow_config(
    domain_name: str,
    s3_bucket: Optional[str] = None,
    s3_prefix: Optional[str] = None,
    s3_region: Optional[str] = None
) -> WorkflowConfig:
    """
    Create workflow config with domain-specific defaults and overrides
    
    Args:
        domain_name: Domain name for workflow
        s3_bucket: Explicit S3 bucket override
        s3_prefix: Explicit S3 prefix override
        s3_region: Explicit S3 region override
        
    Returns:
        WorkflowConfig optimized for the domain
    """
    # Get domain-specific defaults
    domain_config = get_domain_specific_config(domain_name)
    
    # Apply hierarchy: explicit override > domain default > settings.yaml default
    final_bucket = s3_bucket or domain_config.get("s3_bucket")
    final_prefix = s3_prefix or domain_config.get("s3_prefix") 
    final_region = s3_region or domain_config.get("s3_region")
    
    logger.info(f"Creating config for domain '{domain_name}':")
    if domain_config:
        logger.info(f"  Domain-specific defaults: {domain_config}")
    
    return create_workflow_config(
        s3_bucket=final_bucket,
        s3_prefix=final_prefix, 
        s3_region=final_region
    )

# Convenience functions
def get_default_s3_config() -> S3Config:
    """Get default S3 config from settings.yaml"""
    workflow_config = create_workflow_config()
    return workflow_config.s3

def get_s3_path(domain_name: str, workflow_config: WorkflowConfig) -> str:
    """
    Generate S3 path for domain workflow
    
    Args:
        domain_name: Domain name
        workflow_config: Workflow configuration
        
    Returns:
        Complete S3 path: s3://bucket/prefix/domain_name/
    """
    s3_config = workflow_config.s3
    
    # Build path: prefix + domain_name 
    if s3_config.prefix:
        # Remove trailing slash from prefix and add domain
        prefix_clean = s3_config.prefix.rstrip('/')
        path = f"{prefix_clean}/{domain_name}/"
    else:
        path = f"{domain_name}/"
    
    return f"s3://{s3_config.bucket}/{path}"