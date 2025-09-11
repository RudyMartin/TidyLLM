"""
TidyLLM Onboarding Configuration Templates
==========================================

Configuration templates for different deployment scenarios.
"""

from typing import Dict, Any

class ConfigTemplates:
    """Configuration templates for TidyLLM onboarding."""
    
    @staticmethod
    def get_corporate_template() -> Dict[str, Any]:
        """Get corporate deployment template."""
        return {
            'aws': {
                'region': 'us-east-1',
                'profile': 'corporate',
                'sso_enabled': True,
                'proxy_host': None,
                'proxy_port': None
            },
            'database': {
                'host': 'corporate-db.company.com',
                'port': 5432,
                'name': 'tidyllm_corporate',
                'user': 'tidyllm_user',
                'ssl_mode': 'require'
            },
            'security': {
                'encryption': True,
                'audit_logging': True,
                'data_masking': True,
                'sso_provider': 'okta',
                'compliance_mode': 'sox'
            },
            'models': {
                'default_provider': 'bedrock',
                'default_model': 'anthropic.claude-3-sonnet-20240229-v1:0',
                'fallback_provider': 'openai',
                'rate_limiting': True
            },
            'deployment': {
                'environment': 'production',
                'scaling': 'horizontal',
                'monitoring': True,
                'backup_enabled': True
            }
        }
    
    @staticmethod
    def get_development_template() -> Dict[str, Any]:
        """Get development deployment template."""
        return {
            'aws': {
                'region': 'us-east-1',
                'profile': 'default',
                'sso_enabled': False
            },
            'database': {
                'host': 'localhost',
                'port': 5432,
                'name': 'tidyllm_dev',
                'user': 'postgres',
                'ssl_mode': 'disable'
            },
            'security': {
                'encryption': False,
                'audit_logging': False,
                'data_masking': False
            },
            'models': {
                'default_provider': 'bedrock',
                'default_model': 'anthropic.claude-3-haiku-20240307-v1:0',
                'rate_limiting': False
            },
            'deployment': {
                'environment': 'development',
                'scaling': 'single',
                'monitoring': False,
                'backup_enabled': False
            }
        }
    
    @staticmethod
    def get_staging_template() -> Dict[str, Any]:
        """Get staging deployment template."""
        return {
            'aws': {
                'region': 'us-east-1',
                'profile': 'staging',
                'sso_enabled': True
            },
            'database': {
                'host': 'staging-db.company.com',
                'port': 5432,
                'name': 'tidyllm_staging',
                'user': 'tidyllm_user',
                'ssl_mode': 'require'
            },
            'security': {
                'encryption': True,
                'audit_logging': True,
                'data_masking': True
            },
            'models': {
                'default_provider': 'bedrock',
                'default_model': 'anthropic.claude-3-sonnet-20240229-v1:0',
                'rate_limiting': True
            },
            'deployment': {
                'environment': 'staging',
                'scaling': 'horizontal',
                'monitoring': True,
                'backup_enabled': True
            }
        }
    
    @staticmethod
    def get_local_template() -> Dict[str, Any]:
        """Get local deployment template (matches current staging settings)."""
        return {
            'aws': {
                'region': 'us-east-1',
                'profile': 'default',
                'sso_enabled': False
            },
            'database': {
                'host': 'vectorqa-cluster.cluster-cu562e4m02nq.us-east-1.rds.amazonaws.com',
                'port': 5432,
                'name': 'vectorqa',
                'user': 'vectorqa_user',
                'ssl_mode': 'require'
            },
            'security': {
                'encryption': True,
                'audit_logging': True,
                'data_masking': True
            },
            'models': {
                'default_provider': 'bedrock',
                'default_model': 'anthropic.claude-3-sonnet-20240229-v1:0',
                'rate_limiting': True
            },
            'deployment': {
                'environment': 'local',
                'scaling': 'single',
                'monitoring': True,
                'backup_enabled': True
            }
        }
    
    @staticmethod
    def get_template_by_name(name: str) -> Dict[str, Any]:
        """Get template by name."""
        templates = {
            'corporate': ConfigTemplates.get_corporate_template,
            'development': ConfigTemplates.get_development_template,
            'staging': ConfigTemplates.get_staging_template,
            'local': ConfigTemplates.get_local_template
        }
        
        if name in templates:
            return templates[name]()
        else:
            raise ValueError(f"Unknown template: {name}")
