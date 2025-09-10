"""
TidyLLM Onboarding Connection Validator
======================================

Validates connections to all TidyLLM services and components.
"""

import streamlit as st
import time
from typing import Dict, Any, List, Tuple
from pathlib import Path
import sys

# Add parent directory to Python path for tidyllm imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class ConnectionValidator:
    """Validates connections to TidyLLM services."""
    
    def __init__(self):
        self.results = {}
    
    def validate_aws_connectivity(self) -> Dict[str, Any]:
        """Validate AWS connectivity and services."""
        result = {
            's3': {'status': 'unknown', 'message': '', 'latency': 0},
            'bedrock': {'status': 'unknown', 'message': '', 'latency': 0},
            'sts': {'status': 'unknown', 'message': '', 'latency': 0}
        }
        
        try:
            from tidyllm.infrastructure.session.unified import UnifiedSessionManager
            session_manager = UnifiedSessionManager()
            
            # Test S3
            start_time = time.time()
            try:
                s3_client = session_manager.get_s3_client()
                s3_client.list_buckets()
                result['s3'] = {
                    'status': 'success',
                    'message': 'S3 connection successful',
                    'latency': (time.time() - start_time) * 1000
                }
            except Exception as e:
                result['s3'] = {
                    'status': 'error',
                    'message': f'S3 connection failed: {e}',
                    'latency': (time.time() - start_time) * 1000
                }
            
            # Test Bedrock
            start_time = time.time()
            try:
                bedrock_client = session_manager.get_bedrock_client()
                bedrock_client.list_foundation_models()
                result['bedrock'] = {
                    'status': 'success',
                    'message': 'Bedrock connection successful',
                    'latency': (time.time() - start_time) * 1000
                }
            except Exception as e:
                result['bedrock'] = {
                    'status': 'error',
                    'message': f'Bedrock connection failed: {e}',
                    'latency': (time.time() - start_time) * 1000
                }
            
            # Test STS
            start_time = time.time()
            try:
                sts_client = session_manager.get_sts_client()
                identity = sts_client.get_caller_identity()
                result['sts'] = {
                    'status': 'success',
                    'message': f'STS connection successful - Account: {identity["Account"]}',
                    'latency': (time.time() - start_time) * 1000
                }
            except Exception as e:
                result['sts'] = {
                    'status': 'error',
                    'message': f'STS connection failed: {e}',
                    'latency': (time.time() - start_time) * 1000
                }
                
        except Exception as e:
            result['error'] = f'Failed to initialize session manager: {e}'
        
        return result
    
    def validate_database_connectivity(self) -> Dict[str, Any]:
        """Validate database connectivity."""
        result = {'status': 'unknown', 'message': '', 'latency': 0}
        
        try:
            from tidyllm.infrastructure.session.unified import UnifiedSessionManager
            session_manager = UnifiedSessionManager()
            
            start_time = time.time()
            try:
                conn = session_manager.get_postgres_connection()
                if conn:
                    result = {
                        'status': 'success',
                        'message': 'PostgreSQL connection successful',
                        'latency': (time.time() - start_time) * 1000
                    }
                else:
                    result = {
                        'status': 'error',
                        'message': 'PostgreSQL connection failed - no connection returned',
                        'latency': (time.time() - start_time) * 1000
                    }
            except Exception as e:
                result = {
                    'status': 'error',
                    'message': f'PostgreSQL connection failed: {e}',
                    'latency': (time.time() - start_time) * 1000
                }
                
        except Exception as e:
            result = {
                'status': 'error',
                'message': f'Failed to initialize session manager: {e}',
                'latency': 0
            }
        
        return result
    
    def validate_gateways(self) -> Dict[str, Any]:
        """Validate all TidyLLM gateways."""
        result = {}
        
        try:
            from tidyllm.gateways.corporate_llm_gateway import CorporateLLMGateway
            from tidyllm.gateways.ai_processing_gateway import AIProcessingGateway
            from tidyllm.gateways.database_gateway import DatabaseGateway
            from tidyllm.gateways.workflow_optimizer_gateway import WorkflowOptimizerGateway
            
            from tidyllm.infrastructure.session.unified import UnifiedSessionManager
            session_manager = UnifiedSessionManager()
            
            gateways = {
                'corporate_llm': CorporateLLMGateway,
                'ai_processing': AIProcessingGateway,
                'database': DatabaseGateway,
                'workflow_optimizer': WorkflowOptimizerGateway
            }
            
            for name, gateway_class in gateways.items():
                start_time = time.time()
                try:
                    gateway = gateway_class(session_manager=session_manager)
                    result[name] = {
                        'status': 'success',
                        'message': f'{name} gateway initialized successfully',
                        'latency': (time.time() - start_time) * 1000
                    }
                except Exception as e:
                    result[name] = {
                        'status': 'error',
                        'message': f'{name} gateway failed: {e}',
                        'latency': (time.time() - start_time) * 1000
                    }
                    
        except Exception as e:
            result['error'] = f'Failed to validate gateways: {e}'
        
        return result
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete validation of all TidyLLM components."""
        return {
            'aws': self.validate_aws_connectivity(),
            'database': self.validate_database_connectivity(),
            'gateways': self.validate_gateways(),
            'timestamp': time.time()
        }
