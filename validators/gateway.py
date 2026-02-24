"""
Gateway Validator for TidyLLM
============================

Validates all TidyLLM 4-gateway architecture components with timeout protection.
"""

import time
from typing import Dict, Any
from .base import BaseValidator


class GatewayValidator(BaseValidator):
    """Validator for TidyLLM 4-gateway architecture."""
    
    def validate_gateways(self) -> Dict[str, Any]:
        """
        Validate all TidyLLM gateways - Corporate-safe initialization.
        
        Returns:
            Validation results for all 4 gateways
        """
        
        # Detect corporate environment first
        env_info = self.detect_corporate_environment()
        
        if self.corporate_mode:
            return self._validate_gateways_corporate_safe()
        else:
            return self._validate_gateways_standard()
    
    def _validate_gateways_corporate_safe(self) -> Dict[str, Any]:
        """Validate gateways in corporate mode with basic checks only."""
        
        result = {
            'corporate_llm': self.corporate_safe_result('CorporateLLMGateway', 'Corporate mode - gateway import validation only'),
            'ai_processing': self.corporate_safe_result('AIProcessingGateway', 'Corporate mode - gateway import validation only'),
            'database': self.corporate_safe_result('DatabaseGateway', 'Corporate mode - gateway import validation only'),
            'workflow_optimizer': self.corporate_safe_result('WorkflowOptimizerGateway', 'Corporate mode - gateway import validation only'),
            'overall_status': 'corporate_safe',
            'corporate_mode': True
        }
        
        # Test basic imports without initialization
        try:
            from tidyllm.gateways.corporate_llm_gateway import CorporateLLMGateway
            result['corporate_llm']['import_status'] = 'success'
        except ImportError as e:
            result['corporate_llm']['import_error'] = str(e)
            
        try:
            from tidyllm.gateways.ai_processing_gateway import AIProcessingGateway
            result['ai_processing']['import_status'] = 'success'
        except ImportError as e:
            result['ai_processing']['import_error'] = str(e)
            
        try:
            from tidyllm.gateways.database_gateway import DatabaseGateway
            result['database']['import_status'] = 'success'
        except ImportError as e:
            result['database']['import_error'] = str(e)
            
        try:
            from tidyllm.gateways.workflow_optimizer_gateway import WorkflowOptimizerGateway
            result['workflow_optimizer']['import_status'] = 'success'
        except ImportError as e:
            result['workflow_optimizer']['import_error'] = str(e)
        
        return result
    
    def _validate_gateways_standard(self) -> Dict[str, Any]:
        """Validate gateways in standard mode with full initialization."""
        
        result = {}
        
        try:
            from tidyllm.gateways.corporate_llm_gateway import CorporateLLMGateway
            from tidyllm.gateways.ai_processing_gateway import AIProcessingGateway
            from tidyllm.gateways.database_gateway import DatabaseGateway
            from tidyllm.gateways.workflow_optimizer_gateway import WorkflowOptimizerGateway
            
            from tidyllm.infrastructure.session.unified import get_global_session_manager
            session_manager = get_global_session_manager()
            
            gateways = {
                'corporate_llm': CorporateLLMGateway,
                'ai_processing': AIProcessingGateway,
                'database': DatabaseGateway,
                'workflow_optimizer': WorkflowOptimizerGateway
            }
            
            for name, gateway_class in gateways.items():
                result[name] = self.run_with_timeout(
                    lambda gc=gateway_class, n=name: self._test_gateway_initialization(gc, n, session_manager),
                    f'{name}_gateway'
                )
                    
        except Exception as e:
            result['error'] = f'Failed to validate gateways: {e}'
        
        # Determine overall status
        if result:
            statuses = [r.get('status') for r in result.values() if isinstance(r, dict)]
            if all(status == 'success' for status in statuses):
                result['overall_status'] = 'success'
            elif any(status == 'success' for status in statuses):
                result['overall_status'] = 'partial'
            else:
                result['overall_status'] = 'failed'
        else:
            result['overall_status'] = 'error'
        
        return result
    
    def _test_gateway_initialization(self, gateway_class, name: str, session_manager) -> Dict[str, Any]:
        """Test individual gateway initialization."""
        
        start_time = time.time()
        try:
            # Create gateway with proper parameters
            if name == 'database':
                from tidyllm.gateways.database_gateway import DatabaseGatewayConfig
                config = DatabaseGatewayConfig()
                gateway = gateway_class(config)
            else:
                gateway = gateway_class()
            
            # Set session manager after creation
            gateway.session_manager = session_manager
            
            return {
                'status': 'success',
                'message': f'{name} gateway initialized successfully',
                'latency': (time.time() - start_time) * 1000
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'{name} gateway failed: {e}',
                'latency': (time.time() - start_time) * 1000
            }