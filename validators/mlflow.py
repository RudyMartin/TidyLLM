"""
MLflow Validator for TidyLLM
===========================

Corporate-safe MLflow validation that uses TidyLLM's MLflowIntegrationService.
"""

from typing import Dict, Any
from .base import BaseValidator


class MLflowValidator(BaseValidator):
    """Corporate-safe MLflow validation using TidyLLM's MLflowIntegrationService."""
    
    def validate_mlflow_connectivity(self) -> Dict[str, Any]:
        """
        Validate MLflow connectivity using TidyLLM's MLflow service.
        
        Returns:
            MLflow validation results
        """
        
        # Detect corporate environment first
        env_info = self.detect_corporate_environment()
        
        if self.corporate_mode:
            return self._validate_mlflow_corporate_safe()
        else:
            return self._validate_mlflow_standard()
    
    def _validate_mlflow_corporate_safe(self) -> Dict[str, Any]:
        """Validate MLflow in corporate mode with basic checks only."""
        
        return self.corporate_safe_result(
            'MLflow',
            'Corporate environment - MLflow validation uses service integration only'
        )
    
    def _validate_mlflow_standard(self) -> Dict[str, Any]:
        """Validate MLflow in standard mode with full service integration."""
        
        return self.run_with_timeout(self._test_mlflow_service, 'MLflow')
    
    def _test_mlflow_service(self) -> Dict[str, Any]:
        """Test MLflow using TidyLLM's MLflowIntegrationService."""
        
        try:
            from tidyllm.services.mlflow_integration_service import MLflowIntegrationService
            
            # Use TidyLLM's MLflow service for validation
            mlflow_service = MLflowIntegrationService()
            health_status = mlflow_service.health_check()
            
            if health_status.get('healthy', False):
                return {
                    'status': 'success',
                    'message': f'MLflow integration healthy - {health_status.get("routes_available", 0)} routes available',
                    'details': health_status
                }
            else:
                return {
                    'status': 'error',
                    'message': health_status.get('last_error', 'MLflow integration unhealthy'),
                    'details': health_status
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'MLflow service validation failed: {e}'
            }