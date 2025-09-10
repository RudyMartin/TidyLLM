"""
TidyLLM Onboarding Preflight Tests
=================================

Pre-flight validation tests for the onboarding system.
"""

import streamlit as st
import sys
import os
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to Python path for tidyllm imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class PreflightTester:
    """Pre-flight tests for TidyLLM onboarding system."""
    
    def __init__(self):
        self.results = {}
    
    def test_imports(self) -> Dict[str, Any]:
        """Test if all required TidyLLM imports are available."""
        result = {
            'status': 'success',
            'message': 'All imports successful',
            'components': {}
        }
        
        components = [
            ('UnifiedSessionManager', 'tidyllm.infrastructure.session.unified'),
            ('CorporateLLMGateway', 'tidyllm.gateways.corporate_llm_gateway'),
            ('AIProcessingGateway', 'tidyllm.gateways.ai_processing_gateway'),
            ('DatabaseGateway', 'tidyllm.gateways.database_gateway'),
            ('WorkflowOptimizerGateway', 'tidyllm.gateways.workflow_optimizer_gateway'),
            ('DomainRAG', 'tidyllm.knowledge_systems.core.domain_rag'),
            ('BracketRegistry', 'tidyllm.flow.examples.bracket_registry')
        ]
        
        failed_imports = []
        
        for name, module in components:
            try:
                __import__(module)
                result['components'][name] = 'success'
            except ImportError as e:
                result['components'][name] = f'failed: {e}'
                failed_imports.append(name)
        
        if failed_imports:
            result['status'] = 'error'
            result['message'] = f'Failed imports: {", ".join(failed_imports)}'
        
        return result
    
    def test_environment_variables(self) -> Dict[str, Any]:
        """Test required environment variables."""
        result = {
            'status': 'success',
            'message': 'All required environment variables found',
            'variables': {}
        }
        
        required_vars = [
            'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY',
            'AWS_DEFAULT_REGION'
        ]
        
        missing_vars = []
        
        for var in required_vars:
            value = os.environ.get(var)
            if value:
                result['variables'][var] = 'present'
            else:
                result['variables'][var] = 'missing'
                missing_vars.append(var)
        
        if missing_vars:
            result['status'] = 'warning'
            result['message'] = f'Missing variables: {", ".join(missing_vars)}'
        
        return result
    
    def test_dependencies(self) -> Dict[str, Any]:
        """Test required Python dependencies."""
        result = {
            'status': 'success',
            'message': 'All dependencies available',
            'dependencies': {}
        }
        
        dependencies = [
            'streamlit',
            'pandas',
            'plotly',
            'polars',
            'yaml',
            'boto3',
            'psycopg2'
        ]
        
        missing_deps = []
        
        for dep in dependencies:
            try:
                __import__(dep)
                result['dependencies'][dep] = 'available'
            except ImportError:
                result['dependencies'][dep] = 'missing'
                missing_deps.append(dep)
        
        if missing_deps:
            result['status'] = 'error'
            result['message'] = f'Missing dependencies: {", ".join(missing_deps)}'
        
        return result
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all preflight tests."""
        return {
            'imports': self.test_imports(),
            'environment': self.test_environment_variables(),
            'dependencies': self.test_dependencies(),
            'overall_status': 'success'  # Will be updated based on results
        }
