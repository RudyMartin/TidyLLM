"""
Workflow Validator for TidyLLM
=============================

Validates TidyLLM workflow files and configurations.
"""

from typing import Dict, Any, List
from pathlib import Path
from .base import BaseValidator


class WorkflowValidator(BaseValidator):
    """Validator for TidyLLM workflow files and configurations."""
    
    def validate_workflow_files(self, workflow_files: List[str] = None) -> Dict[str, Any]:
        """
        Validate TidyLLM workflow files.
        
        Args:
            workflow_files: Optional list of workflow files to validate
            
        Returns:
            Workflow validation results
        """
        
        if workflow_files is None:
            # Auto-discover workflow files
            workflow_files = self._discover_workflow_files()
        
        if not workflow_files:
            return {
                'status': 'success',
                'message': 'No workflow files found to validate',
                'file_count': 0
            }
        
        # Detect corporate environment
        env_info = self.detect_corporate_environment()
        
        if self.corporate_mode:
            return self._validate_workflows_corporate_safe(workflow_files)
        else:
            return self._validate_workflows_standard(workflow_files)
    
    def _validate_workflows_corporate_safe(self, workflow_files: List[str]) -> Dict[str, Any]:
        """Validate workflows in corporate mode with basic checks only."""
        
        result = {
            'status': 'corporate_safe',
            'message': f'Corporate mode - validated {len(workflow_files)} workflow files (syntax check only)',
            'file_count': len(workflow_files),
            'files': [],
            'corporate_mode': True
        }
        
        for file_path in workflow_files:
            file_result = {'path': file_path, 'status': 'corporate_safe', 'message': 'Syntax check only'}
            
            try:
                # Basic file existence and YAML syntax check
                if Path(file_path).exists():
                    import yaml
                    with open(file_path, 'r') as f:
                        yaml.safe_load(f)
                    file_result['message'] = 'YAML syntax valid'
                else:
                    file_result['status'] = 'error'
                    file_result['message'] = 'File not found'
                    
            except yaml.YAMLError as e:
                file_result['status'] = 'error'
                file_result['message'] = f'YAML syntax error: {e}'
            except Exception as e:
                file_result['status'] = 'error'
                file_result['message'] = f'Validation error: {e}'
            
            result['files'].append(file_result)
        
        return result
    
    def _validate_workflows_standard(self, workflow_files: List[str]) -> Dict[str, Any]:
        """Validate workflows in standard mode with full validation."""
        
        result = {
            'status': 'testing',
            'file_count': len(workflow_files),
            'files': [],
            'corporate_mode': False
        }
        
        success_count = 0
        
        for file_path in workflow_files:
            file_result = self.run_with_timeout(
                lambda fp=file_path: self._validate_single_workflow(fp),
                f'workflow_{Path(file_path).name}'
            )
            
            if file_result.get('status') == 'success':
                success_count += 1
            
            result['files'].append(file_result)
        
        # Determine overall status
        if success_count == len(workflow_files):
            result['status'] = 'success'
            result['message'] = f'All {len(workflow_files)} workflow files validated successfully'
        elif success_count > 0:
            result['status'] = 'partial'
            result['message'] = f'{success_count}/{len(workflow_files)} workflow files validated successfully'
        else:
            result['status'] = 'failed'
            result['message'] = 'No workflow files validated successfully'
        
        return result
    
    def _validate_single_workflow(self, file_path: str) -> Dict[str, Any]:
        """Validate a single workflow file."""
        
        try:
            if not Path(file_path).exists():
                return {
                    'path': file_path,
                    'status': 'error',
                    'message': 'File not found'
                }
            
            import yaml
            with open(file_path, 'r') as f:
                workflow_data = yaml.safe_load(f)
            
            # Basic workflow structure validation
            required_fields = ['name', 'version', 'steps']
            missing_fields = [field for field in required_fields if field not in workflow_data]
            
            if missing_fields:
                return {
                    'path': file_path,
                    'status': 'error',
                    'message': f'Missing required fields: {", ".join(missing_fields)}'
                }
            
            # Validate steps
            steps = workflow_data.get('steps', [])
            if not isinstance(steps, list) or len(steps) == 0:
                return {
                    'path': file_path,
                    'status': 'error',
                    'message': 'Workflow must have at least one step'
                }
            
            return {
                'path': file_path,
                'status': 'success',
                'message': f'Workflow valid - {len(steps)} steps found',
                'step_count': len(steps)
            }
            
        except yaml.YAMLError as e:
            return {
                'path': file_path,
                'status': 'error',
                'message': f'YAML syntax error: {e}'
            }
        except Exception as e:
            return {
                'path': file_path,
                'status': 'error',
                'message': f'Validation error: {e}'
            }
    
    def _discover_workflow_files(self) -> List[str]:
        """Auto-discover workflow files in TidyLLM directories."""
        
        workflow_dirs = [
            Path(__file__).parent.parent / "workflow_configs",
            Path(__file__).parent.parent / "workflows",
            Path(__file__).parent.parent.parent / "tidyllm" / "workflow_configs"
        ]
        
        workflow_files = []
        
        for workflow_dir in workflow_dirs:
            if workflow_dir.exists():
                workflow_files.extend([
                    str(f) for f in workflow_dir.glob("*.yaml")
                ])
                workflow_files.extend([
                    str(f) for f in workflow_dir.glob("*.yml")
                ])
        
        return workflow_files