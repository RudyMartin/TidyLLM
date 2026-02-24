#!/usr/bin/env python3
"""
TidyLLM MLflow Integration Test
==============================

Tests MLflow connectivity, experiment tracking, and database integration.
Verifies MLflow can log experiments and connect to PostgreSQL backend.
"""

import sys
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class MLflowTester:
    """Test MLflow integration and functionality."""
    
    def __init__(self):
        self.admin_path = Path(__file__).parent
        self.settings_path = self.admin_path / "settings.yaml"
        self.test_results = []
        self.config = None
        self.mlflow = None
        
    def log_test(self, test_name: str, status: str, details: Dict[str, Any]):
        """Log test result with timestamp and details."""
        result = {
            'timestamp': datetime.now().isoformat(),
            'test_name': test_name,
            'status': status,
            'details': details
        }
        self.test_results.append(result)
        
        # Status indicators (no Unicode for Windows compatibility)
        status_prefix = {
            'PASS': '[PASS]',
            'FAIL': '[FAIL]',
            'WARN': '[WARN]', 
            'INFO': '[INFO]'
        }
        
        prefix = status_prefix.get(status, '[TEST]')
        print(f"{prefix} {test_name}: {details.get('message', 'Test completed')}")
        
        if status == 'FAIL' and 'error' in details:
            print(f"   Error: {details['error']}")
        elif 'value' in details:
            print(f"   Value: {details['value']}")
    
    def load_config(self) -> bool:
        """Load configuration from settings.yaml."""
        try:
            with open(self.settings_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            if 'integrations' not in self.config or 'mlflow' not in self.config['integrations']:
                self.log_test(
                    "MLflow Config Load",
                    "FAIL",
                    {"message": "No MLflow configuration found in settings.yaml"}
                )
                return False
            
            self.log_test(
                "MLflow Config Load", 
                "PASS",
                {"message": "MLflow configuration loaded successfully"}
            )
            return True
            
        except Exception as e:
            self.log_test(
                "MLflow Config Load",
                "FAIL", 
                {"message": "Failed to load settings.yaml", "error": str(e)}
            )
            return False
    
    def test_mlflow_import(self) -> bool:
        """Test MLflow package import and basic functionality."""
        try:
            import mlflow
            self.mlflow = mlflow
            
            version = mlflow.version.VERSION
            self.log_test(
                "MLflow Import",
                "PASS",
                {"message": "MLflow imported successfully", "version": version}
            )
            return True
            
        except ImportError:
            self.log_test(
                "MLflow Import",
                "FAIL",
                {"message": "MLflow not installed", "fix": "pip install mlflow"}
            )
            return False
        except Exception as e:
            self.log_test(
                "MLflow Import",
                "FAIL",
                {"message": "MLflow import error", "error": str(e)}
            )
            return False
    
    def test_tracking_uri_config(self) -> bool:
        """Test MLflow tracking URI configuration."""
        if not self.config:
            return False
        
        mlflow_config = self.config['integrations']['mlflow']
        
        if not mlflow_config.get('enabled', False):
            self.log_test(
                "MLflow Tracking URI",
                "INFO",
                {"message": "MLflow integration disabled in config"}
            )
            return True
        
        tracking_uri = mlflow_config.get('tracking_uri')
        if not tracking_uri:
            self.log_test(
                "MLflow Tracking URI", 
                "FAIL",
                {"message": "No tracking_uri specified in config"}
            )
            return False
        
        # Set tracking URI
        try:
            self.mlflow.set_tracking_uri(tracking_uri)
            current_uri = self.mlflow.get_tracking_uri()
            
            self.log_test(
                "MLflow Tracking URI",
                "PASS",
                {"message": "Tracking URI configured", "uri": current_uri}
            )
            return True
            
        except Exception as e:
            self.log_test(
                "MLflow Tracking URI",
                "FAIL", 
                {"message": "Failed to set tracking URI", "error": str(e)}
            )
            return False
    
    def test_database_connectivity(self) -> bool:
        """Test MLflow database backend connectivity."""
        if not self.config or not self.mlflow:
            return False
        
        mlflow_config = self.config['integrations']['mlflow']
        tracking_uri = mlflow_config.get('tracking_uri')
        
        if not tracking_uri or not tracking_uri.startswith('postgresql://'):
            self.log_test(
                "MLflow Database",
                "WARN",
                {"message": "Not using PostgreSQL backend", "uri_type": "file" if not tracking_uri else "other"}
            )
            return True
        
        try:
            # Try to connect to the MLflow store
            from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT
            from mlflow.tracking import MlflowClient
            
            client = MlflowClient(tracking_uri)
            
            # Test basic operations
            experiments = client.search_experiments(max_results=1)
            
            self.log_test(
                "MLflow Database",
                "PASS",
                {"message": "Database backend accessible", "experiments_found": len(experiments)}
            )
            return True
            
        except Exception as e:
            self.log_test(
                "MLflow Database",
                "FAIL",
                {"message": "Database backend connection failed", "error": str(e)}
            )
            return False
    
    def test_experiment_operations(self) -> bool:
        """Test MLflow experiment creation and logging."""
        if not self.mlflow:
            return False
        
        try:
            # Create test experiment
            experiment_name = f"tidyllm_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            experiment_id = self.mlflow.create_experiment(experiment_name)
            self.mlflow.set_experiment(experiment_name)
            
            self.log_test(
                "MLflow Experiment Creation",
                "PASS", 
                {"message": "Test experiment created", "experiment_id": experiment_id, "name": experiment_name}
            )
            
            # Test logging
            with self.mlflow.start_run() as run:
                # Log parameters
                self.mlflow.log_param("test_param", "test_value")
                self.mlflow.log_metric("test_metric", 0.95)
                
                # Log artifact (create temp file)
                temp_file = Path("/tmp/test_artifact.txt") if sys.platform != "win32" else Path("temp_test_artifact.txt")
                with open(temp_file, 'w') as f:
                    f.write("MLflow test artifact")
                
                self.mlflow.log_artifact(str(temp_file))
                temp_file.unlink()  # Clean up
                
                run_id = run.info.run_id
            
            self.log_test(
                "MLflow Logging",
                "PASS",
                {"message": "Successfully logged params, metrics, and artifacts", "run_id": run_id}
            )
            
            # Clean up test experiment
            try:
                self.mlflow.delete_experiment(experiment_id)
            except:
                pass  # Some MLflow versions don't support deletion
            
            return True
            
        except Exception as e:
            self.log_test(
                "MLflow Experiment Operations",
                "FAIL",
                {"message": "Experiment operations failed", "error": str(e)}
            )
            return False
    
    def test_existing_experiments(self) -> bool:
        """Test access to existing MLflow experiments.""" 
        if not self.mlflow:
            return False
        
        try:
            from mlflow.tracking import MlflowClient
            
            client = MlflowClient()
            experiments = client.search_experiments()
            
            if not experiments:
                self.log_test(
                    "MLflow Existing Experiments",
                    "INFO",
                    {"message": "No existing experiments found"}
                )
                return True
            
            # Get details about existing experiments
            experiment_details = []
            for exp in experiments[:5]:  # Limit to first 5
                runs = client.search_runs(exp.experiment_id, max_results=1)
                experiment_details.append({
                    "name": exp.name,
                    "id": exp.experiment_id,
                    "run_count": len(runs)
                })
            
            self.log_test(
                "MLflow Existing Experiments", 
                "PASS",
                {"message": f"Found {len(experiments)} experiments", "details": experiment_details}
            )
            return True
            
        except Exception as e:
            self.log_test(
                "MLflow Existing Experiments",
                "FAIL",
                {"message": "Failed to access existing experiments", "error": str(e)}
            )
            return False
    
    def test_tidyllm_experiment(self) -> bool:
        """Test for TidyLLM-specific experiments."""
        if not self.mlflow or not self.config:
            return False
        
        mlflow_config = self.config['integrations']['mlflow']
        tidyllm_experiment = mlflow_config.get('experiment_name', 'tidyllm-workflows')
        
        try:
            from mlflow.tracking import MlflowClient
            
            client = MlflowClient()
            
            # Try to get the TidyLLM experiment
            try:
                experiment = client.get_experiment_by_name(tidyllm_experiment)
                if experiment:
                    runs = client.search_runs(experiment.experiment_id, max_results=10)
                    
                    self.log_test(
                        "TidyLLM Experiment",
                        "PASS",
                        {"message": f"TidyLLM experiment found", "name": tidyllm_experiment, "runs": len(runs)}
                    )
                    return True
                else:
                    self.log_test(
                        "TidyLLM Experiment",
                        "WARN", 
                        {"message": f"TidyLLM experiment '{tidyllm_experiment}' not found - will be created on first use"}
                    )
                    return True
                    
            except Exception as e:
                self.log_test(
                    "TidyLLM Experiment", 
                    "WARN",
                    {"message": f"TidyLLM experiment '{tidyllm_experiment}' not accessible", "error": str(e)}
                )
                return True  # Not critical
                
        except Exception as e:
            self.log_test(
                "TidyLLM Experiment",
                "FAIL",
                {"message": "Failed to check TidyLLM experiment", "error": str(e)}
            )
            return False
    
    def test_mlflow_ui_access(self) -> bool:
        """Test if MLflow UI would be accessible."""
        if not self.config:
            return False
        
        mlflow_config = self.config['integrations']['mlflow'] 
        tracking_uri = mlflow_config.get('tracking_uri', '')
        
        if tracking_uri.startswith('postgresql://'):
            # Extract connection details
            import urllib.parse
            parsed = urllib.parse.urlparse(tracking_uri)
            
            self.log_test(
                "MLflow UI Access",
                "INFO",
                {"message": "MLflow UI available via 'mlflow ui' command", "backend": "PostgreSQL", "host": parsed.hostname}
            )
            return True
        elif tracking_uri.startswith('file://') or not tracking_uri:
            self.log_test(
                "MLflow UI Access", 
                "INFO",
                {"message": "MLflow UI available via 'mlflow ui' command", "backend": "local_files"}
            )
            return True
        else:
            self.log_test(
                "MLflow UI Access",
                "WARN",
                {"message": "MLflow UI access depends on tracking URI type", "uri_type": tracking_uri.split('://')[0] if '://' in tracking_uri else "unknown"}
            )
            return True
    
    def run_all_tests(self) -> bool:
        """Run all MLflow tests and return overall status."""
        print("TIDYLLM MLFLOW INTEGRATION TEST")
        print("=" * 50)
        print(f"Testing MLflow at: {datetime.now().isoformat()}")
        print()
        
        # Run tests in order
        tests_passed = 0
        total_tests = 7
        
        if not self.load_config():
            print("Cannot continue without valid configuration")
            return False
        
        if self.test_mlflow_import():
            tests_passed += 1
        else:
            print("Cannot continue without MLflow package")
            return False
        
        if self.test_tracking_uri_config():
            tests_passed += 1
            
        if self.test_database_connectivity():
            tests_passed += 1
            
        if self.test_experiment_operations():
            tests_passed += 1
            
        if self.test_existing_experiments():
            tests_passed += 1
            
        if self.test_tidyllm_experiment():
            tests_passed += 1
            
        if self.test_mlflow_ui_access():
            tests_passed += 1
        
        # Summary
        print()
        print("=" * 50)
        print("MLFLOW TEST SUMMARY")
        print("=" * 50)
        
        passed_tests = len([r for r in self.test_results if r['status'] == 'PASS'])
        failed_tests = len([r for r in self.test_results if r['status'] == 'FAIL'])
        warned_tests = len([r for r in self.test_results if r['status'] == 'WARN'])
        info_tests = len([r for r in self.test_results if r['status'] == 'INFO'])
        
        print(f"Tests Run: {len(self.test_results)}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Warnings: {warned_tests}")
        print(f"Info: {info_tests}")
        
        if failed_tests == 0:
            overall_status = "WORKING"
            print(f"\nMLFLOW STATUS: WORKING")
            print("MLflow integration is fully operational")
        elif passed_tests > failed_tests:
            overall_status = "PARTIAL" 
            print(f"\nMLFLOW STATUS: PARTIAL")
            print("MLflow has some issues but basic functionality works")
        else:
            overall_status = "BROKEN"
            print(f"\nMLFLOW STATUS: BROKEN")
            print("MLflow integration has critical issues")
        
        # Save results
        self.save_test_results(overall_status)
        
        return overall_status == "WORKING"
    
    def save_test_results(self, overall_status: str):
        """Save test results to JSON file."""
        results_file = self.admin_path / f"mlflow_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': overall_status,
            'test_results': self.test_results,
            'summary': {
                'total_tests': len(self.test_results),
                'passed': len([r for r in self.test_results if r['status'] == 'PASS']),
                'failed': len([r for r in self.test_results if r['status'] == 'FAIL']),
                'warnings': len([r for r in self.test_results if r['status'] == 'WARN']),
                'info': len([r for r in self.test_results if r['status'] == 'INFO'])
            }
        }
        
        try:
            with open(results_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"\nMLflow test results saved: {results_file}")
        except Exception as e:
            print(f"\nWarning: Could not save test results: {e}")


def main():
    """Main entry point."""
    tester = MLflowTester()
    success = tester.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())