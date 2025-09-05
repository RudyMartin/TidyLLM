#!/usr/bin/env python3
"""
Deployment Pre-Flight Checker

Specialized pre-flight checks for deployment environments.
Validates that the deployment bundle is ready for the target environment.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional

class DeploymentPreFlight:
    """Pre-flight checks for deployment environments"""
    
    def __init__(self, target_env: str = "auto"):
        self.target_env = target_env
        self.project_root = Path.cwd()
        self.bundle_path = self.project_root / "migration_bundles" / "production_ready_bundle"
        self.results = {
            "environment": target_env,
            "checks": {},
            "overall_status": "unknown"
        }
    
    def detect_environment(self) -> str:
        """Auto-detect the deployment environment"""
        env_vars = os.environ
        
        # SageMaker detection
        if "SAGEMAKER_NOTEBOOK_INSTANCE_NAME" in env_vars:
            return "sagemaker"
        elif "SAGEMAKER_ENDPOINT_NAME" in env_vars:
            return "sagemaker"
        
        # Docker detection
        elif "CONTAINER_NAME" in env_vars:
            return "docker"
        elif os.path.exists("/.dockerenv"):
            return "docker"
        
        # Lambda detection
        elif "AWS_LAMBDA_FUNCTION_NAME" in env_vars:
            return "lambda"
        elif "LAMBDA_RUNTIME_DIR" in env_vars:
            return "lambda"
        
        # Local development
        elif "VECTORQA_ENV" in env_vars:
            return env_vars["VECTORQA_ENV"]
        
        # Default to local
        else:
            return "local"
    
    def run_deployment_preflight(self) -> bool:
        """Run comprehensive deployment pre-flight checks"""
        print("🚀 DEPLOYMENT PRE-FLIGHT CHECKS")
        print("=" * 60)
        
        # Auto-detect environment if needed
        if self.target_env == "auto":
            self.target_env = self.detect_environment()
        
        print(f"🎯 Target Environment: {self.target_env.upper()}")
        print(f"📁 Current Directory: {self.project_root}")
        print()
        
        # Run environment-specific checks
        checks = [
            self._check_bundle_structure(),
            self._check_environment_specific_requirements(),
            self._check_import_compatibility(),
            self._check_configuration(),
            self._check_dependencies(),
            self._check_launcher_availability(),
            self._check_validation_tools()
        ]
        
        # Determine overall status
        passed_checks = sum(1 for check in checks if check.get('status') == 'passed')
        total_checks = len(checks)
        
        self.results["overall_status"] = "passed" if passed_checks == total_checks else "failed"
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 DEPLOYMENT PRE-FLIGHT SUMMARY")
        print("=" * 60)
        print(f"🎯 Environment: {self.target_env.upper()}")
        print(f"✅ Passed: {passed_checks}/{total_checks}")
        print(f"📊 Status: {self.results['overall_status'].upper()}")
        
        if self.results["overall_status"] == "passed":
            print("🎉 Deployment pre-flight checks PASSED!")
            print("🚀 Ready for deployment!")
        else:
            print("❌ Deployment pre-flight checks FAILED!")
            print("🔧 Please fix issues before deployment")
        
        return self.results["overall_status"] == "passed"
    
    def _check_bundle_structure(self) -> Dict[str, Any]:
        """Check deployment bundle structure"""
        print("📦 Checking Bundle Structure...")
        
        if not self.bundle_path.exists():
            return {
                "status": "failed",
                "message": "Production bundle not found",
                "details": f"Bundle not found at: {self.bundle_path}"
            }
        
        required_components = [
            "src", "config", "launchers", "validation", "docs"
        ]
        
        missing_components = []
        for component in required_components:
            component_path = self.bundle_path / component
            if not component_path.exists():
                missing_components.append(component)
        
        if missing_components:
            return {
                "status": "failed",
                "message": f"Missing bundle components: {', '.join(missing_components)}",
                "details": f"Bundle incomplete at: {self.bundle_path}"
            }
        
        self.results["checks"]["bundle_structure"] = {
            "status": "passed",
            "message": "Bundle structure complete",
            "details": f"All components present in {self.bundle_path}"
        }
        
        print("  ✅ Bundle structure complete")
        return self.results["checks"]["bundle_structure"]
    
    def _check_environment_specific_requirements(self) -> Dict[str, Any]:
        """Check environment-specific requirements"""
        print(f"🌍 Checking {self.target_env} Environment Requirements...")
        
        env_requirements = {
            "sagemaker": {
                "env_vars": ["AWS_DEFAULT_REGION"],
                "paths": ["/home/sagemaker-user"],
                "permissions": ["s3", "bedrock"]
            },
            "docker": {
                "env_vars": ["CONTAINER_NAME"],
                "paths": ["/app"],
                "permissions": ["file_system"]
            },
            "lambda": {
                "env_vars": ["AWS_LAMBDA_FUNCTION_NAME"],
                "paths": ["/var/task"],
                "permissions": ["lambda"]
            },
            "local": {
                "env_vars": [],
                "paths": [],
                "permissions": ["local"]
            }
        }
        
        requirements = env_requirements.get(self.target_env, {})
        
        # Check environment variables
        missing_env_vars = []
        for env_var in requirements.get("env_vars", []):
            if env_var not in os.environ:
                missing_env_vars.append(env_var)
        
        # Check paths
        missing_paths = []
        for path in requirements.get("paths", []):
            if not os.path.exists(path):
                missing_paths.append(path)
        
        if missing_env_vars or missing_paths:
            return {
                "status": "failed",
                "message": f"Environment requirements not met",
                "details": {
                    "missing_env_vars": missing_env_vars,
                    "missing_paths": missing_paths
                }
            }
        
        self.results["checks"]["environment_requirements"] = {
            "status": "passed",
            "message": f"{self.target_env} environment requirements met",
            "details": "All required environment variables and paths present"
        }
        
        print(f"  ✅ {self.target_env} environment requirements met")
        return self.results["checks"]["environment_requirements"]
    
    def _check_import_compatibility(self) -> Dict[str, Any]:
        """Check import compatibility for target environment"""
        print("🔧 Checking Import Compatibility...")
        
        try:
            # Add bundle src to path
            bundle_src = self.bundle_path / "src"
            if not bundle_src.exists():
                return {
                    "status": "failed",
                    "message": "Bundle src directory not found",
                    "details": f"Expected: {bundle_src}"
                }
            
            # Test critical imports
            import sys
            original_path = sys.path.copy()
            sys.path.insert(0, str(bundle_src))
            
            try:
                # Test config import
                from config.settings import config
                config_ok = True
            except ImportError:
                config_ok = False
            
            # Test core imports
            try:
                from core.config import CONFIG
                core_ok = True
            except ImportError:
                core_ok = False
            
            sys.path = original_path
            
            if not config_ok and not core_ok:
                return {
                    "status": "failed",
                    "message": "Critical imports failed",
                    "details": "Both config and core imports failed"
                }
            
            self.results["checks"]["import_compatibility"] = {
                "status": "passed",
                "message": "Import compatibility verified",
                "details": f"Config: {'✅' if config_ok else '❌'}, Core: {'✅' if core_ok else '❌'}"
            }
            
            print("  ✅ Import compatibility verified")
            return self.results["checks"]["import_compatibility"]
            
        except Exception as e:
            return {
                "status": "failed",
                "message": f"Import compatibility check failed: {e}",
                "details": str(e)
            }
    
    def _check_configuration(self) -> Dict[str, Any]:
        """Check configuration for target environment"""
        print("⚙️ Checking Configuration...")
        
        config_file = self.bundle_path / "config" / f"{self.target_env}_config.json"
        
        if not config_file.exists():
            return {
                "status": "failed",
                "message": f"Environment config not found: {config_file.name}",
                "details": f"Expected: {config_file}"
            }
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Validate config structure
            required_keys = ["debug_mode", "log_level"]
            missing_keys = [key for key in required_keys if key not in config]
            
            if missing_keys:
                return {
                    "status": "failed",
                    "message": f"Invalid config structure",
                    "details": f"Missing keys: {missing_keys}"
                }
            
            self.results["checks"]["configuration"] = {
                "status": "passed",
                "message": f"Configuration valid for {self.target_env}",
                "details": f"Config loaded: {list(config.keys())}"
            }
            
            print(f"  ✅ Configuration valid for {self.target_env}")
            return self.results["checks"]["configuration"]
            
        except Exception as e:
            return {
                "status": "failed",
                "message": f"Configuration check failed: {e}",
                "details": str(e)
            }
    
    def _check_dependencies(self) -> Dict[str, Any]:
        """Check dependencies for target environment"""
        print("📦 Checking Dependencies...")
        
        req_file = self.bundle_path / "requirements.txt"
        if not req_file.exists():
            return {
                "status": "failed",
                "message": "Requirements file not found",
                "details": f"Expected: {req_file}"
            }
        
        try:
            # Check if key dependencies are available
            key_deps = ["streamlit", "boto3", "pandas"]
            missing_deps = []
            
            for dep in key_deps:
                try:
                    __import__(dep.replace('-', '_'))
                except ImportError:
                    missing_deps.append(dep)
            
            if missing_deps:
                return {
                    "status": "failed",
                    "message": f"Missing dependencies: {', '.join(missing_deps)}",
                    "details": f"Install with: pip install {' '.join(missing_deps)}"
                }
            
            self.results["checks"]["dependencies"] = {
                "status": "passed",
                "message": "Dependencies satisfied",
                "details": f"All key dependencies available: {key_deps}"
            }
            
            print("  ✅ Dependencies satisfied")
            return self.results["checks"]["dependencies"]
            
        except Exception as e:
            return {
                "status": "failed",
                "message": f"Dependency check failed: {e}",
                "details": str(e)
            }
    
    def _check_launcher_availability(self) -> Dict[str, Any]:
        """Check if environment-specific launcher is available"""
        print("🚀 Checking Launcher Availability...")
        
        launcher_file = self.bundle_path / "launchers" / f"launch_{self.target_env}.py"
        
        if not launcher_file.exists():
            return {
                "status": "failed",
                "message": f"Launcher not found: launch_{self.target_env}.py",
                "details": f"Expected: {launcher_file}"
            }
        
        # Check if launcher is executable
        if not os.access(launcher_file, os.X_OK):
            return {
                "status": "failed",
                "message": f"Launcher not executable: {launcher_file.name}",
                "details": f"Make executable with: chmod +x {launcher_file}"
            }
        
        self.results["checks"]["launcher_availability"] = {
            "status": "passed",
            "message": f"Launcher available: {launcher_file.name}",
            "details": f"Launcher ready for {self.target_env} deployment"
        }
        
        print(f"  ✅ Launcher available: launch_{self.target_env}.py")
        return self.results["checks"]["launcher_availability"]
    
    def _check_validation_tools(self) -> Dict[str, Any]:
        """Check if validation tools are available"""
        print("🧪 Checking Validation Tools...")
        
        validation_dir = self.bundle_path / "validation"
        if not validation_dir.exists():
            return {
                "status": "failed",
                "message": "Validation directory not found",
                "details": f"Expected: {validation_dir}"
            }
        
        validation_script = validation_dir / "run_validation.py"
        if not validation_script.exists():
            return {
                "status": "failed",
                "message": "Validation script not found",
                "details": f"Expected: {validation_script}"
            }
        
        self.results["checks"]["validation_tools"] = {
            "status": "passed",
            "message": "Validation tools available",
            "details": f"Validation script ready: {validation_script}"
        }
        
        print("  ✅ Validation tools available")
        return self.results["checks"]["validation_tools"]
    
    def get_results(self) -> Dict[str, Any]:
        """Get detailed results"""
        return self.results
    
    def save_results(self, output_file: str = "deployment_preflight_results.json"):
        """Save results to file"""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"📄 Results saved to: {output_file}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deployment Pre-Flight Checker")
    parser.add_argument("--env", default="auto", 
                       choices=["auto", "local", "sagemaker", "docker", "lambda"],
                       help="Target environment (default: auto-detect)")
    parser.add_argument("--save-results", help="Save results to file")
    
    args = parser.parse_args()
    
    preflight = DeploymentPreFlight(target_env=args.env)
    success = preflight.run_deployment_preflight()
    
    if args.save_results:
        preflight.save_results(args.save_results)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
