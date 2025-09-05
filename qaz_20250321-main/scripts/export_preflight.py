#!/usr/bin/env python3
"""
Export Pre-Flight Checker

Pre-flight checks before packing for export.
Ensures the codebase is ready for deployment bundle creation.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional

class ExportPreFlight:
    """Pre-flight checks for export/bundle creation"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.results = {
            "checks": {},
            "overall_status": "unknown",
            "recommendations": []
        }
    
    def run_export_preflight(self) -> bool:
        """Run comprehensive export pre-flight checks"""
        print("📦 EXPORT PRE-FLIGHT CHECKS")
        print("=" * 60)
        print(f"📁 Project Root: {self.project_root}")
        print()
        
        # Run export-specific checks
        checks = [
            self._check_code_quality(),
            self._check_import_structure(),
            self._check_configuration_files(),
            self._check_dependencies(),
            self._check_security(),
            self._check_file_structure(),
            self._check_robust_tools(),
            self._check_environment_tests()
        ]
        
        # Determine overall status
        passed_checks = sum(1 for check in checks if check.get('status') == 'passed')
        total_checks = len(checks)
        
        self.results["overall_status"] = "passed" if passed_checks == total_checks else "failed"
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 EXPORT PRE-FLIGHT SUMMARY")
        print("=" * 60)
        print(f"✅ Passed: {passed_checks}/{total_checks}")
        print(f"📊 Status: {self.results['overall_status'].upper()}")
        
        if self.results["overall_status"] == "passed":
            print("🎉 Export pre-flight checks PASSED!")
            print("📦 Ready for bundle creation!")
        else:
            print("❌ Export pre-flight checks FAILED!")
            print("🔧 Please fix issues before export")
        
        # Print recommendations
        if self.results["recommendations"]:
            print("\n💡 RECOMMENDATIONS:")
            for rec in self.results["recommendations"]:
                print(f"  • {rec}")
        
        return self.results["overall_status"] == "passed"
    
    def _check_code_quality(self) -> Dict[str, Any]:
        """Check code quality and syntax"""
        print("🔍 Checking Code Quality...")
        
        # Check for syntax errors
        syntax_errors = []
        python_files = list(self.project_root.rglob("*.py"))
        
        for py_file in python_files:
            try:
                with open(py_file, 'r') as f:
                    compile(f.read(), py_file, 'exec')
            except SyntaxError as e:
                syntax_errors.append(f"{py_file.relative_to(self.project_root)}: {e}")
            except Exception as e:
                syntax_errors.append(f"{py_file.relative_to(self.project_root)}: {e}")
        
        if syntax_errors:
            return {
                "status": "failed",
                "message": f"Found {len(syntax_errors)} syntax errors",
                "details": syntax_errors[:5]  # Show first 5
            }
        
        self.results["checks"]["code_quality"] = {
            "status": "passed",
            "message": f"Code quality verified ({len(python_files)} Python files)",
            "details": "No syntax errors found"
        }
        
        print(f"  ✅ Code quality verified ({len(python_files)} Python files)")
        return self.results["checks"]["code_quality"]
    
    def _check_import_structure(self) -> Dict[str, Any]:
        """Check import structure for deployment compatibility"""
        print("🔧 Checking Import Structure...")
        
        # Count relative imports
        relative_imports = []
        src_dir = self.project_root / "src"
        
        if src_dir.exists():
            for py_file in src_dir.rglob("*.py"):
                try:
                    with open(py_file, 'r') as f:
                        content = f.read()
                        if "from .." in content:
                            relative_imports.append(str(py_file.relative_to(self.project_root)))
                except Exception:
                    pass
        
        if relative_imports:
            self.results["recommendations"].append(
                f"Consider converting {len(relative_imports)} relative imports to absolute for better deployment compatibility"
            )
        
        # Check for robust import tools
        import_helper = self.project_root / "src" / "utils" / "import_helper.py"
        if not import_helper.exists():
            return {
                "status": "failed",
                "message": "Robust import helper not found",
                "details": "Missing src/utils/import_helper.py"
            }
        
        self.results["checks"]["import_structure"] = {
            "status": "passed",
            "message": "Import structure ready for deployment",
            "details": f"Found {len(relative_imports)} relative imports (handled by robust tools)"
        }
        
        print(f"  ✅ Import structure ready ({len(relative_imports)} relative imports)")
        return self.results["checks"]["import_structure"]
    
    def _check_configuration_files(self) -> Dict[str, Any]:
        """Check configuration files"""
        print("⚙️ Checking Configuration Files...")
        
        required_configs = [
            "src/config/settings.py",
            "src/config/environments.py",
            "requirements.txt"
        ]
        
        missing_configs = []
        for config in required_configs:
            config_path = self.project_root / config
            if not config_path.exists():
                missing_configs.append(config)
        
        if missing_configs:
            return {
                "status": "failed",
                "message": f"Missing configuration files: {', '.join(missing_configs)}",
                "details": "Required for deployment bundle"
            }
        
        self.results["checks"]["configuration_files"] = {
            "status": "passed",
            "message": "All configuration files present",
            "details": f"Found {len(required_configs)} required config files"
        }
        
        print(f"  ✅ Configuration files ready ({len(required_configs)} files)")
        return self.results["checks"]["configuration_files"]
    
    def _check_dependencies(self) -> Dict[str, Any]:
        """Check dependencies"""
        print("📦 Checking Dependencies...")
        
        # Check requirements files
        req_files = ["requirements.txt", "requirements_demo.txt"]
        missing_req = []
        
        for req_file in req_files:
            req_path = self.project_root / req_file
            if not req_path.exists():
                missing_req.append(req_file)
        
        if missing_req:
            return {
                "status": "failed",
                "message": f"Missing requirements files: {', '.join(missing_req)}",
                "details": "Required for deployment"
            }
        
        # Check key dependencies
        key_deps = ["streamlit", "boto3", "pandas", "numpy"]
        missing_deps = []
        
        for dep in key_deps:
            try:
                __import__(dep.replace('-', '_'))
            except ImportError:
                missing_deps.append(dep)
        
        if missing_deps:
            self.results["recommendations"].append(
                f"Install missing dependencies: pip install {' '.join(missing_deps)}"
            )
        
        self.results["checks"]["dependencies"] = {
            "status": "passed",
            "message": "Dependencies configured",
            "details": f"Requirements files: {len(req_files)}, Key deps: {len(key_deps)}"
        }
        
        print(f"  ✅ Dependencies configured ({len(req_files)} req files)")
        return self.results["checks"]["dependencies"]
    
    def _check_security(self) -> Dict[str, Any]:
        """Check security issues"""
        print("🔒 Checking Security...")
        
        # Check for exposed API keys
        api_key_patterns = [
            r"sk-[a-zA-Z0-9]{48}",
            r"AIza[0-9A-Za-z-_]{35}",
            r"AKIA[0-9A-Z]{16}",
            r"Bearer [a-zA-Z0-9]{32,}"
        ]
        
        exposed_keys = []
        for pattern in api_key_patterns:
            import re
            for py_file in self.project_root.rglob("*.py"):
                try:
                    with open(py_file, 'r') as f:
                        content = f.read()
                        if re.search(pattern, content):
                            exposed_keys.append(str(py_file.relative_to(self.project_root)))
                except Exception:
                    pass
        
        if exposed_keys:
            return {
                "status": "failed",
                "message": f"Found {len(exposed_keys)} files with potential API keys",
                "details": "Security risk - remove API keys before export"
            }
        
        self.results["checks"]["security"] = {
            "status": "passed",
            "message": "Security check passed",
            "details": "No exposed API keys found"
        }
        
        print("  ✅ Security check passed")
        return self.results["checks"]["security"]
    
    def _check_file_structure(self) -> Dict[str, Any]:
        """Check file structure for export"""
        print("📁 Checking File Structure...")
        
        required_dirs = ["src", "tests", "scripts"]
        missing_dirs = []
        
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                missing_dirs.append(dir_name)
        
        if missing_dirs:
            return {
                "status": "failed",
                "message": f"Missing required directories: {', '.join(missing_dirs)}",
                "details": "Required for deployment bundle"
            }
        
        # Check for large files that shouldn't be exported
        large_files = []
        for file_path in self.project_root.rglob("*"):
            if file_path.is_file() and file_path.stat().st_size > 10 * 1024 * 1024:  # 10MB
                large_files.append(str(file_path.relative_to(self.project_root)))
        
        if large_files:
            self.results["recommendations"].append(
                f"Consider excluding large files from export: {', '.join(large_files[:3])}"
            )
        
        self.results["checks"]["file_structure"] = {
            "status": "passed",
            "message": "File structure ready for export",
            "details": f"Required dirs: {len(required_dirs)}, Large files: {len(large_files)}"
        }
        
        print(f"  ✅ File structure ready ({len(required_dirs)} required dirs)")
        return self.results["checks"]["file_structure"]
    
    def _check_robust_tools(self) -> Dict[str, Any]:
        """Check robust deployment tools"""
        print("🛠️ Checking Robust Tools...")
        
        robust_tools = [
            "scripts/create_robust_bundle.py",
            "tests/test_environment_compatibility.py",
            "scripts/validate_deployment_structure.py"
        ]
        
        missing_tools = []
        for tool in robust_tools:
            tool_path = self.project_root / tool
            if not tool_path.exists():
                missing_tools.append(tool)
        
        if missing_tools:
            return {
                "status": "failed",
                "message": f"Missing robust tools: {', '.join(missing_tools)}",
                "details": "Required for deployment bundle creation"
            }
        
        self.results["checks"]["robust_tools"] = {
            "status": "passed",
            "message": "Robust deployment tools available",
            "details": f"Found {len(robust_tools)} robust tools"
        }
        
        print(f"  ✅ Robust tools available ({len(robust_tools)} tools)")
        return self.results["checks"]["robust_tools"]
    
    def _check_environment_tests(self) -> Dict[str, Any]:
        """Check environment compatibility tests"""
        print("🧪 Checking Environment Tests...")
        
        try:
            # Import and test environment compatibility
            sys.path.insert(0, str(self.project_root))
            
            test_module = self.project_root / "tests" / "test_environment_compatibility.py"
            if not test_module.exists():
                return {
                    "status": "failed",
                    "message": "Environment compatibility tests not found",
                    "details": "Missing tests/test_environment_compatibility.py"
                }
            
            # Run a quick test
            from tests.test_environment_compatibility import EnvironmentTester
            tester = EnvironmentTester()
            
            # Test local environment
            try:
                result = tester.test_local_development()
                test_ok = True
            except Exception as e:
                test_ok = False
                test_error = str(e)
            
            if not test_ok:
                return {
                    "status": "failed",
                    "message": "Environment tests failed",
                    "details": f"Local test error: {test_error}"
                }
            
            self.results["checks"]["environment_tests"] = {
                "status": "passed",
                "message": "Environment compatibility tests available",
                "details": "Local environment test passed"
            }
            
            print("  ✅ Environment tests available")
            return self.results["checks"]["environment_tests"]
            
        except Exception as e:
            return {
                "status": "failed",
                "message": f"Environment test check failed: {e}",
                "details": str(e)
            }
    
    def get_results(self) -> Dict[str, Any]:
        """Get detailed results"""
        return self.results
    
    def save_results(self, output_file: str = "export_preflight_results.json"):
        """Save results to file"""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"📄 Results saved to: {output_file}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Export Pre-Flight Checker")
    parser.add_argument("--save-results", help="Save results to file")
    
    args = parser.parse_args()
    
    preflight = ExportPreFlight()
    success = preflight.run_export_preflight()
    
    if args.save_results:
        preflight.save_results(args.save_results)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
