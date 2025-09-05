#!/usr/bin/env python3
"""
Deployment Structure Validator

Validates that the deployment bundle has the correct structure and imports
before deployment to catch issues early.
"""

import os
import sys
import ast
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple

class DeploymentValidator:
    """Validates deployment bundle structure"""
    
    def __init__(self, bundle_path: str):
        self.bundle_path = Path(bundle_path)
        self.errors = []
        self.warnings = []
    
    def validate(self) -> bool:
        """Run all validation checks"""
        print("🔍 Validating deployment bundle structure...")
        print("=" * 50)
        
        # Run validation checks
        self._check_directory_structure()
        self._check_import_structure()
        self._check_configuration_files()
        self._test_imports()
        self._check_dependencies()
        
        # Report results
        self._report_results()
        
        return len(self.errors) == 0
    
    def _check_directory_structure(self):
        """Check that required directories exist"""
        print("📁 Checking directory structure...")
        
        required_dirs = [
            "src",
            "config", 
            "database",
            "scripts"
        ]
        
        for dir_name in required_dirs:
            dir_path = self.bundle_path / dir_name
            if not dir_path.exists():
                self.errors.append(f"Missing required directory: {dir_name}")
            else:
                print(f"✅ {dir_name}/ exists")
    
    def _check_import_structure(self):
        """Check for problematic import patterns"""
        print("\n🔍 Checking import structure...")
        
        src_path = self.bundle_path / "src"
        if not src_path.exists():
            self.errors.append("src directory not found")
            return
        
        relative_imports = []
        circular_imports = []
        
        for py_file in src_path.rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                # Check for relative imports
                if "from .." in content:
                    relative_imports.append(str(py_file.relative_to(src_path)))
                
                # Parse AST to check for circular imports
                try:
                    tree = ast.parse(content)
                    imports = self._extract_imports(tree)
                    # TODO: Add circular import detection logic
                except SyntaxError:
                    self.errors.append(f"Syntax error in {py_file.relative_to(src_path)}")
                    
            except Exception as e:
                self.warnings.append(f"Could not analyze {py_file.name}: {e}")
        
        if relative_imports:
            self.warnings.append(f"Found {len(relative_imports)} files with relative imports")
            for file in relative_imports[:3]:  # Show first 3
                print(f"⚠️  Relative imports in: {file}")
        else:
            print("✅ No relative imports found")
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements from AST"""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        return imports
    
    def _check_configuration_files(self):
        """Check configuration files"""
        print("\n⚙️  Checking configuration files...")
        
        config_files = [
            "config/settings.py",
            "config/environments.py",
            "deployment_config.json"
        ]
        
        for config_file in config_files:
            file_path = self.bundle_path / config_file
            if not file_path.exists():
                self.warnings.append(f"Missing config file: {config_file}")
            else:
                print(f"✅ {config_file} exists")
    
    def _test_imports(self):
        """Test that critical imports work"""
        print("\n🧪 Testing critical imports...")
        
        # Add bundle src to Python path
        src_path = self.bundle_path / "src"
        config_path = self.bundle_path / "config"
        
        original_path = sys.path.copy()
        try:
            sys.path.insert(0, str(src_path))
            sys.path.insert(0, str(config_path))
            
            # Test critical imports
            test_imports = [
                ("config.settings", "CONFIG"),
                ("backend.core.config", "CONFIG"),
                ("t_dashboard", "tab_evaluation_dashboard")
            ]
            
            for module_name, attr_name in test_imports:
                try:
                    module = __import__(module_name, fromlist=[attr_name])
                    getattr(module, attr_name)
                    print(f"✅ {module_name}.{attr_name} import successful")
                except ImportError as e:
                    self.errors.append(f"Import failed: {module_name}.{attr_name} - {e}")
                except AttributeError as e:
                    self.errors.append(f"Attribute error: {module_name}.{attr_name} - {e}")
                    
        finally:
            sys.path = original_path
    
    def _check_dependencies(self):
        """Check that dependencies are properly specified"""
        print("\n📦 Checking dependencies...")
        
        req_files = [
            "requirements.txt",
            "requirements_demo.txt"
        ]
        
        for req_file in req_files:
            file_path = self.bundle_path / req_file
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        deps = f.read().strip().split('\n')
                        deps = [d.strip() for d in deps if d.strip() and not d.startswith('#')]
                        print(f"✅ {req_file}: {len(deps)} dependencies")
                except Exception as e:
                    self.warnings.append(f"Could not read {req_file}: {e}")
            else:
                self.warnings.append(f"Missing {req_file}")
    
    def _report_results(self):
        """Report validation results"""
        print("\n" + "=" * 50)
        print("📊 VALIDATION RESULTS")
        print("=" * 50)
        
        if self.errors:
            print(f"❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"   • {error}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   • {warning}")
        
        if not self.errors and not self.warnings:
            print("✅ All validation checks passed!")
        elif not self.errors:
            print("✅ No critical errors found (warnings can be ignored)")
        else:
            print("❌ Critical errors found - deployment not recommended")

def main():
    """Main validation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate deployment bundle structure")
    parser.add_argument("bundle_path", help="Path to deployment bundle")
    args = parser.parse_args()
    
    validator = DeploymentValidator(args.bundle_path)
    success = validator.validate()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
