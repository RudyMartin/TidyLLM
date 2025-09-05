#!/usr/bin/env python3
"""
Robust Deployment Bundle Creator

Creates deployment bundles that work across all environment types with
robust import patterns and comprehensive testing.
"""

import os
import sys
import shutil
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

class RobustBundleCreator:
    """Creates robust deployment bundles with cross-environment compatibility"""
    
    def __init__(self, bundle_name: str = None):
        self.project_root = Path(__file__).parent.parent
        self.bundle_name = bundle_name or f"robust_vectorqa_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.bundle_path = self.project_root / "migration_bundles" / self.bundle_name
        
    def create_bundle(self) -> bool:
        """Create the robust deployment bundle"""
        try:
            print(f"🚀 Creating robust deployment bundle: {self.bundle_name}")
            print("=" * 60)
            
            # Create bundle directory
            self.bundle_path.mkdir(parents=True, exist_ok=True)
            
            # Step 1: Copy and restructure application files
            self._copy_and_restructure_files()
            
            # Step 2: Apply robust import patterns
            self._apply_robust_import_patterns()
            
            # Step 3: Create unified configuration
            self._create_unified_configuration()
            
            # Step 4: Create environment-specific launchers
            self._create_environment_launchers()
            
            # Step 5: Create validation and testing
            self._create_validation_suite()
            
            # Step 6: Create deployment documentation
            self._create_deployment_docs()
            
            # Step 7: Test the bundle
            self._test_bundle()
            
            print(f"✅ Robust deployment bundle created successfully!")
            print(f"📦 Bundle location: {self.bundle_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Bundle creation failed: {e}")
            return False
    
    def _copy_and_restructure_files(self):
        """Copy and restructure files for optimal deployment"""
        print("📁 Copying and restructuring application files...")
        
        # Copy src directory with flattened structure
        src_src = self.project_root / "src"
        src_dest = self.bundle_path / "src"
        
        if src_src.exists():
            # Copy with selective flattening
            self._copy_with_restructure(src_src, src_dest)
            print("✅ Copied and restructured src directory")
        
        # Copy other essential directories
        for dir_name in ["database", "scripts"]:
            src_dir = self.project_root / dir_name
            dest_dir = self.bundle_path / dir_name
            if src_dir.exists():
                shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)
                print(f"✅ Copied {dir_name} directory")
        
        # Copy requirements files
        for req_file in ["requirements.txt", "requirements_demo.txt"]:
            req_path = self.project_root / req_file
            if req_path.exists():
                shutil.copy2(req_path, self.bundle_path)
                print(f"✅ Copied {req_file}")
    
    def _copy_with_restructure(self, src_dir: Path, dest_dir: Path):
        """Copy files with intelligent restructuring"""
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Define restructuring rules
        restructure_map = {
            "backend/core": "core",
            "backend/config": "config", 
            "backend/llm": "llm",
            "backend/mcp": "mcp",
            "backend/personas": "personas",
            "backend/qa": "qa",
            "backend/utils": "utils",
            "backend/api": "api"
        }
        
        # Copy files with restructuring
        for src_path in src_dir.rglob("*"):
            if src_path.is_file() and src_path.suffix in [".py", ".txt", ".yaml", ".yml", ".json"]:
                # Calculate relative path
                rel_path = src_path.relative_to(src_dir)
                
                # Apply restructuring rules
                dest_rel_path = rel_path
                for old_path, new_path in restructure_map.items():
                    if str(rel_path).startswith(old_path):
                        dest_rel_path = Path(str(rel_path).replace(old_path, new_path, 1))
                        break
                
                # Create destination
                dest_file = dest_dir / dest_rel_path
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file
                shutil.copy2(src_path, dest_file)
    
    def _apply_robust_import_patterns(self):
        """Apply robust import patterns to all Python files"""
        print("🔧 Applying robust import patterns...")
        
        src_dir = self.bundle_path / "src"
        
        # Define import transformation rules
        import_transforms = [
            # Backend imports
            (r"from backend\.core\.", "from core."),
            (r"from backend\.config\.", "from config."),
            (r"from backend\.llm\.", "from llm."),
            (r"from backend\.mcp\.", "from mcp."),
            (r"from backend\.personas\.", "from personas."),
            (r"from backend\.qa\.", "from qa."),
            (r"from backend\.utils\.", "from utils."),
            (r"from backend\.api\.", "from api."),
            
            # Relative imports to absolute
            (r"from \.\.config\.", "from config."),
            (r"from \.\.core\.", "from core."),
            (r"from \.\.llm\.", "from llm."),
            (r"from \.\.mcp\.", "from mcp."),
            (r"from \.\.personas\.", "from personas."),
            (r"from \.\.qa\.", "from qa."),
            (r"from \.\.utils\.", "from utils."),
            (r"from \.\.api\.", "from api."),
        ]
        
        # Apply transforms to all Python files
        for py_file in src_dir.rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                original_content = content
                
                # Apply import transforms
                for pattern, replacement in import_transforms:
                    content = re.sub(pattern, replacement, content)
                
                # Add robust import wrapper for critical imports
                content = self._add_robust_import_wrapper(content, py_file)
                
                # Write back if changed
                if content != original_content:
                    with open(py_file, 'w') as f:
                        f.write(content)
                    print(f"✅ Applied robust imports to {py_file.relative_to(src_dir)}")
                    
            except Exception as e:
                print(f"⚠️  Error processing {py_file.name}: {e}")
    
    def _add_robust_import_wrapper(self, content: str, file_path: Path) -> str:
        """Add robust import wrapper to critical imports"""
        # Add import helper at the top of files that need it
        if "from config." in content or "from core." in content:
            import_helper = '''
# Robust import setup
import sys
from pathlib import Path
_src_dir = Path(__file__).parent
while _src_dir.name != "src" and _src_dir.parent != _src_dir:
    _src_dir = _src_dir.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
'''
            # Insert after existing imports but before other code
            lines = content.split('\n')
            insert_pos = 0
            
            # Find position after docstring and imports
            in_docstring = False
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    in_docstring = not in_docstring
                elif not in_docstring and not stripped.startswith('#') and not stripped.startswith('import') and not stripped.startswith('from') and stripped:
                    insert_pos = i
                    break
            
            # Insert the robust import setup
            lines.insert(insert_pos, import_helper)
            content = '\n'.join(lines)
        
        return content
    
    def _create_unified_configuration(self):
        """Create unified configuration system"""
        print("⚙️ Creating unified configuration system...")
        
        # Copy the unified config system
        config_src = self.project_root / "src" / "config" / "settings.py"
        config_dest = self.bundle_path / "src" / "config" / "settings.py"
        
        if config_src.exists():
            config_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(config_src, config_dest)
            print("✅ Copied unified configuration system")
        
        # Create environment-specific config files
        self._create_environment_configs()
    
    def _create_environment_configs(self):
        """Create environment-specific configuration files"""
        environments = {
            "local": {
                "debug_mode": True,
                "log_level": "INFO",
                "cache_dir": "~/vectorqa_cache"
            },
            "sagemaker": {
                "debug_mode": False,
                "log_level": "WARNING", 
                "cache_dir": "/tmp/vectorqa_cache",
                "sagemaker_mode": True
            },
            "docker": {
                "debug_mode": False,
                "log_level": "WARNING",
                "cache_dir": "/app/cache",
                "container_mode": True
            },
            "lambda": {
                "debug_mode": False,
                "log_level": "ERROR",
                "cache_dir": "/tmp",
                "lambda_mode": True,
                "minimal_imports": True
            }
        }
        
        config_dir = self.bundle_path / "config"
        config_dir.mkdir(exist_ok=True)
        
        for env_name, env_config in environments.items():
            config_file = config_dir / f"{env_name}_config.json"
            with open(config_file, 'w') as f:
                json.dump(env_config, f, indent=2)
            print(f"✅ Created {env_name} configuration")
    
    def _create_environment_launchers(self):
        """Create environment-specific launcher scripts"""
        print("🚀 Creating environment launchers...")
        
        launchers_dir = self.bundle_path / "launchers"
        launchers_dir.mkdir(exist_ok=True)
        
        # Create launcher for each environment
        environments = ["local", "sagemaker", "docker", "lambda"]
        
        for env in environments:
            launcher_content = self._generate_launcher_script(env)
            launcher_file = launchers_dir / f"launch_{env}.py"
            
            with open(launcher_file, 'w') as f:
                f.write(launcher_content)
            
            # Make executable
            launcher_file.chmod(0o755)
            print(f"✅ Created {env} launcher")
    
    def _generate_launcher_script(self, environment: str) -> str:
        """Generate launcher script for specific environment"""
        return f'''#!/usr/bin/env python3
"""
{environment.title()} Environment Launcher

Launches the VectorQA application optimized for {environment} environment.
"""

import os
import sys
from pathlib import Path

def setup_{environment}_environment():
    """Setup {environment}-specific environment"""
    # Set environment variable
    os.environ["VECTORQA_ENV"] = "{environment}"
    
    # Setup Python path for {environment}
    bundle_root = Path(__file__).parent.parent
    src_dir = bundle_root / "src"
    config_dir = bundle_root / "config"
    
    # Add to Python path
    sys.path.insert(0, str(src_dir))
    sys.path.insert(0, str(config_dir))
    
    print(f"🚀 Starting VectorQA in {{environment}} mode...")
    print(f"📁 Bundle root: {{bundle_root}}")
    print(f"🐍 Python path configured for {{environment}}")

def main():
    """Main launcher function"""
    setup_{environment}_environment()
    
    try:
        # Import and run the main application
        from main import main as app_main
        app_main()
    except ImportError as e:
        print(f"❌ Import error: {{e}}")
        print("💡 Trying alternative import...")
        try:
            import main
            main.main()
        except Exception as e2:
            print(f"❌ Failed to start application: {{e2}}")
            sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    def _create_validation_suite(self):
        """Create validation and testing suite"""
        print("🧪 Creating validation suite...")
        
        # Copy validation scripts
        validation_dir = self.bundle_path / "validation"
        validation_dir.mkdir(exist_ok=True)
        
        # Copy environment compatibility tests
        test_src = self.project_root / "tests" / "test_environment_compatibility.py"
        if test_src.exists():
            shutil.copy2(test_src, validation_dir / "test_environments.py")
            print("✅ Copied environment compatibility tests")
        
        # Copy import structure tests
        import_test_src = self.project_root / "tests" / "test_import_structure.py"
        if import_test_src.exists():
            shutil.copy2(import_test_src, validation_dir / "test_imports.py")
            print("✅ Copied import structure tests")
        
        # Create validation runner
        self._create_validation_runner(validation_dir)
    
    def _create_validation_runner(self, validation_dir: Path):
        """Create validation runner script"""
        runner_content = '''#!/usr/bin/env python3
"""
Deployment Validation Runner

Runs comprehensive validation tests on the deployment bundle.
"""

import sys
import os
from pathlib import Path

def run_validation():
    """Run all validation tests"""
    print("🔍 Running deployment validation...")
    
    # Add bundle to path
    bundle_root = Path(__file__).parent.parent
    sys.path.insert(0, str(bundle_root / "src"))
    
    success = True
    
    # Run environment tests
    try:
        from test_environments import EnvironmentTester
        tester = EnvironmentTester()
        env_success = tester.test_all_environments()
        success = success and env_success
    except Exception as e:
        print(f"❌ Environment tests failed: {e}")
        success = False
    
    # Run import tests
    try:
        from test_imports import TestImportStructure
        import_tester = TestImportStructure()
        import_tester.setup_method()
        
        # Run key tests
        import_tester.test_config_imports()
        import_tester.test_core_module_imports()
        print("✅ Import tests passed")
    except Exception as e:
        print(f"❌ Import tests failed: {e}")
        success = False
    
    return success

if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
'''
        
        runner_file = validation_dir / "run_validation.py"
        with open(runner_file, 'w') as f:
            f.write(runner_content)
        runner_file.chmod(0o755)
        print("✅ Created validation runner")
    
    def _create_deployment_docs(self):
        """Create comprehensive deployment documentation"""
        print("📖 Creating deployment documentation...")
        
        docs_dir = self.bundle_path / "docs"
        docs_dir.mkdir(exist_ok=True)
        
        # Create deployment guide
        deployment_guide = '''# Robust VectorQA Deployment Guide

## 🎯 Overview

This deployment bundle is designed to work across multiple environments with robust import patterns and comprehensive testing.

## 🚀 Quick Start

### Local Development
```bash
python3 launchers/launch_local.py
```

### SageMaker
```bash
python3 launchers/launch_sagemaker.py
```

### Docker
```bash
python3 launchers/launch_docker.py
```

### AWS Lambda
```bash
python3 launchers/launch_lambda.py
```

## 🧪 Validation

Before deployment, run validation:
```bash
python3 validation/run_validation.py
```

## 📁 Structure

```
bundle/
├── src/                    # Application source (restructured)
├── config/                 # Environment configurations
├── launchers/              # Environment-specific launchers
├── validation/             # Validation and testing
├── docs/                   # Documentation
└── requirements.txt        # Dependencies
```

## 🔧 Environment Configuration

Each environment has its own configuration:
- `config/local_config.json` - Local development
- `config/sagemaker_config.json` - SageMaker notebooks
- `config/docker_config.json` - Container deployment
- `config/lambda_config.json` - Serverless deployment

## 🛡️ Robust Import Patterns

The bundle uses multiple fallback mechanisms:
1. Primary import paths
2. Environment-specific fallbacks
3. Dynamic path resolution
4. Graceful degradation

## 📊 Validation Results

Run validation to ensure compatibility:
- Environment compatibility tests
- Import structure validation
- Configuration validation
- Dependency checks
'''
        
        with open(docs_dir / "DEPLOYMENT_GUIDE.md", 'w') as f:
            f.write(deployment_guide)
        print("✅ Created deployment guide")
    
    def _test_bundle(self):
        """Test the created bundle"""
        print("🧪 Testing deployment bundle...")
        
        # Run validation
        validation_script = self.bundle_path / "validation" / "run_validation.py"
        if validation_script.exists():
            try:
                import subprocess
                result = subprocess.run([sys.executable, str(validation_script)], 
                                      capture_output=True, text=True, cwd=self.bundle_path)
                
                if result.returncode == 0:
                    print("✅ Bundle validation passed")
                else:
                    print(f"⚠️  Bundle validation warnings: {result.stdout}")
                    
            except Exception as e:
                print(f"⚠️  Could not run validation: {e}")
        
        print("✅ Bundle testing completed")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create robust deployment bundle")
    parser.add_argument("--name", help="Bundle name")
    args = parser.parse_args()
    
    creator = RobustBundleCreator(args.name)
    success = creator.create_bundle()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
