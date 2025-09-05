#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dependency Validation Script

Validates that all required dependencies are available and working correctly.
This script helps identify missing dependencies before running the application.
"""

import sys
import subprocess
import importlib
from pathlib import Path


def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"🐍 Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        print("❌ Python 3.11+ required")
        return False
    else:
        print("✅ Python version OK")
        return True


def check_import(module_name, package_name=None):
    """Check if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {package_name or module_name}")
        return True
    except ImportError:
        print(f"❌ {package_name or module_name} - NOT AVAILABLE")
        return False


def check_yaml_libraries():
    """Check YAML libraries specifically"""
    print("\n📋 YAML Libraries:")
    
    yaml_libs = [
        ('yaml', 'PyYAML'),
        ('ruamel.yaml', 'ruamel-yaml'),
        ('ruamel_yaml', 'ruamel_yaml (alt)')
    ]
    
    available = []
    for module, name in yaml_libs:
        if check_import(module, name):
            available.append(name)
    
    if not available:
        print("⚠️ No YAML libraries available - will use fallback parser")
    else:
        print(f"✅ YAML processing available with: {', '.join(available)}")
    
    return available


def check_core_dependencies():
    """Check core application dependencies"""
    print("\n📦 Core Dependencies:")
    
    core_deps = [
        ('streamlit', 'Streamlit'),
        ('plotly', 'Plotly'),
        ('pypdf', 'PyPDF'),
        ('docx', 'python-docx'),
        ('openpyxl', 'openpyxl'),
        ('PIL', 'Pillow'),
        ('boto3', 'boto3'),
        ('watchdog', 'watchdog')
    ]
    
    missing = []
    for module, name in core_deps:
        if not check_import(module, name):
            missing.append(name)
    
    return missing


def check_optional_dependencies():
    """Check optional dependencies"""
    print("\n🔧 Optional Dependencies:")
    
    optional_deps = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('mlflow', 'MLflow'),
        ('dspy', 'DSPy'),
        ('datatable', 'datatable')
    ]
    
    available = []
    for module, name in optional_deps:
        if check_import(module, name):
            available.append(name)
            print(f"✅ {name}")
        else:
            print(f"⚠️ {name} - Optional, not required")
    
    return available


def test_yaml_worker():
    """Test the YAML worker specifically"""
    print("\n🧪 Testing YAML Worker:")
    
    try:
        # Add the workers directory to path
        workers_path = Path("src/backend/mcp/workers")
        if workers_path.exists():
            sys.path.append(str(workers_path))
            
            from yaml_processing_worker import YAMLProcessingWorker
            
            worker = YAMLProcessingWorker()
            stats = worker.get_processing_stats()
            
            print(f"✅ YAML Worker initialized successfully")
            print(f"   Available libraries: {stats['available_libraries']}")
            print(f"   Mode: {stats['mode']}")
            
            # Test with a simple YAML file
            test_yaml = """
            test:
              key: value
              number: 42
              list: [1, 2, 3]
            """
            
            test_file = Path("test_dependencies.yaml")
            with open(test_file, 'w') as f:
                f.write(test_yaml)
            
            result = worker.process_yaml_file(str(test_file))
            
            if result['success']:
                print(f"✅ YAML parsing test successful using {result['library_used']}")
            else:
                print(f"❌ YAML parsing test failed: {result['error']}")
            
            # Clean up
            test_file.unlink()
            
            return True
            
        else:
            print("❌ YAML worker not found")
            return False
            
    except Exception as e:
        print(f"❌ YAML worker test failed: {e}")
        return False


def install_missing_dependencies(missing_deps):
    """Install missing dependencies"""
    if not missing_deps:
        return True
    
    print(f"\n📥 Installing missing dependencies: {', '.join(missing_deps)}")
    
    for dep in missing_deps:
        try:
            print(f"Installing {dep}...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', dep], 
                         check=True, capture_output=True, text=True)
            print(f"✅ {dep} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {dep}: {e}")
            return False
    
    return True


def main():
    """Main validation function"""
    print("🔍 Dependency Validation Report")
    print("=" * 50)
    
    # Check Python version
    python_ok = check_python_version()
    
    # Check YAML libraries
    yaml_libs = check_yaml_libraries()
    
    # Check core dependencies
    missing_core = check_core_dependencies()
    
    # Check optional dependencies
    optional_available = check_optional_dependencies()
    
    # Test YAML worker
    yaml_worker_ok = test_yaml_worker()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SUMMARY:")
    print(f"Python Version: {'✅ OK' if python_ok else '❌ FAILED'}")
    print(f"YAML Libraries: {len(yaml_libs)} available")
    print(f"Core Dependencies: {len(missing_core)} missing")
    print(f"Optional Dependencies: {len(optional_available)} available")
    print(f"YAML Worker: {'✅ OK' if yaml_worker_ok else '❌ FAILED'}")
    
    if missing_core:
        print(f"\n❌ Missing core dependencies: {', '.join(missing_core)}")
        print("Run: pip install -r requirements_streamlit.txt")
        return False
    
    if not yaml_libs:
        print("\n⚠️ No YAML libraries available - some features may not work")
        print("Recommended: pip install PyYAML")
    
    if not yaml_worker_ok:
        print("\n❌ YAML worker test failed")
        return False
    
    print("\n✅ All core dependencies are available!")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
