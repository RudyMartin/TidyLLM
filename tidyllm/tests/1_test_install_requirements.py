#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test 1: Install Requirements Verification

Verifies that all required packages are installed and importable.
Tests the minimal core requirements and optional dependencies.
"""

import sys
import importlib
import pytest
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

class TestInstallRequirements:
    """Test suite for verifying required packages are installed"""
    
    def test_python_version(self):
        """Test that Python version is >= 3.8"""
        assert sys.version_info >= (3, 8), f"Python >= 3.8 required, got {sys.version_info}"
        print(f"✅ Python version: {sys.version}")
    
    def test_core_dependencies(self):
        """Test core dependencies are installed and importable"""
        core_packages = [
            'requests',
            'typing_extensions'
        ]
        
        for package in core_packages:
            try:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
                print(f"✅ {package}: {version}")
                assert module is not None, f"Failed to import {package}"
            except ImportError as e:
                if package == 'typing_extensions' and sys.version_info >= (3, 9):
                    print(f"⚠️  {package}: Not needed for Python >= 3.9")
                    continue
                pytest.fail(f"Core dependency missing: {package} - {e}")
    
    def test_tidyllm_imports(self):
        """Test that TidyLLM core modules can be imported"""
        try:
            import tidyllm
            print(f"✅ tidyllm imported successfully")
            
            # Test basic functionality
            from tidyllm import llm_message, LLMMessage
            print(f"✅ tidyllm.llm_message imported")
            print(f"✅ tidyllm.LLMMessage imported")
            
            # Create a test message
            msg = llm_message("Test message")
            assert isinstance(msg, LLMMessage), "llm_message should return LLMMessage instance"
            print(f"✅ llm_message works correctly")
            
        except ImportError as e:
            pytest.fail(f"TidyLLM import failed: {e}")
    
    def test_optional_mlflow_availability(self):
        """Test MLflow integration availability (optional)"""
        try:
            import mlflow
            version = mlflow.__version__
            print(f"✅ mlflow available: {version}")
            
            # Test gateway availability
            from mlflow import gateway
            print(f"✅ mlflow.gateway available")
            
            return True
        except ImportError:
            print(f"⚠️  mlflow not installed (optional)")
            return False
    
    def test_optional_aws_sdk_availability(self):
        """Test AWS SDK availability (optional)"""
        try:
            import boto3
            import botocore
            boto3_version = boto3.__version__
            print(f"✅ boto3 available: {boto3_version}")
            return True
        except ImportError:
            print(f"⚠️  AWS SDK (boto3) not installed (optional)")
            return False
    
    def test_optional_yaml_support(self):
        """Test YAML support availability (optional)"""
        try:
            import yaml
            version = getattr(yaml, '__version__', 'unknown')
            print(f"✅ PyYAML available: {version}")
            return True
        except ImportError:
            print(f"⚠️  PyYAML not installed (optional)")
            return False
    
    def test_optional_database_support(self):
        """Test database support availability (optional)"""
        postgres_available = False
        try:
            import psycopg2
            version = psycopg2.__version__
            print(f"✅ psycopg2 available: {version}")
            postgres_available = True
        except ImportError:
            print(f"⚠️  psycopg2 not installed (optional)")
        
        return postgres_available
    
    def test_optional_document_processing(self):
        """Test document processing libraries availability (optional)"""
        available_libs = {}
        
        optional_libs = [
            ('PIL', 'Pillow'),
            ('pypdf', 'PyPDF'),
            ('openpyxl', 'OpenPyXL'),
            ('docx', 'python-docx'),
            ('bs4', 'BeautifulSoup4')
        ]
        
        for lib_import, lib_name in optional_libs:
            try:
                module = importlib.import_module(lib_import)
                version = getattr(module, '__version__', 'unknown')
                print(f"✅ {lib_name} available: {version}")
                available_libs[lib_name] = True
            except ImportError:
                print(f"⚠️  {lib_name} not installed (optional)")
                available_libs[lib_name] = False
        
        return available_libs
    
    def test_optional_data_processing(self):
        """Test data processing libraries availability (optional)"""
        available_libs = {}
        
        # Test Polars
        try:
            import polars as pl
            version = pl.__version__
            print(f"✅ Polars available: {version}")
            available_libs['polars'] = True
        except ImportError:
            print(f"⚠️  Polars not installed (optional)")
            available_libs['polars'] = False
        
        # Test Datatable
        try:
            import datatable as dt
            version = dt.__version__
            print(f"✅ Datatable available: {version}")
            available_libs['datatable'] = True
        except ImportError:
            print(f"⚠️  Datatable not installed (optional)")
            available_libs['datatable'] = False
        
        # Test Pandas (for visual layer)
        try:
            import pandas as pd
            version = pd.__version__
            print(f"✅ Pandas available: {version}")
            available_libs['pandas'] = True
        except ImportError:
            print(f"⚠️  Pandas not installed (optional)")
            available_libs['pandas'] = False
        
        return available_libs
    
    def test_installation_summary(self):
        """Generate installation summary report"""
        print("\n" + "="*60)
        print("  INSTALLATION SUMMARY")
        print("="*60)
        
        # Core status
        print(f"✅ Core TidyLLM: Installed and working")
        
        # Optional features
        features = {
            "MLflow Integration": self.test_optional_mlflow_availability(),
            "AWS SDK": self.test_optional_aws_sdk_availability(), 
            "YAML Support": self.test_optional_yaml_support(),
            "Database Support": self.test_optional_database_support(),
        }
        
        data_libs = self.test_optional_data_processing()
        doc_libs = self.test_optional_document_processing()
        
        print(f"\nOptional Features:")
        for feature, available in features.items():
            status = "✅ Available" if available else "⚠️  Not Installed"
            print(f"   {feature}: {status}")
        
        print(f"\nData Processing:")
        for lib, available in data_libs.items():
            status = "✅ Available" if available else "⚠️  Not Installed"
            print(f"   {lib}: {status}")
        
        print(f"\nDocument Processing:")
        for lib, available in doc_libs.items():
            status = "✅ Available" if available else "⚠️  Not Installed"
            print(f"   {lib}: {status}")
        
        # Installation recommendations
        missing_features = [name for name, avail in features.items() if not avail]
        if missing_features:
            print(f"\n📋 To enable additional features:")
            if not features["MLflow Integration"]:
                print("   pip install tidyllm[mlflow]")
            if not features["AWS SDK"]:
                print("   pip install boto3")
            if not features["YAML Support"]:
                print("   pip install pyyaml")
            if not features["Database Support"]:
                print("   pip install psycopg2-binary")
        
        print("="*60)

def test_priority_requirements_check():
    """Priority test to ensure basic requirements are met"""
    # This runs first to catch major issues early
    import sys
    assert sys.version_info >= (3, 8), "Python >= 3.8 required"
    
    try:
        import requests
        import tidyllm
        print("SUCCESS: Core requirements verified")
    except ImportError as e:
        pytest.fail(f"CRITICAL: Core requirements not met: {e}")

if __name__ == "__main__":
    # Run the priority test directly
    test_priority_requirements_check()
    print("PASSED: Installation requirements verified")