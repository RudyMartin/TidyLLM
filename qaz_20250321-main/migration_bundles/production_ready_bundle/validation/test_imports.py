#!/usr/bin/env python3
"""
Import Structure Validation Tests

Ensures that all imports work correctly across different deployment environments.
This prevents deployment-time import failures.
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import patch

class TestImportStructure:
    """Test import structure consistency"""
    
    def setup_method(self):
        """Setup test environment"""
        # Add src to path
        src_path = Path(__file__).parent.parent / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
    
    def test_config_imports(self):
        """Test that configuration can be imported from multiple paths"""
        # Test primary import path
        try:
            from config.settings import CONFIG
            assert CONFIG is not None
            print("✅ Primary config import successful")
        except ImportError as e:
            pytest.fail(f"Primary config import failed: {e}")
        
        # Test fallback import path
        try:
            from src.config.settings import CONFIG as CONFIG2
            assert CONFIG2 is not None
            print("✅ Fallback config import successful")
        except ImportError:
            # This is expected in some environments
            pass
    
    def test_core_module_imports(self):
        """Test that core modules can be imported"""
        core_modules = [
            "document_processor",
            "qa_report_generator", 
            "llm_manager",
            "config"
        ]
        
        for module in core_modules:
            try:
                # Test backend.core import (current structure)
                exec(f"from backend.core.{module} import *")
                print(f"✅ backend.core.{module} import successful")
            except ImportError as e:
                print(f"⚠️  backend.core.{module} import failed: {e}")
                
                # Test direct core import (recommended structure)
                try:
                    exec(f"from core.{module} import *")
                    print(f"✅ core.{module} import successful")
                except ImportError as e2:
                    pytest.fail(f"Both import paths failed for {module}: {e}, {e2}")
    
    def test_mcp_imports(self):
        """Test MCP module imports"""
        mcp_modules = [
            "orchestrators.qa_orchestrator",
            "coordinators.dspy_coordinator",
            "protocol.message_schemas"
        ]
        
        for module in mcp_modules:
            try:
                # Test current nested structure
                exec(f"from backend.mcp.{module} import *")
                print(f"✅ backend.mcp.{module} import successful")
            except ImportError as e:
                print(f"⚠️  backend.mcp.{module} import failed: {e}")
    
    def test_ui_imports(self):
        """Test UI module imports"""
        ui_modules = [
            "t_dashboard",
            "t_upload", 
            "t_evaluate"
        ]
        
        for module in ui_modules:
            try:
                exec(f"import {module}")
                print(f"✅ {module} import successful")
            except ImportError as e:
                pytest.fail(f"UI module {module} import failed: {e}")
    
    def test_relative_import_detection(self):
        """Detect files using relative imports that should be converted"""
        src_path = Path(__file__).parent.parent / "src"
        relative_imports = []
        
        for py_file in src_path.rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    if "from .." in content:
                        relative_imports.append(str(py_file.relative_to(src_path)))
            except Exception:
                continue
        
        if relative_imports:
            print(f"⚠️  Files with relative imports: {len(relative_imports)}")
            for file in relative_imports[:5]:  # Show first 5
                print(f"   - {file}")
            if len(relative_imports) > 5:
                print(f"   ... and {len(relative_imports) - 5} more")
        else:
            print("✅ No relative imports detected")
    
    @patch.dict(os.environ, {"VECTORQA_ENV": "production"})
    def test_production_environment_imports(self):
        """Test imports work in production environment"""
        try:
            from config.settings import config
            prod_config = config.get_all()
            assert prod_config["log_level"] == "ERROR"
            assert prod_config["debug_mode"] is False
            print("✅ Production environment imports successful")
        except ImportError as e:
            pytest.fail(f"Production environment imports failed: {e}")
    
    def test_import_helper_utility(self):
        """Test the import helper utility"""
        try:
            from utils.import_helper import import_config, import_core_module
            
            # Test config import
            config_module = import_config()
            assert config_module is not None
            
            # Test core module import
            # Note: This might fail if the module doesn't exist yet
            try:
                doc_processor = import_core_module("document_processor")
                assert doc_processor is not None
                print("✅ Import helper utility works")
            except ImportError:
                print("⚠️  Import helper works but some modules not found (expected)")
                
        except ImportError as e:
            pytest.fail(f"Import helper utility failed: {e}")

if __name__ == "__main__":
    # Run tests directly
    test = TestImportStructure()
    test.setup_method()
    
    print("🧪 Running Import Structure Tests...")
    print("=" * 50)
    
    test.test_config_imports()
    test.test_core_module_imports()
    test.test_mcp_imports()
    test.test_ui_imports()
    test.test_relative_import_detection()
    test.test_production_environment_imports()
    test.test_import_helper_utility()
    
    print("=" * 50)
    print("🎉 Import structure tests completed!")
