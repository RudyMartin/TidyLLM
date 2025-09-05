#!/usr/bin/env python3
"""
Demo Apps Test Script
Tests all Streamlit demo apps to ensure they're working correctly
"""

import subprocess
import time
import sys
import os
from pathlib import Path

def test_streamlit_import():
    """Test if Streamlit is available"""
    try:
        import streamlit
        print("✅ Streamlit is available")
        return True
    except ImportError:
        print("❌ Streamlit not found. Install with: pip install streamlit")
        return False

def test_app_imports():
    """Test if all demo apps can be imported"""
    apps = [
        "src/main.py",
        "src/enhanced_qa_demo.py", 
        "src/rag_query_demo.py",
        "src/mcp_dashboard.py"
    ]
    
    print("\n🔍 Testing app imports...")
    all_good = True
    
    for app in apps:
        try:
            # Test basic syntax
            with open(app, 'r') as f:
                content = f.read()
            
            # Try to compile
            compile(content, app, 'exec')
            print(f"✅ {app} - Syntax OK")
            
        except Exception as e:
            print(f"❌ {app} - Error: {e}")
            all_good = False
    
    return all_good

def test_core_imports():
    """Test if core modules can be imported"""
    print("\n🔍 Testing core module imports...")
    
    # Add src to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    
    modules = [
        "backend.mcp.orchestrators.qa_orchestrator",
        "backend.mcp.orchestrators.rag_qa_orchestrator", 
        "backend.mcp.orchestrators.llm_enhanced_qa_orchestrator",
        "backend.core.qa_report_generator",
        "backend.personas.control_risks_yaml_config"
    ]
    
    all_good = True
    
    for module in modules:
        try:
            __import__(module)
            print(f"✅ {module} - Import OK")
        except Exception as e:
            print(f"❌ {module} - Error: {e}")
            all_good = False
    
    return all_good

def test_database_connection():
    """Test database connection"""
    print("\n🔍 Testing database connection...")
    
    # Add src to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    
    try:
        from backend.config.credential_manager import credential_manager
        db_config = credential_manager.get_database_config()
        database_url = db_config.get('url')
        
        if database_url and database_url != "your_database_connection_string_here":
            print("✅ Database connection configured")
            return True
        else:
            print("⚠️ Database connection not configured")
            return False
            
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return False

def test_port_availability():
    """Test if demo ports are available"""
    print("\n🔍 Testing port availability...")
    
    ports = [8501, 8502, 8503, 8504]
    all_available = True
    
    for port in ports:
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            
            if result == 0:
                print(f"⚠️ Port {port} is in use")
                all_available = False
            else:
                print(f"✅ Port {port} is available")
                
        except Exception as e:
            print(f"❌ Error checking port {port}: {e}")
            all_available = False
    
    return all_available

def test_sample_documents():
    """Test if sample documents exist"""
    print("\n🔍 Testing sample documents...")
    
    input_dir = Path("input")
    if not input_dir.exists():
        print("⚠️ Input directory not found")
        return False
    
    sample_files = list(input_dir.glob("*.pdf"))
    if sample_files:
        print(f"✅ Found {len(sample_files)} sample documents")
        for file in sample_files[:3]:  # Show first 3
            print(f"   - {file.name}")
        return True
    else:
        print("⚠️ No sample PDF documents found in input/")
        return False

def test_config_files():
    """Test if config files exist"""
    print("\n🔍 Testing config files...")
    
    config_files = [
        "dev_configs/qa_criteria_full.yaml",
        "dev_configs/qa_criteria_simplified.yaml",
        "src/config/qa_criteria_full.yaml",
        "src/config/qa_criteria_simplified.yaml"
    ]
    
    all_exist = True
    
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"✅ {config_file}")
        else:
            print(f"❌ {config_file} - Missing")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests"""
    print("🧪 VectorQA Sage Demo Apps Test")
    print("=" * 50)
    
    tests = [
        ("Streamlit Import", test_streamlit_import),
        ("App Imports", test_app_imports),
        ("Core Module Imports", test_core_imports),
        ("Database Connection", test_database_connection),
        ("Port Availability", test_port_availability),
        ("Sample Documents", test_sample_documents),
        ("Config Files", test_config_files)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} - Exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Demo apps are ready.")
        print("\n🚀 To launch demo apps:")
        print("   ./launch_demo.sh")
        print("   or")
        print("   python3 scripts/launch_demo_apps.py")
    else:
        print("⚠️ Some tests failed. Please fix issues before demo.")
        print("\n🔧 Common fixes:")
        print("   - Install missing dependencies: pip install streamlit")
        print("   - Check database connection: python3 scripts/debug_credentials.py")
        print("   - Add sample documents to input/ directory")
        print("   - Free up ports 8501-8504 if in use")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
