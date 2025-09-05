#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo Preparation Script
Comprehensive testing and validation for VectorQA Sage demo
"""

import sys
import os
import subprocess
import time
from pathlib import Path
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def run_command(command, description, timeout=30):
    """Run a command and return success status"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        if result.returncode == 0:
            print(f"✅ {description} - SUCCESS")
            return True, result.stdout
        else:
            print(f"❌ {description} - FAILED")
            print(f"   Error: {result.stderr}")
            return False, result.stderr
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - TIMEOUT")
        return False, "Timeout"
    except Exception as e:
        print(f"💥 {description} - ERROR: {e}")
        return False, str(e)

def test_database_connection():
    """Test database connectivity"""
    print("\n" + "="*60)
    print("🗄️  DATABASE CONNECTION TEST")
    print("="*60)
    
    success, output = run_command(
        "python3 scripts/test_database_connection.py",
        "Database Connection Test"
    )
    
    if success:
        print("✅ Database connection is working")
    else:
        print("⚠️  Database connection failed - will use mock data")
    
    return success

def test_mock_live_context():
    """Test mock live context system"""
    print("\n" + "="*60)
    print("🎭 MOCK LIVE CONTEXT TEST")
    print("="*60)
    
    success, output = run_command(
        "python3 scripts/test_mock_live_context.py",
        "Mock Live Context System"
    )
    
    if success:
        print("✅ Mock live context is working")
        # Extract key metrics from output
        if "Success Rate: 100.0%" in output:
            print("✅ Performance metrics are accurate")
    else:
        print("❌ Mock live context failed")
    
    return success

def test_mcp_hierarchy():
    """Test MCP hierarchy components"""
    print("\n" + "="*60)
    print("🤖 MCP HIERARCHY TEST")
    print("="*60)
    
    success, output = run_command(
        "python3 tests/test_mcp_hierarchy_simple.py",
        "MCP Hierarchy Components"
    )
    
    if success:
        print("✅ MCP hierarchy is working")
        if "All tests passed" in output or "SUCCESS" in output:
            print("✅ All MCP components are functional")
    else:
        print("❌ MCP hierarchy test failed")
    
    return success

def test_live_context_integration():
    """Test live context integration"""
    print("\n" + "="*60)
    print("🔗 LIVE CONTEXT INTEGRATION TEST")
    print("="*60)
    
    success, output = run_command(
        "python3 tests/test_live_context_integration.py",
        "Live Context Integration"
    )
    
    if success:
        print("✅ Live context integration is working")
    else:
        print("⚠️  Live context integration failed - will use fallback")
    
    return success

def check_streamlit_apps():
    """Check if Streamlit apps can be imported"""
    print("\n" + "="*60)
    print("📱 STREAMLIT APPS CHECK")
    print("="*60)
    
    apps_to_check = [
        ("src/rag_query_demo.py", "RAG Query Demo"),
        ("src/enhanced_qa_demo.py", "Enhanced QA Demo"),
        ("src/mcp_dashboard.py", "MCP Dashboard"),
        ("src/qa_demo.py", "Basic QA Demo")
    ]
    
    all_working = True
    for app_path, app_name in apps_to_check:
        try:
            # Try to import the app
            spec = subprocess.run(
                f"python3 -c 'import sys; sys.path.append(\"src\"); exec(open(\"{app_path}\").read())'",
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            if spec.returncode == 0:
                print(f"✅ {app_name} - Import successful")
            else:
                print(f"❌ {app_name} - Import failed")
                print(f"   Error: {spec.stderr}")
                all_working = False
        except Exception as e:
            print(f"❌ {app_name} - Error: {e}")
            all_working = False
    
    return all_working

def check_sample_documents():
    """Check for sample documents"""
    print("\n" + "="*60)
    print("📄 SAMPLE DOCUMENTS CHECK")
    print("="*60)
    
    data_dir = Path("data/input/reviews")
    if data_dir.exists():
        pdf_files = list(data_dir.glob("*.pdf"))
        if pdf_files:
            print(f"✅ Found {len(pdf_files)} sample PDF documents")
            for pdf in pdf_files[:3]:  # Show first 3
                print(f"   📄 {pdf.name}")
            if len(pdf_files) > 3:
                print(f"   ... and {len(pdf_files) - 3} more")
            return True
        else:
            print("⚠️  No PDF files found in data/input/reviews/")
            return False
    else:
        print("⚠️  data/input/reviews/ directory not found")
        return False

def generate_demo_status_report():
    """Generate comprehensive demo status report"""
    print("\n" + "="*60)
    print("📊 DEMO STATUS REPORT")
    print("="*60)
    
    # Test all components
    db_working = test_database_connection()
    mock_working = test_mock_live_context()
    mcp_working = test_mcp_hierarchy()
    integration_working = test_live_context_integration()
    streamlit_working = check_streamlit_apps()
    docs_available = check_sample_documents()
    
    # Generate summary
    print("\n" + "="*60)
    print("🎯 DEMO READINESS SUMMARY")
    print("="*60)
    
    components = [
        ("Database Connection", db_working),
        ("Mock Live Context", mock_working),
        ("MCP Hierarchy", mcp_working),
        ("Live Context Integration", integration_working),
        ("Streamlit Apps", streamlit_working),
        ("Sample Documents", docs_available)
    ]
    
    working_count = sum(1 for _, status in components if status)
    total_count = len(components)
    
    print(f"Overall Status: {working_count}/{total_count} components working")
    
    for component, status in components:
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {component}")
    
    # Demo recommendations
    print("\n" + "="*60)
    print("💡 DEMO RECOMMENDATIONS")
    print("="*60)
    
    if working_count >= 5:
        print("🎉 EXCELLENT - Demo is ready to go!")
        print("   All core components are working")
        print("   Focus on showcasing the full pipeline")
    elif working_count >= 4:
        print("👍 GOOD - Demo is mostly ready")
        print("   Core functionality is working")
        print("   Have backup scenarios ready")
    elif working_count >= 3:
        print("⚠️  FAIR - Demo needs preparation")
        print("   Some components need attention")
        print("   Focus on working features")
    else:
        print("🚨 NEEDS WORK - Demo requires fixes")
        print("   Multiple components are failing")
        print("   Consider postponing or using mock demos")
    
    # Specific recommendations
    if not db_working:
        print("   💡 Use mock data for live context demo")
    if not streamlit_working:
        print("   💡 Prepare command-line demos as backup")
    if not docs_available:
        print("   💡 Create sample documents for demo")
    
    return working_count >= 4  # Return True if demo is ready

def create_quick_start_guide():
    """Create a quick start guide for the demo"""
    print("\n" + "="*60)
    print("📋 QUICK START GUIDE")
    print("="*60)
    
    guide = """
# VectorQA Sage Demo - Quick Start Guide

## 🚀 Launch Commands

### 1. MCP Dashboard (System Overview)
```bash
cd src
streamlit run mcp_dashboard.py
```

### 2. Enhanced QA Demo (Document Processing)
```bash
cd src
streamlit run enhanced_qa_demo.py
```

### 3. RAG Query Demo (Intelligent Search)
```bash
cd src
streamlit run rag_query_demo.py
```

### 4. Basic QA Demo (Simple Processing)
```bash
cd src
streamlit run qa_demo.py
```

## 🧪 Test Commands

### Mock Live Context Test
```bash
python3 scripts/test_mock_live_context.py
```

### MCP Hierarchy Test
```bash
python3 tests/test_mcp_hierarchy_simple.py
```

### Live Context Integration Test
```bash
python3 tests/test_live_context_integration.py
```

## 📄 Sample Documents
Location: `data/input/reviews/`
Use any PDF files in this directory for document processing demos.

## 🎯 Demo Flow
1. Start with MCP Dashboard to show system architecture
2. Use Enhanced QA Demo for document processing
3. Switch to RAG Query Demo for intelligent search
4. Show test results to prove system reliability

## 🚨 Emergency Commands
If Streamlit apps fail, use these command-line demos:
```bash
python3 scripts/test_mock_live_context.py
python3 scripts/test_rag_with_database.py
```
"""
    
    # Save guide to file
    with open("DEMO_QUICK_START.md", "w") as f:
        f.write(guide)
    
    print("✅ Quick start guide saved to DEMO_QUICK_START.md")
    print("📋 Use this guide during the demo for quick reference")

def main():
    """Main demo preparation function"""
    print("🚀 VectorQA Sage Demo Preparation")
    print("="*60)
    print("Testing all components for demo readiness...")
    
    # Run comprehensive tests
    demo_ready = generate_demo_status_report()
    
    # Create quick start guide
    create_quick_start_guide()
    
    # Final status
    print("\n" + "="*60)
    if demo_ready:
        print("🎉 DEMO PREPARATION COMPLETE!")
        print("✅ System is ready for demonstration")
        print("📋 Check DEMO_QUICK_START.md for launch commands")
    else:
        print("⚠️  DEMO PREPARATION NEEDS ATTENTION")
        print("🔧 Some components need to be fixed before demo")
        print("📋 Check the status report above for details")
    
    print("="*60)

if __name__ == "__main__":
    main()
