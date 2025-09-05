#!/usr/bin/env python3
"""
Simple Test - Check Core Functionality
"""

import sys
from pathlib import Path

def test_1_imports():
    print("1. Testing basic imports...")
    try:
        import streamlit
        print("   OK: streamlit")
    except:
        print("   FAIL: streamlit")
    
    try:
        import PyPDF2
        print("   OK: PyPDF2")
    except:
        print("   FAIL: PyPDF2")

def test_2_favorites_prompt():
    print("\n2. Testing favorites prompt...")
    try:
        from scripts.demo_favorites_prompt import FavoritesPromptDemo
        demo = FavoritesPromptDemo()
        print("   OK: FavoritesPromptDemo created")
    except Exception as e:
        print("   FAIL: " + str(e))

def test_3_llm_gateway():
    print("\n3. Testing LLM gateway...")
    try:
        from backend.llm.unified_llm_gateway import UnifiedLLMGateway
        print("   OK: UnifiedLLMGateway imported")
    except Exception as e:
        print("   FAIL: " + str(e))

def test_4_document_functions():
    print("\n4. Testing document functions...")
    try:
        with open("simple_demo.py", "r") as f:
            content = f.read()
            if "extract_text_from_pdf" in content:
                print("   OK: extract_text_from_pdf function exists")
            else:
                print("   FAIL: extract_text_from_pdf function missing")
    except Exception as e:
        print("   FAIL: " + str(e))

def main():
    print("Simple Core Functionality Test")
    print("=" * 30)
    
    test_1_imports()
    test_2_favorites_prompt()
    test_3_llm_gateway()
    test_4_document_functions()
    
    print("\nTest completed!")

if __name__ == "__main__":
    main()
