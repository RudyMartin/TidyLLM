#!/usr/bin/env python3
"""
Test with Real Documents
"""

import sys
from pathlib import Path

def test_pdf_extraction():
    print("Testing PDF extraction with real documents...")
    
    test_pdf = Path("input/tests/test_document.pdf")
    if test_pdf.exists():
        print("Found test document: " + str(test_pdf))
        
        try:
            import PyPDF2
            with open(test_pdf, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
            print("PyPDF2 extracted " + str(len(text)) + " characters")
            print("First 200 chars: " + text[:200] + "...")
        except Exception as e:
            print("PyPDF2 failed: " + str(e))
        
        try:
            import pdfplumber
            with pdfplumber.open(test_pdf) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
            print("pdfplumber extracted " + str(len(text)) + " characters")
            print("First 200 chars: " + text[:200] + "...")
        except Exception as e:
            print("pdfplumber failed: " + str(e))
    else:
        print("Test document not found!")

def test_favorites_prompt():
    print("\nTesting Favorites Prompt with real document...")
    try:
        from scripts.demo_favorites_prompt import FavoritesPromptDemo
        demo = FavoritesPromptDemo()
        print("FavoritesPromptDemo created successfully")
        
        # Test with 1 paper
        results = demo.run_demo(1)
        print("Demo results: " + str(results))
        
    except Exception as e:
        print("Favorites Prompt failed: " + str(e))

def main():
    print("Testing with Real Documents")
    print("=" * 30)
    
    test_pdf_extraction()
    test_favorites_prompt()
    
    print("\nTest completed!")

if __name__ == "__main__":
    main()
