#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Explore References in PDF Documents

Script to explore PDF documents and identify reference sections for bibliography extraction.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def explore_pdf_for_references(pdf_path: str):
    """Explore a PDF document for reference sections"""
    print(f"🔍 Exploring: {pdf_path}")
    print("=" * 60)
    
    try:
        from backend.mcp.workers.modern_pdf_processor import ModernPDFProcessor
        
        processor = ModernPDFProcessor()
        result = processor.process_pdf(pdf_path)
        
        if not result['success']:
            print(f"❌ Failed to process PDF: {result.get('error', 'Unknown error')}")
            return
        
        text_content = result['processing']['text_content']
        lines = text_content.split('\n')
        
        # Look for reference section indicators
        reference_indicators = [
            'references', 'bibliography', 'works cited', 'literature cited',
            'REFERENCES', 'BIBLIOGRAPHY', 'WORKS CITED', 'LITERATURE CITED'
        ]
        
        reference_sections = []
        current_section = None
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            
            # Check if this line indicates a reference section
            for indicator in reference_indicators:
                if indicator in line_lower:
                    print(f"✅ Found reference section: '{line.strip()}' (line {i+1})")
                    current_section = {
                        'start_line': i,
                        'header': line.strip(),
                        'content': []
                    }
                    break
            
            # If we're in a reference section, collect content
            if current_section:
                current_section['content'].append(line)
                
                # Check for end of reference section (next major section)
                if (line.strip().isupper() and len(line.strip()) > 3 and 
                    not any(indicator in line_lower for indicator in reference_indicators) and
                    len(current_section['content']) > 10):
                    current_section['end_line'] = i
                    reference_sections.append(current_section)
                    current_section = None
        
        # If we found reference sections, analyze them
        if reference_sections:
            print(f"\n📊 Found {len(reference_sections)} reference section(s):")
            for i, section in enumerate(reference_sections):
                print(f"\n--- Reference Section {i+1} ---")
                print(f"Header: {section['header']}")
                print(f"Lines: {section['start_line']+1} to {section.get('end_line', 'end')}")
                
                # Show first few reference entries
                content_lines = section['content'][:20]  # First 20 lines
                print("Sample content:")
                for line in content_lines:
                    if line.strip():
                        print(f"  {line.strip()}")
                
                if len(section['content']) > 20:
                    print(f"  ... and {len(section['content']) - 20} more lines")
        else:
            print("❌ No reference sections found")
            
            # Show some sample content to help identify patterns
            print("\n📄 Sample document content:")
            for i, line in enumerate(lines[:50]):
                if line.strip():
                    print(f"  Line {i+1}: {line.strip()}")
        
        return reference_sections
        
    except Exception as e:
        print(f"❌ Error exploring PDF: {e}")
        return None

def main():
    """Main function to explore multiple PDFs"""
    print("🔍 PDF Reference Section Explorer")
    print("=" * 60)
    
    # List of PDFs to explore (focusing on academic papers)
    pdf_files = [
        "input/omnibus/all/2309.11495 - Chain-of-Verification Reduces Hallucination in Large Language Models.pdf",
        "input/omnibus/all/2209.07067.pdf",
        "input/omnibus/all/2210.03102.pdf",
        "input/omnibus/all/2205.14415.pdf",
        "input/omnibus/all/2012.10619.pdf",
        "input/omnibus/all/1806.00663.pdf",
        "input/omnibus/all/1606.08339v1_zhao.pdf",
        "input/omnibus/all/2403.05530v3.pdf",
        "input/omnibus/all/2406.07496v1_teerxtgrad.pdf",
        "input/omnibus/all/2410.05258v1_dif_transformer.pdf"
    ]
    
    found_references = []
    
    for pdf_file in pdf_files:
        if os.path.exists(pdf_file):
            reference_sections = explore_pdf_for_references(pdf_file)
            if reference_sections:
                found_references.append({
                    'file': pdf_file,
                    'sections': reference_sections
                })
            print("\n" + "="*60 + "\n")
        else:
            print(f"⚠️  File not found: {pdf_file}")
    
    # Summary
    print("📊 Summary of Reference Sections Found:")
    print("=" * 60)
    
    if found_references:
        for item in found_references:
            print(f"📄 {os.path.basename(item['file'])}: {len(item['sections'])} reference section(s)")
    else:
        print("❌ No reference sections found in any documents")
    
    return found_references

if __name__ == "__main__":
    main()
