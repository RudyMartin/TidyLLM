#!/usr/bin/env python3
"""
Simple PDF Sorter for Knowledge Base
====================================

Sorts PDFs into sop, checklist, modeling based on filename keywords.
No Unicode characters - Windows console friendly.
"""

import os
import shutil
from pathlib import Path

def sort_pdfs():
    """Sort PDFs into the three folders"""
    
    source_dir = Path("knowledge_base/pdfs")
    base_dir = Path("knowledge_base")
    
    # Create target folders
    sop_dir = base_dir / "sop"
    checklist_dir = base_dir / "checklist"
    modeling_dir = base_dir / "modeling"
    
    for folder in [sop_dir, checklist_dir, modeling_dir]:
        folder.mkdir(exist_ok=True)
    
    # Classification rules
    checklist_keywords = [
        'supervisory', 'board', 'regulatory', 'stress-testing', 'bcbs', 
        'instructions', 'reporting', 'compliance', 'sr-', 'occ'
    ]
    
    modeling_keywords = [
        'model-validation', 'validation', 'ml-model', 'algorithm', 'method',
        'tool', 'technical', 'statistical', 'distract', 'gem15', 'abench',
        'longmem', 'mich'
    ]
    
    print("SIMPLE PDF SORTER")
    print("="*50)
    print(f"Source: {source_dir}")
    print("="*50)
    
    if not source_dir.exists():
        print("ERROR: knowledge_base/pdfs folder not found!")
        return
    
    pdf_files = list(source_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found to sort")
        return
    
    sorted_counts = {'checklist': 0, 'modeling': 0, 'sop': 0}
    
    for pdf_file in pdf_files:
        filename_lower = pdf_file.name.lower()
        
        # Determine category
        if any(keyword in filename_lower for keyword in checklist_keywords):
            category = 'checklist'
            target_dir = checklist_dir
        elif any(keyword in filename_lower for keyword in modeling_keywords):
            category = 'modeling'
            target_dir = modeling_dir
        else:
            category = 'sop'
            target_dir = sop_dir
        
        # Move file
        try:
            target_path = target_dir / pdf_file.name
            shutil.move(str(pdf_file), str(target_path))
            sorted_counts[category] += 1
            print(f"[{category.upper():9}] {pdf_file.name}")
        except Exception as e:
            print(f"[ERROR] Failed to move {pdf_file.name}: {e}")
    
    print("="*50)
    print("SORTING COMPLETE")
    print("="*50)
    print(f"CHECKLIST: {sorted_counts['checklist']} files -> knowledge_base/checklist/")
    print(f"MODELING:  {sorted_counts['modeling']} files -> knowledge_base/modeling/")
    print(f"SOP:       {sorted_counts['sop']} files -> knowledge_base/sop/")
    print("="*50)
    print("Ready for one-click domain RAG builder!")

if __name__ == "__main__":
    sort_pdfs()