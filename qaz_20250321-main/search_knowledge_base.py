#!/usr/bin/env python3
"""
Knowledge Base Search Tool
Search through the knowledge base for papers and documents
"""

import os
from pathlib import Path
import json

class KnowledgeBaseSearcher:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.knowledge_base_dir = self.project_root / "knowledge_base"
        
    def search_papers(self, query, search_type="filename"):
        """Search for papers in the knowledge base"""
        results = []
        
        if not self.knowledge_base_dir.exists():
            print(f"❌ Knowledge base directory not found: {self.knowledge_base_dir}")
            return results
            
        # Search through all PDF files
        for pdf_path in self.knowledge_base_dir.rglob("*.pdf"):
            filename = pdf_path.name.lower()
            relative_path = pdf_path.relative_to(self.project_root)
            
            if search_type == "filename":
                if query.lower() in filename:
                    results.append({
                        'filename': pdf_path.name,
                        'path': str(relative_path),
                        'size_mb': pdf_path.stat().st_size / (1024 * 1024),
                        'category': self._get_category(relative_path)
                    })
            elif search_type == "path":
                if query.lower() in str(relative_path).lower():
                    results.append({
                        'filename': pdf_path.name,
                        'path': str(relative_path),
                        'size_mb': pdf_path.stat().st_size / (1024 * 1024),
                        'category': self._get_category(relative_path)
                    })
        
        return results
    
    def _get_category(self, relative_path):
        """Extract category from path"""
        parts = relative_path.parts
        if len(parts) >= 3:
            return f"{parts[1]}/{parts[2]}"
        elif len(parts) >= 2:
            return parts[1]
        return "unknown"
    
    def list_all_papers(self):
        """List all papers in the knowledge base"""
        all_papers = []
        
        for pdf_path in self.knowledge_base_dir.rglob("*.pdf"):
            relative_path = pdf_path.relative_to(self.project_root)
            all_papers.append({
                'filename': pdf_path.name,
                'path': str(relative_path),
                'size_mb': pdf_path.stat().st_size / (1024 * 1024),
                'category': self._get_category(relative_path)
            })
        
        return all_papers
    
    def get_paper_info(self, filename):
        """Get detailed info about a specific paper"""
        for pdf_path in self.knowledge_base_dir.rglob("*.pdf"):
            if pdf_path.name == filename:
                relative_path = pdf_path.relative_to(self.project_root)
                return {
                    'filename': pdf_path.name,
                    'path': str(relative_path),
                    'full_path': str(pdf_path),
                    'size_mb': pdf_path.stat().st_size / (1024 * 1024),
                    'category': self._get_category(relative_path),
                    'exists': True
                }
        
        return {'exists': False, 'filename': filename}

def main():
    searcher = KnowledgeBaseSearcher()
    
    print("🔍 KNOWLEDGE BASE SEARCH TOOL")
    print("=" * 50)
    
    # Search for FLASH ATTENTION specifically
    print("\n🎯 Searching for FLASH ATTENTION paper...")
    flash_results = searcher.search_papers("flash attention", "filename")
    
    if flash_results:
        print(f"✅ Found {len(flash_results)} FLASH ATTENTION papers:")
        for paper in flash_results:
            print(f"   📄 {paper['filename']}")
            print(f"      📁 {paper['path']}")
            print(f"      📊 {paper['size_mb']:.1f} MB")
            print(f"      🏷️  {paper['category']}")
            print()
    else:
        print("❌ No FLASH ATTENTION papers found")
    
    # Search for attention-related papers
    print("\n🔍 Searching for attention-related papers...")
    attention_results = searcher.search_papers("attention", "filename")
    
    if attention_results:
        print(f"✅ Found {len(attention_results)} attention-related papers:")
        for paper in attention_results:
            print(f"   📄 {paper['filename']}")
            print(f"      📁 {paper['path']}")
            print(f"      📊 {paper['size_mb']:.1f} MB")
            print(f"      🏷️  {paper['category']}")
            print()
    
    # Show all papers in attention_mechanisms category
    print("\n📚 All papers in attention_mechanisms category:")
    attention_category = searcher.search_papers("attention_mechanisms", "path")
    
    if attention_category:
        for paper in attention_category:
            print(f"   📄 {paper['filename']}")
            print(f"      📁 {paper['path']}")
            print(f"      📊 {paper['size_mb']:.1f} MB")
            print()
    
    # Show summary
    print("\n📊 KNOWLEDGE BASE SUMMARY:")
    all_papers = searcher.list_all_papers()
    
    categories = {}
    total_size = 0
    
    for paper in all_papers:
        category = paper['category']
        if category not in categories:
            categories[category] = {'count': 0, 'size': 0}
        categories[category]['count'] += 1
        categories[category]['size'] += paper['size_mb']
        total_size += paper['size_mb']
    
    print(f"   📄 Total papers: {len(all_papers)}")
    print(f"   📊 Total size: {total_size:.1f} MB")
    print(f"   📁 Categories: {len(categories)}")
    
    print("\n📁 By Category:")
    for category, stats in sorted(categories.items()):
        print(f"   🏷️  {category}: {stats['count']} papers ({stats['size']:.1f} MB)")

if __name__ == "__main__":
    main()
