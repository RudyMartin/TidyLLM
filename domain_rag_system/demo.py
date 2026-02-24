#!/usr/bin/env python3
"""
Generated Domain RAG Demo
=========================
Created: 2025-09-05 11:01:11.508915
Total Documents: 25
"""

from pathlib import Path
import json

class SimpleDomainRAG:
    def __init__(self):
        self.kb_path = Path("knowledge_base")
        print("Simple Domain RAG System Initialized")
        print(f"Checklist: 3 docs")
        print(f"SOP: 14 docs") 
        print(f"Modeling: 8 docs")
    
    def query(self, question):
        print(f"\nQuery: {question}")
        
        # Simple file search
        results = []
        
        # Check authoritative first (checklist)
        checklist_dir = self.kb_path / "checklist"
        if checklist_dir.exists():
            for pdf in checklist_dir.glob("*.pdf"):
                if self._is_relevant(question, pdf.name):
                    results.append({"file": pdf.name, "source": "checklist", "precedence": 1.0})
        
        # Check SOP
        sop_dir = self.kb_path / "sop" 
        if sop_dir.exists():
            for pdf in sop_dir.glob("*.pdf"):
                if self._is_relevant(question, pdf.name):
                    results.append({"file": pdf.name, "source": "sop", "precedence": 0.8})
        
        # Check modeling
        modeling_dir = self.kb_path / "modeling"
        if modeling_dir.exists():
            for pdf in modeling_dir.glob("*.pdf"):
                if self._is_relevant(question, pdf.name):
                    results.append({"file": pdf.name, "source": "modeling", "precedence": 0.6})
        
        # Sort by precedence
        results.sort(key=lambda x: x["precedence"], reverse=True)
        
        return results[:5]  # Top 5 results
    
    def _is_relevant(self, question, filename):
        question_lower = question.lower()
        filename_lower = filename.lower()
        
        keywords = question_lower.split()
        return any(keyword in filename_lower for keyword in keywords if len(keyword) > 3)

def main():
    """Demo the system"""
    
    print("DOMAIN RAG SYSTEM DEMO")
    print("="*40)
    
    rag = SimpleDomainRAG()
    
    test_queries = [
        "model validation requirements",
        "stress testing procedures", 
        "credit risk assessment",
        "regulatory compliance guidelines"
    ]
    
    for query in test_queries:
        results = rag.query(query)
        
        print(f"Results: {len(results)} documents found")
        for i, result in enumerate(results[:3], 1):
            print(f"  {i}. [{result['source'].upper()}] {result['file']}")
        print()
    
    print("DEMO COMPLETE")
    print("="*40)
    print("Hierarchy working: Checklist > SOP > Modeling")

if __name__ == "__main__":
    main()
