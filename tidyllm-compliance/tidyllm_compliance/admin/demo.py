#!/usr/bin/env python3
"""
TidyLLM Internal Domain RAG Demo
===============================
Self-referential conflict resolution for TidyLLM documentation.

Generated: 2025-09-05T12:30:37.494207
Total Documents: 12
"""

from pathlib import Path
import json

class TidyLLMInternalRAG:
    def __init__(self):
        print("TidyLLM Internal Domain RAG System Initialized")
        print("Purpose: Resolve conflicts in TidyLLM's own documentation")
        
        # Load hierarchy
        with open("manifest.json") as f:
            self.manifest = json.load(f)
        
        hierarchy = self.manifest['hierarchy']
        for level_name, level_data in hierarchy.items():
            count = len(level_data['files'])
            if count > 0:
                print(f"{level_data['level']}: {count} docs")
    
    def query(self, question):
        """Query internal documentation with hierarchical precedence"""
        print(f"\nQuery: {question}")
        
        # Simple relevance matching (would use real embeddings)
        results = []
        
        # Check each hierarchy level
        hierarchy = self.manifest['hierarchy']
        for level_name, level_data in hierarchy.items():
            for doc_path in level_data['files']:
                if self._is_relevant(question, doc_path):
                    results.append({
                        'file': Path(doc_path).name,
                        'level': level_data['level'],
                        'precedence': level_data['precedence'],
                        'path': doc_path
                    })
        
        # Sort by precedence (highest first)
        results.sort(key=lambda x: x['precedence'], reverse=True)
        
        print(f"Results: {len(results)} documents found")
        for i, result in enumerate(results[:3], 1):
            print(f"  {i}. [{result['level']}] {result['file']}")
        
        return results[:5]
    
    def _is_relevant(self, question, doc_path):
        """Check if document is relevant to question"""
        question_lower = question.lower()
        path_lower = doc_path.lower()
        
        # Simple keyword matching
        keywords = question_lower.split()
        return any(keyword in path_lower for keyword in keywords if len(keyword) > 3)

def main():
    """Demo the internal RAG system"""
    
    print("TIDYLLM INTERNAL DOMAIN RAG DEMO")
    print("="*50)
    
    rag = TidyLLMInternalRAG()
    
    # Test queries for conflict resolution
    test_queries = ['What is the official session management pattern?', 'Should we use UnifiedSessionManager or Gateway pattern?', 'Which embedding system is primary: tidyllm-sentence or tidyllm-vectorqa?', 'How should MLflow be integrated with TidyLLM?', 'What is the approved workflow system: RAG2DAG or HeirOS?', 'Which database should be used: PostgreSQL or SQLite?', 'What are the current constraints for this codebase?', 'How should AWS credentials be managed?', 'What is the proper drop zones architecture?', 'Which demos are currently functional?', 'What is the deployment strategy?', 'How should dependencies be managed?', 'What is the current architecture documentation standard?', 'How should new features be documented?', 'What are the deprecated patterns to avoid?']
    
    for query in test_queries[:5]:  # Test first 5 queries
        results = rag.query(query)
        print()
    
    print("DEMO COMPLETE")
    print("="*50)
    print("Hierarchy: CRITICAL > ARCHITECTURE > CURRENT > RECENT > HISTORICAL > EXAMPLES")

if __name__ == "__main__":
    main()
