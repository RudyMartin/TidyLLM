#!/usr/bin/env python3
"""
Simple QA Demo for Validation Reports
====================================

A working demonstration of question-answering on validation reports
using text matching on our research paper collection.
"""

import json
from pathlib import Path
from typing import List, Dict, Any

class SimpleQA:
    """Simple QA system using text matching."""
    
    def __init__(self):
        self.papers = []
        
    def load_papers(self, kb_path="paper_repository/repository_index.json"):
        """Load papers from knowledge base."""
        try:
            with open(kb_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for paper_id, info in data.get("papers", {}).items():
                self.papers.append({
                    "id": paper_id,
                    "title": info.get("title", ""),
                    "authors": [name.encode('ascii', 'ignore').decode('ascii') for name in info.get("authors", [])],
                    "y_score": info.get("y_score", 0),
                    "collections": info.get("collections", []),
                    "searchable": f"{info.get('title', '')} {' '.join(info.get('authors', []))}".lower()
                })
            
            print(f"Loaded {len(self.papers)} papers")
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False
    
    def search(self, question: str, top_k=3):
        """Search papers using simple text matching."""
        question_words = set(question.lower().split())
        
        scores = []
        for paper in self.papers:
            text_words = set(paper["searchable"].split())
            overlap = len(question_words & text_words)
            score = overlap / len(question_words) if question_words else 0
            scores.append((paper, score))
        
        # Sort by score and return top results
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(paper, score) for paper, score in scores[:top_k] if score > 0]
    
    def demo_qa(self):
        """Run QA demo."""
        questions = [
            "mathematical frameworks signal decomposition",
            "transformer attention mechanisms", 
            "noise separation challenges",
            "adaptive filtering methods",
            "information quality evaluation"
        ]
        
        print("="*70)
        print("VALIDATION REPORT QA DEMO")
        print("="*70)
        
        if not self.load_papers():
            print("Failed to load papers")
            return
            
        # Show collections
        collections = {}
        for paper in self.papers:
            for col in paper['collections']:
                collections[col] = collections.get(col, 0) + 1
                
        print("\\nCollections Available:")
        for col, count in collections.items():
            print(f"  - {col}: {count} papers")
        
        print("\\nQUESTION & ANSWER EXAMPLES:")
        print("-" * 70)
        
        for i, question in enumerate(questions, 1):
            print(f"\\nQ{i}: {question}")
            
            results = self.search(question, top_k=2)
            
            if results:
                print("Relevant Papers:")
                for j, (paper, score) in enumerate(results, 1):
                    title = paper['title'][:50] + "..." if len(paper['title']) > 50 else paper['title']
                    print(f"  {j}. {title}")
                    print(f"     Y-Score: {paper['y_score']}, Match: {score:.2f}")
                    print(f"     Collections: {', '.join(paper['collections'])}")
            else:
                print("  No matches found")
        
        print("\\n" + "="*70)
        print("DEMO STATISTICS")
        print("="*70)
        print(f"Total Papers: {len(self.papers)}")
        print(f"Total Collections: {len(collections)}")
        print(f"Average Y-Score: {sum(p['y_score'] for p in self.papers)/len(self.papers):.2f}")
        
        # Show top papers by Y-score
        top_papers = sorted(self.papers, key=lambda x: x['y_score'], reverse=True)[:5]
        print("\\nTop 5 Papers by Y-Score:")
        for i, paper in enumerate(top_papers, 1):
            title = paper['title'][:40] + "..." if len(paper['title']) > 40 else paper['title']
            print(f"  {i}. {title} (Y-Score: {paper['y_score']})")

if __name__ == "__main__":
    print("Starting Simple QA Demo...")
    qa = SimpleQA()
    qa.demo_qa()
    print("Demo completed!")