#!/usr/bin/env python3
"""
QA Embeddings Demo for Validation Reports - Fixed Version
=========================================================

Demonstrates how to use embeddings for question-answering on validation reports
using our collected research papers as the knowledge base.

Usage:
    python qa_embeddings_demo_fixed.py
"""

import sys
import os
from pathlib import Path
import json
from typing import List, Dict, Any, Tuple

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from sentence.tfidf.embeddings import fit_transform
    from sentence.utils.similarity import cosine_similarity
    EMBEDDINGS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Embeddings not available: {e}")
    EMBEDDINGS_AVAILABLE = False

# Sample validation report questions
VALIDATION_QUESTIONS = [
    "What are the mathematical frameworks for signal decomposition?",
    "How do transformer attention mechanisms work?",
    "What are the key challenges in noise separation?",
    "Which adaptive filtering methods are most effective?",
    "How do you evaluate information quality in research papers?"
]

class ValidationReportQA:
    """QA system for validation reports using embeddings."""
    
    def __init__(self, knowledge_base_path: str = "paper_repository/repository_index.json"):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.papers = []
        self.paper_texts = []
        self.embeddings = None
        
    def load_knowledge_base(self) -> bool:
        """Load papers from repository index."""
        try:
            if not self.knowledge_base_path.exists():
                print(f"Knowledge base not found: {self.knowledge_base_path}")
                return False
                
            with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.papers = []
            self.paper_texts = []
            
            for paper_id, paper_info in data.get("papers", {}).items():
                self.papers.append({
                    "id": paper_id,
                    "title": paper_info.get("title", ""),
                    "authors": paper_info.get("authors", []),
                    "y_score": paper_info.get("y_score", 0),
                    "collections": paper_info.get("collections", [])
                })
                
                # Create searchable text from paper metadata
                title = paper_info.get('title', '')
                authors = ' '.join(paper_info.get('authors', []))
                text = f"{title} {authors}"
                self.paper_texts.append(text)
            
            print(f"Loaded {len(self.papers)} papers from knowledge base")
            return True
            
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
            return False
    
    def build_embeddings(self) -> bool:
        """Build embeddings for the knowledge base."""
        if not EMBEDDINGS_AVAILABLE:
            print("Embeddings not available - using simple text matching")
            return False
            
        try:
            if not self.paper_texts:
                print("No papers loaded")
                return False
                
            # Use TidyLLM embeddings - fit_transform returns (embeddings, model)
            self.embeddings, _ = fit_transform(self.paper_texts)
            print(f"Built embeddings: {len(self.embeddings)} vectors, dimension: {len(self.embeddings[0])}")
            return True
            
        except Exception as e:
            print(f"Error building embeddings: {e}")
            return False
    
    def answer_question(self, question: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Answer a question using embeddings similarity."""
        if not EMBEDDINGS_AVAILABLE or self.embeddings is None:
            return self._fallback_search(question, top_k)
        
        try:
            # Get question embedding
            question_embeddings, _ = fit_transform([question])
            question_embedding = question_embeddings[0]
            
            # Calculate similarities
            similarities = []
            for i, paper_embedding in enumerate(self.embeddings):
                similarity = cosine_similarity(question_embedding, paper_embedding)
                similarities.append((i, similarity))
            
            # Sort by similarity and get top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_results = similarities[:top_k]
            
            # Format results
            results = []
            for idx, similarity in top_results:
                paper = self.papers[idx]
                results.append({
                    "paper_id": paper["id"],
                    "title": paper["title"],
                    "authors": paper["authors"],
                    "y_score": paper["y_score"],
                    "collections": paper["collections"],
                    "similarity": round(similarity, 4)
                })
            
            return results
            
        except Exception as e:
            print(f"Error answering question: {e}")
            return self._fallback_search(question, top_k)
    
    def _fallback_search(self, question: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Fallback text matching when embeddings unavailable."""
        question_words = set(question.lower().split())
        
        scores = []
        for i, (paper, text) in enumerate(zip(self.papers, self.paper_texts)):
            text_words = set(text.lower().split())
            overlap = len(question_words & text_words)
            score = overlap / len(question_words) if question_words else 0
            scores.append((i, score))
        
        # Sort and get top results
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in scores[:top_k]:
            paper = self.papers[idx]
            results.append({
                "paper_id": paper["id"],
                "title": paper["title"],
                "authors": paper["authors"],
                "y_score": paper["y_score"],
                "collections": paper["collections"],
                "similarity": round(score, 4)
            })
        
        return results
    
    def run_demo_qa(self):
        """Run demo QA session."""
        print("\\n" + "="*80)
        print("VALIDATION REPORT QA EMBEDDINGS DEMO")
        print("="*80)
        
        # Load knowledge base
        if not self.load_knowledge_base():
            print("Failed to load knowledge base")
            return
        
        # Build embeddings
        embedding_status = self.build_embeddings()
        method = "TidyLLM Embeddings" if embedding_status else "Text Matching Fallback"
        print(f"Method: {method}")
        
        # Run sample questions
        print("\\nSAMPLE VALIDATION QUESTIONS & ANSWERS:")
        print("-" * 80)
        
        for i, question in enumerate(VALIDATION_QUESTIONS, 1):
            print(f"\\nQuestion {i}: {question}")
            
            results = self.answer_question(question, top_k=2)
            
            if results:
                print("Top Relevant Papers:")
                for j, result in enumerate(results, 1):
                    print(f"   {j}. {result['title'][:60]}...")
                    print(f"      Authors: {', '.join(result['authors'][:2])}")
                    print(f"      Y-Score: {result['y_score']}")
                    print(f"      Similarity: {result['similarity']}")
                    print(f"      Collections: {', '.join(result['collections'])}")
            else:
                print("   No relevant papers found")
        
        print("\\n" + "="*80)
        print("DEMO SUMMARY")
        print("="*80)
        print(f"Knowledge Base: {len(self.papers)} papers loaded")
        print(f"Embedding Method: {method}")
        print(f"Collections Available: {len(set(col for paper in self.papers for col in paper['collections']))}")
        
        # Show collections
        collections = {}
        for paper in self.papers:
            for col in paper['collections']:
                if col not in collections:
                    collections[col] = []
                collections[col].append(paper['title'][:40] + "...")
        
        print("\\nCollections in Knowledge Base:")
        for col, papers in collections.items():
            print(f"  - {col}: {len(papers)} papers")
        
        print("\\nDemo completed successfully!")

def main():
    """Main demo function."""
    print("Starting Validation Report QA Embeddings Demo...")
    
    # Change to whitepapers directory
    demo_dir = Path(__file__).parent
    os.chdir(demo_dir)
    
    # Create and run QA system
    qa_system = ValidationReportQA()
    qa_system.run_demo_qa()

if __name__ == "__main__":
    main()