#!/usr/bin/env python3
"""
QA Embeddings Demo for Validation Reports
=========================================

Demonstrates how to use embeddings for question-answering on validation reports
using our collected research papers as the knowledge base.

Usage:
    python qa_embeddings_demo.py
"""

import sys
import os
from pathlib import Path
import json
from typing import List, Dict, Any, Tuple
import numpy as np

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from sentence.tfidf.embeddings import TFIDFModel, fit_transform
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
    "How do you evaluate information quality in research papers?",
    "What metrics are used for Y=R+S+N framework assessment?",
    "How does reservoir computing handle signal-noise separation?",
    "What are the advantages of variational mode decomposition?",
    "How do sparse representation methods work for denoising?",
    "What are the performance benefits of mixture-of-head attention?"
]

class ValidationReportQA:
    """QA system for validation reports using embeddings."""
    
    def __init__(self, knowledge_base_path: str = "paper_repository/repository_index.json"):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.papers = []
        self.paper_texts = []
        self.embeddings = None
        self.model = None
        
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
                text = f"{paper_info.get('title', '')} {' '.join(paper_info.get('authors', []))}"
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
                
            # Use TidyLLM embeddings
            self.embeddings, self.model = fit_transform(self.paper_texts)
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
            question_embedding = self.model.transform([question])[0]
            
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
                    "similarity": round(similarity, 4),
                    "text": self.paper_texts[idx]
                })
            
            return results
            
        except Exception as e:
            print(f"Error answering question: {e}")
            return []
    
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
                "similarity": round(score, 4),
                "text": self.paper_texts[idx]
            })
        
        return results
    
    def run_demo_qa(self):
        """Run demo QA session."""
        print("\\n" + "="*80)
        print("VALIDATION REPORT QA EMBEDDINGS DEMO")
        print("="*80)
        
        # Load knowledge base
        if not self.load_knowledge_base():
            print("❌ Failed to load knowledge base")
            return
        
        # Build embeddings
        embedding_status = self.build_embeddings()
        method = "TidyLLM Embeddings" if embedding_status else "Text Matching Fallback"
        print(f"Method: {method}")
        
        # Run sample questions
        print("\\nSAMPLE VALIDATION QUESTIONS & ANSWERS:")
        print("-" * 80)
        
        for i, question in enumerate(VALIDATION_QUESTIONS[:5], 1):
            print(f"\\nQuestion {i}: {question}")
            
            results = self.answer_question(question, top_k=2)
            
            if results:
                print("Top Relevant Papers:")
                for j, result in enumerate(results, 1):
                    print(f"   {j}. {result['title']}")
                    print(f"      Authors: {', '.join(result['authors'])}")
                    print(f"      Y-Score: {result['y_score']}")
                    print(f"      Similarity: {result['similarity']}")
                    print(f"      Collections: {', '.join(result['collections'])}")
            else:
                print("   No relevant papers found")
        
        # Interactive QA
        print("\\n" + "="*80)
        print("INTERACTIVE QA MODE (type 'quit' to exit)")
        print("="*80)
        
        while True:
            try:
                question = input("\\nYour question: ").strip()
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not question:
                    continue
                
                results = self.answer_question(question, top_k=3)
                
                if results:
                    print("\\nRelevant Papers:")
                    for i, result in enumerate(results, 1):
                        print(f"   {i}. {result['title']} (Score: {result['similarity']})")
                        print(f"      Collections: {', '.join(result['collections'])}")
                else:
                    print("   No relevant papers found")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("\\nDemo completed!")

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