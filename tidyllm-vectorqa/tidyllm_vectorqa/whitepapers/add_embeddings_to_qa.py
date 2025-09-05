#!/usr/bin/env python3
"""
How to Add Embeddings to QA System
=================================

Step-by-step guide and working implementation to add embeddings
to the validation QA system.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from sentence.tfidf.embeddings import fit_transform
    from sentence.utils.similarity import cosine_similarity
    EMBEDDINGS_AVAILABLE = True
    print("SUCCESS: TidyLLM embeddings available")
except ImportError as e:
    print(f"WARNING: Embeddings not available: {e}")
    EMBEDDINGS_AVAILABLE = False

class EmbeddingsQASystem:
    """QA system with embeddings integration."""
    
    def __init__(self):
        self.papers = []
        self.paper_texts = []
        self.paper_embeddings = None
        self.embedding_model = None
        
    def load_knowledge_base(self):
        """Step 1: Load papers and prepare text."""
        kb_path = "paper_repository/repository_index.json"
        
        try:
            with open(kb_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for paper_id, info in data.get("papers", {}).items():
                # Clean authors to avoid encoding issues
                clean_authors = []
                for author in info.get("authors", []):
                    try:
                        # Remove problematic characters
                        clean_author = author.encode('ascii', 'ignore').decode('ascii')
                        clean_authors.append(clean_author)
                    except:
                        clean_authors.append("Unknown")
                
                paper = {
                    "id": paper_id,
                    "title": info.get("title", ""),
                    "authors": clean_authors,
                    "y_score": info.get("y_score", 0),
                    "collections": info.get("collections", [])
                }
                
                # Create comprehensive text for embeddings
                title = info.get("title", "")
                authors_text = " ".join(clean_authors)
                collections_text = " ".join(info.get("collections", []))
                
                # Rich text combining all metadata
                paper_text = f"{title} {authors_text} {collections_text}"
                
                self.papers.append(paper)
                self.paper_texts.append(paper_text)
            
            print(f"Loaded {len(self.papers)} papers for embeddings")
            return True
            
        except Exception as e:
            print(f"Error loading papers: {e}")
            return False
    
    def create_embeddings(self):
        """Step 2: Build embeddings for all papers."""
        if not EMBEDDINGS_AVAILABLE:
            print("FALLBACK: Using text matching instead of embeddings")
            return False
            
        if not self.paper_texts:
            print("ERROR: No papers loaded")
            return False
        
        try:
            print("Building embeddings...")
            # Use TidyLLM fit_transform - returns (embeddings, model)
            self.paper_embeddings, self.embedding_model = fit_transform(self.paper_texts)
            
            print(f"SUCCESS: Built {len(self.paper_embeddings)} embeddings")
            print(f"Embedding dimension: {len(self.paper_embeddings[0])}")
            
            return True
            
        except Exception as e:
            print(f"ERROR building embeddings: {e}")
            return False
    
    def search_with_embeddings(self, question: str, top_k=3):
        """Step 3: Use embeddings for semantic search."""
        if not self.paper_embeddings:
            return self._text_search_fallback(question, top_k)
        
        try:
            # Create embedding for question
            question_embeddings, _ = fit_transform([question])
            question_embedding = question_embeddings[0]
            
            # Calculate similarity with all papers
            similarities = []
            for i, paper_embedding in enumerate(self.paper_embeddings):
                similarity = cosine_similarity(question_embedding, paper_embedding)
                similarities.append((i, similarity))
            
            # Sort by similarity and get top results
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            results = []
            for idx, similarity in similarities[:top_k]:
                if similarity > 0.01:  # Minimum threshold
                    results.append((self.papers[idx], similarity))
            
            return results
            
        except Exception as e:
            print(f"Embeddings search failed: {e}")
            return self._text_search_fallback(question, top_k)
    
    def _text_search_fallback(self, question: str, top_k=3):
        """Fallback text matching when embeddings unavailable."""
        question_words = set(question.lower().split())
        
        scores = []
        for i, paper in enumerate(self.papers):
            text_words = set(self.paper_texts[i].lower().split())
            overlap = len(question_words & text_words)
            score = overlap / len(question_words) if question_words else 0
            scores.append((paper, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(paper, score) for paper, score in scores[:top_k] if score > 0]
    
    def demo_embeddings_integration(self):
        """Step 4: Demonstrate embeddings integration."""
        print("\\n" + "="*60)
        print("EMBEDDINGS INTEGRATION DEMO")
        print("="*60)
        
        # Load knowledge base
        if not self.load_knowledge_base():
            print("Failed to load knowledge base")
            return
        
        # Build embeddings
        embeddings_success = self.create_embeddings()
        method = "Semantic Embeddings" if embeddings_success else "Text Matching"
        print(f"Search Method: {method}")
        
        # Test questions
        test_questions = [
            "signal decomposition mathematical framework",
            "transformer attention mechanism",
            "noise separation reservoir computing",
            "adaptive filtering dynamic",
            "sparse representation denoising"
        ]
        
        print("\\nTesting Questions:")
        print("-" * 60)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\\nQ{i}: {question}")
            
            results = self.search_with_embeddings(question, top_k=2)
            
            if results:
                print("Results:")
                for j, (paper, score) in enumerate(results, 1):
                    title_short = paper['title'][:40] + "..." if len(paper['title']) > 40 else paper['title']
                    print(f"  {j}. {title_short}")
                    print(f"     Score: {score:.4f}, Y-Score: {paper['y_score']}")
                    print(f"     Collections: {', '.join(paper['collections'])}")
            else:
                print("  No matches found")
        
        # Show embedding benefits
        if embeddings_success:
            print("\\n" + "="*60)
            print("EMBEDDINGS BENEFITS")
            print("="*60)
            print("- Semantic similarity (beyond keyword matching)")
            print("- Better relevance ranking")
            print("- Handles synonyms and related concepts")
            print("- Scalable to large document collections")
            print("- Quantified similarity scores")
        
        print("\\nDemo completed!")

def show_integration_steps():
    """Show how to integrate embeddings into existing QA system."""
    print("HOW TO ADD EMBEDDINGS TO QA SYSTEM")
    print("="*50)
    print("""
STEP 1: Import embeddings module
    from sentence.tfidf.embeddings import fit_transform
    from sentence.utils.similarity import cosine_similarity

STEP 2: Prepare text data
    paper_texts = [f"{title} {authors} {collections}" for each paper]

STEP 3: Build embeddings
    embeddings, model = fit_transform(paper_texts)

STEP 4: Search with embeddings
    question_embeddings, _ = fit_transform([question])
    similarities = [cosine_similarity(q_emb, p_emb) for p_emb in embeddings]

STEP 5: Rank and return results
    sorted_results = sorted by similarity score
    """)

def main():
    """Main demonstration."""
    print("Starting Embeddings Integration Demo...")
    
    # Change to correct directory
    demo_dir = Path(__file__).parent
    import os
    os.chdir(demo_dir)
    
    # Show integration steps
    show_integration_steps()
    
    # Run live demo
    qa_system = EmbeddingsQASystem()
    qa_system.demo_embeddings_integration()

if __name__ == "__main__":
    main()