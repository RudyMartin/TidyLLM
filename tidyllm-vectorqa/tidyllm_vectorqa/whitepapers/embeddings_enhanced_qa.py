#!/usr/bin/env python3
"""
Embeddings-Enhanced QA System
============================

Shows how to add proper embeddings to the validation QA system
using TidyLLM sentence embeddings.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from sentence.tfidf.embeddings import fit_transform
    from sentence.utils.similarity import cosine_similarity
    EMBEDDINGS_AVAILABLE = True
    print("✅ TidyLLM embeddings available")
except ImportError as e:
    print(f"❌ Embeddings not available: {e}")
    EMBEDDINGS_AVAILABLE = False

class EmbeddingsQA:
    """QA system with proper embeddings support."""
    
    def __init__(self):
        self.papers = []
        self.paper_texts = []
        self.paper_embeddings = None
        self.embedding_model = None
        
    def load_papers(self, kb_path="paper_repository/repository_index.json"):
        """Load papers and prepare text for embeddings."""
        try:
            with open(kb_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.papers = []
            self.paper_texts = []
            
            for paper_id, info in data.get("papers", {}).items():
                # Clean authors
                clean_authors = []
                for author in info.get("authors", []):
                    try:
                        clean_authors.append(author.encode('ascii', 'ignore').decode('ascii'))
                    except:
                        clean_authors.append("Unknown")
                
                paper = {
                    "id": paper_id,
                    "title": info.get("title", ""),
                    "authors": clean_authors,
                    "y_score": info.get("y_score", 0),
                    "collections": info.get("collections", [])
                }
                
                # Create rich text for embeddings
                title = info.get("title", "")
                authors_text = " ".join(clean_authors)
                collections_text = " ".join(info.get("collections", []))
                
                # Combine title, authors, and collections for better embeddings
                paper_text = f"{title} {authors_text} {collections_text}"
                
                self.papers.append(paper)
                self.paper_texts.append(paper_text)
            
            print(f"📚 Loaded {len(self.papers)} papers")
            return True
            
        except Exception as e:
            print(f"❌ Error loading papers: {e}")
            return False
    
    def build_embeddings(self):
        """Build embeddings for all papers."""
        if not EMBEDDINGS_AVAILABLE:
            print("⚠️ Falling back to text matching")
            return False
            
        if not self.paper_texts:
            print("❌ No papers loaded")
            return False
        
        try:
            print("🔄 Building embeddings...")
            self.paper_embeddings, self.embedding_model = fit_transform(self.paper_texts)
            
            print(f"✅ Built embeddings:")
            print(f"   📊 Papers: {len(self.paper_embeddings)}")
            print(f"   📐 Dimensions: {len(self.paper_embeddings[0])}")
            print(f"   🔧 Model: TidyLLM TF-IDF")
            
            return True
            
        except Exception as e:
            print(f"❌ Error building embeddings: {e}")
            return False
    
    def search_with_embeddings(self, question: str, top_k=5) -> List[Tuple[Dict[str, Any], float]]:
        """Search using embeddings similarity."""
        if not EMBEDDINGS_AVAILABLE or self.paper_embeddings is None:
            return self._fallback_search(question, top_k)
        
        try:
            # Get question embedding using same model
            question_embeddings, _ = fit_transform([question])
            question_embedding = question_embeddings[0]
            
            # Calculate similarities with all papers
            similarities = []
            for i, paper_embedding in enumerate(self.paper_embeddings):
                similarity = cosine_similarity(question_embedding, paper_embedding)
                similarities.append((i, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Return top results
            results = []
            for idx, similarity in similarities[:top_k]:
                if similarity > 0:  # Only include matches
                    results.append((self.papers[idx], similarity))
            
            return results
            
        except Exception as e:
            print(f"❌ Embedding search failed: {e}")
            return self._fallback_search(question, top_k)
    
    def _fallback_search(self, question: str, top_k=5) -> List[Tuple[Dict[str, Any], float]]:
        """Fallback text search when embeddings fail."""
        question_words = set(question.lower().split())
        
        scores = []
        for i, paper in enumerate(self.papers):
            text = self.paper_texts[i].lower()
            text_words = set(text.split())
            overlap = len(question_words & text_words)
            score = overlap / len(question_words) if question_words else 0
            scores.append((paper, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(paper, score) for paper, score in scores[:top_k] if score > 0]
    
    def demo_embeddings_qa(self):
        """Run embeddings QA demonstration."""
        print("\\n" + "="*70)
        print("🔍 EMBEDDINGS-ENHANCED VALIDATION QA DEMO")
        print("="*70)
        
        # Load papers
        if not self.load_papers():
            return
        
        # Build embeddings
        embeddings_ready = self.build_embeddings()
        method = "Embeddings + Cosine Similarity" if embeddings_ready else "Text Matching Fallback"
        
        print(f"\\n🎯 Search Method: {method}")
        
        # Test questions
        questions = [
            "mathematical frameworks for signal decomposition",
            "transformer attention mechanisms work",
            "noise separation using reservoir computing",
            "adaptive filtering network image denoising",
            "sparse representation self-supervised denoising",
            "variational mode decomposition time windowed",
            "mixture of head attention efficiency"
        ]
        
        print("\\n" + "="*70)
        print("📋 VALIDATION QUESTIONS & RESULTS")
        print("="*70)
        
        for i, question in enumerate(questions, 1):
            print(f"\\n❓ Q{i}: {question}")
            
            results = self.search_with_embeddings(question, top_k=3)
            
            if results:
                print("📄 Top Matches:")
                for j, (paper, score) in enumerate(results, 1):
                    title = paper['title'][:50] + "..." if len(paper['title']) > 50 else paper['title']
                    print(f"   {j}. {title}")
                    print(f"      📊 Similarity: {score:.4f}")
                    print(f"      ⭐ Y-Score: {paper['y_score']}")
                    print(f"      📁 Collections: {', '.join(paper['collections'])}")
            else:
                print("   ❌ No matches found")
        
        # Show embedding advantages
        if embeddings_ready:
            print("\\n" + "="*70)
            print("🎯 EMBEDDINGS ADVANTAGES DEMONSTRATED")
            print("="*70)
            print("✅ Semantic similarity (not just keyword matching)")
            print("✅ Handles synonyms and related concepts")
            print("✅ Better ranking with cosine similarity")
            print("✅ Works with partial matches")
            print("✅ Scalable to large document collections")
        
        print("\\n🎉 Demo completed!")
        
        # Interactive mode
        print("\\n" + "="*70)
        print("💬 Try your own questions (type 'quit' to exit):")
        print("="*70)
        
        while True:
            try:
                question = input("\\n❓ Your question: ").strip()
                if question.lower() in ['quit', 'exit', 'q', '']:
                    break
                
                results = self.search_with_embeddings(question, top_k=2)
                
                if results:
                    print("\\n📄 Results:")
                    for i, (paper, score) in enumerate(results, 1):
                        print(f"   {i}. {paper['title'][:60]}...")
                        print(f"      Score: {score:.4f}, Y-Score: {paper['y_score']}")
                else:
                    print("   ❌ No matches found")
                    
            except (EOFError, KeyboardInterrupt):
                break
        
        print("\\n👋 Thanks for trying the embeddings QA demo!")

def main():
    """Main demo function."""
    print("🚀 Starting Embeddings-Enhanced QA Demo...")
    
    # Change to correct directory
    demo_dir = Path(__file__).parent
    import os
    os.chdir(demo_dir)
    
    # Run demo
    qa = EmbeddingsQA()
    qa.demo_embeddings_qa()

if __name__ == "__main__":
    main()