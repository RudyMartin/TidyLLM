#!/usr/bin/env python3
"""
QA System with Paper Content Extraction
=======================================

Enhanced QA system that extracts actual content from papers and 
generates responses based on the paper content, not just metadata.
"""

import sys
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import PyPDF2
import io

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

class ContentQA:
    """QA system that extracts and uses actual paper content."""
    
    def __init__(self):
        self.papers = []
        self.paper_contents = []
        self.paper_embeddings = None
        self.embedding_model = None
        
    def extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF file."""
        try:
            if not os.path.exists(pdf_path):
                return f"PDF file not found: {pdf_path}"
                
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                # Extract text from all pages (limit to first 10 pages for demo)
                max_pages = min(len(pdf_reader.pages), 10)
                for page_num in range(max_pages):
                    try:
                        page = pdf_reader.pages[page_num]
                        text += page.extract_text() + "\n"
                    except Exception as e:
                        print(f"Error extracting page {page_num}: {e}")
                        continue
                
                # Clean up the text
                text = text.replace('\n', ' ').replace('\t', ' ')
                # Remove excessive whitespace
                text = ' '.join(text.split())
                
                return text[:5000] if len(text) > 5000 else text  # Limit for demo
                
        except Exception as e:
            return f"Error extracting PDF text: {e}"
    
    def load_papers_with_content(self, kb_path="paper_repository/repository_index.json"):
        """Load papers and extract their content."""
        try:
            with open(kb_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.papers = []
            self.paper_contents = []
            
            print("Loading papers with content extraction...")
            
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
                    "collections": info.get("collections", []),
                    "file_path": info.get("file_path", "")
                }
                
                # Extract content from PDF
                pdf_path = info.get("file_path", "")
                if pdf_path and os.path.exists(pdf_path):
                    print(f"   Extracting content from: {paper['title'][:50]}...")
                    content = self.extract_pdf_text(pdf_path)
                else:
                    content = f"Title: {paper['title']}. Authors: {', '.join(clean_authors)}"
                
                # Combine title, authors, and content for embeddings
                full_content = f"Title: {paper['title']}\nAuthors: {', '.join(clean_authors)}\nContent: {content}"
                
                self.papers.append(paper)
                self.paper_contents.append(full_content)
            
            print(f"SUCCESS: Loaded {len(self.papers)} papers with content")
            return True
            
        except Exception as e:
            print(f"ERROR: Error loading papers: {e}")
            return False
    
    def build_content_embeddings(self):
        """Build embeddings using paper content."""
        if not EMBEDDINGS_AVAILABLE:
            print("⚠️ Falling back to text matching")
            return False
            
        if not self.paper_contents:
            print("❌ No paper content loaded")
            return False
        
        try:
            print("🔄 Building embeddings from paper content...")
            self.paper_embeddings, self.embedding_model = fit_transform(self.paper_contents)
            
            print(f"✅ Built content-based embeddings:")
            print(f"   📊 Papers: {len(self.paper_embeddings)}")
            print(f"   📐 Dimensions: {len(self.paper_embeddings[0])}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error building embeddings: {e}")
            return False
    
    def search_content(self, question: str, top_k=3) -> List[Tuple[Dict[str, Any], float, str]]:
        """Search using content embeddings and return relevant excerpts."""
        if not EMBEDDINGS_AVAILABLE or self.paper_embeddings is None:
            return self._fallback_content_search(question, top_k)
        
        try:
            # Get question embedding
            question_embeddings, _ = fit_transform([question])
            question_embedding = question_embeddings[0]
            
            # Calculate similarities
            similarities = []
            for i, paper_embedding in enumerate(self.paper_embeddings):
                similarity = cosine_similarity(question_embedding, paper_embedding)
                similarities.append((i, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Return top results with content excerpts
            results = []
            for idx, similarity in similarities[:top_k]:
                if similarity > 0.01:
                    paper = self.papers[idx]
                    content = self.paper_contents[idx]
                    
                    # Extract relevant excerpt (first 500 chars of content section)
                    content_start = content.find("Content: ") + 9
                    excerpt = content[content_start:content_start + 500] if content_start > 8 else content[:500]
                    
                    results.append((paper, similarity, excerpt))
            
            return results
            
        except Exception as e:
            print(f"❌ Content search failed: {e}")
            return self._fallback_content_search(question, top_k)
    
    def _fallback_content_search(self, question: str, top_k=3) -> List[Tuple[Dict[str, Any], float, str]]:
        """Fallback content search using text matching."""
        question_words = set(question.lower().split())
        
        scores = []
        for i, paper in enumerate(self.papers):
            content = self.paper_contents[i].lower()
            content_words = set(content.split())
            overlap = len(question_words & content_words)
            score = overlap / len(question_words) if question_words else 0
            
            # Extract excerpt for fallback too
            excerpt = self.paper_contents[i][:500] + "..." if len(self.paper_contents[i]) > 500 else self.paper_contents[i]
            
            scores.append((paper, score, excerpt))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(paper, score, excerpt) for paper, score, excerpt in scores[:top_k] if score > 0]
    
    def generate_answer(self, question: str, relevant_papers: List[Tuple[Dict[str, Any], float, str]]) -> str:
        """Generate an answer based on retrieved paper content."""
        if not relevant_papers:
            return "I couldn't find any relevant papers to answer your question."
        
        # Simple template-based response (could be enhanced with LLM)
        answer_parts = [
            f"Based on the research papers in our knowledge base, here's what I found about '{question}':\n"
        ]
        
        for i, (paper, score, excerpt) in enumerate(relevant_papers, 1):
            answer_parts.append(f"\n{i}. From '{paper['title']}' (Y-Score: {paper['y_score']}, Relevance: {score:.3f}):")
            answer_parts.append(f"   Authors: {', '.join(paper['authors'])}")
            answer_parts.append(f"   Excerpt: \"{excerpt}...\"\n")
        
        answer_parts.append(f"\nThis answer was generated from {len(relevant_papers)} relevant research papers.")
        
        return "".join(answer_parts)
    
    def demo_content_qa(self):
        """Run content-based QA demonstration."""
        print("\n" + "="*70)
        print("📚 CONTENT-BASED VALIDATION QA DEMO")
        print("="*70)
        
        # Load papers with content
        if not self.load_papers_with_content():
            return
        
        # Build embeddings
        embeddings_ready = self.build_content_embeddings()
        method = "Content Embeddings" if embeddings_ready else "Content Text Matching"
        
        print(f"\n🎯 Search Method: {method}")
        
        # Test questions
        questions = [
            "What are heat engines and how do they work?",
            "How does AI art generation relate to creativity?",
            "What mathematical frameworks are used for signal processing?",
            "What are the main challenges in noise separation?"
        ]
        
        print("\n" + "="*70)
        print("❓ CONTENT-BASED Q&A EXAMPLES")
        print("="*70)
        
        for i, question in enumerate(questions, 1):
            print(f"\n🔍 Question {i}: {question}")
            print("-" * 50)
            
            # Search for relevant papers
            relevant_papers = self.search_content(question, top_k=2)
            
            if relevant_papers:
                # Generate answer
                answer = self.generate_answer(question, relevant_papers)
                print(answer)
            else:
                print("❌ No relevant papers found for this question.")
        
        print("\n" + "="*70)
        print("✨ CONTENT-BASED QA BENEFITS")
        print("="*70)
        print("✅ Answers based on actual paper content, not just metadata")
        print("✅ Direct quotes and excerpts from research papers")
        print("✅ Relevance scoring based on content similarity")
        print("✅ Y-scores for paper quality assessment")
        print("✅ Semantic understanding of research content")
        
        print("\n🎉 Content-based QA demo completed!")

def main():
    """Main demo function."""
    print("🚀 Starting Content-Based QA Demo...")
    
    # Change to correct directory
    demo_dir = Path(__file__).parent
    os.chdir(demo_dir)
    
    # Run demo
    qa = ContentQA()
    qa.demo_content_qa()

if __name__ == "__main__":
    main()