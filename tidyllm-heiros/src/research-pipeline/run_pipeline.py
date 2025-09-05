"""
Run the Research Paper Processing Pipeline
=========================================

Execute the complete pipeline to download, extract, embed, and store research papers
about mathematical decomposition and residual risk
"""

import os
import sys
import logging
from datetime import datetime
from pdf_extractor_embeddings import ResearchPipelineManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Run the complete research pipeline"""
    
    print("=" * 80)
    print("TidyLLM-HeirOS Research Paper Processing Pipeline")
    print("Mathematical Decomposition & Residual Risk Papers")
    print("=" * 80)
    
    # Configuration from environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Please set: export OPENAI_API_KEY='your-key-here'")
        return False
    
    # PostgreSQL connection
    postgres_connection = os.getenv("POSTGRES_CONNECTION", 
        "postgresql://postgres:password@localhost:5432/research_papers_db"
    )
    
    print(f"OpenAI API Key: {'*' * (len(openai_api_key) - 8)}{openai_api_key[-8:]}")
    print(f"PostgreSQL: {postgres_connection.split('@')[1] if '@' in postgres_connection else 'localhost'}")
    print()
    
    try:
        # Initialize pipeline
        print("Initializing pipeline...")
        pipeline = ResearchPipelineManager(openai_api_key, postgres_connection)
        
        # Extended list of target papers for mathematical decomposition research
        extended_papers = [
            {
                "title": "Deep learning of dynamics and signal-noise decomposition",
                "url": "https://arxiv.org/pdf/1808.02578.pdf",
                "arxiv_id": "1808.02578",
                "authors": ["Steven L. Brunton", "Joshua L. Proctor", "J. Nathan Kutz"]
            },
            {
                "title": "Signal and Noise: A Framework for Reducing Uncertainty",
                "url": "https://arxiv.org/pdf/2508.13144.pdf", 
                "arxiv_id": "2508.13144",
                "authors": ["Authors from ArXiv"]
            },
            {
                "title": "Efficient Orthogonal Decomposition with Automatic Basis Extraction", 
                "url": "https://arxiv.org/pdf/2404.17290.pdf",
                "arxiv_id": "2404.17290",
                "authors": ["Authors from ArXiv"]
            },
            {
                "title": "Orthogonal Mode Decomposition for Finite Discrete Signals",
                "url": "https://arxiv.org/pdf/2409.07242.pdf",
                "arxiv_id": "2409.07242",
                "authors": ["Authors from ArXiv"]
            },
            {
                "title": "Signal and Noise Statistics Oblivious Orthogonal Matching Pursuit",
                "url": "https://arxiv.org/pdf/1806.00650.pdf",
                "arxiv_id": "1806.00650", 
                "authors": ["Authors from ArXiv"]
            }
        ]
        
        # Update pipeline with extended paper list
        pipeline.target_papers = extended_papers
        
        print(f"Target papers: {len(pipeline.target_papers)}")
        print()
        
        # Process all papers
        print("Starting paper processing...")
        pipeline.process_all_papers()
        
        # Test search functionality
        print("\n" + "=" * 50)
        print("TESTING SEARCH FUNCTIONALITY")
        print("=" * 50)
        
        test_queries = [
            "mathematical decomposition Y = R + S + N",
            "residual risk orthogonal projection", 
            "signal noise separation empirical mode",
            "variance decomposition bias noise",
            "systematic error superfluous noise",
            "context collapse prevention electrical"
        ]
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            print("-" * 40)
            
            results = pipeline.search_papers(query, limit=3)
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"{i}. {result['title']}")
                    print(f"   Authors: {', '.join(result.get('authors', ['Unknown']))}")
                    print(f"   Similarity: {result['similarity']:.3f}")
                    print(f"   Type: {result['chunk_type']}")
                    print(f"   Text: {result['chunk_text'][:150]}...")
                    print()
            else:
                print("   No results found")
        
        # Summary statistics
        print("\n" + "=" * 50)
        print("PIPELINE SUMMARY")
        print("=" * 50)
        
        try:
            # Get database statistics
            import psycopg2
            with psycopg2.connect(postgres_connection) as conn:
                with conn.cursor() as cur:
                    # Count papers
                    cur.execute("SELECT COUNT(*) FROM research_papers;")
                    paper_count = cur.fetchone()[0]
                    
                    # Count embeddings
                    cur.execute("SELECT COUNT(*) FROM paper_embeddings;")
                    embedding_count = cur.fetchone()[0]
                    
                    # Get chunk type distribution
                    cur.execute("""
                        SELECT chunk_type, COUNT(*) 
                        FROM paper_embeddings 
                        GROUP BY chunk_type 
                        ORDER BY COUNT(*) DESC;
                    """)
                    chunk_distribution = cur.fetchall()
                    
                    print(f"Papers processed: {paper_count}")
                    print(f"Embedding chunks: {embedding_count}")
                    print(f"Average chunks per paper: {embedding_count/paper_count:.1f}")
                    print("\nChunk type distribution:")
                    for chunk_type, count in chunk_distribution:
                        print(f"  {chunk_type}: {count}")
                        
        except Exception as e:
            print(f"Could not retrieve statistics: {e}")
        
        print("\n✅ Pipeline completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        logging.exception("Pipeline execution failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)