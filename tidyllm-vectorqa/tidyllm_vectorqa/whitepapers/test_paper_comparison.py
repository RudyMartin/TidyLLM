#!/usr/bin/env python3
"""Test paper comparison with TidyLLM transformers"""

from dotenv import load_dotenv
load_dotenv()

from paper_repository import get_paper_repository

def test_paper_comparison():
    """Test the embedding comparison functionality"""
    print("Testing paper comparison with TidyLLM transformers...")
    print("=" * 60)
    
    # Get repository
    repo = get_paper_repository()
    
    # Check available papers
    papers = list(repo.index.get("papers", {}).keys())
    print(f"Available papers: {len(papers)}")
    
    for paper_id in papers:
        paper_info = repo.index["papers"][paper_id]
        print(f"- {paper_id}: {paper_info.get('title', 'No title')[:50]}...")
    
    if len(papers) < 2:
        print("ERROR: Need at least 2 papers for comparison")
        return
    
    print(f"\nGenerating comparison report for {len(papers)} papers...")
    
    # Generate comparison report
    report = repo.generate_embedding_comparison_report(papers)
    
    if report.get("success"):
        print("SUCCESS: Analysis successful!")
        
        # Show results
        stats = report["collection_stats"]
        print(f"\nCollection Statistics:")
        print(f"- Total papers: {report['total_papers']}")
        print(f"- Average similarity: {stats['average_similarity_percent']}%")
        print(f"- Total comparisons: {stats['total_comparisons']}")
        print(f"- Model: {report['embedding_model']}")
        
        # Show analysis summary
        print(f"\nAnalysis Summary:")
        for point in report["analysis_summary"]:
            print(f"  {point}")
        
        # Show most similar pair
        if stats["most_similar_pair"]:
            pair = stats["most_similar_pair"]
            print(f"\nMost Similar Pair ({pair['similarity_percent']}%):")
            print(f"  Paper 1: {pair['paper1']['title']}")
            print(f"  Paper 2: {pair['paper2']['title']}")
        
        # Show top 3 similarities
        print(f"\nTop Similarities:")
        for i, sim in enumerate(report["pairwise_similarities"][:3]):
            print(f"  {i+1}. {sim['similarity_percent']}% - {sim['paper1']['title'][:30]}... vs {sim['paper2']['title'][:30]}...")
        
        # Show model details
        model_info = report["model_details"]
        print(f"\nModel Details:")
        print(f"  - Vocabulary size: {model_info['vocab_size']:,}")
        print(f"  - Embedding dimension: {model_info['embedding_dimension']:,}")
        print(f"  - Attention heads: {model_info['attention_heads']}")
        
    else:
        print(f"ERROR: Analysis failed: {report.get('message', 'Unknown error')}")
        if "TidyLLM" in str(report.get('message', '')):
            print("TIP: Make sure tidyllm-sentence is properly installed")

if __name__ == '__main__':
    test_paper_comparison()