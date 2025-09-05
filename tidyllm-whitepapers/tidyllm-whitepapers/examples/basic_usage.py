#!/usr/bin/env python3
"""
TidyLLM-Papers Basic Usage Examples

Demonstrates core functionality and integration with TidyLLM ecosystem.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import tidyllm
    LLMDATA_AVAILABLE = True
    print("🔗 LLMData available - full integration enabled")
except ImportError:
    LLMDATA_AVAILABLE = False
    print("⚠️ LLMData not available - standalone mode")

from tidyllm_papers import papers, discover, analyze, cite

def example_basic_discovery():
    """Basic paper discovery from ArXiv"""
    print("\n" + "="*60)
    print("📄 BASIC PAPER DISCOVERY")
    print("="*60)
    
    # Simple discovery
    research = (papers("machine learning attention mechanisms")
               | discover.arxiv(limit=5))
    
    print(f"Found {len(research.papers)} papers")
    
    # Display results
    for i, paper in enumerate(research.papers, 1):
        print(f"\n{i}. {paper.title}")
        print(f"   Authors: {', '.join(paper.authors[:3])}")
        print(f"   Published: {paper.published_date}")
        print(f"   Categories: {', '.join(paper.categories)}")
        
    return research

def example_content_analysis():
    """Download and analyze paper content"""
    print("\n" + "="*60)
    print("📚 CONTENT ANALYSIS")
    print("="*60)
    
    # Discover, download, and analyze
    research = (papers("neural attention")
               | discover.arxiv(limit=3)
               | analyze.download("./example_papers", max_papers=2)
               | analyze.content())
    
    print(f"Downloaded {research.stats['downloaded']} papers")
    print(f"Analyzed {research.stats['analyzed']} papers")
    
    # Show content analysis
    if 'content_extraction' in research.analysis_results:
        extraction_results = research.analysis_results['content_extraction']
        print(f"Content extraction: {extraction_results['processed']} processed")
    
    # Show sample content
    for paper in research.papers:
        if paper.content:
            print(f"\n📄 {paper.title[:50]}...")
            print(f"Content length: {len(paper.content)} characters")
            print(f"Sample content: {paper.content[:200]}...")
            break
    
    return research

def example_citation_analysis():
    """Analyze citations and references"""
    print("\n" + "="*60)
    print("🔗 CITATION ANALYSIS") 
    print("="*60)
    
    # Full citation workflow
    research = (papers("transformer architecture")
               | discover.arxiv(limit=5)
               | analyze.abstracts()  # Analyze abstracts instead of full content
               | cite.extract_references())
    
    if 'abstracts' in research.analysis_results:
        abstract_results = research.analysis_results['abstracts']
        print(f"Abstracts analyzed: {abstract_results['papers_with_abstracts']}")
        print(f"Common keywords: {list(abstract_results['common_keywords'].keys())[:10]}")
    
    # Format bibliography
    research = research | cite.format_bibtex("example_bibliography.bib")
    
    if 'bibtex' in research.analysis_results:
        print(f"Generated {research.analysis_results['bibtex']['entries_count']} BibTeX entries")
        print("Sample BibTeX entry:")
        sample_entry = research.analysis_results['bibtex']['content'].split('\n\n')[0]
        print(sample_entry)
    
    return research

def example_llm_integration():
    """Integration with LLM for analysis"""
    print("\n" + "="*60)
    print("🤖 LLM INTEGRATION")
    print("="*60)
    
    if not LLMDATA_AVAILABLE:
        print("⚠️ LLMData not available - skipping LLM integration example")
        return None
    
    try:
        # Discover papers and convert to attachments
        research = (papers("attention mechanisms")
                   | discover.sample(2))  # Use sample for quick demo
        
        print(f"Discovered {len(research.papers)} papers")
        
        # Convert to LLMData format (would normally route to LLM)
        from tidyllm_papers.attachments import as_attachments
        
        attachments = as_attachments(research, include_abstracts=True, include_pdfs=False)
        
        if attachments:
            print("✅ Successfully converted to LLMData attachments")
            print(f"Text content length: {len(attachments.text_content) if hasattr(attachments, 'text_content') else 0}")
        else:
            print("❌ Failed to create attachments")
        
        return research
        
    except Exception as e:
        print(f"❌ LLM integration failed: {e}")
        return None

def example_metadata_analysis():
    """Analyze paper metadata and trends"""
    print("\n" + "="*60)
    print("📊 METADATA ANALYSIS")
    print("="*60)
    
    # Discover more papers for better statistics
    research = (papers("deep learning")
               | discover.arxiv(limit=15)
               | analyze.metadata())
    
    if 'metadata' in research.analysis_results:
        metadata = research.analysis_results['metadata']
        
        print("Publication Years:")
        for year, count in sorted(metadata['publication_years'].items()):
            print(f"  {year}: {count} papers")
        
        print("\nTop Categories:")
        for category, count in list(metadata['categories'].items())[:5]:
            print(f"  {category}: {count} papers")
        
        print("\nCollaboration Patterns:")
        collab = metadata['collaboration_patterns']
        print(f"  Single author papers: {collab['single_author_papers']}")
        print(f"  Multi-author papers: {collab['multi_author_papers']}")
        print(f"  Average authors per paper: {collab['average_authors_per_paper']:.1f}")
        
        print("\nMost Productive Authors:")
        for author, count in list(metadata['most_productive_authors'].items())[:5]:
            print(f"  {author}: {count} papers")
    
    return research

def example_advanced_workflow():
    """Complete advanced research workflow"""
    print("\n" + "="*60)
    print("🚀 ADVANCED WORKFLOW")
    print("="*60)
    
    # Multi-step research pipeline
    research = (papers("graph neural networks")
               | discover.by_category(["cs.LG", "cs.AI"], limit=8)
               | analyze.abstracts()
               | analyze.metadata()
               | cite.extract_references())
    
    print(f"Discovered {len(research.papers)} papers across categories")
    
    # Show comprehensive results
    print(f"Collection statistics:")
    for key, value in research.stats.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Save results
    research.save_to_file("advanced_workflow_results.json")
    print("💾 Results saved to advanced_workflow_results.json")
    
    return research

def main():
    """Run all examples"""
    print("🧪 TidyLLM-Papers Examples")
    print("="*60)
    
    examples = [
        ("Basic Discovery", example_basic_discovery),
        ("Content Analysis", example_content_analysis), 
        ("Citation Analysis", example_citation_analysis),
        ("Metadata Analysis", example_metadata_analysis),
        ("LLM Integration", example_llm_integration),
        ("Advanced Workflow", example_advanced_workflow)
    ]
    
    results = {}
    
    for name, example_func in examples:
        try:
            print(f"\n🔄 Running: {name}")
            result = example_func()
            results[name] = result
            print(f"✅ {name} completed successfully")
            
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            results[name] = None
    
    # Summary
    print("\n" + "="*60)
    print("📈 EXAMPLES SUMMARY")
    print("="*60)
    
    successful = sum(1 for result in results.values() if result is not None)
    print(f"Successful examples: {successful}/{len(examples)}")
    
    for name, result in results.items():
        status = "✅" if result is not None else "❌"
        print(f"{status} {name}")
    
    print("\n🎯 Next Steps:")
    print("- Explore tidyllm-papers documentation")
    print("- Integrate with your LLM workflows") 
    print("- Try custom research queries")
    print("- Set up enterprise paper monitoring")

if __name__ == "__main__":
    main()