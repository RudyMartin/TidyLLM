"""
Search Script for Y=R+S+N Analysis
=================================

Standalone script to perform searches and track results in PostgreSQL.
This can be called from Streamlit or run independently.

Usage:
    python search_script.py "neural networks" "ArXiv" 10
"""

import sys
import os
import argparse
import json
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import required modules
try:
    from backend_config import get_backend_config
    from search_tracker import YRSNSearchTracker, SearchSession, SearchResult, generate_session_id
    # Import from the correct path
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'yrsn'))
    from research_framework import ResearchFramework
    import requests
    import xml.etree.ElementTree as ET
    from urllib.parse import quote_plus
    print("All imports successful")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def search_arxiv(query, max_results=10):
    """Search ArXiv for papers"""
    print(f"Searching ArXiv for: '{query}' (max {max_results} results)")
    
    try:
        # ArXiv API
        base_url = "http://export.arxiv.org/api/query?"
        search_query = f"search_query=all:{quote_plus(query)}&start=0&max_results={max_results}"
        
        print(f"API URL: {base_url + search_query}")
        response = requests.get(base_url + search_query, timeout=30)
        response.raise_for_status()
        
        # Parse XML response
        root = ET.fromstring(response.content)
        papers = []
        
        # Namespace for ArXiv
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        entries = root.findall('atom:entry', ns)
        print(f"Found {len(entries)} entries in API response")
        
        for i, entry in enumerate(entries):
            try:
                title = entry.find('atom:title', ns).text.strip()
                authors = [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)]
                abstract = entry.find('atom:summary', ns).text.strip()
                published = entry.find('atom:published', ns).text[:10]
                arxiv_id = entry.find('atom:id', ns).text.split('/')[-1]
                
                papers.append({
                    'title': title,
                    'authors': authors,
                    'abstract': abstract,
                    'published': published,
                    'arxiv_id': arxiv_id,
                    'source': 'ArXiv'
                })
                
                print(f"  {i+1}. {title[:60]}...")
                
            except Exception as e:
                print(f"Warning: Error parsing entry {i+1}: {e}")
                continue
        
        print(f"Successfully parsed {len(papers)} papers")
        return papers
        
    except Exception as e:
        print(f"ArXiv search error: {e}")
        return []

def analyze_and_track_search(query, source, papers, max_results=10):
    """Analyze papers with Y=R+S+N framework and track in database"""
    print(f"\n Starting Y=R+S+N analysis for {len(papers)} papers...")
    
    if not papers:
        print(" No papers to analyze")
        return False
    
    try:
        # Initialize framework and tracker
        framework = ResearchFramework()
        backend_config = get_backend_config()
        tracker = YRSNSearchTracker(backend_config)
        
        print(" Framework and tracker initialized")
        
        # Analyze each paper
        search_results = []
        analyses = []
        
        for i, paper in enumerate(papers):
            print(f"\n Analyzing paper {i+1}: {paper['title'][:50]}...")
            
            try:
                # Perform Y=R+S+N analysis (test without search_query first)
                analysis = framework.analyze_paper_content(
                    title=paper['title'],
                    abstract=paper['abstract']
                )
                
                analyses.append(analysis)
                
                # Calculate semantic relevance for tracking (simplified)
                # TODO: Implement proper semantic relevance calculation
                semantic_relevance = 0.5  # Default relevance
                
                # Create search result object
                search_result = SearchResult(
                    paper_id=paper.get('arxiv_id', f"unknown_{i}"),
                    title=paper['title'],
                    authors=paper['authors'],
                    abstract=paper['abstract'],
                    source=paper['source'],
                    y_score=analysis.y_score,
                    r_score=analysis.relevant,
                    s_score=analysis.superfluous,
                    n_score=analysis.noise,
                    context_risk=analysis.superfluous + (1.5 * analysis.noise),
                    semantic_relevance=semantic_relevance
                )
                
                search_results.append(search_result)
                
                print(f"   Y Score: {analysis.y_score:.3f} | R: {analysis.relevant:.1%} | S: {analysis.superfluous:.1%} | N: {analysis.noise:.1%}")
                print(f"   Semantic Relevance: {semantic_relevance:.3f}")
                
            except Exception as e:
                print(f"    Analysis error: {e}")
                continue
        
        if not search_results:
            print(" No successful analyses to track")
            return False
        
        print(f"\n Tracking search session with {len(search_results)} results...")
        
        # Calculate session summary
        avg_y = sum(r.y_score for r in search_results) / len(search_results)
        avg_r = sum(r.r_score for r in search_results) / len(search_results)
        avg_s = sum(r.s_score for r in search_results) / len(search_results)
        avg_n = sum(r.n_score for r in search_results) / len(search_results)
        avg_risk = sum(r.context_risk for r in search_results) / len(search_results)
        
        # Get top paper by Y score
        top_paper = max(search_results, key=lambda x: x.y_score)
        
        # Infer research domain
        research_domain = tracker.infer_research_domain(query, [r.title for r in search_results])
        
        # Create search session
        session = SearchSession(
            session_id=generate_session_id(),
            query=query,
            search_source=source,
            timestamp=datetime.now(),
            total_results=len(search_results),
            avg_y_score=avg_y,
            avg_r_score=avg_r,
            avg_s_score=avg_s,
            avg_n_score=avg_n,
            avg_context_risk=avg_risk,
            top_paper_title=top_paper.title,
            research_domain=research_domain
        )
        
        print(f" Session Summary:")
        print(f"   Query: {query}")
        print(f"   Source: {source}")
        print(f"   Domain: {research_domain}")
        print(f"   Results: {len(search_results)}")
        print(f"   Avg Y Score: {avg_y:.3f}")
        print(f"   Avg Context Risk: {avg_risk:.3f}")
        print(f"   Top Paper: {top_paper.title[:60]}...")
        
        # Log the search session
        session_id = tracker.log_search_session(session, search_results)
        print(f" Search session logged with ID: {session_id}")
        
        return True
        
    except Exception as e:
        print(f" Tracking error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function for command line execution"""
    parser = argparse.ArgumentParser(description='Search and analyze papers with Y=R+S+N framework')
    parser.add_argument('query', help='Search query')
    parser.add_argument('source', choices=['ArXiv', 'PubMed', 'Multiple'], default='ArXiv', help='Search source')
    parser.add_argument('max_results', type=int, default=10, help='Maximum results to return')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Y=R+S+N SEARCH AND ANALYSIS SCRIPT")
    print("=" * 80)
    print(f"Query: {args.query}")
    print(f"Source: {args.source}")
    print(f"Max Results: {args.max_results}")
    print()
    
    # Perform search
    if args.source == 'ArXiv':
        papers = search_arxiv(args.query, args.max_results)
    else:
        print(f" {args.source} not implemented yet")
        return
    
    if not papers:
        print(" No papers found")
        return
    
    # Analyze and track
    success = analyze_and_track_search(args.query, args.source, papers, args.max_results)
    
    if success:
        print(f"\n SUCCESS! Search tracked in database")
        print("Check the YRSN Searches tab in Streamlit to see the results")
    else:
        print(f"\n FAILED to track search")

def search_and_track(query, source="ArXiv", max_results=10):
    """
    Function to be called from Streamlit
    Returns: (success, message, results_count, papers)
    """
    try:
        print(f" search_and_track called: query='{query}', source='{source}', max_results={max_results}")
        
        # Perform search
        if source == 'ArXiv':
            papers = search_arxiv(query, max_results)
        else:
            return False, f"{source} not implemented yet", 0, []
        
        if not papers:
            return False, "No papers found", 0, []
        
        # Analyze and track
        success = analyze_and_track_search(query, source, papers, max_results)
        
        if success:
            return True, f"Successfully tracked {len(papers)} papers", len(papers), papers
        else:
            return False, "Failed to track search", len(papers), papers
            
    except Exception as e:
        error_msg = f"Search script error: {e}"
        print(f" {error_msg}")
        import traceback
        traceback.print_exc()
        return False, error_msg, 0, []

if __name__ == "__main__":
    main()