#!/usr/bin/env python3
"""Test remove from collection functionality"""

from dotenv import load_dotenv
load_dotenv()

from paper_repository import get_paper_repository

def test_remove_from_collections():
    """Test removing papers from collections"""
    print("Testing remove from collection functionality...")
    print("=" * 50)
    
    repo = get_paper_repository()
    
    # Show current state
    papers = list(repo.index.get("papers", {}).keys())
    collections = repo.get_collections()
    
    print("Current state:")
    for paper_id in papers:
        paper_info = repo.index["papers"][paper_id]
        paper_collections = paper_info.get("collections", [])
        print(f"  Paper {paper_id}: {paper_collections}")
    
    print(f"\nCollections:")
    for collection in collections:
        collection_papers = repo.index.get("collections", {}).get(collection["name"], {}).get("papers", [])
        print(f"  {collection['name']}: {collection_papers}")
    
    # Test removing a paper from a collection
    if len(papers) > 0 and len(collections) > 0:
        paper_id = "2303.13534"  # The AI Art paper
        collection_name = "Test Collection"
        
        print(f"\nRemoving paper '{paper_id}' from collection '{collection_name}'...")
        result = repo.remove_from_collection(paper_id, collection_name)
        print(f"Remove result: {result}")
        
        # Check updated state
        print(f"\nUpdated state:")
        paper_info = repo.index["papers"][paper_id]
        paper_collections = paper_info.get("collections", [])
        print(f"  Paper {paper_id}: {paper_collections}")
        
        collection_papers = repo.index.get("collections", {}).get(collection_name, {}).get("papers", [])
        print(f"  Collection '{collection_name}': {collection_papers}")
        
        # Test removing again (should fail)
        print(f"\nTrying to remove again (should fail)...")
        result2 = repo.remove_from_collection(paper_id, collection_name)
        print(f"Second remove result: {result2}")
    
    print("\nTest complete!")

if __name__ == '__main__':
    test_remove_from_collections()