#!/usr/bin/env python3
"""Test collection functionality"""

from dotenv import load_dotenv
load_dotenv()

from paper_repository import get_paper_repository

def test_collections():
    """Test creating collections and adding papers"""
    print("Testing collection functionality...")
    print("=" * 50)
    
    repo = get_paper_repository()
    
    # Get available papers
    papers = list(repo.index.get("papers", {}).keys())
    print(f"Available papers: {len(papers)}")
    for paper_id in papers:
        title = repo.index["papers"][paper_id].get("title", "Unknown")
        print(f"  - {paper_id}: {title[:50]}...")
    
    # Test creating a collection
    print(f"\nCreating test collection...")
    result = repo.create_collection("Test Collection", "Testing paper associations")
    print(f"Create result: {result}")
    
    # Get collections
    collections = repo.get_collections()
    print(f"\nAvailable collections: {len(collections)}")
    for collection in collections:
        print(f"  - {collection['name']}: {collection['description']}")
    
    if len(papers) >= 1 and len(collections) >= 1:
        paper_id = papers[0]
        collection_name = collections[0]["name"]
        
        print(f"\nAdding paper '{paper_id}' to collection '{collection_name}'...")
        result = repo.add_to_collection(paper_id, collection_name)
        print(f"Add result: {result}")
        
        # Check if paper was added
        updated_collections = repo.get_collections()
        for collection in updated_collections:
            if collection["name"] == collection_name:
                print(f"Collection '{collection_name}' now has {collection['paper_count']} papers")
                
                # Check the actual papers in collection
                collection_papers = repo.index.get("collections", {}).get(collection_name, {}).get("papers", [])
                print(f"Papers in collection: {collection_papers}")
                
                # Check if paper has collection reference
                paper_collections = repo.index["papers"][paper_id].get("collections", [])
                print(f"Paper collections: {paper_collections}")
    
    print("\nTest complete!")

if __name__ == '__main__':
    test_collections()