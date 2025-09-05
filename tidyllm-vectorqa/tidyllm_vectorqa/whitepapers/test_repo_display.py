"""
Quick test to see what the Paper Repository page would show
"""

def test_repository_display():
    """Test what the repository page should display"""
    try:
        from paper_repository import get_paper_repository
        from backend_config import get_backend_config
        
        print("=== TESTING PAPER REPOSITORY DISPLAY ===")
        
        backend_config = get_backend_config()
        repo = get_paper_repository(backend_config)
        
        # Repository statistics
        stats = repo.get_repository_stats()
        print(f"📄 Total Papers: {stats['total_papers']}")
        print(f"💾 Total Size: {stats['total_size_mb']} MB")
        print(f"📁 Repository: Local + S3")
        print(f"🔍 Embeddings: Ready")
        
        # Source breakdown
        if stats["source_breakdown"]:
            print("\n📊 Papers by Source:")
            for source, count in stats["source_breakdown"].items():
                print(f"   📚 {source}: {count}")
        
        # List papers
        papers = repo.list_papers(limit=20)
        print(f"\n📋 Papers in Repository ({len(papers)}):")
        
        if papers:
            for paper in papers:
                print(f"\n📄 {paper['title'][:60]}... ({paper['source']})")
                print(f"   Authors: {', '.join(paper['authors'][:3])}")
                print(f"   Source: {paper['source']} | Size: {paper['file_size_mb']} MB")
                print(f"   Downloaded: {paper['download_date'][:10]}")
                if paper.get('y_score'):
                    print(f"   Y Score: {paper['y_score']:.3f}")
        else:
            print("   No papers in repository. Download some papers from the Search tab!")
        
        # Collections
        collections = repo.get_collections()
        print(f"\n📁 Collections ({len(collections)}):")
        if collections:
            for collection in collections:
                print(f"   📁 {collection['name']} ({collection['paper_count']} papers)")
                print(f"      Description: {collection['description']}")
        else:
            print("   No collections yet.")
            
        print(f"\n📂 Repository Path: {stats['repository_path']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_repository_display()