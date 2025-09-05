#!/usr/bin/env python3
"""
Check paper and file counts from PostgreSQL database
Alternative to S3 direct access
"""
import psycopg2
from psycopg2.extras import RealDictCursor

def check_papers_in_database():
    password = "REMOVED_PASSWORD"
    
    try:
        print("="*80)
        print("PAPERS & FILES FROM DATABASE RECORDS")
        print("="*80)
        
        conn = psycopg2.connect(
            host="vectorqa-cluster.cluster-cu562e4m02nq.us-east-1.rds.amazonaws.com",
            port=5432,
            database="vectorqa", 
            user="vectorqa_user",
            password=password,
            sslmode="require",
            connect_timeout=10
        )
        
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Check research paper tables
        print("\n" + "-"*60)
        print("RESEARCH PAPERS")
        print("-"*60)
        
        # Downloaded papers
        cursor.execute("SELECT COUNT(*) as count FROM yrsn_downloaded_papers;")
        downloaded_papers = cursor.fetchone()['count']
        print(f"Downloaded papers: {downloaded_papers}")
        
        if downloaded_papers > 0:
            cursor.execute("SELECT title, s3_key, file_size FROM yrsn_downloaded_papers LIMIT 3;")
            samples = cursor.fetchall()
            print("Sample downloads:")
            for paper in samples:
                size_mb = paper['file_size'] / (1024*1024) if paper['file_size'] else 0
                print(f"  {paper['title'][:50]}... | S3: {paper['s3_key']} | Size: {size_mb:.1f} MB")
        
        # Paper collections
        cursor.execute("SELECT COUNT(*) as count FROM yrsn_paper_collections;")
        collections = cursor.fetchone()['count']
        print(f"Paper collections: {collections}")
        
        # Search results (may indicate available papers)
        cursor.execute("SELECT COUNT(*) as count FROM yrsn_search_results;")
        search_results = cursor.fetchone()['count']
        print(f"Search results: {search_results}")
        
        # Document processing
        print("\n" + "-"*60)
        print("DOCUMENT PROCESSING")
        print("-"*60)
        
        # Document metadata
        cursor.execute("SELECT COUNT(*) as count FROM document_metadata;")
        doc_metadata = cursor.fetchone()['count']
        print(f"Document metadata records: {doc_metadata}")
        
        if doc_metadata > 0:
            cursor.execute("SELECT file_name, file_size, s3_path FROM document_metadata LIMIT 3;")
            docs = cursor.fetchall()
            print("Sample documents:")
            for doc in docs:
                size_mb = doc['file_size'] / (1024*1024) if doc['file_size'] else 0
                print(f"  {doc['file_name']} | S3: {doc['s3_path']} | Size: {size_mb:.1f} MB")
        
        # Document chunks (processed content)
        cursor.execute("SELECT COUNT(*) as count FROM document_chunks;")
        chunks = cursor.fetchone()['count']
        print(f"Document chunks: {chunks}")
        
        # Embeddings
        cursor.execute("SELECT COUNT(*) as count FROM chunk_embeddings;")
        embeddings = cursor.fetchone()['count']
        print(f"Chunk embeddings: {embeddings}")
        
        # S3 path analysis
        print("\n" + "-"*60)
        print("S3 PATH ANALYSIS")
        print("-"*60)
        
        # Check for S3 paths in different tables
        s3_tables = [
            ('yrsn_downloaded_papers', 's3_key'),
            ('document_metadata', 's3_path')
        ]
        
        total_s3_files = 0
        for table, column in s3_tables:
            cursor.execute(f"SELECT COUNT(*) as count FROM {table} WHERE {column} IS NOT NULL AND {column} != '';")
            count = cursor.fetchone()['count']
            print(f"S3 files in {table}: {count}")
            total_s3_files += count
            
            if count > 0:
                cursor.execute(f"SELECT {column} FROM {table} WHERE {column} IS NOT NULL AND {column} != '' LIMIT 3;")
                paths = cursor.fetchall()
                for path_row in paths:
                    path = path_row[column]
                    if 'papers/papers/' in path:
                        print(f"  Found papers/papers/ path: {path}")
        
        # Search for any references to papers/papers/
        print(f"\nTotal S3 file references: {total_s3_files}")
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Database indicates {downloaded_papers + doc_metadata} files processed")
        print(f"Document chunks: {chunks} (indicates {doc_metadata} documents were processed)")
        print(f"S3 file references: {total_s3_files}")
        
        cursor.close()
        conn.close()
        
        return downloaded_papers, doc_metadata, chunks, total_s3_files
        
    except Exception as e:
        print(f"ERROR: Failed to check database: {e}")
        return 0, 0, 0, 0

if __name__ == "__main__":
    papers, docs, chunks, s3_refs = check_papers_in_database()
    print(f"\nESTIMATE: {papers + docs} papers/documents likely in S3 based on database records")
    if chunks > 0:
        print(f"Processing confirmed: {chunks} document chunks indicate active document processing")