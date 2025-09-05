#!/usr/bin/env python3
"""
Check Database Schema

This script checks the PostgreSQL database schema to understand
the embedding dimension requirements and current table structure.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

import psycopg2
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_database_schema():
    """Check the database schema for embedding dimensions"""
    
    try:
        # Import credential manager
        from backend.config.credential_manager import credential_manager
        
        # Get database URL from credential manager
        db_config = credential_manager.get_database_config()
        database_url = db_config.get('url')
        
        if not database_url:
            logger.warning("DATABASE_URL not found in credential manager")
            print("🔍 Checking Database Schema...")
            print("❌ DATABASE_URL not found in credential manager")
            print("\n💡 To set up database connection:")
            print("1. Create environ_settings/.env.local")
            print("2. Add DATABASE_URL=postgresql://user:password@host:port/database")
            print("3. Set APP_ENV=local")
            return
        
        print("🔍 Checking Database Schema...")
        print(f"📊 Database URL: {database_url[:50]}..." if len(database_url) > 50 else f"📊 Database URL: {database_url}")
        
        # Connect to database
        conn = psycopg2.connect(database_url)
        cursor = conn.cursor()
        
        # Check if document_chunks table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'document_chunks'
            );
        """)
        
        table_exists = cursor.fetchone()[0]
        print(f"📋 document_chunks table exists: {table_exists}")
        
        if table_exists:
            # Get table structure
            cursor.execute("""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns 
                WHERE table_name = 'document_chunks'
                ORDER BY ordinal_position;
            """)
            
            columns = cursor.fetchall()
            print("\n📋 document_chunks table structure:")
            print("-" * 80)
            for col in columns:
                print(f"  {col[0]:<20} {col[1]:<20} {'NULL' if col[2] == 'YES' else 'NOT NULL':<10} {col[3] or ''}")
            
            # Check embedding column specifically
            cursor.execute("""
                SELECT column_name, data_type, character_maximum_length
                FROM information_schema.columns 
                WHERE table_name = 'document_chunks' AND column_name = 'embedding';
            """)
            
            embedding_col = cursor.fetchone()
            if embedding_col:
                print(f"\n🔢 Embedding column: {embedding_col[0]} ({embedding_col[1]})")
                if embedding_col[2]:
                    print(f"   Expected dimensions: {embedding_col[2]}")
            
            # Check pgvector extension
            cursor.execute("""
                SELECT extname, extversion 
                FROM pg_extension 
                WHERE extname = 'vector';
            """)
            
            pgvector = cursor.fetchone()
            if pgvector:
                print(f"\n✅ pgvector extension: {pgvector[0]} v{pgvector[1]}")
            else:
                print("\n❌ pgvector extension not found")
            
            # Check sample data
            cursor.execute("""
                SELECT COUNT(*) FROM document_chunks;
            """)
            
            count = cursor.fetchone()[0]
            print(f"\n📊 Total document chunks: {count}")
            
            if count > 0:
                # Check embedding dimensions in actual data
                cursor.execute("""
                    SELECT embedding_model, 
                           CASE 
                               WHEN embedding IS NOT NULL THEN embedding::text
                               ELSE NULL 
                           END as embedding_sample
                    FROM document_chunks 
                    WHERE embedding IS NOT NULL 
                    LIMIT 3;
                """)
                
                samples = cursor.fetchall()
                print(f"\n🔢 Sample embedding data:")
                for sample in samples:
                    model = sample[0]
                    embedding_text = sample[1]
                    if embedding_text:
                        # Parse the vector text to get dimensions
                        # Vector format: [0.1,0.2,0.3,...]
                        try:
                            # Remove brackets and split by comma
                            values = embedding_text.strip('[]').split(',')
                            dimensions = len(values)
                            print(f"   Model: {model}, Dimensions: {dimensions}")
                        except:
                            print(f"   Model: {model}, Raw: {embedding_text[:50]}...")
                    else:
                        print(f"   Model: {model}, No embedding data")
        
        cursor.close()
        conn.close()
        
        print("\n✅ Database schema check completed!")
        
    except Exception as e:
        print(f"❌ Database schema check failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_database_schema()
