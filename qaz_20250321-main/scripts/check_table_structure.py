#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check Database Table Structure

This script checks the structure of existing database tables.
"""

import os
from dotenv import load_dotenv

def check_table_structure():
    """Check the structure of database tables"""
    
    print("🔍 Checking Database Table Structure")
    print("=" * 40)
    
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        
        # Load environment variables
        load_dotenv('src/backend/config/credentials.env')
        database_url = os.getenv('DATABASE_URL')
        
        if not database_url:
            print("❌ DATABASE_URL not found")
            return False
        
        # Connect to database
        conn = psycopg2.connect(database_url)
        
        # Check document_chunks table structure
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = 'document_chunks'
                ORDER BY ordinal_position;
            """)
            columns = cursor.fetchall()
            
            print("📋 document_chunks table structure:")
            print("-" * 40)
            for col in columns:
                print(f"  {col['column_name']}: {col['data_type']} ({'NULL' if col['is_nullable'] == 'YES' else 'NOT NULL'})")
        
        # Check chunk_embeddings table structure
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = 'chunk_embeddings'
                ORDER BY ordinal_position;
            """)
            columns = cursor.fetchall()
            
            print("\n📋 chunk_embeddings table structure:")
            print("-" * 40)
            for col in columns:
                print(f"  {col['column_name']}: {col['data_type']} ({'NULL' if col['is_nullable'] == 'YES' else 'NOT NULL'})")
        
        # Check sample data
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT COUNT(*) as count FROM document_chunks;")
            result = cursor.fetchone()
            print(f"\n📊 document_chunks has {result['count']} rows")
            
            cursor.execute("SELECT COUNT(*) as count FROM chunk_embeddings;")
            result = cursor.fetchone()
            print(f"📊 chunk_embeddings has {result['count']} rows")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error checking table structure: {e}")
        return False

if __name__ == "__main__":
    check_table_structure()
