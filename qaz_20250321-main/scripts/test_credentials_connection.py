#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Database Connection using Credentials File

Simple script to test the connection to Aurora PostgreSQL using credentials.env
"""

import os
import sys

def test_connection():
    """Test the database connection using credentials file"""
    
    print("🔍 Testing Aurora PostgreSQL Connection")
    print("=" * 40)
    
    try:
        # Import dependencies
        import psycopg2
        from dotenv import load_dotenv
        
        # Load credentials
        credentials_path = "src/backend/config/credentials.env"
        load_dotenv(credentials_path)
        
        # Get database URL
        database_url = os.getenv('DATABASE_URL')
        
        if not database_url:
            print("❌ DATABASE_URL not found in credentials")
            return False
        
        print("Connecting to: vectorqa-cluster.cluster-cu562e4m02nq.us-east-1.rds.amazonaws.com")
        
        # Test connection
        conn = psycopg2.connect(database_url)
        
        # Test basic query
        with conn.cursor() as cursor:
            cursor.execute("SELECT version();")
            version = cursor.fetchone()
            print("✅ Connected successfully!")
            print("Database: {}".format(version[0]))
        
        # Test pgvector extension
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
            vector_ext = cursor.fetchone()
            
            if vector_ext:
                print("✅ pgvector extension is available")
            else:
                print("⚠️  pgvector extension not found")
        
        # List tables
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """)
            tables = cursor.fetchall()
            
            print("📋 Available tables ({}):".format(len(tables)))
            for table in tables:
                print("  - {}".format(table[0]))
        
        conn.close()
        return True
        
    except ImportError as e:
        print("❌ Missing dependency: {}".format(e))
        return False
    except Exception as e:
        print("❌ Connection failed: {}".format(e))
        return False

if __name__ == "__main__":
    success = test_connection()
    if success:
        print("\n🎉 Database connection test passed!")
        print("Your RAG orchestrator is ready to use the database!")
    else:
        print("\n❌ Database connection test failed!")
        sys.exit(1)
