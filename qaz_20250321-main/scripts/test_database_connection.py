#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Database Connection

Simple script to test the connection to Aurora PostgreSQL.
"""

import os
import psycopg2
from dotenv import load_dotenv

def test_connection():
    """Test the database connection"""
    
    # Load environment variables
    load_dotenv('.env.local')
    
    database_url = os.getenv('DATABASE_URL')
    
    if not database_url:
        print("❌ DATABASE_URL not found in environment")
        print("Please run: python scripts/setup_database_connection.py")
        return False
    
    try:
        print("🔍 Testing connection to Aurora PostgreSQL...")
        print("Host: vectorqa-cluster.cluster-cu562e4m02nq.us-east-1.rds.amazonaws.com")
        
        # Connect to database
        conn = psycopg2.connect(database_url)
        
        # Test basic query
        with conn.cursor() as cursor:
            cursor.execute("SELECT version();")
            version = cursor.fetchone()
            print("✅ Connected successfully!")
            print("Database: {}".format(version[0]))
        
        # Test pgvector
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
        
    except Exception as e:
        print("❌ Connection failed: {}".format(e))
        return False

if __name__ == "__main__":
    success = test_connection()
    if success:
        print("\n🎉 Database connection test passed!")
    else:
        print("\n❌ Database connection test failed!")
        sys.exit(1)
