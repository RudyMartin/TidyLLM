#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup Credentials Script

This script helps you set up your credentials.env file with your actual database password.
"""

import os
import getpass
import shutil

def setup_credentials():
    """Set up the credentials file with actual database password"""
    
    print("🔐 Setting up Credentials File")
    print("=" * 40)
    
    # Paths
    template_path = "src/backend/config/credentials_template.env"
    credentials_path = "src/backend/config/credentials.env"
    
    # Check if template exists
    if not os.path.exists(template_path):
        print("❌ Template file not found: {}".format(template_path))
        return False
    
    # Get database password
    print("Enter your Aurora PostgreSQL password:")
    db_password = getpass.getpass()
    
    try:
        # Read template
        with open(template_path, 'r') as f:
            content = f.read()
        
        # Replace password placeholder
        content = content.replace("DB_PASSWORD=your_actual_password_here", 
                                "DB_PASSWORD={}".format(db_password))
        
        # Write to credentials file
        with open(credentials_path, 'w') as f:
            f.write(content)
        
        print("✅ Credentials file created: {}".format(credentials_path))
        print("⚠️  This file is already in .gitignore for security")
        
        return True
        
    except Exception as e:
        print("❌ Failed to create credentials file: {}".format(e))
        return False

def test_database_connection():
    """Test the database connection using credentials file"""
    
    print("\n🔍 Testing Database Connection")
    print("=" * 30)
    
    try:
        # Import here to avoid issues if not installed
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
        
        # Test connection
        print("Connecting to Aurora PostgreSQL...")
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
                print("⚠️  pgvector extension not found - you may need to enable it")
        
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
        print("Please install: pip install psycopg2-binary python-dotenv")
        return False
    except Exception as e:
        print("❌ Connection failed: {}".format(e))
        return False

def main():
    """Main function"""
    
    print("🚀 Credentials Setup for Aurora PostgreSQL")
    print("=" * 50)
    
    # Step 1: Setup credentials file
    if setup_credentials():
        # Step 2: Test connection
        if test_database_connection():
            print("\n🎉 Database connection setup complete!")
            print("\n📝 Your RAG orchestrator will now use the database")
            print("📝 You can test with: python scripts/test_database_connection.py")
        else:
            print("\n⚠️  Connection test failed. Please check your password.")
    else:
        print("❌ Setup failed.")

if __name__ == "__main__":
    main()
