#!/usr/bin/env python3
"""
Simple PostgreSQL Connection for Air-Gapped Environment
Just update config.yaml and run!
"""

import yaml
import psycopg2
import sys
import os


def load_config():
    """Load configuration from YAML file"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def test_postgres_connection():
    """Test PostgreSQL connection using config.yaml"""
    print("="*50)
    print("PostgreSQL Connection Test")
    print("="*50)
    
    # Load configuration
    config = load_config()
    pg_config = config['postgres']
    
    print(f"\nConnecting to:")
    print(f"  Host: {pg_config['host']}")
    print(f"  Database: {pg_config['database']}")
    print(f"  User: {pg_config['username']}")
    
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host=pg_config['host'],
            port=pg_config['port'],
            database=pg_config['database'],
            user=pg_config['username'],
            password=pg_config['password'],
            sslmode=pg_config.get('ssl_mode', 'require')
        )
        
        print("\n✅ Connected successfully!")
        
        # Run test query
        cursor = conn.cursor()
        cursor.execute("SELECT version()")
        version = cursor.fetchone()[0]
        print(f"\nPostgreSQL Version: {version[:50]}...")
        
        # Get current database info
        cursor.execute("SELECT current_database(), current_user, now()")
        db_info = cursor.fetchone()
        print(f"Database: {db_info[0]}")
        print(f"User: {db_info[1]}")
        print(f"Server Time: {db_info[2]}")
        
        # List tables
        cursor.execute("""
            SELECT tablename 
            FROM pg_tables 
            WHERE schemaname = 'public' 
            LIMIT 5
        """)
        tables = cursor.fetchall()
        
        if tables:
            print("\nTables (first 5):")
            for table in tables:
                print(f"  - {table[0]}")
        else:
            print("\nNo tables found in public schema")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"\n❌ Connection failed: {str(e)}")
        return False


def run_query(query):
    """Run a custom SQL query"""
    config = load_config()
    pg_config = config['postgres']
    
    try:
        conn = psycopg2.connect(
            host=pg_config['host'],
            port=pg_config['port'],
            database=pg_config['database'],
            user=pg_config['username'],
            password=pg_config['password'],
            sslmode=pg_config.get('ssl_mode', 'require')
        )
        
        cursor = conn.cursor()
        cursor.execute(query)
        
        # If SELECT query, fetch results
        if query.strip().upper().startswith('SELECT'):
            results = cursor.fetchall()
            for row in results:
                print(row)
        else:
            conn.commit()
            print(f"Query executed. Rows affected: {cursor.rowcount}")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run custom query from command line
        query = ' '.join(sys.argv[1:])
        print(f"Running query: {query}")
        run_query(query)
    else:
        # Run connection test
        test_postgres_connection()