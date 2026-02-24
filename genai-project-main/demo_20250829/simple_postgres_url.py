#!/usr/bin/env python3
"""
PostgreSQL Connection using Database URL
Handles special characters by URL encoding the password
"""

import psycopg2
import urllib.parse
import yaml
import sys


def load_config():
    """Load configuration from YAML file"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def test_postgres_connection_url():
    """Test PostgreSQL connection using database URL"""
    print("="*50)
    print("PostgreSQL Connection Test (Database URL)")
    print("="*50)
    
    config = load_config()
    pg_config = config['postgres']
    
    # URL encode the password to handle special characters
    password_encoded = urllib.parse.quote_plus(pg_config['password'])
    
    # Build database URL
    db_url = f"postgresql://{pg_config['username']}:{password_encoded}@{pg_config['host']}:{pg_config['port']}/{pg_config['database']}?sslmode={pg_config.get('ssl_mode', 'require')}"
    
    print(f"\nConnecting to:")
    print(f"  Host: {pg_config['host']}")
    print(f"  Database: {pg_config['database']}")
    print(f"  User: {pg_config['username']}")
    print(f"  URL: postgresql://{pg_config['username']}:***@{pg_config['host']}:{pg_config['port']}/{pg_config['database']}")
    
    try:
        # Connect using URL
        conn = psycopg2.connect(db_url)
        
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
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"\n❌ Connection failed: {str(e)}")
        print(f"\nTroubleshooting:")
        print(f"- Make sure your password is correct in config.yaml")
        print(f"- Special characters in password are automatically URL-encoded")
        print(f"- Check host, port, database name, and username")
        return False


def run_query_url(query):
    """Run a custom SQL query using URL connection"""
    config = load_config()
    pg_config = config['postgres']
    
    # URL encode the password
    password_encoded = urllib.parse.quote_plus(pg_config['password'])
    
    # Build database URL
    db_url = f"postgresql://{pg_config['username']}:{password_encoded}@{pg_config['host']}:{pg_config['port']}/{pg_config['database']}?sslmode={pg_config.get('ssl_mode', 'require')}"
    
    try:
        conn = psycopg2.connect(db_url)
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
        run_query_url(query)
    else:
        # Run connection test
        test_postgres_connection_url()