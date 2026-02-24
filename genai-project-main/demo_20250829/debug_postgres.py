#!/usr/bin/env python3
"""
Debug PostgreSQL Connection - Shows exactly what's happening
"""

import psycopg2
import urllib.parse
import yaml
import sys


def debug_connection():
    """Debug PostgreSQL connection with detailed output"""
    print("="*60)
    print("PostgreSQL Connection Debug")
    print("="*60)
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    pg_config = config['postgres']
    
    print(f"\n📋 Raw Configuration:")
    print(f"  Host: {pg_config['host']}")
    print(f"  Port: {pg_config['port']}")
    print(f"  Database: {pg_config['database']}")
    print(f"  Username: {pg_config['username']}")
    print(f"  Password: {pg_config['password'][:3]}...{pg_config['password'][-3:]}")
    print(f"  SSL Mode: {pg_config.get('ssl_mode', 'require')}")
    
    # Show URL encoding
    password_encoded = urllib.parse.quote_plus(pg_config['password'])
    print(f"\n🔧 URL Encoding:")
    print(f"  Original: {pg_config['password']}")
    print(f"  Encoded:  {password_encoded}")
    
    # Try method 1: Individual parameters (most reliable)
    print(f"\n🔍 Method 1: Individual Parameters")
    try:
        conn = psycopg2.connect(
            host=pg_config['host'],
            port=pg_config['port'],
            database=pg_config['database'],
            user=pg_config['username'],
            password=pg_config['password'],  # Raw password
            sslmode=pg_config.get('ssl_mode', 'require')
        )
        print("✅ Method 1 SUCCESS!")
        cursor = conn.cursor()
        cursor.execute("SELECT version()")
        version = cursor.fetchone()[0]
        print(f"  PostgreSQL: {version[:50]}...")
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Method 1 FAILED: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
    
    # Try method 2: URL with encoding
    print(f"\n🔍 Method 2: Database URL")
    try:
        db_url = f"postgresql://{pg_config['username']}:{password_encoded}@{pg_config['host']}:{pg_config['port']}/{pg_config['database']}?sslmode={pg_config.get('ssl_mode', 'require')}"
        print(f"  URL: postgresql://{pg_config['username']}:***@{pg_config['host']}:{pg_config['port']}/{pg_config['database']}")
        
        conn = psycopg2.connect(db_url)
        print("✅ Method 2 SUCCESS!")
        cursor = conn.cursor()
        cursor.execute("SELECT version()")
        version = cursor.fetchone()[0]
        print(f"  PostgreSQL: {version[:50]}...")
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Method 2 FAILED: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
    
    # Try method 3: URL without SSL
    print(f"\n🔍 Method 3: Database URL (no SSL)")
    try:
        db_url = f"postgresql://{pg_config['username']}:{password_encoded}@{pg_config['host']}:{pg_config['port']}/{pg_config['database']}"
        conn = psycopg2.connect(db_url)
        print("✅ Method 3 SUCCESS!")
        cursor = conn.cursor()
        cursor.execute("SELECT version()")
        version = cursor.fetchone()[0]
        print(f"  PostgreSQL: {version[:50]}...")
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Method 3 FAILED: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
    
    # Try method 4: Individual parameters without SSL
    print(f"\n🔍 Method 4: Individual Parameters (no SSL)")
    try:
        conn = psycopg2.connect(
            host=pg_config['host'],
            port=pg_config['port'],
            database=pg_config['database'],
            user=pg_config['username'],
            password=pg_config['password'],
            sslmode='disable'
        )
        print("✅ Method 4 SUCCESS!")
        cursor = conn.cursor()
        cursor.execute("SELECT version()")
        version = cursor.fetchone()[0]
        print(f"  PostgreSQL: {version[:50]}...")
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Method 4 FAILED: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
    
    print(f"\n💡 Troubleshooting Tips:")
    print(f"  1. Check if the host is reachable")
    print(f"  2. Verify port {pg_config['port']} is open")
    print(f"  3. Confirm database name exists")
    print(f"  4. Test credentials manually")
    print(f"  5. Check firewall/security groups")
    
    return False


if __name__ == "__main__":
    debug_connection()