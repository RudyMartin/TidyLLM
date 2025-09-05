#!/usr/bin/env python3
"""
Test script to verify error tracking tables are live and accessible
"""

import os
import sys
import psycopg2
from datetime import datetime, timedelta
import json

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_database_connection():
    """Test basic database connectivity using credential manager"""
    try:
        # Import credential manager
        from backend.config.credential_manager import credential_manager
        
        # Get database URL from credential manager
        db_config = credential_manager.get_database_config()
        database_url = db_config.get('url')
        
        if not database_url:
            print("❌ DATABASE_URL not found in credential manager")
            print("💡 Check your credentials file or environment variables")
            return False
            
        print(f"🔗 Connecting to database using credential manager...")
        conn = psycopg2.connect(database_url)
        cursor = conn.cursor()
        
        # Test basic connection
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"✅ Connected successfully! PostgreSQL version: {version[0]}")
        
        return conn, cursor
        
    except ImportError as e:
        print(f"❌ Could not import credential manager: {e}")
        return False
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def test_error_tracking_tables(conn, cursor):
    """Test if error tracking tables exist and are accessible"""
    tables_to_check = [
        'prompt_history',
        'prompt_pipeline_errors', 
        'error_patterns',
        'alert_history',
        'real_time_context',
        'batch_processing_status',
        'mlflow_integration'
    ]
    
    print("\n📊 Testing error tracking tables...")
    
    for table in tables_to_check:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table};")
            count = cursor.fetchone()[0]
            print(f"✅ {table}: {count} records")
        except Exception as e:
            print(f"❌ {table}: {e}")

def test_sample_queries(conn, cursor):
    """Run some sample queries to test functionality"""
    print("\n🔍 Running sample queries...")
    
    queries = [
        ("Recent errors by severity", 
         "SELECT severity, COUNT(*) FROM prompt_pipeline_errors WHERE created_at > NOW() - INTERVAL '24 hours' GROUP BY severity;"),
        
        ("Top error patterns", 
         "SELECT error_type, COUNT(*) FROM error_patterns GROUP BY error_type ORDER BY COUNT(*) DESC LIMIT 5;"),
        
        ("Recent prompt history", 
         "SELECT task_type, success, created_at FROM prompt_history ORDER BY created_at DESC LIMIT 3;"),
        
        ("Active alerts", 
         "SELECT alert_type, message, sent_at FROM alert_history WHERE status = 'sent' ORDER BY sent_at DESC LIMIT 3;")
    ]
    
    for query_name, query in queries:
        try:
            print(f"\n📋 {query_name}:")
            cursor.execute(query)
            results = cursor.fetchall()
            if results:
                for row in results:
                    print(f"   {row}")
            else:
                print("   No results")
        except Exception as e:
            print(f"   ❌ Query failed: {e}")

def test_mock_data_integration(conn, cursor):
    """Test if mock data is properly integrated"""
    print("\n🎯 Testing mock data integration...")
    
    try:
        # Check for specific mock data patterns
        cursor.execute("""
            SELECT COUNT(*) FROM prompt_pipeline_errors 
            WHERE error_message LIKE '%mock%' OR error_message LIKE '%test%'
        """)
        mock_count = cursor.fetchone()[0]
        print(f"✅ Found {mock_count} mock/test error records")
        
        # Check for realistic error patterns
        cursor.execute("""
            SELECT error_type, COUNT(*) 
            FROM prompt_pipeline_errors 
            GROUP BY error_type 
            ORDER BY COUNT(*) DESC
        """)
        error_types = cursor.fetchall()
        print(f"✅ Error types found: {[et[0] for et in error_types]}")
        
    except Exception as e:
        print(f"❌ Mock data test failed: {e}")

def main():
    """Main test function"""
    print("🚀 Testing Error Tracking Database Connection")
    print("=" * 50)
    
    # Test connection
    result = test_database_connection()
    if not result:
        print("❌ Cannot proceed without database connection")
        return
    
    conn, cursor = result
    
    try:
        # Test tables
        test_error_tracking_tables(conn, cursor)
        
        # Test sample queries
        test_sample_queries(conn, cursor)
        
        # Test mock data
        test_mock_data_integration(conn, cursor)
        
        print("\n🎉 All tests completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    finally:
        cursor.close()
        conn.close()
        print("🔌 Database connection closed")

if __name__ == "__main__":
    main()
