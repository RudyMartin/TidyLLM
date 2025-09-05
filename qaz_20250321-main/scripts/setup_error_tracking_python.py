#!/usr/bin/env python3
"""
Setup script for error tracking tables using credential manager
"""

import os
import sys
import psycopg2
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def setup_error_tracking():
    """Set up error tracking tables using credential manager"""
    
    print("🚀 Setting up Error Tracking Tables")
    print("=" * 50)
    
    try:
        # Import credential manager
        from backend.config.credential_manager import credential_manager
        
        # Get database URL from credential manager
        db_config = credential_manager.get_database_config()
        database_url = db_config.get('url')
        
        if not database_url:
            print("❌ DATABASE_URL not found in credential manager")
            return False
            
        print(f"🔗 Connecting to database using credential manager...")
        conn = psycopg2.connect(database_url)
        cursor = conn.cursor()
        
        # Test basic connection
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"✅ Connected successfully! PostgreSQL version: {version[0]}")
        
        # Read and execute schema creation
        schema_file = Path("database/simple_error_tracking_schema.sql")
        if not schema_file.exists():
            print(f"❌ Schema file not found: {schema_file}")
            return False
            
        print(f"📋 Creating schema from {schema_file}...")
        with open(schema_file, 'r') as f:
            schema_sql = f.read()
        
        # Split and execute SQL statements
        statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
        
        for i, statement in enumerate(statements, 1):
            if statement:
                try:
                    cursor.execute(statement)
                    print(f"   ✅ Statement {i} executed successfully")
                except Exception as e:
                    print(f"   ⚠️  Statement {i} failed (may already exist): {e}")
        
        conn.commit()
        print("✅ Schema creation completed")
        
        # Read and execute mock data insertion
        mock_data_file = Path("database/mock_data_error_tracking.sql")
        if mock_data_file.exists():
            print(f"📊 Inserting mock data from {mock_data_file}...")
            with open(mock_data_file, 'r') as f:
                mock_data_sql = f.read()
            
            # Split and execute SQL statements
            statements = [stmt.strip() for stmt in mock_data_sql.split(';') if stmt.strip()]
            
            for i, statement in enumerate(statements, 1):
                if statement:
                    try:
                        cursor.execute(statement)
                        print(f"   ✅ Mock data statement {i} executed successfully")
                    except Exception as e:
                        print(f"   ⚠️  Mock data statement {i} failed: {e}")
            
            conn.commit()
            print("✅ Mock data insertion completed")
        else:
            print(f"⚠️  Mock data file not found: {mock_data_file}")
        
        # Verify tables were created
        print("\n🔍 Verifying table creation...")
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
                AND table_name IN (
                    'prompt_history', 
                    'prompt_pipeline_errors', 
                    'error_patterns', 
                    'alert_history', 
                    'real_time_context', 
                    'batch_processing_status',
                    'mlflow_integration'
                )
            ORDER BY table_name;
        """)
        
        tables = cursor.fetchall()
        if tables:
            print("✅ Created tables:")
            for table in tables:
                print(f"   - {table[0]}")
        else:
            print("❌ No expected tables found")
            return False
        
        # Test table counts
        print("\n📊 Testing table counts...")
        for table in tables:
            table_name = table[0]
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
                count = cursor.fetchone()[0]
                print(f"   ✅ {table_name}: {count} records")
            except Exception as e:
                print(f"   ❌ {table_name}: {e}")
        
        print("\n🎉 Error tracking setup completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Could not import credential manager: {e}")
        return False
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'conn' in locals():
            cursor.close()
            conn.close()
            print("🔌 Database connection closed")

if __name__ == "__main__":
    success = setup_error_tracking()
    sys.exit(0 if success else 1)
