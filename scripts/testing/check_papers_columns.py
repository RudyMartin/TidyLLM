#!/usr/bin/env python3
"""
Check actual column names in paper tables
"""
import psycopg2
from psycopg2.extras import RealDictCursor

def check_table_columns():
    password = "REMOVED_PASSWORD"
    
    try:
        conn = psycopg2.connect(
            host="vectorqa-cluster.cluster-cu562e4m02nq.us-east-1.rds.amazonaws.com",
            port=5432,
            database="vectorqa", 
            user="vectorqa_user",
            password=password,
            sslmode="require"
        )
        
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Check paper-related tables
        paper_tables = [
            'yrsn_downloaded_papers',
            'yrsn_paper_collections', 
            'document_metadata',
            'document_chunks'
        ]
        
        for table in paper_tables:
            print(f"\n=== {table.upper()} ===")
            
            # Get column info
            cursor.execute(f"""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns 
                WHERE table_name = '{table}'
                ORDER BY ordinal_position;
            """)
            columns = cursor.fetchall()
            
            print("Columns:")
            for col in columns:
                print(f"  {col['column_name']:<25} {col['data_type']:<15} {'NULL' if col['is_nullable'] == 'YES' else 'NOT NULL'}")
            
            # Get row count and sample
            cursor.execute(f"SELECT COUNT(*) as count FROM {table};")
            count = cursor.fetchone()['count']
            print(f"Rows: {count}")
            
            if count > 0:
                cursor.execute(f"SELECT * FROM {table} LIMIT 2;")
                samples = cursor.fetchall()
                print("Sample data:")
                for i, row in enumerate(samples, 1):
                    print(f"  Row {i}: {dict(row)}")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    check_table_columns()