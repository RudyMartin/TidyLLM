#!/usr/bin/env python3
"""
Show the exact location and full content of the PostgreSQL record
"""

import yaml
import json
from pathlib import Path

def show_exact_record():
    """Show exactly where the record is and its full content"""
    print("EXACT RECORD LOCATION AND CONTENT")
    print("=" * 50)
    
    # Load credentials
    settings_path = Path("C:/Users/marti/AI-Scoring/tidyllm/admin/settings.yaml")
    with open(settings_path, 'r') as f:
        config = yaml.safe_load(f)
    
    pg_creds = config['credentials']['postgresql']
    
    print("DATABASE CONNECTION DETAILS:")
    print(f"  Host: {pg_creds['host']}")
    print(f"  Port: {pg_creds['port']}")
    print(f"  Database: {pg_creds['database']}")
    print(f"  Username: {pg_creds['username']}")
    print(f"  Table: v2_ai_processing_evidence_test")
    print()
    
    import psycopg2
    
    # Connect and get the record
    conn = psycopg2.connect(
        host=pg_creds['host'],
        port=pg_creds['port'],
        database=pg_creds['database'],
        user=pg_creds['username'],
        password=pg_creds['password'],
        sslmode=pg_creds['ssl_mode']
    )
    
    cursor = conn.cursor()
    
    print("EXACT SQL TO ACCESS THE RECORD:")
    print("-" * 35)
    sql_command = """
    SELECT id, experiment_id, ai_response, confidence_score, created_at
    FROM v2_ai_processing_evidence_test
    ORDER BY created_at DESC
    LIMIT 1;
    """
    print(sql_command)
    
    # Execute and get the record
    cursor.execute(sql_command)
    record = cursor.fetchone()
    
    if record:
        id_val, experiment_id, ai_response, confidence_score, created_at = record
        
        print("FULL RECORD CONTENT:")
        print("-" * 20)
        print(f"ID: {id_val}")
        print(f"Experiment ID: {experiment_id}")
        print(f"Confidence Score: {confidence_score}")
        print(f"Created At: {created_at}")
        print()
        
        print("FULL AI RESPONSE JSON:")
        print("-" * 22)
        # Parse and pretty print the JSON
        try:
            ai_data = json.loads(ai_response)
            print(json.dumps(ai_data, indent=2))
        except:
            print(ai_response)
        
        print("\nHOW TO ACCESS THIS RECORD:")
        print("-" * 27)
        print("1. Connect to PostgreSQL using:")
        print(f"   psql -h {pg_creds['host']} -U {pg_creds['username']} -d {pg_creds['database']}")
        print()
        print("2. Or use any PostgreSQL client with these credentials:")
        print(f"   Host: {pg_creds['host']}")
        print(f"   Port: {pg_creds['port']}")
        print(f"   Database: {pg_creds['database']}")
        print(f"   Username: {pg_creds['username']}")
        print(f"   Password: [from settings.yaml]")
        print()
        print("3. Then run this SQL:")
        print("   SELECT * FROM v2_ai_processing_evidence_test;")
        print()
        
        print("RECORD PHYSICAL LOCATION:")
        print("-" * 25)
        print("AWS RDS PostgreSQL Cluster:")
        print(f"  Cluster: vectorqa-cluster")
        print(f"  Region: us-east-1")
        print(f"  Full Host: {pg_creds['host']}")
        print(f"  Schema: public")
        print(f"  Table: v2_ai_processing_evidence_test")
        print(f"  Record ID: {id_val}")
        print(f"  Created: {created_at}")
        
    else:
        print("No records found")
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    show_exact_record()