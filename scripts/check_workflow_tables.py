#!/usr/bin/env python3
"""
Check database tables for TidyLLM-HeirOS workflow storage
Identify existing tables and suggest new schema if needed
"""
import psycopg2
from psycopg2.extras import RealDictCursor

def analyze_workflow_tables():
    """Analyze existing tables for workflow storage potential"""
    
    conn = psycopg2.connect(
        host='vectorqa-cluster.cluster-cu562e4m02nq.us-east-1.rds.amazonaws.com',
        port=5432,
        database='vectorqa',
        user='vectorqa_user', 
        password='REMOVED_PASSWORD',
        sslmode='require'
    )
    
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    print("="*80)
    print("TIDYLLM-HEIROS WORKFLOW DATABASE ANALYSIS")
    print("="*80)
    
    # 1. Check for existing workflow-related tables
    print("\n1. EXISTING WORKFLOW-RELATED TABLES:")
    print("-" * 50)
    
    workflow_keywords = ['workflow', 'dag', 'node', 'process', 'execution', 'sparse', 'agreement', 'heiros', 'batch']
    
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """)
    
    all_tables = [row['table_name'] for row in cursor.fetchall()]
    workflow_tables = []
    
    for table_name in all_tables:
        if any(keyword in table_name.lower() for keyword in workflow_keywords):
            workflow_tables.append(table_name)
    
    if workflow_tables:
        for table in workflow_tables:
            cursor.execute("""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns 
                WHERE table_name = %s
                ORDER BY ordinal_position;
            """, (table,))
            
            columns = cursor.fetchall()
            
            cursor.execute(f"SELECT COUNT(*) as count FROM {table};")
            row_count = cursor.fetchone()['count']
            
            print(f"\nTABLE: {table} ({row_count} rows)")
            for col in columns:
                nullable = "NULL" if col['is_nullable'] == 'YES' else "NOT NULL"
                default = f" DEFAULT {col['column_default']}" if col['column_default'] else ""
                print(f"  {col['column_name']:<25} {col['data_type']:<15} {nullable}{default}")
    else:
        print("No existing workflow tables found.")
    
    # 2. Check existing tables that could store workflows
    print(f"\n2. EXISTING TABLES SUITABLE FOR WORKFLOW STORAGE:")
    print("-" * 50)
    
    suitable_tables = []
    for table_name in all_tables:
        if any(word in table_name.lower() for word in ['events', 'metadata', 'data', 'records']):
            suitable_tables.append(table_name)
    
    for table in suitable_tables[:5]:  # Show first 5
        cursor.execute(f"SELECT COUNT(*) as count FROM {table};")
        row_count = cursor.fetchone()['count']
        print(f"  {table:<30} {row_count:>8} rows")
    
    # 3. Suggest new workflow tables schema
    print(f"\n3. SUGGESTED WORKFLOW TABLES SCHEMA:")
    print("-" * 50)
    
    workflow_schema = {
        "heiros_workflows": {
            "description": "Main workflow definitions",
            "columns": [
                "workflow_id UUID PRIMARY KEY",
                "name VARCHAR(255) NOT NULL",
                "description TEXT",
                "compliance_level VARCHAR(50)",
                "workflow_json JSONB NOT NULL",
                "version VARCHAR(20) DEFAULT '1.0'",
                "status VARCHAR(20) DEFAULT 'active'",
                "created_date TIMESTAMP DEFAULT NOW()",
                "updated_date TIMESTAMP DEFAULT NOW()",
                "created_by VARCHAR(100)",
                "tags JSONB"
            ]
        },
        
        "heiros_executions": {
            "description": "Workflow execution history", 
            "columns": [
                "execution_id UUID PRIMARY KEY",
                "workflow_id UUID REFERENCES heiros_workflows(workflow_id)",
                "execution_date TIMESTAMP DEFAULT NOW()",
                "status VARCHAR(20)",
                "duration_ms INTEGER",
                "context_data JSONB",
                "result_data JSONB",
                "node_results JSONB",
                "compliance_report JSONB",
                "executed_by VARCHAR(100)",
                "error_message TEXT"
            ]
        },
        
        "heiros_sparse_agreements": {
            "description": "SPARSE agreement definitions",
            "columns": [
                "agreement_id UUID PRIMARY KEY", 
                "title VARCHAR(255) NOT NULL",
                "description TEXT",
                "business_purpose TEXT",
                "business_owner VARCHAR(100)",
                "technical_owner VARCHAR(100)",
                "risk_level VARCHAR(20)",
                "status VARCHAR(20) DEFAULT 'pending'",
                "agreement_json JSONB NOT NULL",
                "created_date TIMESTAMP DEFAULT NOW()",
                "approved_date TIMESTAMP",
                "expiry_date TIMESTAMP",
                "execution_count INTEGER DEFAULT 0"
            ]
        },
        
        "heiros_node_templates": {
            "description": "Reusable workflow node templates",
            "columns": [
                "template_id UUID PRIMARY KEY",
                "name VARCHAR(255) NOT NULL",
                "description TEXT", 
                "node_type VARCHAR(50)",
                "template_json JSONB NOT NULL",
                "category VARCHAR(100)",
                "version VARCHAR(20) DEFAULT '1.0'",
                "created_date TIMESTAMP DEFAULT NOW()",
                "usage_count INTEGER DEFAULT 0"
            ]
        }
    }
    
    for table_name, schema_info in workflow_schema.items():
        print(f"\nCREATE TABLE {table_name} (")
        for column in schema_info["columns"]:
            print(f"  {column},")
        print(f");")
        print(f"-- {schema_info['description']}")
    
    # 4. Check if we can create the tables
    print(f"\n4. CURRENT DATABASE PERMISSIONS:")
    print("-" * 50)
    
    try:
        # Try to check if we have CREATE privileges
        cursor.execute("""
            SELECT has_database_privilege('vectorqa_user', 'vectorqa', 'CREATE') as can_create;
        """)
        can_create = cursor.fetchone()['can_create']
        print(f"Can create tables: {'YES' if can_create else 'NO'}")
        
        if can_create:
            print("\nRECOMMENDATION: Create dedicated workflow tables")
        else:
            print("\nRECOMMENDATION: Use existing tables with JSONB columns")
            print("Suggested options:")
            print("  1. Use 'events_raw' table for workflow executions")
            print("  2. Use 'document_metadata' table for workflow definitions")
            print("  3. Use 'mlflow_integration' table for SPARSE agreements")
        
    except Exception as e:
        print(f"Permission check failed: {e}")
    
    cursor.close()
    conn.close()
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    analyze_workflow_tables()