#!/usr/bin/env python3
"""
Document PostgreSQL Database Schema
Connects to PostgreSQL and documents all tables, columns, and relationships
"""

    # #future_fix: Convert to use enhanced service infrastructure
import psycopg2
import os
from datetime import datetime
import json

def get_connection():
    """Get PostgreSQL connection using environment variables or defaults"""
    conn_params = {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': os.getenv('POSTGRES_PORT', '5432'),
        'database': os.getenv('POSTGRES_DB', 'mlflow'),
        'user': os.getenv('POSTGRES_USER', 'mlflowuser'),
        'password': os.getenv('POSTGRES_PASSWORD', 'mlflowpassword')
    }
    
    try:
    # #future_fix: Convert to use enhanced service infrastructure
        return psycopg2.connect(**conn_params)
    except Exception as e:
        print(f"Connection failed with defaults, error: {e}")
        # Try alternate database name
        conn_params['database'] = 'postgres'
        try:
    # #future_fix: Convert to use enhanced service infrastructure
            return psycopg2.connect(**conn_params)
        except Exception as e2:
            print(f"Connection failed with postgres db: {e2}")
            raise

def document_schema(conn):
    """Document all tables and their structure"""
    cur = conn.cursor()
    
    # Get all schemas
    cur.execute("""
        SELECT schema_name 
        FROM information_schema.schemata 
        WHERE schema_name NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
        ORDER BY schema_name
    """)
    schemas = cur.fetchall()
    
    documentation = {
        'generated_at': datetime.now().isoformat(),
        'schemas': {}
    }
    
    for schema_row in schemas:
        schema_name = schema_row[0]
        print(f"\nSchema: {schema_name}")
        
        # Get all tables in schema
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = %s 
            AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """, (schema_name,))
        
        tables = cur.fetchall()
        schema_doc = {'tables': {}}
        
        for table_row in tables:
            table_name = table_row[0]
            print(f"  Table: {table_name}")
            
            # Get columns
            cur.execute("""
                SELECT 
                    column_name,
                    data_type,
                    character_maximum_length,
                    is_nullable,
                    column_default
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s
                ORDER BY ordinal_position
            """, (schema_name, table_name))
            
            columns = cur.fetchall()
            
            # Get primary keys
            cur.execute("""
                SELECT a.attname
                FROM pg_index i
                JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
                WHERE i.indrelid = %s::regclass AND i.indisprimary
            """, (f"{schema_name}.{table_name}",))
            
            primary_keys = [row[0] for row in cur.fetchall()]
            
            # Get foreign keys
            cur.execute("""
                SELECT
                    tc.constraint_name, 
                    kcu.column_name, 
                    ccu.table_schema AS foreign_table_schema,
                    ccu.table_name AS foreign_table_name,
                    ccu.column_name AS foreign_column_name 
                FROM information_schema.table_constraints AS tc 
                JOIN information_schema.key_column_usage AS kcu
                  ON tc.constraint_name = kcu.constraint_name
                  AND tc.table_schema = kcu.table_schema
                JOIN information_schema.constraint_column_usage AS ccu
                  ON ccu.constraint_name = tc.constraint_name
                  AND ccu.table_schema = tc.table_schema
                WHERE tc.constraint_type = 'FOREIGN KEY' 
                  AND tc.table_schema = %s
                  AND tc.table_name = %s
            """, (schema_name, table_name))
            
            foreign_keys = cur.fetchall()
            
            # Get row count
            # Use psycopg2.sql for safe identifier interpolation
            from psycopg2 import sql
            cur.execute(
                sql.SQL("SELECT COUNT(*) FROM {}.{}").format(
                    sql.Identifier(schema_name),
                    sql.Identifier(table_name)
                )
            )
            result = cur.fetchone()
            row_count = result[0] if isinstance(result, tuple) else result['count']
            
            table_doc = {
                'row_count': row_count,
                'columns': [],
                'primary_keys': primary_keys,
                'foreign_keys': []
            }
            
            for col in columns:
                col_doc = {
                    'name': col[0],
                    'type': col[1],
                    'max_length': col[2],
                    'nullable': col[3] == 'YES',
                    'default': col[4],
                    'is_primary': col[0] in primary_keys
                }
                table_doc['columns'].append(col_doc)
            
            for fk in foreign_keys:
                fk_doc = {
                    'constraint': fk[0],
                    'column': fk[1],
                    'references': f"{fk[2]}.{fk[3]}.{fk[4]}"
                }
                table_doc['foreign_keys'].append(fk_doc)
            
            schema_doc['tables'][table_name] = table_doc
        
        documentation['schemas'][schema_name] = schema_doc
    
    cur.close()
    return documentation

def generate_markdown_report(doc):
    """Generate markdown documentation from schema documentation"""
    md = []
    md.append("# PostgreSQL Database Schema Documentation")
    md.append(f"\n**Generated**: {doc['generated_at']}")
    md.append("\n---\n")
    
    # Summary statistics
    total_tables = sum(len(s['tables']) for s in doc['schemas'].values())
    total_rows = sum(
        t['row_count'] 
        for s in doc['schemas'].values() 
        for t in s['tables'].values()
    )
    
    md.append("## Database Summary\n")
    md.append(f"- **Schemas**: {len(doc['schemas'])}")
    md.append(f"- **Tables**: {total_tables}")
    md.append(f"- **Total Rows**: {total_rows:,}")
    md.append("\n---\n")
    
    # Schema details
    for schema_name, schema_data in sorted(doc['schemas'].items()):
        if not schema_data['tables']:
            continue
            
        md.append(f"## Schema: `{schema_name}`\n")
        
        for table_name, table_data in sorted(schema_data['tables'].items()):
            md.append(f"### Table: `{schema_name}.{table_name}`")
            md.append(f"**Rows**: {table_data['row_count']:,}\n")
            
            if table_data['primary_keys']:
                md.append(f"**Primary Keys**: {', '.join(table_data['primary_keys'])}\n")
            
            # Columns table
            md.append("#### Columns\n")
            md.append("| Column | Type | Nullable | Default | Primary |")
            md.append("|--------|------|----------|---------|---------|")
            
            for col in table_data['columns']:
                type_str = col['type']
                if col['max_length']:
                    type_str += f"({col['max_length']})"
                
                nullable = "YES" if col['nullable'] else "NO"
                default = col['default'] if col['default'] else "-"
                primary = "PK" if col['is_primary'] else "-"
                
                # Escape pipe characters in default values
                if default != "-":
                    default = default.replace("|", "\\|")
                
                md.append(f"| {col['name']} | {type_str} | {nullable} | {default} | {primary} |")
            
            # Foreign keys
            if table_data['foreign_keys']:
                md.append("\n#### Foreign Keys\n")
                for fk in table_data['foreign_keys']:
                    md.append(f"- `{fk['column']}` → `{fk['references']}` ({fk['constraint']})")
            
            md.append("")  # Empty line between tables
    
    return "\n".join(md)

def main():
    """Main function to document database"""
    print("Connecting to PostgreSQL database...")
    
    try:
        conn = get_connection()
        print("Connected successfully!")
        
        print("\nDocumenting database schema...")
        doc = document_schema(conn)
        
        # Save JSON documentation
        json_file = "database_schema.json"
        with open(json_file, 'w') as f:
            json.dump(doc, f, indent=2)
        print(f"\nSaved JSON documentation to {json_file}")
        
        # Generate and save markdown
        markdown = generate_markdown_report(doc)
        md_file = "database_schema.md"
        with open(md_file, 'w') as f:
            f.write(markdown)
        print(f"Saved Markdown documentation to {md_file}")
        
        conn.close()
        
        # Print summary
        print("\n" + "="*60)
        print("DATABASE SUMMARY")
        print("="*60)
        for schema_name, schema_data in sorted(doc['schemas'].items()):
            if schema_data['tables']:
                print(f"\nSchema: {schema_name}")
                for table_name, table_data in sorted(schema_data['tables'].items()):
                    print(f"  - {table_name}: {table_data['row_count']} rows, {len(table_data['columns'])} columns")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())