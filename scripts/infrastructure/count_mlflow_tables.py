#!/usr/bin/env python3
"""
Count and analyze MLflow tables in PostgreSQL
"""
import psycopg2
from psycopg2.extras import RealDictCursor

def count_mlflow_tables():
    password = "REMOVED_PASSWORD"
    
    try:
        conn = psycopg2.connect(
            host="vectorqa-cluster.cluster-cu562e4m02nq.us-east-1.rds.amazonaws.com",
            port=5432,
            database="vectorqa", 
            user="vectorqa_user",
            password=password,
            sslmode="require",
            connect_timeout=10
        )
        
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        print("="*80)
        print("MLFLOW TABLES IN POSTGRESQL")
        print("="*80)
        
        # Get all tables
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        all_tables = cursor.fetchall()
        all_table_names = [t['table_name'] for t in all_tables]
        
        # Identify MLflow tables (standard MLflow schema)
        mlflow_patterns = [
            'experiment',
            'run',
            'metric',
            'param', 
            'tag',
            'model',
            'alembic'  # MLflow migration table
        ]
        
        mlflow_tables = []
        other_tables = []
        
        for table_name in all_table_names:
            is_mlflow = any(pattern in table_name.lower() for pattern in mlflow_patterns)
            if is_mlflow:
                mlflow_tables.append(table_name)
            else:
                other_tables.append(table_name)
        
        # Display results
        print(f"\nTOTAL TABLES: {len(all_table_names)}")
        print(f"MLFLOW TABLES: {len(mlflow_tables)}")
        print(f"OTHER TABLES: {len(other_tables)}")
        
        print(f"\n" + "-"*60)
        print("MLFLOW TABLES:")
        print("-"*60)
        for table in sorted(mlflow_tables):
            # Get row count
            cursor.execute(f"SELECT COUNT(*) as count FROM {table};")
            count = cursor.fetchone()['count']
            print(f"  {table:<30} {count:>10} rows")
        
        print(f"\n" + "-"*60)
        print("OTHER/CUSTOM TABLES:")
        print("-"*60)
        for table in sorted(other_tables):
            # Get row count
            cursor.execute(f"SELECT COUNT(*) as count FROM {table};")
            count = cursor.fetchone()['count']
            print(f"  {table:<30} {count:>10} rows")
        
        # Analyze MLflow table relationships
        print(f"\n" + "-"*60)
        print("MLFLOW SCHEMA ANALYSIS:")
        print("-"*60)
        
        # Core MLflow tables analysis
        core_tables = {
            'experiments': 'Experiment definitions',
            'runs': 'Individual experiment runs', 
            'metrics': 'Run metrics/measurements',
            'latest_metrics': 'Latest metric values',
            'params': 'Run parameters',
            'tags': 'Run/experiment tags',
            'models': 'Registered models',
            'model_versions': 'Model version history'
        }
        
        for table, description in core_tables.items():
            if table in all_table_names:
                cursor.execute(f"SELECT COUNT(*) as count FROM {table};")
                count = cursor.fetchone()['count']
                status = "ACTIVE" if count > 0 else "EMPTY"
                print(f"  {status} {table:<20} {count:>6} rows - {description}")
            else:
                print(f"  MISSING {table:<20}    n/a rows - {description} (missing)")
        
        cursor.close()
        conn.close()
        
        return len(mlflow_tables), len(other_tables), len(all_table_names)
        
    except Exception as e:
        print(f"ERROR: {e}")
        return 0, 0, 0

if __name__ == "__main__":
    mlflow_count, other_count, total_count = count_mlflow_tables()
    print(f"\n" + "="*80)
    print(f"SUMMARY: {mlflow_count} MLflow tables, {other_count} custom tables, {total_count} total")
    print("="*80)