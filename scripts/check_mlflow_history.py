#!/usr/bin/env python3
"""
MLflow History Log Checker
Check MLflow tracking data stored in PostgreSQL
"""
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from datetime import datetime

def check_mlflow_history():
    password = "REMOVED_PASSWORD"
    
    try:
        print("="*80)
        print("MLFLOW HISTORY LOG CHECKER")
        print("="*80)
        
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
        
        # 1. Check experiments
        print("\n" + "-"*60)
        print("EXPERIMENTS")
        print("-"*60)
        cursor.execute("""
            SELECT experiment_id, name, lifecycle_stage, creation_time, last_update_time
            FROM experiments 
            ORDER BY creation_time DESC 
            LIMIT 10;
        """)
        experiments = cursor.fetchall()
        
        if experiments:
            for exp in experiments:
                created = datetime.fromtimestamp(exp['creation_time']/1000) if exp['creation_time'] else 'Unknown'
                updated = datetime.fromtimestamp(exp['last_update_time']/1000) if exp['last_update_time'] else 'Unknown'
                print(f"ID: {exp['experiment_id']} | Name: {exp['name']} | Stage: {exp['lifecycle_stage']}")
                print(f"   Created: {created} | Updated: {updated}")
        else:
            print("No experiments found")
        
        # 2. Check runs
        print("\n" + "-"*60)
        print("RECENT RUNS")
        print("-"*60)
        cursor.execute("""
            SELECT run_uuid, experiment_id, status, start_time, end_time, source_name
            FROM runs 
            ORDER BY start_time DESC 
            LIMIT 10;
        """)
        runs = cursor.fetchall()
        
        if runs:
            for run in runs:
                start = datetime.fromtimestamp(run['start_time']/1000) if run['start_time'] else 'Unknown'
                end = datetime.fromtimestamp(run['end_time']/1000) if run['end_time'] else 'Running'
                print(f"Run: {run['run_uuid'][:8]}... | Exp: {run['experiment_id']} | Status: {run['status']}")
                print(f"   Started: {start} | Ended: {end} | Source: {run['source_name']}")
        else:
            print("No runs found")
        
        # 3. Check metrics
        print("\n" + "-"*60)
        print("METRICS")
        print("-"*60)
        cursor.execute("""
            SELECT run_uuid, key, value, timestamp, step
            FROM metrics 
            ORDER BY timestamp DESC 
            LIMIT 15;
        """)
        metrics = cursor.fetchall()
        
        if metrics:
            for metric in metrics:
                timestamp = datetime.fromtimestamp(metric['timestamp']/1000) if metric['timestamp'] else 'Unknown'
                print(f"Run: {metric['run_uuid'][:8]}... | {metric['key']}: {metric['value']} | Step: {metric['step']} | Time: {timestamp}")
        else:
            print("No metrics found")
        
        # 4. Check latest metrics summary
        print("\n" + "-"*60)
        print("LATEST METRICS SUMMARY")
        print("-"*60)
        cursor.execute("""
            SELECT run_uuid, key, value, timestamp
            FROM latest_metrics 
            ORDER BY timestamp DESC 
            LIMIT 15;
        """)
        latest_metrics = cursor.fetchall()
        
        if latest_metrics:
            for metric in latest_metrics:
                timestamp = datetime.fromtimestamp(metric['timestamp']/1000) if metric['timestamp'] else 'Unknown'
                print(f"Run: {metric['run_uuid'][:8]}... | {metric['key']}: {metric['value']} | Time: {timestamp}")
        else:
            print("No latest metrics found")
        
        # 5. Check for TidyLLM/Gateway specific data
        print("\n" + "-"*60)
        print("TIDYLLM/GATEWAY ACTIVITY")
        print("-"*60)
        
        # Look for runs with TidyLLM or Gateway sources
        cursor.execute("""
            SELECT run_uuid, experiment_id, source_name, source_version, start_time
            FROM runs 
            WHERE source_name ILIKE '%tidyllm%' 
               OR source_name ILIKE '%gateway%'
               OR source_name ILIKE '%dspy%'
            ORDER BY start_time DESC 
            LIMIT 10;
        """)
        tidyllm_runs = cursor.fetchall()
        
        if tidyllm_runs:
            print("TidyLLM/Gateway runs found:")
            for run in tidyllm_runs:
                start = datetime.fromtimestamp(run['start_time']/1000) if run['start_time'] else 'Unknown'
                print(f"  Run: {run['run_uuid'][:8]}... | Source: {run['source_name']} | Version: {run['source_version']} | Started: {start}")
        else:
            print("No TidyLLM/Gateway specific runs found yet")
        
        # 6. Database size and activity summary
        print("\n" + "-"*60)
        print("DATABASE ACTIVITY SUMMARY")
        print("-"*60)
        
        # Count totals
        cursor.execute("SELECT COUNT(*) FROM experiments;")
        total_experiments = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(*) FROM runs;")
        total_runs = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(*) FROM metrics;")
        total_metrics = cursor.fetchone()['count']
        
        # Get date range
        cursor.execute("""
            SELECT 
                MIN(start_time) as earliest_run,
                MAX(start_time) as latest_run
            FROM runs 
            WHERE start_time IS NOT NULL;
        """)
        date_range = cursor.fetchone()
        
        print(f"Total Experiments: {total_experiments}")
        print(f"Total Runs: {total_runs}")
        print(f"Total Metrics: {total_metrics}")
        
        if date_range and date_range['earliest_run'] and date_range['latest_run']:
            earliest = datetime.fromtimestamp(date_range['earliest_run']/1000)
            latest = datetime.fromtimestamp(date_range['latest_run']/1000)
            print(f"Activity Range: {earliest} to {latest}")
        else:
            print("No run activity timestamps found")
        
        # 7. Check for any error or system logs
        print("\n" + "-"*60)
        print("SYSTEM TABLES")
        print("-"*60)
        
        # Check if there are any custom tables that might contain logs
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
              AND (table_name ILIKE '%log%' 
                OR table_name ILIKE '%error%'
                OR table_name ILIKE '%event%'
                OR table_name ILIKE '%history%')
            ORDER BY table_name;
        """)
        log_tables = cursor.fetchall()
        
        if log_tables:
            print("Log/Event tables found:")
            for table in log_tables:
                table_name = table['table_name']
                cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
                count = cursor.fetchone()['count']
                print(f"  {table_name}: {count} records")
                
                # Show sample from error_patterns if it exists
                if 'error' in table_name and count > 0:
                    cursor.execute(f"SELECT * FROM {table_name} LIMIT 3;")
                    samples = cursor.fetchall()
                    for sample in samples:
                        print(f"    Sample: {dict(sample)}")
        else:
            print("No specific log/event tables found")
        
        cursor.close()
        conn.close()
        
        print("\n" + "="*80)
        print("MLFLOW HISTORY CHECK COMPLETE")
        print("="*80)
        
    except Exception as e:
        print(f"ERROR: Failed to check MLflow history: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_mlflow_history()