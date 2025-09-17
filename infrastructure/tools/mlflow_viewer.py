#!/usr/bin/env python3
"""
Show the last 5 MLflow records from the PostgreSQL backend
"""

import yaml
import mlflow
import mlflow.tracking
from pathlib import Path
from datetime import datetime

def show_last_5_mlflow_records():
    """Show the last 5 MLflow experiment runs with all details"""
    print("LAST 5 MLFLOW RECORDS FROM POSTGRESQL")
    print("=" * 60)
    
    # Load credentials and connect
    settings_path = Path("C:/Users/marti/AI-Scoring/tidyllm/admin/settings.yaml")
    with open(settings_path, 'r') as f:
        config = yaml.safe_load(f)
    
    mlflow_uri = config['services']['mlflow']['backend_store_uri']
    mlflow.set_tracking_uri(mlflow_uri)
    
    print(f"MLflow Backend: PostgreSQL on AWS RDS")
    print(f"Database: vectorqa")
    print(f"Host: {config['credentials']['postgresql']['host']}")
    print()
    
    client = mlflow.tracking.MlflowClient()
    
    # Get ALL runs across ALL experiments, sorted by start time
    print("SEARCHING ALL EXPERIMENTS FOR RECENT RUNS...")
    print("-" * 45)
    
    # First get all experiments
    all_experiments = client.search_experiments()
    print(f"Total experiments in database: {len(all_experiments)}")
    
    # Get all runs from all experiments
    all_runs = []
    for exp in all_experiments:
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=5  # Get top 5 from each experiment
        )
        all_runs.extend(runs)
    
    # Sort all runs by start time and get the last 5
    all_runs.sort(key=lambda x: x.info.start_time, reverse=True)
    last_5_runs = all_runs[:5]
    
    print(f"Total runs found: {len(all_runs)}")
    print(f"Showing last 5 runs:\n")
    
    # Show detailed info for each of the last 5 runs
    for i, run in enumerate(last_5_runs, 1):
        print(f"{'='*60}")
        print(f"RECORD #{i} - MLflow Run")
        print(f"{'='*60}")
        
        # Get experiment info
        experiment = client.get_experiment(run.info.experiment_id)
        
        print("RUN IDENTIFICATION:")
        print(f"  Run ID: {run.info.run_id}")
        print(f"  Run Name: {run.info.run_name or 'Unnamed'}")
        print(f"  Experiment: {experiment.name}")
        print(f"  Experiment ID: {experiment.experiment_id}")
        print(f"  Status: {run.info.status}")
        print(f"  Start Time: {datetime.fromtimestamp(run.info.start_time/1000)}")
        if run.info.end_time:
            duration = (run.info.end_time - run.info.start_time) / 1000
            print(f"  Duration: {duration:.2f} seconds")
        print()
        
        # Show parameters
        params = run.data.params
        if params:
            print(f"PARAMETERS ({len(params)} total):")
            # Show key parameters
            key_params = ['model', 'boss_template_version', 'dspy_after_optimization', 
                         'boss_template_a', 'boss_template_b', 'ai_model']
            
            for key in key_params:
                if key in params:
                    print(f"  {key}: {params[key]}")
            
            # Show any other interesting parameters
            other_params = {k: v for k, v in params.items() 
                          if k not in key_params and len(v) < 50}
            if other_params:
                for key, value in list(other_params.items())[:5]:
                    print(f"  {key}: {value}")
        else:
            print("PARAMETERS: None")
        print()
        
        # Show metrics
        metrics = run.data.metrics
        if metrics:
            print(f"METRICS ({len(metrics)} total):")
            
            # Token metrics
            token_metrics = {k: v for k, v in metrics.items() if 'token' in k.lower()}
            if token_metrics:
                print("  Token Counts:")
                for key, value in sorted(token_metrics.items()):
                    print(f"    {key}: {value:.0f}")
            
            # Cost metrics
            cost_metrics = {k: v for k, v in metrics.items() if 'cost' in k.lower()}
            if cost_metrics:
                print("  Costs:")
                for key, value in sorted(cost_metrics.items()):
                    if 'percent' not in key:
                        print(f"    {key}: ${value:.4f}")
                    else:
                        print(f"    {key}: {value:.1f}%")
            
            # Accuracy/Quality metrics
            quality_metrics = {k: v for k, v in metrics.items() 
                             if any(x in k.lower() for x in ['accuracy', 'confidence', 'satisfaction'])}
            if quality_metrics:
                print("  Quality Metrics:")
                for key, value in sorted(quality_metrics.items()):
                    print(f"    {key}: {value:.2f}")
            
            # Other important metrics
            other_metrics = {k: v for k, v in metrics.items() 
                           if k not in token_metrics and k not in cost_metrics and k not in quality_metrics}
            if other_metrics:
                print("  Other Metrics:")
                for key, value in list(other_metrics.items())[:5]:
                    print(f"    {key}: {value}")
        else:
            print("METRICS: None")
        
        print()
    
    # Summary statistics
    print("=" * 60)
    print("SUMMARY STATISTICS:")
    print("=" * 60)
    
    # Count runs by experiment
    exp_counts = {}
    for run in all_runs:
        exp_id = run.info.experiment_id
        exp = client.get_experiment(exp_id)
        exp_name = exp.name
        exp_counts[exp_name] = exp_counts.get(exp_name, 0) + 1
    
    print("Runs by Experiment:")
    for exp_name, count in sorted(exp_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {exp_name}: {count} runs")
    
    print(f"\nTotal Experiments: {len(all_experiments)}")
    print(f"Total Runs: {len(all_runs)}")
    print(f"Database Location: AWS RDS PostgreSQL")
    print(f"Connection: {mlflow_uri[:50]}...")

if __name__ == "__main__":
    show_last_5_mlflow_records()