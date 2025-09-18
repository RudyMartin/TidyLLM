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

def export_project_mlflow_csv(project_name: str, output_path: str = None):
    """Export MLflow records for a specific project to CSV for verification"""
    import csv
    import pandas as pd
    import json

    print(f"EXPORTING MLFLOW DATA FOR PROJECT: {project_name}")
    print("=" * 60)

    # Load project configuration for dynamic step mapping
    step_mapping = {}
    try:
        project_config_path = Path(f"tidyllm/workflows/projects/{project_name}/project_config.json")
        if project_config_path.exists():
            with open(project_config_path, 'r', encoding='utf-8') as f:
                project_config = json.load(f)

            print(f"✓ Loaded project config: {project_config_path}")

            # Create step mapping from config
            for step in project_config.get('steps', []):
                step_id = step.get('step_id', '')
                step_number = step.get('step_number', 'auto')
                step_name = step.get('step_name', step_id)

                step_mapping[step_id] = {
                    'number': step_number,
                    'name': step_name,
                    'type': step.get('step_type', 'unknown')
                }

            print(f"✓ Mapped {len(step_mapping)} steps from config")
            for step_id, info in step_mapping.items():
                print(f"  - {step_id}: {info['number']} ({info['name']})")
        else:
            print(f"⚠ No project config found at {project_config_path}")
            print("  Using fallback step detection")
    except Exception as e:
        print(f"⚠ Error loading project config: {e}")
        print("  Using fallback step detection")

    # Load credentials and connect
    settings_path = Path("C:/Users/marti/AI-Scoring/tidyllm/admin/settings.yaml")
    with open(settings_path, 'r') as f:
        config = yaml.safe_load(f)

    mlflow_uri = config['services']['mlflow']['backend_store_uri']
    mlflow.set_tracking_uri(mlflow_uri)

    client = mlflow.tracking.MlflowClient()

    # Get all experiments and runs
    all_experiments = client.search_experiments()
    all_runs = []

    for exp in all_experiments:
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=1000  # Get more runs for project analysis
        )
        all_runs.extend(runs)

    # Filter runs for specific project (check user_id, audit_reason, parameters)
    project_runs = []
    for run in all_runs:
        params = run.data.params
        metrics = run.data.metrics

        # Check if run belongs to this project
        is_project_run = False

        # Check parameters for project indicators
        if any(project_name.lower() in str(v).lower() for v in params.values()):
            is_project_run = True

        # Check for QAQC-specific audit reasons
        if project_name == "alex_qaqc":
            qaqc_indicators = ['qaqc', 'metadata_extraction', 'analysis_steps',
                             'results_aggregation', 'recording_questions']
            if any(indicator in str(params.get('audit_reason', '')).lower()
                   for indicator in qaqc_indicators):
                is_project_run = True

        if is_project_run:
            project_runs.append(run)

    print(f"Found {len(project_runs)} runs for project {project_name}")

    if not project_runs:
        print("No runs found for this project. Create CSV template for future runs.")
        # Create empty template
        csv_data = [{
            'run_id': 'template',
            'timestamp': datetime.now().isoformat(),
            'step_name': 'sample_step',
            'step_number': '1.0',
            'step_type': 'template',
            'user_id': 'sample_user',
            'audit_reason': f'{project_name}_template',
            'processing_time_ms': 0,
            'success': True,
            'confidence_score': 0.0,
            'model': 'claude-3-sonnet',
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0
        }]
    else:
        # Extract data for CSV
        csv_data = []
        for run in project_runs:
            params = run.data.params
            metrics = run.data.metrics

            # Extract step information from audit_reason using dynamic mapping
            audit_reason = params.get('audit_reason', '')
            step_name = 'unknown'
            step_number = 'auto'
            step_type = 'unknown'

            # Try to match against configured steps first
            step_matched = False
            if step_mapping:
                for step_id, step_info in step_mapping.items():
                    if step_id in audit_reason:
                        step_name = step_id
                        step_number = step_info['number']
                        step_type = step_info['type']
                        step_matched = True
                        break

            # Fallback to hardcoded detection if no mapping found
            if not step_matched:
                if 'metadata_extraction' in audit_reason:
                    step_name = 'metadata_extraction'
                    step_number = 1
                elif 'analysis_steps' in audit_reason:
                    step_name = 'analysis_steps'
                    step_number = 2
                elif 'results_aggregation' in audit_reason:
                    step_name = 'results_aggregation'
                    step_number = 3
                elif 'recording_questions' in audit_reason:
                    step_name = 'recording_questions'
                    step_number = 4

            csv_row = {
                'run_id': run.info.run_id,
                'timestamp': datetime.fromtimestamp(run.info.start_time/1000).isoformat(),
                'step_name': step_name,
                'step_number': step_number,
                'step_type': step_type,
                'user_id': params.get('user_id', 'unknown'),
                'audit_reason': audit_reason,
                'processing_time_ms': metrics.get('processing_time_ms', 0),
                'success': metrics.get('success', 1.0) == 1.0,
                'confidence_score': metrics.get('confidence_score', 0.0),
                'model': params.get('model', 'unknown'),
                'input_tokens': metrics.get('input_tokens', 0),
                'output_tokens': metrics.get('output_tokens', 0),
                'total_tokens': metrics.get('total_tokens', 0),
                'experiment_name': client.get_experiment(run.info.experiment_id).name
            }
            csv_data.append(csv_row)

    def _sort_by_step_number(row):
        """Sort by step_number, handling hierarchical numbers like 1.1.1"""
        step_num = row['step_number']
        if isinstance(step_num, str) and step_num != 'auto':
            # Convert "1.2.3" to [1, 2, 3] for proper sorting
            try:
                return [int(x) for x in step_num.split('.')]
            except:
                return [999, 0]  # Put unparseable numbers at end
        elif isinstance(step_num, (int, float)):
            return [int(step_num), 0]
        else:
            # Fallback to timestamp for 'auto' or unknown
            return [999, 1]

    # Sort by step_number first, then by timestamp
    csv_data.sort(key=lambda x: (_sort_by_step_number(x), x['timestamp']))

    # Set output path
    if output_path is None:
        output_path = f"mlflow_{project_name}.csv"

    # Write CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        if csv_data:
            fieldnames = csv_data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)

    print(f"✓ Exported {len(csv_data)} records to: {output_path}")
    print("\nCSV Columns:")
    if csv_data:
        for col in csv_data[0].keys():
            print(f"  - {col}")

    # Show sample data
    print(f"\nSample Records (first 3):")
    for i, row in enumerate(csv_data[:3]):
        print(f"\nRecord {i+1}:")
        print(f"  Step: {row['step_name']} (#{row['step_number']})")
        print(f"  Time: {row['timestamp']}")
        print(f"  User: {row['user_id']}")
        print(f"  Audit: {row['audit_reason']}")
        print(f"  Success: {row['success']}")
        print(f"  Tokens: {row['total_tokens']}")

    return output_path

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "export":
        project = sys.argv[2] if len(sys.argv) > 2 else "alex_qaqc"
        export_project_mlflow_csv(project)
    else:
        show_last_5_mlflow_records()