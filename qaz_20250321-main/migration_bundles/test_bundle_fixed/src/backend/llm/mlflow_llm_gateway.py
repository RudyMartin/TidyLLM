#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLflow-Enhanced LLM Gateway

This module extends the LLM Gateway with MLflow experiment tracking
for comprehensive LLM performance analysis and optimization.
"""

import logging
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import hashlib

import mlflow
from mlflow.tracking import MlflowClient

from .llm_gateway import LLMGateway, LLMResponse, LLMCallMetrics

logger = logging.getLogger(__name__)


class MLflowLLMGateway(LLMGateway):
    """MLflow-enhanced LLM Gateway with experiment tracking"""
    
    def __init__(self, experiment_name: str = "llm-gateway", 
                 tracking_uri: Optional[str] = None,
                 enable_tracking: bool = True):
        super().__init__()
        
        self.experiment_name = experiment_name
        self.enable_tracking = enable_tracking
        
        # Initialize MLflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        if enable_tracking:
            self._setup_mlflow()
        
        logger.info(f"MLflow LLM Gateway initialized with experiment: {experiment_name}")
    
    def _setup_mlflow(self):
        """Setup MLflow experiment"""
        try:
            # Set experiment
            mlflow.set_experiment(self.experiment_name)
            
            # Get or create experiment
            client = MlflowClient()
            experiment = client.get_experiment_by_name(self.experiment_name)
            
            if experiment is None:
                logger.info(f"Creating new MLflow experiment: {self.experiment_name}")
            else:
                logger.info(f"Using existing MLflow experiment: {self.experiment_name}")
                
        except Exception as e:
            logger.warning(f"Failed to setup MLflow: {e}")
            self.enable_tracking = False
    
    def call_llm(self, agent_name: str, task_type: str, prompt: str, 
                 model_preference: Optional[str] = None) -> LLMResponse:
        """Enhanced LLM call with MLflow tracking"""
        
        if not self.enable_tracking:
            return super().call_llm(agent_name, task_type, prompt, model_preference)
        
        # Generate unique run name
        run_name = f"{agent_name}_{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            with mlflow.start_run(run_name=run_name, nested=True) as run:
                # Log input parameters
                self._log_input_parameters(agent_name, task_type, prompt, model_preference)
                
                # Make LLM call
                start_time = time.time()
                response = super().call_llm(agent_name, task_type, prompt, model_preference)
                end_time = time.time()
                
                # Log response metrics
                self._log_response_metrics(response, end_time - start_time)
                
                # Log artifacts
                self._log_artifacts(prompt, response, run)
                
                # Log additional metadata
                self._log_metadata(agent_name, task_type, response)
                
                logger.info(f"MLflow tracked LLM call: {run.info.run_id}")
                
                return response
                
        except Exception as e:
            logger.error(f"MLflow tracking failed: {e}")
            # Fallback to regular call without tracking
            return super().call_llm(agent_name, task_type, prompt, model_preference)
    
    def _log_input_parameters(self, agent_name: str, task_type: str, 
                            prompt: str, model_preference: Optional[str]):
        """Log input parameters to MLflow"""
        try:
            # Basic parameters
            params = {
                "agent_name": agent_name,
                "task_type": task_type,
                "prompt_length": len(prompt),
                "prompt_word_count": len(prompt.split()),
                "model_preference": model_preference or "auto"
            }
            
            # Add prompt complexity metrics
            params.update(self._calculate_prompt_complexity(prompt))
            
            mlflow.log_params(params)
            
        except Exception as e:
            logger.warning(f"Failed to log input parameters: {e}")
    
    def _log_response_metrics(self, response: LLMResponse, actual_response_time: float):
        """Log response metrics to MLflow"""
        try:
            metrics = {
                "response_time": response.response_time,
                "actual_response_time": actual_response_time,
                "cost": response.cost,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "total_tokens": response.input_tokens + response.output_tokens,
                "success": 1.0 if response.success else 0.0
            }
            
            # Calculate derived metrics
            if response.input_tokens > 0:
                metrics["cost_per_input_token"] = response.cost / response.input_tokens
            if response.output_tokens > 0:
                metrics["cost_per_output_token"] = response.cost / response.output_tokens
            if (response.input_tokens + response.output_tokens) > 0:
                metrics["cost_per_total_token"] = response.cost / (response.input_tokens + response.output_tokens)
            
            # Log only numerical metrics (model_name is already logged as a tag)
            mlflow.log_metrics(metrics)
            
        except Exception as e:
            logger.warning(f"Failed to log response metrics: {e}")
    
    def _log_artifacts(self, prompt: str, response: LLMResponse, run):
        """Log artifacts to MLflow"""
        try:
            # Create temporary files for artifacts
            temp_dir = Path("temp_mlflow_artifacts")
            temp_dir.mkdir(exist_ok=True)
            
            # Log prompt
            prompt_file = temp_dir / "prompt.txt"
            with open(prompt_file, 'w') as f:
                f.write(prompt)
            mlflow.log_artifact(str(prompt_file), "prompt")
            
            # Log response
            response_file = temp_dir / "response.txt"
            with open(response_file, 'w') as f:
                f.write(response.content)
            mlflow.log_artifact(str(response_file), "response")
            
            # Log response metadata
            metadata_file = temp_dir / "response_metadata.json"
            metadata = {
                "model_name": response.model_name,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost": response.cost,
                "response_time": response.response_time,
                "success": response.success,
                "error": response.error
            }
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            mlflow.log_artifact(str(metadata_file), "metadata")
            
            # Clean up temp files
            for file in temp_dir.glob("*"):
                file.unlink()
            temp_dir.rmdir()
            
        except Exception as e:
            logger.warning(f"Failed to log artifacts: {e}")
    
    def _log_metadata(self, agent_name: str, task_type: str, response: LLMResponse):
        """Log additional metadata"""
        try:
            # Add tags for easy filtering
            mlflow.set_tags({
                "agent_name": agent_name,
                "task_type": task_type,
                "model_name": response.model_name,
                "success": str(response.success),
                "tracking_version": "1.0"
            })
            
        except Exception as e:
            logger.warning(f"Failed to log metadata: {e}")
    
    def _calculate_prompt_complexity(self, prompt: str) -> Dict[str, Any]:
        """Calculate prompt complexity metrics"""
        try:
            words = prompt.split()
            sentences = prompt.split('.')
            
            return {
                "prompt_complexity_score": len(words) * len(sentences) / 100,  # Simple heuristic
                "avg_words_per_sentence": len(words) / len(sentences) if len(sentences) > 1 else len(words),
                "has_code": 1.0 if any(char in prompt for char in ['{', '}', '(', ')', ';']) else 0.0,
                "has_numbers": 1.0 if any(char.isdigit() for char in prompt) else 0.0,
                "has_special_chars": 1.0 if any(char in prompt for char in ['@', '#', '$', '%', '&']) else 0.0
            }
        except Exception:
            return {}
    
    def get_experiment_performance(self, filter_string: Optional[str] = None) -> Dict[str, Any]:
        """Get performance analysis from MLflow experiments"""
        try:
            if not self.enable_tracking:
                return {"error": "MLflow tracking not enabled"}
            
            # Search runs
            runs = mlflow.search_runs(
                experiment_names=[self.experiment_name],
                filter_string=filter_string
            )
            
            if runs.empty:
                return {"message": "No runs found"}
            
            # Calculate performance metrics
            performance = {
                "total_runs": len(runs),
                "success_rate": runs["metrics.success"].mean() if "metrics.success" in runs.columns else 0,
                "avg_response_time": runs["metrics.response_time"].mean() if "metrics.response_time" in runs.columns else 0,
                "avg_cost": runs["metrics.cost"].mean() if "metrics.cost" in runs.columns else 0,
                "total_cost": runs["metrics.cost"].sum() if "metrics.cost" in runs.columns else 0,
                "avg_tokens": runs["metrics.total_tokens"].mean() if "metrics.total_tokens" in runs.columns else 0
            }
            
            # Model breakdown (only if metrics columns exist)
            if "params.model_name" in runs.columns and any(col.startswith("metrics.") for col in runs.columns):
                try:
                    available_metrics = [col for col in runs.columns if col.startswith("metrics.")]
                    agg_dict = {}
                    
                    if "metrics.cost" in available_metrics:
                        agg_dict["metrics.cost"] = ["mean", "sum", "count"]
                    if "metrics.response_time" in available_metrics:
                        agg_dict["metrics.response_time"] = "mean"
                    if "metrics.success" in available_metrics:
                        agg_dict["metrics.success"] = "mean"
                    
                    if agg_dict:
                        model_breakdown = runs.groupby("params.model_name").agg(agg_dict).round(4)
                        performance["model_breakdown"] = model_breakdown.to_dict()
                except Exception as e:
                    logger.warning(f"Failed to create model breakdown: {e}")
            
            # Task breakdown (only if metrics columns exist)
            if "params.task_type" in runs.columns and any(col.startswith("metrics.") for col in runs.columns):
                try:
                    available_metrics = [col for col in runs.columns if col.startswith("metrics.")]
                    agg_dict = {}
                    
                    if "metrics.cost" in available_metrics:
                        agg_dict["metrics.cost"] = ["mean", "sum", "count"]
                    if "metrics.response_time" in available_metrics:
                        agg_dict["metrics.response_time"] = "mean"
                    if "metrics.success" in available_metrics:
                        agg_dict["metrics.success"] = "mean"
                    
                    if agg_dict:
                        task_breakdown = runs.groupby("params.task_type").agg(agg_dict).round(4)
                        performance["task_breakdown"] = task_breakdown.to_dict()
                except Exception as e:
                    logger.warning(f"Failed to create task breakdown: {e}")
            
            return performance
            
        except Exception as e:
            logger.error(f"Failed to get experiment performance: {e}")
            return {"error": str(e)}
    
    def get_cost_optimization_recommendations(self) -> Dict[str, Any]:
        """Get cost optimization recommendations based on historical data"""
        try:
            if not self.enable_tracking:
                return {"error": "MLflow tracking not enabled"}
            
            # Get recent runs
            runs = mlflow.search_runs(
                experiment_names=[self.experiment_name],
                max_results=1000
            )
            
            if runs.empty:
                return {"message": "No historical data available"}
            
            recommendations = {
                "best_cost_performance": {},
                "model_recommendations": {},
                "task_optimizations": {}
            }
            
            # Find best cost-performance models (only if metrics columns exist)
            if "params.model_name" in runs.columns and "metrics.cost" in runs.columns:
                try:
                    available_metrics = [col for col in runs.columns if col.startswith("metrics.")]
                    agg_dict = {}
                    
                    if "metrics.cost" in available_metrics:
                        agg_dict["metrics.cost"] = "mean"
                    if "metrics.response_time" in available_metrics:
                        agg_dict["metrics.response_time"] = "mean"
                    if "metrics.success" in available_metrics:
                        agg_dict["metrics.success"] = "mean"
                    
                    if agg_dict:
                        model_performance = runs.groupby("params.model_name").agg(agg_dict)
                        
                        # Best cost-effective model
                        if "metrics.cost" in model_performance.columns:
                            best_cost_model = model_performance["metrics.cost"].idxmin()
                            recommendations["best_cost_performance"]["model"] = best_cost_model
                            recommendations["best_cost_performance"]["avg_cost"] = model_performance.loc[best_cost_model, "metrics.cost"]
                except Exception as e:
                    logger.warning(f"Failed to create cost performance analysis: {e}")
            
            # Model recommendations by task (only if metrics columns exist)
            if "params.task_type" in runs.columns and any(col.startswith("metrics.") for col in runs.columns):
                try:
                    for task_type in runs["params.task_type"].unique():
                        task_runs = runs[runs["params.task_type"] == task_type]
                        if not task_runs.empty and "params.model_name" in task_runs.columns:
                            available_metrics = [col for col in task_runs.columns if col.startswith("metrics.")]
                            agg_dict = {}
                            
                            if "metrics.cost" in available_metrics:
                                agg_dict["metrics.cost"] = "mean"
                            if "metrics.response_time" in available_metrics:
                                agg_dict["metrics.response_time"] = "mean"
                            if "metrics.success" in available_metrics:
                                agg_dict["metrics.success"] = "mean"
                            
                            if agg_dict:
                                task_performance = task_runs.groupby("params.model_name").agg(agg_dict)
                                
                                if not task_performance.empty and "metrics.cost" in task_performance.columns:
                                    best_model = task_performance["metrics.cost"].idxmin()
                                    recommendations["model_recommendations"][task_type] = {
                                        "recommended_model": best_model,
                                        "avg_cost": task_performance.loc[best_model, "metrics.cost"]
                                    }
                                    
                                    if "metrics.response_time" in task_performance.columns:
                                        recommendations["model_recommendations"][task_type]["avg_response_time"] = task_performance.loc[best_model, "metrics.response_time"]
                except Exception as e:
                    logger.warning(f"Failed to create task recommendations: {e}")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to get optimization recommendations: {e}")
            return {"error": str(e)}
    
    def export_experiment_data(self, output_path: str = "mlflow_export") -> str:
        """Export experiment data for external analysis"""
        try:
            if not self.enable_tracking:
                return "MLflow tracking not enabled"
            
            output_dir = Path(output_path)
            output_dir.mkdir(exist_ok=True)
            
            # Get all runs
            runs = mlflow.search_runs(
                experiment_names=[self.experiment_name]
            )
            
            # Export to JSON
            export_file = output_dir / f"llm_experiment_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            runs.to_json(export_file, orient='records', indent=2)
            
            logger.info(f"Exported experiment data to: {export_file}")
            return str(export_file)
            
        except Exception as e:
            logger.error(f"Failed to export experiment data: {e}")
            return f"Export failed: {str(e)}"


class MLflowExperimentAnalyzer:
    """Utility class for analyzing MLflow LLM experiments"""
    
    def __init__(self, experiment_name: str = "llm-gateway"):
        self.experiment_name = experiment_name
        self.client = MlflowClient()
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get comprehensive experiment summary"""
        try:
            # Get experiment
            experiment = self.client.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                return {"error": f"Experiment {self.experiment_name} not found"}
            
            # Get runs
            runs = mlflow.search_runs(
                experiment_names=[self.experiment_name]
            )
            
            if runs.empty:
                return {"message": "No runs found in experiment"}
            
            summary = {
                "experiment_name": self.experiment_name,
                "experiment_id": experiment.experiment_id,
                "total_runs": len(runs),
                "date_range": {
                    "earliest": runs["start_time"].min(),
                    "latest": runs["start_time"].max()
                }
            }
            
            # Performance metrics
            if "metrics.cost" in runs.columns:
                summary["cost_analysis"] = {
                    "total_cost": runs["metrics.cost"].sum(),
                    "avg_cost_per_run": runs["metrics.cost"].mean(),
                    "min_cost": runs["metrics.cost"].min(),
                    "max_cost": runs["metrics.cost"].max()
                }
            
            if "metrics.response_time" in runs.columns:
                summary["performance_analysis"] = {
                    "avg_response_time": runs["metrics.response_time"].mean(),
                    "min_response_time": runs["metrics.response_time"].min(),
                    "max_response_time": runs["metrics.response_time"].max()
                }
            
            return summary
            
        except Exception as e:
            return {"error": str(e)}
    
    def generate_performance_report(self, output_path: str = "mlflow_performance_report.md") -> str:
        """Generate a markdown performance report"""
        try:
            summary = self.get_experiment_summary()
            
            if "error" in summary:
                return f"Failed to generate report: {summary['error']}"
            
            report_content = f"""# LLM Gateway Performance Report

## Experiment Summary
- **Experiment Name**: {summary.get('experiment_name', 'N/A')}
- **Total Runs**: {summary.get('total_runs', 0)}
- **Date Range**: {summary.get('date_range', {}).get('earliest', 'N/A')} to {summary.get('date_range', {}).get('latest', 'N/A')}

## Cost Analysis
- **Total Cost**: ${summary.get('cost_analysis', {}).get('total_cost', 0):.4f}
- **Average Cost per Run**: ${summary.get('cost_analysis', {}).get('avg_cost_per_run', 0):.4f}
- **Cost Range**: ${summary.get('cost_analysis', {}).get('min_cost', 0):.4f} - ${summary.get('cost_analysis', {}).get('max_cost', 0):.4f}

## Performance Analysis
- **Average Response Time**: {summary.get('performance_analysis', {}).get('avg_response_time', 0):.2f}s
- **Response Time Range**: {summary.get('performance_analysis', {}).get('min_response_time', 0):.2f}s - {summary.get('performance_analysis', {}).get('max_response_time', 0):.2f}s

## Recommendations
1. Monitor cost trends and set up alerts for budget overruns
2. Analyze response time patterns to optimize model selection
3. Review failed runs to identify common issues
4. Consider implementing automated model selection based on historical performance

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            with open(output_path, 'w') as f:
                f.write(report_content)
            
            return output_path
            
        except Exception as e:
            return f"Failed to generate report: {str(e)}"
