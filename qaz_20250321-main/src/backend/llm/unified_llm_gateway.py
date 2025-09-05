#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified LLM Gateway - Single Gateway with MLflow Integration

This module provides a unified LLM Gateway that integrates with MLflow
and can handle both local (ZLLM) and remote LLM calls with comprehensive
tracking and management.
"""

import logging
import time
import json
import os
import requests
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
from collections import defaultdict

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


@dataclass
class LLMCallMetrics:
    """Data structure for LLM call metrics"""
    timestamp: str
    batch_id: str
    agent_name: str
    model_name: str
    task_type: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: float
    response_time: float
    success: bool
    error: Optional[str] = None
    cache_hit: bool = False
    gateway_type: str = "unified"  # "local" or "remote"


@dataclass
class LLMResponse:
    """Standardized LLM response structure"""
    content: str
    input_tokens: int
    output_tokens: int
    cost: float
    model_name: str
    response_time: float
    success: bool
    error: Optional[str] = None
    gateway_type: str = "unified"


class UnifiedLLMGateway:
    """Unified LLM Gateway with MLflow integration"""
    
    def __init__(self, 
                 experiment_name: Optional[str] = None,
                 tracking_uri: Optional[str] = None,
                 enable_tracking: Optional[bool] = None,
                 local_gateway_url: Optional[str] = None,
                 remote_gateway_url: Optional[str] = None,
                 default_gateway: Optional[str] = None,
                 env_name: Optional[str] = None):
        
        # Import environment-aware configuration
        try:
            from ..core.mlflow_config import get_mlflow_config, setup_mlflow_environment
            self.mlflow_config = get_mlflow_config(env_name)
            
            # Setup environment if not already done
            if not os.getenv('MLFLOW_TRACKING_URI'):
                setup_mlflow_environment(env_name)
            
            # Use environment config as defaults, override with explicit parameters
            self.experiment_name = experiment_name or self.mlflow_config.get_experiment_name()
            self.enable_tracking = enable_tracking if enable_tracking is not None else self.mlflow_config.is_tracking_enabled()
            self.local_gateway_url = local_gateway_url or self.mlflow_config.get_gateway_config()['local_url']
            self.remote_gateway_url = remote_gateway_url or self.mlflow_config.get_gateway_config().get('remote_url')
            self.default_gateway = default_gateway or self.mlflow_config.get_gateway_config()['default_gateway']
            
            # Use environment tracking URI if not explicitly provided
            if tracking_uri:
                self.tracking_uri = tracking_uri
            else:
                self.tracking_uri = self.mlflow_config.get_tracking_uri()
                
        except ImportError:
            # Fallback to direct configuration if mlflow_config not available
            self.experiment_name = experiment_name or "unified-llm-gateway"
            self.enable_tracking = enable_tracking if enable_tracking is not None else True
            self.local_gateway_url = local_gateway_url or "http://localhost:11434"
            self.remote_gateway_url = remote_gateway_url
            self.default_gateway = default_gateway or "local"
            self.tracking_uri = tracking_uri
            self.mlflow_config = None
        
        self.experiment_name = experiment_name
        self.enable_tracking = enable_tracking
        self.local_gateway_url = local_gateway_url
        self.remote_gateway_url = remote_gateway_url
        self.default_gateway = default_gateway
        
        # Initialize MLflow using centralized configuration
        if enable_tracking:
            self._setup_mlflow()
        
        # Initialize metrics tracking
        self.metrics_tracker = LLMMetricsTracker()
        
        logger.info(f"Unified LLM Gateway initialized with experiment: {experiment_name}")
    
    def _setup_mlflow(self):
        """Setup MLflow experiment using centralized configuration"""
        try:
            # Use centralized MLflow configuration
            if hasattr(self, 'mlflow_config'):
                # Setup tracking URI from centralized config
                tracking_uri = self.mlflow_config.get_tracking_uri()
                if tracking_uri:
                    mlflow.set_tracking_uri(tracking_uri)
                
                # Set experiment from centralized config
                experiment_name = self.mlflow_config.get_experiment_name()
                mlflow.set_experiment(experiment_name)
                
                client = MlflowClient()
                experiment = client.get_experiment_by_name(experiment_name)
                
                if experiment is None:
                    logger.info(f"Creating new MLflow experiment: {experiment_name}")
                else:
                    logger.info(f"Using existing MLflow experiment: {experiment_name}")
            else:
                # Fallback to direct setup
                mlflow.set_experiment(self.experiment_name)
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
                 model_preference: Optional[str] = None,
                 gateway_preference: Optional[str] = None) -> LLMResponse:
        """Unified LLM call with MLflow tracking"""
        
        # Determine which gateway to use
        gateway_type = gateway_preference or self.default_gateway
        
        # Generate unique run name
        run_name = f"{agent_name}_{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            with mlflow.start_run(run_name=run_name, nested=True) as run:
                # Log input parameters
                self._log_input_parameters(agent_name, task_type, prompt, model_preference, gateway_type)
                
                # Make LLM call
                start_time = time.time()
                
                if gateway_type == "local":
                    response = self._call_local_llm(agent_name, task_type, prompt, model_preference)
                elif gateway_type == "remote":
                    response = self._call_remote_llm(agent_name, task_type, prompt, model_preference)
                else:
                    # Try local first, fallback to remote
                    try:
                        response = self._call_local_llm(agent_name, task_type, prompt, model_preference)
                    except Exception as e:
                        logger.warning(f"Local gateway failed, trying remote: {e}")
                        response = self._call_remote_llm(agent_name, task_type, prompt, model_preference)
                
                end_time = time.time()
                response.response_time = end_time - start_time
                response.gateway_type = gateway_type
                
                # Log response metrics
                self._log_response_metrics(response, end_time - start_time, gateway_type)
                
                # Log artifacts
                self._log_artifacts(prompt, response, run)
                
                # Log additional metadata
                self._log_metadata(agent_name, task_type, response, gateway_type)
                
                # Track metrics
                self._track_metrics(agent_name, task_type, response, gateway_type)
                
                logger.info(f"MLflow tracked LLM call: {run.info.run_id} via {gateway_type} gateway")
                
                return response
                
        except Exception as e:
            logger.error(f"MLflow tracking failed: {e}")
            # Fallback to regular call without tracking
            return self._call_local_llm(agent_name, task_type, prompt, model_preference)
    
    def _call_local_llm(self, agent_name: str, task_type: str, prompt: str, 
                        model_preference: Optional[str] = None) -> LLMResponse:
        """Call local ZLLM gateway"""
        try:
            payload = {
                "model": model_preference or "llama2",
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(
                f"{self.local_gateway_url}/api/generate", 
                json=payload, 
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            content = result.get('response', '')
            
            # Estimate tokens (rough calculation)
            input_tokens = len(prompt.split()) * 1.3  # Rough estimate
            output_tokens = len(content.split()) * 1.3
            
            return LLMResponse(
                content=content,
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens),
                cost=0.0,  # Local models are free
                model_name=model_preference or "llama2",
                response_time=0.0,  # Will be set by caller
                success=True,
                gateway_type="local"
            )
            
        except Exception as e:
            logger.error(f"Local LLM call failed: {e}")
            raise
    
    def _call_remote_llm(self, agent_name: str, task_type: str, prompt: str, 
                         model_preference: Optional[str] = None) -> LLMResponse:
        """Call remote LLM gateway"""
        if not self.remote_gateway_url:
            raise ValueError("Remote gateway URL not configured")
        
        try:
            payload = {
                "agent_name": agent_name,
                "task_type": task_type,
                "prompt": prompt,
                "model_preference": model_preference
            }
            
            response = requests.post(
                f"{self.remote_gateway_url}/api/llm/call",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            
            return LLMResponse(
                content=result.get('content', ''),
                input_tokens=result.get('input_tokens', 0),
                output_tokens=result.get('output_tokens', 0),
                cost=result.get('cost', 0.0),
                model_name=result.get('model_name', 'unknown'),
                response_time=0.0,  # Will be set by caller
                success=result.get('success', False),
                error=result.get('error'),
                gateway_type="remote"
            )
            
        except Exception as e:
            logger.error(f"Remote LLM call failed: {e}")
            raise
    
    def _log_input_parameters(self, agent_name: str, task_type: str, prompt: str, 
                             model_preference: Optional[str], gateway_type: str):
        """Log input parameters to MLflow"""
        mlflow.log_params({
            "agent_name": agent_name,
            "task_type": task_type,
            "prompt_length": len(prompt),
            "model_preference": model_preference or "default",
            "gateway_type": gateway_type
        })
    
    def _log_response_metrics(self, response: LLMResponse, response_time: float, gateway_type: str):
        """Log response metrics to MLflow"""
        mlflow.log_metrics({
            "response_time": response_time,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "total_tokens": response.input_tokens + response.output_tokens,
            "cost": response.cost,
            "success": response.success
        })
        
        mlflow.log_params({
            "model_used": response.model_name,
            "gateway_type": gateway_type
        })
    
    def _log_artifacts(self, prompt: str, response: LLMResponse, run):
        """Log artifacts to MLflow"""
        # Save prompt and response to files
        artifacts_dir = Path("mlflow_artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        
        prompt_file = artifacts_dir / f"prompt_{run.info.run_id}.txt"
        response_file = artifacts_dir / f"response_{run.info.run_id}.txt"
        
        with open(prompt_file, 'w') as f:
            f.write(prompt)
        
        with open(response_file, 'w') as f:
            f.write(response.content)
        
        mlflow.log_artifact(str(prompt_file))
        mlflow.log_artifact(str(response_file))
    
    def _log_metadata(self, agent_name: str, task_type: str, response: LLMResponse, gateway_type: str):
        """Log additional metadata to MLflow"""
        mlflow.log_dict({
            "agent_name": agent_name,
            "task_type": task_type,
            "gateway_type": gateway_type,
            "model_name": response.model_name,
            "success": response.success,
            "error": response.error
        }, "metadata.json")
    
    def _track_metrics(self, agent_name: str, task_type: str, response: LLMResponse, gateway_type: str):
        """Track metrics for analytics"""
        metrics = LLMCallMetrics(
            timestamp=datetime.now().isoformat(),
            batch_id=f"batch_{datetime.now().strftime('%Y%m%d_%H')}",
            agent_name=agent_name,
            model_name=response.model_name,
            task_type=task_type,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            total_tokens=response.input_tokens + response.output_tokens,
            cost=response.cost,
            response_time=response.response_time,
            success=response.success,
            error=response.error,
            gateway_type=gateway_type
        )
        
        self.metrics_tracker.track_call(metrics)


class LLMMetricsTracker:
    """Tracks all LLM call metrics"""
    
    def __init__(self, storage_path: str = "llm_metrics"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.current_batch_metrics = defaultdict(list)
        self.lock = threading.Lock()
    
    def track_call(self, metrics: LLMCallMetrics):
        """Track a single LLM call"""
        with self.lock:
            self.current_batch_metrics[metrics.batch_id].append(metrics)
            self.save_metrics(metrics)
            
            logger.info(f"Tracked LLM call: {metrics.agent_name} -> {metrics.model_name} "
                       f"({metrics.total_tokens} tokens, ${metrics.cost:.4f}) via {metrics.gateway_type}")
    
    def save_metrics(self, metrics: LLMCallMetrics):
        """Save metrics to file"""
        metrics_file = self.storage_path / f"metrics_{metrics.batch_id}.jsonl"
        
        with open(metrics_file, 'a') as f:
            f.write(json.dumps(asdict(metrics)) + '\n')
    
    def get_batch_metrics(self, batch_id: str) -> List[LLMCallMetrics]:
        """Get all metrics for a specific batch"""
        return self.current_batch_metrics.get(batch_id, [])
    
    def get_batch_summary(self, batch_id: str) -> Dict[str, Any]:
        """Get summary metrics for a batch"""
        metrics = self.get_batch_metrics(batch_id)
        
        if not metrics:
            return {}
        
        total_calls = len(metrics)
        total_tokens = sum(m.total_tokens for m in metrics)
        total_cost = sum(m.cost for m in metrics)
        success_rate = sum(1 for m in metrics if m.success) / total_calls
        avg_response_time = sum(m.response_time for m in metrics) / total_calls
        
        gateway_breakdown = defaultdict(lambda: {'calls': 0, 'tokens': 0, 'cost': 0})
        for m in metrics:
            gateway_breakdown[m.gateway_type]['calls'] += 1
            gateway_breakdown[m.gateway_type]['tokens'] += m.total_tokens
            gateway_breakdown[m.gateway_type]['cost'] += m.cost
        
        return {
            'batch_id': batch_id,
            'total_calls': total_calls,
            'total_tokens': total_tokens,
            'total_cost': total_cost,
            'success_rate': success_rate,
            'avg_response_time': avg_response_time,
            'gateway_breakdown': dict(gateway_breakdown)
        }
