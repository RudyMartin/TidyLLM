#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Gateway - Centralized LLM Management System

This module provides a centralized gateway for all LLM interactions,
including metrics tracking, cost management, model routing, and caching.
"""

import logging
import time
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
from collections import defaultdict

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
            # Store in current batch
            self.current_batch_metrics[metrics.batch_id].append(metrics)
            
            # Save to file
            self.save_metrics(metrics)
            
            logger.info(f"Tracked LLM call: {metrics.agent_name} -> {metrics.model_name} "
                       f"({metrics.total_tokens} tokens, ${metrics.cost:.4f})")
    
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
        
        model_breakdown = defaultdict(lambda: {'calls': 0, 'tokens': 0, 'cost': 0})
        agent_breakdown = defaultdict(lambda: {'calls': 0, 'tokens': 0, 'cost': 0})
        
        for metric in metrics:
            model_breakdown[metric.model_name]['calls'] += 1
            model_breakdown[metric.model_name]['tokens'] += metric.total_tokens
            model_breakdown[metric.model_name]['cost'] += metric.cost
            
            agent_breakdown[metric.agent_name]['calls'] += 1
            agent_breakdown[metric.agent_name]['tokens'] += metric.total_tokens
            agent_breakdown[metric.agent_name]['cost'] += metric.cost
        
        return {
            'batch_id': batch_id,
            'total_calls': total_calls,
            'total_tokens': total_tokens,
            'total_cost': total_cost,
            'success_rate': success_rate,
            'average_response_time': avg_response_time,
            'model_breakdown': dict(model_breakdown),
            'agent_breakdown': dict(agent_breakdown)
        }


class LLMCostManager:
    """Manages LLM costs and budgets"""
    
    def __init__(self):
        # Model costs per 1K tokens (input + output)
        self.model_costs = {
            'gpt-4': 0.03,
            'gpt-4-turbo': 0.01,
            'gpt-3.5-turbo': 0.002,
            'claude-3-sonnet': 0.015,
            'claude-3-haiku': 0.00025,
            'claude-3-opus': 0.075
        }
        
        self.budget_limits = {}
        self.current_batch_costs = defaultdict(float)
    
    def calculate_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for a model call"""
        total_tokens = input_tokens + output_tokens
        cost_per_1k = self.model_costs.get(model_name, 0.01)  # Default fallback
        return (total_tokens / 1000) * cost_per_1k
    
    def set_batch_budget(self, batch_id: str, budget_limit: float):
        """Set budget limit for a batch"""
        self.budget_limits[batch_id] = budget_limit
        logger.info(f"Set budget limit for batch {batch_id}: ${budget_limit}")
    
    def check_budget_limit(self, batch_id: str, additional_cost: float = 0) -> bool:
        """Check if batch is within budget"""
        if batch_id not in self.budget_limits:
            return True  # No limit set
        
        current_cost = self.current_batch_costs[batch_id] + additional_cost
        limit = self.budget_limits[batch_id]
        
        if current_cost > limit:
            logger.warning(f"Batch {batch_id} would exceed budget: ${current_cost:.2f} > ${limit}")
            return False
        
        return True
    
    def add_batch_cost(self, batch_id: str, cost: float):
        """Add cost to batch total"""
        self.current_batch_costs[batch_id] += cost
        logger.debug(f"Added ${cost:.4f} to batch {batch_id}, total: ${self.current_batch_costs[batch_id]:.2f}")


class LLMModelRouter:
    """Routes requests to optimal models"""
    
    def __init__(self):
        self.task_model_mapping = {
            'simple_classification': 'gpt-3.5-turbo',
            'complex_classification': 'gpt-4',
            'standards_extraction': 'claude-3-sonnet',
            'digest_generation': 'gpt-4',
            'analysis': 'gpt-4',
            'summarization': 'claude-3-sonnet',
            'code_generation': 'gpt-4',
            'default': 'gpt-4'
        }
    
    def select_model(self, task_type: str, prompt: str, model_preference: Optional[str] = None) -> str:
        """Select optimal model for task"""
        if model_preference:
            return model_preference
        
        # Simple heuristics based on prompt length and task type
        prompt_length = len(prompt)
        
        if task_type == 'simple_classification' and prompt_length < 2000:
            return 'gpt-3.5-turbo'
        elif task_type == 'summarization':
            return 'claude-3-sonnet'
        elif task_type == 'analysis' or prompt_length > 4000:
            return 'gpt-4'
        else:
            return self.task_model_mapping.get(task_type, 'gpt-4')


class LLMCacheManager:
    """Caches LLM responses to reduce costs"""
    
    def __init__(self, cache_dir: str = "llm_cache", ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600
        self.cache_hits = 0
        self.cache_misses = 0
    
    def generate_cache_key(self, prompt: str, task_type: str) -> str:
        """Generate cache key for prompt and task"""
        content = f"{task_type}:{prompt}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def check_cache(self, prompt: str, task_type: str) -> Optional[LLMResponse]:
        """Check if response is cached"""
        cache_key = self.generate_cache_key(prompt, task_type)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            # Check if cache is still valid
            if time.time() - cache_file.stat().st_mtime < self.ttl_seconds:
                try:
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                    
                    self.cache_hits += 1
                    logger.info(f"Cache hit for task: {task_type}")
                    
                    return LLMResponse(
                        content=cached_data['content'],
                        input_tokens=cached_data['input_tokens'],
                        output_tokens=cached_data['output_tokens'],
                        cost=cached_data['cost'],
                        model_name=cached_data['model_name'],
                        response_time=cached_data['response_time'],
                        success=cached_data['success']
                    )
                except Exception as e:
                    logger.warning(f"Failed to load cached response: {e}")
            else:
                # Cache expired, remove file
                cache_file.unlink()
        
        self.cache_misses += 1
        return None
    
    def cache_response(self, prompt: str, task_type: str, response: LLMResponse):
        """Cache LLM response"""
        cache_key = self.generate_cache_key(prompt, task_type)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(asdict(response), f, indent=2)
            
            logger.debug(f"Cached response for task: {task_type}")
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")


class LLMRateLimiter:
    """Rate limiting for LLM calls"""
    
    def __init__(self):
        self.rate_limits = {
            'gpt-4': {'requests_per_minute': 50, 'tokens_per_minute': 100000},
            'gpt-3.5-turbo': {'requests_per_minute': 100, 'tokens_per_minute': 200000},
            'claude-3-sonnet': {'requests_per_minute': 75, 'tokens_per_minute': 150000}
        }
        self.request_counts = defaultdict(list)
        self.token_counts = defaultdict(list)
        self.lock = threading.Lock()
    
    def check_rate_limit(self, model_name: str, estimated_tokens: int = 0):
        """Check if request is within rate limits"""
        with self.lock:
            current_time = time.time()
            window_start = current_time - 60  # 1 minute window
            
            # Clean old entries
            self.request_counts[model_name] = [t for t in self.request_counts[model_name] if t > window_start]
            self.token_counts[model_name] = [t for t in self.token_counts[model_name] if t > window_start]
            
            # Check request rate
            if model_name in self.rate_limits:
                limit = self.rate_limits[model_name]
                
                if len(self.request_counts[model_name]) >= limit['requests_per_minute']:
                    raise RateLimitExceededError(f"Request rate limit exceeded for {model_name}")
                
                if sum(self.token_counts[model_name]) + estimated_tokens >= limit['tokens_per_minute']:
                    raise RateLimitExceededError(f"Token rate limit exceeded for {model_name}")
            
            # Add current request
            self.request_counts[model_name].append(current_time)
            self.token_counts[model_name].append(estimated_tokens)


class LLMGateway:
    """Centralized gateway for all LLM interactions"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.metrics_tracker = LLMMetricsTracker()
        self.cost_manager = LLMCostManager()
        self.model_router = LLMModelRouter()
        self.cache_manager = LLMCacheManager()
        self.rate_limiter = LLMRateLimiter()
        
        self.current_batch_id = None
        self.active_batches = set()
        
        # Load configuration
        if config_path:
            self.load_config(config_path)
        
        logger.info("LLM Gateway initialized")
    
    def load_config(self, config_path: str):
        """Load gateway configuration"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Apply configuration
            if 'model_costs' in config:
                self.cost_manager.model_costs.update(config['model_costs'])
            
            if 'rate_limits' in config:
                self.rate_limiter.rate_limits.update(config['rate_limits'])
            
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load configuration: {e}")
    
    def start_batch(self, batch_id: str, budget_limit: Optional[float] = None):
        """Start a new batch processing session"""
        self.current_batch_id = batch_id
        self.active_batches.add(batch_id)
        
        if budget_limit:
            self.cost_manager.set_batch_budget(batch_id, budget_limit)
        
        logger.info(f"Started batch: {batch_id}")
    
    def end_batch(self, batch_id: str):
        """End a batch processing session"""
        self.active_batches.discard(batch_id)
        logger.info(f"Ended batch: {batch_id}")
    
    def call_llm(self, agent_name: str, task_type: str, prompt: str, 
                 model_preference: Optional[str] = None) -> LLMResponse:
        """Single entry point for all LLM calls"""
        
        if not self.current_batch_id:
            raise ValueError("No active batch. Call start_batch() first.")
        
        batch_id = self.current_batch_id
        start_time = time.time()
        
        try:
            # Step 1: Check cache
            cached_response = self.cache_manager.check_cache(prompt, task_type)
            if cached_response:
                # Track cached response
                metrics = LLMCallMetrics(
                    timestamp=datetime.now().isoformat(),
                    batch_id=batch_id,
                    agent_name=agent_name,
                    model_name=cached_response.model_name,
                    task_type=task_type,
                    input_tokens=cached_response.input_tokens,
                    output_tokens=cached_response.output_tokens,
                    total_tokens=cached_response.input_tokens + cached_response.output_tokens,
                    cost=cached_response.cost,
                    response_time=time.time() - start_time,
                    success=True,
                    cache_hit=True
                )
                self.metrics_tracker.track_call(metrics)
                return cached_response
            
            # Step 2: Model selection
            model_name = self.model_router.select_model(task_type, prompt, model_preference)
            
            # Step 3: Estimate tokens for rate limiting
            estimated_tokens = len(prompt.split()) * 1.3  # Rough estimation
            
            # Step 4: Rate limiting
            self.rate_limiter.check_rate_limit(model_name, int(estimated_tokens))
            
            # Step 5: Budget check
            estimated_cost = self.cost_manager.calculate_cost(model_name, int(estimated_tokens), 0)
            if not self.cost_manager.check_budget_limit(batch_id, estimated_cost):
                raise BudgetExceededError(f"Batch {batch_id} would exceed budget")
            
            # Step 6: Make LLM call (mock implementation for now)
            response = self._make_llm_call(model_name, prompt)
            
            # Step 7: Calculate actual costs
            actual_cost = self.cost_manager.calculate_cost(
                model_name, response.input_tokens, response.output_tokens
            )
            self.cost_manager.add_batch_cost(batch_id, actual_cost)
            
            # Step 8: Track metrics
            end_time = time.time()
            metrics = LLMCallMetrics(
                timestamp=datetime.now().isoformat(),
                batch_id=batch_id,
                agent_name=agent_name,
                model_name=model_name,
                task_type=task_type,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                total_tokens=response.input_tokens + response.output_tokens,
                cost=actual_cost,
                response_time=end_time - start_time,
                success=response.success,
                error=response.error
            )
            self.metrics_tracker.track_call(metrics)
            
            # Step 9: Cache response
            if response.success:
                self.cache_manager.cache_response(prompt, task_type, response)
            
            return response
            
        except Exception as e:
            # Track failed calls
            end_time = time.time()
            metrics = LLMCallMetrics(
                timestamp=datetime.now().isoformat(),
                batch_id=batch_id,
                agent_name=agent_name,
                model_name=model_name if 'model_name' in locals() else 'unknown',
                task_type=task_type,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                cost=0,
                response_time=end_time - start_time,
                success=False,
                error=str(e)
            )
            self.metrics_tracker.track_call(metrics)
            raise
    
    def _make_llm_call(self, model_name: str, prompt: str) -> LLMResponse:
        """Make actual LLM call (mock implementation)"""
        # This is a mock implementation
        # In production, this would call actual LLM APIs
        
        # Simulate processing time
        time.sleep(0.1)
        
        # Mock response
        mock_content = f"Mock response from {model_name} for prompt: {prompt[:50]}..."
        input_tokens = len(prompt.split())
        output_tokens = len(mock_content.split())
        
        return LLMResponse(
            content=mock_content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=0.0,  # Will be calculated by cost manager
            model_name=model_name,
            response_time=0.1,
            success=True
        )
    
    def generate_batch_report(self, batch_id: str) -> Dict[str, Any]:
        """Generate comprehensive report for a batch"""
        metrics_summary = self.metrics_tracker.get_batch_summary(batch_id)
        
        if not metrics_summary:
            return {"error": f"No metrics found for batch {batch_id}"}
        
        # Add cache statistics
        cache_stats = {
            'cache_hits': self.cache_manager.cache_hits,
            'cache_misses': self.cache_manager.cache_misses,
            'cache_hit_rate': self.cache_manager.cache_hits / (self.cache_manager.cache_hits + self.cache_manager.cache_misses) if (self.cache_manager.cache_hits + self.cache_manager.cache_misses) > 0 else 0
        }
        
        # Add cost analysis
        cost_analysis = {
            'cost_per_document': metrics_summary['total_cost'] / metrics_summary['total_calls'] if metrics_summary['total_calls'] > 0 else 0,
            'cost_per_token': metrics_summary['total_cost'] / metrics_summary['total_tokens'] if metrics_summary['total_tokens'] > 0 else 0,
            'tokens_per_call': metrics_summary['total_tokens'] / metrics_summary['total_calls'] if metrics_summary['total_calls'] > 0 else 0
        }
        
        report = {
            'batch_id': batch_id,
            'timestamp': datetime.now().isoformat(),
            'processing_summary': {
                'total_calls': metrics_summary['total_calls'],
                'total_tokens': metrics_summary['total_tokens'],
                'total_cost': metrics_summary['total_cost'],
                'success_rate': metrics_summary['success_rate'],
                'average_response_time': metrics_summary['average_response_time']
            },
            'model_breakdown': metrics_summary['model_breakdown'],
            'agent_breakdown': metrics_summary['agent_breakdown'],
            'cache_statistics': cache_stats,
            'cost_analysis': cost_analysis
        }
        
        # Save report
        report_file = Path("llm_utilization_reports") / f"batch_{batch_id}_report.json"
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Generated batch report: {report_file}")
        return report


class RateLimitExceededError(Exception):
    """Raised when rate limit is exceeded"""
    pass


class BudgetExceededError(Exception):
    """Raised when budget limit is exceeded"""
    pass
