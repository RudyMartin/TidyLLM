#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test 9: Multi-Model Comparison

Tests comparison across different LLM models (Claude, GPT, etc.) with performance
metrics and response quality analysis.

IMPORTANT FOR AGENTS/LLMs:
- DO NOT use mock responses when real API credentials are available
- Test ACTUAL model responses and compare quality metrics
- SAVE model response comparisons to tests/EVIDENCE folder
- Measure performance, accuracy, and response characteristics
"""

import os
import sys
import json
import pytest
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tidyllm.settings_loader import SettingsLoader

# Core TidyLLM imports
try:
    from tidyllm import llm_message, chat, claude, bedrock, LLMMessage
    TIDYLLM_AVAILABLE = True
except ImportError:
    TIDYLLM_AVAILABLE = False

class TestMultiModelComparison:
    """Test suite for multi-model comparison functionality"""
    
    def save_evidence(self, evidence_data, test_name):
        """Save model comparison evidence"""
        evidence_dir = Path(__file__).parent / "EVIDENCE"
        evidence_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"evidence_models_{test_name}_{timestamp}.json"
        evidence_path = evidence_dir / filename
        
        with open(evidence_path, 'w') as f:
            json.dump(evidence_data, f, indent=2, default=str)
        
        print(f"Model comparison evidence saved: {evidence_path}")
        return evidence_path
    
    @pytest.fixture
    def settings_loader(self):
        """Fixture providing initialized SettingsLoader"""
        admin_settings_path = Path(__file__).parent.parent / "tidyllm" / "admin" / "settings.yaml"
        return SettingsLoader(str(admin_settings_path))
    
    def test_claude_vs_bedrock_response_quality(self, settings_loader):
        """Compare Claude and Bedrock response quality"""
        from tidyllm.model_comparison import ModelComparator, ComparisonMetrics
        
        # Test prompt for comparison
        test_prompt = "Explain quantum computing in simple terms"
        
        comparator = ModelComparator()
        
        # Compare Claude vs Bedrock
        comparison = comparator.compare_models(
            prompt=test_prompt,
            models=["claude-3-haiku", "amazon.titan-text-express-v1"],
            metrics=["response_time", "token_count", "quality_score"]
        )
        
        # Validate comparison results
        assert comparison.prompt == test_prompt
        assert len(comparison.model_results) == 2
        assert comparison.winner is not None
        
        # Check individual model results
        for model_result in comparison.model_results:
            assert model_result.model_name in ["claude-3-haiku", "amazon.titan-text-express-v1"]
            assert model_result.response_time_ms > 0
            assert len(model_result.response_text) > 0
            assert model_result.token_count > 0
        
        # Save evidence
        evidence_path = self.save_evidence(comparison.__dict__, "claude_vs_bedrock")
        
        print(f" Model comparison completed")
        print(f"   Winner: {comparison.winner}")
        print(f"   Models tested: {[r.model_name for r in comparison.model_results]}")
    
    def test_performance_benchmarking(self, settings_loader):
        """Test performance across multiple models"""
        from tidyllm.model_comparison import ModelComparator, PerformanceBenchmark
        
        benchmark = PerformanceBenchmark()
        
        # Run performance test
        results = benchmark.run_benchmark(
            models=["claude-3-haiku", "amazon.titan-text-express-v1"],
            test_prompts=[
                "What is machine learning?",
                "Explain photosynthesis",
                "Describe the solar system"
            ],
            iterations=2
        )
        
        # Validate benchmark results
        assert len(results.model_performances) == 2
        assert results.total_prompts == 3
        assert results.iterations_per_prompt == 2
        
        # Check performance metrics
        for model_perf in results.model_performances:
            assert model_perf.avg_response_time_ms > 0
            assert model_perf.total_requests == 6  # 3 prompts * 2 iterations
            assert model_perf.success_rate >= 0
        
        # Save evidence
        evidence_path = self.save_evidence(results.__dict__, "performance_benchmark")
        
        print(f" Performance benchmark completed")
        print(f"   Models: {[p.model_name for p in results.model_performances]}")
        print(f"   Total requests: {sum(p.total_requests for p in results.model_performances)}")

def test_priority_model_comparison_check():
    """Priority test for model comparison readiness"""
    try:
        from tidyllm.model_comparison import (
            ModelComparator, ComparisonMetrics, ModelResult, ComparisonResult,
            PerformanceBenchmark, ModelPerformance, BenchmarkResult
        )
        
        # Test basic model comparison functionality
        comparator = ModelComparator()
        assert comparator is not None
        
        # Test comparison metrics
        metrics = ComparisonMetrics()
        assert metrics is not None
        assert hasattr(metrics, 'calculate_quality_score')
        
        # Test performance benchmark
        benchmark = PerformanceBenchmark()
        assert benchmark is not None
        
        print("SUCCESS: Multi-model comparison system implemented and working")
        
    except Exception as e:
        pytest.fail(f"CRITICAL: Model comparison check failed: {e}")

if __name__ == "__main__":
    test_priority_model_comparison_check()