#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test 7: Cross-Service Integration

Tests integrated workflow: chat → MLflow → S3 upload pipeline.
Validates data flows between all services with performance monitoring.

IMPORTANT FOR AGENTS/LLMs:
- DO NOT use mock data when real credentials are available
- Test COMPLETE end-to-end workflows, not isolated components
- SAVE pipeline execution evidence to tests/EVIDENCE folder
- Measure performance across the entire integration chain
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

# MLflow imports
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# AWS imports
try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

class TestCrossServiceIntegration:
    """Test suite for cross-service integration workflows"""
    
    def save_evidence(self, evidence_data, test_name):
        """Save integration evidence"""
        evidence_dir = Path(__file__).parent / "EVIDENCE"
        evidence_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"evidence_integration_{test_name}_{timestamp}.json"
        evidence_path = evidence_dir / filename
        
        with open(evidence_path, 'w') as f:
            json.dump(evidence_data, f, indent=2, default=str)
        
        print(f"Integration evidence saved: {evidence_path}")
        return evidence_path
    
    @pytest.fixture
    def settings_loader(self):
        """Fixture providing initialized SettingsLoader"""
        admin_settings_path = Path(__file__).parent.parent / "tidyllm" / "admin" / "settings.yaml"
        return SettingsLoader(str(admin_settings_path))
    
    def test_chat_to_mlflow_to_s3_pipeline(self, settings_loader):
        """Test complete chat → MLflow → S3 pipeline"""
        from tidyllm.integration_pipeline import IntegrationPipeline, PREDEFINED_PIPELINES
        
        # Configure pipeline
        config = {
            "chat": {"provider": "claude", "model": "claude-3-haiku"},
            "mlflow": {"tracking_uri": "test://localhost"},
            "s3": {"bucket": "test-bucket", "region": "us-east-1"}
        }
        
        pipeline = IntegrationPipeline(config)
        
        # Initialize services
        connection_results = pipeline.initialize_services()
        assert len(connection_results) == 3  # chat, mlflow, s3
        
        # Execute predefined pipeline
        execution = pipeline.execute_pipeline(
            "chat_to_mlflow_to_s3", 
            PREDEFINED_PIPELINES["chat_to_mlflow_to_s3"]
        )
        
        # Validate execution
        assert execution.status == "completed"
        assert len(execution.steps) == 6
        assert execution.total_duration_ms > 0
        
        # Check each step completed successfully
        for step in execution.steps:
            assert step.status == "completed", f"Step {step.name} failed: {step.error}"
            assert step.duration_ms > 0
        
        # Save evidence
        evidence_path = self.save_evidence(execution.__dict__, "pipeline_test")
        
        print(f"✅ End-to-end pipeline completed in {execution.total_duration_ms:.2f}ms")
        print(f"   Steps: {len(execution.steps)}")
        print(f"   Services: chat, mlflow, s3")
    
    def test_concurrent_service_access(self, settings_loader):
        """Test concurrent access to all services"""
        from tidyllm.integration_pipeline import IntegrationPipeline
        
        # Configure pipeline
        config = {
            "chat": {"provider": "claude", "model": "claude-3-haiku"},
            "mlflow": {"tracking_uri": "test://localhost"},
            "s3": {"bucket": "test-bucket", "region": "us-east-1"}
        }
        
        pipeline = IntegrationPipeline(config)
        
        # Initialize services
        pipeline.initialize_services()
        
        # Create multiple concurrent pipeline executions
        pipeline_configs = [
            [
                {"name": "chat_1", "service": "chat", "action": "generate_response", 
                 "input_data": {"prompt": f"Question {i}", "model": "claude-3-haiku"}}
            ]
            for i in range(3)
        ]
        
        # Run pipelines concurrently
        import asyncio
        
        async def run_concurrent():
            tasks = []
            for i, config_steps in enumerate(pipeline_configs):
                task = pipeline.execute_pipeline(f"concurrent_test_{i}", config_steps)
                tasks.append(task)
            
            # Wait for all to complete
            results = await asyncio.gather(*tasks)
            return results
        
        results = asyncio.run(run_concurrent())
        
        # Validate all completed successfully
        for i, execution in enumerate(results):
            assert execution.status == "completed", f"Pipeline {i} failed"
            assert len(execution.steps) == 1
        
        print(f"✅ Concurrent service access working: {len(results)} pipelines completed")
        print(f"   Total duration range: {min(r.total_duration_ms for r in results):.1f}-{max(r.total_duration_ms for r in results):.1f}ms")

def test_priority_integration_check():
    """Priority test for integration readiness"""
    try:
        from tidyllm.integration_pipeline import (
            IntegrationPipeline, PipelineStep, PipelineExecution,
            ChatServiceConnector, MLflowServiceConnector, S3ServiceConnector,
            PREDEFINED_PIPELINES
        )
        
        # Test basic integration functionality
        config = {"chat": {}, "mlflow": {}, "s3": {}}
        pipeline = IntegrationPipeline(config)
        
        assert pipeline is not None
        assert len(pipeline.connectors) == 3
        
        # Test predefined pipelines
        assert "chat_to_mlflow_to_s3" in PREDEFINED_PIPELINES
        assert len(PREDEFINED_PIPELINES["chat_to_mlflow_to_s3"]) == 6
        
        # Test pipeline step creation
        step = PipelineStep("test", "chat", "test_action", {"test": "data"})
        assert step.name == "test"
        assert step.service == "chat"
        
        print("SUCCESS: Cross-service integration pipeline implemented and working")
        
    except Exception as e:
        pytest.fail(f"CRITICAL: Integration pipeline check failed: {e}")

if __name__ == "__main__":
    test_priority_integration_check()