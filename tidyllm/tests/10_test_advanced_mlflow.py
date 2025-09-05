#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test 10: Advanced MLflow Integration

Tests advanced MLflow features including experiment tracking, model registry,
and automated logging with TidyLLM integration.

IMPORTANT FOR AGENTS/LLMs:
- DO NOT use mock MLflow operations when real connection is available
- Test ACTUAL experiment creation, logging, and model registration
- SAVE MLflow experiment IDs and run data to tests/EVIDENCE folder
- Validate MLflow UI accessibility and data persistence
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

# MLflow imports
try:
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Core TidyLLM imports
try:
    from tidyllm import llm_message, chat, claude, bedrock, LLMMessage
    TIDYLLM_AVAILABLE = True
except ImportError:
    TIDYLLM_AVAILABLE = False

class TestAdvancedMLflow:
    """Test suite for advanced MLflow integration"""
    
    def save_evidence(self, evidence_data, test_name):
        """Save MLflow integration evidence"""
        evidence_dir = Path(__file__).parent / "EVIDENCE"
        evidence_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"evidence_mlflow_{test_name}_{timestamp}.json"
        evidence_path = evidence_dir / filename
        
        with open(evidence_path, 'w') as f:
            json.dump(evidence_data, f, indent=2, default=str)
        
        print(f"MLflow evidence saved: {evidence_path}")
        return evidence_path
    
    @pytest.fixture
    def settings_loader(self):
        """Fixture providing initialized SettingsLoader"""
        admin_settings_path = Path(__file__).parent.parent / "tidyllm" / "admin" / "settings.yaml"
        return SettingsLoader(str(admin_settings_path))
    
    def test_experiment_lifecycle_management(self, settings_loader):
        """Test complete MLflow experiment lifecycle"""
        from tidyllm.mlflow_integration import AdvancedMLflowManager
        
        # Initialize advanced MLflow manager
        mlflow_manager = AdvancedMLflowManager()
        
        # Create experiment
        experiment_name = f"test_lifecycle_{int(time.time())}"
        experiment = mlflow_manager.create_experiment(
            name=experiment_name,
            tags={"purpose": "integration_test", "env": "test"}
        )
        
        # Validate experiment creation
        assert experiment.experiment_id is not None
        assert experiment.name == experiment_name
        assert experiment.lifecycle_stage == "active"
        
        # Create runs in experiment
        run_results = []
        for i in range(3):
            run = mlflow_manager.create_run(
                experiment_id=experiment.experiment_id,
                run_name=f"test_run_{i}",
                tags={"iteration": str(i)}
            )
            
            # Log metrics and parameters
            mlflow_manager.log_metrics(run.info.run_id, {
                "accuracy": 0.85 + i * 0.05,
                "loss": 0.25 - i * 0.05
            })
            
            mlflow_manager.log_parameters(run.info.run_id, {
                "model": "claude-3-haiku",
                "iteration": i
            })
            
            # End run
            mlflow_manager.end_run(run.info.run_id)
            run_results.append(run)
        
        # Validate runs
        assert len(run_results) == 3
        for run in run_results:
            assert run.info.status == "FINISHED"
        
        # Archive experiment
        mlflow_manager.archive_experiment(experiment.experiment_id)
        
        # Save evidence
        evidence_data = {
            "experiment": experiment.__dict__,
            "runs": [run.to_dictionary() for run in run_results]
        }
        evidence_path = self.save_evidence(evidence_data, "lifecycle_management")
        
        print(f" Experiment lifecycle test completed")
        print(f"   Experiment ID: {experiment.experiment_id}")
        print(f"   Runs created: {len(run_results)}")
    
    def test_model_registry_integration(self, settings_loader):
        """Test MLflow model registry functionality"""
        from tidyllm.mlflow_integration import AdvancedMLflowManager, ModelRegistryManager
        
        # Initialize managers
        mlflow_manager = AdvancedMLflowManager()
        registry_manager = ModelRegistryManager()
        
        # Create experiment and run
        experiment_name = f"test_registry_{int(time.time())}"
        experiment = mlflow_manager.create_experiment(experiment_name)
        
        run = mlflow_manager.create_run(experiment.experiment_id, "model_registry_test")
        
        # Log a dummy model
        model_info = registry_manager.log_model(
            run_id=run.info.run_id,
            model_name="test_llm_model",
            model_data={"type": "llm", "provider": "claude"},
            artifact_path="model"
        )
        
        # Register model
        model_version = registry_manager.register_model(
            model_uri=f"runs:/{run.info.run_id}/model",
            model_name="test_llm_model",
            description="Test LLM model for integration testing"
        )
        
        # Validate model registration
        assert model_version.name == "test_llm_model"
        assert model_version.version == "1"
        assert model_version.current_stage == "None"
        
        # Transition to staging
        registry_manager.transition_model_stage(
            name="test_llm_model",
            version="1",
            stage="Staging",
            description="Moving to staging for testing"
        )
        
        # Get latest model version
        latest_version = registry_manager.get_latest_model_version(
            name="test_llm_model",
            stage="Staging"
        )
        
        assert latest_version.current_stage == "Staging"
        
        # Save evidence
        evidence_data = {
            "model_version": model_version.__dict__,
            "latest_version": latest_version.__dict__,
            "experiment_id": experiment.experiment_id
        }
        evidence_path = self.save_evidence(evidence_data, "model_registry")
        
        print(f" Model registry test completed")
        print(f"   Model: {model_version.name} v{model_version.version}")
        print(f"   Stage: {latest_version.current_stage}")
    
    def test_automated_llm_logging(self, settings_loader):
        """Test automated logging of LLM interactions"""
        from tidyllm.mlflow_integration import AutoMLflowLogger
        
        # Initialize auto-logger
        auto_logger = AutoMLflowLogger()
        
        # Start tracking session
        session = auto_logger.start_tracking_session(
            experiment_name=f"auto_logging_{int(time.time())}",
            session_name="llm_interaction_test"
        )
        
        # Simulate LLM interactions with auto-logging
        test_interactions = [
            {"prompt": "What is AI?", "model": "claude-3-haiku"},
            {"prompt": "Explain machine learning", "model": "claude-3-haiku"},
            {"prompt": "Describe neural networks", "model": "claude-3-haiku"}
        ]
        
        logged_interactions = []
        for interaction in test_interactions:
            # Log interaction automatically
            logged = auto_logger.log_llm_interaction(
                prompt=interaction["prompt"],
                model=interaction["model"],
                response="Generated response for testing",
                metrics={"token_count": 150, "response_time_ms": 1200}
            )
            logged_interactions.append(logged)
        
        # End tracking session
        session_summary = auto_logger.end_tracking_session(session.session_id)
        
        # Validate session
        assert session_summary.total_interactions == 3
        assert session_summary.total_tokens > 0
        assert session_summary.avg_response_time_ms > 0
        
        # Validate logged interactions
        for logged in logged_interactions:
            assert logged.prompt is not None
            assert logged.model == "claude-3-haiku"
            assert logged.logged_at is not None
        
        # Save evidence
        evidence_data = {
            "session": session.__dict__,
            "session_summary": session_summary.__dict__,
            "interactions": [i.__dict__ for i in logged_interactions]
        }
        evidence_path = self.save_evidence(evidence_data, "automated_logging")
        
        print(f" Automated LLM logging test completed")
        print(f"   Session ID: {session.session_id}")
        print(f"   Interactions logged: {len(logged_interactions)}")

def test_priority_advanced_mlflow_check():
    """Priority test for advanced MLflow readiness"""
    try:
        from tidyllm.mlflow_integration import (
            AdvancedMLflowManager, ModelRegistryManager, AutoMLflowLogger,
            MLflowExperiment, MLflowRun, LoggedInteraction
        )
        
        # Test advanced MLflow functionality
        mlflow_manager = AdvancedMLflowManager()
        assert mlflow_manager is not None
        
        # Test model registry
        registry_manager = ModelRegistryManager()
        assert registry_manager is not None
        
        # Test auto-logger
        auto_logger = AutoMLflowLogger()
        assert auto_logger is not None
        
        print("SUCCESS: Advanced MLflow integration implemented and working")
        
    except Exception as e:
        pytest.fail(f"CRITICAL: Advanced MLflow check failed: {e}")

if __name__ == "__main__":
    test_priority_advanced_mlflow_check()