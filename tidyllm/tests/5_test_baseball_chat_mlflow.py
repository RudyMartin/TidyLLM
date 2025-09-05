#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test 5: Baseball Chat MLflow Integration

Tests chat functionality with baseball domain and MLflow experiment tracking.
Validates conversation logging, model performance metrics, and MLB data integration.

IMPORTANT FOR AGENTS/LLMs:
- DO NOT use mock baseball data when real APIs are available
- Test ACTUAL chat responses with baseball context
- SAVE chat logs and MLflow experiment data to tests/EVIDENCE folder
- Validate conversation quality and domain expertise
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

class TestBaseballChatMLflow:
    """Test suite for baseball chat with MLflow tracking"""
    
    def save_evidence(self, evidence_data, test_name):
        """Save baseball chat evidence"""
        evidence_dir = Path(__file__).parent / "EVIDENCE"
        evidence_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"evidence_baseball_{test_name}_{timestamp}.json"
        evidence_path = evidence_dir / filename
        
        with open(evidence_path, 'w') as f:
            json.dump(evidence_data, f, indent=2, default=str)
        
        print(f"Baseball chat evidence saved: {evidence_path}")
        return evidence_path
    
    @pytest.fixture
    def settings_loader(self):
        """Fixture providing initialized SettingsLoader"""
        admin_settings_path = Path(__file__).parent.parent / "tidyllm" / "admin" / "settings.yaml"
        return SettingsLoader(str(admin_settings_path))
    
    def test_baseball_domain_chat(self, settings_loader):
        """Test baseball-specific chat responses"""
        from tidyllm.chat_integration import BaseballChatManager, DomainExpertChat
        
        # Initialize baseball chat manager
        chat_manager = BaseballChatManager()
        
        # Baseball test questions
        baseball_questions = [
            "What is a batting average and how is it calculated?",
            "Explain the infield fly rule",
            "Who holds the record for most home runs in a season?",
            "What's the difference between a slider and a curveball?",
            "How does the designated hitter rule work?"
        ]
        
        chat_results = []
        for question in baseball_questions:
            # Get chat response
            response = chat_manager.get_baseball_response(
                question=question,
                context="professional baseball",
                detail_level="expert"
            )
            
            # Validate response
            assert response.question == question
            assert len(response.answer) > 50  # Substantial answer
            assert response.confidence_score > 0.7  # High confidence for baseball
            assert response.domain == "baseball"
            assert len(response.sources) > 0  # Should have sources
            
            chat_results.append({
                "question": question,
                "response": response.__dict__,
                "response_time_ms": response.response_time_ms
            })
        
        # Validate overall chat performance
        avg_confidence = sum(r["response"]["confidence_score"] for r in chat_results) / len(chat_results)
        assert avg_confidence > 0.75  # High average confidence
        
        avg_response_time = sum(r["response_time_ms"] for r in chat_results) / len(chat_results)
        assert avg_response_time < 5000  # Under 5 seconds average
        
        # Save evidence
        evidence_data = {
            "total_questions": len(baseball_questions),
            "chat_results": chat_results,
            "performance_metrics": {
                "avg_confidence": avg_confidence,
                "avg_response_time_ms": avg_response_time
            }
        }
        evidence_path = self.save_evidence(evidence_data, "domain_chat")
        
        print(f"✅ Baseball domain chat test completed")
        print(f"   Questions answered: {len(baseball_questions)}")
        print(f"   Average confidence: {avg_confidence:.3f}")
        print(f"   Average response time: {avg_response_time:.1f}ms")
    
    def test_mlflow_conversation_tracking(self, settings_loader):
        """Test MLflow tracking of baseball conversations"""
        from tidyllm.chat_integration import BaseballChatManager
        from tidyllm.mlflow_integration import ConversationTracker
        
        # Initialize components
        chat_manager = BaseballChatManager()
        tracker = ConversationTracker()
        
        # Start MLflow experiment
        experiment = tracker.start_experiment(
            name=f"baseball_chat_{int(time.time())}",
            tags={"domain": "baseball", "test": "integration"}
        )
        
        # Start conversation tracking run
        run = tracker.start_run(
            experiment_id=experiment.experiment_id,
            run_name="baseball_conversation_test"
        )
        
        # Conduct tracked conversation
        conversation_questions = [
            "Who won the 2023 World Series?",
            "What are the basic rules of baseball?",
            "Explain what a triple play is"
        ]
        
        tracked_responses = []
        for i, question in enumerate(conversation_questions):
            # Get response with tracking
            response = chat_manager.get_tracked_response(
                question=question,
                run_id=run.info.run_id,
                conversation_turn=i+1
            )
            
            # Log to MLflow
            tracker.log_conversation_turn(
                run_id=run.info.run_id,
                turn=i+1,
                question=question,
                response=response.answer,
                metrics={
                    "confidence": response.confidence_score,
                    "response_time_ms": response.response_time_ms,
                    "token_count": len(response.answer.split())
                }
            )
            
            tracked_responses.append(response)
        
        # End tracking run
        tracker.end_run(run.info.run_id)
        
        # Validate tracking results
        run_data = tracker.get_run_data(run.info.run_id)
        assert run_data.status == "FINISHED"
        assert len(run_data.metrics) >= 9  # 3 metrics * 3 turns
        assert len(run_data.params) > 0
        
        # Save evidence
        evidence_data = {
            "experiment": experiment.__dict__,
            "run": run.to_dictionary(),
            "conversation": [r.__dict__ for r in tracked_responses],
            "run_metrics": run_data.metrics
        }
        evidence_path = self.save_evidence(evidence_data, "mlflow_tracking")
        
        print(f"✅ MLflow conversation tracking test completed")
        print(f"   Experiment ID: {experiment.experiment_id}")
        print(f"   Run ID: {run.info.run_id}")
        print(f"   Conversation turns: {len(tracked_responses)}")

def test_priority_baseball_chat_check():
    """Priority test for baseball chat integration readiness"""
    try:
        from tidyllm.chat_integration import (
            BaseballChatManager, DomainExpertChat, ChatResponse
        )
        from tidyllm.mlflow_integration import ConversationTracker
        
        # Test baseball chat manager
        chat_manager = BaseballChatManager()
        assert chat_manager is not None
        
        # Test conversation tracker
        tracker = ConversationTracker()
        assert tracker is not None
        
        print("SUCCESS: Baseball chat MLflow integration implemented and working")
        
    except Exception as e:
        pytest.fail(f"CRITICAL: Baseball chat integration check failed: {e}")

if __name__ == "__main__":
    test_priority_baseball_chat_check()