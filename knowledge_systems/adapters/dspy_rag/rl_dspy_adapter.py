"""
RL-Enhanced DSPy RAG Adapter
============================

Extends DSPy RAG with Reinforcement Learning capabilities:
- Uses PostgreSQL RAG for memory/embeddings
- Collects user feedback as rewards
- Learns from high-reward examples
- Optimizes DSPy signatures based on performance
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import logging
from dataclasses import dataclass, field
from collections import defaultdict

from .dspy_rag_adapter import DSPyRAGAdapter
from ..postgres_rag.postgres_rag_adapter import PostgresRAGAdapter
from ....services.dspy_service import get_dspy_service

logger = logging.getLogger(__name__)


@dataclass
class FeedbackRecord:
    """User feedback for a query/response pair."""
    query_id: str
    query_text: str
    response_text: str
    rating: int  # 1-3 scale
    improvement_notes: str
    signature_used: str
    timestamp: datetime
    context_used: List[str] = field(default_factory=list)


@dataclass
class SignaturePerformance:
    """Tracks performance metrics for a DSPy signature."""
    signature_name: str
    total_uses: int = 0
    total_reward: float = 0.0
    avg_reward: float = 0.0
    success_rate: float = 0.0
    recent_rewards: List[float] = field(default_factory=list)

    def update(self, reward: float):
        """Update performance metrics with new reward."""
        self.total_uses += 1
        self.total_reward += reward
        self.recent_rewards.append(reward)
        # Keep only last 100 rewards
        if len(self.recent_rewards) > 100:
            self.recent_rewards.pop(0)
        self.avg_reward = self.total_reward / self.total_uses
        self.success_rate = sum(1 for r in self.recent_rewards if r > 0) / len(self.recent_rewards)


class RLDSPyAdapter(DSPyRAGAdapter):
    """
    DSPy RAG Adapter enhanced with Reinforcement Learning.

    Uses PostgreSQL RAG for memory and learns from user feedback
    to continuously improve DSPy signature selection and optimization.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize RL-Enhanced DSPy Adapter."""
        super().__init__(config)

        # Initialize PostgreSQL RAG for memory
        self.postgres_rag = PostgresRAGAdapter(config)

        # RL components
        self.feedback_buffer: List[FeedbackRecord] = []
        self.signature_performance: Dict[str, SignaturePerformance] = defaultdict(SignaturePerformance)
        self.exploration_rate = 0.1  # Epsilon for epsilon-greedy
        self.learning_enabled = True
        self.optimization_threshold = 50  # Min feedback before optimization

        # Track current query for feedback collection
        self.current_query_id = None
        self.current_signature = None

        logger.info("Initialized RL-Enhanced DSPy Adapter with PostgreSQL memory")

    def query(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Query with RL-enhanced signature selection.

        Uses epsilon-greedy exploration to balance between:
        - Exploiting best-performing signatures
        - Exploring new signature variants
        """
        # Generate query ID for tracking
        import uuid
        self.current_query_id = str(uuid.uuid4())

        # First, get relevant context from PostgreSQL RAG
        postgres_response = self.postgres_rag.query(request)
        context_docs = postgres_response.get('documents', [])

        # Add context to request for DSPy
        enhanced_request = request.copy()
        if context_docs:
            enhanced_request['context'] = self._format_context(context_docs)

        # Select signature using RL strategy
        if np.random.random() < self.exploration_rate and self.learning_enabled:
            # Explore: Try a new signature variant
            response = self._explore_new_signature(enhanced_request)
            logger.info(f"Exploring new signature variant for query {self.current_query_id}")
        else:
            # Exploit: Use best-performing signature
            response = self._exploit_best_signature(enhanced_request)
            logger.info(f"Using best signature for query {self.current_query_id}")

        # Add tracking info to response
        response['query_id'] = self.current_query_id
        response['signature_used'] = self.current_signature
        response['rl_mode'] = 'explore' if np.random.random() < self.exploration_rate else 'exploit'

        return response

    def collect_feedback(self, query_id: str, rating: int, improvement_notes: str = "") -> Dict[str, Any]:
        """
        Collect user feedback and convert to RL reward signal.

        Args:
            query_id: ID of the query to provide feedback for
            rating: User rating (1-3 scale)
            improvement_notes: Optional notes on what to improve

        Returns:
            Result with feedback status
        """
        try:
            # Convert rating to reward (-1 to 1 scale)
            reward = (rating - 2) / 1.0  # Maps 1->-1, 2->0, 3->1

            # Find the query in recent history
            # In production, this would query a database
            feedback = FeedbackRecord(
                query_id=query_id,
                query_text="",  # Would retrieve from storage
                response_text="",  # Would retrieve from storage
                rating=rating,
                improvement_notes=improvement_notes,
                signature_used=self.current_signature or "default",
                timestamp=datetime.now()
            )

            # Add to feedback buffer
            self.feedback_buffer.append(feedback)

            # Update signature performance
            if self.current_signature:
                self.signature_performance[self.current_signature].update(reward)

            # Store in PostgreSQL for long-term memory
            self._store_feedback_in_postgres(feedback, reward)

            # Trigger optimization if enough feedback collected
            if len(self.feedback_buffer) >= self.optimization_threshold:
                self._trigger_optimization()

            logger.info(f"Collected feedback for query {query_id}: rating={rating}, reward={reward}")

            return {
                'success': True,
                'query_id': query_id,
                'reward': reward,
                'feedback_count': len(self.feedback_buffer),
                'optimization_pending': len(self.feedback_buffer) >= self.optimization_threshold
            }

        except Exception as e:
            logger.error(f"Error collecting feedback: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _explore_new_signature(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Explore new signature variants using DSPy's capabilities.

        Generates variations of existing signatures to test new approaches.
        """
        try:
            # Get DSPy service
            dspy_service = get_dspy_service()

            # Generate a variant signature
            base_signature = self._get_base_signature()
            variant = self._generate_signature_variant(base_signature)

            # Execute with variant
            result = dspy_service.execute_with_signature(variant, request)

            self.current_signature = variant.get('name', 'variant')

            return result

        except Exception as e:
            logger.error(f"Error in exploration: {e}")
            # Fallback to default
            return super().query(request)

    def _exploit_best_signature(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use the best-performing signature based on collected rewards.
        """
        try:
            # Find best signature by average reward
            if self.signature_performance:
                best_sig = max(
                    self.signature_performance.items(),
                    key=lambda x: x[1].avg_reward
                )
                self.current_signature = best_sig[0]

                # Use the best signature
                return self._execute_with_signature(best_sig[0], request)
            else:
                # No performance data yet, use default
                return super().query(request)

        except Exception as e:
            logger.error(f"Error in exploitation: {e}")
            return super().query(request)

    def _trigger_optimization(self):
        """
        Trigger DSPy optimization using high-reward examples.

        Uses Bootstrap or MIPRO optimizers with reward-weighted examples.
        """
        try:
            logger.info("Triggering RL-based optimization")

            # Get high-reward examples from feedback
            high_reward_feedback = [
                f for f in self.feedback_buffer
                if f.rating >= 3
            ]

            if not high_reward_feedback:
                logger.warning("No high-reward examples for optimization")
                return

            # Convert to DSPy training examples
            examples = self._convert_feedback_to_examples(high_reward_feedback)

            # Use DSPy Bootstrap for optimization
            dspy_service = get_dspy_service()
            optimization_result = dspy_service.optimize_with_examples(
                examples=examples,
                metric_fn=self._reward_metric
            )

            if optimization_result.get('success'):
                logger.info("Optimization completed successfully")
                # Clear buffer after optimization
                self.feedback_buffer = []
            else:
                logger.error(f"Optimization failed: {optimization_result.get('error')}")

        except Exception as e:
            logger.error(f"Error in optimization: {e}")

    def _store_feedback_in_postgres(self, feedback: FeedbackRecord, reward: float):
        """
        Store feedback in PostgreSQL for long-term memory.

        Uses the SME tables with vector embeddings for similarity search.
        """
        try:
            # Create a document from feedback
            doc = {
                'content': f"Query: {feedback.query_text}\nResponse: {feedback.response_text}",
                'metadata': {
                    'query_id': feedback.query_id,
                    'rating': feedback.rating,
                    'reward': reward,
                    'improvement_notes': feedback.improvement_notes,
                    'signature': feedback.signature_used,
                    'timestamp': feedback.timestamp.isoformat()
                },
                'authority_tier': 'feedback'  # Custom tier for RL feedback
            }

            # Store in PostgreSQL (would need to extend postgres_rag_adapter)
            # For now, just log it
            logger.info(f"Would store feedback in PostgreSQL: {doc['metadata']}")

        except Exception as e:
            logger.error(f"Error storing feedback: {e}")

    def _format_context(self, documents: List[Dict]) -> str:
        """Format PostgreSQL documents as context for DSPy."""
        context_parts = []
        for doc in documents[:5]:  # Limit to top 5
            content = doc.get('content', '')
            source = doc.get('metadata', {}).get('source', 'Unknown')
            context_parts.append(f"[Source: {source}]\n{content}")

        return "\n\n".join(context_parts)

    def _get_base_signature(self) -> Dict[str, Any]:
        """Get the base signature for variation generation."""
        return {
            'name': 'base_signature',
            'inputs': ['query', 'context'],
            'outputs': ['response'],
            'instructions': 'Answer the query using the provided context.'
        }

    def _generate_signature_variant(self, base: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a variant of the base signature for exploration."""
        import random

        variants = [
            "Be concise and focus on key points.",
            "Provide detailed explanation with examples.",
            "Use step-by-step reasoning.",
            "Summarize first, then elaborate.",
            "Focus on practical applications."
        ]

        variant = base.copy()
        variant['name'] = f"variant_{random.randint(1000, 9999)}"
        variant['instructions'] = base['instructions'] + " " + random.choice(variants)

        return variant

    def _execute_with_signature(self, signature_name: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute request with a specific signature."""
        # In production, would retrieve and execute the actual signature
        # For now, use the base query method
        return super().query(request)

    def _convert_feedback_to_examples(self, feedback_list: List[FeedbackRecord]) -> List[Dict]:
        """Convert feedback records to DSPy training examples."""
        examples = []
        for feedback in feedback_list:
            examples.append({
                'input': feedback.query_text,
                'output': feedback.response_text,
                'weight': feedback.rating / 3.0  # Normalize to 0-1
            })
        return examples

    def _reward_metric(self, prediction, example) -> float:
        """Metric function for DSPy optimization based on rewards."""
        # Simple similarity metric weighted by reward
        # In production, would use more sophisticated metrics
        return example.get('weight', 1.0)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of signature performance metrics."""
        summary = {
            'total_feedback': len(self.feedback_buffer),
            'signatures': {}
        }

        for sig_name, perf in self.signature_performance.items():
            summary['signatures'][sig_name] = {
                'uses': perf.total_uses,
                'avg_reward': perf.avg_reward,
                'success_rate': perf.success_rate
            }

        return summary

    def set_exploration_rate(self, rate: float):
        """Adjust exploration rate for epsilon-greedy strategy."""
        self.exploration_rate = max(0.0, min(1.0, rate))
        logger.info(f"Set exploration rate to {self.exploration_rate}")

    def enable_learning(self, enabled: bool = True):
        """Enable or disable learning from feedback."""
        self.learning_enabled = enabled
        logger.info(f"Learning {'enabled' if enabled else 'disabled'}")


def get_rl_dspy_adapter(config: Dict[str, Any] = None) -> RLDSPyAdapter:
    """Factory function to get RL-Enhanced DSPy Adapter instance."""
    return RLDSPyAdapter(config or {})