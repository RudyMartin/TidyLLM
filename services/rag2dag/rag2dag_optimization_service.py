"""
RAG2DAG Optimization Service
============================

Core service for RAG workflow optimization, pattern detection, and performance analysis.
Follows the enterprise service pattern similar to MLflow services.
"""

import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

from tidyllm.rag2dag.converter import RAG2DAGConverter, RAGPatternType
from tidyllm.rag2dag.config import RAG2DAGConfig
from .rag2dag_pattern_service import RAG2DAGPatternService
from .rag2dag_execution_service import RAG2DAGExecutionService

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of RAG2DAG optimization analysis."""
    should_optimize: bool
    pattern_detected: Optional[RAGPatternType]
    estimated_speedup: float
    estimated_cost_factor: float
    dag_workflow: Optional[Dict[str, Any]]
    optimization_reason: str
    confidence_score: float = 0.0


@dataclass
class OptimizationSuggestion:
    """Suggestion for workflow optimization."""
    performance_gain: int  # Percentage improvement
    cost_impact: str       # "reduced", "neutral", "increased"
    complexity_reduction: bool
    suggested_changes: List[str]
    dag_preview: Dict[str, Any]
    confidence_score: float = 0.0


class RAG2DAGOptimizationService:
    """
    Enterprise RAG2DAG Optimization Service.

    Provides:
    - Intelligent pattern detection and optimization analysis
    - Workflow design suggestions and performance estimates
    - DAG workflow generation and execution coordination
    - Performance tracking and analytics
    - Integration with corporate governance and compliance
    """

    def __init__(self):
        """Initialize the RAG2DAG optimization service."""
        self.config = RAG2DAGConfig.create_default_config()
        self.converter = RAG2DAGConverter(self.config)

        # Initialize sub-services
        self.pattern_service = RAG2DAGPatternService(self.config)
        self.execution_service = RAG2DAGExecutionService(self.config)

        # Service state and metrics
        self.optimization_stats = {
            "total_analyzed": 0,
            "optimizations_applied": 0,
            "avg_speedup": 0.0,
            "patterns_detected": {},
            "cost_savings_total": 0.0,
            "service_uptime": datetime.now()
        }

        # Performance thresholds
        self.optimization_thresholds = {
            "min_files_for_optimization": 3,
            "min_complexity_score": 5,
            "min_estimated_speedup": 1.3,
            "confidence_threshold": 0.7
        }

    def analyze_request_optimization(self, request: str, context: str = "", source_files: List[str] = None) -> OptimizationResult:
        """
        Analyze a request for RAG2DAG optimization opportunities.

        This is the primary service method for optimization analysis.

        Args:
            request: The user request/query to analyze
            context: Additional context about the request
            source_files: List of source files involved

        Returns:
            OptimizationResult with detailed analysis and recommendations
        """
        self.optimization_stats["total_analyzed"] += 1

        try:
            # Step 1: Pattern detection
            pattern_result = self.pattern_service.detect_pattern(
                request=request,
                context=context,
                source_files=source_files or []
            )

            if not pattern_result.pattern_detected:
                return OptimizationResult(
                    should_optimize=False,
                    pattern_detected=None,
                    estimated_speedup=1.0,
                    estimated_cost_factor=1.0,
                    dag_workflow=None,
                    optimization_reason="No optimizable RAG pattern detected",
                    confidence_score=pattern_result.confidence
                )

            # Step 2: Optimization potential analysis
            optimization_analysis = self._analyze_optimization_potential(
                pattern_result.pattern_detected,
                request,
                source_files or [],
                pattern_result.confidence
            )

            # Step 3: Generate DAG workflow if beneficial
            dag_workflow = None
            if optimization_analysis["should_optimize"]:
                dag_workflow = self._generate_dag_workflow(
                    pattern_result.pattern_detected,
                    request,
                    context
                )

                # Update statistics
                self.optimization_stats["optimizations_applied"] += 1
                self.optimization_stats["patterns_detected"][pattern_result.pattern_detected.value] = \
                    self.optimization_stats["patterns_detected"].get(pattern_result.pattern_detected.value, 0) + 1

            return OptimizationResult(
                should_optimize=optimization_analysis["should_optimize"],
                pattern_detected=pattern_result.pattern_detected,
                estimated_speedup=optimization_analysis["speedup"],
                estimated_cost_factor=optimization_analysis["cost_factor"],
                dag_workflow=dag_workflow,
                optimization_reason=optimization_analysis["reason"],
                confidence_score=pattern_result.confidence
            )

        except Exception as e:
            logger.error(f"RAG2DAG optimization analysis failed: {e}")
            return OptimizationResult(
                should_optimize=False,
                pattern_detected=None,
                estimated_speedup=1.0,
                estimated_cost_factor=1.0,
                dag_workflow=None,
                optimization_reason=f"Analysis error: {str(e)}",
                confidence_score=0.0
            )

    def suggest_workflow_optimization(self, workflow_description: str, expected_load: str = "medium") -> OptimizationSuggestion:
        """
        Suggest optimizations for workflow designs.

        Args:
            workflow_description: Description of the workflow to optimize
            expected_load: Expected load level ("low", "medium", "high")

        Returns:
            OptimizationSuggestion with specific recommendations
        """
        try:
            # Use pattern service for analysis
            analysis = self.pattern_service.analyze_workflow_description(
                workflow_description,
                expected_load
            )

            # Generate DAG preview if optimization is beneficial
            dag_preview = {}
            if analysis["performance_gain"] > 20:
                pattern_type = RAGPatternType(analysis.get("optimization_type", "multi_source"))
                dag_preview = self._create_dag_preview(pattern_type, workflow_description)

            return OptimizationSuggestion(
                performance_gain=analysis["performance_gain"],
                cost_impact=analysis["cost_impact"],
                complexity_reduction=analysis["complexity_reduction"],
                suggested_changes=analysis["parallel_opportunities"],
                dag_preview=dag_preview,
                confidence_score=analysis.get("confidence", 0.5)
            )

        except Exception as e:
            logger.error(f"Workflow optimization suggestion failed: {e}")
            return OptimizationSuggestion(
                performance_gain=0,
                cost_impact="neutral",
                complexity_reduction=False,
                suggested_changes=[],
                dag_preview={},
                confidence_score=0.0
            )

    def execute_optimized_workflow(self, dag_workflow: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an optimized DAG workflow.

        Args:
            dag_workflow: The DAG workflow definition
            input_data: Input data for the workflow

        Returns:
            Execution results with performance metrics
        """
        try:
            # Delegate to execution service
            result = self.execution_service.execute_dag_workflow(dag_workflow, input_data)

            # Update performance statistics
            if "execution_time" in result and "speedup_factor" in result:
                current_avg = self.optimization_stats["avg_speedup"]
                total_optimizations = self.optimization_stats["optimizations_applied"]

                if total_optimizations > 0:
                    new_avg = ((current_avg * (total_optimizations - 1)) + result["speedup_factor"]) / total_optimizations
                    self.optimization_stats["avg_speedup"] = new_avg

            return result

        except Exception as e:
            logger.error(f"Optimized workflow execution failed: {e}")
            return {"error": str(e), "success": False}

    def get_service_stats(self) -> Dict[str, Any]:
        """Get service performance statistics and metrics."""
        uptime = datetime.now() - self.optimization_stats["service_uptime"]

        return {
            **self.optimization_stats.copy(),
            "service_uptime_hours": uptime.total_seconds() / 3600,
            "optimization_rate": (
                self.optimization_stats["optimizations_applied"] /
                max(self.optimization_stats["total_analyzed"], 1) * 100
            ),
            "thresholds": self.optimization_thresholds.copy(),
            "service_health": "healthy" if self.optimization_stats["total_analyzed"] > 0 else "initializing"
        }

    def update_optimization_thresholds(self, thresholds: Dict[str, Any]) -> bool:
        """Update optimization thresholds for fine-tuning."""
        try:
            self.optimization_thresholds.update(thresholds)
            logger.info(f"Updated optimization thresholds: {thresholds}")
            return True
        except Exception as e:
            logger.error(f"Failed to update thresholds: {e}")
            return False

    # Private methods
    def _analyze_optimization_potential(self, pattern: RAGPatternType, request: str, source_files: List[str], confidence: float) -> Dict[str, Any]:
        """Analyze whether optimization would be beneficial."""
        pattern_def = self.converter.patterns.get(pattern)
        if not pattern_def:
            return {"should_optimize": False, "reason": "Unknown pattern type"}

        # Calculate optimization score
        file_count = len(source_files)
        complexity_score = pattern_def.complexity_score
        request_length = len(request.split())

        optimization_score = 0

        # File count factor
        if file_count >= self.optimization_thresholds["min_files_for_optimization"]:
            optimization_score += min(file_count, 10)  # Cap at 10

        # Complexity factor
        optimization_score += complexity_score

        # Request complexity factor
        if request_length > 50:
            optimization_score += 3
        elif request_length > 20:
            optimization_score += 1

        # Pattern-specific bonuses
        pattern_bonuses = {
            RAGPatternType.MULTI_SOURCE: 4,
            RAGPatternType.RESEARCH_SYNTHESIS: 3,
            RAGPatternType.COMPARATIVE_ANALYSIS: 3,
            RAGPatternType.FACT_CHECKING: 2,
            RAGPatternType.KNOWLEDGE_EXTRACTION: 2,
            RAGPatternType.DOCUMENT_PIPELINE: 1
        }
        optimization_score += pattern_bonuses.get(pattern, 0)

        # Confidence adjustment
        optimization_score *= confidence

        should_optimize = (
            optimization_score >= self.optimization_thresholds["min_complexity_score"] and
            confidence >= self.optimization_thresholds["confidence_threshold"]
        )

        if should_optimize:
            estimated_speedup = min(1.5 + (optimization_score * 0.2), 6.0)
            return {
                "should_optimize": True,
                "speedup": estimated_speedup,
                "cost_factor": pattern_def.estimated_cost_factor,
                "reason": f"High optimization potential (score: {optimization_score:.1f}, confidence: {confidence:.2f})"
            }
        else:
            return {
                "should_optimize": False,
                "speedup": 1.0,
                "cost_factor": 1.0,
                "reason": f"Insufficient optimization potential (score: {optimization_score:.1f}, confidence: {confidence:.2f})"
            }

    def _generate_dag_workflow(self, pattern: RAGPatternType, request: str, context: str) -> Dict[str, Any]:
        """Generate DAG workflow for the detected pattern."""
        try:
            dag_nodes = self.converter.generate_dag_from_pattern(pattern, request, [])

            workflow = {
                "pattern_type": pattern.value,
                "nodes": [],
                "dependencies": {},
                "parallel_groups": [],
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "service_version": "1.0.0",
                    "request_hash": hash(request) % 10000
                }
            }

            for node in dag_nodes:
                workflow["nodes"].append({
                    "node_id": node.node_id,
                    "operation": node.operation,
                    "instruction": node.instruction,
                    "model_config": {
                        "model_id": getattr(node.model_config, 'model_id', 'claude-3-5-sonnet'),
                        "max_tokens": getattr(node.model_config, 'max_tokens', 1000),
                        "temperature": getattr(node.model_config, 'temperature', 0.7)
                    },
                    "timeout_seconds": node.timeout_seconds,
                    "retry_attempts": node.retry_attempts
                })

                if node.input_from:
                    workflow["dependencies"][node.node_id] = node.input_from

                if node.parallel_group:
                    if node.parallel_group not in workflow["parallel_groups"]:
                        workflow["parallel_groups"].append(node.parallel_group)

            return workflow

        except Exception as e:
            logger.error(f"DAG workflow generation failed: {e}")
            return {}

    def _create_dag_preview(self, pattern_type: RAGPatternType, workflow_description: str) -> Dict[str, Any]:
        """Create a preview of what the DAG workflow would look like."""
        pattern_def = self.converter.patterns.get(pattern_type)
        if not pattern_def:
            return {}

        return {
            "pattern_type": pattern_type.value,
            "pattern_name": pattern_def.name,
            "estimated_nodes": len(pattern_def.node_template),
            "parallel_groups": len(pattern_def.parallel_groups),
            "estimated_speedup": f"{pattern_def.estimated_cost_factor:.1f}x",
            "complexity_score": pattern_def.complexity_score,
            "preview_operations": [node.get("operation", "unknown") for node in pattern_def.node_template[:5]]
        }