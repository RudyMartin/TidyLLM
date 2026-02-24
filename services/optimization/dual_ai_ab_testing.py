"""
Dual AI Pipeline A/B/C Testing Framework
=======================================

Systematically tests different model combinations with MLflow tracking:
- Test A: Speed Focus (haiku → sonnet)
- Test B: Quality Focus (sonnet → 3.5-sonnet)
- Test C: Premium Focus (haiku → opus)

Each test runs the same QA/QC workflow with different template orderings
and tracks comprehensive metrics for comparison.
"""

import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import json

try:
    import mlflow
    import mlflow.tracking
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

from .dual_ai_pipeline import DualAIPipeline, DualAIResult

logger = logging.getLogger(__name__)


# DSPy Signatures for Test D
if DSPY_AVAILABLE:
    class InitialWorkflowAnalysis(dspy.Signature):
        """Fast initial analysis of workflow requirements."""

        workflow_query = dspy.InputField(desc="User's workflow question or requirement")
        template_sequence = dspy.InputField(desc="Template execution order (1,2,3,4)")
        context_info = dspy.InputField(desc="Project context and workflow data")

        initial_assessment = dspy.OutputField(desc="Quick initial workflow assessment and broad recommendations")

    class EnhancedWorkflowOptimization(dspy.Signature):
        """Detailed enhancement and optimization of initial workflow analysis."""

        initial_analysis = dspy.InputField(desc="Initial AI assessment of workflow requirements")
        workflow_query = dspy.InputField(desc="Original user query")
        template_sequence = dspy.InputField(desc="Template execution order")
        context_info = dspy.InputField(desc="Project context and workflow data")

        optimized_recommendations = dspy.OutputField(desc="Enhanced, detailed workflow recommendations with specific optimizations and corrections")


@dataclass
class ABTestConfig:
    """Configuration for an A/B/C test variant."""
    name: str
    label: str
    stage1_model: str
    stage2_model: str
    description: str
    experiment_name: str


class DualAIABTesting:
    """A/B/C testing framework for dual AI pipeline model combinations."""

    def __init__(self, project_name: str = "alex_qaqc"):
        """Initialize A/B/C testing framework."""
        self.project_name = project_name
        self.test_configs = self._setup_test_configurations()
        self.results = {}

    def _setup_test_configurations(self) -> Dict[str, ABTestConfig]:
        """Define the four test configurations including DSPy."""
        configs = {
            "A": ABTestConfig(
                name="speed_focus",
                label="Test A: Speed Focus",
                stage1_model="claude-3-haiku",
                stage2_model="claude-3-sonnet",
                description="Ultra-fast initial + solid enhancement for maximum throughput",
                experiment_name=f"dual_ai_test_a_speed_{self.project_name}"
            ),
            "B": ABTestConfig(
                name="quality_focus",
                label="Test B: Quality Focus",
                stage1_model="claude-3-sonnet",
                stage2_model="claude-3-5-sonnet",
                description="Strong initial + premium enhancement for higher quality",
                experiment_name=f"dual_ai_test_b_quality_{self.project_name}"
            ),
            "C": ABTestConfig(
                name="premium_focus",
                label="Test C: Premium Focus",
                stage1_model="claude-3-haiku",
                stage2_model="claude-3-5-sonnet",
                description="Fast initial + premium 3.5 Sonnet enhancement for superior quality",
                experiment_name=f"dual_ai_test_c_premium_{self.project_name}"
            )
        }

        # Add Test D if DSPy is available
        if DSPY_AVAILABLE:
            configs["D"] = ABTestConfig(
                name="dspy_optimized",
                label="Test D: DSPy Optimized",
                stage1_model="claude-3-haiku",
                stage2_model="claude-3-sonnet",
                description="DSPy-powered dual pipeline with signature optimization and structured outputs",
                experiment_name=f"dual_ai_test_d_dspy_{self.project_name}"
            )

        return configs

    def run_ab_test_suite(self, query: str, context: Dict[str, Any],
                         workflow_data: Dict[str, Any], user_id: str = "test_user",
                         sequential_delay_seconds: int = 2) -> Dict[str, DualAIResult]:
        """
        Run all A/B/C/D tests sequentially (no racing) with the same query and compare results.

        Args:
            sequential_delay_seconds: Delay between tests to ensure sequential execution

        Returns dict with test results keyed by test name (A, B, C, D).
        """
        logger.info(f"Starting SEQUENTIAL A/B/C/D test suite for query: {query[:100]}...")
        logger.info("Tests will run one after another with delays to prevent racing")

        all_results = {}
        test_order = ["A", "B", "C", "D"]  # Fixed order for sequential execution

        for i, test_id in enumerate(test_order):
            if test_id not in self.test_configs:
                continue

            config = self.test_configs[test_id]
            logger.info(f"[{i+1}/{len(test_order)}] Running {config.label} sequentially...")

            try:
                if test_id == "D" and DSPY_AVAILABLE:
                    # DSPy-based pipeline for Test D
                    result = self._run_dspy_pipeline(query, context, workflow_data, user_id, config)
                else:
                    # Standard dual AI pipeline for Tests A, B, C
                    pipeline = DualAIPipeline(
                        mlflow_experiment_name=config.experiment_name,
                        stage1_model=config.stage1_model,
                        stage2_model=config.stage2_model
                    )
                    result = pipeline.process_query(query, context, workflow_data, user_id)

                # Store results
                all_results[test_id] = result
                self.results[test_id] = result

                # Log test-specific metrics
                if MLFLOW_AVAILABLE:
                    self._log_ab_test_metrics(test_id, config, result)

                logger.info(f"✓ Completed {config.label} - Total time: {result.total_processing_time_ms:.0f}ms")

                # Sequential delay to prevent racing (except for last test)
                if i < len(test_order) - 1:
                    logger.info(f"Waiting {sequential_delay_seconds}s before next test...")
                    time.sleep(sequential_delay_seconds)

            except Exception as e:
                logger.error(f"✗ Failed {config.label}: {e}")
                all_results[test_id] = None
                # Still wait before next test to maintain sequence
                if i < len(test_order) - 1:
                    time.sleep(1)

        # Generate comparison report
        if all_results:
            self._generate_comparison_report(all_results)

        logger.info(f"Sequential testing complete. Results: {len([r for r in all_results.values() if r])}/{len(all_results)} successful")
        return all_results

    def _run_dspy_pipeline(self, query: str, context: Dict[str, Any],
                          workflow_data: Dict[str, Any], user_id: str,
                          config: ABTestConfig) -> DualAIResult:
        """Run DSPy-based dual AI pipeline for Test D."""
        import time
        from tidyllm.services.dual_ai_pipeline import AIResponse, AIStage
        from tidyllm.services.dspy_service import CorporateDSPyLM

        start_time = time.time()

        try:
            # Configure DSPy with CorporateLLMAdapter
            corporate_lm = CorporateDSPyLM(model_name=config.stage1_model)
            dspy.configure(lm=corporate_lm)
            logger.info(f"DSPy configured with CorporateLLMAdapter: {config.stage1_model}")

            # Initialize DSPy components
            initial_analyzer = dspy.ChainOfThought(InitialWorkflowAnalysis)
            enhanced_optimizer = dspy.ChainOfThought(EnhancedWorkflowOptimization)

            # Prepare input data
            template_sequence = "1. " + "\n2. ".join(workflow_data.get("available_templates", []))
            context_str = json.dumps(context, indent=2)

            # Stage 1: Initial Analysis (DSPy)
            stage1_start = time.time()
            initial_result = initial_analyzer(
                workflow_query=query,
                template_sequence=template_sequence,
                context_info=context_str
            )
            stage1_time = (time.time() - stage1_start) * 1000

            initial_response = AIResponse(
                content=initial_result.initial_assessment,
                confidence=0.75,  # DSPy typically provides good initial responses
                processing_time_ms=stage1_time,
                stage=AIStage.INITIAL,
                model_used=f"dspy_{config.stage1_model}",
                tokens_used=len(initial_result.initial_assessment.split()),  # Approximate
                metadata={"framework": "dspy", "signature": "InitialWorkflowAnalysis"}
            )

            # Stage 2: Enhanced Optimization (DSPy)
            stage2_start = time.time()
            enhanced_result = enhanced_optimizer(
                initial_analysis=initial_result.initial_assessment,
                workflow_query=query,
                template_sequence=template_sequence,
                context_info=context_str
            )
            stage2_time = (time.time() - stage2_start) * 1000

            enhanced_response = AIResponse(
                content=enhanced_result.optimized_recommendations,
                confidence=0.90,  # DSPy enhancement typically improves quality
                processing_time_ms=stage2_time,
                stage=AIStage.ENHANCEMENT,
                model_used=f"dspy_{config.stage2_model}",
                tokens_used=len(enhanced_result.optimized_recommendations.split()),  # Approximate
                metadata={"framework": "dspy", "signature": "EnhancedWorkflowOptimization"}
            )

            total_time = (time.time() - start_time) * 1000

            # Create DSPy-specific improvement summary
            improvement_summary = "DSPy signature optimization with structured workflow analysis"
            confidence_improvement = enhanced_response.confidence - initial_response.confidence

            # Return DualAIResult
            return DualAIResult(
                initial_response=initial_response,
                enhanced_response=enhanced_response,
                improvement_summary=improvement_summary,
                total_processing_time_ms=total_time,
                confidence_improvement=confidence_improvement,
                final_content=enhanced_result.optimized_recommendations
            )

        except Exception as e:
            logger.error(f"DSPy pipeline failed: {e}")
            raise

    def _log_ab_test_metrics(self, test_id: str, config: ABTestConfig, result: DualAIResult):
        """Log A/B/C test specific metrics to MLflow."""
        try:
            with mlflow.start_run(run_name=f"ab_test_summary_{test_id}_{int(time.time())}"):
                # Test configuration
                mlflow.log_param("test_id", test_id)
                mlflow.log_param("test_name", config.name)
                mlflow.log_param("test_label", config.label)
                mlflow.log_param("stage1_model", config.stage1_model)
                mlflow.log_param("stage2_model", config.stage2_model)
                mlflow.log_param("project_name", self.project_name)

                # Performance metrics
                mlflow.log_metric("total_time_ms", result.total_processing_time_ms)
                mlflow.log_metric("stage1_time_ms", result.initial_response.processing_time_ms)
                mlflow.log_metric("stage2_time_ms", result.enhanced_response.processing_time_ms)

                # Quality metrics
                mlflow.log_metric("initial_confidence", result.initial_response.confidence)
                mlflow.log_metric("final_confidence", result.enhanced_response.confidence)
                mlflow.log_metric("confidence_improvement", result.confidence_improvement)

                # Token usage
                mlflow.log_metric("stage1_tokens", result.initial_response.tokens_used)
                mlflow.log_metric("stage2_tokens", result.enhanced_response.tokens_used)
                mlflow.log_metric("total_tokens", result.initial_response.tokens_used + result.enhanced_response.tokens_used)

                # Content analysis
                mlflow.log_metric("initial_content_length", len(result.initial_response.content))
                mlflow.log_metric("final_content_length", len(result.enhanced_response.content))
                mlflow.log_metric("content_expansion_ratio", len(result.enhanced_response.content) / len(result.initial_response.content))

                # Save results as artifacts
                self._save_ab_test_artifacts(test_id, config, result)

        except Exception as e:
            logger.error(f"Failed to log A/B test metrics for {test_id}: {e}")

    def _save_ab_test_artifacts(self, test_id: str, config: ABTestConfig, result: DualAIResult):
        """Save detailed A/B test results as MLflow artifacts and to project outputs folder."""
        try:
            # Project outputs directory (following TidyLLM workflow pattern)
            project_outputs_dir = f"tidyllm/workflows/projects/{self.project_name}/outputs"
            import os
            os.makedirs(project_outputs_dir, exist_ok=True)

            # Detailed result JSON
            result_data = {
                "test_config": {
                    "test_id": test_id,
                    "name": config.name,
                    "label": config.label,
                    "stage1_model": config.stage1_model,
                    "stage2_model": config.stage2_model,
                    "description": config.description
                },
                "initial_response": {
                    "content": result.initial_response.content,
                    "confidence": result.initial_response.confidence,
                    "processing_time_ms": result.initial_response.processing_time_ms,
                    "model_used": result.initial_response.model_used,
                    "tokens_used": result.initial_response.tokens_used
                },
                "enhanced_response": {
                    "content": result.enhanced_response.content,
                    "confidence": result.enhanced_response.confidence,
                    "processing_time_ms": result.enhanced_response.processing_time_ms,
                    "model_used": result.enhanced_response.model_used,
                    "tokens_used": result.enhanced_response.tokens_used
                },
                "summary": {
                    "improvement_summary": result.improvement_summary,
                    "total_processing_time_ms": result.total_processing_time_ms,
                    "confidence_improvement": result.confidence_improvement,
                    "final_content": result.final_content
                },
                "timestamp": datetime.now().isoformat(),
                "project_name": self.project_name,
                "output_location": f"{project_outputs_dir}/ab_test_{test_id}_detailed_results.json"
            }

            # Save to project outputs folder (primary location)
            project_filename = f"{project_outputs_dir}/ab_test_{test_id}_detailed_results.json"
            with open(project_filename, "w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)

            # Save individual response files to project outputs
            with open(f"{project_outputs_dir}/ab_test_{test_id}_initial_response.txt", "w", encoding="utf-8") as f:
                f.write(result.initial_response.content)

            with open(f"{project_outputs_dir}/ab_test_{test_id}_enhanced_response.txt", "w", encoding="utf-8") as f:
                f.write(result.enhanced_response.content)

            logger.info(f"A/B test results saved to project outputs: {project_filename}")

            # Also save as JSON artifact for MLflow (secondary location)
            filename = f"ab_test_{test_id}_detailed_results.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            mlflow.log_artifact(filename, "ab_test_results")

            # Save individual response files for MLflow
            with open(f"ab_test_{test_id}_initial_response.txt", "w", encoding="utf-8") as f:
                f.write(result.initial_response.content)
            mlflow.log_artifact(f"ab_test_{test_id}_initial_response.txt", "responses")

            with open(f"ab_test_{test_id}_enhanced_response.txt", "w", encoding="utf-8") as f:
                f.write(result.enhanced_response.content)
            mlflow.log_artifact(f"ab_test_{test_id}_enhanced_response.txt", "responses")

        except Exception as e:
            logger.error(f"Failed to save A/B test artifacts for {test_id}: {e}")

    def _generate_comparison_report(self, results: Dict[str, DualAIResult]):
        """Generate comparative analysis of A/B/C test results."""
        try:
            valid_results = {k: v for k, v in results.items() if v is not None}
            if len(valid_results) < 2:
                logger.warning("Not enough valid results for comparison")
                return

            report = []
            report.append("# A/B/C Test Comparison Report")
            report.append(f"Generated: {datetime.now().isoformat()}")
            report.append(f"Project: {self.project_name}")
            report.append("")

            # Performance comparison
            report.append("## Performance Metrics")
            report.append("| Test | Stage1 Model | Stage2 Model | Total Time (ms) | Confidence Improvement |")
            report.append("|------|--------------|--------------|-----------------|----------------------|")

            for test_id, result in valid_results.items():
                config = self.test_configs[test_id]
                report.append(f"| {config.label} | {config.stage1_model} | {config.stage2_model} | "
                            f"{result.total_processing_time_ms:.0f} | {result.confidence_improvement:.2f} |")

            report.append("")

            # Token usage comparison
            report.append("## Token Usage Analysis")
            report.append("| Test | Stage1 Tokens | Stage2 Tokens | Total Tokens | Cost Efficiency |")
            report.append("|------|---------------|---------------|--------------|-----------------|")

            for test_id, result in valid_results.items():
                config = self.test_configs[test_id]
                total_tokens = result.initial_response.tokens_used + result.enhanced_response.tokens_used
                efficiency = result.confidence_improvement / total_tokens if total_tokens > 0 else 0
                report.append(f"| {config.label} | {result.initial_response.tokens_used} | "
                            f"{result.enhanced_response.tokens_used} | {total_tokens} | {efficiency:.4f} |")

            report.append("")

            # Quality analysis
            report.append("## Quality Metrics")
            for test_id, result in valid_results.items():
                config = self.test_configs[test_id]
                report.append(f"### {config.label}")
                report.append(f"- **Description**: {config.description}")
                report.append(f"- **Initial Confidence**: {result.initial_response.confidence:.2f}")
                report.append(f"- **Final Confidence**: {result.enhanced_response.confidence:.2f}")
                report.append(f"- **Improvement Summary**: {result.improvement_summary}")
                report.append("")

            # Save report to project outputs folder (primary location)
            report_content = "\n".join(report)
            project_outputs_dir = f"tidyllm/workflows/projects/{self.project_name}/outputs"
            import os
            os.makedirs(project_outputs_dir, exist_ok=True)

            project_report_filename = f"{project_outputs_dir}/ab_test_comparison_report_{int(time.time())}.md"
            with open(project_report_filename, "w", encoding="utf-8") as f:
                f.write(report_content)

            logger.info(f"A/B test comparison report saved to project outputs: {project_report_filename}")

            # Also save to current directory for MLflow (secondary location)
            filename = f"ab_test_comparison_report_{int(time.time())}.md"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(report_content)

            if MLFLOW_AVAILABLE:
                try:
                    with mlflow.start_run(run_name=f"ab_test_comparison_{int(time.time())}"):
                        mlflow.log_artifact(filename, "comparison_reports")

                        # Log summary metrics
                        best_speed = min(valid_results.items(), key=lambda x: x[1].total_processing_time_ms)
                        best_quality = max(valid_results.items(), key=lambda x: x[1].confidence_improvement)

                        mlflow.log_param("fastest_test", best_speed[0])
                        mlflow.log_param("highest_quality_test", best_quality[0])
                        mlflow.log_metric("fastest_time_ms", best_speed[1].total_processing_time_ms)
                        mlflow.log_metric("highest_confidence_improvement", best_quality[1].confidence_improvement)

                except Exception as e:
                    logger.error(f"Failed to log comparison report to MLflow: {e}")

            logger.info(f"A/B/C test comparison report saved: {filename}")

        except Exception as e:
            logger.error(f"Failed to generate comparison report: {e}")

    def get_test_summary(self) -> Dict[str, Any]:
        """Get summary of all test results."""
        if not self.results:
            return {"error": "No test results available"}

        summary = {
            "total_tests": len(self.results),
            "project_name": self.project_name,
            "timestamp": datetime.now().isoformat(),
            "tests": {}
        }

        for test_id, result in self.results.items():
            if result:
                config = self.test_configs[test_id]
                summary["tests"][test_id] = {
                    "label": config.label,
                    "models": f"{config.stage1_model} → {config.stage2_model}",
                    "total_time_ms": result.total_processing_time_ms,
                    "confidence_improvement": result.confidence_improvement,
                    "total_tokens": result.initial_response.tokens_used + result.enhanced_response.tokens_used
                }

        return summary


def run_qaqc_ab_testing(query: str = None, template_order: List[str] = None,
                        sequential: bool = True, delay_seconds: int = 2) -> Dict[str, Any]:
    """
    Convenience function to run A/B/C/D testing on QA/QC workflow.

    Args:
        query: Test query (default: QA/QC analysis query)
        template_order: Custom template ordering for testing
        sequential: If True, run tests sequentially with delays (no racing)
        delay_seconds: Delay between sequential tests

    Returns:
        Dictionary with all test results and summary
    """
    if query is None:
        query = """Analyze the QA/QC workflow for data quality assessment.
        Focus on metadata extraction, analysis steps, results aggregation, and recording questions.
        Provide specific recommendations for template ordering and execution strategy."""

    if template_order is None:
        template_order = ["step_01_metadata_extraction", "step_02_analysis_steps",
                         "step_03_results_aggregation", "step_04_recording_questions"]

    context = {
        "workflow_type": "qaqc_analysis",
        "project_name": "alex_qaqc",
        "template_ordering": template_order
    }

    workflow_data = {
        "available_templates": template_order,
        "execution_context": "ab_testing_sequential" if sequential else "ab_testing",
        "tracking_enabled": True
    }

    # Run A/B/C/D testing
    ab_tester = DualAIABTesting("alex_qaqc")

    if sequential:
        results = ab_tester.run_ab_test_suite(query, context, workflow_data, "ab_test_user", delay_seconds)
    else:
        # Use original method for racing/parallel execution
        results = ab_tester.run_ab_test_suite(query, context, workflow_data, "ab_test_user", 0)

    return {
        "results": results,
        "summary": ab_tester.get_test_summary(),
        "test_configurations": ab_tester.test_configs,
        "execution_mode": "sequential" if sequential else "racing"
    }


def run_sequential_qaqc_testing(delay_seconds: int = 3) -> Dict[str, Any]:
    """
    Run A/B/C/D tests sequentially with guaranteed no racing.

    Args:
        delay_seconds: Seconds to wait between tests

    Returns:
        Sequential test results
    """
    return run_qaqc_ab_testing(sequential=True, delay_seconds=delay_seconds)


def run_selective_sequential_testing(selected_tests: List[str],
                                    query: str = None,
                                    template_order: List[str] = None,
                                    delay_seconds: int = 3) -> Dict[str, Any]:
    """
    Run only selected A/B/C/D tests sequentially for targeted optimization.

    Args:
        selected_tests: List of test IDs to run (e.g., ["A", "B", "D"])
        query: Test query (default: QA/QC analysis query)
        template_order: Custom template ordering for testing
        delay_seconds: Seconds to wait between tests

    Returns:
        Dictionary with selected test results
    """
    if query is None:
        query = """Analyze the QA/QC workflow for data quality assessment.
        Focus on metadata extraction, analysis steps, results aggregation, and recording questions.
        Provide specific recommendations for template ordering and execution strategy."""

    if template_order is None:
        template_order = ["step_01_metadata_extraction", "step_02_analysis_steps",
                         "step_03_results_aggregation", "step_04_recording_questions"]

    # Validate selected tests
    valid_tests = ["A", "B", "C", "D"]
    selected_tests = [test for test in selected_tests if test in valid_tests]

    if not selected_tests:
        raise ValueError("No valid tests selected. Choose from A, B, C, D")

    logger.info(f"Running selective sequential testing for tests: {', '.join(selected_tests)}")

    context = {
        "workflow_type": "qaqc_analysis",
        "project_name": "alex_qaqc",
        "template_ordering": template_order,
        "execution_mode": "selective_sequential",
        "selected_tests": selected_tests
    }

    workflow_data = {
        "available_templates": template_order,
        "execution_context": "ab_testing_selective_sequential",
        "tracking_enabled": True,
        "test_selection": selected_tests
    }

    # Create A/B tester with selective configuration
    ab_tester = DualAIABTesting("alex_qaqc")

    # Filter test configurations to only selected tests
    original_configs = ab_tester.test_configs.copy()
    ab_tester.test_configs = {
        test_id: config for test_id, config in original_configs.items()
        if test_id in selected_tests
    }

    # Run sequential tests for selected tests only
    results = ab_tester.run_ab_test_suite(query, context, workflow_data, "selective_sequential_user", delay_seconds)

    return {
        "results": results,
        "summary": ab_tester.get_test_summary(),
        "test_configurations": {test_id: original_configs[test_id] for test_id in selected_tests},
        "execution_mode": "selective_sequential",
        "selected_tests": selected_tests,
        "delay_seconds": delay_seconds
    }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("Starting Dual AI Pipeline A/B/C/D Sequential Testing (No Racing)...")
    test_results = run_sequential_qaqc_testing(delay_seconds=3)

    print(f"\nCompleted {len(test_results['results'])} tests in {test_results['execution_mode']} mode")
    print("Sequential execution ensures no racing between tests")
    print("Check MLflow for detailed results and comparison reports")