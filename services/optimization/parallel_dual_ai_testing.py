"""
Parallel Dual AI Pipeline A/B/C/D Testing Framework
==================================================

Uses existing TidyLLM parallelization infrastructure to run A/B/C/D tests concurrently
instead of sequentially. Leverages BaseWorker patterns and async execution for performance.
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import json
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

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
from .dual_ai_ab_testing import ABTestConfig, DualAIABTesting

# Import existing parallelization infrastructure
from tidyllm.infrastructure.workers.base_worker import BaseWorker, WorkerTask, TaskPriority, WorkerMetrics

logger = logging.getLogger(__name__)


@dataclass
class ParallelTestTask:
    """Task configuration for parallel A/B testing."""
    test_id: str
    config: ABTestConfig
    query: str
    context: Dict[str, Any]
    workflow_data: Dict[str, Any]
    user_id: str
    task_priority: TaskPriority = TaskPriority.NORMAL


class ParallelDualAITester(BaseWorker):
    """
    Parallel A/B/C/D testing worker using existing TidyLLM infrastructure.

    Executes multiple test configurations concurrently for performance comparison.
    """

    def __init__(self,
                 project_name: str = "alex_qaqc",
                 max_concurrent_tests: int = 4,
                 worker_name: str = "parallel_ab_tester"):
        """
        Initialize parallel A/B testing worker.

        Args:
            project_name: Project identifier for tracking
            max_concurrent_tests: Maximum tests to run in parallel
            worker_name: Worker identifier
        """
        super().__init__(
            worker_name=worker_name,
            max_queue_size=20,
            max_concurrent_tasks=max_concurrent_tests,
            task_timeout=600.0  # 10 minutes for comprehensive tests
        )

        self.project_name = project_name
        self.sequential_tester = DualAIABTesting(project_name)
        self.test_results = {}
        self.start_time = None

        logger.info(f"Parallel A/B Tester initialized for {max_concurrent_tests} concurrent tests")

    async def process_task(self, task_input) -> Any:
        """Process a single test task (BaseWorker abstract method)."""
        if isinstance(task_input, ParallelTestTask):
            return self._execute_single_test(task_input)
        else:
            raise ValueError(f"Unsupported task input type: {type(task_input)}")

    def validate_input(self, task_input: Any) -> bool:
        """Validate task input data (BaseWorker abstract method)."""
        return isinstance(task_input, ParallelTestTask)

    async def run_parallel_ab_test_suite(self,
                                       query: str,
                                       context: Dict[str, Any],
                                       workflow_data: Dict[str, Any],
                                       user_id: str = "parallel_test_user") -> Dict[str, DualAIResult]:
        """
        Run all A/B/C/D tests in parallel using worker infrastructure.

        Args:
            query: Test query for all configurations
            context: Test context data
            workflow_data: Workflow configuration
            user_id: User identifier for tracking

        Returns:
            Dictionary with test results keyed by test ID
        """
        self.start_time = time.time()
        logger.info(f"Starting PARALLEL A/B/C/D test suite for query: {query[:100]}...")
        logger.info("Tests will run concurrently using TidyLLM worker infrastructure")

        # Create parallel test tasks
        test_tasks = []
        for test_id, config in self.sequential_tester.test_configs.items():
            task = ParallelTestTask(
                test_id=test_id,
                config=config,
                query=query,
                context=context,
                workflow_data=workflow_data,
                user_id=user_id,
                task_priority=TaskPriority.NORMAL
            )
            test_tasks.append(task)

        # Execute tests in parallel using ThreadPoolExecutor
        all_results = await self._execute_parallel_tests(test_tasks)

        # Generate comparison report
        if all_results:
            self._generate_parallel_comparison_report(all_results)

        total_time = time.time() - self.start_time
        logger.info(f"Parallel testing complete in {total_time:.2f}s. Results: {len([r for r in all_results.values() if r])}/{len(all_results)} successful")

        return all_results

    async def _execute_parallel_tests(self, test_tasks: List[ParallelTestTask]) -> Dict[str, DualAIResult]:
        """Execute tests in parallel using concurrent.futures."""
        all_results = {}

        # Use ThreadPoolExecutor for true parallelism of AI calls
        with ThreadPoolExecutor(max_workers=self.max_concurrent_tasks) as executor:
            # Submit all test tasks
            future_to_test = {}
            for task in test_tasks:
                future = executor.submit(self._execute_single_test, task)
                future_to_test[future] = task.test_id

            logger.info(f"Submitted {len(test_tasks)} tests for parallel execution")

            # Collect results as they complete
            for future in as_completed(future_to_test):
                test_id = future_to_test[future]
                try:
                    result = future.result()
                    all_results[test_id] = result
                    if result:
                        logger.info(f"✓ Completed Test {test_id} - Total time: {result.total_processing_time_ms:.0f}ms")
                    else:
                        logger.error(f"✗ Failed Test {test_id} - No result returned")
                except Exception as e:
                    logger.error(f"✗ Failed Test {test_id}: {e}")
                    all_results[test_id] = None

        return all_results

    def _execute_single_test(self, task: ParallelTestTask) -> Optional[DualAIResult]:
        """
        Execute a single A/B test configuration.

        Args:
            task: Test task configuration

        Returns:
            Test result or None if failed
        """
        test_start = time.time()
        logger.info(f"[PARALLEL] Starting {task.config.label}...")

        try:
            if task.test_id == "D" and DSPY_AVAILABLE:
                # DSPy-based pipeline for Test D
                result = self.sequential_tester._run_dspy_pipeline(
                    task.query, task.context, task.workflow_data, task.user_id, task.config
                )
            else:
                # Standard dual AI pipeline for Tests A, B, C
                pipeline = DualAIPipeline(
                    mlflow_experiment_name=task.config.experiment_name,
                    stage1_model=task.config.stage1_model,
                    stage2_model=task.config.stage2_model
                )
                result = pipeline.process_query(
                    task.query, task.context, task.workflow_data, task.user_id
                )

            # Log test-specific metrics in parallel
            if MLFLOW_AVAILABLE and result:
                self.sequential_tester._log_ab_test_metrics(task.test_id, task.config, result)

            test_time = time.time() - test_start
            logger.info(f"[PARALLEL] ✓ Completed {task.config.label} in {test_time:.2f}s")

            return result

        except Exception as e:
            test_time = time.time() - test_start
            logger.error(f"[PARALLEL] ✗ Failed {task.config.label} after {test_time:.2f}s: {e}")
            return None

    def _generate_parallel_comparison_report(self, results: Dict[str, DualAIResult]):
        """Generate comparison report highlighting parallel execution benefits."""
        try:
            valid_results = {k: v for k, v in results.items() if v is not None}
            if len(valid_results) < 2:
                logger.warning("Not enough valid results for parallel comparison")
                return

            total_parallel_time = time.time() - self.start_time

            # Calculate what sequential time would have been
            total_sequential_time = sum(
                result.total_processing_time_ms / 1000
                for result in valid_results.values()
            )

            speedup_factor = total_sequential_time / total_parallel_time if total_parallel_time > 0 else 1.0

            report = []
            report.append("# Parallel A/B/C/D Test Comparison Report")
            report.append(f"Generated: {datetime.now().isoformat()}")
            report.append(f"Project: {self.project_name}")
            report.append(f"Execution Mode: PARALLEL")
            report.append("")

            # Parallel performance summary
            report.append("## Parallel Execution Performance")
            report.append(f"- **Total Parallel Execution Time**: {total_parallel_time:.2f}s")
            report.append(f"- **Estimated Sequential Time**: {total_sequential_time:.2f}s")
            report.append(f"- **Parallel Speedup Factor**: {speedup_factor:.2f}x")
            report.append(f"- **Time Saved**: {total_sequential_time - total_parallel_time:.2f}s")
            report.append("")

            # Performance comparison table
            report.append("## Individual Test Performance")
            report.append("| Test | Stage1 Model | Stage2 Model | Total Time (ms) | Confidence Improvement | Parallel Efficiency |")
            report.append("|------|--------------|--------------|-----------------|----------------------|-------------------|")

            for test_id, result in valid_results.items():
                config = self.sequential_tester.test_configs[test_id]
                efficiency = (result.total_processing_time_ms / 1000) / total_parallel_time
                report.append(f"| {config.label} | {config.stage1_model} | {config.stage2_model} | "
                            f"{result.total_processing_time_ms:.0f} | {result.confidence_improvement:.2f} | {efficiency:.2f} |")

            report.append("")

            # Quality vs Speed analysis
            report.append("## Quality vs Speed vs Parallel Efficiency Analysis")
            for test_id, result in valid_results.items():
                config = self.sequential_tester.test_configs[test_id]
                report.append(f"### {config.label}")
                report.append(f"- **Description**: {config.description}")
                report.append(f"- **Parallel Processing Time**: {result.total_processing_time_ms/1000:.2f}s")
                report.append(f"- **Quality Improvement**: {result.confidence_improvement:.2f}")
                report.append(f"- **Efficiency Score**: {result.confidence_improvement / (result.total_processing_time_ms/1000):.3f}")
                report.append("")

            # Parallel execution insights
            report.append("## Parallel Execution Insights")
            report.append(f"- **Concurrency**: {self.max_concurrent_tasks} tests executed simultaneously")
            report.append(f"- **Infrastructure**: TidyLLM BaseWorker + ThreadPoolExecutor")
            report.append(f"- **Resource Efficiency**: {speedup_factor:.1f}x faster than sequential execution")
            report.append(f"- **Throughput Gain**: {((speedup_factor - 1) * 100):.1f}% improvement")

            # Save report to project outputs folder (primary location)
            report_content = "\n".join(report)
            project_outputs_dir = f"tidyllm/workflows/projects/{self.project_name}/outputs"
            import os
            os.makedirs(project_outputs_dir, exist_ok=True)

            project_report_filename = f"{project_outputs_dir}/parallel_ab_test_comparison_report_{int(time.time())}.md"
            with open(project_report_filename, "w", encoding="utf-8") as f:
                f.write(report_content)

            logger.info(f"Parallel A/B test comparison report saved to project outputs: {project_report_filename}")

            # Also save to current directory for MLflow (secondary location)
            filename = f"parallel_ab_test_comparison_report_{int(time.time())}.md"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(report_content)

            if MLFLOW_AVAILABLE:
                try:
                    with mlflow.start_run(run_name=f"parallel_ab_test_comparison_{int(time.time())}"):
                        mlflow.log_artifact(filename, "parallel_comparison_reports")

                        # Log parallel performance metrics
                        mlflow.log_metric("parallel_execution_time_seconds", total_parallel_time)
                        mlflow.log_metric("sequential_time_estimate_seconds", total_sequential_time)
                        mlflow.log_metric("parallel_speedup_factor", speedup_factor)
                        mlflow.log_metric("concurrent_tests_executed", len(valid_results))
                        mlflow.log_param("execution_mode", "parallel")
                        mlflow.log_param("max_concurrent_tasks", self.max_concurrent_tasks)

                except Exception as e:
                    logger.error(f"Failed to log parallel comparison report to MLflow: {e}")

            logger.info(f"Parallel A/B/C/D test comparison report saved: {filename}")
            logger.info(f"Parallel speedup achieved: {speedup_factor:.2f}x faster than sequential")

        except Exception as e:
            logger.error(f"Failed to generate parallel comparison report: {e}")


# Convenience functions for parallel testing

def run_parallel_qaqc_ab_testing(query: str = None, template_order: List[str] = None,
                                max_concurrent_tests: int = 4) -> Dict[str, Any]:
    """
    Run A/B/C/D tests in parallel for maximum performance.

    Args:
        query: Test query (default: QA/QC analysis query)
        template_order: Custom template ordering for testing
        max_concurrent_tests: Maximum concurrent test execution

    Returns:
        Dictionary with all test results and parallel performance metrics
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
        "template_ordering": template_order,
        "execution_mode": "parallel"
    }

    workflow_data = {
        "available_templates": template_order,
        "execution_context": "ab_testing_parallel",
        "tracking_enabled": True,
        "parallel_optimization": True
    }

    # Create parallel tester
    parallel_tester = ParallelDualAITester("alex_qaqc", max_concurrent_tests)

    # Run parallel tests using asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        results = loop.run_until_complete(
            parallel_tester.run_parallel_ab_test_suite(query, context, workflow_data, "parallel_test_user")
        )
    finally:
        loop.close()

    return {
        "results": results,
        "summary": parallel_tester.sequential_tester.get_test_summary(),
        "test_configurations": parallel_tester.sequential_tester.test_configs,
        "execution_mode": "parallel",
        "max_concurrent_tests": max_concurrent_tests,
        "parallel_infrastructure": "TidyLLM BaseWorker + ThreadPoolExecutor"
    }


def run_selective_parallel_testing(selected_tests: List[str],
                                   query: str = None,
                                   template_order: List[str] = None,
                                   max_concurrent_tests: int = 4,
                                   status_callback: callable = None) -> Dict[str, Any]:
    """
    Run only selected A/B/C/D tests in parallel for targeted optimization.

    Args:
        selected_tests: List of test IDs to run (e.g., ["A", "B", "D"])
        query: Test query (default: QA/QC analysis query)
        template_order: Custom template ordering for testing
        max_concurrent_tests: Maximum concurrent test execution

    Returns:
        Dictionary with selected test results and parallel performance metrics
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

    # Import monitor for status tracking
    from .test_execution_monitor import TestExecutionMonitor, TestStatus

    # Initialize execution monitor
    monitor = TestExecutionMonitor(
        selected_tests=selected_tests,
        execution_mode="parallel",
        timeout_seconds=300,
        status_callback=status_callback
    )
    monitor.start_monitoring()
    monitor.print_status_summary()

    logger.info(f"Running selective parallel testing for tests: {', '.join(selected_tests)}")

    context = {
        "workflow_type": "qaqc_analysis",
        "project_name": "alex_qaqc",
        "template_ordering": template_order,
        "execution_mode": "selective_parallel",
        "selected_tests": selected_tests
    }

    workflow_data = {
        "available_templates": template_order,
        "execution_context": "ab_testing_selective_parallel",
        "tracking_enabled": True,
        "parallel_optimization": True,
        "test_selection": selected_tests
    }

    # Create parallel tester with selective configuration
    parallel_tester = ParallelDualAITester("alex_qaqc", min(max_concurrent_tests, len(selected_tests)))

    # Filter test configurations to only selected tests
    original_configs = parallel_tester.sequential_tester.test_configs.copy()
    parallel_tester.sequential_tester.test_configs = {
        test_id: config for test_id, config in original_configs.items()
        if test_id in selected_tests
    }

    # Run parallel tests using asyncio with monitoring
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # Update monitor status before starting tests
        for test_id in selected_tests:
            monitor.update_test_status(test_id, TestStatus.STARTING)

        results = loop.run_until_complete(
            parallel_tester.run_parallel_ab_test_suite(query, context, workflow_data, "selective_parallel_user")
        )

        # Update monitor with results
        for test_id in selected_tests:
            if test_id in results and results[test_id] is not None:
                monitor.update_test_status(test_id, TestStatus.COMPLETED, result=results[test_id])
            else:
                monitor.update_test_status(test_id, TestStatus.FAILED, error="Test execution failed")

    except Exception as e:
        logger.error(f"Parallel testing failed: {e}")
        # Log system error with full exception details
        monitor.log_system_error("parallel_test_execution", f"Parallel testing failed: {str(e)}", e)
        # Mark all tests as failed
        for test_id in selected_tests:
            monitor.update_test_status(test_id, TestStatus.FAILED, error=str(e), exception=e)
        results = {}
    finally:
        loop.close()

    # Print final status
    monitor.print_status_summary()

    # Generate executive report if tests completed successfully
    executive_report_data = None
    if monitor.should_generate_report():
        try:
            from .test_execution_monitor import create_monitored_test_wrapper
            from tidyllm.workflows.templates.test.executive_report_generator import ExecutiveReportGenerator

            logger.info("Generating executive performance report...")
            report_data = monitor.get_results_for_report()

            generator = ExecutiveReportGenerator("alex_qaqc")
            report_path = generator.generate_report(report_data)
            logger.info(f"Executive report generated: {report_path}")
            executive_report_data = {"report_path": report_path, "report_data": report_data}

        except Exception as e:
            logger.error(f"Failed to generate executive report: {e}")

    # Cleanup monitor
    monitor.stop_monitoring()

    return {
        "results": results,
        "summary": parallel_tester.sequential_tester.get_test_summary(),
        "test_configurations": {test_id: original_configs[test_id] for test_id in selected_tests},
        "execution_mode": "selective_parallel",
        "selected_tests": selected_tests,
        "max_concurrent_tests": min(max_concurrent_tests, len(selected_tests)),
        "parallel_infrastructure": "TidyLLM BaseWorker + ThreadPoolExecutor",
        "monitor_status": monitor.get_status_report(),
        "executive_report": executive_report_data
    }


def compare_sequential_vs_parallel_testing(query: str = None, template_order: List[str] = None) -> Dict[str, Any]:
    """
    Run both sequential and parallel testing to compare performance benefits.

    Returns:
        Comprehensive comparison of sequential vs parallel execution
    """
    logger.info("Starting Sequential vs Parallel A/B Testing Comparison...")

    # Import sequential testing
    from .dual_ai_ab_testing import run_sequential_qaqc_testing

    # Run sequential tests
    logger.info("Phase 1: Running Sequential Tests...")
    sequential_start = time.time()
    sequential_results = run_sequential_qaqc_testing(delay_seconds=1)
    sequential_time = time.time() - sequential_start

    # Run parallel tests
    logger.info("Phase 2: Running Parallel Tests...")
    parallel_start = time.time()
    parallel_results = run_parallel_qaqc_ab_testing(query, template_order, max_concurrent_tests=4)
    parallel_time = time.time() - parallel_start

    # Calculate performance comparison
    speedup = sequential_time / parallel_time if parallel_time > 0 else 1.0
    efficiency = ((speedup - 1) * 100) if speedup > 1 else 0

    comparison = {
        "sequential_results": sequential_results,
        "parallel_results": parallel_results,
        "performance_comparison": {
            "sequential_execution_time": sequential_time,
            "parallel_execution_time": parallel_time,
            "speedup_factor": speedup,
            "efficiency_improvement_percent": efficiency,
            "time_saved_seconds": sequential_time - parallel_time
        },
        "recommendation": "parallel" if speedup > 1.2 else "sequential"
    }

    logger.info(f"Performance Comparison Complete:")
    logger.info(f"  Sequential: {sequential_time:.2f}s")
    logger.info(f"  Parallel:   {parallel_time:.2f}s")
    logger.info(f"  Speedup:    {speedup:.2f}x")
    logger.info(f"  Efficiency: {efficiency:.1f}% improvement")

    return comparison


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("Starting Parallel A/B/C/D Testing with TidyLLM Infrastructure...")
    test_results = run_parallel_qaqc_ab_testing(max_concurrent_tests=4)

    print(f"\nCompleted {len(test_results['results'])} tests in {test_results['execution_mode']} mode")
    print(f"Parallel infrastructure: {test_results['parallel_infrastructure']}")
    print("Check MLflow for detailed results and parallel performance reports")