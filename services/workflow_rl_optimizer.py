"""
TidyLLM Workflow RL Optimization Functions
==========================================

Core TidyLLM functions for RL-enhanced workflow optimization.
Clean, focused functions that provide RL capabilities to any workflow.

Usage:
    from packages.tidyllm.services.workflow_rl_optimizer import (
        optimize_workflow_execution,
        calculate_step_reward,
        update_rl_factors
    )
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

try:
    # Import RL components
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent.parent))

    from domain.services.rl_factor_optimizer import RLFactorOptimizer
    from domain.services.cumulative_learning_pipeline import CumulativeLearningPipeline
    from domain.services.step_attributes import BaseStep
    RL_COMPONENTS_AVAILABLE = True
except ImportError:
    RL_COMPONENTS_AVAILABLE = False

logger = logging.getLogger(__name__)


def optimize_workflow_execution(
    project_id: str,
    workflow_config: Dict[str, Any],
    context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    TidyLLM function to optimize workflow execution with RL.

    Args:
        project_id: Project identifier
        workflow_config: Workflow configuration
        context: Execution context (files, params, etc.)

    Returns:
        Optimized execution plan with RL factors
    """
    if not RL_COMPONENTS_AVAILABLE:
        return {"error": "RL components not available", "fallback_mode": True}

    try:
        rl_optimizer = RLFactorOptimizer(project_id=project_id)

        # Get optimized factors for workflow
        optimization_plan = {
            "project_id": project_id,
            "optimization_timestamp": datetime.now().isoformat(),
            "rl_enabled": True,
            "optimized_steps": []
        }

        # Handle different workflow formats
        steps = []
        if 'steps' in workflow_config:
            steps_data = workflow_config['steps']
            if isinstance(steps_data, dict):
                steps = list(steps_data.values())
            elif isinstance(steps_data, list):
                steps = steps_data
        elif 'prompt_steps' in workflow_config:
            steps_data = workflow_config['prompt_steps']
            if isinstance(steps_data, dict):
                steps = list(steps_data.values())
            elif isinstance(steps_data, list):
                steps = steps_data

        for i, step_config in enumerate(steps):
            step_type = step_config.get('step_type', 'process')

            # Get RL factors for this step
            rl_factors = rl_optimizer.optimize_step_factors(
                step_config=step_config,
                performance_metrics=context.get('historical_data', {}) if context else {}
            )

            optimized_step = {
                **step_config,
                "rl_factors": rl_factors,
                "optimization_level": "high"
            }

            optimization_plan["optimized_steps"].append(optimized_step)

        return optimization_plan

    except Exception as e:
        logger.error(f"Workflow optimization failed: {e}")
        return {"error": str(e), "fallback_mode": True}


def calculate_step_reward(
    step_config: Dict[str, Any],
    execution_result: Dict[str, Any],
    execution_time: float,
    success: bool = True
) -> float:
    """
    TidyLLM function to calculate RL reward for step execution.

    Args:
        step_config: Step configuration
        execution_result: Execution result
        execution_time: Time taken for execution
        success: Whether execution was successful

    Returns:
        RL reward value
    """
    if not RL_COMPONENTS_AVAILABLE:
        return 0.0

    try:
        # Create BaseStep for reward calculation
        base_step = BaseStep(
            step_name=step_config.get('step_name', 'unknown'),
            step_type=step_config.get('step_type', 'process'),
            description=step_config.get('description', ''),
            position=step_config.get('position', 0)
        )

        # Use learning pipeline for reward calculation
        learning_pipeline = CumulativeLearningPipeline(
            project_id=step_config.get('project_id', 'default')
        )

        reward = learning_pipeline.calculate_step_reward(
            step=base_step,
            result=execution_result,
            execution_time=execution_time,
            success=success
        )

        return reward

    except Exception as e:
        logger.error(f"Reward calculation failed: {e}")
        return 0.0 if success else -0.5


def update_rl_factors(
    project_id: str,
    step_type: str,
    reward: float,
    context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    TidyLLM function to update RL factors based on feedback.

    Args:
        project_id: Project identifier
        step_type: Type of step executed
        reward: Reward received
        context: Additional context for learning

    Returns:
        Updated RL factors
    """
    if not RL_COMPONENTS_AVAILABLE:
        return {"error": "RL components not available"}

    try:
        rl_optimizer = RLFactorOptimizer(project_id=project_id)

        # Process feedback using existing method
        rl_optimizer.process_feedback('explicit', reward, context)

        # Get updated factors
        updated_factors = rl_optimizer.optimize_step_factors(
            step_config={'step_type': step_type},
            performance_metrics={'rewards': [reward], 'latencies': [1.0]}
        )

        return {
            "update_successful": True,
            "updated_factors": updated_factors,
            "reward_processed": reward,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"RL factor update failed: {e}")
        return {"error": str(e), "update_successful": False}


def get_rl_performance_summary(project_id: str) -> Dict[str, Any]:
    """
    TidyLLM function to get RL performance summary.

    Args:
        project_id: Project identifier

    Returns:
        Performance summary with metrics
    """
    if not RL_COMPONENTS_AVAILABLE:
        return {"error": "RL components not available"}

    try:
        rl_optimizer = RLFactorOptimizer(project_id=project_id)
        learning_pipeline = CumulativeLearningPipeline(
            project_id=project_id
        )

        rl_summary = rl_optimizer.get_optimization_report()
        learning_summary = learning_pipeline.get_performance_report()

        return {
            "project_id": project_id,
            "summary_timestamp": datetime.now().isoformat(),
            "rl_performance": rl_summary,
            "learning_metrics": learning_summary,
            "tidyllm_service": "workflow_rl_optimizer"
        }

    except Exception as e:
        logger.error(f"Performance summary failed: {e}")
        return {"error": str(e)}


def create_rl_enhanced_step(
    step_config: Dict[str, Any],
    project_id: str
) -> Dict[str, Any]:
    """
    TidyLLM function to create an RL-enhanced step configuration.

    Args:
        step_config: Original step configuration
        project_id: Project identifier

    Returns:
        Enhanced step configuration with RL factors
    """
    if not RL_COMPONENTS_AVAILABLE:
        return step_config  # Return original if RL not available

    try:
        step_type = step_config.get('step_type', 'process')

        # Get RL optimization for this step
        optimization_result = optimize_workflow_execution(
            project_id=project_id,
            workflow_config={'steps': [step_config]},
            context={}
        )

        if optimization_result.get('error'):
            return step_config  # Return original on error

        optimized_steps = optimization_result.get('optimized_steps', [])
        if optimized_steps:
            return optimized_steps[0]

        return step_config

    except Exception as e:
        logger.error(f"Step enhancement failed: {e}")
        return step_config


# TidyLLM service metadata
TIDYLLM_RL_SERVICE_INFO = {
    "service_name": "workflow_rl_optimizer",
    "service_type": "tidyllm_optimization_functions",
    "version": "1.0.0",
    "functions": [
        "optimize_workflow_execution",
        "calculate_step_reward",
        "update_rl_factors",
        "get_rl_performance_summary",
        "create_rl_enhanced_step"
    ],
    "dependencies": ["domain.services.rl_factor_optimizer"],
    "rl_components_available": RL_COMPONENTS_AVAILABLE
}