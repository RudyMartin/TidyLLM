"""
Project-Based Model Selection Strategy
======================================

Automatically selects the optimal AI model based on project type,
task complexity, and requirements.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class ProjectType(Enum):
    """Types of projects with different model requirements."""
    LEGAL_COMPLIANCE = "legal_compliance"
    CODE_GENERATION = "code_generation"
    DATA_ANALYSIS = "data_analysis"
    CREATIVE_WRITING = "creative_writing"
    CUSTOMER_SUPPORT = "customer_support"
    TECHNICAL_DOCS = "technical_docs"
    RESEARCH = "research"
    QUICK_ANSWERS = "quick_answers"
    IMAGE_ANALYSIS = "image_analysis"
    COMPLEX_REASONING = "complex_reasoning"


class TaskComplexity(Enum):
    """Task complexity levels."""
    SIMPLE = "simple"      # Quick, straightforward tasks
    MODERATE = "moderate"  # Standard complexity
    COMPLEX = "complex"    # Multi-step, nuanced tasks
    CRITICAL = "critical"  # High-stakes, need best quality


@dataclass
class ModelProfile:
    """Profile for each model's capabilities."""
    model_id: str
    name: str
    strengths: List[str]
    weaknesses: List[str]
    cost_per_1k_tokens: float  # Approximate
    speed_rating: int  # 1-10, 10 being fastest
    quality_rating: int  # 1-10, 10 being best
    context_window: int
    supports_vision: bool = False
    supports_tools: bool = False


# Model profiles based on real-world performance
MODEL_PROFILES = {
    "claude-3-haiku": ModelProfile(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        name="Claude 3 Haiku",
        strengths=["speed", "cost-effective", "good for simple tasks"],
        weaknesses=["less nuanced", "shorter context"],
        cost_per_1k_tokens=0.00025,
        speed_rating=10,
        quality_rating=6,
        context_window=200000,
        supports_vision=True
    ),
    "claude-3-sonnet": ModelProfile(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        name="Claude 3 Sonnet",
        strengths=["balanced", "good reasoning", "reliable"],
        weaknesses=["not the fastest", "not the cheapest"],
        cost_per_1k_tokens=0.003,
        speed_rating=7,
        quality_rating=8,
        context_window=200000,
        supports_vision=True
    ),
    "claude-3.5-sonnet": ModelProfile(
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        name="Claude 3.5 Sonnet",
        strengths=["excellent reasoning", "code generation", "analysis"],
        weaknesses=["higher cost"],
        cost_per_1k_tokens=0.003,
        speed_rating=7,
        quality_rating=9,
        context_window=200000,
        supports_vision=True,
        supports_tools=True
    ),
    "claude-3-opus": ModelProfile(
        model_id="anthropic.claude-3-opus-20240229-v1:0",
        name="Claude 3 Opus",
        strengths=["best quality", "complex reasoning", "nuanced understanding"],
        weaknesses=["expensive", "slower"],
        cost_per_1k_tokens=0.015,
        speed_rating=5,
        quality_rating=10,
        context_window=200000,
        supports_vision=True
    ),
    "llama-3.1-8b": ModelProfile(
        model_id="meta.llama3-1-8b-instruct-v1:0",
        name="Llama 3.1 8B",
        strengths=["very fast", "low cost", "good for simple tasks"],
        weaknesses=["limited reasoning", "smaller model"],
        cost_per_1k_tokens=0.0003,
        speed_rating=10,
        quality_rating=5,
        context_window=128000
    ),
    "llama-3.1-70b": ModelProfile(
        model_id="meta.llama3-1-70b-instruct-v1:0",
        name="Llama 3.1 70B",
        strengths=["good quality", "open source", "cost-effective"],
        weaknesses=["not as nuanced as Claude"],
        cost_per_1k_tokens=0.00265,
        speed_rating=6,
        quality_rating=7,
        context_window=128000
    ),
    "llama-3.1-405b": ModelProfile(
        model_id="meta.llama3-1-405b-instruct-v1:0",
        name="Llama 3.1 405B",
        strengths=["very capable", "large context", "good reasoning"],
        weaknesses=["expensive", "slower"],
        cost_per_1k_tokens=0.00532,
        speed_rating=4,
        quality_rating=8,
        context_window=128000
    ),
    "titan-lite": ModelProfile(
        model_id="amazon.titan-text-lite-v1",
        name="Titan Lite",
        strengths=["very cheap", "fast", "AWS native"],
        weaknesses=["basic capabilities"],
        cost_per_1k_tokens=0.00015,
        speed_rating=10,
        quality_rating=4,
        context_window=4000
    ),
    "titan-express": ModelProfile(
        model_id="amazon.titan-text-express-v1",
        name="Titan Express",
        strengths=["cheap", "reliable", "AWS native"],
        weaknesses=["limited complexity"],
        cost_per_1k_tokens=0.0002,
        speed_rating=9,
        quality_rating=5,
        context_window=8000
    ),
    "titan-premier": ModelProfile(
        model_id="amazon.titan-text-premier-v1:0",
        name="Titan Premier",
        strengths=["good quality", "AWS optimized", "balanced"],
        weaknesses=["not best in class"],
        cost_per_1k_tokens=0.0005,
        speed_rating=7,
        quality_rating=6,
        context_window=32000
    )
}


class ProjectModelSelector:
    """Selects optimal models based on project requirements."""

    def __init__(self):
        """Initialize the model selector."""
        self.model_profiles = MODEL_PROFILES
        self.project_configs = self._init_project_configs()

    def _init_project_configs(self) -> Dict[ProjectType, Dict]:
        """Initialize project-specific model configurations."""
        return {
            ProjectType.LEGAL_COMPLIANCE: {
                "primary": "claude-3-opus",  # Need highest accuracy
                "secondary": "claude-3.5-sonnet",
                "budget": "claude-3-sonnet",
                "requirements": ["high accuracy", "nuanced understanding", "compliance"]
            },
            ProjectType.CODE_GENERATION: {
                "primary": "claude-3.5-sonnet",  # Best for code
                "secondary": "claude-3-sonnet",
                "budget": "llama-3.1-70b",
                "requirements": ["code understanding", "debugging", "refactoring"]
            },
            ProjectType.DATA_ANALYSIS: {
                "primary": "claude-3.5-sonnet",
                "secondary": "llama-3.1-405b",
                "budget": "llama-3.1-70b",
                "requirements": ["analytical", "statistical", "visualization"]
            },
            ProjectType.CREATIVE_WRITING: {
                "primary": "claude-3-opus",
                "secondary": "claude-3.5-sonnet",
                "budget": "claude-3-sonnet",
                "requirements": ["creativity", "style", "narrative"]
            },
            ProjectType.CUSTOMER_SUPPORT: {
                "primary": "claude-3-haiku",  # Fast responses
                "secondary": "llama-3.1-8b",
                "budget": "titan-express",
                "requirements": ["speed", "friendliness", "accuracy"]
            },
            ProjectType.TECHNICAL_DOCS: {
                "primary": "claude-3.5-sonnet",
                "secondary": "claude-3-sonnet",
                "budget": "llama-3.1-70b",
                "requirements": ["technical accuracy", "clarity", "structure"]
            },
            ProjectType.RESEARCH: {
                "primary": "claude-3-opus",
                "secondary": "claude-3.5-sonnet",
                "budget": "llama-3.1-405b",
                "requirements": ["deep analysis", "synthesis", "citations"]
            },
            ProjectType.QUICK_ANSWERS: {
                "primary": "claude-3-haiku",
                "secondary": "llama-3.1-8b",
                "budget": "titan-lite",
                "requirements": ["speed", "basic accuracy"]
            },
            ProjectType.IMAGE_ANALYSIS: {
                "primary": "claude-3.5-sonnet",  # Vision capable
                "secondary": "claude-3-sonnet",
                "budget": "claude-3-haiku",
                "requirements": ["vision", "description", "analysis"]
            },
            ProjectType.COMPLEX_REASONING: {
                "primary": "claude-3-opus",
                "secondary": "claude-3.5-sonnet",
                "budget": "llama-3.1-405b",
                "requirements": ["multi-step reasoning", "logic", "problem solving"]
            }
        }

    def select_model(
        self,
        project_type: ProjectType,
        complexity: TaskComplexity,
        budget_mode: bool = False,
        requires_vision: bool = False,
        requires_tools: bool = False,
        min_context_window: int = 4000
    ) -> str:
        """
        Select the optimal model for a given project and task.

        Args:
            project_type: Type of project
            complexity: Task complexity level
            budget_mode: Whether to optimize for cost
            requires_vision: Whether task needs image understanding
            requires_tools: Whether task needs tool/function calling
            min_context_window: Minimum required context window

        Returns:
            Model ID string
        """
        config = self.project_configs.get(project_type)

        if not config:
            # Default fallback
            return "claude-3-sonnet" if not budget_mode else "claude-3-haiku"

        # Start with base selection
        if budget_mode:
            model_key = config["budget"]
        elif complexity == TaskComplexity.CRITICAL:
            model_key = config["primary"]
        elif complexity == TaskComplexity.COMPLEX:
            model_key = config["primary"]
        elif complexity == TaskComplexity.MODERATE:
            model_key = config["secondary"]
        else:  # SIMPLE
            model_key = config["budget"]

        # Get the model profile
        model = self.model_profiles.get(model_key)

        # Check constraints
        if requires_vision and not model.supports_vision:
            # Need to upgrade to vision-capable model
            for key in [config["primary"], config["secondary"]]:
                alt_model = self.model_profiles.get(key)
                if alt_model and alt_model.supports_vision:
                    model_key = key
                    model = alt_model
                    break

        if requires_tools and not model.supports_tools:
            # Currently only Claude 3.5 Sonnet supports tools well
            model_key = "claude-3.5-sonnet"
            model = self.model_profiles["claude-3.5-sonnet"]

        if model.context_window < min_context_window:
            # Need larger context window
            for key, profile in self.model_profiles.items():
                if profile.context_window >= min_context_window:
                    model_key = key
                    model = profile
                    break

        logger.info(
            f"Selected {model.name} for {project_type.value} "
            f"(complexity: {complexity.value}, budget_mode: {budget_mode})"
        )

        return model.model_id

    def get_model_recommendation(
        self,
        project_type: ProjectType,
        complexity: TaskComplexity
    ) -> Dict[str, Any]:
        """
        Get detailed model recommendation with rationale.

        Returns:
            Dictionary with recommendation details
        """
        primary = self.project_configs[project_type]["primary"]
        secondary = self.project_configs[project_type]["secondary"]
        budget = self.project_configs[project_type]["budget"]

        primary_model = self.model_profiles[primary]
        secondary_model = self.model_profiles[secondary]
        budget_model = self.model_profiles[budget]

        return {
            "project_type": project_type.value,
            "complexity": complexity.value,
            "recommendations": {
                "optimal": {
                    "model": primary_model.name,
                    "model_id": primary_model.model_id,
                    "rationale": f"Best quality for {project_type.value}",
                    "cost_per_1k": primary_model.cost_per_1k_tokens,
                    "speed_rating": primary_model.speed_rating,
                    "quality_rating": primary_model.quality_rating
                },
                "balanced": {
                    "model": secondary_model.name,
                    "model_id": secondary_model.model_id,
                    "rationale": "Good balance of cost and quality",
                    "cost_per_1k": secondary_model.cost_per_1k_tokens,
                    "speed_rating": secondary_model.speed_rating,
                    "quality_rating": secondary_model.quality_rating
                },
                "budget": {
                    "model": budget_model.name,
                    "model_id": budget_model.model_id,
                    "rationale": "Most cost-effective option",
                    "cost_per_1k": budget_model.cost_per_1k_tokens,
                    "speed_rating": budget_model.speed_rating,
                    "quality_rating": budget_model.quality_rating
                }
            },
            "requirements": self.project_configs[project_type]["requirements"]
        }

    def estimate_cost(
        self,
        project_type: ProjectType,
        estimated_tokens: int,
        complexity: TaskComplexity
    ) -> Dict[str, float]:
        """
        Estimate costs for different model options.

        Args:
            project_type: Type of project
            estimated_tokens: Estimated total tokens
            complexity: Task complexity

        Returns:
            Cost estimates for each model tier
        """
        config = self.project_configs[project_type]

        costs = {}
        for tier, model_key in [
            ("optimal", config["primary"]),
            ("balanced", config["secondary"]),
            ("budget", config["budget"])
        ]:
            model = self.model_profiles[model_key]
            cost = (estimated_tokens / 1000) * model.cost_per_1k_tokens
            costs[tier] = {
                "model": model.name,
                "total_cost": round(cost, 4),
                "cost_per_1k": model.cost_per_1k_tokens
            }

        return costs


def demo_model_selection():
    """Demonstrate the model selection system."""
    selector = ProjectModelSelector()

    print("=== Project-Based Model Selection Demo ===\n")

    # Example 1: Legal compliance document
    print("1. Legal Compliance Document Review")
    print("-" * 40)
    legal_model = selector.select_model(
        ProjectType.LEGAL_COMPLIANCE,
        TaskComplexity.CRITICAL,
        budget_mode=False
    )
    print(f"Selected Model: {legal_model}")

    recommendation = selector.get_model_recommendation(
        ProjectType.LEGAL_COMPLIANCE,
        TaskComplexity.CRITICAL
    )
    print(f"Rationale: {recommendation['recommendations']['optimal']['rationale']}")
    print(f"Quality Rating: {recommendation['recommendations']['optimal']['quality_rating']}/10")
    print()

    # Example 2: Customer support chatbot
    print("2. Customer Support Quick Response")
    print("-" * 40)
    support_model = selector.select_model(
        ProjectType.CUSTOMER_SUPPORT,
        TaskComplexity.SIMPLE,
        budget_mode=True
    )
    print(f"Selected Model: {support_model}")

    recommendation = selector.get_model_recommendation(
        ProjectType.CUSTOMER_SUPPORT,
        TaskComplexity.SIMPLE
    )
    print(f"Rationale: Fast responses at low cost")
    print(f"Speed Rating: {recommendation['recommendations']['budget']['speed_rating']}/10")
    print(f"Cost per 1K tokens: ${recommendation['recommendations']['budget']['cost_per_1k']}")
    print()

    # Example 3: Code generation with budget constraint
    print("3. Code Generation (Budget Mode)")
    print("-" * 40)
    code_model = selector.select_model(
        ProjectType.CODE_GENERATION,
        TaskComplexity.MODERATE,
        budget_mode=True
    )
    print(f"Selected Model: {code_model}")

    # Cost estimation
    costs = selector.estimate_cost(
        ProjectType.CODE_GENERATION,
        estimated_tokens=50000,
        complexity=TaskComplexity.MODERATE
    )
    print("\nCost Comparison for 50K tokens:")
    for tier, details in costs.items():
        print(f"  {tier}: {details['model']} - ${details['total_cost']}")
    print()

    # Example 4: Image analysis task
    print("4. Image Analysis Task")
    print("-" * 40)
    image_model = selector.select_model(
        ProjectType.IMAGE_ANALYSIS,
        TaskComplexity.MODERATE,
        requires_vision=True
    )
    print(f"Selected Model: {image_model}")
    print("Note: Automatically selected vision-capable model")
    print()

    # Show all project type recommendations
    print("\n=== Default Model Recommendations by Project Type ===")
    print("-" * 60)
    for project_type in ProjectType:
        config = selector.project_configs[project_type]
        primary = selector.model_profiles[config["primary"]]
        print(f"{project_type.value:20} -> {primary.name:20} (Quality: {primary.quality_rating}/10)")


if __name__ == "__main__":
    demo_model_selection()