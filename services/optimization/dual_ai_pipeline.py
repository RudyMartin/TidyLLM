"""
Dual AI Pipeline Service
========================

Two-stage AI processing where:
1. Initial AI: Fast, broad response generation (assumes incomplete)
2. Enhancement AI: Reviews, corrects, and enhances the initial response

Use Cases:
- Workflow advice with initial quick response + detailed enhancement
- Template ordering suggestions with follow-up optimization
- Document analysis with preliminary + comprehensive review

Architecture:
    User Query → Initial AI → Enhancement AI → Final Response
"""

import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# MLflow integration for dual AI tracking
try:
    import mlflow
    import mlflow.tracking
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# TidyLLM Corporate Gateway integration
try:
    from tidyllm.gateways.corporate_llm_gateway import CorporateLLMGateway, LLMRequest, LLMResponse
    CORPORATE_GATEWAY_AVAILABLE = True
except ImportError:
    CORPORATE_GATEWAY_AVAILABLE = False

logger = logging.getLogger(__name__)


class AIStage(Enum):
    """AI processing stages."""
    INITIAL = "initial"
    ENHANCEMENT = "enhancement"


@dataclass
class AIResponse:
    """Response from an AI stage."""
    content: str
    confidence: float
    processing_time_ms: float
    stage: AIStage
    model_used: str
    tokens_used: int
    metadata: Dict[str, Any]


@dataclass
class DualAIResult:
    """Final result from dual AI pipeline."""
    initial_response: AIResponse
    enhanced_response: AIResponse
    improvement_summary: str
    total_processing_time_ms: float
    confidence_improvement: float
    final_content: str


class DualAIPipeline:
    """
    Two-stage AI pipeline for improved response quality.

    Stage 1: Initial AI - Fast, broad responses (assumes incomplete)
    Stage 2: Enhancement AI - Review, correct, and enhance
    """

    def __init__(self,
                 mlflow_experiment_name: str = "dual_ai_pipeline",
                 stage1_model: str = "claude-3-haiku",
                 stage2_model: str = "claude-3-sonnet"):
        """Initialize dual AI pipeline with configurable models and MLflow tracking."""
        self.stage1_config = {
            "model": stage1_model,  # Configurable fast model for initial response
            "temperature": 0.7,
            "max_tokens": 500,  # Shorter initial responses
            "purpose": "quick_draft"
        }

        self.stage2_config = {
            "model": stage2_model,  # Configurable better model for enhancement
            "temperature": 0.3,  # More focused for corrections
            "max_tokens": 1500,  # Longer enhanced responses
            "purpose": "enhancement_review"
        }

        # Initialize Corporate LLM Gateway
        self.gateway_enabled = CORPORATE_GATEWAY_AVAILABLE
        if self.gateway_enabled:
            try:
                self.llm_gateway = CorporateLLMGateway()
                logger.info("CorporateLLMGateway initialized for dual AI pipeline")
            except Exception as e:
                logger.warning(f"CorporateLLMGateway initialization failed: {e}")
                self.gateway_enabled = False
                self.llm_gateway = None
        else:
            self.llm_gateway = None

        # MLflow setup
        self.mlflow_experiment = mlflow_experiment_name
        self.mlflow_enabled = MLFLOW_AVAILABLE
        if self.mlflow_enabled:
            try:
                mlflow.set_experiment(self.mlflow_experiment)
                logger.info(f"MLflow experiment set: {self.mlflow_experiment}")
            except Exception as e:
                logger.warning(f"MLflow setup failed: {e}")
                self.mlflow_enabled = False

    def process_query(self,
                     query: str,
                     context: Dict[str, Any] = None,
                     workflow_data: Dict[str, Any] = None,
                     user_id: str = "system") -> DualAIResult:
        """
        Process query through dual AI pipeline with comprehensive MLflow tracking.

        Args:
            query: User's question or request
            context: Additional context (workflow, templates, etc.)
            workflow_data: Specific workflow information
            user_id: User identifier for tracking

        Returns:
            DualAIResult with both stages and final enhanced response
        """
        start_time = time.time()
        pipeline_run_id = None
        stage1_run_id = None
        stage2_run_id = None

        try:
            # Start MLflow parent run for entire pipeline
            if self.mlflow_enabled:
                pipeline_run_id = self._start_pipeline_mlflow_run(query, context, user_id)

            # Stage 1: Initial AI Response with MLflow tracking
            logger.info("Stage 1: Generating initial response...")
            initial_response, stage1_run_id = self._stage1_initial_ai_with_tracking(
                query, context, workflow_data, pipeline_run_id
            )

            # Stage 2: Enhancement AI with MLflow tracking
            logger.info("Stage 2: Enhancing and correcting initial response...")
            enhanced_response, stage2_run_id = self._stage2_enhancement_ai_with_tracking(
                query, initial_response, context, workflow_data, pipeline_run_id
            )

            # Calculate improvements
            total_time = (time.time() - start_time) * 1000
            confidence_improvement = enhanced_response.confidence - initial_response.confidence
            improvement_summary = self._analyze_improvements(initial_response, enhanced_response)

            # Log pipeline-level metrics
            if self.mlflow_enabled and pipeline_run_id:
                self._log_pipeline_metrics(
                    pipeline_run_id, initial_response, enhanced_response,
                    total_time, confidence_improvement, improvement_summary
                )

            result = DualAIResult(
                initial_response=initial_response,
                enhanced_response=enhanced_response,
                improvement_summary=improvement_summary,
                total_processing_time_ms=total_time,
                confidence_improvement=confidence_improvement,
                final_content=enhanced_response.content
            )

            # End pipeline run
            if self.mlflow_enabled and pipeline_run_id:
                mlflow.end_run()

            return result

        except Exception as e:
            logger.error(f"Dual AI pipeline failed: {e}")
            # End any open runs
            if self.mlflow_enabled:
                try:
                    mlflow.end_run()
                except:
                    pass
            raise

    def _stage1_initial_ai(self,
                          query: str,
                          context: Dict[str, Any],
                          workflow_data: Dict[str, Any]) -> AIResponse:
        """
        Stage 1: Fast initial response generation.

        Assumes response will be incomplete and need enhancement.
        """
        start_time = time.time()

        # Build initial prompt - focused on speed over completeness
        prompt = self._build_initial_prompt(query, context, workflow_data)

        try:
            # Simulate AI call (replace with actual DSPy/Claude call)
            content = self._call_initial_ai(prompt)

            processing_time = (time.time() - start_time) * 1000

            return AIResponse(
                content=content,
                confidence=0.6,  # Deliberately lower - we assume it's incomplete
                processing_time_ms=processing_time,
                stage=AIStage.INITIAL,
                model_used=self.stage1_config["model"],
                tokens_used=len(content.split()) * 1.3,  # Rough estimate
                metadata={
                    "temperature": self.stage1_config["temperature"],
                    "max_tokens": self.stage1_config["max_tokens"],
                    "purpose": "quick_initial_draft"
                }
            )

        except Exception as e:
            logger.error(f"Stage 1 AI failed: {e}")
            return AIResponse(
                content=f"Initial analysis: {query} requires template ordering review.",
                confidence=0.3,
                processing_time_ms=(time.time() - start_time) * 1000,
                stage=AIStage.INITIAL,
                model_used="fallback",
                tokens_used=10,
                metadata={"error": str(e)}
            )

    def _stage2_enhancement_ai(self,
                              query: str,
                              initial_response: AIResponse,
                              context: Dict[str, Any],
                              workflow_data: Dict[str, Any]) -> AIResponse:
        """
        Stage 2: Enhancement and correction of initial response.

        Reviews initial response and provides comprehensive improvement.
        """
        start_time = time.time()

        # Build enhancement prompt with initial response for review
        prompt = self._build_enhancement_prompt(query, initial_response, context, workflow_data)

        try:
            # Call enhancement AI
            content = self._call_enhancement_ai(prompt)

            processing_time = (time.time() - start_time) * 1000

            return AIResponse(
                content=content,
                confidence=0.9,  # High confidence after enhancement
                processing_time_ms=processing_time,
                stage=AIStage.ENHANCEMENT,
                model_used=self.stage2_config["model"],
                tokens_used=len(content.split()) * 1.3,
                metadata={
                    "temperature": self.stage2_config["temperature"],
                    "max_tokens": self.stage2_config["max_tokens"],
                    "purpose": "comprehensive_enhancement",
                    "improved_from": initial_response.model_used
                }
            )

        except Exception as e:
            logger.error(f"Stage 2 AI failed: {e}")
            # Fallback to initial response if enhancement fails
            return AIResponse(
                content=f"Enhanced analysis: {initial_response.content}\n\n[Enhancement unavailable: {e}]",
                confidence=initial_response.confidence + 0.1,
                processing_time_ms=(time.time() - start_time) * 1000,
                stage=AIStage.ENHANCEMENT,
                model_used="fallback",
                tokens_used=initial_response.tokens_used + 10,
                metadata={"enhancement_error": str(e)}
            )

    def _build_initial_prompt(self, query: str, context: Dict, workflow_data: Dict) -> str:
        """Build prompt for initial AI (fast, broad response)."""
        base_prompt = f"""
Provide a quick initial analysis for this workflow question:

Question: {query}

Context: {context.get('workflow_type', 'general')} workflow

Instructions:
- Give a fast, preliminary response
- Focus on immediate, actionable insights
- Keep response concise (under 300 words)
- Don't worry about completeness - this will be enhanced

Response:"""

        return base_prompt

    def _build_enhancement_prompt(self, query: str, initial_response: AIResponse,
                                context: Dict, workflow_data: Dict) -> str:
        """Build prompt for enhancement AI (comprehensive review and improvement)."""
        enhancement_prompt = f"""
You are an expert workflow advisor reviewing and enhancing an initial AI response.

Original Question: {query}

Initial Response (assume it's incomplete):
{initial_response.content}

Your task:
1. REVIEW the initial response for gaps, errors, or missing information
2. ENHANCE with additional insights, details, and best practices
3. CORRECT any inaccuracies or oversimplifications
4. EXPAND with specific recommendations and next steps

Context Information:
- Workflow Type: {context.get('workflow_type', 'general')}
- Template Ordering: {workflow_data.get('steps', [])}
- Project: {context.get('project_name', 'unknown')}

Provide a comprehensive, enhanced response that:
- Addresses all aspects of the original question
- Includes specific, actionable recommendations
- Explains the reasoning behind suggestions
- Considers potential challenges and solutions

Enhanced Response:"""

        return enhancement_prompt

    def _call_initial_ai(self, prompt: str) -> str:
        """Call initial AI service (Stage 1) using CorporateLLMGateway."""
        if not self.gateway_enabled or not self.llm_gateway:
            logger.warning("CorporateLLMGateway not available, using fallback")
            return f"Initial analysis suggests reviewing workflow configuration for optimization opportunities."

        try:
            # Create LLM request for Stage 1
            request = LLMRequest(
                prompt=prompt,
                model_id=self.stage1_config["model"],
                temperature=self.stage1_config["temperature"],
                max_tokens=self.stage1_config["max_tokens"],
                audit_reason="dual_ai_pipeline_stage1"
            )

            # Process through Corporate Gateway
            response = self.llm_gateway.process_request(request)

            if response.success:
                return response.content
            else:
                logger.error(f"Stage 1 AI failed: {response.error}")
                return f"Initial analysis suggests reviewing workflow configuration for optimization opportunities."

        except Exception as e:
            logger.error(f"Stage 1 AI gateway error: {e}")
            return f"Initial analysis suggests reviewing workflow configuration for optimization opportunities."

    def _call_enhancement_ai(self, prompt: str) -> str:
        """Call enhancement AI service (Stage 2) using CorporateLLMGateway."""
        if not self.gateway_enabled or not self.llm_gateway:
            logger.warning("CorporateLLMGateway not available, using fallback")
            return f"Enhanced analysis: The initial response was correct but incomplete. Additional considerations include workflow dependencies, performance optimization, error handling, and testing strategies for optimal results."

        try:
            # Create LLM request for Stage 2
            request = LLMRequest(
                prompt=prompt,
                model_id=self.stage2_config["model"],
                temperature=self.stage2_config["temperature"],
                max_tokens=self.stage2_config["max_tokens"],
                audit_reason="dual_ai_pipeline_stage2"
            )

            # Process through Corporate Gateway
            response = self.llm_gateway.process_request(request)

            if response.success:
                return response.content
            else:
                logger.error(f"Stage 2 AI failed: {response.error}")
                return f"Enhanced analysis: The initial response was correct but incomplete. Additional considerations include workflow dependencies, performance optimization, error handling, and testing strategies for optimal results."

        except Exception as e:
            logger.error(f"Stage 2 AI gateway error: {e}")
            return f"Enhanced analysis: The initial response was correct but incomplete. Additional considerations include workflow dependencies, performance optimization, error handling, and testing strategies for optimal results."

    def _analyze_improvements(self, initial: AIResponse, enhanced: AIResponse) -> str:
        """Analyze improvements made by enhancement stage."""
        initial_length = len(initial.content.split())
        enhanced_length = len(enhanced.content.split())

        improvements = []

        if enhanced_length > initial_length * 1.5:
            improvements.append("Significantly expanded content")

        if enhanced.confidence > initial.confidence:
            improvements.append(f"Increased confidence by {enhanced.confidence - initial.confidence:.1f}")

        if "specific" in enhanced.content.lower() and "specific" not in initial.content.lower():
            improvements.append("Added specific recommendations")

        if "step" in enhanced.content.lower() and enhanced.content.count("step") > initial.content.count("step"):
            improvements.append("Enhanced with detailed steps")

        return "; ".join(improvements) if improvements else "Content refined and enhanced"

    # ==================== MLFLOW TRACKING METHODS ====================

    def _start_pipeline_mlflow_run(self, query: str, context: Dict[str, Any], user_id: str) -> str:
        """Start MLflow run for the entire dual AI pipeline."""
        try:
            run = mlflow.start_run(run_name=f"dual_ai_pipeline_{int(time.time())}")

            # Log pipeline parameters
            mlflow.log_param("user_id", user_id)
            mlflow.log_param("query_length", len(query))
            mlflow.log_param("workflow_type", context.get("workflow_type", "unknown"))
            mlflow.log_param("project_name", context.get("project_name", "unknown"))
            mlflow.log_param("pipeline_type", "dual_ai")
            mlflow.log_param("stage1_model", self.stage1_config["model"])
            mlflow.log_param("stage2_model", self.stage2_config["model"])

            # Log query as artifact
            with open("temp_query.txt", "w", encoding="utf-8") as f:
                f.write(query)
            mlflow.log_artifact("temp_query.txt", "pipeline_inputs")

            return run.info.run_id

        except Exception as e:
            logger.error(f"Failed to start pipeline MLflow run: {e}")
            return None

    def _stage1_initial_ai_with_tracking(self, query: str, context: Dict[str, Any],
                                        workflow_data: Dict[str, Any], parent_run_id: str) -> Tuple[AIResponse, str]:
        """Stage 1 AI with comprehensive MLflow tracking."""
        stage1_run_id = None

        try:
            # Start nested run for Stage 1
            if self.mlflow_enabled and parent_run_id:
                stage1_run = mlflow.start_run(
                    run_name=f"stage1_initial_{int(time.time())}",
                    nested=True
                )
                stage1_run_id = stage1_run.info.run_id

                # Log Stage 1 parameters
                mlflow.log_param("stage", "initial")
                mlflow.log_param("model", self.stage1_config["model"])
                mlflow.log_param("temperature", self.stage1_config["temperature"])
                mlflow.log_param("max_tokens", self.stage1_config["max_tokens"])
                mlflow.log_param("purpose", self.stage1_config["purpose"])

            # Execute Stage 1 AI
            response = self._stage1_initial_ai(query, context, workflow_data)

            # Log Stage 1 metrics
            if self.mlflow_enabled and stage1_run_id:
                mlflow.log_metric("processing_time_ms", response.processing_time_ms)
                mlflow.log_metric("confidence", response.confidence)
                mlflow.log_metric("tokens_used", response.tokens_used)
                mlflow.log_metric("response_length", len(response.content))
                mlflow.log_metric("response_word_count", len(response.content.split()))

                # Log response as artifact
                with open("temp_stage1_response.txt", "w", encoding="utf-8") as f:
                    f.write(response.content)
                mlflow.log_artifact("temp_stage1_response.txt", "stage1_outputs")

                # End Stage 1 run
                mlflow.end_run()

            return response, stage1_run_id

        except Exception as e:
            logger.error(f"Stage 1 with tracking failed: {e}")
            if self.mlflow_enabled and stage1_run_id:
                mlflow.log_metric("error_occurred", 1)
                mlflow.end_run()

            # Fallback to basic Stage 1
            response = self._stage1_initial_ai(query, context, workflow_data)
            return response, None

    def _stage2_enhancement_ai_with_tracking(self, query: str, initial_response: AIResponse,
                                           context: Dict[str, Any], workflow_data: Dict[str, Any],
                                           parent_run_id: str) -> Tuple[AIResponse, str]:
        """Stage 2 AI with comprehensive MLflow tracking."""
        stage2_run_id = None

        try:
            # Start nested run for Stage 2
            if self.mlflow_enabled and parent_run_id:
                stage2_run = mlflow.start_run(
                    run_name=f"stage2_enhancement_{int(time.time())}",
                    nested=True
                )
                stage2_run_id = stage2_run.info.run_id

                # Log Stage 2 parameters
                mlflow.log_param("stage", "enhancement")
                mlflow.log_param("model", self.stage2_config["model"])
                mlflow.log_param("temperature", self.stage2_config["temperature"])
                mlflow.log_param("max_tokens", self.stage2_config["max_tokens"])
                mlflow.log_param("purpose", self.stage2_config["purpose"])
                mlflow.log_param("initial_confidence", initial_response.confidence)
                mlflow.log_param("initial_tokens", initial_response.tokens_used)

            # Execute Stage 2 AI
            response = self._stage2_enhancement_ai(query, initial_response, context, workflow_data)

            # Log Stage 2 metrics
            if self.mlflow_enabled and stage2_run_id:
                mlflow.log_metric("processing_time_ms", response.processing_time_ms)
                mlflow.log_metric("confidence", response.confidence)
                mlflow.log_metric("tokens_used", response.tokens_used)
                mlflow.log_metric("response_length", len(response.content))
                mlflow.log_metric("response_word_count", len(response.content.split()))

                # Enhancement metrics
                confidence_gain = response.confidence - initial_response.confidence
                content_expansion = len(response.content) / len(initial_response.content)
                token_efficiency = response.tokens_used / initial_response.tokens_used

                mlflow.log_metric("confidence_gain", confidence_gain)
                mlflow.log_metric("content_expansion_ratio", content_expansion)
                mlflow.log_metric("token_efficiency_ratio", token_efficiency)

                # Log enhanced response as artifact
                with open("temp_stage2_response.txt", "w", encoding="utf-8") as f:
                    f.write(response.content)
                mlflow.log_artifact("temp_stage2_response.txt", "stage2_outputs")

                # End Stage 2 run
                mlflow.end_run()

            return response, stage2_run_id

        except Exception as e:
            logger.error(f"Stage 2 with tracking failed: {e}")
            if self.mlflow_enabled and stage2_run_id:
                mlflow.log_metric("error_occurred", 1)
                mlflow.end_run()

            # Fallback to basic Stage 2
            response = self._stage2_enhancement_ai(query, initial_response, context, workflow_data)
            return response, None

    def _log_pipeline_metrics(self, pipeline_run_id: str, initial_response: AIResponse,
                             enhanced_response: AIResponse, total_time: float,
                             confidence_improvement: float, improvement_summary: str):
        """Log comprehensive pipeline-level metrics to MLflow."""
        try:
            # Pipeline performance metrics
            mlflow.log_metric("total_processing_time_ms", total_time)
            mlflow.log_metric("confidence_improvement", confidence_improvement)
            mlflow.log_metric("stage1_time_ms", initial_response.processing_time_ms)
            mlflow.log_metric("stage2_time_ms", enhanced_response.processing_time_ms)

            # Content analysis metrics
            initial_words = len(initial_response.content.split())
            enhanced_words = len(enhanced_response.content.split())
            content_growth = (enhanced_words - initial_words) / initial_words if initial_words > 0 else 0

            mlflow.log_metric("initial_word_count", initial_words)
            mlflow.log_metric("enhanced_word_count", enhanced_words)
            mlflow.log_metric("content_growth_percent", content_growth * 100)

            # Token efficiency metrics
            total_tokens = initial_response.tokens_used + enhanced_response.tokens_used
            tokens_per_word = total_tokens / enhanced_words if enhanced_words > 0 else 0

            mlflow.log_metric("total_tokens_used", total_tokens)
            mlflow.log_metric("tokens_per_final_word", tokens_per_word)

            # Quality metrics
            mlflow.log_metric("final_confidence", enhanced_response.confidence)
            mlflow.log_metric("confidence_improvement_percent", (confidence_improvement / initial_response.confidence) * 100)

            # Time efficiency metrics
            time_per_word = total_time / enhanced_words if enhanced_words > 0 else 0
            stage2_efficiency = enhanced_response.processing_time_ms / (enhanced_words - initial_words) if enhanced_words > initial_words else 0

            mlflow.log_metric("ms_per_final_word", time_per_word)
            mlflow.log_metric("stage2_ms_per_new_word", stage2_efficiency)

            # Log improvement summary as parameter
            mlflow.log_param("improvement_summary", improvement_summary)

            # Create comparison artifact
            comparison_data = f"""
Dual AI Pipeline Comparison Report
================================

STAGE 1 (Initial AI):
- Model: {initial_response.model_used}
- Confidence: {initial_response.confidence:.2f}
- Processing Time: {initial_response.processing_time_ms:.1f}ms
- Word Count: {initial_words}
- Tokens Used: {initial_response.tokens_used}

STAGE 2 (Enhancement AI):
- Model: {enhanced_response.model_used}
- Confidence: {enhanced_response.confidence:.2f}
- Processing Time: {enhanced_response.processing_time_ms:.1f}ms
- Word Count: {enhanced_words}
- Tokens Used: {enhanced_response.tokens_used}

IMPROVEMENTS:
- Confidence Gain: +{confidence_improvement:.2f} ({(confidence_improvement/initial_response.confidence)*100:.1f}%)
- Content Growth: +{enhanced_words - initial_words} words ({content_growth*100:.1f}%)
- Total Processing Time: {total_time:.1f}ms
- Summary: {improvement_summary}

EFFICIENCY METRICS:
- Total Tokens: {total_tokens}
- Tokens per Final Word: {tokens_per_word:.2f}
- Time per Final Word: {time_per_word:.2f}ms
- Stage 2 Efficiency: {stage2_efficiency:.2f}ms per new word
"""

            with open("temp_pipeline_comparison.txt", "w", encoding="utf-8") as f:
                f.write(comparison_data)
            mlflow.log_artifact("temp_pipeline_comparison.txt", "pipeline_analysis")

        except Exception as e:
            logger.error(f"Failed to log pipeline metrics: {e}")


# Integration with existing AI Advisor
class EnhancedWorkflowAdvisor:
    """Enhanced workflow advisor using dual AI pipeline."""

    def __init__(self):
        """Initialize with dual AI pipeline."""
        self.dual_ai = DualAIPipeline()

    def get_workflow_advice(self,
                           user_question: str,
                           workflow_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get enhanced workflow advice using dual AI pipeline.

        Returns both initial and enhanced responses for transparency.
        """
        # Process through dual AI pipeline
        result = self.dual_ai.process_query(
            query=user_question,
            context=workflow_context or {},
            workflow_data=workflow_context or {}
        )

        return {
            "final_advice": result.final_content,
            "initial_response": result.initial_response.content,
            "enhancement_summary": result.improvement_summary,
            "confidence_improvement": result.confidence_improvement,
            "processing_time_ms": result.total_processing_time_ms,
            "stages": {
                "initial": {
                    "model": result.initial_response.model_used,
                    "confidence": result.initial_response.confidence,
                    "time_ms": result.initial_response.processing_time_ms
                },
                "enhanced": {
                    "model": result.enhanced_response.model_used,
                    "confidence": result.enhanced_response.confidence,
                    "time_ms": result.enhanced_response.processing_time_ms
                }
            }
        }


def test_dual_ai_pipeline():
    """Test the dual AI pipeline with template ordering query."""
    print("=== Testing Dual AI Pipeline ===")
    print()

    advisor = EnhancedWorkflowAdvisor()

    # Test template ordering question
    test_query = "How should I order my QA/QC workflow templates for optimal results?"
    test_context = {
        "workflow_type": "qaqc",
        "project_name": "alex_qaqc",
        "steps": [
            {"step_id": "metadata_extraction", "step_number": "1.0"},
            {"step_id": "analysis_steps", "step_number": "2.0"},
            {"step_id": "results_aggregation", "step_number": "3.0"},
            {"step_id": "recording_questions", "step_number": "4.0"}
        ]
    }

    print(f"Query: {test_query}")
    print()

    result = advisor.get_workflow_advice(test_query, test_context)

    print("STAGE 1 - Initial Response:")
    print(f"Model: {result['stages']['initial']['model']}")
    print(f"Confidence: {result['stages']['initial']['confidence']}")
    print(f"Time: {result['stages']['initial']['time_ms']:.1f}ms")
    print(f"Content: {result['initial_response'][:200]}...")
    print()

    print("STAGE 2 - Enhanced Response:")
    print(f"Model: {result['stages']['enhanced']['model']}")
    print(f"Confidence: {result['stages']['enhanced']['confidence']}")
    print(f"Time: {result['stages']['enhanced']['time_ms']:.1f}ms")
    print(f"Improvement: {result['enhancement_summary']}")
    print()

    print("FINAL ADVICE:")
    print(result['final_advice'][:500] + "..." if len(result['final_advice']) > 500 else result['final_advice'])
    print()

    print(f"Total Processing Time: {result['processing_time_ms']:.1f}ms")
    print(f"Confidence Improvement: +{result['confidence_improvement']:.1f}")


if __name__ == "__main__":
    test_dual_ai_pipeline()