#!/usr/bin/env python3
"""
Workflow AI Advisor
==================

DSPy-powered AI system for providing intelligent workflow advice based on:
- Criteria analysis
- Template field optimization
- Recent activity patterns
- Final results evaluation
"""

import json
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import dspy

class WorkflowContextAnalyzer(dspy.Signature):
    """Analyze workflow context to provide intelligent recommendations."""

    workflow_criteria = dspy.InputField(desc="JSON criteria defining workflow requirements and document qualifiers")
    template_fields = dspy.InputField(desc="Template fields configuration with validation rules and defaults")
    recent_activity = dspy.InputField(desc="Recent workflow executions, test results, and user interactions")
    final_results = dspy.InputField(desc="Latest workflow execution results and performance metrics")
    user_question = dspy.InputField(desc="User's specific question or request for workflow advice")

    advisor_response = dspy.OutputField(desc="Comprehensive workflow advice with specific recommendations, insights, and actionable suggestions")

class WorkflowOptimizer(dspy.Signature):
    """Provide workflow optimization recommendations."""

    current_workflow = dspy.InputField(desc="Current workflow configuration and performance data")
    bottlenecks = dspy.InputField(desc="Identified performance bottlenecks and inefficiencies")
    usage_patterns = dspy.InputField(desc="User behavior and workflow usage patterns")

    optimization_plan = dspy.OutputField(desc="Detailed optimization recommendations with priority levels and expected impact")

class TemplateFieldAdvisor(dspy.Signature):
    """Advise on template field configuration and validation."""

    current_fields = dspy.InputField(desc="Current template field definitions and configurations")
    validation_errors = dspy.InputField(desc="Recent validation errors or field-related issues")
    use_cases = dspy.InputField(desc="Intended use cases and document types for the workflow")

    field_recommendations = dspy.OutputField(desc="Template field improvement suggestions with validation rules and default values")

class WorkflowAIAdvisor:
    """AI-powered workflow advisor using DSPy for intelligent recommendations."""

    def __init__(self):
        # Initialize DSPy modules
        self.context_analyzer = dspy.ChainOfThought(WorkflowContextAnalyzer)
        self.optimizer = dspy.ChainOfThought(WorkflowOptimizer)
        self.field_advisor = dspy.ChainOfThought(TemplateFieldAdvisor)

        # Initialize RAG knowledge base
        try:
            from rag_knowledge_base import workflow_knowledge_base
            self.knowledge_base = workflow_knowledge_base
            self.rag_enabled = True
        except ImportError:
            self.knowledge_base = None
            self.rag_enabled = False

        # System prompt for workflow expertise
        self.system_context = """You are an expert workflow architect specializing in document processing and analysis workflows.

Your expertise includes:
- RAG (Retrieval-Augmented Generation) system integration
- Sequential workflow optimization
- Template field design and validation
- Document processing pipelines
- Performance optimization strategies
- Error handling and recovery patterns

When providing advice:
1. Be specific and actionable
2. Consider performance implications
3. Account for user experience and ease of use
4. Suggest concrete improvements with expected benefits
5. Identify potential issues before they become problems
6. Recommend best practices from workflow engineering

Focus on practical, implementable solutions that improve workflow efficiency, reliability, and user satisfaction."""

    def get_workflow_advice(self,
                          criteria: Dict[str, Any],
                          template_fields: Dict[str, Any],
                          recent_activity: List[Dict[str, Any]],
                          final_results: Dict[str, Any],
                          user_question: str) -> Dict[str, Any]:
        """Get comprehensive workflow advice based on current context."""

        try:
            # Enhance with RAG knowledge if available
            rag_context = ""
            if self.rag_enabled and self.knowledge_base:
                # Query knowledge base for relevant information
                relevant_docs = self.knowledge_base.query_knowledge_base(user_question)

                if relevant_docs:
                    rag_context = "\n\nRelevant Knowledge Base Information:\n"
                    for doc in relevant_docs:
                        rag_context += f"\n{doc['type'].title()}: {json.dumps(doc['content'], indent=2)}\n"

            # Prepare context data
            criteria_json = json.dumps(criteria, indent=2) if criteria else "No criteria defined"
            fields_json = json.dumps(template_fields, indent=2) if template_fields else "No template fields defined"
            activity_json = json.dumps(recent_activity[-5:], indent=2) if recent_activity else "No recent activity"
            results_json = json.dumps(final_results, indent=2) if final_results else "No recent results"

            # Get AI advice with RAG enhancement
            enhanced_user_question = f"{self.system_context}\n\nUser Question: {user_question}{rag_context}"

            with dspy.context(lm=dspy.LM(model="claude-3-sonnet-20240229")):
                response = self.context_analyzer(
                    workflow_criteria=criteria_json,
                    template_fields=fields_json,
                    recent_activity=activity_json,
                    final_results=results_json,
                    user_question=enhanced_user_question
                )

            # Store interaction for learning (for future RAG improvements)
            if self.rag_enabled:
                self._store_interaction_for_learning(user_question, response.advisor_response, {
                    "criteria": criteria,
                    "template_fields": template_fields,
                    "recent_activity": recent_activity,
                    "final_results": final_results
                })

            return {
                "success": True,
                "advice": response.advisor_response,
                "context_analyzed": {
                    "criteria_provided": bool(criteria),
                    "fields_analyzed": len(template_fields) if template_fields else 0,
                    "recent_executions": len(recent_activity) if recent_activity else 0,
                    "results_available": bool(final_results),
                    "rag_enhanced": self.rag_enabled and bool(rag_context)
                },
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "advice": f"I apologize, but I encountered an error while analyzing your workflow. Error: {str(e)}. Please try asking a more specific question or check your workflow configuration.",
                "timestamp": datetime.now().isoformat()
            }

    def optimize_workflow(self,
                         workflow_config: Dict[str, Any],
                         performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get workflow optimization recommendations."""

        try:
            # Analyze performance bottlenecks
            bottlenecks = self._identify_bottlenecks(performance_data)
            usage_patterns = self._analyze_usage_patterns(performance_data)

            with dspy.context(lm=dspy.LM(model="claude-3-sonnet-20240229")):
                response = self.optimizer(
                    current_workflow=json.dumps(workflow_config, indent=2),
                    bottlenecks=json.dumps(bottlenecks, indent=2),
                    usage_patterns=json.dumps(usage_patterns, indent=2)
                )

            return {
                "success": True,
                "optimization_plan": response.optimization_plan,
                "bottlenecks_identified": bottlenecks,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "optimization_plan": f"Unable to generate optimization plan: {str(e)}"
            }

    def advise_template_fields(self,
                              current_fields: Dict[str, Any],
                              validation_issues: List[str],
                              use_cases: List[str]) -> Dict[str, Any]:
        """Get template field configuration advice."""

        try:
            with dspy.context(lm=dspy.LM(model="claude-3-sonnet-20240229")):
                response = self.field_advisor(
                    current_fields=json.dumps(current_fields, indent=2),
                    validation_errors=json.dumps(validation_issues, indent=2),
                    use_cases=json.dumps(use_cases, indent=2)
                )

            return {
                "success": True,
                "field_recommendations": response.field_recommendations,
                "issues_addressed": len(validation_issues),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "field_recommendations": f"Unable to analyze template fields: {str(e)}"
            }

    def _identify_bottlenecks(self, performance_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks from execution data."""
        bottlenecks = []

        if not performance_data:
            return bottlenecks

        # Analyze step execution times
        if 'execution_results' in performance_data:
            for step_result in performance_data['execution_results']:
                processing_time = step_result.get('processing_time_ms', 0)

                # Flag steps taking longer than 500ms
                if processing_time > 500:
                    bottlenecks.append({
                        "type": "slow_step",
                        "step": step_result.get('step_name', 'unknown'),
                        "processing_time_ms": processing_time,
                        "severity": "high" if processing_time > 1000 else "medium"
                    })

        # Check overall processing time
        total_time = performance_data.get('summary', {}).get('total_processing_time_ms', 0)
        if total_time > 2000:
            bottlenecks.append({
                "type": "slow_overall",
                "total_time_ms": total_time,
                "severity": "high" if total_time > 5000 else "medium"
            })

        return bottlenecks

    def _analyze_usage_patterns(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze workflow usage patterns."""
        return {
            "execution_frequency": "unknown",
            "common_file_types": ["pdf", "docx", "md"],
            "average_file_count": performance_data.get('summary', {}).get('input_files_processed', 1),
            "success_rate": performance_data.get('summary', {}).get('success_rate', 1.0),
            "most_used_features": ["document_validation", "entity_extraction", "analysis"]
        }

    def get_quick_suggestions(self, context_type: str, context_data: Dict[str, Any]) -> List[str]:
        """Get quick suggestions based on context type."""
        suggestions = []

        if context_type == "criteria":
            suggestions = [
                "Consider adding file size limits to prevent processing large documents",
                "Review quality threshold settings - 0.85 is good for most use cases",
                "Add language detection to ensure document compatibility",
                "Define clear validation rules for document completeness"
            ]

        elif context_type == "template_fields":
            suggestions = [
                "Use enum values for fields with limited options to prevent errors",
                "Set appropriate default values for optional fields",
                "Add validation patterns for structured fields like review_id",
                "Consider making input_files required for all workflows"
            ]

        elif context_type == "performance":
            if context_data.get('total_processing_time_ms', 0) > 2000:
                suggestions.append("Consider parallel processing for multiple documents")
            if context_data.get('steps_completed', 0) < 5:
                suggestions.append("Review workflow configuration for incomplete steps")
            suggestions.append("Enable caching for frequently accessed RAG data")

        elif context_type == "results":
            success_rate = context_data.get('success_rate', 1.0)
            if success_rate < 0.9:
                suggestions.append("Investigate frequent failure patterns")
                suggestions.append("Add more robust error handling and recovery")
            suggestions.append("Monitor confidence scores to ensure quality output")

        return suggestions[:3]  # Return top 3 suggestions

    def _store_interaction_for_learning(self, question: str, advice: str, context: Dict[str, Any]):
        """Store interaction data for continuous learning and RAG improvement."""
        try:
            # Create learning data directory
            learning_dir = Path("tidyllm/workflows/ai_advisor/learning_data")
            learning_dir.mkdir(exist_ok=True)

            # Store interaction for future analysis and knowledge base updates
            interaction_data = {
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "advice": advice,
                "context": context,
                "interaction_id": f"INT_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:19]}"
            }

            # Save to learning file
            learning_file = learning_dir / f"interactions_{datetime.now().strftime('%Y%m')}.jsonl"
            with open(learning_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(interaction_data) + '\n')

        except Exception as e:
            # Silent fail for learning - don't impact user experience
            pass

    def analyze_learning_patterns(self) -> Dict[str, Any]:
        """Analyze interaction patterns for knowledge base improvements."""
        try:
            learning_dir = Path("tidyllm/workflows/ai_advisor/learning_data")
            if not learning_dir.exists():
                return {"status": "no_data", "patterns": []}

            interactions = []
            for learning_file in learning_dir.glob("interactions_*.jsonl"):
                with open(learning_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            interactions.append(json.loads(line))

            # Analyze patterns
            patterns = {
                "total_interactions": len(interactions),
                "common_question_types": self._analyze_question_types(interactions),
                "frequent_contexts": self._analyze_contexts(interactions),
                "improvement_opportunities": self._identify_improvements(interactions)
            }

            return {"status": "success", "patterns": patterns}

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _analyze_question_types(self, interactions: List[Dict]) -> Dict[str, int]:
        """Analyze common question types from interactions."""
        question_types = {}
        keywords = {
            "optimization": ["optimize", "performance", "speed", "faster"],
            "troubleshooting": ["error", "issue", "problem", "troubleshoot", "debug"],
            "best_practices": ["best practice", "recommend", "should", "better way"],
            "configuration": ["configure", "setup", "field", "template", "validation"],
            "integration": ["integrate", "connect", "rag", "system"]
        }

        for interaction in interactions:
            question = interaction.get("question", "").lower()
            for category, terms in keywords.items():
                if any(term in question for term in terms):
                    question_types[category] = question_types.get(category, 0) + 1
                    break
            else:
                question_types["general"] = question_types.get("general", 0) + 1

        return question_types

    def _analyze_contexts(self, interactions: List[Dict]) -> Dict[str, int]:
        """Analyze common context patterns."""
        context_patterns = {
            "has_criteria": 0,
            "has_template_fields": 0,
            "has_recent_activity": 0,
            "has_results": 0,
            "complex_workflows": 0  # workflows with >3 steps
        }

        for interaction in interactions:
            context = interaction.get("context", {})
            if context.get("criteria"):
                context_patterns["has_criteria"] += 1
            if context.get("template_fields"):
                context_patterns["has_template_fields"] += 1
            if context.get("recent_activity"):
                context_patterns["has_recent_activity"] += 1
            if context.get("final_results"):
                context_patterns["has_results"] += 1

            # Check workflow complexity
            criteria = context.get("criteria", {})
            if isinstance(criteria, dict):
                steps = criteria.get("steps", [])
                if len(steps) > 3:
                    context_patterns["complex_workflows"] += 1

        return context_patterns

    def _identify_improvements(self, interactions: List[Dict]) -> List[str]:
        """Identify opportunities for knowledge base improvements."""
        improvements = []

        # Analyze if certain question types need more knowledge
        question_types = self._analyze_question_types(interactions)

        if question_types.get("troubleshooting", 0) > question_types.get("optimization", 0) * 2:
            improvements.append("Add more troubleshooting guides to knowledge base")

        if question_types.get("configuration", 0) > len(interactions) * 0.3:
            improvements.append("Expand template field configuration examples")

        if question_types.get("integration", 0) > len(interactions) * 0.2:
            improvements.append("Add more RAG integration patterns and examples")

        # Add general improvement if we have enough data
        if len(interactions) > 50:
            improvements.append("Consider creating specialized knowledge bases for different workflow types")

        return improvements

    def get_specialized_advisor_for_workflow_type(self, workflow_type: str) -> 'WorkflowAIAdvisor':
        """Future: Get specialized advisor for specific workflow types."""
        # Placeholder for future specialization
        # Could return different advisor instances optimized for:
        # - Financial document processing
        # - Academic research analysis
        # - Regulatory compliance workflows
        # - Technical documentation processing
        return self

# Initialize global advisor instance
workflow_advisor = WorkflowAIAdvisor()