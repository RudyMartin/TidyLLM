"""
DSPy Advisor Service
====================

Workflow-specific DSPy service for AI-powered workflow advice.
Separate from DSPyService to avoid breaking existing functionality.
Uses same gateway hooks for MLflow integration.
"""

import dspy
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

# Import existing gateway pattern from DSPyService
from tidyllm.services.dspy_service import CorporateDSPyLM

logger = logging.getLogger(__name__)


class WorkflowAdvice(dspy.Signature):
    """DSPy signature for workflow advice generation."""

    criteria = dspy.InputField(desc="Workflow criteria and validation rules as JSON")
    template_fields = dspy.InputField(desc="Template field configuration as JSON")
    recent_activity = dspy.InputField(desc="Recent workflow execution data as JSON")
    final_results = dspy.InputField(desc="Latest workflow results as JSON")
    user_question = dspy.InputField(desc="User's specific question about the workflow")
    use_cases = dspy.InputField(desc="Workflow use cases and context")

    reasoning = dspy.OutputField(desc="Step-by-step analysis of the workflow situation")
    advice = dspy.OutputField(desc="Detailed advice with specific recommendations")
    context_analyzed = dspy.OutputField(desc="Summary of context data analyzed as JSON")


class DSPyAdvisor:
    """DSPy-powered workflow advisor with corporate gateway integration."""

    def __init__(self, model_name: str = "claude-3-sonnet"):
        self.model_name = model_name
        self.advisor_module = None
        self._configured = False

    def configure_advisor(self):
        """Configure DSPy with corporate gateway (same pattern as DSPyService)."""
        try:
            # Check if already configured in this thread
            current_lm = getattr(dspy.settings, 'lm', None)
            if current_lm is not None and hasattr(current_lm, 'model_name'):
                # Already configured - reuse existing configuration
                self.advisor_module = dspy.ChainOfThought(WorkflowAdvice)
                self._configured = True
                logger.info(f"DSPyAdvisor reusing existing DSPy configuration")
                return True

            # Use same gateway hook pattern as DSPyService
            corporate_lm = CorporateDSPyLM(model_name=self.model_name)
            dspy.configure(lm=corporate_lm)

            # Create workflow advice module
            self.advisor_module = dspy.ChainOfThought(WorkflowAdvice)
            self._configured = True

            logger.info(f"DSPyAdvisor configured with {self.model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to configure DSPyAdvisor: {e}")
            # Try to create module without reconfiguration
            try:
                self.advisor_module = dspy.ChainOfThought(WorkflowAdvice)
                self._configured = True
                logger.info("DSPyAdvisor using existing DSPy settings")
                return True
            except Exception as e2:
                logger.error(f"Failed to create advisor module: {e2}")
                return False

    def get_workflow_advice(self,
                          criteria: Dict = None,
                          template_fields: Dict = None,
                          recent_activity: List = None,
                          final_results: Dict = None,
                          user_question: str = "",
                          use_cases: List = None) -> Dict:
        """
        Get AI-powered workflow advice using DSPy.

        Args:
            criteria: Workflow criteria and rules
            template_fields: Template field configuration
            recent_activity: Recent execution data
            final_results: Latest results
            user_question: User's question
            use_cases: Workflow context and use cases

        Returns:
            Dict with advice, reasoning, and context analysis
        """

        if not self._configured:
            if not self.configure_advisor():
                return {
                    'success': False,
                    'advice': 'DSPy Advisor configuration failed. Please check system setup.',
                    'error': 'Configuration error'
                }

        try:
            # Prepare input data - convert to JSON strings for DSPy
            criteria_json = json.dumps(criteria or {}, indent=2)
            template_fields_json = json.dumps(template_fields or {}, indent=2)
            recent_activity_json = json.dumps(recent_activity or [], indent=2)
            final_results_json = json.dumps(final_results or {}, indent=2)
            use_cases_list = use_cases or ['general workflow', 'document processing']
            use_cases_str = ', '.join(use_cases_list)

            logger.info(f"DSPyAdvisor: About to call advisor_module with question: {user_question[:100]}...")
            logger.info(f"DSPyAdvisor: advisor_module type: {type(self.advisor_module)}")

            # Get DSPy prediction
            prediction = self.advisor_module(
                criteria=criteria_json,
                template_fields=template_fields_json,
                recent_activity=recent_activity_json,
                final_results=final_results_json,
                user_question=user_question,
                use_cases=use_cases_str
            )

            logger.info(f"DSPyAdvisor: Got prediction type: {type(prediction)}")
            logger.info(f"DSPyAdvisor: Prediction advice length: {len(getattr(prediction, 'advice', ''))}")

            # Parse context analysis
            context_analyzed = {}
            try:
                context_analyzed = json.loads(prediction.context_analyzed)
            except (json.JSONDecodeError, AttributeError):
                # Create context summary from input data
                context_analyzed = {
                    'criteria_provided': bool(criteria),
                    'fields_analyzed': len(template_fields or {}),
                    'recent_executions': len(recent_activity or []),
                    'results_available': bool(final_results),
                    'use_cases_count': len(use_cases_list)
                }

            return {
                'success': True,
                'advice': prediction.advice,
                'reasoning': prediction.reasoning,
                'context_analyzed': context_analyzed,
                'model_used': self.model_name
            }

        except Exception as e:
            logger.error(f"DSPy workflow advice failed: {e}")

            # Provide fallback advice based on question content
            fallback_advice = self._get_fallback_advice(user_question, use_cases_list)

            return {
                'success': False,
                'advice': fallback_advice,
                'error': str(e),
                'fallback': True
            }

    def _get_fallback_advice(self, user_question: str, use_cases: List[str]) -> str:
        """Generate fallback advice when DSPy fails."""

        question_lower = user_question.lower()

        if any(word in question_lower for word in ['template', 'field']):
            return """**Template Field Configuration Best Practices:**

🔧 **Configuration Tips:**
- Use clear, descriptive field names
- Set appropriate data types (text, number, date)
- Define required vs optional fields
- Add validation patterns for consistency

📋 **Validation Rules:**
- Format validation (email, phone, date formats)
- Length constraints (min/max characters)
- Value ranges for numerical fields
- Required field validation

💡 **Workflow-Specific:**
- Standardize metadata extraction fields
- Define clear analysis step fields
- Ensure consistent result formats"""

        elif any(word in question_lower for word in ['optimize', 'improve', 'performance']):
            return """**Workflow Optimization Strategies:**

⚡ **Performance:**
- Streamline processing steps
- Implement parallel processing
- Cache frequently used data
- Optimize field configurations

📊 **Quality:**
- Add comprehensive validation
- Implement error handling
- Create detailed logging
- Set up monitoring

🔄 **Process:**
- Standardize naming conventions
- Document workflow steps
- Create reusable templates
- Implement version control"""

        else:
            use_case_context = f" for {', '.join(use_cases)}" if use_cases else ""
            return f"""**General Workflow Advice{use_case_context}:**

🎯 **Best Practices:**
- Start with clear requirements
- Design modular, reusable components
- Implement proper error handling
- Test thoroughly before deployment

📋 **Quality Assurance:**
- Validate all inputs and outputs
- Document each step clearly
- Monitor execution performance
- Maintain audit trails

🔧 **Technical Excellence:**
- Follow established patterns
- Use existing services and utilities
- Implement proper logging
- Ensure scalability"""


# Global advisor instance
_advisor_instance = None

def get_advisor(model_name: str = "claude-3-sonnet") -> DSPyAdvisor:
    """Get or create global DSPyAdvisor instance."""
    global _advisor_instance

    if _advisor_instance is None or _advisor_instance.model_name != model_name:
        _advisor_instance = DSPyAdvisor(model_name=model_name)

    return _advisor_instance