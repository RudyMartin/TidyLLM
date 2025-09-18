"""
AI Advisor Tab - Modular Component
Extracted from flow_creator_v3.py for easier maintenance
"""

import streamlit as st
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

def render_ai_advisor_tab(workflow_manager, registry):
    """Render the AI Advisor page with chat interface."""
    st.header("🤖 AI Workflow Advisor")
    st.markdown("Get intelligent advice about your workflows from AI - ask about optimization, troubleshooting, or best practices")

    # Import the DSPyAdvisor
    try:
        from tidyllm.services.dspy_advisor import get_advisor
        advisor_available = True
    except ImportError as e:
        st.info(f"AI Advisor currently unavailable: {e}")
        advisor_available = False
        return

    # Context gathering section
    st.subheader("🔍 Workflow Context")

    col1, col2 = st.columns(2)

    with col1:
        # Get current workflow context
        workflows = registry.workflows
        selected_workflow = None

        if workflows:
            workflow_names = list(workflows.keys())

            # Set default to Alex QAQC if available
            default_index = 0
            if "alex_qaqc" in workflow_names:
                default_index = workflow_names.index("alex_qaqc") + 1  # +1 because of "All Workflows"

            selected_workflow_name = st.selectbox(
                "Select workflow for context",
                options=["All Workflows"] + workflow_names,
                index=default_index,
                help="Choose a specific workflow or analyze all workflows"
            )

            if selected_workflow_name != "All Workflows":
                selected_workflow = workflows[selected_workflow_name]

                # Show context checkboxes
                st.write("**Include in analysis:**")
                include_criteria = st.checkbox("📋 Criteria & Document Qualifiers", value=True)
                include_fields = st.checkbox("📝 Template Fields", value=True)
                include_activity = st.checkbox("📊 Recent Activity", value=True)
                include_results = st.checkbox("📈 Latest Results", value=True)
            else:
                include_criteria = include_fields = include_activity = include_results = True
        else:
            st.warning("⚠️ No workflows found. Create a workflow first to get context-aware advice.")
            include_criteria = include_fields = include_activity = include_results = False

    with col2:
        # Quick context insights
        if selected_workflow:
            st.write("**📋 Quick Context Preview:**")

            # Show workflow summary
            workflow_type = selected_workflow.get('workflow_type', 'unknown')
            steps = len(selected_workflow.get('steps', []))
            rag_systems = len(selected_workflow.get('rag_integration', []))

            st.write(f"• **Type**: {workflow_type}")
            st.write(f"• **Steps**: {steps}")
            st.write(f"• **RAG Systems**: {rag_systems}")

            # Template fields count
            template_fields = selected_workflow.get('template_fields', {})
            st.write(f"• **Template Fields**: {len(template_fields)}")

    # Chat interface
    st.subheader("💬 AI Chat Interface")

    # Initialize chat history
    if "ai_advisor_messages" not in st.session_state:
        st.session_state.ai_advisor_messages = [
            {
                "role": "assistant",
                "content": "Hello! 👋 I'm your AI Workflow Advisor. I can help you optimize workflows, troubleshoot issues, and suggest best practices. What would you like to know about your workflows?"
            }
        ]

    # Chat history display
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.ai_advisor_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Quick suggestion buttons
    st.write("**🚀 Quick Questions:**")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("⚡ Optimize Performance", help="Get performance optimization tips"):
            quick_question = "How can I optimize the performance of my workflow? What are the main bottlenecks to look for?"
            st.session_state.ai_advisor_pending_question = quick_question

    with col2:
        if st.button("📝 Improve Template Fields", help="Get field configuration advice"):
            quick_question = "How can I improve my template field configuration? What validation rules should I add?"
            st.session_state.ai_advisor_pending_question = quick_question

    with col3:
        if st.button("🔧 Troubleshoot Issues", help="Get troubleshooting help"):
            quick_question = "My workflow is having issues. Can you help me troubleshoot common problems?"
            st.session_state.ai_advisor_pending_question = quick_question

    with col4:
        if st.button("📋 Best Practices", help="Get workflow best practices"):
            quick_question = "What are the best practices for designing efficient document processing workflows?"
            st.session_state.ai_advisor_pending_question = quick_question

    # Chat input
    if prompt := st.chat_input("Ask about your workflow..."):
        _handle_ai_advisor_chat(prompt, selected_workflow, include_criteria, include_fields, include_activity, include_results, advisor_available)

    # Handle pending questions from quick buttons
    if hasattr(st.session_state, 'ai_advisor_pending_question'):
        pending_question = st.session_state.ai_advisor_pending_question
        # Clear the pending question FIRST to prevent infinite loop
        del st.session_state.ai_advisor_pending_question

        _handle_ai_advisor_chat(
            pending_question,
            selected_workflow,
            include_criteria, include_fields, include_activity, include_results,
            advisor_available
        )

    # Context summary sidebar
    with st.sidebar:
        if selected_workflow:
            st.subheader("📊 Workflow Context")

            # Show what context will be included
            context_items = []
            if include_criteria:
                context_items.append("✅ Criteria & qualifiers")
            if include_fields:
                context_items.append("✅ Template fields")
            if include_activity:
                context_items.append("✅ Recent activity")
            if include_results:
                context_items.append("✅ Latest results")

            if context_items:
                st.write("**Included in AI analysis:**")
                for item in context_items:
                    st.write(item)
            else:
                st.write("❌ No context selected")

            # Quick suggestions based on context
            suggestions = _get_quick_workflow_suggestions(selected_workflow)
            if suggestions:
                st.write("**💡 Quick Suggestions:**")
                for suggestion in suggestions:
                    st.write(f"• {suggestion}")

def _handle_ai_advisor_chat(user_input: str, selected_workflow: Dict,
                           include_criteria: bool, include_fields: bool,
                           include_activity: bool, include_results: bool,
                           advisor_available: bool):
    """Handle AI advisor chat interaction."""

    # Add user message to chat
    st.session_state.ai_advisor_messages.append({
        "role": "user",
        "content": user_input
    })

    if not advisor_available:
        response = "I apologize, but the AI Advisor system is currently unavailable. Please check the system configuration and try again."
    else:
        # Gather context data
        criteria = {}
        template_fields = {}
        recent_activity = []
        final_results = {}
        use_cases = []

        if selected_workflow:
            if include_criteria:
                criteria = _load_workflow_criteria(selected_workflow)

            if include_fields:
                template_fields = selected_workflow.get('template_fields', {})

            if include_activity:
                recent_activity = _get_recent_workflow_activity()

            if include_results:
                final_results = _get_latest_workflow_results()

            # Extract use cases from workflow context
            workflow_desc = selected_workflow.get('description', '')
            workflow_type = selected_workflow.get('workflow_type', selected_workflow.get('template', {}).get('workflow_type', 'unknown'))

            if 'pdf' in workflow_desc.lower() or 'document' in workflow_desc.lower():
                use_cases.append('document processing')
            if 'analysis' in workflow_desc.lower():
                use_cases.append('analysis workflows')
            if 'sequential' in workflow_desc.lower():
                use_cases.append('sequential processing')
            if workflow_type == 'analysis':
                use_cases.append('multi-step analysis')
            elif workflow_type == 'template':
                use_cases.append('template workflows')

        # Default fallback use cases
        if not use_cases:
            use_cases = ['general workflow', 'document processing']

        # Get AI response using DSPyAdvisor
        try:
            from tidyllm.services.dspy_advisor import get_advisor

            # Ensure use_cases has default values
            if not use_cases:
                use_cases = ['document processing', 'quality assurance', 'data validation']

            with st.spinner("🤖 AI is analyzing your workflow..."):
                # Get DSPyAdvisor instance
                advisor = get_advisor(model_name="claude-3-sonnet")

                # Call with all expected parameters
                advice_result = advisor.get_workflow_advice(
                    criteria=criteria,
                    template_fields=template_fields,
                    recent_activity=recent_activity,
                    final_results=final_results,
                    user_question=user_input,
                    use_cases=use_cases
                )

            if advice_result.get('success', False):
                response = advice_result['advice']

                # Add context info
                context_info = advice_result.get('context_analyzed', {})
                # Defensive check for context_info type
                if isinstance(context_info, dict):
                    if any(context_info.values()):
                        context_summary = []
                        if context_info.get('criteria_provided'):
                            context_summary.append("criteria")
                        if context_info.get('fields_analyzed', 0) > 0:
                            context_summary.append(f"{context_info['fields_analyzed']} template fields")
                        if context_info.get('recent_executions', 0) > 0:
                            context_summary.append(f"{context_info['recent_executions']} recent executions")
                        if context_info.get('results_available'):
                            context_summary.append("latest results")

                        if context_summary:
                            response += f"\n\n*Analysis based on: {', '.join(context_summary)}*"
            else:
                response = advice_result.get('advice', 'Sorry, I encountered an error while analyzing your workflow.')

        except Exception as e:
            # Provide helpful fallback responses based on the question
            user_input_lower = user_input.lower()
            if "template" in user_input_lower and "field" in user_input_lower:
                response = """**Template Field Configuration Best Practices:**

🔧 **Configuration Tips:**
- Use clear, descriptive field names
- Set appropriate data types (text, number, date)
- Define required vs optional fields
- Add validation patterns for consistency

📋 **Validation Rules to Consider:**
- Format validation (email, phone, date formats)
- Length constraints (min/max characters)
- Value ranges for numerical fields
- Required field validation
- Business logic validation

💡 **Alex QAQC Specific:**
- Metadata extraction fields should be standardized
- Analysis step fields need clear definitions
- Results aggregation requires consistent formats
- Recording questions should have structured formats

*Note: For detailed workflow analysis, please check your workflow advisor configuration.*"""

            elif "improve" in user_input_lower or "optimize" in user_input_lower:
                response = """**Workflow Improvement Suggestions:**

⚡ **Performance Optimization:**
- Review and streamline processing steps
- Implement parallel processing where possible
- Cache frequently used data
- Optimize template field configurations

📊 **Quality Improvements:**
- Add comprehensive validation rules
- Implement error handling
- Create detailed logging
- Set up monitoring and alerts

🔄 **Process Enhancement:**
- Standardize naming conventions
- Document all workflow steps
- Create reusable templates
- Implement version control

*For specific recommendations based on your workflow data, please ensure the workflow advisor is properly configured.*"""

            elif "flow" in user_input_lower and "about" in user_input_lower:
                response = """**Alex QAQC Workflow Overview:**

🎯 **Purpose:** Quality Assurance and Quality Control for data validation

📁 **Structure:**
- **Criteria:** Validation rules and quality standards
- **Templates:** Step-by-step process templates
- **Inputs:** Documents and data for validation
- **Outputs:** Processed results and reports

🔄 **Process Flow:**
1. Metadata extraction
2. Analysis steps execution
3. Results aggregation
4. Quality control recording

This workflow ensures data quality and generates comprehensive validation reports."""

            else:
                response = f"I apologize, but I encountered a technical error: {str(e)}. However, I can still help you with general workflow questions about the Alex QAQC workflow. Try asking about template fields, validation rules, or workflow optimization."

    # Add AI response to chat
    st.session_state.ai_advisor_messages.append({
        "role": "assistant",
        "content": response
    })

    # Trigger rerun to show new messages
    st.rerun()

def _load_workflow_criteria(workflow: Dict) -> Dict:
    """Load criteria for the specified workflow."""
    try:
        from tidyllm.utils.clean_json_io import load_json

        # Try to load criteria.json from the workflow's criteria directory
        workflow_name = workflow.get('workflow_name', '').lower().replace(' ', '_')
        criteria_file = Path(f"tidyllm/workflows/projects/{workflow_name}/criteria/criteria.json")

        if criteria_file.exists():
            return load_json(criteria_file)
        else:
            # Try the templates directory as fallback
            criteria_file = Path("tidyllm/workflows/projects/templates/criteria/criteria.json")
            if criteria_file.exists():
                return load_json(criteria_file)
    except Exception:
        pass

    return {}

def _get_recent_workflow_activity() -> List[Dict]:
    """Get recent workflow activity and executions."""
    activity = []

    try:
        # Load recent test results
        results_dir = Path("tidyllm/workflows/projects/templates/outputs")
        if results_dir.exists():
            result_files = list(results_dir.glob("final_REV*.json"))
            result_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            for result_file in result_files[:3]:  # Last 3 executions
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        result_data = json.load(f)

                    activity.append({
                        "type": "workflow_execution",
                        "timestamp": datetime.fromtimestamp(result_file.stat().st_mtime).isoformat(),
                        "file": result_file.name,
                        "summary": result_data.get('summary', {}),
                        "status": "completed"
                    })
                except Exception:
                    continue
    except Exception:
        pass

    return activity

def _get_latest_workflow_results() -> Dict:
    """Get the latest workflow execution results."""
    try:
        from tidyllm.utils.clean_json_io import load_json

        results_dir = Path("tidyllm/workflows/projects/templates/outputs")
        if results_dir.exists():
            result_files = list(results_dir.glob("final_REV*.json"))
            if result_files:
                # Get most recent file
                latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
                return load_json(latest_file)
    except Exception:
        pass

    return {}

def _get_quick_workflow_suggestions(workflow: Dict) -> List[str]:
    """Get quick suggestions for the workflow."""
    suggestions = []

    # Check template fields
    template_fields = workflow.get('template_fields', {})
    if len(template_fields) < 3:
        suggestions.append("Consider adding more template fields for better customization")

    # Check RAG integration
    rag_systems = workflow.get('rag_integration', [])
    if len(rag_systems) < 2:
        suggestions.append("Add multiple RAG systems for better analysis coverage")

    # Check steps
    steps = workflow.get('steps', [])
    if len(steps) < 4:
        suggestions.append("Consider adding more processing steps for comprehensive analysis")

    return suggestions[:3]