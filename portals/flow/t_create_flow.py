"""
Create Flow Tab - Modular Component
Extracted from flow_creator_v3.py for easier maintenance
"""

import streamlit as st
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

def render_create_flow_tab(workflow_manager, registry):
    """Render the Create Flow tab content."""
    st.markdown("## ➕ **Create New Flow**")

    # Get available workflow types from registry
    workflow_types = registry.get_workflow_types()

    if not workflow_types or workflow_types == ["unknown"]:
        st.warning("⚠️ No workflow types found in registry. Using defaults.")
        workflow_types = ["mvr", "analysis", "rag_creation", "custom"]

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🎯 **Workflow Configuration**")

        # Basic workflow information
        workflow_name = st.text_input(
            "Workflow Name",
            placeholder="e.g., financial_risk_analysis",
            help="Unique name for your workflow"
        )

        workflow_type = st.selectbox(
            "Workflow Type",
            workflow_types,
            help="Choose the type that best fits your use case"
        )

        description = st.text_area(
            "Description",
            placeholder="Describe what this workflow does...",
            help="Brief description of the workflow's purpose"
        )

        # RAG Integration Options
        st.subheader("🧠 **RAG Integration**")
        rag_systems = ["ai_powered", "postgres", "intelligent", "sme", "dspy"]
        selected_rags = st.multiselect(
            "RAG Systems to Integrate",
            rag_systems,
            default=["ai_powered"],
            help="Choose which RAG systems to include"
        )

    with col2:
        st.subheader("📋 **Workflow Steps**")

        if 'workflow_steps' not in st.session_state:
            st.session_state.workflow_steps = [
                {"step_name": "initial_analysis", "step_type": "extraction", "template": ""},
                {"step_name": "detailed_review", "step_type": "analysis", "template": ""},
                {"step_name": "final_output", "step_type": "aggregation", "template": ""}
            ]

        steps = st.session_state.workflow_steps

        # Step configuration
        for i, step in enumerate(steps):
            with st.expander(f"Step {i+1}: {step['step_name']}", expanded=False):
                step['step_name'] = st.text_input(f"Step Name {i+1}", value=step['step_name'], key=f"step_name_{i}")
                step['step_type'] = st.selectbox(
                    f"Step Type {i+1}",
                    ["extraction", "analysis", "aggregation", "validation", "custom"],
                    index=["extraction", "analysis", "aggregation", "validation", "custom"].index(step['step_type']),
                    key=f"step_type_{i}"
                )
                step['template'] = st.text_input(f"Template {i+1}", value=step['template'], key=f"template_{i}")

        # Add/Remove steps
        col_add, col_remove = st.columns(2)
        with col_add:
            if st.button("➕ Add Step"):
                st.session_state.workflow_steps.append({
                    "step_name": f"step_{len(steps)+1}",
                    "step_type": "custom",
                    "template": ""
                })
                st.rerun()

        with col_remove:
            if st.button("➖ Remove Step") and len(steps) > 1:
                st.session_state.workflow_steps.pop()
                st.rerun()

    # Creation controls
    st.markdown("---")
    col_create, col_preview = st.columns([1, 1])

    with col_create:
        if st.button("🚀 **Create Workflow**", type="primary", use_container_width=True):
            if workflow_name and description:
                # Create workflow data structure
                workflow_data = {
                    "workflow_id": workflow_name,
                    "workflow_name": workflow_name.replace('_', ' ').title(),
                    "workflow_type": workflow_type,
                    "description": description,
                    "rag_integration": selected_rags,
                    "steps": st.session_state.workflow_steps,
                    "created_date": datetime.now().isoformat(),
                    "flow_encoding": f"@{workflow_name}#process!analyze@output"
                }

                # Save workflow
                try:
                    workflows_dir = Path("tidyllm/workflows/projects") / workflow_name
                    workflows_dir.mkdir(parents=True, exist_ok=True)

                    config_file = workflows_dir / "project_config.json"
                    from tidyllm.utils.clean_json_io import save_json
                    save_json(config_file, workflow_data)

                    # Create directory structure
                    (workflows_dir / "criteria").mkdir(exist_ok=True)
                    (workflows_dir / "templates").mkdir(exist_ok=True)
                    (workflows_dir / "inputs").mkdir(exist_ok=True)
                    (workflows_dir / "outputs").mkdir(exist_ok=True)
                    (workflows_dir / "resources").mkdir(exist_ok=True)

                    st.success(f"✅ **Created workflow**: {workflow_name}")
                    st.success(f"📁 **Location**: {workflows_dir}")
                    st.balloons()

                    # Clear form
                    del st.session_state.workflow_steps
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ **Creation failed**: {e}")
            else:
                st.warning("⚠️ Please fill in workflow name and description")

    with col_preview:
        if st.button("👁️ **Preview Configuration**", use_container_width=True):
            if workflow_name:
                st.json({
                    "workflow_id": workflow_name,
                    "workflow_type": workflow_type,
                    "description": description,
                    "rag_integration": selected_rags,
                    "steps": st.session_state.workflow_steps
                })
            else:
                st.info("Enter a workflow name to see preview")