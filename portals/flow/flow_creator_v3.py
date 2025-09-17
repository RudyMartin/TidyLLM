"""
TidyLLM Flow Creator V3 Portal
==============================

Next-generation workflow creation and management portal that integrates:
- Create Flow: Build new workflows from templates or scratch
- Existing Flow: Browse and manage existing workflows
- Flow Designer: Advanced workflow configuration with RAG integration
- Workflow Registry: Integration with comprehensive workflow definitions

Architecture: Portal → WorkflowManager → RAG Ecosystem Integration
"""

import streamlit as st
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import traceback

# Import TidyLLM components
try:
    from tidyllm.services.unified_rag_manager import UnifiedRAGManager, RAGSystemType
    from tidyllm.services.unified_flow_manager import UnifiedFlowManager, WorkflowSystemType, WorkflowStatus
    from tidyllm.infrastructure.session.unified import UnifiedSessionManager
except ImportError:
    st.error("ERROR: TidyLLM components not available. Please check installation.")
    st.stop()


class WorkflowRegistry:
    """Manages the comprehensive workflow registry with 17+ workflow definitions."""

    def __init__(self):
        self.workflows_base_path = Path("tidyllm/workflows/definitions/workflows")
        self.workflow_registry_path = Path("tidyllm/workflows/workflow_registry")
        self.workflows = self._load_workflow_registry()

    def _load_workflow_registry(self) -> Dict[str, Any]:
        """Load workflows from the project folder structure (criteria, outputs, resources, templates)."""
        try:
            workflows = {}

            # Scan workflow project directories
            if self.workflows_base_path.exists():
                for project_dir in self.workflows_base_path.iterdir():
                    if project_dir.is_dir() and project_dir.name != "__pycache__":
                        workflow_data = self._load_project_workflow(project_dir)
                        if workflow_data:
                            workflows[project_dir.name] = workflow_data

            # If no workflows found, use defaults
            if not workflows:
                workflows = self._create_default_registry()

            return workflows

        except Exception as e:
            st.warning(f"WARNING: Could not load workflow registry: {e}")
            return self._create_default_registry()

    def _load_project_workflow(self, project_dir: Path) -> Dict[str, Any]:
        """Load workflow data from project directory structure."""
        try:
            workflow_data = {
                "workflow_id": project_dir.name,
                "workflow_name": project_dir.name.replace('_', ' ').title(),
                "workflow_type": self._detect_workflow_type(project_dir),
                "description": self._load_project_description(project_dir),
                "project_structure": {
                    "criteria": (project_dir / "criteria").exists(),
                    "outputs": (project_dir / "outputs").exists(),
                    "resources": (project_dir / "resources").exists(),
                    "templates": (project_dir / "templates").exists(),
                    "inputs": (project_dir / "inputs").exists()
                },
                "rag_integration": self._detect_rag_integration(project_dir),
                "flow_encoding": f"@{project_dir.name}#process!analyze@output"
            }

            # Load criteria if available
            criteria_file = project_dir / f"{project_dir.name}_criteria.json"
            if criteria_file.exists():
                with open(criteria_file, 'r', encoding='utf-8') as f:
                    criteria_data = json.load(f)
                    workflow_data["criteria"] = criteria_data

            return workflow_data

        except Exception as e:
            st.warning(f"WARNING: Could not load project {project_dir.name}: {e}")
            return None

    def _detect_workflow_type(self, project_dir: Path) -> str:
        """Detect workflow type based on project name and contents."""
        name = project_dir.name.lower()
        if "mvr" in name:
            return "mvr"
        elif "code_review" in name or "review" in name:
            return "code_review"
        elif "analysis" in name:
            return "analysis"
        elif "rag" in name:
            return "rag_creation"
        else:
            return "custom"

    def _load_project_description(self, project_dir: Path) -> str:
        """Load project description from README.md if available."""
        readme_file = project_dir / "README.md"
        if readme_file.exists():
            try:
                with open(readme_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Extract first paragraph as description
                    lines = content.split('\n')
                    for line in lines:
                        if line.strip() and not line.startswith('#'):
                            return line.strip()
            except:
                pass
        return f"Workflow for {project_dir.name.replace('_', ' ')}"

    def _detect_rag_integration(self, project_dir: Path) -> List[str]:
        """Detect RAG integration based on project files."""
        rag_systems = []

        # Check for DSPy integration
        if any(f.name.endswith('_dspy_signature.py') for f in project_dir.iterdir()):
            rag_systems.append("dspy")

        # Check for SME integration
        if any(f.name.endswith('_sme.py') for f in project_dir.iterdir()):
            rag_systems.append("sme")

        # Default RAG systems for most workflows
        if not rag_systems:
            rag_systems = ["ai_powered", "intelligent"]

        return rag_systems

    def _create_default_registry(self) -> Dict[str, Any]:
        """Create default workflow registry if none exists."""
        return {
            "process_mvr": {
                "workflow_id": "process_mvr",
                "workflow_name": "Process MVR",
                "workflow_type": "mvr",
                "description": "4-stage MVR analysis workflow with domain RAG peer review",
                "rag_integration": ["ai_powered", "postgres", "intelligent"],
                "criteria": {
                    "scoring_rubric": {"accuracy": 0.3, "completeness": 0.3, "compliance": 0.4},
                    "weight_scheme": {"primary": 0.6, "secondary": 0.3, "tertiary": 0.1}
                },
                "flow_encoding": "@mvr#process!extract@compliance_data"
            },
            "financial_analysis": {
                "workflow_id": "financial_analysis",
                "workflow_name": "Financial Analysis",
                "workflow_type": "analysis",
                "description": "Financial model risk analysis with regulatory compliance",
                "rag_integration": ["ai_powered", "sme", "dspy"],
                "criteria": {
                    "scoring_rubric": {"risk_assessment": 0.4, "compliance": 0.3, "validation": 0.3}
                },
                "flow_encoding": "@financial#analyze!risk@model_validation"
            },
            "domain_rag": {
                "workflow_id": "domain_rag",
                "workflow_name": "Domain RAG Creation",
                "workflow_type": "rag_creation",
                "description": "Create domain-specific RAG systems using multiple orchestrators",
                "rag_integration": ["ai_powered", "postgres", "intelligent", "sme", "dspy"],
                "criteria": {
                    "scoring_rubric": {"retrieval_quality": 0.4, "response_accuracy": 0.4, "performance": 0.2}
                },
                "flow_encoding": "@rag#create!embed@vector_search"
            }
        }

    def get_workflow_types(self) -> List[str]:
        """Get list of available workflow types."""
        return list(set(w.get("workflow_type", "unknown") for w in self.workflows.values()))

    def get_workflows_by_type(self, workflow_type: str) -> Dict[str, Any]:
        """Get workflows filtered by type."""
        return {
            wid: workflow for wid, workflow in self.workflows.items()
            if workflow.get("workflow_type") == workflow_type
        }

    def get_rag_compatible_workflows(self) -> Dict[str, Any]:
        """Get workflows that integrate with RAG systems."""
        return {
            wid: workflow for wid, workflow in self.workflows.items()
            if workflow.get("rag_integration")
        }


class WorkflowManager:
    """Manages workflow creation, deployment, and integration with RAG ecosystem."""

    def __init__(self):
        # Use UnifiedFlowManager for all workflow operations
        self.flow_manager = UnifiedFlowManager(auto_load_credentials=True)
        self.registry = WorkflowRegistry()

        # Legacy compatibility
        self.usm = self.flow_manager.usm
        self.rag_manager = self.flow_manager.rag_manager
        self.templates_dir = self.flow_manager.templates_dir
        self.active_workflows_dir = self.flow_manager.workflows_dir

    def get_available_rag_systems(self) -> Dict[str, bool]:
        """Check availability of RAG systems for workflow integration."""
        return self.flow_manager.rag_manager.get_available_systems() if self.flow_manager.rag_manager else {}

    def get_available_workflow_systems(self) -> Dict[str, bool]:
        """Check availability of workflow systems."""
        return self.flow_manager.get_available_systems()

    def create_workflow_from_template(self, template_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create new workflow instance from template."""
        try:
            # Convert template_id to WorkflowSystemType
            system_type = WorkflowSystemType(template_id)
            return self.flow_manager.create_workflow(system_type, config)
        except ValueError:
            return {"success": False, "error": f"Unknown workflow type: {template_id}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def deploy_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Deploy workflow with RAG system integration."""
        return self.flow_manager.deploy_workflow(workflow_id)

    def get_active_workflows(self) -> List[Dict[str, Any]]:
        """Get list of active workflows."""
        return self.flow_manager.get_all_workflows()

    def get_workflow_health(self) -> Dict[str, Any]:
        """Get workflow system health status."""
        return self.flow_manager.health_check()

    def get_workflow_metrics(self) -> Dict[str, Any]:
        """Get workflow performance metrics."""
        return self.flow_manager.get_performance_metrics()


class FlowCreatorV3Portal:
    """Main portal for Flow Creator V3 with Create Flow, Existing Flow, and Flow Designer capabilities."""

    def __init__(self):
        self.workflow_manager = WorkflowManager()
        self.registry = WorkflowRegistry()

    def render_portal(self):
        """Render the main Flow Creator V3 portal."""
        st.set_page_config(
            page_title="TidyLLM Flow Creator V3",
            page_icon="+",
            layout="wide"
        )

        st.title("+ TidyLLM Flow Creator V3")
        st.markdown("**Next-generation workflow creation and deployment with RAG ecosystem integration**")

        # Status indicators
        self._render_status_indicators()

        # Main tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "+ Create Flow",
            "= Existing Flows",
            "* Flow Designer",
            "> Test Runner",
            "? AI Advisor",
            "^ Workflow Monitor",
            "# Health Dashboard"
        ])

        with tab1:
            self._render_create_flow_page()

        with tab2:
            self._render_existing_flows_page()

        with tab3:
            self._render_flow_designer_page()

        with tab4:
            self._render_test_runner_page()

        with tab5:
            self._render_ai_advisor_page()

        with tab6:
            self._render_workflow_monitor_page()

        with tab7:
            self._render_health_dashboard_page()

    def _render_status_indicators(self):
        """Render system status indicators."""
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            # RAG Systems Status
            rag_status = self.workflow_manager.get_available_rag_systems()
            available_rags = sum(1 for available in rag_status.values() if available)
            total_rags = len(rag_status)

            if available_rags > 0:
                st.success(f"OK: RAG Systems: {available_rags}/{total_rags}")
            else:
                st.error("ERROR: No RAG Systems Available")

        with col2:
            # Workflow Systems Status
            workflow_status = self.workflow_manager.get_available_workflow_systems()
            available_workflows = sum(1 for available in workflow_status.values() if available)
            total_workflow_types = len(workflow_status)

            if available_workflows > 0:
                st.success(f"+ Workflow Types: {available_workflows}/{total_workflow_types}")
            else:
                st.error("ERROR: No Workflow Systems Available")

        with col3:
            # Active Workflows
            active_workflows = self.workflow_manager.get_active_workflows()
            if active_workflows:
                st.info(f"🏃 Active: {len(active_workflows)} workflows")
            else:
                st.info("SLEEP: No active workflows")

        with col4:
            # UFM Health Status
            try:
                health = self.workflow_manager.get_workflow_health()
                if health.get("success") and health.get("overall_healthy"):
                    st.success("OK: UFM Healthy")
                else:
                    st.warning("WARNING: UFM Issues")
            except:
                st.error("ERROR: UFM Error")

    def _render_create_flow_page(self):
        """Render the Create Flow page."""
        st.header("+ Create New Workflow")
        st.markdown("Build workflows from templates or create custom flows")

        # Template Selection
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("= Select Template")

            workflow_types = self.registry.get_workflow_types()
            selected_type = st.selectbox(
                "Workflow Type",
                ["All"] + workflow_types,
                index=0
            )

            # Filter workflows by type
            if selected_type == "All":
                available_workflows = self.registry.workflows
            else:
                available_workflows = self.registry.get_workflows_by_type(selected_type)

            selected_template = st.selectbox(
                "Workflow Template",
                list(available_workflows.keys()),
                format_func=lambda x: available_workflows[x]["workflow_name"]
            )

            if selected_template:
                template = available_workflows[selected_template]

                with st.expander("FILE: Template Details"):
                    st.markdown(f"**Name:** {template['workflow_name']}")
                    st.markdown(f"**Type:** {template['workflow_type']}")
                    st.markdown(f"**Description:** {template['description']}")

                    if template.get("rag_integration"):
                        st.markdown(f"**RAG Integration:** {', '.join(template['rag_integration'])}")

                    if template.get("flow_encoding"):
                        st.markdown(f"**Flow Encoding:** `{template['flow_encoding']}`")

                    # Show project structure
                    if template.get("project_structure"):
                        st.markdown("**DIR: Project Structure:**")
                        structure = template["project_structure"]
                        for folder, exists in structure.items():
                            icon = "OK:" if exists else "ERROR:"
                            st.markdown(f"  {icon} {folder}/")

                    # Show criteria if available
                    if template.get("criteria"):
                        with st.expander("= Scoring Criteria"):
                            criteria = template["criteria"]
                            if isinstance(criteria, dict):
                                if "scoring_rubric" in criteria:
                                    st.markdown("**Scoring Rubric:**")
                                    for metric, weight in criteria["scoring_rubric"].items():
                                        st.markdown(f"- {metric}: {weight:.1%}")
                                st.json(criteria)

        with col2:
            st.subheader("TOOL: Configure Workflow")

            if selected_template:
                template = available_workflows[selected_template]

                with st.form("create_workflow_form"):
                    # Basic Configuration
                    workflow_name = st.text_input(
                        "Workflow Name",
                        value=f"Custom {template['workflow_name']}",
                        help="Human-readable name for your workflow instance"
                    )

                    domain = st.text_input(
                        "Domain",
                        value="general",
                        help="Domain for RAG system integration (e.g., financial, legal, technical)"
                    )

                    description = st.text_area(
                        "Description",
                        value=template.get("description", ""),
                        help="Brief description of what this workflow will do"
                    )

                    # RAG System Selection
                    st.markdown("**AI: RAG System Integration**")
                    rag_status = self.workflow_manager.get_available_rag_systems()
                    suggested_rags = template.get("rag_integration", [])

                    selected_rags = []
                    for rag_type in RAGSystemType:
                        rag_name = rag_type.value
                        is_available = rag_status.get(rag_name, False)
                        is_suggested = rag_name in suggested_rags

                        # Default selection logic
                        default_selected = is_suggested and is_available

                        col_check, col_status = st.columns([3, 1])
                        with col_check:
                            if st.checkbox(
                                f"{rag_name.replace('_', ' ').title()}",
                                value=default_selected,
                                disabled=not is_available,
                                key=f"rag_{rag_name}"
                            ):
                                selected_rags.append(rag_name)

                        with col_status:
                            if is_available:
                                if is_suggested:
                                    st.success("OK: ⭐")
                                else:
                                    st.success("OK:")
                            else:
                                st.error("ERROR:")

                    # Document Upload Section
                    st.markdown("**FILE: Document Upload (up to 20 files):**")

                    uploaded_files = st.file_uploader(
                        "Choose input documents for workflow processing",
                        type=['pdf', 'docx', 'txt', 'md', 'csv', 'json', 'xlsx'],
                        accept_multiple_files=True,
                        help="Supports PDF, Word, text, markdown, CSV, JSON, and Excel files. Maximum 20 files.",
                        key="workflow_file_upload"
                    )

                    file_info = None
                    if uploaded_files:
                        if len(uploaded_files) > 20:
                            st.error("ERROR: Maximum 20 files allowed. Please remove some files.")
                        else:
                            st.success(f"OK: {len(uploaded_files)} files uploaded")

                            # Display file summary
                            total_size = 0
                            file_list = []
                            for file in uploaded_files:
                                file_size = len(file.read()) / (1024 * 1024)  # MB
                                file.seek(0)  # Reset file pointer
                                total_size += file_size
                                file_list.append({
                                    "name": file.name,
                                    "size_mb": round(file_size, 2),
                                    "type": file.type or "unknown"
                                })

                            with st.expander(f"FILE: Upload Details ({total_size:.1f} MB total)"):
                                for file_data in file_list:
                                    st.markdown(f"- **{file_data['name']}** ({file_data['size_mb']} MB, {file_data['type']})")

                            file_info = {
                                "files": file_list,
                                "total_count": len(uploaded_files),
                                "total_size_mb": round(total_size, 2)
                            }

                    # Advanced Settings
                    with st.expander("TOOL: Advanced Settings"):
                        priority = st.selectbox("Priority", ["low", "medium", "high"], index=1)
                        auto_deploy = st.checkbox("Auto-deploy after creation", value=True)
                        enable_monitoring = st.checkbox("Enable monitoring", value=True)

                    # Submit
                    submitted = st.form_submit_button("> Create Workflow", width="stretch")

                    if submitted:
                        config = {
                            "name": workflow_name,
                            "domain": domain,
                            "description": description,
                            "rag_systems": selected_rags,
                            "priority": priority,
                            "auto_deploy": auto_deploy,
                            "enable_monitoring": enable_monitoring,
                            "uploaded_files": file_info
                        }

                        # Save uploaded files to project inputs folder if files were uploaded
                        if uploaded_files and len(uploaded_files) <= 20:
                            import tempfile
                            import os
                            from pathlib import Path

                            # Create project-specific inputs directory
                            project_inputs_dir = Path(f"tidyllm/workflows/definitions/workflows/{selected_template}/inputs")
                            project_inputs_dir.mkdir(parents=True, exist_ok=True)

                            saved_files = []
                            for file in uploaded_files:
                                # Save file to inputs directory
                                file_path = project_inputs_dir / file.name
                                with open(file_path, "wb") as f:
                                    f.write(file.read())
                                saved_files.append(str(file_path))
                                file.seek(0)  # Reset for potential reuse

                            config["input_file_paths"] = saved_files
                            st.success(f"OK: {len(saved_files)} files saved to {project_inputs_dir}")

                        result = self.workflow_manager.create_workflow_from_template(
                            selected_template, config
                        )

                        if result["success"]:
                            st.success(f"OK: Workflow created: {result['workflow_id']}")

                            if auto_deploy:
                                with st.spinner("Deploying workflow..."):
                                    deploy_result = self.workflow_manager.deploy_workflow(result['workflow_id'])

                                if deploy_result["success"]:
                                    st.success("> Workflow deployed successfully!")

                                    # Show deployment details
                                    with st.expander("^ Deployment Details"):
                                        for rag_deploy in deploy_result["rag_deployment"]:
                                            status_icon = {
                                                "deployed": "OK:",
                                                "failed": "ERROR:",
                                                "unavailable": "WARNING:",
                                                "error": "BOOM:"
                                            }.get(rag_deploy["status"], "?")

                                            st.markdown(f"{status_icon} **{rag_deploy['rag_type']}**: {rag_deploy['status']}")
                                else:
                                    st.error(f"ERROR: Deployment failed: {deploy_result.get('error')}")
                        else:
                            st.error(f"ERROR: Creation failed: {result.get('error')}")

    def _render_existing_flows_page(self):
        """Render the Existing Flows page."""
        st.header("= Existing Workflows")
        st.markdown("Browse and manage existing workflows")

        # Active workflows
        active_workflows = self.workflow_manager.get_active_workflows()

        if not active_workflows:
            st.info("SLEEP: No active workflows found. Create your first workflow in the 'Create Flow' tab.")
            return

        # Workflow filters
        col1, col2, col3 = st.columns(3)

        with col1:
            status_filter = st.selectbox(
                "Filter by Status",
                ["All", "created", "deployed", "running", "completed", "error"]
            )

        with col2:
            type_filter = st.selectbox(
                "Filter by Type",
                ["All"] + list(set(w.get("template", {}).get("workflow_type", "unknown") for w in active_workflows))
            )

        with col3:
            search_term = st.text_input("SEARCH: Search workflows", placeholder="Enter workflow name or ID...")

        # Filter workflows
        filtered_workflows = active_workflows

        if status_filter != "All":
            filtered_workflows = [w for w in filtered_workflows if w.get("status") == status_filter]

        if type_filter != "All":
            filtered_workflows = [w for w in filtered_workflows if w.get("template", {}).get("workflow_type") == type_filter]

        if search_term:
            filtered_workflows = [
                w for w in filtered_workflows
                if search_term.lower() in w.get("workflow_id", "").lower()
                or search_term.lower() in w.get("config", {}).get("name", "").lower()
            ]

        # Display workflows
        st.markdown(f"**Found {len(filtered_workflows)} workflows**")

        for workflow in filtered_workflows:
            with st.expander(f"+ {workflow.get('config', {}).get('name', workflow['workflow_id'])}", expanded=False):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"**ID:** `{workflow['workflow_id']}`")
                    st.markdown(f"**Status:** {workflow.get('status', 'unknown')}")
                    st.markdown(f"**Created:** {workflow.get('created_at', 'unknown')}")
                    st.markdown(f"**Template:** {workflow.get('template', {}).get('workflow_name', 'unknown')}")
                    st.markdown(f"**Domain:** {workflow.get('config', {}).get('domain', 'general')}")

                    if workflow.get("rag_systems"):
                        st.markdown(f"**RAG Systems:** {', '.join(workflow['rag_systems'])}")

                with col2:
                    # Action buttons
                    if workflow.get("status") == "created":
                        if st.button(f"> Deploy", key=f"deploy_{workflow['workflow_id']}"):
                            with st.spinner("Deploying..."):
                                result = self.workflow_manager.deploy_workflow(workflow['workflow_id'])

                            if result["success"]:
                                st.success("OK: Deployed!")
                                st.rerun()
                            else:
                                st.error(f"ERROR: Failed: {result.get('error')}")

                    if st.button(f"^ Monitor", key=f"monitor_{workflow['workflow_id']}"):
                        st.info("+ Monitoring functionality coming soon...")

                    if st.button(f"🗑️ Delete", key=f"delete_{workflow['workflow_id']}"):
                        st.warning("WARNING: Delete confirmation needed")

                # RAG deployment status
                if workflow.get("rag_deployment"):
                    st.markdown("**AI: RAG Deployment Status:**")
                    for rag_deploy in workflow["rag_deployment"]:
                        status_icon = {
                            "deployed": "OK:",
                            "failed": "ERROR:",
                            "unavailable": "WARNING:",
                            "error": "BOOM:"
                        }.get(rag_deploy["status"], "?")

                        st.markdown(f"  {status_icon} {rag_deploy['rag_type']}: {rag_deploy['status']}")

    def _render_flow_designer_page(self):
        """Render the Flow Designer page."""
        st.header("* Flow Designer")
        st.markdown("Advanced workflow configuration and RAG integration design")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("TOOL: Design Tools")

            design_mode = st.radio(
                "Design Mode",
                ["Visual Builder", "Code Editor", "Template Wizard"]
            )

            if design_mode == "Visual Builder":
                st.markdown("*** Visual Workflow Builder**")
                st.info("🚧 Visual builder coming soon!")
                st.markdown("""
                **Features:**
                - Drag & drop workflow components
                - Visual RAG system connections
                - Real-time flow validation
                - Interactive testing
                """)

            elif design_mode == "Code Editor":
                st.markdown("**CODE: Flow Code Editor**")

                # Sample workflow code
                sample_code = {
                    "workflow_id": "custom_workflow",
                    "workflow_name": "Custom Analysis Workflow",
                    "workflow_type": "custom",
                    "rag_integration": ["ai_powered", "dspy"],
                    "flow_encoding": "@custom#analyze!process@output",
                    "stages": [
                        {
                            "stage": "input",
                            "description": "Document input and validation",
                            "operations": ["validate", "classify", "route"]
                        },
                        {
                            "stage": "process",
                            "description": "RAG-enhanced processing",
                            "operations": ["rag_query", "analyze", "synthesize"]
                        },
                        {
                            "stage": "output",
                            "description": "Result generation",
                            "operations": ["format", "validate", "deliver"]
                        }
                    ]
                }

                workflow_code = st.text_area(
                    "Workflow JSON",
                    value=json.dumps(sample_code, indent=2),
                    height=400
                )

                if st.button("OK: Validate Code"):
                    try:
                        parsed_workflow = json.loads(workflow_code)
                        st.success("OK: Valid workflow JSON!")

                        # Show workflow preview
                        with st.expander("FILE: Workflow Preview"):
                            st.json(parsed_workflow)

                    except json.JSONDecodeError as e:
                        st.error(f"ERROR: Invalid JSON: {e}")

            else:  # Template Wizard
                st.markdown("**WIZARD: Template Wizard**")

                wizard_step = st.selectbox(
                    "Wizard Step",
                    ["1. Basic Info", "2. RAG Configuration", "3. Flow Definition", "4. Review & Create"]
                )

                if wizard_step == "1. Basic Info":
                    st.markdown("**Step 1: Basic Workflow Information**")
                    with st.form("wizard_step1"):
                        wf_name = st.text_input("Workflow Name")
                        wf_type = st.selectbox("Workflow Type", ["analysis", "processing", "synthesis", "classification"])
                        wf_desc = st.text_area("Description")

                        if st.form_submit_button("Next ➡️"):
                            st.session_state.wizard_step1 = {
                                "name": wf_name, "type": wf_type, "description": wf_desc
                            }
                            st.success("OK: Step 1 Complete! Move to Step 2.")

        with col2:
            st.subheader("^ RAG Integration Designer")

            # RAG system availability
            rag_status = self.workflow_manager.get_available_rag_systems()

            st.markdown("**AI: Available RAG Systems:**")
            for rag_type, is_available in rag_status.items():
                status_icon = "OK:" if is_available else "ERROR:"
                st.markdown(f"{status_icon} **{rag_type.replace('_', ' ').title()}**")

            st.markdown("---")

            # RAG Integration Patterns
            st.markdown("**🔗 Integration Patterns:**")

            pattern = st.selectbox(
                "Select Pattern",
                [
                    "Single RAG System",
                    "Parallel RAG Processing",
                    "Sequential RAG Chain",
                    "Conditional RAG Routing",
                    "Hybrid Multi-RAG"
                ]
            )

            if pattern == "Single RAG System":
                st.info("🎯 Use one RAG system for the entire workflow")
                selected_rag = st.selectbox("RAG System", [k for k, v in rag_status.items() if v])

            elif pattern == "Parallel RAG Processing":
                st.info("FAST: Run multiple RAG systems in parallel and combine results")
                parallel_rags = st.multiselect("RAG Systems", [k for k, v in rag_status.items() if v])

            elif pattern == "Sequential RAG Chain":
                st.info("🔗 Chain RAG systems where output of one feeds into next")
                st.markdown("**Define Chain Order:**")

            # Flow Visualization
            st.markdown("---")
            st.markdown("**🌊 Flow Visualization:**")

            # Simple flow diagram
            if pattern == "Single RAG System":
                st.markdown("""
                ```
                Input → RAG System → Processing → Output
                ```
                """)
            elif pattern == "Parallel RAG Processing":
                st.markdown("""
                ```
                Input → ┌─ RAG System 1 ─┐
                        ├─ RAG System 2 ─┼─ Combine → Output
                        └─ RAG System 3 ─┘
                ```
                """)

    def _render_workflow_monitor_page(self):
        """Render the Workflow Monitor page."""
        st.header("^ Workflow Monitor")
        st.markdown("Monitor active workflows and system performance")

        # Get active workflows
        active_workflows = self.workflow_manager.get_active_workflows()

        if not active_workflows:
            st.info("SLEEP: No active workflows to monitor")
            return

        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_workflows = len(active_workflows)
            st.metric("Total Workflows", total_workflows)

        with col2:
            deployed_workflows = len([w for w in active_workflows if w.get("status") == "deployed"])
            st.metric("Deployed", deployed_workflows)

        with col3:
            rag_integrations = sum(len(w.get("rag_systems", [])) for w in active_workflows)
            st.metric("RAG Integrations", rag_integrations)

        with col4:
            # Success rate (mock data)
            st.metric("Success Rate", "94%", "2%")

        # Workflow details
        st.markdown("---")
        st.subheader("+ Active Workflow Details")

        for workflow in active_workflows:
            status_color = {
                "created": "BLUE:",
                "deployed": "GREEN:",
                "running": "YELLOW:",
                "completed": "OK:",
                "error": "RED:"
            }.get(workflow.get("status"), "WHITE:")

            with st.expander(f"{status_color} {workflow.get('config', {}).get('name', workflow['workflow_id'])}"):

                # Basic info
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**Status:** {workflow.get('status', 'unknown')}")
                    st.markdown(f"**Created:** {workflow.get('created_at', 'unknown')}")
                    st.markdown(f"**Template:** {workflow.get('template', {}).get('workflow_name', 'unknown')}")

                with col2:
                    st.markdown(f"**Domain:** {workflow.get('config', {}).get('domain', 'general')}")
                    if workflow.get("deployed_at"):
                        st.markdown(f"**Deployed:** {workflow['deployed_at']}")

                # RAG system status
                if workflow.get("rag_deployment"):
                    st.markdown("**AI: RAG System Status:**")

                    rag_cols = st.columns(len(workflow["rag_deployment"]))

                    for i, rag_deploy in enumerate(workflow["rag_deployment"]):
                        with rag_cols[i]:
                            status_icon = {
                                "deployed": "OK:",
                                "failed": "ERROR:",
                                "unavailable": "WARNING:",
                                "error": "BOOM:"
                            }.get(rag_deploy["status"], "?")

                            st.markdown(f"**{rag_deploy['rag_type']}**")
                            st.markdown(f"{status_icon} {rag_deploy['status']}")

                            if rag_deploy.get("system_id"):
                                st.markdown(f"ID: `{rag_deploy['system_id'][:8]}...`")

    def _render_test_runner_page(self):
        """Render the Test Runner page for document upload and flow execution."""
        st.header("> Test Runner")
        st.markdown("Upload documents and run active workflows for testing and validation")

        # Document Upload Section
        st.subheader("Document Upload")

        uploaded_files = st.file_uploader(
            "Choose documents for workflow testing",
            type=['pdf', 'docx', 'txt', 'md', 'csv', 'json', 'xlsx'],
            accept_multiple_files=True,
            help="Supports PDF, Word, text, markdown, CSV, JSON, and Excel files. Maximum 20 files.",
            key="test_runner_file_upload"
        )

        if uploaded_files:
            st.success(f"+ {len(uploaded_files)} file(s) uploaded")

            # Show uploaded files
            st.write("**Uploaded Files:**")
            for i, file in enumerate(uploaded_files, 1):
                file_size = len(file.getvalue()) / 1024  # KB
                st.write(f"{i}. {file.name} ({file_size:.1f} KB)")

        # Active Workflows Selection
        st.subheader("Select Active Workflow")

        # Get available workflows
        workflows = self.registry.workflows
        active_workflows = [w for w in workflows.values() if w.get('status') != 'disabled']

        if active_workflows:
            workflow_options = {}
            for workflow in active_workflows:
                workflow_id = workflow.get('workflow_id', 'unknown')
                workflow_name = workflow.get('workflow_name', 'Unnamed Workflow')
                workflow_options[f"{workflow_name} ({workflow_id})"] = workflow

            selected_workflow_key = st.selectbox(
                "Choose workflow to execute",
                options=list(workflow_options.keys()),
                help="Select an active workflow to run with your uploaded documents"
            )

            if selected_workflow_key:
                selected_workflow = workflow_options[selected_workflow_key]

                # Show workflow details
                with st.expander("Workflow Details", expanded=False):
                    st.json(selected_workflow)

                # Template Fields Configuration
                st.subheader("Configure Template Fields")

                template_fields = selected_workflow.get('template_fields', {})
                field_values = {}

                if template_fields:
                    col1, col2 = st.columns(2)

                    field_items = list(template_fields.items())
                    mid_point = len(field_items) // 2

                    with col1:
                        for field_name, field_spec in field_items[:mid_point]:
                            field_values[field_name] = self._render_template_field_input(
                                field_name, field_spec, uploaded_files
                            )

                    with col2:
                        for field_name, field_spec in field_items[mid_point:]:
                            field_values[field_name] = self._render_template_field_input(
                                field_name, field_spec, uploaded_files
                            )
                else:
                    st.info("No template fields defined for this workflow")

                # Execution Controls
                st.subheader("Execute Workflow")

                col1, col2, col3 = st.columns([2, 1, 1])

                with col1:
                    execution_mode = st.radio(
                        "Execution Mode",
                        ["Simulation", "Full Execution"],
                        help="Simulation: Mock execution for testing. Full Execution: Real processing."
                    )

                with col2:
                    if st.button("+ Run Workflow", type="primary", disabled=not uploaded_files):
                        if uploaded_files and selected_workflow:
                            self._execute_test_workflow(
                                selected_workflow,
                                uploaded_files,
                                field_values,
                                execution_mode == "Full Execution"
                            )

                with col3:
                    if st.button("Clear All", help="Clear uploaded files and reset form"):
                        st.rerun()

        else:
            st.warning("No active workflows available. Please create a workflow first.")

        # Recent Test Results
        st.subheader("Recent Test Results")
        self._render_recent_test_results()

    def _render_template_field_input(self, field_name: str, field_spec: Dict, uploaded_files: List):
        """Render input widget for a template field."""
        field_type = field_spec.get('type', 'string')
        description = field_spec.get('description', f'Value for {field_name}')
        default_value = field_spec.get('default')
        required = field_spec.get('required', True)

        label = f"{field_name}{'*' if required else ''}"

        # Special handling for input_files
        if field_name == 'input_files' and uploaded_files:
            return [f.name for f in uploaded_files]

        # Handle different field types
        if field_type == 'string':
            if field_spec.get('enum'):
                return st.selectbox(label, field_spec['enum'], help=description)
            else:
                return st.text_input(label, value=default_value or "", help=description)

        elif field_type in ['integer', 'number']:
            min_val = field_spec.get('range', [0, 100])[0] if 'range' in field_spec else 0
            max_val = field_spec.get('range', [0, 100])[1] if 'range' in field_spec else 100
            return st.number_input(label, min_value=min_val, max_value=max_val,
                                 value=default_value or min_val, help=description)

        elif field_type == 'boolean':
            return st.checkbox(label, value=default_value or False, help=description)

        elif field_type == 'array':
            input_val = st.text_area(label, help=f"{description} (one item per line)")
            return input_val.strip().split('\n') if input_val.strip() else []

        else:
            return st.text_input(label, value=str(default_value or ""), help=description)

    def _execute_test_workflow(self, workflow: Dict, uploaded_files: List, field_values: Dict, full_execution: bool):
        """Execute the selected workflow with uploaded documents."""
        try:
            st.info(f"{'Executing' if full_execution else 'Simulating'} workflow: {workflow.get('workflow_name')}")

            # Generate unique execution ID
            from datetime import datetime
            execution_id = f"TEST_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Save uploaded files to temporary location
            import tempfile
            import os

            temp_dir = Path(tempfile.mkdtemp(prefix="flow_test_"))
            input_files = []

            for uploaded_file in uploaded_files:
                file_path = temp_dir / uploaded_file.name
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())
                input_files.append(str(file_path))

            # Update field values with actual file paths
            if 'input_files' in field_values:
                field_values['input_files'] = input_files

            if full_execution:
                # Import and run the sequential workflow test
                import sys
                sys.path.append(str(Path("tidyllm/workflows/definitions/workflows/templates")))

                try:
                    from test_sequential_flow import run_sequential_flow_test

                    # Modify input files for the test
                    original_get_input_files = None

                    # Run the workflow
                    with st.spinner("Executing workflow steps..."):
                        result = run_sequential_flow_test()

                    # Display results
                    st.success("+ Workflow execution completed successfully!")

                    with st.expander("Execution Results", expanded=True):
                        st.json(result)

                    # Show processing summary
                    summary = result.get('summary', {})
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Processing Time", f"{summary.get('total_processing_time_ms', 0):.1f}ms")
                    with col2:
                        st.metric("Steps Completed", f"{summary.get('steps_completed', 0)}/5")
                    with col3:
                        st.metric("Success Rate", f"{summary.get('success_rate', 0):.0%}")
                    with col4:
                        st.metric("Files Processed", summary.get('input_files_processed', 0))

                except Exception as e:
                    st.error(f"ERROR: Workflow execution failed: {str(e)}")
                    st.code(traceback.format_exc())

            else:
                # Simulation mode
                with st.spinner("Simulating workflow execution..."):
                    import time
                    time.sleep(2)  # Simulate processing time

                st.success("+ Workflow simulation completed!")

                # Mock results
                mock_result = {
                    "execution_id": execution_id,
                    "workflow_id": workflow.get('workflow_id'),
                    "simulation": True,
                    "input_files": len(uploaded_files),
                    "template_fields": field_values,
                    "estimated_processing_time": "~2.5 seconds",
                    "steps": [
                        {"step": 1, "name": "Input Validation", "status": "simulated"},
                        {"step": 2, "name": "Data Extraction", "status": "simulated"},
                        {"step": 3, "name": "Analysis", "status": "simulated"},
                        {"step": 4, "name": "Synthesis", "status": "simulated"},
                        {"step": 5, "name": "Output Generation", "status": "simulated"}
                    ]
                }

                with st.expander("Simulation Results", expanded=True):
                    st.json(mock_result)

            # Cleanup temp files
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

        except Exception as e:
            st.error(f"ERROR: Test execution failed: {str(e)}")
            st.code(traceback.format_exc())

    def _render_recent_test_results(self):
        """Show recent test execution results."""
        # For now, show placeholder - could be enhanced to read from outputs directory
        results_dir = Path("tidyllm/workflows/definitions/workflows/templates/outputs")

        if results_dir.exists():
            result_files = list(results_dir.glob("final_REV*.json"))
            if result_files:
                # Sort by modification time, most recent first
                result_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

                st.write(f"**{len(result_files)} recent test result(s):**")

                for i, result_file in enumerate(result_files[:5]):  # Show last 5
                    with st.expander(f"{result_file.name} ({datetime.fromtimestamp(result_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')})", expanded=False):
                        try:
                            with open(result_file, 'r', encoding='utf-8') as f:
                                result_data = json.load(f)

                            # Show summary
                            summary = result_data.get('summary', {})
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric("Processing Time", f"{summary.get('total_processing_time_ms', 0):.1f}ms")
                            with col2:
                                st.metric("Steps", f"{summary.get('steps_completed', 0)}")
                            with col3:
                                st.metric("Files", f"{summary.get('input_files_processed', 0)}")

                            # Show full results
                            if st.button(f"View Full Results", key=f"view_result_{i}"):
                                st.json(result_data)

                        except Exception as e:
                            st.error(f"Could not load result file: {e}")
            else:
                st.info("No test results found. Run a workflow test to see results here.")
        else:
            st.info("No results directory found.")

    def _render_ai_advisor_page(self):
        """Render the AI Advisor page with chat interface."""
        st.header("? AI Workflow Advisor")
        st.markdown("Get intelligent advice about your workflows from AI - ask about optimization, troubleshooting, or best practices")

        # Import the workflow advisor
        try:
            import sys
            sys.path.append(str(Path("tidyllm/workflows/ai_advisor")))
            from workflow_advisor import workflow_advisor
            advisor_available = True
        except ImportError as e:
            st.error(f"AI Advisor not available: {e}")
            advisor_available = False
            return

        # Context gathering section
        st.subheader("Workflow Context")

        col1, col2 = st.columns(2)

        with col1:
            # Get current workflow context
            workflows = self.registry.workflows
            selected_workflow = None

            if workflows:
                workflow_names = list(workflows.keys())
                selected_workflow_name = st.selectbox(
                    "Select workflow for context",
                    options=["All Workflows"] + workflow_names,
                    help="Choose a specific workflow or analyze all workflows"
                )

                if selected_workflow_name != "All Workflows":
                    selected_workflow = workflows[selected_workflow_name]

                    # Show context checkboxes
                    st.write("**Include in analysis:**")
                    include_criteria = st.checkbox("Criteria & Document Qualifiers", value=True)
                    include_fields = st.checkbox("Template Fields", value=True)
                    include_activity = st.checkbox("Recent Activity", value=True)
                    include_results = st.checkbox("Latest Results", value=True)
                else:
                    include_criteria = include_fields = include_activity = include_results = True
            else:
                st.warning("No workflows found. Create a workflow first to get context-aware advice.")
                include_criteria = include_fields = include_activity = include_results = False

        with col2:
            # Quick context insights
            if selected_workflow:
                st.write("**Quick Context Preview:**")

                # Show workflow summary
                workflow_type = selected_workflow.get('workflow_type', 'unknown')
                steps = len(selected_workflow.get('steps', []))
                rag_systems = len(selected_workflow.get('rag_integration', []))

                st.write(f"- Type: {workflow_type}")
                st.write(f"- Steps: {steps}")
                st.write(f"- RAG Systems: {rag_systems}")

                # Template fields count
                template_fields = selected_workflow.get('template_fields', {})
                st.write(f"- Template Fields: {len(template_fields)}")

        # Chat interface
        st.subheader("AI Chat Interface")

        # Initialize chat history
        if "ai_advisor_messages" not in st.session_state:
            st.session_state.ai_advisor_messages = [
                {
                    "role": "assistant",
                    "content": "Hello! I'm your AI Workflow Advisor. I can help you optimize workflows, troubleshoot issues, and suggest best practices. What would you like to know about your workflows?"
                }
            ]

        # Chat history display
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.ai_advisor_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Quick suggestion buttons
        st.write("**Quick Questions:**")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("Optimize Performance", help="Get performance optimization tips"):
                quick_question = "How can I optimize the performance of my workflow? What are the main bottlenecks to look for?"
                st.session_state.ai_advisor_pending_question = quick_question

        with col2:
            if st.button("Improve Template Fields", help="Get field configuration advice"):
                quick_question = "How can I improve my template field configuration? What validation rules should I add?"
                st.session_state.ai_advisor_pending_question = quick_question

        with col3:
            if st.button("Troubleshoot Issues", help="Get troubleshooting help"):
                quick_question = "My workflow is having issues. Can you help me troubleshoot common problems?"
                st.session_state.ai_advisor_pending_question = quick_question

        with col4:
            if st.button("Best Practices", help="Get workflow best practices"):
                quick_question = "What are the best practices for designing efficient document processing workflows?"
                st.session_state.ai_advisor_pending_question = quick_question

        # Chat input
        if prompt := st.chat_input("Ask about your workflow..."):
            self._handle_ai_advisor_chat(prompt, selected_workflow, include_criteria, include_fields, include_activity, include_results, advisor_available)

        # Handle pending questions from quick buttons
        if hasattr(st.session_state, 'ai_advisor_pending_question'):
            self._handle_ai_advisor_chat(
                st.session_state.ai_advisor_pending_question,
                selected_workflow,
                include_criteria, include_fields, include_activity, include_results,
                advisor_available
            )
            del st.session_state.ai_advisor_pending_question

        # Context summary sidebar
        with st.sidebar:
            if selected_workflow:
                st.subheader("Workflow Context")

                # Show what context will be included
                context_items = []
                if include_criteria:
                    context_items.append("+ Criteria & qualifiers")
                if include_fields:
                    context_items.append("+ Template fields")
                if include_activity:
                    context_items.append("+ Recent activity")
                if include_results:
                    context_items.append("+ Latest results")

                if context_items:
                    st.write("**Included in AI analysis:**")
                    for item in context_items:
                        st.write(item)
                else:
                    st.write("No context selected")

                # Quick suggestions based on context
                suggestions = self._get_quick_workflow_suggestions(selected_workflow)
                if suggestions:
                    st.write("**Quick Suggestions:**")
                    for suggestion in suggestions:
                        st.write(f"• {suggestion}")

    def _handle_ai_advisor_chat(self, user_input: str, selected_workflow: Dict,
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

            if selected_workflow:
                if include_criteria:
                    criteria = self._load_workflow_criteria(selected_workflow)

                if include_fields:
                    template_fields = selected_workflow.get('template_fields', {})

                if include_activity:
                    recent_activity = self._get_recent_workflow_activity()

                if include_results:
                    final_results = self._get_latest_workflow_results()

            # Get AI response
            try:
                from workflow_advisor import workflow_advisor

                with st.spinner("AI is analyzing your workflow..."):
                    advice_result = workflow_advisor.get_workflow_advice(
                        criteria=criteria,
                        template_fields=template_fields,
                        recent_activity=recent_activity,
                        final_results=final_results,
                        user_question=user_input
                    )

                if advice_result.get('success', False):
                    response = advice_result['advice']

                    # Add context info
                    context_info = advice_result.get('context_analyzed', {})
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
                response = f"I apologize, but I encountered an error while processing your request: {str(e)}. Please try asking a more specific question."

        # Add AI response to chat
        st.session_state.ai_advisor_messages.append({
            "role": "assistant",
            "content": response
        })

        # Trigger rerun to show new messages
        st.rerun()

    def _load_workflow_criteria(self, workflow: Dict) -> Dict:
        """Load criteria for the specified workflow."""
        try:
            # Try to load criteria.json from the workflow's criteria directory
            workflow_name = workflow.get('workflow_name', '').lower().replace(' ', '_')
            criteria_file = Path(f"tidyllm/workflows/definitions/workflows/{workflow_name}/criteria/criteria.json")

            if criteria_file.exists():
                with open(criteria_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Try the templates directory as fallback
                criteria_file = Path("tidyllm/workflows/definitions/workflows/templates/criteria/criteria.json")
                if criteria_file.exists():
                    with open(criteria_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
        except Exception:
            pass

        return {}

    def _get_recent_workflow_activity(self) -> List[Dict]:
        """Get recent workflow activity and executions."""
        activity = []

        try:
            # Load recent test results
            results_dir = Path("tidyllm/workflows/definitions/workflows/templates/outputs")
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

    def _get_latest_workflow_results(self) -> Dict:
        """Get the latest workflow execution results."""
        try:
            results_dir = Path("tidyllm/workflows/definitions/workflows/templates/outputs")
            if results_dir.exists():
                result_files = list(results_dir.glob("final_REV*.json"))
                if result_files:
                    # Get most recent file
                    latest_file = max(result_files, key=lambda x: x.stat().st_mtime)

                    with open(latest_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
        except Exception:
            pass

        return {}

    def _get_quick_workflow_suggestions(self, workflow: Dict) -> List[str]:
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

    def _render_health_dashboard_page(self):
        """Render the Health Dashboard page."""
        st.header("# Health Dashboard")
        st.markdown("Comprehensive health monitoring for workflow and RAG systems")

        # Overall health metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("+ Workflow Systems")
            workflow_health = self.workflow_manager.get_workflow_health()

            if workflow_health.get("success"):
                overall_healthy = workflow_health.get("overall_healthy", False)
                if overall_healthy:
                    st.success("OK: All Systems Healthy")
                else:
                    st.warning("WARNING: Some Issues Detected")

                # Individual system health
                systems = workflow_health.get("systems", {})
                for system_name, health_info in systems.items():
                    is_healthy = health_info.get("healthy", False)
                    icon = "OK:" if is_healthy else "ERROR:"

                    with st.expander(f"{icon} {system_name.replace('_', ' ').title()}"):
                        st.markdown(f"**Available:** {'Yes' if health_info.get('available') else 'No'}")
                        st.markdown(f"**Has Template:** {'Yes' if health_info.get('has_template') else 'No'}")
                        st.markdown(f"**Active Workflows:** {health_info.get('active_workflows', 0)}")
                        if health_info.get("last_check"):
                            st.markdown(f"**Last Check:** {health_info['last_check']}")
            else:
                st.error(f"ERROR: Health check failed: {workflow_health.get('error', 'Unknown error')}")

        with col2:
            st.subheader("AI: RAG Systems")
            rag_status = self.workflow_manager.get_available_rag_systems()

            available_count = sum(1 for available in rag_status.values() if available)
            total_count = len(rag_status)

            if available_count == total_count:
                st.success(f"OK: All {total_count} RAG Systems Available")
            elif available_count > 0:
                st.warning(f"WARNING: {available_count}/{total_count} RAG Systems Available")
            else:
                st.error("ERROR: No RAG Systems Available")

            for rag_type, is_available in rag_status.items():
                icon = "OK:" if is_available else "ERROR:"
                st.markdown(f"{icon} **{rag_type.replace('_', ' ').title()}**")

        with col3:
            st.subheader("^ Performance Metrics")
            metrics = self.workflow_manager.get_workflow_metrics()

            if metrics.get("success"):
                st.metric("Total Workflows", metrics.get("total_workflows", 0))

                # Status distribution
                status_dist = metrics.get("status_distribution", {})
                for status, count in status_dist.items():
                    if count > 0:
                        status_icon = {
                            "created": "BLUE:",
                            "deployed": "GREEN:",
                            "running": "YELLOW:",
                            "completed": "OK:",
                            "failed": "RED:",
                            "archived": "DIR:"
                        }.get(status, "WHITE:")

                        st.markdown(f"{status_icon} **{status.title()}:** {count}")
            else:
                st.error("ERROR: Metrics unavailable")

        # Detailed health information
        st.markdown("---")
        st.subheader("SEARCH: Detailed Health Information")

        # Refresh button
        if st.button("+ Refresh Health Status", use_container_width=True):
            # Clear any cached health data
            if hasattr(self.workflow_manager.flow_manager, 'health_cache'):
                self.workflow_manager.flow_manager.health_cache.clear()
            st.success("OK: Health status refreshed!")
            st.rerun()

        # System status table
        workflow_health = self.workflow_manager.get_workflow_health()
        if workflow_health.get("success"):
            systems = workflow_health.get("systems", {})

            # Create a table view
            system_data = []
            for system_name, health_info in systems.items():
                system_data.append({
                    "System": system_name.replace('_', ' ').title(),
                    "Status": "OK: Healthy" if health_info.get("healthy") else "ERROR: Unhealthy",
                    "Available": "Yes" if health_info.get("available") else "No",
                    "Template": "Yes" if health_info.get("has_template") else "No",
                    "Active Workflows": health_info.get("active_workflows", 0),
                    "Last Check": health_info.get("last_check", "Never")[:19] if health_info.get("last_check") else "Never"
                })

            if system_data:
                import pandas as pd
                df = pd.DataFrame(system_data)
                st.dataframe(df, use_container_width=True)

        # Health history (future enhancement)
        st.markdown("---")
        st.subheader("📈 Health Trends")
        st.info("🚧 Health trend visualization coming soon!")
        st.markdown("""
        **Planned Features:**
        - Health status over time
        - Performance trend charts
        - Alert history
        - System reliability metrics
        """)


def main():
    """Main entry point for Flow Creator V3 Portal."""
    try:
        portal = FlowCreatorV3Portal()
        portal.render_portal()
    except Exception as e:
        st.error(f"ERROR: Portal Error: {e}")
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()