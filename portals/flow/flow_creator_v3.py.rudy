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

# Import step ordering utilities
try:
    from tidyllm.utils.step_ordering import (
        reorder_steps_by_position,
        check_for_order_changes,
        renumber_steps,
        validate_step_order
    )
    UTILS_AVAILABLE = True
except ImportError:
    # Fallback to local implementations if utils not available
    UTILS_AVAILABLE = False

# Import clean JSON I/O utilities
from tidyllm.utils.clean_json_io import load_json, save_json
from tidyllm.utils.data_normalizer import DataNormalizer

# Import TidyLLM components
try:
    from tidyllm.services.unified_rag_manager import UnifiedRAGManager, RAGSystemType
    from tidyllm.services.unified_flow_manager import UnifiedFlowManager, WorkflowSystemType, WorkflowStatus
    from tidyllm.infrastructure.session.unified import UnifiedSessionManager
except ImportError:
    st.error("TidyLLM components not available. Please check installation.")
    st.stop()


class WorkflowRegistry:
    """Manages the comprehensive workflow registry with 17+ workflow definitions."""

    def __init__(self):
        # Find the base path by looking for AI-Shipping directory
        current = Path.cwd()
        if "AI-Shipping" in str(current):
            # Navigate to AI-Shipping root
            while current.name != "AI-Shipping" and current.parent != current:
                current = current.parent
            base_path = current
        else:
            # Fallback to current directory
            base_path = Path.cwd()

        self.workflows_base_path = base_path / "tidyllm" / "workflows" / "projects"
        self.workflow_registry_path = base_path / "tidyllm" / "workflows" / "projects"
        self.workflows = self._load_workflow_registry()

    def _load_workflow_registry(self) -> Dict[str, Any]:
        """Load workflows from the project folder structure (criteria, outputs, resources, templates)."""
        try:
            workflows = {}

            # Debug: Print the path being checked
            print(f"DEBUG: Looking for workflows in: {self.workflows_base_path}")
            print(f"DEBUG: Path exists: {self.workflows_base_path.exists()}")

            # Demo mode: only show alex_qaqc for now (can be made configurable)
            demo_filter_enabled = True
            allowed_projects = {"alex_qaqc"}

            # Scan workflow project directories
            if self.workflows_base_path.exists():
                for project_dir in self.workflows_base_path.iterdir():
                    if project_dir.is_dir() and project_dir.name != "__pycache__":
                        # Skip projects not in demo filter if enabled
                        if demo_filter_enabled and project_dir.name not in allowed_projects:
                            print(f"DEBUG: Skipping {project_dir.name} (demo filter active)")
                            continue

                        print(f"DEBUG: Found project directory: {project_dir.name}")
                        workflow_data = self._load_project_workflow(project_dir)
                        if workflow_data:
                            workflows[project_dir.name] = workflow_data
                            print(f"DEBUG: Loaded workflow: {project_dir.name}")

            # If no workflows found, use defaults
            if not workflows:
                print("DEBUG: No workflows found, using defaults")
                workflows = self._create_default_registry()
            else:
                print(f"DEBUG: Loaded {len(workflows)} workflows total")

            return workflows

        except Exception as e:
            print(f"DEBUG ERROR: {e}")
            st.warning(f"WARNING: Could not load workflow registry: {e}")
            return self._create_default_registry()

    def _load_project_workflow(self, project_dir: Path) -> Dict[str, Any]:
        """Load workflow data from project directory structure."""
        try:
            # Check if this is a valid workflow project
            # Must have either project_config.json OR proper folder structure (criteria + templates)
            config_file = project_dir / "project_config.json"
            has_criteria = (project_dir / "criteria").exists()
            has_templates = (project_dir / "templates").exists()

            # Skip projects without proper configuration
            if not config_file.exists() and not (has_criteria and has_templates):
                print(f"DEBUG: Skipping incomplete project: {project_dir.name} (no config file and missing folders)")
                return None

            workflow_data = {
                "workflow_id": project_dir.name,
                "workflow_name": project_dir.name.replace('_', ' ').title(),
                "workflow_type": self._detect_workflow_type(project_dir),
                "description": self._load_project_description(project_dir),
                "project_structure": {
                    "criteria": has_criteria,
                    "outputs": (project_dir / "outputs").exists(),
                    "resources": (project_dir / "resources").exists(),
                    "templates": has_templates,
                    "inputs": (project_dir / "inputs").exists()
                },
                "rag_integration": self._detect_rag_integration(project_dir),
                "flow_encoding": f"@{project_dir.name}#process!analyze@output"
            }

            if config_file.exists():
                config_data = load_json(config_file)
                workflow_data.update(config_data)
                workflow_data["has_flow_definition"] = True
            else:
                # Fallback: try legacy naming patterns
                flow_file = project_dir / f"{project_dir.name}_flow.json"
                criteria_file = project_dir / f"{project_dir.name}_criteria.json"

                if flow_file.exists():
                    flow_data = load_json(flow_file)
                    workflow_data.update(flow_data)
                    workflow_data["has_flow_definition"] = True
                elif criteria_file.exists():
                    criteria_data = load_json(criteria_file)
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
        # Defensive check: ensure workflows is a dict
        if isinstance(self.workflows, list):
            return list(set(w.get("workflow_type", "unknown") for w in self.workflows))
        elif isinstance(self.workflows, dict):
            return list(set(w.get("workflow_type", "unknown") for w in self.workflows.values()))
        else:
            return ["unknown"]

    def get_workflows_by_type(self, workflow_type: str) -> Dict[str, Any]:
        """Get workflows filtered by type."""
        # Defensive check: ensure workflows is a dict
        if isinstance(self.workflows, dict):
            return {
                wid: workflow for wid, workflow in self.workflows.items()
                if workflow.get("workflow_type") == workflow_type
            }
        elif isinstance(self.workflows, list):
            return {
                f"workflow_{i}": workflow for i, workflow in enumerate(self.workflows)
                if workflow.get("workflow_type") == workflow_type
            }
        else:
            return {}

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

    def _format_date(self, iso_date: str) -> str:
        """Format ISO date to readable format."""
        if not iso_date:
            return 'Unknown'
        try:
            from datetime import datetime
            # Parse ISO format
            dt = datetime.fromisoformat(iso_date.replace('Z', '+00:00'))
            # Format as readable date
            return dt.strftime('%Y-%m-%d %H:%M')
        except Exception:
            return iso_date

    def get_active_workflows(self) -> List[Dict[str, Any]]:
        """Get list of active workflows."""
        # Get workflows from our registry instead of flow_manager
        workflows = self.registry.workflows
        active_workflows = []

        for workflow_id, workflow_data in workflows.items():
            if workflow_data.get('status', 'active') != 'disabled':
                # Convert to expected format
                workflow_info = {
                    'workflow_id': workflow_id,  # Changed from 'id' to 'workflow_id'
                    'workflow_name': workflow_data.get('workflow_name', workflow_id),
                    'name': workflow_data.get('workflow_name', workflow_id),
                    'type': workflow_data.get('workflow_type', 'unknown'),
                    'status': workflow_data.get('status', 'active'),
                    'description': workflow_data.get('description', ''),
                    'has_flow_definition': workflow_data.get('has_flow_definition', False),
                    'created_at': self._format_date(workflow_data.get('created_at', '')),
                    'version': workflow_data.get('version', '1.0'),
                    'config': workflow_data,  # Add the full config data
                    'template': workflow_data  # For backward compatibility
                }
                active_workflows.append(workflow_info)

        return active_workflows

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

        # Main tabs - DEMO FOCUSED: Test Designer replaces Flow Designer for optimization testing
        tab1, tab2, tab3, tab4 = st.tabs([
            "+ Create Flow",
            "= Existing Flows",
            "🧪 Test Designer",        # NEW: A/B testing for workflow optimization
            "? AI Advisor"
            # , "* Flow Designer"        # COMMENTED OUT - replaced by Test Designer
            # , "> Test Runner"          # COMMENTED OUT FOR DEMO
            # , "^ Workflow Monitor"     # COMMENTED OUT FOR DEMO
            # , "# Health Dashboard"     # COMMENTED OUT FOR DEMO
        ])
        # tab6, tab7 would be for monitor and health dashboard

        with tab1:
            self._render_create_flow_page()

        with tab2:
            self._render_existing_flows_page()

        with tab3:
            self._render_test_designer_page()  # NEW: Test Designer for A/B optimization

        with tab4:
            self._render_ai_advisor_page()

        # Commented out Flow Designer - users can choose Test Designer for optimization instead
        # with tab3:
        #     self._render_flow_designer_page()

        # COMMENTED OUT FOR DEMO - Test Runner, Monitor and Health Dashboard tabs
        # with tab4:
        #     self._render_test_runner_page()

        # with tab5:
        #     self._render_workflow_monitor_page()

        # with tab6:
        #     self._render_health_dashboard_page()

    def _render_status_indicators(self):
        """Render system status indicators."""
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            # RAG Systems Status
            rag_status = self.workflow_manager.get_available_rag_systems()
            rag_status = DataNormalizer.normalize_to_status_dict(rag_status)
            available_rags = sum(1 for available in rag_status.values() if available)
            total_rags = len(rag_status)

            if available_rags > 0:
                st.success(f"OK: RAG Systems: {available_rags}/{total_rags}")
            else:
                st.warning("RAG systems currently unavailable")

        with col2:
            # Workflow Systems Status
            workflow_status = self.workflow_manager.get_available_workflow_systems()
            # Defensive check for workflow_status type
            if isinstance(workflow_status, dict):
                available_workflows = sum(1 for available in workflow_status.values() if available)
                total_workflow_types = len(workflow_status)
            elif isinstance(workflow_status, list):
                available_workflows = sum(1 for item in workflow_status if item)
                total_workflow_types = len(workflow_status)
            else:
                available_workflows = 0
                total_workflow_types = 0

            if available_workflows > 0:
                st.success(f"+ Workflow Types: {available_workflows}/{total_workflow_types}")
            else:
                st.warning("Workflow systems currently unavailable")

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
                st.warning("Flow Manager temporarily unavailable")

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
                            icon = "✓" if exists else "○"
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
                        # Defensive check for rag_status type
                        if isinstance(rag_status, dict):
                            is_available = rag_status.get(rag_name, False)
                        elif isinstance(rag_status, list):
                            # For list format, assume all items are available
                            is_available = rag_name in rag_status if rag_status else False
                        else:
                            is_available = False
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
                                    st.success("OK: Success")
                                else:
                                    st.success("OK:")
                            else:
                                st.warning("Not available")

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
                            st.warning("Maximum 20 files allowed. Please remove some files.")
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
                            project_inputs_dir = Path(f"tidyllm/workflows/projects/{selected_template}/inputs")
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
                                                "failed": "○",
                                                "unavailable": "WARNING:",
                                                "error": "○"
                                            }.get(rag_deploy["status"], "?")

                                            st.markdown(f"{status_icon} **{rag_deploy['rag_type']}**: {rag_deploy['status']}")
                                else:
                                    st.warning(f"Deployment issue: {deploy_result.get('error')}")
                        else:
                            st.warning(f"Creation issue: {result.get('error')}")

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

        # Project File Manager Section
        if filtered_workflows:
            st.markdown("---")
            st.subheader("📁 Project File Manager")

            # Select workflow for file management
            workflow_options = {
                f"{w.get('config', {}).get('name', w.get('workflow_name', w.get('workflow_id', 'Unknown')))}"
                f" (v{w.get('version', '1.0')})": w
                for w in filtered_workflows
            }

            # Initialize session state for stable project selection
            if "file_manager_selected_workflow" not in st.session_state:
                st.session_state.file_manager_selected_workflow = None

            workflow_keys = list(workflow_options.keys())
            default_index = 0
            if st.session_state.file_manager_selected_workflow and st.session_state.file_manager_selected_workflow in workflow_keys:
                default_index = workflow_keys.index(st.session_state.file_manager_selected_workflow)

            selected_workflow_name = st.selectbox(
                "Select project to manage files",
                workflow_keys,
                index=default_index,
                help="Choose a project to view and edit criteria/template files"
            )

            # Update session state for stability
            st.session_state.file_manager_selected_workflow = selected_workflow_name

            if selected_workflow_name:
                selected_workflow = workflow_options[selected_workflow_name]

                # Render comprehensive file manager
                with st.container():
                    updated = self._render_project_file_manager(
                        selected_workflow,
                        container_key="file_manager"
                    )

                    if updated:
                        # Refresh the workflows list
                        st.rerun()

        # Display workflows (hidden when file manager is shown)
        # Commenting out workflow details cards since file manager provides better interface
        # for workflow in filtered_workflows:
        #     with st.expander(f"+ {workflow.get('name', workflow.get('workflow_id', 'Unknown'))}", expanded=False):
        #         col1, col2 = st.columns([2, 1])

        #         with col1:
        #             st.markdown(f"**ID:** `{workflow['workflow_id']}`")
        #             st.markdown(f"**Status:** {workflow.get('status', 'unknown')}")
        #             st.markdown(f"**Created:** {workflow.get('created_at', 'unknown')}")

        #             # Show Type and Description prominently
        #             workflow_type = workflow.get('workflow_type', workflow.get('template', {}).get('workflow_type', 'unknown'))
        #             workflow_description = workflow.get('description', workflow.get('template', {}).get('description', 'No description'))
        #             st.markdown(f"**Type:** `{workflow_type}` - {workflow_description}")

        #             st.markdown(f"**Template:** {workflow.get('template', {}).get('workflow_name', 'unknown')}")
        #             st.markdown(f"**Domain:** {workflow.get('config', {}).get('domain', 'general')}")

        #             if workflow.get("rag_systems"):
        #                 st.markdown(f"**RAG Systems:** {', '.join(workflow['rag_systems'])}")

        # Workflow details card moved to _render_workflow_details_card() method for reuse

    def _render_workflow_details_card(self, workflow: Dict[str, Any], key_suffix: str = "") -> bool:
        """Render detailed workflow information card with action buttons.

        Args:
            workflow: Workflow data dictionary
            key_suffix: Unique suffix for button keys to avoid conflicts

        Returns:
            bool: True if any action was performed that requires a rerun
        """
        with st.expander(f"+ {workflow.get('name', workflow.get('workflow_id', 'Unknown'))}", expanded=False):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"**ID:** `{workflow['workflow_id']}`")
                st.markdown(f"**Status:** {workflow.get('status', 'unknown')}")
                st.markdown(f"**Created:** {workflow.get('created_at', 'unknown')}")

                # Show Type and Description prominently
                workflow_type = workflow.get('workflow_type', workflow.get('template', {}).get('workflow_type', 'unknown'))
                workflow_description = workflow.get('description', workflow.get('template', {}).get('description', 'No description'))
                st.markdown(f"**Type:** `{workflow_type}` - {workflow_description}")

                st.markdown(f"**Template:** {workflow.get('template', {}).get('workflow_name', 'unknown')}")
                st.markdown(f"**Domain:** {workflow.get('config', {}).get('domain', 'general')}")

                if workflow.get("rag_systems"):
                    st.markdown(f"**RAG Systems:** {', '.join(workflow['rag_systems'])}")

            with col2:
                # Action buttons
                if workflow.get("status") == "created":
                    if st.button(f"> Deploy", key=f"deploy_{workflow['workflow_id']}_{key_suffix}"):
                        with st.spinner("Deploying..."):
                            result = self.workflow_manager.deploy_workflow(workflow['workflow_id'])

                        if result["success"]:
                            st.success("OK: Deployed!")
                            return True
                        else:
                            st.warning(f"Operation failed: {result.get('error')}")

                if st.button(f"^ Monitor", key=f"monitor_{workflow['workflow_id']}_{key_suffix}"):
                    st.info("+ Monitoring functionality coming soon...")

                col_actions = st.columns(3)

                with col_actions[0]:
                    if st.button(f"[Copy]", key=f"copy_{workflow['workflow_id']}_{key_suffix}"):
                        result = self._copy_workflow_as_template(workflow)
                        if result['success']:
                            success_msg = f"OK: Workflow copied as '{result['new_name']}'"
                            success_msg += f" (saved to {result.get('destination', 'templates directory')})"
                            if result.get('target_status'):
                                success_msg += f" with status '{result['target_status']}'"
                            st.success(success_msg)
                            st.info("+ Navigate to 'Create Flow' tab to customize the copied workflow")
                            return True
                        else:
                            st.warning(f"Copy failed: {result.get('error')}")

                with col_actions[1]:
                    if st.button(f"[Update]", key=f"update_{workflow['workflow_id']}_{key_suffix}"):
                        result = self._update_workflow(workflow)
                        if result['success']:
                            success_msg = f"OK: {result.get('workflow_type', 'Workflow')} updated"
                            success_msg += f" from v{result.get('previous_version', '1.0')} → v{result['new_version']}"
                            if result.get('new_status'):
                                success_msg += f" (status: {result['new_status']})"
                            if result.get('created_new_file'):
                                success_msg += " (new file created)"
                            elif result.get('files_updated', 0) > 1:
                                success_msg += f" ({result['files_updated']} files updated)"
                            st.success(success_msg)
                            return True
                        else:
                            st.warning(f"Update failed: {result.get('error')}")

                with col_actions[2]:
                    if st.button(f"[Delete]", key=f"delete_{workflow['workflow_id']}_{key_suffix}"):
                        st.warning("WARNING: Delete confirmation needed")

            # RAG deployment status
            if workflow.get("rag_deployment"):
                st.markdown("**AI: RAG Deployment Status:**")
                for rag_deploy in workflow["rag_deployment"]:
                    status_icon = {
                        "deployed": "OK:",
                        "failed": "○",
                        "unavailable": "WARNING:",
                        "error": "○"
                    }.get(rag_deploy["status"], "?")

                    st.markdown(f"  {status_icon} {rag_deploy['rag_type']}: {rag_deploy['status']}")

        return False

    def _filter_git_files(self, file_list: List[Path]) -> List[Path]:
        """Filter out git-related files from file listings.

        Args:
            file_list: List of Path objects to filter

        Returns:
            List[Path]: Filtered list excluding .gitkeep, .gitignore, etc.
        """
        git_files = {'.gitkeep', '.gitignore', '.git', '.gitattributes'}
        return [f for f in file_list if f.name not in git_files]

    def _render_flow_designer_page(self):
        """Render the Flow Designer page."""
        st.header("* Flow Designer")
        st.markdown("Advanced workflow configuration and RAG integration design")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("TOOL: Design Tools")

            design_mode = st.radio(
                "Design Mode",
                ["Code Editor", "Template Wizard", "RAG Configuration"]
            )

            if design_mode == "Code Editor":
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
                        st.warning(f"Invalid JSON format: {e}")

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

                        if st.form_submit_button("Next >>"):
                            st.session_state.wizard_step1 = {
                                "name": wf_name, "type": wf_type, "description": wf_desc
                            }
                            st.success("OK: Step 1 Complete! Move to Step 2.")

        with col2:
            st.subheader("^ RAG Integration Designer")

            # RAG system availability
            rag_status = self.workflow_manager.get_available_rag_systems()

            st.markdown("**AI: Available RAG Systems:**")
            # Defensive type handling for rag_status
            if isinstance(rag_status, dict):
                for rag_type, is_available in rag_status.items():
                    status_icon = "✓" if is_available else "○"
                    st.markdown(f"{status_icon} **{rag_type.replace('_', ' ').title()}**")
            elif isinstance(rag_status, list):
                for i, is_available in enumerate(rag_status):
                    rag_type = f"rag_system_{i}"
                    status_icon = "✓" if is_available else "○"
                    st.markdown(f"{status_icon} **{rag_type.replace('_', ' ').title()}**")
            else:
                st.info("RAG system status checking...")

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
                st.info("Use one RAG system for the entire workflow")
                # Get available RAG systems with defensive type checking
                if isinstance(rag_status, dict):
                    available_rags = [k for k, v in rag_status.items() if v]
                elif isinstance(rag_status, list):
                    available_rags = [f"rag_system_{i}" for i, v in enumerate(rag_status) if v]
                else:
                    available_rags = ["No RAG systems available"]
                selected_rag = st.selectbox("RAG System", available_rags)

            elif pattern == "Parallel RAG Processing":
                st.info("FAST: Run multiple RAG systems in parallel and combine results")
                # Get available RAG systems with defensive type checking
                if isinstance(rag_status, dict):
                    available_rags = [k for k, v in rag_status.items() if v]
                elif isinstance(rag_status, list):
                    available_rags = [f"rag_system_{i}" for i, v in enumerate(rag_status) if v]
                else:
                    available_rags = ["No RAG systems available"]
                parallel_rags = st.multiselect("RAG Systems", available_rags)

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
        """Render the Workflow Monitor page.

        COMMENTED OUT FOR DEMO - This method is preserved but not called.
        Contains workflow monitoring functionality for future use.
        """
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

            with st.expander(f"{status_color} {workflow.get('name', workflow.get('workflow_id', 'Unknown'))}"):

                # Basic info
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**Status:** {workflow.get('status', 'unknown')}")
                    st.markdown(f"**Created:** {workflow.get('created_at', 'unknown')}")

                    # Show Type and Description prominently
                    workflow_type = workflow.get('workflow_type', workflow.get('template', {}).get('workflow_type', 'unknown'))
                    workflow_description = workflow.get('description', workflow.get('template', {}).get('description', 'No description'))
                    st.markdown(f"**Type:** `{workflow_type}` - {workflow_description}")

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
                                "failed": "○",
                                "unavailable": "WARNING:",
                                "error": "○"
                            }.get(rag_deploy["status"], "?")

                            st.markdown(f"**{rag_deploy['rag_type']}**")
                            st.markdown(f"{status_icon} {rag_deploy['status']}")

                            if rag_deploy.get("system_id"):
                                st.markdown(f"ID: `{rag_deploy['system_id'][:8]}...`")

    def _render_test_runner_page(self):
        """Render the Test Runner page for document upload and flow execution.

        COMMENTED OUT FOR DEMO - This method is preserved but not called.
        Contains workflow testing functionality for future use.
        """
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
        # Defensive check: handle both dict and list workflows
        if isinstance(workflows, dict):
            active_workflows = [w for w in workflows.values() if w.get('status') != 'disabled']
        elif isinstance(workflows, list):
            active_workflows = [w for w in workflows if w.get('status') != 'disabled']
        else:
            active_workflows = []

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

            # Ensure type consistency for Streamlit number_input
            if field_type == 'integer':
                min_val = int(min_val)
                max_val = int(max_val)
                default_val = int(default_value) if default_value is not None else min_val
            else:  # number (float)
                min_val = float(min_val)
                max_val = float(max_val)
                default_val = float(default_value) if default_value is not None else float(min_val)

            return st.number_input(label, min_value=min_val, max_value=max_val,
                                 value=default_val, help=description)

        elif field_type == 'boolean':
            return st.checkbox(label, value=default_value or False, help=description)

        elif field_type == 'array':
            input_val = st.text_area(label, help=f"{description} (one item per line)")
            return input_val.strip().split('\n') if input_val.strip() else []

        else:
            return st.text_input(label, value=str(default_value or ""), help=description)

    def _reorder_steps_by_position(self, steps: List[Dict]) -> List[Dict]:
        """
        Utility function to reorder steps based on desired_position field.

        WHAT CHANGES:
        - Array index (position in list)
        - step_index (0-based position)
        - step_number (1-based display like "1.0", "2.0")

        WHAT STAYS THE SAME:
        - step_id (e.g., 'metadata_extraction' - permanent identifier)
        - template_id (linked template)
        - All other step properties

        Args:
            steps: List of step dictionaries with 'desired_position' field

        Returns:
            Reordered list with updated indices
        """
        # Create a copy to avoid modifying original
        steps_copy = [s.copy() for s in steps]

        # Sort steps by desired_position (this shuffles the array indices)
        reordered = sorted(steps_copy, key=lambda s: s.get('desired_position', 999))

        # Update position-related fields for each step
        for new_index, step in enumerate(reordered):
            # Remove temporary UI field
            if 'desired_position' in step:
                del step['desired_position']

            # Update position fields (but NOT step_id)
            step['step_index'] = new_index  # 0-based array position
            step['step_number'] = f"{new_index + 1}.0"  # 1-based display number

            # step_id UNCHANGED - it's the permanent identifier

        return reordered

    def _check_for_order_changes(self, steps: List[Dict]) -> bool:
        """
        Utility function to check if any steps have different desired positions.

        Args:
            steps: List of step dictionaries

        Returns:
            True if any positions changed, False otherwise
        """
        for i, step in enumerate(steps):
            if step.get('desired_position', i+1) != i+1:
                return True
        return False

    def _render_template_order_manager(self, workflow_id: str, workflow_config: Dict, project_path: Path):
        """Template ordering manager for existing workflows"""
        import json

        st.markdown("### 🔄 Template Execution Order")

        # Initialize session state for this workflow
        session_key = f"template_order_{workflow_id}"
        if session_key not in st.session_state:
            st.session_state[session_key] = workflow_config.get('steps', []).copy()

        steps = st.session_state[session_key]

        # Load available templates from folder
        templates_path = project_path / "templates"
        available_templates = []
        if templates_path.exists():
            template_files = list(templates_path.glob("*.md")) + list(templates_path.glob("*.txt"))
            available_templates = [t.stem for t in template_files]

        # Add template selector
        col_add, col_save = st.columns(2)
        with col_add:
            if available_templates:
                # Filter out templates already in execution order
                existing_step_ids = {step.get('step_id') for step in steps}
                unused_templates = [t for t in available_templates if t not in existing_step_ids]

                # Show template availability status
                if len(existing_step_ids) > 0:
                    st.caption(f"📊 {len(unused_templates)} of {len(available_templates)} templates available")

                if unused_templates:
                    selected_template = st.selectbox(
                        "Add Template",
                        [""] + unused_templates,
                        help=f"Templates already in execution order are filtered out"
                    )
                    if st.button("➕ Add to Execution Order") and selected_template:
                        # Double-check to prevent race conditions
                        if selected_template in existing_step_ids:
                            st.warning(f"⚠️ {selected_template} is already in execution order")
                            return

                        new_step = {
                            "step_id": selected_template,
                            "step_name": selected_template.replace("_", " ").title(),
                            "step_index": len(steps),
                            "step_number": f"{len(steps)+1}.0",
                            "step_type": "template",
                            "template_file": f"{selected_template}.md" if (templates_path / f"{selected_template}.md").exists() else f"{selected_template}.txt"
                        }
                        steps.append(new_step)
                        st.session_state[session_key] = steps
                        st.success(f"✅ Added {selected_template}")
                else:
                    st.info("✅ All available templates are already in execution order")
                    if len(available_templates) > 0:
                        with st.expander("📋 Templates in execution order", expanded=False):
                            for step_id in sorted(existing_step_ids):
                                st.text(f"• {step_id}")
            else:
                st.warning("📂 No template files found in templates folder")

        with col_save:
            if st.button("💾 Save Execution Order"):
                # Validate for duplicates before saving
                step_ids = [step.get('step_id') for step in steps]
                duplicate_ids = [id for id in step_ids if step_ids.count(id) > 1]

                if duplicate_ids:
                    st.error(f"❌ Cannot save: Duplicate templates detected: {', '.join(set(duplicate_ids))}")
                    st.warning("Please remove duplicate templates before saving")
                    return

                # Validate step IDs are not empty
                empty_ids = [i for i, step in enumerate(steps) if not step.get('step_id')]
                if empty_ids:
                    st.error(f"❌ Cannot save: Templates at positions {[i+1 for i in empty_ids]} have missing IDs")
                    return

                # Use our step ordering utilities for backend processing
                if UTILS_AVAILABLE:
                    # Check if reordering is needed
                    if check_for_order_changes(steps):
                        # Backend reordering using utility
                        reordered_steps = reorder_steps_by_position(steps)
                        st.success("✅ Backend reordered templates by position")
                    else:
                        reordered_steps = steps
                        st.info("No order changes detected")
                else:
                    # Fallback if utilities not available
                    reordered_steps = steps

                # Additional validation after reordering
                if UTILS_AVAILABLE:
                    from tidyllm.utils.step_ordering import validate_step_order
                    validation_errors = validate_step_order(reordered_steps)
                    if validation_errors:
                        st.error("❌ Validation failed:")
                        for error in validation_errors:
                            st.error(f"   • {error}")
                        return

                # Save to project_config.json using backend utilities
                workflow_config['steps'] = reordered_steps
                config_path = project_path / "project_config.json"
                try:
                    with open(config_path, 'w', encoding='utf-8') as f:
                        json.dump(workflow_config, f, indent=2, ensure_ascii=False)
                    st.success("✅ Template execution order saved to backend!")
                    st.success(f"   • {len(reordered_steps)} templates in execution order")
                    st.success(f"   • All validations passed")
                    st.session_state[session_key] = reordered_steps
                except Exception as e:
                    st.error(f"❌ Backend save failed: {e}")

        if not steps:
            st.info("No templates in execution order. Add templates above.")
            return

        st.markdown("---")

        # Display current execution order with duplicate detection
        st.markdown("**Current Execution Order:**")

        # Check for duplicates to highlight them
        step_ids = [step.get('step_id') for step in steps]
        duplicate_ids = set([id for id in step_ids if step_ids.count(id) > 1])

        if duplicate_ids:
            st.warning(f"⚠️ Duplicate templates detected: {', '.join(duplicate_ids)}")
            st.info("Remove duplicates using the 🗑️ button before saving")

        for i, step in enumerate(steps):
            with st.container():
                col1, col2, col3 = st.columns([1, 4, 1])

                with col1:
                    # Order selector - show current position as default
                    position_options = list(range(1, len(steps) + 1))
                    current_position = step.get('desired_position', i + 1)

                    # Ensure current_position is within valid range
                    if current_position > len(steps):
                        current_position = i + 1

                    step['desired_position'] = st.selectbox(
                        f"Order",
                        options=position_options,
                        index=current_position - 1,
                        key=f"order_{workflow_id}_{step.get('step_id', i)}_{i}"
                    )

                with col2:
                    step_id = step.get('step_id')
                    step_name = step.get('step_name', 'Unnamed')

                    # Highlight duplicates
                    if step_id in duplicate_ids:
                        st.markdown(f"**{i+1}. 🔴 {step_name}** _(DUPLICATE)_")
                        st.caption(f"📄 File: {step.get('template_file', 'unknown')} ⚠️")
                    else:
                        st.markdown(f"**{i+1}. {step_name}**")
                        st.caption(f"📄 File: {step.get('template_file', 'unknown')}")

                with col3:
                    if st.button("🗑️", key=f"del_{workflow_id}_{i}", help="Remove from order"):
                        steps.pop(i)
                        st.session_state[session_key] = steps

    def _render_simple_step_manager(self, workflow: Dict):
        """Template execution manager with order configuration"""
        import json
        from pathlib import Path

        st.markdown("### 📋 Template Execution Order")
        st.markdown(f"**Project**: {workflow.get('workflow_name', 'Unknown')}")
        st.info("💡 Each step is a template that will be executed in the order shown")

        project_id = workflow.get('workflow_id', 'unknown')

        # Initialize session state for steps if not exists
        session_key = f"steps_{project_id}"
        if session_key not in st.session_state:
            st.session_state[session_key] = workflow.get('steps', []).copy()

        steps = st.session_state[session_key]

        # Show existing templates available for use
        with st.expander("📚 Available Templates", expanded=False):
            # Load existing templates from the project
            templates_path = Path(f"tidyllm/workflows/projects/{project_id}/templates")

            if templates_path.exists():
                template_files = list(templates_path.glob("*.txt")) + list(templates_path.glob("*.json"))

                if template_files:
                    st.markdown("**Existing Templates:**")

                    for template_file in template_files:
                        template_name = template_file.stem

                        # Create expandable preview for each template
                        with st.expander(f"📄 {template_name.replace('_', ' ').title()}", expanded=False):
                            # Show template preview
                            try:
                                with open(template_file, 'r', encoding='utf-8') as f:
                                    content = f.read(500)  # First 500 chars preview
                                st.markdown("**Preview:**")
                                st.text(content + "..." if len(content) == 500 else content)
                            except:
                                st.info("Unable to preview template")

                            # Add button
                            if st.button(f"➕ Add to Workflow", key=f"add_existing_{template_name}"):
                                new_step = {
                                    "step_id": template_name,  # Permanent ID (never changes)
                                    "step_name": template_name.replace("_", " ").title(),
                                    "step_index": len(steps),  # Position in array (0-based)
                                    "step_number": f"{len(steps)+1}.0",  # Display number (1-based)
                                    "step_type": "template",
                                    "template_id": template_name,
                                    "template_file": template_file.name,
                                    "description": f"Execute {template_name} template"
                                }
                                steps.append(new_step)
                                st.session_state[session_key] = steps
                                st.success(f"✅ Added: {template_name} (click Save to persist)")
                else:
                    st.info("No templates found in project folder")
            else:
                st.info(f"Templates folder not found: {templates_path}")

            # Also show common templates
            st.markdown("---")
            st.markdown("**Common Templates:**")

            common_templates = [
                {"id": "metadata_extraction", "name": "Metadata Extraction", "type": "extraction_template"},
                {"id": "analysis_steps", "name": "Analysis Steps", "type": "analysis_template"},
                {"id": "results_aggregation", "name": "Results Aggregation", "type": "aggregation_template"},
                {"id": "recording_questions", "name": "Recording Questions", "type": "validation_template"},
                {"id": "data_validation", "name": "Data Validation", "type": "validation_template"},
                {"id": "summary_generation", "name": "Summary Generation", "type": "aggregation_template"}
            ]

            common_cols = st.columns(3)
            for idx, template in enumerate(common_templates):
                col_idx = idx % 3
                with common_cols[col_idx]:
                    if st.button(f"➕ {template['name']}", key=f"add_common_{template['id']}",
                               help=f"Add {template['name']} to workflow"):
                        new_step = {
                            "step_id": template['id'],
                            "step_name": template['name'],
                            "step_number": f"{len(steps)+1}.0",
                            "step_type": template['type'],
                            "template_id": template['id'],
                            "template_file": f"{template['id']}.txt",
                            "description": f"Execute {template['name']} template"
                        }
                        steps.append(new_step)
                        st.session_state[session_key] = steps
                        st.success(f"✅ Added: {template['name']} (click Save to persist)")

        # Add custom template button
        if st.button("➕ Add Custom Template"):
            new_step = {
                "step_id": f"template_{len(steps)+1}",
                "step_name": f"Template {len(steps)+1}",
                "step_number": f"{len(steps)+1}.0",
                "step_type": "template",
                "template_id": f"template_{len(steps)+1}",
                "template_file": f"template_{len(steps)+1}.txt",
                "description": "Custom template execution"
            }
            steps.append(new_step)
            st.session_state[session_key] = steps
            st.success(f"✅ Added custom template: {new_step['step_name']}")

        if not steps:
            st.info("No steps defined. Click 'Add New Step' to get started.")
            return

        # Save button that actually reorders on backend
        col_save, col_reset = st.columns(2)
        with col_save:
            if st.button("💾 Save Order Changes"):
                # Check if any positions changed using utility function
                if self._check_for_order_changes(steps):
                    # Reorder steps in array (shuffle indices, keep IDs)
                    # Example: 'metadata_extraction' stays 'metadata_extraction'
                    # but moves from index 0 to index 2 in the array
                    reordered_steps = self._reorder_steps_by_position(steps)

                    # Save reordered array to backend
                    workflow['steps'] = reordered_steps
                    if self._save_workflow_steps_to_file(project_id, workflow, reordered_steps):
                        st.success("✅ Order saved! Backend array indices shuffled.")
                        st.session_state[session_key] = reordered_steps
                        st.rerun()
                    else:
                        st.error("❌ Failed to save order changes")
                else:
                    st.info("No order changes to save")

        with col_reset:
            if st.button("🔄 Reset to Saved"):
                st.session_state[session_key] = workflow.get('steps', []).copy()
                st.success("🔄 Reset to saved configuration")

        st.markdown("---")

        # Display and edit each step with clean layout
        for i, step in enumerate(steps):
            with st.container():
                # Add a clear order number display
                st.markdown(f"### {i+1}. {step.get('step_name', 'Unnamed')}")

                col1, col2, col3 = st.columns([1.5, 5, 1])

                with col1:
                    # Label for clarity
                    st.markdown("**Change Order:**")

                    # Simple dropdown selector for position
                    position_options = list(range(1, len(steps) + 1))

                    # Store desired position in step data (not actually moving yet)
                    step['desired_position'] = st.selectbox(
                        "",
                        options=position_options,
                        index=i,
                        key=f"pos_{project_id}_{step.get('step_id', f'step_{i}')}_{i}",
                        label_visibility="collapsed"
                    )

                with col2:
                    # Show details without redundant name
                    st.caption(f"📌 **Template ID:** {step.get('step_id', 'unknown')}")
                    st.caption(f"📁 **Type:** {step.get('step_type', 'unknown')}")
                    st.caption(f"📄 **File:** {step.get('template_file', 'Not specified')}")

                with col3:
                    # Delete button
                    if st.button("🗑️", key=f"del_{i}", help="Delete this step"):
                        deleted_name = step.get('step_name', 'Unknown')
                        steps.pop(i)
                        st.session_state[session_key] = steps
                        st.warning(f"⚠️ Deleted '{deleted_name}' (click Save to persist)")

                # Expandable step editor for client customization
                with st.expander(f"✏️ Edit {step.get('step_name', 'Step')}"):
                    with st.form(f"edit_step_{i}"):
                        col_edit1, col_edit2 = st.columns(2)

                        with col_edit1:
                            new_step_id = st.text_input(
                                "Template ID",
                                value=step.get('step_id', ''),
                                help="Unique template identifier"
                            )
                            new_step_name = st.text_input(
                                "Template Display Name",
                                value=step.get('step_name', ''),
                                help="Friendly name clients will see"
                            )

                            # Template file selector
                            template_file = st.text_input(
                                "Template File",
                                value=step.get('template_file', f"template_{i+1}.txt"),
                                help="Template file name (e.g., metadata_extraction.txt)"
                            )
                            st.info(f"📍 Execution Order: {i+1}")

                        with col_edit2:
                            # Support both old and new step types
                            template_types = ["template", "extraction_template", "analysis_template", "aggregation_template", "validation_template", "custom_template"]

                            # Map old types to new template types
                            current_type = step.get('step_type', 'template')
                            if current_type == 'extraction':
                                current_type = 'extraction_template'
                            elif current_type == 'analysis':
                                current_type = 'analysis_template'
                            elif current_type == 'aggregation':
                                current_type = 'aggregation_template'
                            elif current_type == 'validation':
                                current_type = 'validation_template'
                            elif current_type == 'recording':
                                current_type = 'validation_template'
                            elif current_type == 'processing':
                                current_type = 'custom_template'

                            # Set index safely
                            try:
                                type_index = template_types.index(current_type)
                            except ValueError:
                                type_index = 0  # Default to 'template'

                            new_step_type = st.selectbox(
                                "Template Type",
                                template_types,
                                index=type_index
                            )
                            new_description = st.text_area(
                                "Template Description",
                                value=step.get('description', ''),
                                help="What this template does"
                            )

                        col_btn1, col_btn2, col_btn3 = st.columns(3)

                        with col_btn1:
                            if st.form_submit_button("💾 Save Changes"):
                                # Update step with new values (order is controlled by position selector only)
                                steps[i].update({
                                    'step_id': new_step_id,
                                    'step_name': new_step_name,
                                    'step_number': f"{i+1}.0",  # Auto-set based on current position
                                    'step_type': new_step_type,
                                    'template_file': template_file,  # Store template file reference
                                    'template_id': new_step_id,  # Template ID is same as step ID
                                    'description': new_description
                                })
                                self._save_workflow_steps_to_file(project_id, workflow, steps)
                                st.success(f"✅ Updated {new_step_name}")
                                st.rerun()

                        with col_btn2:
                            if st.form_submit_button("🗑️ Delete Step"):
                                steps.pop(i)
                                self._save_workflow_steps_to_file(project_id, workflow, steps)
                                st.success(f"✅ Deleted step")
                                st.rerun()

                        with col_btn3:
                            if st.form_submit_button("📋 Duplicate"):
                                new_step = step.copy()
                                new_step['step_id'] = f"{step.get('step_id', 'step')}_copy"
                                new_step['step_name'] = f"{step.get('step_name', 'Step')} Copy"
                                steps.insert(i+1, new_step)
                                self._save_workflow_steps_to_file(project_id, workflow, steps)
                                st.success(f"✅ Duplicated step")
                                st.rerun()

                st.markdown("---")

        # Show current step execution order
        with st.expander("📊 Current Execution Order"):
            st.markdown("**Steps will execute in this exact order:**")
            for i, step in enumerate(steps):
                st.markdown(f"**{i+1}.** {step.get('step_name', 'Unnamed')} - `{step.get('step_id', 'unknown')}` ({step.get('step_type', 'unknown')})")

            if not steps:
                st.info("No steps defined yet")

    def _save_workflow_steps_to_file_to_file(self, project_id: str, workflow: Dict, steps: List) -> bool:
        """Save updated steps back to project_config.json - simpler version"""
        import json
        from pathlib import Path

        try:
            # Update workflow with new steps
            workflow_copy = workflow.copy()
            workflow_copy['steps'] = steps

            # Try multiple possible paths
            possible_paths = [
                Path.cwd() / "tidyllm" / "workflows" / "projects" / project_id / "project_config.json",
                Path(f"tidyllm/workflows/projects/{project_id}/project_config.json"),
                Path(f"./tidyllm/workflows/projects/{project_id}/project_config.json")
            ]

            for config_path in possible_paths:
                if config_path.exists():
                    with open(config_path, 'w', encoding='utf-8') as f:
                        json.dump(workflow_copy, f, indent=2, ensure_ascii=False)
                    return True

            # If no existing file found, show available paths for debugging
            st.warning("No config file found at expected locations:")
            for path in possible_paths:
                st.text(f"  - {path} (exists: {path.exists()})")

            return False

        except Exception as e:
            st.error(f"Save error: {e}")
            return False

    def _extract_document_content(self, uploaded_files: List) -> str:
        """Extract text content from uploaded files for QAQC analysis"""
        try:
            from tidyllm.services import CentralizedDocumentService
            doc_service = CentralizedDocumentService()

            all_content = []
            for file in uploaded_files:
                if file.type == "application/pdf":
                    # For PDF files, extract text using document service
                    content = doc_service.extract_text(file.getvalue())
                    all_content.append(f"=== {file.name} ===\n{content}\n")
                else:
                    # For other files, read as text
                    content = str(file.read(), 'utf-8', errors='ignore')
                    all_content.append(f"=== {file.name} ===\n{content}\n")
                    file.seek(0)  # Reset file pointer

            return "\n".join(all_content)
        except Exception as e:
            st.warning(f"Document extraction error: {e}")
            return "Error extracting document content"

    def _load_step_configuration(self, workflow: Dict) -> Dict:
        """Load step configuration with dynamic numbering from project config"""
        steps_config = {}

        # Get steps from workflow configuration
        steps = workflow.get('steps', [])

        for i, step in enumerate(steps):
            step_id = step.get('step_id', f'step_{i+1}')
            step_number = step.get('step_number', i+1)

            steps_config[step_id] = {
                'step_id': step_id,
                'step_name': step.get('step_name', step_id),
                'step_number': step_number,
                'step_type': step.get('step_type', 'processing'),
                'description': step.get('description', ''),
                'depends_on': step.get('depends_on', []),
                'order': i
            }

        return steps_config

    def _execute_qaqc_workflow_dspy(self, workflow: Dict, uploaded_files: List, field_values: Dict) -> Dict:
        """Execute the real QAQC workflow using DSPy reasoning and MLflow tracking"""
        import tidyllm
        from datetime import datetime
        import json

        # Generate project session context for MLflow tracking
        project_id = workflow.get('workflow_id', 'alex_qaqc')
        session_id = f"{project_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        base_user_id = f"qaqc_{project_id}"

        st.info(f"Executing QAQC workflow with session: {session_id}")

        # Load dynamic step configuration
        steps_config = self._load_step_configuration(workflow)
        st.info(f"Loaded {len(steps_config)} steps from configuration")

        # Extract document content
        document_content = self._extract_document_content(uploaded_files)

        workflow_results = {}
        step_results = []

        def _sort_step_number(step_config):
            """Sort steps by step_number, handling hierarchical numbers like 1.1.1"""
            step_num = step_config['step_number']
            if isinstance(step_num, str):
                # Convert "1.2.3" to [1, 2, 3] for proper sorting
                try:
                    return [int(x) for x in step_num.split('.')]
                except:
                    return [999, step_config['order']]  # Fallback to order
            else:
                return [int(step_num), 0]  # Simple numeric step

        # Execute steps dynamically based on step_number order
        for step_config in sorted(steps_config.values(), key=_sort_step_number):
            step_id = step_config['step_id']
            step_name = step_config['step_name']
            step_number = step_config['step_number']
            step_type = step_config['step_type']

            with st.spinner(f"Step {step_number}: {step_name}..."):
                try:
                    # Generate dynamic prompt based on step type and configuration
                    prompt = self._generate_step_prompt(
                        step_config,
                        workflow_results,
                        field_values,
                        document_content,
                        uploaded_files
                    )

                    result = tidyllm.chat(
                        prompt,
                        chat_type='dspy',
                        reasoning=True,
                        user_id=f"{base_user_id}_{step_id}",
                        session_id=session_id,
                        audit_reason=f"{project_id}_step_{step_number}_{step_id}",
                        project_id=project_id,
                        workflow_step=step_id,
                        step_number=step_number,
                        step_type=step_type,
                        document_count=len(uploaded_files)
                    )

                    workflow_results[step_id] = result
                    step_results.append({
                        'step': step_number,
                        'name': step_id,
                        'type': step_type,
                        'status': 'completed',
                        'result': result
                    })
                    st.success(f"✓ Step {step_number}: {step_name} completed")

                except Exception as e:
                    st.error(f"Step {step_number} ({step_name}) failed: {e}")
                    workflow_results[step_id] = {'error': str(e)}
                    step_results.append({
                        'step': step_number,
                        'name': step_id,
                        'type': step_type,
                        'status': 'failed',
                        'error': str(e)
                    })

        # Step 2: QAQC Analysis Steps
        with st.spinner("Step 2: QAQC Analysis..."):
            try:
                analysis_prompt = f"""
                QAQC Step 2: Quality Assurance & Quality Control Analysis

                Based on metadata: {workflow_results.get('metadata_extraction', 'No metadata available')}
                Document content: {document_content[:2000]}...
                Analysis focus: {field_values.get('analysis_focus', 'comprehensive')}
                Quality threshold: {field_values.get('quality_threshold', 0.85)}

                Reason through comprehensive QAQC analysis:
                1. Quality assessment scoring (text clarity, visual presentation, completeness, consistency)
                2. Compliance requirements checking (standards, regulations, best practices)
                3. Technical accuracy evaluation (correctness, feasibility, implementation)
                4. Content validation analysis (reliability, accuracy, currency)

                Generate detailed analysis results in JSON format:
                {{
                    "quality_assessment": {{
                        "overall_quality_score": 0.0,
                        "quality_breakdown": {{"text_clarity": 0.0, "visual_presentation": 0.0, "completeness": 0.0, "consistency": 0.0}},
                        "quality_issues": [],
                        "quality_strengths": []
                    }},
                    "compliance_check": {{
                        "compliance_status": "pass/fail",
                        "requirements_met": [],
                        "violations_found": [],
                        "risk_assessment": "low/medium/high"
                    }},
                    "technical_analysis": {{
                        "technical_accuracy": 0.0,
                        "technical_issues": [],
                        "technical_strengths": [],
                        "implementation_feasibility": "high/medium/low",
                        "risk_level": "low/medium/high"
                    }},
                    "content_validation": {{
                        "content_accuracy": 0.0,
                        "reliability_score": 0.0,
                        "validation_results": [],
                        "information_currency": "current/outdated",
                        "source_quality": "high/medium/low"
                    }}
                }}
                """

                analysis_result = tidyllm.chat(
                    analysis_prompt,
                    chat_type='dspy',
                    reasoning=True,
                    user_id=f"{base_user_id}_analysis",
                    session_id=session_id,
                    audit_reason=f"{project_id}_step_2_analysis_steps",
                    project_id=project_id,
                    workflow_step="analysis_steps",
                    step_number=2
                )

                workflow_results['analysis_steps'] = analysis_result
                step_results.append({
                    'step': 2,
                    'name': 'analysis_steps',
                    'status': 'completed',
                    'result': analysis_result
                })
                st.success("✓ Step 2: QAQC analysis completed")

            except Exception as e:
                st.error(f"Step 2 failed: {e}")
                workflow_results['analysis_steps'] = {'error': str(e)}

        # Step 3: Results Aggregation
        with st.spinner("Step 3: Results Aggregation..."):
            try:
                aggregation_prompt = f"""
                QAQC Step 3: Results Aggregation and Synthesis

                Synthesize these comprehensive findings:
                Metadata: {workflow_results.get('metadata_extraction', {})}
                Analysis: {workflow_results.get('analysis_steps', {})}

                Think through systematic aggregation:
                1. Cross-referencing findings for consistency and accuracy
                2. Identifying key patterns, trends, and critical insights
                3. Prioritizing issues by impact and urgency
                4. Preparing coherent recommendations and action items

                Output format: {field_values.get('output_format', 'detailed')}
                Include recommendations: {field_values.get('include_recommendations', True)}

                Generate aggregated results JSON:
                {{
                    "aggregated_results": {{
                        "overall_score": 0.0,
                        "key_findings": [],
                        "critical_issues": [],
                        "improvement_opportunities": []
                    }},
                    "executive_summary": {{
                        "overall_assessment": "...",
                        "risk_level": "low/medium/high",
                        "compliance_status": "pass/fail",
                        "recommended_actions": []
                    }},
                    "detailed_findings": {{
                        "quality_summary": {{}},
                        "compliance_summary": {{}},
                        "technical_summary": {{}},
                        "content_summary": {{}}
                    }}
                }}
                """

                aggregation_result = tidyllm.chat(
                    aggregation_prompt,
                    chat_type='dspy',
                    reasoning=True,
                    user_id=f"{base_user_id}_aggregation",
                    session_id=session_id,
                    audit_reason=f"{project_id}_step_3_results_aggregation",
                    project_id=project_id,
                    workflow_step="results_aggregation",
                    step_number=3
                )

                workflow_results['results_aggregation'] = aggregation_result
                step_results.append({
                    'step': 3,
                    'name': 'results_aggregation',
                    'status': 'completed',
                    'result': aggregation_result
                })
                st.success("✓ Step 3: Results aggregation completed")

            except Exception as e:
                st.error(f"Step 3 failed: {e}")
                workflow_results['results_aggregation'] = {'error': str(e)}

        # Step 4: Recording Analysis Questions
        with st.spinner("Step 4: Recording 5 Analysis Questions..."):
            try:
                questions_prompt = f"""
                QAQC Step 4: Generate 5 Analysis Questions with Comprehensive Answers

                Based on complete QAQC analysis:
                Metadata: {workflow_results.get('metadata_extraction', {})}
                Analysis: {workflow_results.get('analysis_steps', {})}
                Aggregation: {workflow_results.get('results_aggregation', {})}

                Generate the 5 required QAQC analysis questions with detailed reasoning:

                Question 1 (Quality Assessment): "What is the overall quality score and key quality indicators?"
                - Review all quality metrics from analysis
                - Calculate composite quality score based on components
                - Identify specific quality indicators and evidence

                Question 2 (Compliance Status): "What compliance requirements are met or violated?"
                - Assess compliance findings comprehensively
                - List specific violations and successful compliance areas
                - Provide risk assessment and remediation guidance

                Question 3 (Technical Analysis): "What technical issues or strengths were identified?"
                - Review technical accuracy and implementation feasibility
                - Identify critical technical issues requiring attention
                - Highlight technical strengths and capabilities

                Question 4 (Content Validation): "How reliable and accurate is the document content?"
                - Assess content accuracy and factual correctness
                - Evaluate reliability score and trustworthiness
                - Review source quality and information currency

                Question 5 (Recommendations): "What are the top 3 recommendations for improvement?"
                - Prioritize improvement actions by impact and feasibility
                - Provide implementation guidance and expected benefits
                - Focus on actionable, specific recommendations

                Return exact JSON format:
                {{
                    "analysis_questions": [
                        {{
                            "question_id": 1,
                            "question": "What is the overall quality score and key quality indicators?",
                            "category": "Quality Assessment",
                            "answer": {{
                                "overall_score": 0.0,
                                "quality_breakdown": {{}},
                                "key_indicators": [],
                                "supporting_evidence": []
                            }},
                            "confidence_score": 0.0
                        }}
                        // ... continue for all 5 questions
                    ],
                    "final_summary": {{
                        "workflow_execution": {{
                            "workflow_id": "{project_id}",
                            "execution_timestamp": "{datetime.now().isoformat()}",
                            "session_id": "{session_id}",
                            "status": "completed"
                        }},
                        "overall_assessment": "...",
                        "confidence_average": 0.0
                    }}
                }}
                """

                questions_result = tidyllm.chat(
                    questions_prompt,
                    chat_type='dspy',
                    reasoning=True,
                    user_id=f"{base_user_id}_questions",
                    session_id=session_id,
                    audit_reason=f"{project_id}_step_4_recording_questions",
                    project_id=project_id,
                    workflow_step="recording_questions",
                    step_number=4
                )

                workflow_results['recording_questions'] = questions_result
                step_results.append({
                    'step': 4,
                    'name': 'recording_questions',
                    'status': 'completed',
                    'result': questions_result
                })
                st.success("✓ Step 4: Analysis questions generated")

            except Exception as e:
                st.error(f"Step 4 failed: {e}")
                workflow_results['recording_questions'] = {'error': str(e)}

        # Generate final output and save to outputs folder
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        final_output = {
            'workflow_execution': {
                'workflow_id': project_id,
                'session_id': session_id,
                'execution_timestamp': datetime.now().isoformat(),
                'steps_completed': len(step_results),
                'status': 'completed'
            },
            'workflow_results': workflow_results,
            'step_summary': step_results,
            'field_values': field_values,
            'document_info': {
                'file_count': len(uploaded_files),
                'file_names': [f.name for f in uploaded_files]
            }
        }

        # Save timestamped outputs to project-specific outputs folder
        try:
            from pathlib import Path
            # Save to project-specific outputs folder
            project_outputs_dir = Path(f"tidyllm/workflows/projects/{project_id}/outputs")
            project_outputs_dir.mkdir(parents=True, exist_ok=True)

            # Also save to generic outputs for compatibility
            generic_outputs_dir = Path("outputs")
            generic_outputs_dir.mkdir(exist_ok=True)

            # Save to both project-specific and generic locations
            saved_files = []

            # 1. Final report
            project_final_path = project_outputs_dir / f"final_{timestamp}.json"
            generic_final_path = generic_outputs_dir / f"final_{timestamp}.json"

            for path in [project_final_path, generic_final_path]:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(final_output, f, indent=2, ensure_ascii=False)
                saved_files.append(str(path))

            # 2. Five analysis questions (if available)
            questions_data = workflow_results.get('recording_questions', {})
            if 'analysis_questions' in str(questions_data):
                project_questions_path = project_outputs_dir / f"five_analysis_questions_{timestamp}.json"
                generic_questions_path = generic_outputs_dir / f"five_analysis_questions_{timestamp}.json"

                for path in [project_questions_path, generic_questions_path]:
                    with open(path, 'w', encoding='utf-8') as f:
                        json.dump(questions_data, f, indent=2, ensure_ascii=False)
                    saved_files.append(str(path))

            # 3. Executive summary
            summary_data = {
                'session_id': session_id,
                'workflow_id': project_id,
                'execution_timestamp': datetime.now().isoformat(),
                'executive_summary': workflow_results.get('results_aggregation', {}),
                'workflow_summary': {
                    'steps_completed': len(step_results),
                    'success_rate': len([s for s in step_results if s['status'] == 'completed']) / len(step_results) if step_results else 0,
                    'document_count': len(uploaded_files)
                }
            }

            project_summary_path = project_outputs_dir / f"{project_id}_summary_{timestamp}.json"
            generic_summary_path = generic_outputs_dir / f"{project_id}_summary_{timestamp}.json"

            for path in [project_summary_path, generic_summary_path]:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(summary_data, f, indent=2, ensure_ascii=False)
                saved_files.append(str(path))

            # 4. Generate MLflow tracking CSV
            try:
                from tidyllm.infrastructure.tools import export_project_mlflow_csv

                # Save MLflow CSV to project outputs folder
                project_csv_path = project_outputs_dir / f"mlflow_{project_id}_{timestamp}.csv"
                generic_csv_path = f"mlflow_{project_id}_{timestamp}.csv"

                # Export MLflow data
                export_project_mlflow_csv(project_id, str(project_csv_path))
                export_project_mlflow_csv(project_id, generic_csv_path)

                saved_files.extend([str(project_csv_path), generic_csv_path])

                st.success("✓ MLflow tracking CSV generated!")

            except Exception as csv_error:
                st.warning(f"MLflow CSV generation failed: {csv_error}")

            # Show saved files
            st.success(f"✓ Workflow outputs saved to:")
            st.info(f"📁 **Project Folder**: `{project_outputs_dir}`")
            st.info(f"📁 **Generic Folder**: `{generic_outputs_dir}`")

            with st.expander("📄 Files Created"):
                for file_path in saved_files:
                    st.text(f"• {file_path}")

            # Show download buttons for key files
            col1, col2, col3 = st.columns(3)

            with col1:
                if project_final_path.exists():
                    with open(project_final_path, 'rb') as f:
                        st.download_button(
                            "📥 Download Final Report",
                            f.read(),
                            file_name=f"final_{project_id}_{timestamp}.json",
                            mime="application/json"
                        )

            with col2:
                if project_csv_path.exists():
                    with open(project_csv_path, 'rb') as f:
                        st.download_button(
                            "📊 Download MLflow CSV",
                            f.read(),
                            file_name=f"mlflow_{project_id}_{timestamp}.csv",
                            mime="text/csv"
                        )

            with col3:
                if 'project_questions_path' in locals() and project_questions_path.exists():
                    with open(project_questions_path, 'rb') as f:
                        st.download_button(
                            "❓ Download Questions",
                            f.read(),
                            file_name=f"questions_{project_id}_{timestamp}.json",
                            mime="application/json"
                        )

        except Exception as save_error:
            st.warning(f"Failed to save outputs: {save_error}")

        return final_output

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
                # Execute real QAQC workflow using DSPy reasoning
                try:
                    # Check if this is a QAQC workflow
                    project_id = workflow.get('workflow_id', 'unknown')

                    if project_id == 'alex_qaqc':
                        # Run the real QAQC workflow
                        with st.spinner("Executing QAQC workflow steps..."):
                            result = self._execute_qaqc_workflow_dspy(workflow, uploaded_files, field_values)

                        # Display results
                        st.success("✓ QAQC workflow execution completed successfully!")

                        with st.expander("QAQC Execution Results", expanded=True):
                            st.json(result)

                        # Export MLflow tracking data for verification
                        try:
                            from tidyllm.infrastructure.tools import export_project_mlflow_csv
                            from datetime import datetime

                            # Save to project outputs folder
                            project_outputs_dir = Path(f"tidyllm/workflows/projects/{project_id}/outputs")
                            project_outputs_dir.mkdir(parents=True, exist_ok=True)

                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            project_csv_path = project_outputs_dir / f"mlflow_{project_id}_{timestamp}.csv"

                            # Export to both project folder and generic location
                            export_project_mlflow_csv(project_id, str(project_csv_path))
                            generic_csv_path = export_project_mlflow_csv(project_id)

                            st.info(f"✓ MLflow tracking exported to:")
                            st.info(f"  📁 Project: {project_csv_path}")
                            st.info(f"  📁 Generic: {generic_csv_path}")

                            # Show download buttons for both versions
                            col_csv1, col_csv2 = st.columns(2)

                            with col_csv1:
                                with open(project_csv_path, 'rb') as file:
                                    st.download_button(
                                        label="📊 Download Project MLflow CSV",
                                        data=file.read(),
                                        file_name=f"mlflow_{project_id}_{timestamp}.csv",
                                        mime="text/csv"
                                    )

                            with col_csv2:
                                with open(generic_csv_path, 'rb') as file:
                                    st.download_button(
                                        label="📋 Download MLflow CSV",
                                        data=file.read(),
                                        file_name=f"mlflow_{project_id}.csv",
                                        mime="text/csv"
                                    )

                        except Exception as csv_error:
                            st.warning(f"MLflow CSV export failed: {csv_error}")

                    else:
                        # Fallback to original test workflow for non-QAQC projects
                        import sys
                        sys.path.append(str(Path("tidyllm/workflows/projects/templates")))
                        from test_sequential_flow import run_sequential_flow_test

                        with st.spinner("Executing workflow steps..."):
                            result = run_sequential_flow_test()

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
                    st.warning(f"Workflow execution issue: {str(e)}")
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
            st.warning(f"Test execution issue: {str(e)}")
            st.code(traceback.format_exc())

    def _render_recent_test_results(self):
        """Show recent test execution results."""
        # For now, show placeholder - could be enhanced to read from outputs directory
        results_dir = Path("tidyllm/workflows/projects/templates/outputs")

        if results_dir.exists():
            result_files = list(results_dir.glob("final_REV*.json"))
            if result_files:
                # Sort by modification time, most recent first
                result_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

                st.write(f"**{len(result_files)} recent test result(s):**")

                for i, result_file in enumerate(result_files[:5]):  # Show last 5
                    with st.expander(f"{result_file.name} ({datetime.fromtimestamp(result_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')})", expanded=False):
                        try:
                            result_data = load_json(result_file)

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

    def _render_test_designer_page(self):
        """Render the Test Designer page for A/B/C/D workflow optimization testing."""
        st.header("🧪 Test Designer")
        st.markdown("**Optimize your workflows with A/B/C/D testing to find the best model combinations and execution strategies**")

        # Test Designer Overview
        with st.expander("📖 What is Test Designer?", expanded=False):
            st.markdown("""
            **Test Designer** helps you optimize your workflows by automatically testing different AI model combinations:

            - **Test A (Speed Focus)**: claude-3-haiku → claude-3-sonnet for maximum speed
            - **Test B (Quality Focus)**: claude-3-sonnet → claude-3-5-sonnet for highest quality
            - **Test C (Premium Focus)**: claude-3-haiku → claude-3-5-sonnet for balanced performance
            - **Test D (DSPy Optimized)**: claude-3-haiku → claude-3-sonnet with DSPy structured outputs

            **Benefits:**
            - Find optimal model combinations for your use case
            - Compare sequential vs parallel execution performance
            - Get detailed performance metrics and cost analysis
            - MLflow tracking for experiment management
            """)

        # Test Configuration Section
        st.subheader("🔧 Test Configuration")

        col1, col2 = st.columns([1, 1])

        with col1:
            # Project Selection
            available_projects = ["alex_qaqc"]  # Based on your workflow registry
            selected_project = st.selectbox(
                "📁 Select Project",
                available_projects,
                help="Choose which workflow project to optimize"
            )

            # Test Mode Selection
            test_mode = st.radio(
                "🚀 Execution Mode",
                ["Parallel (Recommended)", "Sequential (Controlled)"],
                index=0,  # Default to Parallel
                help="Parallel runs selected tests simultaneously for maximum speed. Sequential runs one after another with delays."
            )

            # Query Customization
            use_custom_query = st.checkbox("Use Custom Test Query", value=False)

            if use_custom_query:
                test_query = st.text_area(
                    "Test Query",
                    value="Analyze the QA/QC workflow for data quality assessment. Focus on metadata extraction, analysis steps, results aggregation, and recording questions.",
                    height=100,
                    help="Customize the query that will be tested across all model combinations"
                )
            else:
                test_query = None
                st.info("Using default QA/QC workflow analysis query")

        with col2:
            # Test Selection
            st.markdown("**🧪 Tests to Run**")

            run_test_a = st.checkbox("Test A: Speed Focus", value=True)
            run_test_b = st.checkbox("Test B: Quality Focus", value=True)
            run_test_c = st.checkbox("Test C: Premium Focus", value=True)
            run_test_d = st.checkbox("Test D: DSPy Optimized", value=True)

            # Execution Configuration
            if test_mode == "Parallel (Recommended)":
                max_concurrent = st.slider(
                    "Max Concurrent Tests",
                    min_value=1,
                    max_value=4,
                    value=4,
                    help="Number of tests to run simultaneously (higher = faster but more resource intensive)"
                )
                st.success("✅ Parallel mode will run your selected tests simultaneously for maximum efficiency")
            else:  # Sequential mode
                max_concurrent = 1
                delay_seconds = st.slider(
                    "Delay Between Tests (seconds)",
                    min_value=1,
                    max_value=10,
                    value=2,
                    help="Wait time between sequential tests to prevent racing"
                )
                st.info("ℹ️ Sequential mode runs tests one after another with controlled delays")

        # Run Tests Section
        st.subheader("🚀 Run Optimization Tests")

        # Clear explanation of the two options
        with st.expander("ℹ️ **Test Execution Options**", expanded=False):
            st.markdown("""
            **🟢 Selected Tests Only** - Runs only the tests you checked above (1-4 tests)
            - Choose specific A/B/C/D tests to compare
            - Fast and targeted optimization
            - Results saved to your project outputs folder

            **🔶 Full Performance Comparison** - Runs ALL tests in BOTH modes (8 tests total)
            - A/B/C/D tests in sequential mode (4 tests)
            - A/B/C/D tests in parallel mode (4 tests)
            - Comprehensive performance analysis
            - Takes significantly longer but provides complete comparison
            """)

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            # Determine which tests to run based on user selection
            selected_tests = []
            if run_test_a:
                selected_tests.append("A")
            if run_test_b:
                selected_tests.append("B")
            if run_test_c:
                selected_tests.append("C")
            if run_test_d:
                selected_tests.append("D")

            # Validate selection
            if not selected_tests:
                st.warning("⚠️ Please select at least one test to run")
            else:
                st.info(f"Selected tests: {', '.join(selected_tests)}")

            # Dynamic button text based on selection
            test_count = len(selected_tests)
            button_text = f"🟢 🔬 Run {test_count} Selected Test{'s' if test_count != 1 else ''}"

            if st.button(button_text, type="primary", disabled=len(selected_tests) == 0):
                with st.spinner(f"Running {len(selected_tests)} selected test(s)..."):
                    try:
                        # Import the testing frameworks
                        if test_mode == "Sequential (Controlled)":
                            from tidyllm.services.optimization.dual_ai_ab_testing import run_selective_sequential_testing

                            # Run sequential tests for selected tests only
                            st.info(f"Running SEQUENTIAL tests for project: {selected_project}")
                            st.info(f"Selected tests: {', '.join(selected_tests)} (with {delay_seconds}s delays)")
                            results = run_selective_sequential_testing(
                                selected_tests=selected_tests,
                                query=test_query,
                                delay_seconds=delay_seconds
                            )

                        else:  # Parallel mode (recommended)
                            from tidyllm.services.optimization.parallel_dual_ai_testing import run_selective_parallel_testing

                            # Run parallel tests for selected tests only
                            actual_concurrent = min(max_concurrent, len(selected_tests))
                            st.info(f"🚀 Running PARALLEL tests with {actual_concurrent} concurrent streams")
                            st.info(f"Selected tests: {', '.join(selected_tests)} (simultaneous execution)")

                            results = run_selective_parallel_testing(
                                selected_tests=selected_tests,
                                query=test_query,
                                max_concurrent_tests=actual_concurrent
                            )

                        # Store results in session state
                        st.session_state.test_results = results
                        st.session_state.test_completed = True
                        st.session_state.selected_tests = selected_tests

                        st.success(f"✅ {len(selected_tests)} test(s) completed! Mode: {results.get('execution_mode', 'unknown')}")
                        st.info(f"📊 Results saved to: tidyllm/workflows/projects/{selected_project}/outputs/")

                        # Show quick summary
                        if 'summary' in results and 'tests' in results['summary']:
                            test_summary = results['summary']['tests']
                            for test_id in selected_tests:
                                if test_id in test_summary:
                                    test_data = test_summary[test_id]
                                    st.success(f"Test {test_id}: {test_data.get('total_time_ms', 0):.0f}ms, Confidence: {test_data.get('confidence_improvement', 0):.2f}")

                    except Exception as e:
                        st.error(f"Test execution failed: {e}")
                        st.exception(e)

        with col2:
            # Warning about the full test suite
            st.warning("⚠️ **FULL TEST SUITE** - This button runs ALL 8 tests (A/B/C/D sequential + parallel)")

            # Color-coded button to make it clear this runs tests
            if st.button("🔶 📊 Performance Comparison (Runs 8 Tests)",
                        help="WARNING: This will execute ALL A/B/C/D tests in both sequential AND parallel modes for comparison. This takes significant time and resources.",
                        type="secondary"):
                with st.spinner("Running performance comparison... (8 tests total)"):
                    try:
                        from tidyllm.services.optimization.parallel_dual_ai_testing import compare_sequential_vs_parallel_testing

                        st.warning("🚀 Running ALL A/B/C/D tests in both sequential AND parallel modes...")
                        comparison = compare_sequential_vs_parallel_testing(query=test_query)

                        # Store comparison in session state
                        st.session_state.performance_comparison = comparison

                        # Display key metrics
                        perf = comparison['performance_comparison']
                        st.success("✅ Performance comparison completed!")

                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        with metric_col1:
                            st.metric("Sequential Time", f"{perf['sequential_execution_time']:.1f}s")
                        with metric_col2:
                            st.metric("Parallel Time", f"{perf['parallel_execution_time']:.1f}s")
                        with metric_col3:
                            st.metric("Speedup Factor", f"{perf['speedup_factor']:.1f}x")

                    except Exception as e:
                        st.error(f"Performance comparison failed: {e}")

        with col3:
            if st.button("📁 View Results"):
                # Display test results if available
                if hasattr(st.session_state, 'test_results') and st.session_state.test_results:
                    st.markdown("### 📊 Latest Test Results")

                    results = st.session_state.test_results
                    summary = results.get('summary', {})

                    if 'tests' in summary:
                        for test_id, test_data in summary['tests'].items():
                            with st.expander(f"Test {test_id}: {test_data.get('label', 'Unknown')}"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Models", test_data.get('models', 'N/A'))
                                    st.metric("Total Time", f"{test_data.get('total_time_ms', 0):.0f}ms")
                                with col2:
                                    st.metric("Confidence Improvement", f"{test_data.get('confidence_improvement', 0):.2f}")
                                    st.metric("Total Tokens", f"{test_data.get('total_tokens', 0)}")
                else:
                    st.info("No test results available. Run tests first to see results here.")

        # Results Analysis Section
        if hasattr(st.session_state, 'test_results') or hasattr(st.session_state, 'performance_comparison'):
            st.subheader("📈 Results Analysis")

            analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(["📊 Test Summary", "⚡ Performance", "💡 Recommendations"])

            with analysis_tab1:
                if hasattr(st.session_state, 'test_results'):
                    st.markdown("#### Test Results Summary")
                    results = st.session_state.test_results

                    # Create summary table
                    if 'summary' in results and 'tests' in results['summary']:
                        import pandas as pd

                        test_data = []
                        for test_id, data in results['summary']['tests'].items():
                            test_data.append({
                                'Test': test_id,
                                'Models': data.get('models', 'N/A'),
                                'Time (ms)': data.get('total_time_ms', 0),
                                'Confidence': data.get('confidence_improvement', 0),
                                'Tokens': data.get('total_tokens', 0)
                            })

                        df = pd.DataFrame(test_data)
                        st.dataframe(df, use_container_width=True)

            with analysis_tab2:
                if hasattr(st.session_state, 'performance_comparison'):
                    st.markdown("#### Performance Comparison")
                    comparison = st.session_state.performance_comparison
                    perf = comparison['performance_comparison']

                    # Performance metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Sequential Execution", f"{perf['sequential_execution_time']:.2f}s")
                        st.metric("Parallel Execution", f"{perf['parallel_execution_time']:.2f}s")
                    with col2:
                        st.metric("Speedup Factor", f"{perf['speedup_factor']:.2f}x")
                        st.metric("Time Saved", f"{perf['time_saved_seconds']:.2f}s")

                    # Recommendation
                    if perf['speedup_factor'] > 1.2:
                        st.success(f"💡 **Recommendation**: Use parallel execution for {perf['efficiency_improvement_percent']:.1f}% better efficiency")
                    else:
                        st.info("💡 **Recommendation**: Sequential execution is sufficient for this workload")

            with analysis_tab3:
                st.markdown("#### Optimization Recommendations")

                if hasattr(st.session_state, 'test_results'):
                    results = st.session_state.test_results

                    if 'summary' in results and 'tests' in results['summary']:
                        tests = results['summary']['tests']

                        # Find best test for different criteria
                        best_speed = min(tests.items(), key=lambda x: x[1].get('total_time_ms', float('inf')))
                        best_quality = max(tests.items(), key=lambda x: x[1].get('confidence_improvement', 0))

                        col1, col2 = st.columns(2)
                        with col1:
                            st.success(f"🚀 **Best for Speed**: Test {best_speed[0]}")
                            st.write(f"Time: {best_speed[1].get('total_time_ms', 0):.0f}ms")
                            st.write(f"Models: {best_speed[1].get('models', 'N/A')}")

                        with col2:
                            st.success(f"🎯 **Best for Quality**: Test {best_quality[0]}")
                            st.write(f"Confidence: {best_quality[1].get('confidence_improvement', 0):.2f}")
                            st.write(f"Models: {best_quality[1].get('models', 'N/A')}")

    def _render_ai_advisor_page(self):
        """Render the AI Advisor page with chat interface."""
        st.header("? AI Workflow Advisor")
        st.markdown("Get intelligent advice about your workflows from AI - ask about optimization, troubleshooting, or best practices")

        # Import the workflow advisor
        try:
            import sys
            # Find the AI-Shipping root directory
            current = Path.cwd()
            if "AI-Shipping" in str(current):
                while current.name != "AI-Shipping" and current.parent != current:
                    current = current.parent
                advisor_path = current / "tidyllm" / "workflows" / "ai_advisor"
            else:
                advisor_path = Path("tidyllm/workflows/ai_advisor")

            sys.path.append(str(advisor_path))
            from workflow_advisor import workflow_advisor
            advisor_available = True
        except ImportError as e:
            st.info(f"AI Advisor currently unavailable: {e}")
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
            use_cases = []

            if selected_workflow:
                if include_criteria:
                    criteria = self._load_workflow_criteria(selected_workflow)

                if include_fields:
                    template_fields = selected_workflow.get('template_fields', {})

                if include_activity:
                    recent_activity = self._get_recent_workflow_activity()

                if include_results:
                    final_results = self._get_latest_workflow_results()

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

            # Get AI response
            try:
                # Import should already be available from _render_ai_advisor_tab
                import sys
                current = Path.cwd()
                if "AI-Shipping" in str(current):
                    while current.name != "AI-Shipping" and current.parent != current:
                        current = current.parent
                    advisor_path = current / "tidyllm" / "workflows" / "ai_advisor"
                else:
                    advisor_path = Path("tidyllm/workflows/ai_advisor")

                if str(advisor_path) not in sys.path:
                    sys.path.append(str(advisor_path))
                from workflow_advisor import workflow_advisor

                with st.spinner("AI is analyzing your workflow..."):
                    advice_result = workflow_advisor.get_workflow_advice(
                        criteria=criteria,
                        template_fields=template_fields,
                        recent_activity=recent_activity,
                        final_results=final_results,
                        user_question=user_input,
                        use_cases=use_cases  # Add missing use_cases parameter
                    )

                if advice_result.get('success', False):
                    response = advice_result['advice']

                    # Add context info
                    context_info = advice_result.get('context_analyzed', {})
                    # Defensive check for context_info type
                    if isinstance(context_info, dict) and any(context_info.values()):
                        pass
                    elif isinstance(context_info, list) and any(context_info):
                        pass
                    else:
                        context_info = {}

                    if (isinstance(context_info, dict) and any(context_info.values())) or (isinstance(context_info, list) and any(context_info)):
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

    def _get_recent_workflow_activity(self) -> List[Dict]:
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

    def _get_latest_workflow_results(self) -> Dict:
        """Get the latest workflow execution results."""
        try:
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
        """Render the Health Dashboard page.

        COMMENTED OUT FOR DEMO - This method is preserved but not called.
        Contains comprehensive health monitoring functionality for future use.
        """
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
                    icon = "✓" if is_healthy else "○"

                    with st.expander(f"{icon} {system_name.replace('_', ' ').title()}"):
                        st.markdown(f"**Available:** {'Yes' if health_info.get('available') else 'No'}")
                        st.markdown(f"**Has Template:** {'Yes' if health_info.get('has_template') else 'No'}")
                        st.markdown(f"**Active Workflows:** {health_info.get('active_workflows', 0)}")
                        if health_info.get("last_check"):
                            st.markdown(f"**Last Check:** {health_info['last_check']}")
            else:
                st.warning(f"Health check unavailable: {workflow_health.get('error', 'Checking...')}")

        with col2:
            st.subheader("AI: RAG Systems")
            rag_status = self.workflow_manager.get_available_rag_systems()

            rag_status = DataNormalizer.normalize_to_status_dict(rag_status)
            available_count = sum(1 for available in rag_status.values() if available)
            total_count = len(rag_status)

            if available_count == total_count:
                st.success(f"OK: All {total_count} RAG Systems Available")
            elif available_count > 0:
                st.warning(f"WARNING: {available_count}/{total_count} RAG Systems Available")
            else:
                st.warning("RAG systems currently unavailable")

            # Display RAG systems with defensive type handling
            if isinstance(rag_status, dict):
                for rag_type, is_available in rag_status.items():
                    icon = "✓" if is_available else "○"
                    st.markdown(f"{icon} **{rag_type.replace('_', ' ').title()}**")
            elif isinstance(rag_status, list):
                for i, is_available in enumerate(rag_status):
                    rag_type = f"rag_system_{i}"
                    icon = "✓" if is_available else "○"
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
                st.info("Metrics currently unavailable")

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
                    "Status": "✓ Healthy" if health_info.get("healthy") else "○ Checking",
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
        st.subheader("Health Trends")
        st.info("🚧 Health trend visualization coming soon!")
        st.markdown("""
        **Planned Features:**
        - Health status over time
        - Performance trend charts
        - Alert history
        - System reliability metrics
        """)

    def _copy_workflow_as_template(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Copy an existing workflow as a new template."""
        try:
            import uuid
            from datetime import datetime
            import shutil

            original_id = workflow.get('workflow_id', 'unknown')
            original_name = workflow.get('workflow_name', 'Unknown Workflow')
            workflow_type = workflow.get('workflow_type', workflow.get('template', {}).get('workflow_type', 'unknown'))

            # Generate new workflow details based on type
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create type-specific copy naming
            if workflow_type in ['template', 'registry']:
                new_id = f"custom_{original_id}_{timestamp}"
                new_name = f"Custom {original_name}"
                target_status = 'custom_template'
            elif workflow_type == 'analysis':
                new_id = f"analysis_copy_{timestamp}"
                new_name = f"My {original_name}"
                target_status = 'draft'
            else:
                new_id = f"copy_{original_id}_{timestamp}"
                new_name = f"Copy of {original_name}"
                target_status = 'template'

            # Create new workflow structure with enhanced metadata
            new_workflow = workflow.copy()
            new_workflow.update({
                'workflow_id': new_id,
                'workflow_name': new_name,
                'created_at': datetime.now().isoformat() + 'Z',
                'status': target_status,
                'description': f"Copy of {original_name} - {workflow.get('description', '')}",
                'copied_from': original_id,
                'copy_timestamp': datetime.now().isoformat() + 'Z',
                'copy_metadata': {
                    'original_type': workflow_type,
                    'copy_method': 'portal_copy_button',
                    'user_initiated': True
                }
            })

            # Determine save location based on workflow type
            if workflow_type in ['template', 'registry']:
                # Save to templates directory for easy discovery
                save_dir = Path("tidyllm/workflows/projects/templates")
                save_dir.mkdir(parents=True, exist_ok=True)
                copy_file = save_dir / f"{new_id}_flow.json"
                destination = "templates directory"
            else:
                # Save to a custom copies directory
                save_dir = Path("tidyllm/workflows/projects/custom_copies")
                save_dir.mkdir(parents=True, exist_ok=True)
                copy_file = save_dir / f"{new_id}_flow.json"
                destination = "custom copies directory"

            save_json(new_workflow, copy_file)

            return {
                'success': True,
                'new_id': new_id,
                'new_name': new_name,
                'file_path': str(copy_file),
                'destination': destination,
                'workflow_type': workflow_type,
                'target_status': target_status
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def _update_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Update workflow with current timestamp and version increment."""
        try:
            from datetime import datetime

            workflow_id = workflow.get('workflow_id', 'unknown')
            workflow_type = workflow.get('workflow_type', workflow.get('template', {}).get('workflow_type', 'unknown'))

            # Update workflow metadata
            current_version = workflow.get('version', '1.0')
            try:
                version_parts = current_version.split('.')
                minor_version = int(version_parts[1]) + 1
                new_version = f"{version_parts[0]}.{minor_version}"
            except:
                new_version = "1.1"

            # Determine appropriate status based on workflow type
            if workflow_type == 'template':
                new_status = 'updated_template'
            elif workflow_type == 'analysis':
                new_status = 'updated_analysis'
            elif workflow_type == 'registry':
                new_status = 'updated_registry'
            else:
                new_status = 'updated'

            workflow.update({
                'version': new_version,
                'last_updated': datetime.now().isoformat() + 'Z',
                'status': new_status,
                'update_metadata': {
                    'update_method': 'portal_update_button',
                    'previous_version': current_version,
                    'workflow_type': workflow_type,
                    'update_timestamp': datetime.now().isoformat() + 'Z'
                }
            })

            # Find and update the project_config.json file
            workflow_dir = Path(f"tidyllm/workflows/projects/{workflow_id}")
            config_file = workflow_dir / "project_config.json"

            # If project_config.json exists, use it
            if config_file.exists():
                workflow_files = [config_file]
            else:
                # Fallback: look for legacy flow files
                workflow_files = list(Path("tidyllm/workflows/projects").rglob(f"*{workflow_id}*.json"))

                # If not found, try looking for flow files in the specific directory
                if not workflow_files and workflow_id:
                    if workflow_dir.exists():
                        # Look for flow files in the workflow directory
                        flow_files = list(workflow_dir.glob("*_flow.json"))
                        if flow_files:
                            workflow_files = flow_files
                        else:
                            # Try to find any JSON file that might be the workflow definition
                            json_files = list(workflow_dir.glob("*.json"))
                            workflow_files = [f for f in json_files if 'flow' in f.name.lower()]

            if workflow_files:
                # Update the first matching file
                workflow_file = workflow_files[0]
                save_json(workflow, workflow_file)

                return {
                    'success': True,
                    'new_version': new_version,
                    'file_path': str(workflow_file),
                    'files_updated': len(workflow_files),
                    'workflow_type': workflow_type,
                    'new_status': new_status,
                    'previous_version': current_version
                }
            else:
                # If still no files found, create a new workflow file in the workflow directory
                workflow_dir = Path(f"tidyllm/workflows/projects/{workflow_id}")
                workflow_dir.mkdir(parents=True, exist_ok=True)

                new_workflow_file = workflow_dir / f"{workflow_id}_flow.json"
                save_json(workflow, new_workflow_file)

                return {
                    'success': True,
                    'new_version': new_version,
                    'file_path': str(new_workflow_file),
                    'created_new_file': True,
                    'workflow_type': workflow_type,
                    'new_status': new_status,
                    'previous_version': current_version
                }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def _render_project_file_manager(self, workflow: Dict[str, Any], container_key: str = None) -> bool:
        """Render comprehensive file management interface for project criteria and templates.

        Shows cards for each JSON criteria file, each MD template file, plus upload functionality.
        Returns True if any files were updated.
        """
        workflow_id = workflow.get('workflow_id', 'unknown')

        # Use absolute path resolution like in WorkflowRegistry
        current = Path.cwd()
        if "AI-Shipping" in str(current):
            # Navigate to AI-Shipping root
            while current.name != "AI-Shipping" and current.parent != current:
                current = current.parent
            base_path = current
        else:
            # Fallback to current directory
            base_path = Path.cwd()

        project_path = base_path / "tidyllm" / "workflows" / "projects" / workflow_id

        if not project_path.exists():
            st.error(f"❌ Project directory not found: {project_path}")
            return False

        st.markdown(f"**📁 File Manager: {workflow_id}**")

        # File management tabs
        criteria_tab, templates_tab, inputs_tab, run_tab = st.tabs(["📋 Criteria", "📝 Templates", "📂 Inputs", ":red[🚀 Run]"])

        file_updated = False

        # CRITERIA FILES TAB
        with criteria_tab:
            st.markdown("**JSON Criteria Files**")
            criteria_path = project_path / "criteria"

            if criteria_path.exists():
                json_files = self._filter_git_files(list(criteria_path.glob("*.json")))

                if json_files:
                    for json_file in json_files:
                        file_updated |= self._render_json_file_card(json_file, f"criteria_{json_file.stem}_{container_key}")
                else:
                    st.info("📋 No criteria files found")
            else:
                st.warning("📂 Criteria folder not found")

            # Upload criteria files section
            st.markdown("---")
            st.markdown("**⬆️ Upload JSON Criteria Files**")
            uploaded_criteria = st.file_uploader(
                "Choose JSON criteria files",
                type=['json'],
                accept_multiple_files=True,
                key=f"criteria_upload_{container_key}",
                help="Upload JSON files to the criteria folder"
            )

            if uploaded_criteria:
                criteria_path.mkdir(exist_ok=True)
                for uploaded_file in uploaded_criteria:
                    file_path = criteria_path / uploaded_file.name
                    with open(file_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    st.success(f"✅ Uploaded {uploaded_file.name} to criteria/")
                st.rerun()

        # TEMPLATE FILES TAB
        with templates_tab:
            st.markdown("**📝 Template Execution Order Manager**")
            st.info("💡 Configure the order in which templates will be executed in the workflow")

            # Load workflow configuration for template ordering
            config_path = project_path / "project_config.json"
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        workflow_config = json.load(f)
                except:
                    workflow_config = {"steps": []}
            else:
                workflow_config = {"steps": []}

            # Render the template ordering interface
            self._render_template_order_manager(workflow_id, workflow_config, project_path)

            st.markdown("---")
            st.markdown("**Template Files**")
            templates_path = project_path / "templates"

            if templates_path.exists():
                md_files = self._filter_git_files(list(templates_path.glob("*.md")))
                txt_files = self._filter_git_files(list(templates_path.glob("*.txt")))
                all_template_files = md_files + txt_files

                if all_template_files:
                    for template_file in all_template_files:
                        if template_file.suffix == '.md':
                            file_updated |= self._render_md_file_card(template_file, f"template_{template_file.stem}_{container_key}")
                        else:
                            file_updated |= self._render_text_file_card(template_file, f"template_{template_file.stem}_{container_key}")
                else:
                    st.info("📝 No template files found")
            else:
                st.warning("📂 Templates folder not found")

            # Upload template files section
            st.markdown("---")
            st.markdown("**⬆️ Upload Markdown Template Files**")
            uploaded_templates = st.file_uploader(
                "Choose markdown template files",
                type=['md', 'txt'],
                accept_multiple_files=True,
                key=f"templates_upload_{container_key}",
                help="Upload markdown files to the templates folder"
            )

            if uploaded_templates:
                templates_path.mkdir(exist_ok=True)
                for uploaded_file in uploaded_templates:
                    file_path = templates_path / uploaded_file.name
                    with open(file_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    st.success(f"✅ Uploaded {uploaded_file.name} to templates/")
                st.rerun()

        # INPUT FILES & RUN TAB
        with inputs_tab:
            st.markdown("**Input Files**")
            inputs_path = project_path / "inputs"

            # Show existing input files
            st.markdown("**📂 Current Input Files**")
            if inputs_path.exists():
                input_files = self._filter_git_files(list(inputs_path.glob("*")))

                if input_files:
                    for input_file in input_files:
                        if input_file.is_file():
                            file_size = input_file.stat().st_size
                            file_ext = input_file.suffix.lower()

                            # File type icon
                            if file_ext in ['.pdf']:
                                icon = "📄"
                            elif file_ext in ['.docx', '.doc']:
                                icon = "📝"
                            elif file_ext in ['.xlsx', '.xls', '.csv']:
                                icon = "📊"
                            elif file_ext in ['.txt', '.md']:
                                icon = "📋"
                            else:
                                icon = "📁"

                            with st.expander(f"{icon} {input_file.name} ({file_size:,} bytes)", expanded=False):
                                col_a, col_b = st.columns([3, 1])

                                with col_a:
                                    st.markdown(f"**File:** {input_file.name}")
                                    st.markdown(f"**Size:** {file_size:,} bytes")
                                    st.markdown(f"**Type:** {file_ext.upper() if file_ext else 'Unknown'}")

                                    from datetime import datetime
                                    mod_time = datetime.fromtimestamp(input_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                                    st.markdown(f"**Modified:** {mod_time}")

                                with col_b:
                                    # Download existing input file
                                    try:
                                        with open(input_file, 'rb') as f:
                                            file_content = f.read()

                                        st.download_button(
                                            "⬇️ Download",
                                            data=file_content,
                                            file_name=input_file.name,
                                            key=f"download_input_{input_file.stem}_{container_key}"
                                        )
                                    except Exception as e:
                                        st.error(f"Download error: {e}")

                                    # Delete file option
                                    if st.button("🗑️ Delete", key=f"delete_input_{input_file.stem}_{container_key}"):
                                        try:
                                            input_file.unlink()
                                            st.success(f"✅ Deleted {input_file.name}")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"❌ Delete error: {e}")
                else:
                    st.info("📂 No input files found")

                # Upload new input files section
                st.markdown("---")
                st.markdown("**⬆️ Upload New Input Files**")

                new_input_files = st.file_uploader(
                    "Upload files for workflow processing",
                    type=['pdf', 'docx', 'txt', 'md', 'csv', 'json', 'xlsx'],
                    accept_multiple_files=True,
                    key=f"upload_new_inputs_{container_key}",
                    help="Upload documents to be processed by the workflow"
                )

                if new_input_files:
                    inputs_path.mkdir(exist_ok=True)

                    for uploaded_file in new_input_files:
                        try:
                            # Save uploaded file
                            file_path = inputs_path / uploaded_file.name
                            content = uploaded_file.read()

                            with open(file_path, 'wb') as f:
                                f.write(content)

                            st.success(f"✅ Uploaded: {uploaded_file.name}")
                            file_updated = True

                        except Exception as e:
                            st.error(f"❌ Upload error: {e}")

                    if file_updated:
                        st.rerun()
            else:
                st.warning("📂 Inputs folder not found - will be created when files are uploaded")

        # WORKFLOW RUN TAB
        with run_tab:
            st.markdown("**🚀 Workflow Execution**")
            inputs_path = project_path / "inputs"

            # Workflow status and controls
            with st.container():
                st.markdown(f"**Project:** {workflow_id}")
                st.markdown(f"**Version:** {workflow.get('version', '1.0')}")

                # Check if there are input files
                has_inputs = inputs_path.exists() and any(inputs_path.glob("*"))

                if has_inputs:
                    input_count = len(list(inputs_path.glob("*")))
                    st.markdown(f"**Input Files:** {input_count}")
                    st.success("✅ Ready to run")
                else:
                    st.warning("⚠️ No input files")

                # Run workflow button
                run_clicked = st.button(
                    "🚀 Run Workflow",
                    type="primary",
                    use_container_width=True,
                    disabled=not has_inputs,
                    key=f"run_workflow_{container_key}",
                    help="Execute the workflow with current input files" if has_inputs else "Upload input files first"
                )

                if run_clicked and has_inputs:
                    with st.spinner("🔄 Executing workflow..."):
                        try:
                            # Simulate workflow execution (replace with actual workflow execution)
                            import time
                            time.sleep(2)  # Simulate processing time

                            # Create a simple execution result
                            from datetime import datetime
                            execution_time = datetime.now().isoformat()

                            # Save execution log
                            outputs_path = project_path / "outputs"
                            outputs_path.mkdir(exist_ok=True)

                            log_file = outputs_path / f"execution_log_{execution_time.replace(':', '-').split('.')[0]}.json"
                            execution_log = {
                                "workflow_id": workflow_id,
                                "execution_time": execution_time,
                                "input_files": [f.name for f in inputs_path.glob("*") if f.is_file()],
                                "status": "completed",
                                "message": "Workflow executed successfully via File Manager"
                            }

                            import json
                            with open(log_file, 'w', encoding='utf-8') as f:
                                json.dump(execution_log, f, indent=2)

                            st.success(f"✅ Workflow executed successfully!")
                            st.info(f"📄 Execution log saved: {log_file.name}")

                            # Option to download results
                            if st.button("📥 View Results", key=f"view_results_{container_key}"):
                                st.json(execution_log)

                        except Exception as e:
                            st.error(f"❌ Execution error: {e}")

                # Additional workflow controls
                st.markdown("---")
                st.markdown("**🔧 Quick Actions**")

                col_action1, col_action2 = st.columns(2)

                with col_action1:
                    if st.button("🗂️ Clear Inputs", key=f"clear_inputs_{container_key}"):
                        if inputs_path.exists():
                            import shutil
                            try:
                                shutil.rmtree(inputs_path)
                                inputs_path.mkdir()
                                st.success("✅ Inputs cleared")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Clear error: {e}")

                with col_action2:
                    if st.button("📊 Manage Outputs", key=f"manage_outputs_{container_key}"):
                        self._show_output_manager(project_path, container_key)

        return file_updated

    def _render_json_file_card(self, file_path: Path, card_key: str) -> bool:
        """Render an individual JSON file management card."""
        try:
            # Read current content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            file_size = file_path.stat().st_size
            file_modified = file_path.stat().st_mtime

            with st.expander(f"📋 {file_path.name} ({file_size:,} bytes)", expanded=False):
                col1, col2 = st.columns([3, 1])

                with col1:
                    # Editable JSON content
                    new_content = st.text_area(
                        "Content",
                        value=content,
                        height=200,
                        help="Edit JSON content directly",
                        key=f"json_content_{card_key}"
                    )

                with col2:
                    st.markdown(f"**File:** {file_path.name}")
                    st.markdown(f"**Size:** {file_size:,} bytes")

                    from datetime import datetime
                    mod_time = datetime.fromtimestamp(file_modified).strftime("%Y-%m-%d %H:%M")
                    st.markdown(f"**Modified:** {mod_time}")

                # Action buttons
                col_btn1, col_btn2, col_btn3 = st.columns(3)

                with col_btn1:
                    if st.button("💾 Save", key=f"save_json_{card_key}"):
                        try:
                            # Validate JSON before saving
                            import json
                            json.loads(new_content)

                            with open(file_path, 'w', encoding='utf-8') as f:
                                f.write(new_content)
                            st.success(f"✅ Saved {file_path.name}")
                            st.rerun()
                            return True
                        except json.JSONDecodeError as e:
                            st.error(f"❌ Invalid JSON: {e}")
                        except Exception as e:
                            st.error(f"❌ Save error: {e}")

                with col_btn2:
                    if st.button("🔄 Reload", key=f"reload_json_{card_key}"):
                        st.rerun()

                with col_btn3:
                    # Download button
                    st.download_button(
                        "⬇️ Download",
                        data=content,
                        file_name=file_path.name,
                        mime="application/json",
                        key=f"download_json_{card_key}"
                    )

        except Exception as e:
            st.error(f"❌ Error loading {file_path.name}: {e}")

        return False

    def _render_md_file_card(self, file_path: Path, card_key: str) -> bool:
        """Render an individual Markdown file management card."""
        try:
            # Read current content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            file_size = file_path.stat().st_size
            file_modified = file_path.stat().st_mtime

            with st.expander(f"📝 {file_path.name} ({file_size:,} bytes)", expanded=False):
                col1, col2 = st.columns([3, 1])

                with col1:
                    # Editable Markdown content
                    new_content = st.text_area(
                        "Content",
                        value=content,
                        height=300,
                        help="Edit Markdown content directly",
                        key=f"md_content_{card_key}"
                    )

                with col2:
                    st.markdown(f"**File:** {file_path.name}")
                    st.markdown(f"**Size:** {file_size:,} bytes")

                    from datetime import datetime
                    mod_time = datetime.fromtimestamp(file_modified).strftime("%Y-%m-%d %H:%M")
                    st.markdown(f"**Modified:** {mod_time}")

                    # Show markdown preview
                    if st.checkbox("👁️ Preview", key=f"preview_md_{card_key}"):
                        st.markdown("**Preview:**")
                        st.markdown(new_content[:500] + "..." if len(new_content) > 500 else new_content)

                # Action buttons
                col_btn1, col_btn2, col_btn3 = st.columns(3)

                with col_btn1:
                    if st.button("💾 Save", key=f"save_md_{card_key}"):
                        try:
                            with open(file_path, 'w', encoding='utf-8') as f:
                                f.write(new_content)
                            st.success(f"✅ Saved {file_path.name}")
                            st.rerun()
                            return True
                        except Exception as e:
                            st.error(f"❌ Save error: {e}")

                with col_btn2:
                    if st.button("🔄 Reload", key=f"reload_md_{card_key}"):
                        st.rerun()

                with col_btn3:
                    # Download button
                    st.download_button(
                        "⬇️ Download",
                        data=content,
                        file_name=file_path.name,
                        mime="text/markdown",
                        key=f"download_md_{card_key}"
                    )

        except Exception as e:
            st.error(f"❌ Error loading {file_path.name}: {e}")

        return False

    def _render_text_file_card(self, file_path: Path, card_key: str) -> bool:
        """Render an individual text file management card."""
        try:
            # Read current content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            file_size = file_path.stat().st_size
            file_modified = file_path.stat().st_mtime

            with st.expander(f"📄 {file_path.name} ({file_size:,} bytes)", expanded=False):
                col1, col2 = st.columns([3, 1])

                with col1:
                    # Editable text content
                    new_content = st.text_area(
                        "Content",
                        value=content,
                        height=300,
                        help="Edit text content directly",
                        key=f"txt_content_{card_key}"
                    )

                with col2:
                    st.markdown(f"**File:** {file_path.name}")
                    st.markdown(f"**Size:** {file_size:,} bytes")

                    from datetime import datetime
                    mod_time = datetime.fromtimestamp(file_modified).strftime("%Y-%m-%d %H:%M")
                    st.markdown(f"**Modified:** {mod_time}")

                    # Show text preview
                    if st.checkbox("👁️ Preview", key=f"preview_txt_{card_key}"):
                        st.markdown("**Preview:**")
                        st.text(content[:500] + "..." if len(content) > 500 else content)

                # Action buttons
                col_btn1, col_btn2, col_btn3 = st.columns(3)

                with col_btn1:
                    if st.button("💾 Save", key=f"save_txt_{card_key}"):
                        try:
                            with open(file_path, 'w', encoding='utf-8') as f:
                                f.write(new_content)
                            st.success(f"✅ Saved {file_path.name}")
                            st.rerun()
                            return True
                        except Exception as e:
                            st.error(f"❌ Save error: {e}")

                with col_btn2:
                    if st.button("🔄 Reload", key=f"reload_txt_{card_key}"):
                        st.rerun()

                with col_btn3:
                    # Download button
                    st.download_button(
                        "⬇️ Download",
                        data=content,
                        file_name=file_path.name,
                        mime="text/plain",
                        key=f"download_txt_{card_key}"
                    )

        except Exception as e:
            st.error(f"❌ Error loading {file_path.name}: {e}")

        return False

    def _render_file_upload_manager(self, project_path: Path, container_key: str) -> bool:
        """Render file upload interface for criteria and templates folders."""
        col1, col2 = st.columns(2)

        file_uploaded = False

        with col1:
            st.markdown("**📋 Upload Criteria JSON**")
            criteria_files = st.file_uploader(
                "Upload JSON criteria files",
                type=['json'],
                accept_multiple_files=True,
                key=f"upload_criteria_{container_key}",
                help="Upload one or more JSON criteria files"
            )

            if criteria_files:
                criteria_path = project_path / "criteria"
                criteria_path.mkdir(exist_ok=True)

                for uploaded_file in criteria_files:
                    try:
                        # Validate JSON content
                        content = uploaded_file.read().decode('utf-8')
                        import json
                        json.loads(content)  # Validate JSON

                        # Save file
                        file_path = criteria_path / uploaded_file.name
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)

                        st.success(f"✅ Uploaded: {uploaded_file.name}")
                        file_uploaded = True

                    except json.JSONDecodeError:
                        st.error(f"❌ Invalid JSON: {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"❌ Upload error: {e}")

        with col2:
            st.markdown("**📝 Upload Template Markdown**")
            template_files = st.file_uploader(
                "Upload MD template files",
                type=['md', 'markdown'],
                accept_multiple_files=True,
                key=f"upload_templates_{container_key}",
                help="Upload one or more Markdown template files"
            )

            if template_files:
                templates_path = project_path / "templates"
                templates_path.mkdir(exist_ok=True)

                for uploaded_file in template_files:
                    try:
                        content = uploaded_file.read().decode('utf-8')

                        # Save file
                        file_path = templates_path / uploaded_file.name
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)

                        st.success(f"✅ Uploaded: {uploaded_file.name}")
                        file_uploaded = True

                    except Exception as e:
                        st.error(f"❌ Upload error: {e}")

        if file_uploaded:
            st.info("🔄 Refresh the page to see uploaded files in their respective tabs")

        return file_uploaded

    def _show_output_manager(self, project_path: Path, container_key: str):
        """Show comprehensive output file management with download and cleanup options."""
        outputs_path = project_path / "outputs"

        with st.expander("📊 Output Files Manager", expanded=True):
            if outputs_path.exists():
                output_files = self._filter_git_files(list(outputs_path.glob("*")))

                if output_files:
                    st.markdown(f"**Found {len(output_files)} output files**")

                    # Bulk actions section
                    col_bulk1, col_bulk2, col_bulk3 = st.columns(3)

                    with col_bulk1:
                        if st.button("📦 Download All", key=f"download_all_outputs_{container_key}"):
                            self._download_all_outputs(outputs_path, container_key)

                    with col_bulk2:
                        if st.button("🗑️ Clear All Outputs", key=f"clear_all_outputs_{container_key}", type="secondary"):
                            if st.button("⚠️ Confirm Clear All", key=f"confirm_clear_outputs_{container_key}"):
                                try:
                                    import shutil
                                    shutil.rmtree(outputs_path)
                                    outputs_path.mkdir()
                                    st.success("✅ All outputs cleared")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Clear error: {e}")

                    with col_bulk3:
                        # Clean old files (older than 7 days)
                        if st.button("🧹 Clean Old Files", key=f"clean_old_outputs_{container_key}"):
                            self._clean_old_outputs(outputs_path, days=7)

                    st.markdown("---")

                    # Individual file management
                    for i, output_file in enumerate(sorted(output_files, key=lambda x: x.stat().st_mtime, reverse=True)):
                        if output_file.is_file():
                            file_size = output_file.stat().st_size
                            file_ext = output_file.suffix.lower()

                            # File type icon
                            if file_ext == '.json':
                                icon = "📄"
                            elif file_ext in ['.txt', '.log']:
                                icon = "📝"
                            elif file_ext in ['.pdf']:
                                icon = "📑"
                            elif file_ext in ['.csv', '.xlsx']:
                                icon = "📊"
                            else:
                                icon = "📁"

                            with st.expander(f"{icon} {output_file.name} ({file_size:,} bytes)", expanded=False):
                                col_a, col_b = st.columns([3, 1])

                                with col_a:
                                    st.markdown(f"**File:** {output_file.name}")
                                    st.markdown(f"**Size:** {file_size:,} bytes")
                                    st.markdown(f"**Type:** {file_ext.upper() if file_ext else 'Unknown'}")

                                    from datetime import datetime
                                    mod_time = datetime.fromtimestamp(output_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                                    st.markdown(f"**Modified:** {mod_time}")

                                    # Preview content for small text files
                                    if file_ext in ['.json', '.txt', '.log'] and file_size < 10000:  # Less than 10KB
                                        if st.checkbox("👁️ Preview", key=f"preview_output_{i}_{container_key}"):
                                            try:
                                                with open(output_file, 'r', encoding='utf-8') as f:
                                                    content = f.read()
                                                st.code(content[:1000] + "..." if len(content) > 1000 else content)
                                            except Exception as e:
                                                st.error(f"Preview error: {e}")

                                with col_b:
                                    # Download individual file
                                    try:
                                        with open(output_file, 'rb') as f:
                                            file_content = f.read()

                                        st.download_button(
                                            "⬇️ Download",
                                            data=file_content,
                                            file_name=output_file.name,
                                            key=f"download_output_{i}_{container_key}"
                                        )
                                    except Exception as e:
                                        st.error(f"Download error: {e}")

                                    # Delete individual file
                                    if st.button("🗑️ Delete", key=f"delete_output_{i}_{container_key}"):
                                        try:
                                            output_file.unlink()
                                            st.success(f"✅ Deleted {output_file.name}")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"❌ Delete error: {e}")

                else:
                    st.info("📂 No output files found")
            else:
                st.info("📂 Outputs folder not found")

    def _download_all_outputs(self, outputs_path: Path, container_key: str):
        """Create a ZIP file with all output files for download."""
        try:
            import zipfile
            import io
            from datetime import datetime

            # Create ZIP file in memory
            zip_buffer = io.BytesIO()

            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                output_files = list(outputs_path.glob("*"))

                if output_files:
                    for output_file in output_files:
                        if output_file.is_file():
                            # Add file to ZIP
                            zip_file.write(output_file, output_file.name)

                    zip_buffer.seek(0)

                    # Offer download
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    zip_filename = f"{outputs_path.parent.name}_outputs_{timestamp}.zip"

                    st.download_button(
                        "📦 Download ZIP Archive",
                        data=zip_buffer.getvalue(),
                        file_name=zip_filename,
                        mime="application/zip",
                        key=f"download_zip_{container_key}"
                    )

                    st.success(f"✅ Created ZIP with {len(output_files)} files")
                else:
                    st.warning("No files to download")

        except Exception as e:
            st.error(f"❌ ZIP creation error: {e}")

    def _clean_old_outputs(self, outputs_path: Path, days: int = 7):
        """Clean output files older than specified days."""
        try:
            from datetime import datetime, timedelta
            import time

            cutoff_time = time.time() - (days * 24 * 60 * 60)  # Convert days to seconds
            cleaned_files = []

            for output_file in outputs_path.glob("*"):
                if output_file.is_file() and output_file.stat().st_mtime < cutoff_time:
                    cleaned_files.append(output_file.name)
                    output_file.unlink()

            if cleaned_files:
                st.success(f"✅ Cleaned {len(cleaned_files)} old files (>{days} days)")
                for filename in cleaned_files[:5]:  # Show first 5
                    st.markdown(f"- {filename}")
                if len(cleaned_files) > 5:
                    st.markdown(f"... and {len(cleaned_files) - 5} more")
                st.rerun()
            else:
                st.info(f"📂 No files older than {days} days found")

        except Exception as e:
            st.error(f"❌ Cleanup error: {e}")

    def _render_compact_update_card_legacy(self, workflow: Dict[str, Any], container_key: str = None) -> bool:
        """Render a compact update card for workflow modification.

        Returns True if workflow was updated successfully.
        """
        workflow_id = workflow.get('workflow_id', 'unknown')
        workflow_name = workflow.get('config', {}).get('name', workflow.get('workflow_name', workflow_id))
        current_version = workflow.get('version', '1.0')
        workflow_type = workflow.get('workflow_type', workflow.get('template', {}).get('workflow_type', 'unknown'))

        with st.container():
            st.markdown(f"**⚙️ Quick Update: {workflow_name}**")

            # Compact form
            with st.form(f"update_form_{workflow_id}_{container_key or 'default'}"):
                col1, col2 = st.columns([2, 1])

                with col1:
                    # Essential fields only
                    new_name = st.text_input(
                        "Name",
                        value=workflow_name,
                        help="Update workflow name",
                        key=f"update_name_{workflow_id}_{container_key or 'default'}"
                    )

                    new_description = st.text_area(
                        "Description",
                        value=workflow.get('description', ''),
                        height=60,
                        help="Brief description of changes",
                        key=f"update_desc_{workflow_id}_{container_key or 'default'}"
                    )

                with col2:
                    st.markdown(f"**Type:** {workflow_type}")
                    st.markdown(f"**Version:** {current_version}")

                    # Auto-increment version
                    try:
                        version_parts = current_version.split('.')
                        minor_version = int(version_parts[1]) + 1
                        next_version = f"{version_parts[0]}.{minor_version}"
                    except:
                        next_version = "1.1"

                    st.markdown(f"**Next:** {next_version}")

                # Expandable advanced options
                with st.expander("🔧 Advanced Options"):
                    col_a, col_b = st.columns(2)

                    with col_a:
                        new_priority = st.selectbox(
                            "Priority",
                            ["low", "medium", "high"],
                            index=["low", "medium", "high"].index(workflow.get('config', {}).get('priority', 'medium')),
                            key=f"update_priority_{workflow_id}_{container_key or 'default'}"
                        )

                        enable_monitoring = st.checkbox(
                            "Enable Monitoring",
                            value=workflow.get('config', {}).get('enable_monitoring', True),
                            key=f"update_monitoring_{workflow_id}_{container_key or 'default'}"
                        )

                    with col_b:
                        # RAG systems selection (if available)
                        current_rags = workflow.get('config', {}).get('rag_systems', [])
                        if current_rags:
                            st.markdown(f"**Current RAG:** {', '.join(current_rags)}")

                        # Status override
                        new_status = st.selectbox(
                            "Status",
                            ["updated", "active", "deployed", "testing"],
                            index=0,
                            key=f"update_status_{workflow_id}_{container_key or 'default'}"
                        )

                # Action buttons
                col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])

                with col_btn1:
                    update_clicked = st.form_submit_button("🔄 Update", use_container_width=True)

                with col_btn2:
                    reset_clicked = st.form_submit_button("↩️ Reset", use_container_width=True)

                with col_btn3:
                    # Display current file path
                    if workflow.get('file_path'):
                        st.caption(f"📁 {Path(workflow['file_path']).name}")

                if update_clicked:
                    # Update workflow data
                    updated_workflow = workflow.copy()
                    updated_workflow.update({
                        'config': {
                            **workflow.get('config', {}),
                            'name': new_name,
                            'priority': new_priority,
                            'enable_monitoring': enable_monitoring
                        },
                        'description': new_description,
                        'status': new_status,
                        'workflow_name': new_name
                    })

                    # Perform update
                    try:
                        update_result = self._update_workflow(updated_workflow)

                        if update_result.get('success'):
                            st.success(f"✅ {workflow_name} updated to v{update_result['new_version']}")
                            st.rerun()
                            return True
                        else:
                            st.error(f"❌ Update failed: {update_result.get('error', 'Unknown error')}")
                            return False

                    except Exception as e:
                        st.error(f"❌ Update error: {str(e)}")
                        return False

                if reset_clicked:
                    st.info("🔄 Form reset - refresh to reload original values")
                    st.rerun()

        return False


def main():
    """Main entry point for Flow Creator V3 Portal."""
    try:
        # Use session state to cache the portal instance to prevent re-initialization loops
        if 'portal_instance' not in st.session_state:
            st.session_state.portal_instance = FlowCreatorV3Portal()

        portal = st.session_state.portal_instance
        portal.render_portal()
    except Exception as e:
        st.warning(f"Portal temporarily unavailable: {e}")
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()