"""
TidyLLM Flow Creator V3 Portal - MODULAR VERSION
=================================================

Modular version with separated tab components for easier maintenance and focused editing.

Architecture: Portal → WorkflowManager → RAG Ecosystem Integration

MODULAR STRUCTURE:
- t_create_flow.py      - Create new workflows
- t_existing_flows.py   - Browse and manage existing workflows
- t_test_designer.py    - Enhanced 12th-grader friendly A/B/C/D testing
- t_ai_advisor.py       - AI chat interface for workflow advice

FUTURE TABS (Ready for activation):
- t_flow_designer.py    - Visual workflow builder (coming soon)
- t_test_runner.py      - Batch test execution (coming soon)
- t_workflow_monitor.py - Real-time monitoring (coming soon)
- t_health_dashboard.py - System health monitoring (coming soon)
"""

import streamlit as st
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import traceback
import time

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

# IMPORT MODULAR TAB COMPONENTS
try:
    import sys
    import os

    # Add current directory to path for imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    from t_create_flow import render_create_flow_tab
    from t_existing_flows import render_existing_flows_tab
    from t_test_designer import render_test_designer_tab  # Re-enabled
    from t_ai_advisor import render_ai_advisor_tab

    # Future tabs (ready for activation)
    from t_flow_designer import render_flow_designer_placeholder
    from t_test_runner import render_test_runner_placeholder
    from t_workflow_monitor import render_workflow_monitor_placeholder
    from t_health_dashboard import render_health_dashboard_placeholder

    MODULAR_TABS_AVAILABLE = True
except ImportError as e:
    st.error(f"Modular tab components not available: {e}")
    MODULAR_TABS_AVAILABLE = False

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
            config_file = project_dir / "project_config.json"
            has_criteria = (project_dir / "criteria").exists()
            has_templates = (project_dir / "templates").exists()

            # Skip projects without proper configuration
            if not config_file.exists() and not (has_criteria and has_templates):
                print(f"DEBUG: Skipping incomplete project: {project_dir.name}")
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
            "alex_qaqc": {
                "workflow_id": "alex_qaqc",
                "workflow_name": "Alex QAQC",
                "workflow_type": "analysis",
                "description": "Quality assurance and quality control workflow for data validation",
                "rag_integration": ["ai_powered", "intelligent"],
                "flow_encoding": "@alex_qaqc#process!analyze@output"
            }
        }

    def get_workflow_types(self) -> List[str]:
        """Get list of available workflow types."""
        if isinstance(self.workflows, dict):
            return list(set(w.get("workflow_type", "unknown") for w in self.workflows.values()))
        return ["unknown"]

    def get_workflows_by_type(self, workflow_type: str) -> Dict[str, Any]:
        """Get workflows filtered by type."""
        if isinstance(self.workflows, dict):
            return {
                wid: workflow for wid, workflow in self.workflows.items()
                if workflow.get("workflow_type") == workflow_type
            }
        return {}

class WorkflowManager:
    """Manages workflow creation, deployment, and integration with RAG ecosystem."""

    def __init__(self):
        # Use UnifiedFlowManager for all workflow operations
        self.flow_manager = UnifiedFlowManager(auto_load_credentials=True)
        self.registry = WorkflowRegistry()

    def get_available_rag_systems(self):
        """Get available RAG systems with defensive type handling."""
        try:
            # Get RAG systems from the flow manager
            rag_status = self.flow_manager.get_available_rag_systems()
            # Use DataNormalizer to ensure consistent format
            return DataNormalizer.normalize_to_status_dict(rag_status)
        except Exception as e:
            print(f"Warning: Could not get RAG systems: {e}")
            return {"ai_powered": True, "intelligent": False}

    def get_available_workflow_systems(self):
        """Get available workflow systems with defensive type handling."""
        try:
            workflow_status = self.flow_manager.get_available_workflow_systems()
            # Use DataNormalizer to ensure consistent format
            return DataNormalizer.normalize_to_status_dict(workflow_status)
        except Exception as e:
            print(f"Warning: Could not get workflow systems: {e}")
            return {"default": True}

    def get_workflow_health(self):
        """Get workflow system health status."""
        try:
            return self.flow_manager.get_system_health()
        except Exception as e:
            return {"success": False, "error": str(e)}

class FlowCreatorV3Portal:
    """
    MODULAR Flow Creator V3 Portal - Main Interface
    Uses separate tab files for easier maintenance and focused editing
    """

    def __init__(self):
        """Initialize the portal with required managers."""
        self.workflow_manager = WorkflowManager()
        self.registry = self.workflow_manager.registry

        # Initialize session state
        if 'portal_initialized' not in st.session_state:
            st.session_state.portal_initialized = True

    def render_portal(self):
        """Render the main portal interface with modular tabs."""

        # Portal header
        st.set_page_config(
            page_title="TidyLLM Flow Creator V3",
            page_icon="🧪",
            layout="wide"
        )

        st.title("🧪 TidyLLM Flow Creator V3")
        st.markdown("**Next-generation workflow creation and management portal** *(Modular Version)*")

        # System status indicators
        self._render_status_indicators()

        if not MODULAR_TABS_AVAILABLE:
            st.error("❌ **Modular tab components not available**. Please check the modular tab imports.")
            return

        # Main tabs - MODULAR VERSION
        tab1, tab2, tab3, tab4 = st.tabs([
            "➕ Create Flow",
            "📋 Existing Flows",
            "🧪 Test Designer",
            "🤖 AI Advisor"
        ])

        # ACTIVE TABS (using modular components)
        with tab1:
            render_create_flow_tab(self.workflow_manager, self.registry)

        with tab2:
            render_existing_flows_tab(self.workflow_manager, self.registry)

        with tab3:
            render_test_designer_tab(self.workflow_manager, self.registry)

        with tab4:
            render_ai_advisor_tab(self.workflow_manager, self.registry)

        # FUTURE TABS (placeholder functionality)
        # Uncomment these when ready to activate:
        #
        # tab5, tab6, tab7, tab8 = st.tabs([
        #     "🎨 Flow Designer",
        #     "🏃 Test Runner",
        #     "📊 Monitor",
        #     "🏥 Health Dashboard"
        # ])
        #
        # with tab5:
        #     render_flow_designer_placeholder()
        # with tab6:
        #     render_test_runner_placeholder()
        # with tab7:
        #     render_workflow_monitor_placeholder()
        # with tab8:
        #     render_health_dashboard_placeholder()

    def _render_status_indicators(self):
        """Render system status indicators."""
        col1, col2, col3 = st.columns(3)

        with col1:
            # RAG System Status
            rag_status = self.workflow_manager.get_available_rag_systems()
            rag_status = DataNormalizer.normalize_to_status_dict(rag_status)
            available_rags = sum(1 for available in rag_status.values() if available)
            total_rags = len(rag_status)

            if available_rags > 0:
                st.success(f"🧠 **RAG Systems**: {available_rags}/{total_rags} Available")
            else:
                st.warning("🧠 **RAG Systems**: None Available")

        with col2:
            # Workflow System Status
            workflow_status = self.workflow_manager.get_available_workflow_systems()
            workflow_status = DataNormalizer.normalize_to_status_dict(workflow_status)
            available_workflows = sum(1 for available in workflow_status.values() if available)
            total_workflows = len(workflow_status)

            if available_workflows > 0:
                st.success(f"⚙️ **Workflow Systems**: {available_workflows}/{total_workflows} Available")
            else:
                st.warning("⚙️ **Workflow Systems**: None Available")

        with col3:
            # Registry Status
            workflow_count = len(self.registry.workflows)
            if workflow_count > 0:
                st.success(f"📋 **Workflow Registry**: {workflow_count} Workflows")
            else:
                st.warning("📋 **Workflow Registry**: Empty")

def main():
    """Main application entry point."""
    try:
        if 'portal_instance' not in st.session_state:
            st.session_state.portal_instance = FlowCreatorV3Portal()

        portal = st.session_state.portal_instance
        portal.render_portal()
    except Exception as e:
        st.warning(f"Portal temporarily unavailable: {e}")
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()