# mock_demo_c.py
import streamlit as st
from packaging import version
import logging

# --- Streamlit Version Check ---
if version.parse(st.__version__) < version.parse("1.25.0"):
    st.error("Streamlit version too old. Please upgrade to 1.25.0 or higher.")
    st.stop()

# --- Import Config & Client Helpers from admin_demo ---
from core.config_helper import get_config
from core.client_manager import get_clients

# --- Import Live Admin Tabs ---
from tabs import (
    a_config_tab,
    a_logs_tab,
    a_faiss_diagnostics_tab,
    a_admin_cleanup_tab,
    a_pdf_tab,
    a_pgvector_index_tab
)

# --- Load Config with Caching ---
@st.cache_resource
def load_config():
    return get_config()

# --- Config and Clients Setup ---
if "config" not in st.session_state:
    st.session_state["config"] = load_config()
clients = get_clients()

# --- Dummy User Tab Functions (unchanged) ---
class WizardTab:
    @staticmethod
    def render():
        st.header("Wizard Tab")
        st.write("Populate Wizard Tab here.")

class PlanAuditTab:
    @staticmethod
    def render():
        st.header("Plan Audit Tab")
        st.write("Populate Plan Audit Tab here.")

class FaissSearchTab:
    @staticmethod
    def render():
        st.header("FAISS Search Tab")
        st.write("Populate FAISS Search Tab here.")

class GuidedSearchTab:
    @staticmethod
    def render():
        st.header("Guided Search Tab")
        st.write("Populate Guided Search Tab here.")

class DashboardTab:
    @staticmethod
    def render():
        st.header("Dashboard Tab")
        st.write("Populate Dashboard Tab here.")

class GroundtruthTab:
    @staticmethod
    def render():
        st.header("Groundtruth Tab")
        st.write("Populate Groundtruth Tab here.")

class Option2AuditTab:
    @staticmethod
    def render(vector_backend="pgvector"):
        st.header("Option 2: Plan Audit Tab")
        st.write(f"Populate Option 2 Plan Audit Tab here using vector backend: {vector_backend}.")

class Option2TruthTab:
    @staticmethod
    def render():
        st.header("Option 2: Groundtruth Tab")
        st.write("Populate Option 2 Groundtruth Tab here.")

class Option2DashboardTab:
    @staticmethod
    def render():
        st.header("Option 2: Dashboard Tab")
        st.write("Populate Option 2 Dashboard Tab here.")

class Option3AuditTab:
    @staticmethod
    def render():
        st.header("Option 3: Agent Audit Tab")
        st.write("Populate Option 3 Agent Audit Tab here.")

class Option3SynthTab:
    @staticmethod
    def render():
        st.header("Option 3: Synth Tab")
        st.write("Populate Option 3 Synth Tab here.")


# --- Navigation Controls ---
if st.sidebar.button("🏠 Home"):
    st.session_state["admin_tool"] = "None"
    st.experimental_rerun()

# --- Sidebar: Admin Panel ---
st.sidebar.header(" Admin Panel")
admin_tool = st.sidebar.selectbox(
    "Select Admin Tool",
    ["None", "Config", "Logs Viewer", "FAISS Diagnostics", "Cleanup Tab", "PDF Extract", "pgVector Index"],
    key="admin_tool"
)

if admin_tool != "None":
    # --- Live Admin Tabs from admin_demo ---
    if admin_tool == "Config":
        a_config_tab.render()
    elif admin_tool == "Logs Viewer":
        a_logs_tab.render()
    elif admin_tool == "FAISS Diagnostics":
        a_faiss_diagnostics_tab.show_faiss_diagnostics()
    elif admin_tool == "Cleanup Tab":
        a_admin_cleanup_tab.render()
    elif admin_tool == "PDF Extract":
        a_pdf_tab.render()
    elif admin_tool == "pgVector Index":
        a_pgvector_index_tab.render()
else:
    # --- DSPy Workflow Selector (User Mode) ---
    st.sidebar.markdown("### DSPy Workflow Mode")
    selected_dspy_mode = st.sidebar.radio(
        "Choose Workflow:",
        [
            "Option 1: FAISS (Simple)",
            "Option 2: DSPy + pgVector",
            "Option 3: Multi-Agent DSPy"
        ],
        key="dspy_workflow_mode"
    )

    if selected_dspy_mode == "Option 1: FAISS (Simple)":
        tabs = st.tabs(["🧠 Guided Search", "📍 FAISS Search"])
        with tabs[0]:
            GuidedSearchTab.render()
        with tabs[1]:
            FaissSearchTab.render()

    elif selected_dspy_mode == "Option 2: DSPy + pgVector":
        tabs = st.tabs(["📋 Plan Audit", "🧠 Groundtruth", "📊 Dashboard"])
        with tabs[0]:
            Option2AuditTab.render(vector_backend="pgvector")
        with tabs[1]:
            Option2TruthTab.render()
        with tabs[2]:
            Option2DashboardTab.render()

    elif selected_dspy_mode == "Option 3: Multi-Agent DSPy":
        tabs = st.tabs(["📋 Agent Audit", "🧠 Synth", "📊 Dashboard", "🤖 Router"])
        with tabs[0]:
            Option3AuditTab.render()
        with tabs[1]:
            Option3SynthTab.render()
        with tabs[2]:
            Option2DashboardTab.render()  # Reusing Option2 Dashboard Tab for now
        with tabs[3]:
            st.info("Coming soon: multi-agent DSPy orchestration.")
