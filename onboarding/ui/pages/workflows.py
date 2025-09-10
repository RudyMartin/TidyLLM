"""
TidyLLM Onboarding Workflows Page
=================================

Workflow configuration and YAML registry management page.
"""

import streamlit as st
import yaml

def render_workflows_page():
    """Render the workflows page."""
    
    st.markdown('<div class="section-header">⚙️ Workflows - YAML Registry Management</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Configure and manage TidyLLM workflows:
    - **Registry Conversion**: Convert Python workflows to editable YAML
    - **Ad-hoc AI Managers**: Create custom AI managers with dual RAG support
    - **YAML Editor**: Live editing with validation and save functionality
    - **Dual RAG System**: Domain knowledge + work history RAGs per manager
    """)
    
    # Tab interface
    tab1, tab2, tab3 = st.tabs(["📋 Workflow Registry", "🤖 AI Managers", "📝 YAML Editor"])
    
    with tab1:
        render_workflow_registry()
    
    with tab2:
        render_ai_managers()
    
    with tab3:
        render_yaml_editor()

def render_workflow_registry():
    """Render workflow registry interface."""
    st.subheader("Workflow Registry")
    
    # Mock workflow data
    workflows = [
        {
            "name": "QA Processing",
            "description": "Quality assurance document processing workflow",
            "type": "document_processing",
            "status": "active",
            "last_run": "2025-01-09 10:30:00"
        },
        {
            "name": "MVR Analysis",
            "description": "Motor vehicle record analysis workflow",
            "type": "compliance",
            "status": "active",
            "last_run": "2025-01-09 09:15:00"
        },
        {
            "name": "Contract Review",
            "description": "Legal contract review and analysis",
            "type": "legal",
            "status": "draft",
            "last_run": None
        }
    ]
    
    # Workflow list
    for workflow in workflows:
        with st.expander(f"⚙️ {workflow['name']} ({workflow['type']})"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**Description:** {workflow['description']}")
                st.write(f"**Status:** {workflow['status']}")
                if workflow['last_run']:
                    st.write(f"**Last Run:** {workflow['last_run']}")
                else:
                    st.write("**Last Run:** Never")
            
            with col2:
                if st.button(f"Edit", key=f"edit_{workflow['name']}"):
                    st.session_state.selected_workflow = workflow['name']
                    st.rerun()
                
                if st.button(f"Run", key=f"run_{workflow['name']}"):
                    st.info(f"Running workflow: {workflow['name']}")
    
    # Add new workflow
    st.subheader("Add New Workflow")
    
    with st.form("add_workflow_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            new_name = st.text_input("Workflow Name", placeholder="e.g., Invoice Processing")
            new_description = st.text_area("Description", placeholder="Describe the workflow purpose")
        
        with col2:
            new_type = st.selectbox("Workflow Type", ["document_processing", "compliance", "legal", "financial", "custom"])
            new_priority = st.selectbox("Priority", ["low", "medium", "high", "critical"])
        
        submitted = st.form_submit_button("Add Workflow")
        
        if submitted:
            if new_name and new_description:
                st.success(f"Workflow '{new_name}' added successfully!")
                # TODO: Implement actual workflow creation
            else:
                st.error("Please fill in all required fields.")

def render_ai_managers():
    """Render AI managers interface."""
    st.subheader("AI Managers")
    
    # Mock AI manager data
    ai_managers = [
        {
            "name": "Document Processor",
            "description": "Handles document processing workflows",
            "rag_systems": ["domain_knowledge", "work_history"],
            "models": ["claude-3-sonnet", "gpt-4"],
            "status": "active"
        },
        {
            "name": "Compliance Checker",
            "description": "Ensures regulatory compliance",
            "rag_systems": ["policy_database", "regulation_history"],
            "models": ["claude-3-opus"],
            "status": "active"
        }
    ]
    
    # AI manager list
    for manager in ai_managers:
        with st.expander(f"🤖 {manager['name']}"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**Description:** {manager['description']}")
                st.write(f"**RAG Systems:** {', '.join(manager['rag_systems'])}")
                st.write(f"**Models:** {', '.join(manager['models'])}")
                st.write(f"**Status:** {manager['status']}")
            
            with col2:
                if st.button(f"Configure", key=f"config_{manager['name']}"):
                    st.session_state.selected_manager = manager['name']
                    st.rerun()
                
                if st.button(f"Test", key=f"test_{manager['name']}"):
                    st.info(f"Testing AI manager: {manager['name']}")
    
    # Create new AI manager
    st.subheader("Create New AI Manager")
    
    with st.form("create_manager_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            manager_name = st.text_input("Manager Name", placeholder="e.g., Legal Reviewer")
            manager_description = st.text_area("Description", placeholder="Describe the manager's purpose")
        
        with col2:
            rag_systems = st.multiselect(
                "RAG Systems",
                ["domain_knowledge", "work_history", "policy_database", "regulation_history", "custom"],
                default=["domain_knowledge"]
            )
            models = st.multiselect(
                "AI Models",
                ["claude-3-sonnet", "claude-3-opus", "gpt-4", "gpt-3.5-turbo"],
                default=["claude-3-sonnet"]
            )
        
        submitted = st.form_submit_button("Create AI Manager")
        
        if submitted:
            if manager_name and manager_description and rag_systems and models:
                st.success(f"AI Manager '{manager_name}' created successfully!")
                # TODO: Implement actual AI manager creation
            else:
                st.error("Please fill in all required fields.")

def render_yaml_editor():
    """Render YAML editor interface."""
    st.subheader("YAML Editor")
    
    # Sample YAML content
    sample_yaml = """# TidyLLM Workflow Configuration
workflow:
  name: "Document Processing"
  version: "1.0.0"
  description: "Process and analyze documents"
  
  steps:
    - name: "upload"
      type: "file_upload"
      config:
        allowed_types: ["pdf", "docx", "txt"]
        max_size: "10MB"
    
    - name: "extract"
      type: "text_extraction"
      config:
        engine: "tesseract"
        language: "en"
    
    - name: "analyze"
      type: "ai_analysis"
      config:
        model: "claude-3-sonnet"
        prompt: "Analyze this document and extract key information"
    
    - name: "store"
      type: "knowledge_store"
      config:
        domain: "processed_documents"
        vectorize: true

  ai_managers:
    - name: "document_processor"
      rag_systems:
        - "domain_knowledge"
        - "work_history"
      models:
        - "claude-3-sonnet"
        - "gpt-4"
"""
    
    # YAML editor
    yaml_content = st.text_area(
        "YAML Configuration",
        value=sample_yaml,
        height=400,
        help="Edit the YAML configuration for your workflow"
    )
    
    # Editor controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save Configuration", use_container_width=True):
            try:
                # Validate YAML
                yaml.safe_load(yaml_content)
                st.success("Configuration saved successfully!")
                # TODO: Implement actual save functionality
            except yaml.YAMLError as e:
                st.error(f"Invalid YAML: {e}")
    
    with col2:
        if st.button("✅ Validate YAML", use_container_width=True):
            try:
                parsed = yaml.safe_load(yaml_content)
                st.success("YAML is valid!")
                st.json(parsed)
            except yaml.YAMLError as e:
                st.error(f"YAML validation failed: {e}")
    
    with col3:
        if st.button("🔄 Reset to Default", use_container_width=True):
            st.rerun()
    
    # YAML preview
    st.subheader("YAML Preview")
    try:
        parsed_yaml = yaml.safe_load(yaml_content)
        st.json(parsed_yaml)
    except yaml.YAMLError as e:
        st.error(f"Invalid YAML: {e}")
