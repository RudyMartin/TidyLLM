"""
Existing Flows Tab - Modular Component
Extracted from flow_creator_v3.py for easier maintenance
"""

import streamlit as st
from pathlib import Path
from typing import Dict, List, Any, Optional

# Helper functions defined first
def _render_criteria_tab():
    """Render the Criteria tab content."""
    st.markdown("### 📋 **JSON Criteria Files**")

    # Default to alex_qaqc project
    project_path = Path("C:/Users/marti/AI-Shipping/tidyllm/workflows/projects/alex_qaqc")
    criteria_path = project_path / "criteria"

    if criteria_path.exists():
        json_files = list(criteria_path.glob("*.json"))

        if json_files:
            for json_file in json_files:
                with st.expander(f"📄 {json_file.name}", expanded=False):
                    col_preview, col_actions = st.columns([3, 1])

                    with col_preview:
                        try:
                            import json
                            with open(json_file, 'r', encoding='utf-8') as f:
                                content = json.load(f)
                            st.json(content)
                        except Exception as e:
                            st.error(f"Error reading file: {e}")

                    with col_actions:
                        if st.button(f"🗑️ Delete", key=f"tab_del_criteria_{json_file.name}"):
                            json_file.unlink()
                            st.success(f"Deleted {json_file.name}")
                            st.rerun()

                        with open(json_file, 'rb') as f:
                            st.download_button(
                                "⬇️ Download",
                                f.read(),
                                file_name=json_file.name,
                                mime="application/json",
                                key=f"tab_download_criteria_{json_file.name}"
                            )
        else:
            st.info("📋 No criteria files found")
    else:
        st.warning("📂 Criteria folder not found")

def _render_templates_tab():
    """Render the Templates tab content."""
    st.markdown("### 📝 **Template Files**")

    # Default to alex_qaqc project
    project_path = Path("C:/Users/marti/AI-Shipping/tidyllm/workflows/projects/alex_qaqc")
    templates_path = project_path / "templates"

    if templates_path.exists():
        md_files = list(templates_path.glob("*.md"))
        txt_files = list(templates_path.glob("*.txt"))
        all_template_files = md_files + txt_files

        if all_template_files:
            for template_file in all_template_files:
                with st.expander(f"📄 {template_file.name}", expanded=False):
                    col_preview, col_actions = st.columns([3, 1])

                    with col_preview:
                        try:
                            with open(template_file, 'r', encoding='utf-8') as f:
                                content = f.read()
                            if template_file.suffix == '.md':
                                st.markdown(content)
                            else:
                                st.text(content)
                        except Exception as e:
                            st.error(f"Error reading file: {e}")

                    with col_actions:
                        if st.button(f"🗑️ Delete", key=f"tab_del_template_{template_file.name}"):
                            template_file.unlink()
                            st.success(f"Deleted {template_file.name}")
                            st.rerun()

                        with open(template_file, 'rb') as f:
                            st.download_button(
                                "⬇️ Download",
                                f.read(),
                                file_name=template_file.name,
                                mime="text/plain",
                                key=f"tab_download_template_{template_file.name}"
                            )
        else:
            st.info("📝 No template files found")
    else:
        st.warning("📂 Templates folder not found")

def _render_inputs_tab():
    """Render the Inputs tab content."""
    st.markdown("### 📂 **Input Files**")

    # Default to alex_qaqc project
    project_path = Path("C:/Users/marti/AI-Shipping/tidyllm/workflows/projects/alex_qaqc")
    inputs_path = project_path / "inputs"

    if inputs_path.exists():
        input_files = list(inputs_path.glob("*"))
        input_files = [f for f in input_files if f.is_file()]

        if input_files:
            for input_file in input_files:
                with st.expander(f"📄 {input_file.name}", expanded=False):
                    col_info, col_actions = st.columns([3, 1])

                    with col_info:
                        st.write(f"**Size**: {input_file.stat().st_size} bytes")
                        st.write(f"**Type**: {input_file.suffix}")

                        # Show preview for text files
                        if input_file.suffix in ['.txt', '.md', '.json', '.csv']:
                            try:
                                with open(input_file, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                if len(content) > 1000:
                                    st.text(content[:1000] + "...")
                                else:
                                    st.text(content)
                            except:
                                st.write("Binary file - preview not available")

                    with col_actions:
                        if st.button(f"🗑️ Delete", key=f"tab_del_input_{input_file.name}"):
                            input_file.unlink()
                            st.success(f"Deleted {input_file.name}")
                            st.rerun()

                        with open(input_file, 'rb') as f:
                            st.download_button(
                                "⬇️ Download",
                                f.read(),
                                file_name=input_file.name,
                                key=f"tab_download_input_{input_file.name}"
                            )
        else:
            st.info("📂 No input files found")
    else:
        st.warning("📂 Inputs folder not found")

def _render_outputs_tab():
    """Render the Outputs tab content."""
    st.markdown("### 📤 **Output Files**")

    # Default to alex_qaqc project
    project_path = Path("C:/Users/marti/AI-Shipping/tidyllm/workflows/projects/alex_qaqc")
    outputs_path = project_path / "outputs"

    if outputs_path.exists():
        output_files = list(outputs_path.glob("*"))
        output_files = [f for f in output_files if f.is_file()]

        if output_files:
            for output_file in output_files:
                with st.expander(f"📄 {output_file.name}", expanded=False):
                    col_info, col_actions = st.columns([3, 1])

                    with col_info:
                        st.write(f"**Size**: {output_file.stat().st_size} bytes")
                        st.write(f"**Modified**: {output_file.stat().st_mtime}")

                        # Show preview for text files
                        if output_file.suffix in ['.txt', '.md', '.json', '.csv']:
                            try:
                                with open(output_file, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                if len(content) > 1000:
                                    st.text(content[:1000] + "...")
                                else:
                                    st.text(content)
                            except:
                                st.write("Binary file - preview not available")

                    with col_actions:
                        if st.button(f"🗑️ Delete", key=f"tab_del_output_{output_file.name}"):
                            output_file.unlink()
                            st.success(f"Deleted {output_file.name}")
                            st.rerun()

                        with open(output_file, 'rb') as f:
                            st.download_button(
                                "⬇️ Download",
                                f.read(),
                                file_name=output_file.name,
                                key=f"tab_download_output_{output_file.name}"
                            )
        else:
            st.info("📤 No output files found")
    else:
        st.info("📂 Outputs folder will be created when workflows run")

def render_existing_flows_tab(workflow_manager, registry):
    """Render the Existing Flows tab content."""
    st.markdown("## 📋 **Existing Flows**")

    # Add the sub-navigation tabs right at the top level
    main_tab, criteria_tab, templates_tab, inputs_tab, outputs_tab = st.tabs([
        "📋 Workflows", "📋 Criteria", "📝 Templates", "📂 Inputs", "📤 Outputs"
    ])

    with main_tab:
        _render_workflows_list(workflow_manager, registry)

    with criteria_tab:
        _render_criteria_tab()

    with templates_tab:
        _render_templates_tab()

    with inputs_tab:
        _render_inputs_tab()

    with outputs_tab:
        _render_outputs_tab()

def _render_workflows_list(workflow_manager, registry):
    """Render the main workflows list."""
    # Get workflows from registry
    workflows = registry.workflows

    if not workflows:
        st.info("🆕 **No workflows found**. Create your first workflow in the '+ Create Flow' tab!")
        return

    # Display workflows
    st.markdown(f"### 📁 **Available Workflows** ({len(workflows)} total)")

    # Filter and search
    col_search, col_filter = st.columns([2, 1])

    with col_search:
        search_term = st.text_input("🔍 Search workflows", placeholder="Enter workflow name or description...")

    with col_filter:
        workflow_types = list(set(w.get("workflow_type", "unknown") for w in workflows.values()))
        selected_type = st.selectbox("Filter by type", ["All"] + workflow_types)

    # Filter workflows
    filtered_workflows = workflows
    if search_term:
        filtered_workflows = {
            wid: workflow for wid, workflow in workflows.items()
            if search_term.lower() in workflow.get("workflow_name", "").lower() or
               search_term.lower() in workflow.get("description", "").lower()
        }

    if selected_type != "All":
        filtered_workflows = {
            wid: workflow for wid, workflow in filtered_workflows.items()
            if workflow.get("workflow_type") == selected_type
        }

    if not filtered_workflows:
        st.warning("🔍 No workflows match your search criteria")
        return

    # Display workflows in cards
    for workflow_id, workflow in filtered_workflows.items():
        with st.expander(f"📋 **{workflow.get('workflow_name', workflow_id)}**", expanded=False):
            col_info, col_actions = st.columns([2, 1])

            with col_info:
                st.markdown(f"**Type**: {workflow.get('workflow_type', 'unknown')}")
                st.markdown(f"**Description**: {workflow.get('description', 'No description')}")

                # RAG integration info
                rag_systems = workflow.get('rag_integration', [])
                if rag_systems:
                    st.markdown(f"**RAG Systems**: {', '.join(rag_systems)}")

                # Project structure info
                project_structure = workflow.get('project_structure', {})
                if project_structure:
                    structure_icons = []
                    if project_structure.get('criteria'): structure_icons.append("📋 Criteria")
                    if project_structure.get('templates'): structure_icons.append("📝 Templates")
                    if project_structure.get('inputs'): structure_icons.append("📂 Inputs")
                    if project_structure.get('outputs'): structure_icons.append("📤 Outputs")

                    if structure_icons:
                        st.markdown(f"**Structure**: {' | '.join(structure_icons)}")

            with col_actions:
                # Action buttons
                if st.button(f"📝 Edit {workflow_id}", key=f"edit_{workflow_id}"):
                    st.session_state.selected_workflow_for_editing = workflow_id
                    st.success(f"Selected {workflow_id} for editing")

                if st.button(f"📁 Manage Files", key=f"files_{workflow_id}"):
                    st.session_state.file_manager_selected_workflow = workflow_id
                    st.success(f"Opening file manager for {workflow_id}")

    # File Manager Section (Enhanced with Criteria/Templates tabs) - ALWAYS SHOW FOR TESTING
    # if 'file_manager_selected_workflow' in st.session_state:
    if True:  # Temporarily always show to debug the tabs
        st.markdown("---")
        workflow_id = st.session_state.get('file_manager_selected_workflow', 'alex_qaqc')  # Default to alex_qaqc for testing
        st.markdown(f"## 📁 **File Manager: {workflow_id}**")

        # Clear button
        if st.button("❌ Close File Manager"):
            del st.session_state.file_manager_selected_workflow
            st.rerun()

        # Get project path
        project_path = Path("C:/Users/marti/AI-Shipping/tidyllm/workflows/projects") / workflow_id

        if not project_path.exists():
            st.error(f"❌ Project folder not found: {project_path}")
            return

        # File management tabs
        criteria_tab, templates_tab, inputs_tab, outputs_tab = st.tabs([
            "📋 Criteria", "📝 Templates", "📂 Inputs", "📤 Outputs"
        ])

        # CRITERIA TAB
        with criteria_tab:
            st.markdown("### 📋 **JSON Criteria Files**")
            criteria_path = project_path / "criteria"

            if criteria_path.exists():
                json_files = list(criteria_path.glob("*.json"))

                if json_files:
                    for json_file in json_files:
                        with st.expander(f"📄 {json_file.name}", expanded=False):
                            col_preview, col_actions = st.columns([3, 1])

                            with col_preview:
                                try:
                                    import json
                                    with open(json_file, 'r', encoding='utf-8') as f:
                                        content = json.load(f)
                                    st.json(content)
                                except Exception as e:
                                    st.error(f"Error reading file: {e}")

                            with col_actions:
                                if st.button(f"🗑️ Delete", key=f"del_criteria_{json_file.name}"):
                                    json_file.unlink()
                                    st.success(f"Deleted {json_file.name}")
                                    st.rerun()

                                if st.button(f"⬇️ Download", key=f"dl_criteria_{json_file.name}"):
                                    with open(json_file, 'rb') as f:
                                        st.download_button(
                                            "📥 Download File",
                                            f.read(),
                                            file_name=json_file.name,
                                            mime="application/json",
                                            key=f"download_criteria_{json_file.name}"
                                        )
                else:
                    st.info("📋 No criteria files found")
            else:
                st.warning("📂 Criteria folder not found")
                if st.button("📁 Create Criteria Folder"):
                    criteria_path.mkdir(parents=True, exist_ok=True)
                    st.success("✅ Created criteria folder")
                    st.rerun()

            # Upload criteria files
            st.markdown("---")
            st.markdown("**⬆️ Upload JSON Criteria Files**")
            uploaded_criteria = st.file_uploader(
                "Choose JSON criteria files",
                type=['json'],
                accept_multiple_files=True,
                key=f"criteria_upload_{workflow_id}",
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

        # TEMPLATES TAB
        with templates_tab:
            st.markdown("### 📝 **Template Files**")
            templates_path = project_path / "templates"

            if templates_path.exists():
                md_files = list(templates_path.glob("*.md"))
                txt_files = list(templates_path.glob("*.txt"))
                all_template_files = md_files + txt_files

                if all_template_files:
                    for template_file in all_template_files:
                        with st.expander(f"📄 {template_file.name}", expanded=False):
                            col_preview, col_actions = st.columns([3, 1])

                            with col_preview:
                                try:
                                    with open(template_file, 'r', encoding='utf-8') as f:
                                        content = f.read()
                                    if template_file.suffix == '.md':
                                        st.markdown(content)
                                    else:
                                        st.text(content)
                                except Exception as e:
                                    st.error(f"Error reading file: {e}")

                            with col_actions:
                                if st.button(f"🗑️ Delete", key=f"del_template_{template_file.name}"):
                                    template_file.unlink()
                                    st.success(f"Deleted {template_file.name}")
                                    st.rerun()

                                if st.button(f"⬇️ Download", key=f"dl_template_{template_file.name}"):
                                    with open(template_file, 'rb') as f:
                                        st.download_button(
                                            "📥 Download File",
                                            f.read(),
                                            file_name=template_file.name,
                                            mime="text/plain",
                                            key=f"download_template_{template_file.name}"
                                        )
                else:
                    st.info("📝 No template files found")
            else:
                st.warning("📂 Templates folder not found")
                if st.button("📁 Create Templates Folder"):
                    templates_path.mkdir(parents=True, exist_ok=True)
                    st.success("✅ Created templates folder")
                    st.rerun()

            # Upload template files
            st.markdown("---")
            st.markdown("**⬆️ Upload Template Files**")
            uploaded_templates = st.file_uploader(
                "Choose markdown/text template files",
                type=['md', 'txt'],
                accept_multiple_files=True,
                key=f"templates_upload_{workflow_id}",
                help="Upload markdown or text files to the templates folder"
            )

            if uploaded_templates:
                templates_path.mkdir(exist_ok=True)
                for uploaded_file in uploaded_templates:
                    file_path = templates_path / uploaded_file.name
                    with open(file_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    st.success(f"✅ Uploaded {uploaded_file.name} to templates/")
                st.rerun()

        # INPUTS TAB
        with inputs_tab:
            st.markdown("### 📂 **Input Files**")
            inputs_path = project_path / "inputs"

            if inputs_path.exists():
                input_files = list(inputs_path.glob("*"))
                input_files = [f for f in input_files if f.is_file()]

                if input_files:
                    for input_file in input_files:
                        with st.expander(f"📄 {input_file.name}", expanded=False):
                            col_info, col_actions = st.columns([3, 1])

                            with col_info:
                                st.write(f"**Size**: {input_file.stat().st_size} bytes")
                                st.write(f"**Type**: {input_file.suffix}")

                                # Show preview for text files
                                if input_file.suffix in ['.txt', '.md', '.json', '.csv']:
                                    try:
                                        with open(input_file, 'r', encoding='utf-8') as f:
                                            content = f.read()
                                        if len(content) > 1000:
                                            st.text(content[:1000] + "...")
                                        else:
                                            st.text(content)
                                    except:
                                        st.write("Binary file - preview not available")

                            with col_actions:
                                if st.button(f"🗑️ Delete", key=f"del_input_{input_file.name}"):
                                    input_file.unlink()
                                    st.success(f"Deleted {input_file.name}")
                                    st.rerun()

                                with open(input_file, 'rb') as f:
                                    st.download_button(
                                        "⬇️ Download",
                                        f.read(),
                                        file_name=input_file.name,
                                        key=f"download_input_{input_file.name}"
                                    )
                else:
                    st.info("📂 No input files found")
            else:
                st.warning("📂 Inputs folder not found")
                if st.button("📁 Create Inputs Folder"):
                    inputs_path.mkdir(parents=True, exist_ok=True)
                    st.success("✅ Created inputs folder")
                    st.rerun()

            # Upload input files
            st.markdown("---")
            st.markdown("**⬆️ Upload Input Files**")
            uploaded_inputs = st.file_uploader(
                "Choose input files",
                accept_multiple_files=True,
                key=f"inputs_upload_{workflow_id}",
                help="Upload any files to the inputs folder"
            )

            if uploaded_inputs:
                inputs_path.mkdir(exist_ok=True)
                for uploaded_file in uploaded_inputs:
                    file_path = inputs_path / uploaded_file.name
                    with open(file_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    st.success(f"✅ Uploaded {uploaded_file.name} to inputs/")
                st.rerun()

        # OUTPUTS TAB
        with outputs_tab:
            st.markdown("### 📤 **Output Files**")
            outputs_path = project_path / "outputs"

            if outputs_path.exists():
                output_files = list(outputs_path.glob("*"))
                output_files = [f for f in output_files if f.is_file()]

                if output_files:
                    for output_file in output_files:
                        with st.expander(f"📄 {output_file.name}", expanded=False):
                            col_info, col_actions = st.columns([3, 1])

                            with col_info:
                                st.write(f"**Size**: {output_file.stat().st_size} bytes")
                                st.write(f"**Modified**: {output_file.stat().st_mtime}")

                                # Show preview for text files
                                if output_file.suffix in ['.txt', '.md', '.json', '.csv']:
                                    try:
                                        with open(output_file, 'r', encoding='utf-8') as f:
                                            content = f.read()
                                        if len(content) > 1000:
                                            st.text(content[:1000] + "...")
                                        else:
                                            st.text(content)
                                    except:
                                        st.write("Binary file - preview not available")

                            with col_actions:
                                if st.button(f"🗑️ Delete", key=f"del_output_{output_file.name}"):
                                    output_file.unlink()
                                    st.success(f"Deleted {output_file.name}")
                                    st.rerun()

                                with open(output_file, 'rb') as f:
                                    st.download_button(
                                        "⬇️ Download",
                                        f.read(),
                                        file_name=output_file.name,
                                        key=f"download_output_{output_file.name}"
                                    )
                else:
                    st.info("📤 No output files found")
            else:
                st.info("📂 Outputs folder will be created when workflows run")

                if st.button(f"🚀 Run Workflow", key=f"run_{workflow_id}"):
                    _run_workflow(workflow_manager, workflow_id, workflow)

    # Workflow management section
    st.markdown("---")
    st.subheader("🔧 **Workflow Management**")

    management_col1, management_col2, management_col3 = st.columns(3)

    with management_col1:
        if st.button("🔄 **Refresh Workflows**", use_container_width=True):
            # Force reload of registry
            try:
                registry.__init__()  # Reinitialize registry
                st.success("✅ Workflows refreshed!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Refresh failed: {e}")

    with management_col2:
        if st.button("📊 **Export All**", use_container_width=True):
            _export_all_workflows(workflows)

    with management_col3:
        if st.button("📥 **Import Workflow**", use_container_width=True):
            _show_import_interface()

def _run_workflow(workflow_manager, workflow_id, workflow):
    """Execute a workflow."""
    try:
        st.info(f"🚀 **Running workflow**: {workflow_id}")

        # Check if this is alex_qaqc project
        if workflow_id == "alex_qaqc":
            st.info("🎯 **QAQC Workflow** detected - executing specialized process...")

            # Import and run QAQC workflow
            import sys
            sys.path.append(str(Path("tidyllm/workflows/projects/alex_qaqc")))

            try:
                from alex_qaqc_workflow import run_qaqc_workflow
                with st.spinner("Executing QAQC workflow..."):
                    result = run_qaqc_workflow()

                st.success("✅ **QAQC Workflow completed successfully!**")
                with st.expander("🔍 **Execution Results**", expanded=True):
                    st.json(result)

            except ImportError:
                st.warning("⚠️ QAQC workflow module not found. Using generic execution...")
                _run_generic_workflow(workflow_manager, workflow_id, workflow)

        else:
            _run_generic_workflow(workflow_manager, workflow_id, workflow)

    except Exception as e:
        st.error(f"❌ **Workflow execution failed**: {e}")

def _run_generic_workflow(workflow_manager, workflow_id, workflow):
    """Run a generic workflow."""
    try:
        # Use the workflow manager to execute
        result = workflow_manager.execute_workflow(workflow_id)

        st.success(f"✅ **Workflow '{workflow_id}' executed successfully!**")
        with st.expander("📊 **Execution Results**", expanded=True):
            st.json(result)

    except Exception as e:
        st.warning(f"⚠️ Generic execution not available: {e}")
        st.info("💡 Use the Test Designer tab for advanced workflow testing")

def _export_all_workflows(workflows):
    """Export all workflows to JSON."""
    try:
        import json
        from datetime import datetime

        export_data = {
            "export_date": datetime.now().isoformat(),
            "workflow_count": len(workflows),
            "workflows": workflows
        }

        export_json = json.dumps(export_data, indent=2)

        st.download_button(
            label="📥 **Download Workflows JSON**",
            data=export_json,
            file_name=f"tidyllm_workflows_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

        st.success(f"✅ **Ready to export {len(workflows)} workflows**")

    except Exception as e:
        st.error(f"❌ Export failed: {e}")

def _show_import_interface():
    """Show workflow import interface."""
    st.markdown("### 📥 **Import Workflow**")

    uploaded_file = st.file_uploader(
        "Choose a workflow JSON file",
        type=['json'],
        help="Upload a TidyLLM workflow configuration file"
    )

    if uploaded_file is not None:
        try:
            import json
            workflow_data = json.load(uploaded_file)

            st.success("✅ **File loaded successfully!**")

            # Show preview
            with st.expander("👁️ **Preview Import Data**", expanded=False):
                st.json(workflow_data)

            if st.button("📥 **Import Workflow**", type="primary"):
                # TODO: Implement import logic
                st.info("🚧 Import functionality coming soon!")

        except Exception as e:
            st.error(f"❌ **Invalid file**: {e}")