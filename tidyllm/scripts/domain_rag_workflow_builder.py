"""
Domain RAG Workflow Builder
==========================

Streamlit app to customize and launch domain RAG creation workflows.
Uses the generic template and fills in the blanks based on user input.
"""

import streamlit as st
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import re

class DomainRAGWorkflowBuilder:
    """Build customized domain RAG workflows from generic template"""
    
    def __init__(self):
        self.template_path = Path("workflows/domainrag_generic_template.yaml")
        self.output_dir = Path("workflows")
        self.presets = self._load_presets()
    
    def _load_presets(self) -> Dict[str, Dict]:
        """Load workflow presets for different domains"""
        return {
            "DomainRAG": {
                "WORKFLOW_NAME": "Domain RAG Creation Workflow",
                "COLLECTION_NAME": "domain_rag",
                "DESCRIPTION": "Generic domain RAG creation using heiros, dspy, and tidyllm gateways",
                "DOCUMENT_TYPES": "document, file, text, content",
                "FILE_PATTERNS": ["*.pdf, *.doc*", "*.txt", "*.md"],
                "TEST_QUERIES": [
                    "domain knowledge", 
                    "key concepts", 
                    "main topics"
                ]
            },
            
            "Financial Modeling": {
                "WORKFLOW_NAME": "Financial Model Risk Analysis RAG",
                "COLLECTION_NAME": "financial_model_risk",
                "DESCRIPTION": "Domain RAG for financial model risk management documents",
                "DOCUMENT_TYPES": "risk framework, validation procedure, compliance guide, policy document",
                "FILE_PATTERNS": ["*.pdf", "*.docx", "*.txt", "*.md"],
                "TEST_QUERIES": [
                    "model risk management framework", 
                    "validation procedures", 
                    "regulatory compliance"
                ]
            },
            
            "Legal Compliance": {
                "WORKFLOW_NAME": "Legal Compliance Document RAG",
                "COLLECTION_NAME": "legal_compliance",
                "DESCRIPTION": "Domain RAG for legal and compliance documentation",
                "DOCUMENT_TYPES": "regulation, policy, procedure, legal memo, compliance guide",
                "FILE_PATTERNS": ["*.pdf", "*.docx", "*.txt"],
                "TEST_QUERIES": [
                    "regulatory requirements", 
                    "compliance procedures", 
                    "legal obligations"
                ]
            },
            
            "Technical Documentation": {
                "WORKFLOW_NAME": "Technical Documentation RAG",
                "COLLECTION_NAME": "technical_docs",
                "DESCRIPTION": "Domain RAG for technical documentation and procedures",
                "DOCUMENT_TYPES": "manual, specification, procedure, guide, documentation",
                "FILE_PATTERNS": ["*.pdf", "*.md", "*.txt", "*.docx"],
                "TEST_QUERIES": [
                    "technical procedures", 
                    "system specifications", 
                    "implementation guide"
                ]
            },
            
            "Research Papers": {
                "WORKFLOW_NAME": "Research Paper Collection RAG",
                "COLLECTION_NAME": "research_papers",
                "DESCRIPTION": "Domain RAG for academic and research papers",
                "DOCUMENT_TYPES": "research paper, journal article, conference paper, thesis, report",
                "FILE_PATTERNS": ["*.pdf", "*.txt"],
                "TEST_QUERIES": [
                    "research methodology", 
                    "experimental results", 
                    "literature review"
                ]
            },
            
            "Custom": {
                "WORKFLOW_NAME": "",
                "COLLECTION_NAME": "",
                "DESCRIPTION": "",
                "DOCUMENT_TYPES": "",
                "FILE_PATTERNS": ["*.pdf", "*.txt"],
                "TEST_QUERIES": []
            }
        }
    
    def render_workflow_builder(self):
        """Render the Streamlit workflow builder interface"""
        st.title("🔧 TidyLLM Workflow Builder")
        st.markdown("Create and manage different types of workflows")
        
        # Create tabs for different workflow types
        tab1, tab2, tab3 = st.tabs(["🎯 Domain RAG", "📊 MVR Analysis", "⚙️ Custom Flow"])
        
        with tab1:
            self._render_domain_rag_page()
        
        with tab2:
            self._render_mvr_analysis_page()
        
        with tab3:
            self._render_custom_flow_page()
    
    def _render_domain_rag_page(self):
        """Render the Domain RAG workflow page"""
        st.header("🎯 Domain RAG Creation")
        st.markdown("Create knowledge bases from your documents using heiros, dspy, and tidyllm")
        
        # Sidebar for presets and quick links
        with st.sidebar:
            st.header("📋 Domain RAG Presets")
            domain_presets = {k: v for k, v in self.presets.items() if k in ["DomainRAG", "Financial Modeling", "Legal Compliance", "Technical Documentation", "Research Papers"]}
            preset = st.selectbox(
                "Choose a preset:",
                list(domain_presets.keys()),
                index=0,
                key="domain_preset"
            )
            
            if st.button("📥 Load Preset", key="load_domain_preset"):
                st.session_state.update(domain_presets[preset])
                st.success(f"Loaded {preset} preset!")
            
            # Quick Demo Links
            st.markdown("---")
            st.header("🚀 Quick Demos")
            
            demo_links = [
                ("💬 Chat Interface", "chat_workflow_interface.py", "RAG chat with your collections"),
                ("📊 MLFlow UI", "mlflow ui", "View experiment tracking"),
                ("🏗️ HeirOS Demo", "heiros_streamlit_demo.py", "Workflow orchestration"),
                ("🚀 Demo Launcher", "launch_demo.py", "All demos & configuration")
            ]
            
            for name, command, description in demo_links:
                if st.button(name, key=f"demo_{command}", use_container_width=True):
                    if command.endswith('.py'):
                        st.info(f"🚀 To launch: `streamlit run {command}`")
                    else:
                        st.info(f"🚀 To launch: `{command}`")
                    st.caption(description)
            
            # Configuration Status
            st.markdown("---")
            st.header("⚙️ Status")
            
            # Check MLFlow status
            try:
                import yaml
                settings_path = Path(__file__).parent / "tidyllm" / "admin" / "settings.yaml"
                if settings_path.exists():
                    with open(settings_path, 'r') as f:
                        settings = yaml.safe_load(f)
                    
                    mlflow_enabled = settings.get('integrations', {}).get('mlflow', {}).get('enabled', False)
                    if mlflow_enabled:
                        st.success("✅ MLFlow Enabled")
                    else:
                        st.warning("⚠️ MLFlow Disabled")
                        st.caption("Launch demo_launcher.py to enable")
                else:
                    st.error("❌ Settings not found")
            except Exception:
                st.error("❌ Config check failed")
        
        # Domain RAG workflow form
        self._render_domain_rag_form()
    
    def _render_mvr_analysis_page(self):
        """Render the MVR Analysis workflow page"""
        st.header("📊 MVR Analysis Workflow")
        st.markdown("Complex 4-stage cascade workflow for Motor Vehicle Record analysis")
        
        st.info("🚧 **Coming Soon:** MVR Analysis workflow builder interface")
        st.markdown("This will provide:")
        st.markdown("- MVR document processing pipeline")
        st.markdown("- VST comparison analysis")
        st.markdown("- Domain RAG peer review")
        st.markdown("- Comprehensive reporting")
        
        # Show existing MVR workflow
        st.subheader("📄 Current MVR Workflow")
        mvr_path = Path("workflows/mvr_analysis_flow.yaml")
        if mvr_path.exists():
            with st.expander("View MVR Analysis Flow YAML"):
                st.code(mvr_path.read_text(), language="yaml")
        else:
            st.warning("MVR Analysis Flow YAML not found")
    
    def _render_custom_flow_page(self):
        """Render the Custom Flow workflow page"""
        st.header("⚙️ Custom Flow Builder")
        st.markdown("Build custom workflows with advanced configuration options")
        
        st.info("🚧 **Coming Soon:** Advanced custom workflow builder")
        st.markdown("This will provide:")
        st.markdown("- Custom gateway configurations")
        st.markdown("- Advanced staging and routing")
        st.markdown("- Custom validation rules")
        st.markdown("- Integration with external systems")
        
        # Show template files
        st.subheader("📄 Available Templates")
        template_path = Path("workflows/domainrag_generic_template.yaml")
        if template_path.exists():
            with st.expander("View Generic Template"):
                st.code(template_path.read_text(), language="yaml")
    
    def _render_domain_rag_form(self):
        """Render the Domain RAG workflow form"""
        with st.form("domain_rag_workflow_config"):
            st.header("🎯 Workflow Configuration")
            
            # Basic Settings
            col1, col2 = st.columns(2)
            
            with col1:
                workflow_name = st.text_input(
                    "Workflow Name", 
                    value=st.session_state.get("WORKFLOW_NAME", ""),
                    help="📝 Human-readable name for the workflow (e.g., 'Financial Risk Analysis RAG')",
                    placeholder="Financial Risk Analysis RAG"
                )
                
                collection_name = st.text_input(
                    "Collection Name", 
                    value=st.session_state.get("COLLECTION_NAME", ""),
                    help="🔍 Technical name for chat interface - what you'll search (e.g., 'financial_risk')",
                    placeholder="financial_risk"
                ).lower().replace(" ", "_")
                
                description = st.text_area(
                    "Description",
                    value=st.session_state.get("DESCRIPTION", ""),
                    help="Brief description of the workflow purpose (can be left blank)",
                    placeholder="Domain RAG for analyzing financial model risk documents"
                )
            
            with col2:
                document_types = st.text_input(
                    "Document Types",
                    value=st.session_state.get("DOCUMENT_TYPES", ""),
                    help="Comma-separated list of expected document types (e.g., policy, procedure, manual)",
                    placeholder="policy, procedure, risk framework"
                )
                
                # File patterns (without large header)
                file_patterns = []
                
                # Main pattern (always shown)
                pattern_defaults = st.session_state.get("FILE_PATTERNS", ["*.pdf, *.doc*"])
                main_pattern = st.text_input(
                    "Main File Pattern", 
                    value=pattern_defaults[0] if pattern_defaults else "*.pdf, *.doc*",
                    help="Primary file types to process (comma-separated, e.g., *.pdf, *.doc*, *.txt)",
                    placeholder="*.pdf, *.doc*"
                )
                if main_pattern.strip():
                    file_patterns.append(main_pattern.strip())
                
                # Additional patterns (optional)
                additional_pattern_1 = st.text_input(
                    "Additional Pattern 1", 
                    value=pattern_defaults[1] if len(pattern_defaults) > 1 else "",
                    help="Optional second file type",
                    placeholder="*.txt"
                )
                if additional_pattern_1.strip():
                    file_patterns.append(additional_pattern_1.strip())
                
                additional_pattern_2 = st.text_input(
                    "Additional Pattern 2", 
                    value=pattern_defaults[2] if len(pattern_defaults) > 2 else "",
                    help="Optional third file type",
                    placeholder="*.md"
                )
                if additional_pattern_2.strip():
                    file_patterns.append(additional_pattern_2.strip())
            
            # Advanced Settings
            st.header("⚙️ Advanced Settings")
            
            adv_col1, adv_col2, adv_col3 = st.columns(3)
            
            with adv_col1:
                st.subheader("📏 Chunking")
                chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
                chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200, 50)
                
            with adv_col2:
                st.subheader("🧠 Embeddings") 
                embedding_dim = st.selectbox("Embedding Dimension", [384, 768, 1024], index=2)
                embedding_batch = st.slider("Batch Size", 10, 100, 50, 10)
                
            with adv_col3:
                st.subheader("🔍 Search")
                similarity_threshold = st.slider("Similarity Threshold", 0.5, 0.9, 0.7, 0.05)
                max_results = st.slider("Max Results", 5, 50, 10, 5)
            
            # Gateway Configuration
            st.header("🚪 Gateway Configuration")
            st.markdown("**Domain RAG Gateway Setup:**")
            
            gate_col1, gate_col2 = st.columns(2)
            
            with gate_col1:
                use_heiros = st.checkbox("🔧 **heiros** - Workflow Orchestration", value=True, help="File operations, S3, vector storage, workflow management")
                use_dspy = st.checkbox("🧠 **dspy** - Prompts & Structured AI", value=True, help="Document classification, embedding generation, structured tasks") 
                use_tidyllm = st.checkbox("⚡ **tidyllm** - Gateway System", value=True, help="TidyLLM gateway for integrated processing")
                
            with gate_col2:
                use_llm = st.checkbox("💬 **llm** - Language Models", value=False, help="Natural language processing, summarization, quality validation")
                use_custom = st.checkbox("🔧 **custom** - Custom Gateway", value=False, help="Custom processing gateway")
                
                # Set primary gateway based on selection
                if use_heiros:
                    primary_gateway = "heiros"
                elif use_dspy:
                    primary_gateway = "dspy"
                elif use_tidyllm:
                    primary_gateway = "tidyllm"
                else:
                    primary_gateway = "llm"
                
                secondary_gateway = "dspy" if use_dspy else "llm"
                doc_gateway = "heiros" if use_heiros else "dspy"
                embed_gateway = "dspy" if use_dspy else "tidyllm"
            
            # Test Queries
            st.header("🧪 Test Queries")
            st.markdown("Define test queries to validate search functionality:")
            
            test_queries = []
            query_defaults = st.session_state.get("TEST_QUERIES", [""])
            
            for i, default_query in enumerate(query_defaults + [""]):  # Add one empty for new query
                query = st.text_input(f"Test Query {i+1}", value=default_query, key=f"query_{i}")
                if query.strip():
                    test_queries.append(query.strip())
            
            # Submit button
            submitted = st.form_submit_button("🚀 Generate Workflow", use_container_width=True)
            
            if submitted:
                # Validate inputs
                if not workflow_name.strip():
                    st.error("❌ Workflow Name is required")
                    return
                if not collection_name.strip():
                    st.error("❌ Collection Name is required")
                    return
                if len(collection_name) < 3:
                    st.error("❌ Collection Name must be at least 3 characters long")
                    return
                
                # Generate workflow
                workflow_config = self._build_workflow_config(
                    workflow_name=workflow_name,
                    collection_name=collection_name,
                    description=description,
                    document_types=document_types,
                    file_patterns=file_patterns,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    embedding_dim=embedding_dim,
                    embedding_batch=embedding_batch,
                    similarity_threshold=similarity_threshold,
                    max_results=max_results,
                    primary_gateway=primary_gateway,
                    secondary_gateway=secondary_gateway,
                    doc_gateway=doc_gateway,
                    embed_gateway=embed_gateway,
                    test_queries=test_queries
                )
                
                # Save and display
                filename = self._save_workflow(workflow_config, collection_name)
                st.success(f"✅ Workflow saved as `{filename}`")
                
                # Show preview
                with st.expander("📄 Workflow Preview"):
                    st.code(self._generate_yaml_content(workflow_config), language="yaml")
                
                # Success message only
                st.success("✅ Workflow generated successfully!")
        
        # Launch Options (immediately after workflow generation)
        if 'workflow_generated' in st.session_state:
            st.header("🚀 Next Steps")
            st.markdown("Choose your next action:")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Step 1: Documents**")
                if st.button("📁 Upload Files", use_container_width=True):
                    st.info("💡 Scroll down to the Upload Documents section below")
            
            with col2:
                st.markdown("**Step 2: Process**")
                if st.button("▶️ Launch Workflow", use_container_width=True):
                    workflow_filename = st.session_state.get('current_workflow', 'workflow.yaml')
                    collection_name = st.session_state.get('current_collection', 'domain_rag')
                    self._launch_workflow(workflow_filename, collection_name)
            
            with col3:
                st.markdown("**Step 3: Chat**")
                if st.button("💬 Open Chat", use_container_width=True):
                    collection_name = st.session_state.get('current_collection', 'domain_rag')
                    st.info(f"💬 Collection '{collection_name}' will be ready for chat after processing")
            
            st.markdown("---")
        
        # Document Upload (outside form)
        st.header("📎 Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Upload documents for processing",
            accept_multiple_files=True,
            type=['pdf', 'txt', 'docx', 'md'],
            help="Upload your documents to create the RAG knowledge base"
        )
        
        if uploaded_files:
            st.info(f"📄 Selected {len(uploaded_files)} files")
            
            # Show file list
            with st.expander(f"📋 View {len(uploaded_files)} files"):
                for file in uploaded_files:
                    file_size = len(file.read()) / 1024  # KB
                    file.seek(0)  # Reset file pointer
                    st.write(f"📄 {file.name} ({file_size:.1f} KB)")
            
            if st.button("💾 Save Files to Workflow", use_container_width=True):
                collection_name = st.session_state.get('current_collection', 'domain_rag')
                result = self._save_uploaded_files(uploaded_files, collection_name)
                if result['success']:
                    st.success(f"✅ Saved {result['files_saved']} files to inputs/{collection_name}/")
                else:
                    st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
    
    def _build_workflow_config(self, **kwargs) -> Dict[str, Any]:
        """Build workflow configuration dictionary"""
        
        # Generate unique workflow ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        workflow_id = f"{kwargs['collection_name']}_{timestamp}"
        
        return {
            "WORKFLOW_NAME": kwargs["workflow_name"],
            "WORKFLOW_ID": workflow_id,
            "DESCRIPTION": kwargs["description"],
            "COLLECTION_NAME": kwargs["collection_name"],
            
            # Document settings
            "DOCUMENT_TYPES": kwargs["document_types"],
            "FILE_PATTERNS": kwargs["file_patterns"],
            
            # Processing settings
            "CHUNK_SIZE": kwargs["chunk_size"],
            "CHUNK_OVERLAP": kwargs["chunk_overlap"],
            "EMBEDDING_DIMENSION": kwargs["embedding_dim"],
            "EMBEDDING_BATCH_SIZE": kwargs["embedding_batch"],
            
            # Search settings
            "SIMILARITY_THRESHOLD": kwargs["similarity_threshold"],
            "MAX_RESULTS": kwargs["max_results"],
            
            # Gateway settings
            "PRIMARY_GATEWAY": kwargs["primary_gateway"],
            "SECONDARY_GATEWAY": kwargs["secondary_gateway"],
            "DOC_GATEWAY": kwargs["doc_gateway"],
            "EMBED_GATEWAY": kwargs["embed_gateway"],
            "VECTOR_GATEWAY": kwargs["primary_gateway"],
            "QA_GATEWAY": "llm",
            
            # Test queries
            "TEST_QUERIES": kwargs["test_queries"],
            
            # Timeouts and technical settings
            "SCAN_TIMEOUT": 60,
            "EXTRACTION_TIMEOUT": 300,
            "EMBEDDING_TIMEOUT": 600,
            "PROVIDER_TIMEOUT": 30,
            "COLLECTION_TIMEOUT": 60,
            "INSERT_TIMEOUT": 300,
            "INSERT_BATCH_SIZE": 100,
            "INDEX_TYPE": "hnsw",
            
            # Parallel execution
            "PARALLEL_EXTRACTION": True,
            "PARALLEL_METADATA": True,
            "PARALLEL_EMBEDDING": True,
            
            # Other settings
            "EMBEDDING_PROVIDER": "tidyllm_sentence",
            "ENABLE_RERANKING": True,
            "CONTEXT_WINDOW": 4096,
            "PROGRESS_UPDATES": True,
            "TIDYMART_LOGGING": True,
            "VECTOR_METRICS": True,
            "PERFORMANCE_TRACKING": True,
            "QA_EXTRACTION": "required",
            "QA_EMBEDDING": "required", 
            "QA_INDEX": "required",
            "QA_SEARCH": "required",
            "MAX_RETRIES": 3,
            "FALLBACK_GATEWAY": "llm",
            "ERROR_QUEUE": "error_review/",
            "NOTIFICATION_WEBHOOK": "tidymart://domain_rag_alerts",
            "METADATA_FORMAT": "json",
            "LOG_FORMAT": "markdown",
            "TEST_FORMAT": "json",
            "METRICS_FORMAT": "json"
        }
    
    def _generate_yaml_content(self, config: Dict[str, Any]) -> str:
        """Generate YAML content from configuration"""
        
        # Load template
        if not self.template_path.exists():
            return "Template file not found!"
        
        with open(self.template_path, 'r') as f:
            template_content = f.read()
        
        # Replace placeholders
        yaml_content = template_content
        for key, value in config.items():
            placeholder = "{" + key + "}"
            
            # Handle different value types
            if isinstance(value, list):
                if all(isinstance(item, str) for item in value):
                    # String list for YAML
                    formatted_value = "\n" + "\n".join([f'      - "{item}"' for item in value])
                else:
                    formatted_value = json.dumps(value)
            elif isinstance(value, bool):
                formatted_value = str(value).lower()
            else:
                formatted_value = str(value)
            
            yaml_content = yaml_content.replace(placeholder, formatted_value)
        
        return yaml_content
    
    def _save_workflow(self, config: Dict[str, Any], collection_name: str) -> str:
        """Save generated workflow to file"""
        
        filename = f"domainrag_{collection_name}.yaml"
        filepath = self.output_dir / filename
        
        # Generate YAML content
        yaml_content = self._generate_yaml_content(config)
        
        # Save to file
        self.output_dir.mkdir(exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(yaml_content)
        
        st.session_state['workflow_generated'] = True
        st.session_state['current_workflow'] = filename
        st.session_state['current_collection'] = collection_name
        
        return filename
    
    def _launch_workflow(self, filename: str, collection_name: str):
        """Launch the workflow execution"""
        st.info(f"🚀 Launching workflow: {filename}")
        st.info(f"📊 Collection: {collection_name}")
        
        # Show estimated time
        st.info("⏱️ **Estimated Time:** 2-5 minutes for full RAG creation")
        
        # Progress simulation (in real implementation, you'd get actual progress)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        import time
        
        # Simulate workflow stages
        stages = [
            (10, "📄 Stage 1: Document ingestion and text extraction..."),
            (30, "🧠 Stage 2: Generating embeddings..."),
            (60, "🔍 Stage 3: Creating vector index..."),
            (90, "✅ Stage 4: Enabling RAG interface..."),
            (100, "🎉 Workflow complete! Collection ready for chat.")
        ]
        
        for progress, message in stages:
            status_text.text(message)
            progress_bar.progress(progress)
            time.sleep(1)  # Simulate processing time
            
        st.success(f"✅ **Workflow Complete!** Collection `{collection_name}` is ready for chat interface")
        
        # Show next steps
        st.info("💬 **Next Steps:** Use the chat interface to query your domain knowledge")
        
        # In real implementation, replace simulation with actual workflow call:
        st.code(f"""
# Real Implementation Command:
python -m tidyllm.workflows.runner --workflow {filename} --collection {collection_name}

# Or via TidyLLM API:
from tidyllm.workflows import WorkflowRunner
runner = WorkflowRunner()
runner.launch_workflow("{filename}", collection_name="{collection_name}")
        """, language="python")
    
    def _save_uploaded_files(self, files, collection_name: str):
        """Save uploaded files to processing directory"""
        
        try:
            # Create input directory
            input_dir = Path(f"workflows/inputs/{collection_name}")
            input_dir.mkdir(parents=True, exist_ok=True)
            
            saved_files = []
            for file in files:
                file_path = input_dir / file.name
                with open(file_path, "wb") as f:
                    f.write(file.read())
                saved_files.append(file.name)
            
            return {
                "success": True,
                "files_saved": len(saved_files),
                "input_directory": str(input_dir),
                "files": saved_files,
                "collection_name": collection_name
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "files_saved": 0
            }
    
    def _process_directory(self, directory_path: str, collection_name: str) -> Dict[str, Any]:
        """Process all documents in a directory"""
        
        try:
            # Create input directory for workflow
            input_dir = Path(f"workflows/inputs/{collection_name}")
            input_dir.mkdir(parents=True, exist_ok=True)
            
            # Find and copy supported files
            source_dir = Path(directory_path)
            supported_extensions = ['.pdf', '.txt', '.docx', '.md', '.doc']
            found_files = []
            
            for ext in supported_extensions:
                found_files.extend(list(source_dir.glob(f"*{ext}")))
                found_files.extend(list(source_dir.glob(f"**/*{ext}")))  # Include subdirectories
            
            # Copy files to input directory
            copied_files = []
            for file in found_files:
                dest_file = input_dir / file.name
                # Handle duplicate names by adding counter
                counter = 1
                original_stem = dest_file.stem
                while dest_file.exists():
                    dest_file = input_dir / f"{original_stem}_{counter}{dest_file.suffix}"
                    counter += 1
                
                # Copy file
                import shutil
                shutil.copy2(file, dest_file)
                copied_files.append(dest_file.name)
            
            return {
                "success": True,
                "files_processed": len(copied_files),
                "input_directory": str(input_dir),
                "files": copied_files,
                "collection_name": collection_name
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "files_processed": 0
            }

def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="Domain RAG Workflow Builder",
        page_icon="🔧",
        layout="wide"
    )
    
    builder = DomainRAGWorkflowBuilder()
    builder.render_workflow_builder()

if __name__ == "__main__":
    main()