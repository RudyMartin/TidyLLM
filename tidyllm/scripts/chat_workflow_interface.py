"""
TidyLLM Chat Workflow Interface
===============================

Simple chat interface with Flow Agreement sidebar for workflow launching.
Supports both predefined Flow Agreements and bracket commands like [mvr_analysis].
"""

import streamlit as st
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import FlowAgreement classes
try:
    from ..flow.flow_agreements import FlowAgreement, FlowAgreementConfig
except ImportError:
    # Fallback for when flow agreements are not available
    class FlowAgreement:
        def __init__(self, config): pass
    class FlowAgreementConfig:
        def __init__(self, **kwargs): pass

# Import TidyLLM components
try:
    from .rag2dag import RAG2DAGConverter, RAG2DAGConfig
    # FlowAgreement classes already imported above
    from .gateways import get_gateway
except ImportError:
    # Placeholder imports for development
    class RAG2DAGConverter:
        def __init__(self, config): pass
        def create_workflow_from_query(self, query, files): 
            return {"pattern_name": "Mock Pattern", "dag_nodes": []}


class MVRAnalysisFlow(FlowAgreement):
    """MVR Analysis Flow Agreement - 4-stage document processing workflow."""
    
    def __init__(self):
        config = FlowAgreementConfig(
            agreement_id="mvr_analysis_v1",
            agreement_type="MVR Document Analysis",
            created_by="TidyLLM",
            approved_gateways=["dspy", "llm"],
            audit_requirements=["tidymart_storage", "processing_trail"],
            auto_optimizations=["noise_filtering", "sentiment_analysis"]
        )
        super().__init__(config)
        
        # Define the 4-stage workflow
        self.stages = [
            {
                "stage": "mvr_tag",
                "description": "Initial classification and metadata extraction",
                "drop_zone": "mvr_tag/",
                "file_patterns": ["*.pdf", "*.txt", "*.doc", "*.docx"],
                "operations": [
                    "read_top_5_pages",
                    "classify_document", 
                    "extract_metadata",
                    "extract_table_of_contents",
                    "ynsr_noise_analysis",
                    "sentiment_analysis_initial",
                    "create_digest_text",
                    "store_in_tidymart"
                ],
                "routing": {
                    "class_vst": "que/",
                    "class_mvr": "mvr_qa/", 
                    "non_class": "que/"
                }
            },
            {
                "stage": "mvr_qa",
                "description": "MVR vs VST comparison and analysis", 
                "drop_zone": "mvr_qa/",
                "required_metadata": "REV00000 format (REV + 5 alphanumeric)",
                "operations": [
                    "match_mvr_vst_by_metadata",
                    "section_by_section_review",
                    "create_digest_markdown",
                    "create_detail_markdown", 
                    "process_markdown_chain",
                    "store_results_tidymart"
                ],
                "routing": {
                    "processed": "mvr_peer/",
                    "incomplete": "que/"
                }
            },
            {
                "stage": "mvr_peer",
                "description": "Domain RAG peer review analysis",
                "drop_zone": "mvr_peer/",
                "operations": [
                    "load_domain_rag",
                    "analyze_mvr_text",
                    "analyze_digest_review", 
                    "analyze_stepwise_review",
                    "triangulate_analysis",
                    "store_in_tidymart",
                    "save_to_database"
                ],
                "routing": {
                    "completed": "mvr_report/"
                }
            },
            {
                "stage": "mvr_report", 
                "description": "Final report generation",
                "drop_zone": "mvr_report/",
                "operations": [
                    "create_full_analysis_markdown",
                    "generate_pdf_report",
                    "generate_json_summary",
                    "archive_final_outputs"
                ],
                "routing": {
                    "final": "completed/"
                }
            }
        ]
    
    def validate(self) -> bool:
        """Validate MVR flow agreement."""
        # Check TidyMart connection, domain RAG availability, etc.
        return True
    
    def get_gateway_config(self) -> Dict[str, Any]:
        """Gateway configuration for MVR analysis."""
        return {
            "backend": "auto",
            "model": "claude-3-sonnet",
            "specialized_operations": {
                "ynsr_analysis": "dspy",
                "sentiment_analysis": "llm", 
                "domain_rag": "heiros"
            }
        }
    
    def get_drop_zone_config(self) -> Dict[str, Any]:
        """Drop zone configuration for 4-stage MVR workflow."""
        return {
            "workflow_type": "multi_stage_cascade",
            "stages": self.stages,
            "global_settings": {
                "tidymart_storage": True,
                "processing_trail": True,
                "metadata_validation": "REV00000_format"
            }
        }


class WorkflowChatInterface:
    """Main chat interface with Flow Agreement sidebar."""
    
    def __init__(self, ai_manager=None, llm_gateway=None):
        self.ai_manager = ai_manager
        self.llm_gateway = llm_gateway
        self.available_flows = self._load_flow_agreements()
        self.rag2dag_config = RAG2DAGConfig.create_default_config() if 'RAG2DAGConfig' in globals() else None
        
    def _load_flow_agreements(self) -> Dict[str, FlowAgreement]:
        """Load available Flow Agreements."""
        flows = {
            "MVR Analysis": MVRAnalysisFlow(),
            # Add more flows here
            "Research Synthesis": self._create_research_flow(),
            "Compliance Review": self._create_compliance_flow(),
            "Document Classification": self._create_classification_flow()
        }
        return flows
    
    def _create_research_flow(self) -> FlowAgreement:
        """Create research synthesis flow agreement."""
        config = FlowAgreementConfig(
            agreement_id="research_synthesis_v1",
            agreement_type="Research Document Synthesis",
            created_by="TidyLLM",
            approved_gateways=["dspy", "llm"]
        )
        
        class ResearchFlow(FlowAgreement):
            def __init__(self, config):
                super().__init__(config)
            def validate(self): return True
            def get_gateway_config(self): return {"backend": "auto"}
            def get_drop_zone_config(self): return {"pattern": "research_synthesis"}
        
        return ResearchFlow(config)
    
    def _create_compliance_flow(self) -> FlowAgreement:
        """Create compliance review flow agreement.""" 
        config = FlowAgreementConfig(
            agreement_id="compliance_review_v1", 
            agreement_type="Regulatory Compliance Review",
            created_by="TidyLLM",
            approved_gateways=["llm", "heiros"]
        )
        
        class ComplianceFlow(FlowAgreement):
            def __init__(self, config):
                super().__init__(config)
            def validate(self): return True
            def get_gateway_config(self): return {"backend": "auto", "compliance_mode": True}
            def get_drop_zone_config(self): return {"pattern": "compliance_extraction"}
        
        return ComplianceFlow(config)
    
    def _create_classification_flow(self) -> FlowAgreement:
        """Create document classification flow agreement."""
        config = FlowAgreementConfig(
            agreement_id="doc_classification_v1",
            agreement_type="Document Classification & Routing", 
            created_by="TidyLLM",
            approved_gateways=["dspy"]
        )
        
        class ClassificationFlow(FlowAgreement):
            def __init__(self, config):
                super().__init__(config)
            def validate(self): return True
            def get_gateway_config(self): return {"backend": "auto", "classification_mode": True}
            def get_drop_zone_config(self): return {"pattern": "document_classification"}
        
        return ClassificationFlow(config)
    
    def render_sidebar(self):
        """Render the Flow Agreements sidebar (25% width)."""
        with st.sidebar:
            st.header("🔄 Flow Agreements")
            st.markdown("*Select a workflow to get started*")
            
            # Available Flow Agreements
            st.subheader("📋 Available Flows")
            
            for flow_name, flow_agreement in self.available_flows.items():
                if st.button(flow_name, key=f"flow_{flow_name}", use_container_width=True):
                    # Add flow to chat
                    self._add_flow_to_chat(flow_name, flow_agreement)
            
            st.markdown("---")
            
            # Quick Commands
            st.subheader("⚡ Quick Commands")
            st.markdown("""
            **Bracket Commands:**
            - `[mvr_analysis]` - Full MVR workflow
            - `[research_synth]` - Research synthesis
            - `[compliance_check]` - Compliance review
            - `[doc_classify]` - Document classification
            
            **Chat Examples:**
            - "Start MVR analysis workflow"
            - "I need to analyze research papers"
            - "Help me with compliance review"
            """)
            
            st.markdown("---")
            
            # Workflow Status
            st.subheader("📊 Active Workflows") 
            if "active_workflows" in st.session_state:
                for workflow_id, status in st.session_state.active_workflows.items():
                    st.markdown(f"**{workflow_id[:12]}...** - {status['status']}")
            else:
                st.markdown("*No active workflows*")
    
    def _add_flow_to_chat(self, flow_name: str, flow_agreement: FlowAgreement):
        """Add selected flow to chat conversation."""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Add user selection message
        st.session_state.messages.append({
            "role": "user",
            "content": f"I want to use the {flow_name} workflow"
        })
        
        # Add assistant response with flow details
        flow_info = self._get_flow_info(flow_agreement)
        st.session_state.messages.append({
            "role": "assistant", 
            "content": flow_info
        })
    
    def _get_flow_info(self, flow_agreement: FlowAgreement) -> str:
        """Get formatted information about a flow agreement."""
        config = flow_agreement.config
        
        info = f"""
**{config.agreement_type} Workflow Activated** ✅

**Agreement ID:** {config.agreement_id}
**Approved Gateways:** {', '.join(config.approved_gateways)}
**Valid Until:** {config.valid_until.strftime('%Y-%m-%d') if config.valid_until else 'Unlimited'}

"""
        
        # Add stage-specific info for MVR Analysis
        if hasattr(flow_agreement, 'stages'):
            info += "**Workflow Stages:**\n"
            for i, stage in enumerate(flow_agreement.stages, 1):
                info += f"{i}. **{stage['stage']}**: {stage['description']}\n"
        
        info += f"""
**Ready to start!** You can now:
1. Drop files in the workflow folders
2. Ask questions about the process
3. Monitor progress in real-time

Type your question or `[start_workflow]` to begin processing.
"""
        
        return info
    
    def render_chat(self):
        """Render the main chat interface (75% width)."""
        st.title("💬 TidyLLM Workflow Chat")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": """Welcome to TidyLLM Workflow Chat! 👋

I can help you launch and manage document analysis workflows. Here's how to get started:

**Option 1:** Select a Flow Agreement from the sidebar →
**Option 2:** Use bracket commands like `[mvr_analysis]`
**Option 3:** Just tell me what you want to analyze!

What would you like to work on today?"""
                }
            ]
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about workflows or type [command]..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Process user input
            response = self._process_user_input(prompt)
            
            # Add assistant response
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Trigger rerun to show new messages
            st.rerun()
    
    def _process_user_input(self, user_input: str) -> str:
        """Process user input and generate response."""
        input_lower = user_input.lower().strip()
        
        # Check for bracket commands
        if '[' in user_input and ']' in user_input:
            return self._handle_bracket_command(user_input)
        
        # Check for workflow-related keywords
        if any(keyword in input_lower for keyword in ['mvr', 'analysis', 'workflow', 'start']):
            if 'mvr' in input_lower:
                return self._handle_mvr_request(user_input)
            else:
                return self._handle_general_workflow_request(user_input)
        
        # Check for help requests
        if any(keyword in input_lower for keyword in ['help', 'how', 'what can']):
            return self._provide_help()
        
        # Default response
        return self._generate_contextual_response(user_input)
    
    def _handle_bracket_command(self, command: str) -> str:
        """Handle bracket commands like [mvr_analysis]."""
        # Extract command from brackets
        import re
        commands = re.findall(r'\[([^\]]+)\]', command)
        
        if not commands:
            return "I didn't find a valid bracket command. Try `[mvr_analysis]` or `[research_synth]`."
        
        cmd = commands[0].lower()
        
        if cmd == "mvr_analysis":
            return self._start_mvr_workflow()
        elif cmd == "research_synth":
            return self._start_research_workflow() 
        elif cmd == "compliance_check":
            return self._start_compliance_workflow()
        elif cmd == "doc_classify":
            return self._start_classification_workflow()
        elif cmd == "start_workflow":
            return self._start_selected_workflow()
        else:
            return f"Unknown command: `[{cmd}]`. Available commands: `[mvr_analysis]`, `[research_synth]`, `[compliance_check]`, `[doc_classify]`"
    
    def _start_mvr_workflow(self) -> str:
        """Start the MVR analysis workflow."""
        mvr_flow = self.available_flows["MVR Analysis"]
        
        # Create workflow directories
        workflow_id = f"mvr_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize active workflow tracking
        if "active_workflows" not in st.session_state:
            st.session_state.active_workflows = {}
        
        st.session_state.active_workflows[workflow_id] = {
            "status": "initializing",
            "flow_type": "MVR Analysis",
            "created": datetime.now().isoformat(),
            "current_stage": "mvr_tag"
        }
        
        return f"""
**🚀 MVR Analysis Workflow Started!**

**Workflow ID:** `{workflow_id}`

**📁 Drop Zone Structure Created:**
```
{workflow_id}/
├── mvr_tag/          ← Drop initial documents here
├── mvr_qa/           ← Auto-populated from mvr_tag
├── mvr_peer/         ← Auto-populated from mvr_qa  
└── mvr_report/       ← Final reports appear here
```

**🔄 4-Stage Process:**
1. **MVR Tag**: Document classification & metadata extraction
2. **MVR QA**: MVR vs VST comparison analysis
3. **MVR Peer**: Domain RAG peer review
4. **MVR Report**: Final PDF and JSON generation

**Next Steps:**
1. Drop your PDF/DOC files in `{workflow_id}/mvr_tag/`
2. Ensure files have REV00000 format metadata
3. Monitor progress here - I'll update you as files move through stages

**Ready to process!** Drop your documents and I'll handle the rest.
"""
    
    def _start_research_workflow(self) -> str:
        """Start research synthesis workflow."""
        workflow_id = f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if "active_workflows" not in st.session_state:
            st.session_state.active_workflows = {}
        
        st.session_state.active_workflows[workflow_id] = {
            "status": "ready",
            "flow_type": "Research Synthesis", 
            "created": datetime.now().isoformat()
        }
        
        return f"""
**📚 Research Synthesis Workflow Ready!**

**Workflow ID:** `{workflow_id}`

This workflow will:
- Extract key findings from multiple research papers
- Identify common themes and methodologies  
- Synthesize insights across documents
- Generate comprehensive analysis report

**To start:** Drop your research papers in the upload area or tell me what research question you want to explore.
"""
    
    def _start_compliance_workflow(self) -> str:
        """Start compliance review workflow."""
        return """
**⚖️ Compliance Review Workflow Ready!**

This workflow specializes in:
- Extracting regulatory requirements
- Identifying compliance gaps
- Mapping requirements to policies
- Generating compliance matrices

**Ready to analyze compliance documents!** What regulations are you working with?
"""
    
    def _start_classification_workflow(self) -> str:
        """Start document classification workflow."""
        return """
**📂 Document Classification Workflow Ready!**

This workflow will:
- Automatically classify document types
- Extract metadata and key information
- Route documents to appropriate workflows
- Create classification reports

**Drop your mixed documents** and I'll sort them automatically!
"""
    
    def _handle_mvr_request(self, user_input: str) -> str:
        """Handle MVR-specific requests."""
        return """
**MVR Analysis Workflow** 🔬

This is our most sophisticated document analysis workflow, perfect for:
- Motor Vehicle Record analysis
- VST (Verification Support Tool) comparisons
- Regulatory document processing
- Multi-stage peer review

**Would you like me to:**
1. `[mvr_analysis]` - Start full MVR workflow
2. Explain the 4-stage process in detail
3. Set up drop zones for your documents

What specific MVR analysis do you need?
"""
    
    def _handle_general_workflow_request(self, user_input: str) -> str:
        """Handle general workflow requests."""
        return """
**Available Workflows** 🔄

I can help you with several types of analysis:

**📋 Flow Agreements** (select from sidebar):
- **MVR Analysis** - 4-stage regulatory document processing
- **Research Synthesis** - Multi-document research analysis  
- **Compliance Review** - Regulatory requirement extraction
- **Document Classification** - Automatic document sorting

**⚡ Quick Commands:**
- `[mvr_analysis]` - Start MVR workflow
- `[research_synth]` - Research analysis
- `[compliance_check]` - Compliance review

**Or just tell me:** "I need to analyze [type] documents for [purpose]"

What type of documents are you working with?
"""
    
    def _provide_help(self) -> str:
        """Provide help information."""
        return """
**TidyLLM Chat Help** 💡

**🎯 How to Use:**
1. **Select Flow** - Choose from sidebar or use brackets `[mvr_analysis]`
2. **Drop Files** - Upload to generated drop zones
3. **Monitor** - I'll update you on progress automatically

**🔧 Available Commands:**
- `[mvr_analysis]` - Full MVR document workflow
- `[research_synth]` - Research paper synthesis
- `[compliance_check]` - Regulatory compliance review
- `[doc_classify]` - Document classification

**📁 Workflow Types:**
- **MVR Analysis** - 4-stage: tag → qa → peer → report
- **Research** - Multi-document analysis and synthesis
- **Compliance** - Regulatory requirement extraction
- **Classification** - Automatic document sorting

**💬 Just Ask:**
- "Start MVR workflow"
- "Analyze research papers"
- "Help with compliance"
- "What can you do?"

Need help with something specific?
"""
    
    def _generate_contextual_response(self, user_input: str) -> str:
        """Generate contextual response for general queries."""
        return f"""
Thanks for your message! I'm here to help with document analysis workflows.

**I can help you:**
- Launch specialized workflows (MVR, Research, Compliance)
- Set up automated document processing  
- Guide you through multi-stage analysis
- Monitor workflow progress

**Try saying:**
- "I need to analyze documents"
- `[mvr_analysis]` for regulatory workflow
- "Help me with research papers"
- Select a Flow Agreement from the sidebar →

What type of document analysis do you need help with?
"""


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="TidyLLM Workflow Chat",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize the interface
    interface = WorkflowChatInterface()
    
    # Create two-column layout (25% sidebar, 75% chat)
    # Note: Streamlit handles sidebar automatically, main area gets remaining space
    
    # Render sidebar (25%)
    interface.render_sidebar()
    
    # Render main chat area (75%)
    interface.render_chat()


if __name__ == "__main__":
    main()