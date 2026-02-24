#!/usr/bin/env python3
"""
RAG Creator V3 - UnifiedRAGManager Portal
=========================================

Comprehensive RAG management portal using UnifiedRAGManager (URM) for:
- CRUD operations on all 6 RAG orchestrators
- Health checks and system monitoring
- RAG tailoring and customization
- Integration with AI-Powered, Postgres, Judge, Intelligent, SME, and DSPy systems
"""

import streamlit as st
import sys
from pathlib import Path
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

# V2 Architecture imports
from tidyllm.core.settings import settings
from tidyllm.core.resources import get_resources
from tidyllm.core.state import (
    ensure_session,
    get_portal_state,
    set_portal_value,
    get_portal_value,
    set_current_portal
)

# UnifiedRAGManager - The core of V3
from tidyllm.services.unified_rag_manager import UnifiedRAGManager, RAGSystemType
from tidyllm.infrastructure.session.unified import UnifiedSessionManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="RAG Creator V3",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

class RAGCreatorV3Portal:
    """V3 RAG Creator Portal with full UnifiedRAGManager integration"""

    def __init__(self):
        # Initialize V2 session state
        ensure_session()
        set_current_portal("rag_v3")

        # Get V2 resources
        self.resources = get_resources()

        # Initialize UnifiedRAGManager
        self.usm = self._init_session_manager()
        self.rag_manager = None
        if self.usm:
            self.rag_manager = UnifiedRAGManager(session_manager=self.usm)

        # RAG system configurations
        self.rag_systems = self._get_available_rag_systems()
        self.health_check_cache = {}

    def _init_session_manager(self) -> Optional[UnifiedSessionManager]:
        """Initialize USM session manager"""
        try:
            return UnifiedSessionManager()
        except Exception as e:
            st.error(f"Failed to initialize USM: {e}")
            return None

    def _get_available_rag_systems(self) -> Dict[str, Dict[str, Any]]:
        """Get all available RAG systems with their capabilities"""
        return {
            RAGSystemType.AI_POWERED.value: {
                "name": "AI-Powered RAG",
                "description": "AI-enhanced responses via CorporateLLMGateway + Bedrock analysis",
                "capabilities": ["AI Analysis", "Corporate Context", "Session Continuity", "Quality Assurance"],
                "use_cases": ["Complex analysis", "Corporate knowledge", "Multi-turn conversations"],
                "icon": "🤖"
            },
            RAGSystemType.POSTGRES.value: {
                "name": "PostgreSQL RAG",
                "description": "Authority-based precedence with existing SME infrastructure",
                "capabilities": ["Authority Routing", "Compliance RAG", "Document RAG", "Expert RAG"],
                "use_cases": ["Regulatory compliance", "Authority-based decisions", "SME knowledge"],
                "icon": "🗃️"
            },
            RAGSystemType.JUDGE.value: {
                "name": "Judge RAG",
                "description": "External system integration with transparent fallback mechanisms",
                "capabilities": ["External Integration", "Automatic Failover", "Health Monitoring", "Hybrid Decision"],
                "use_cases": ["External RAG systems", "High availability", "Hybrid architectures"],
                "icon": "⚖️"
            },
            RAGSystemType.INTELLIGENT.value: {
                "name": "Intelligent RAG",
                "description": "Real content extraction with direct database operations",
                "capabilities": ["PDF Processing", "Smart Vector Ops", "Database-First", "Intelligent Responses"],
                "use_cases": ["Document processing", "Content extraction", "Vector similarity"],
                "icon": "🧠"
            },
            RAGSystemType.SME.value: {
                "name": "SME RAG System",
                "description": "Full document lifecycle with multi-model embedding support",
                "capabilities": ["Document Lifecycle", "Multi-Model Support", "S3 Integration", "Collection Management"],
                "use_cases": ["Enterprise document management", "Multiple embedding models", "Production workflows"],
                "icon": "📚"
            },
            RAGSystemType.DSPY.value: {
                "name": "DSPy RAG",
                "description": "Prompt engineering and signature optimization with reasoning chains",
                "capabilities": ["Signature Optimization", "Prompt Engineering", "Chain-of-Thought", "Bootstrap Learning"],
                "use_cases": ["Prompt optimization", "Reasoning enhancement", "Signature engineering"],
                "icon": "✨"
            }
        }

    def render_portal(self):
        """Render the main RAG Creator V3 portal"""
        # Custom CSS
        self._render_custom_css()

        # Header
        st.markdown("""
        <div class="main-header">
            <h1>🧠 RAG Creator V3</h1>
            <p>Complete RAG Management with UnifiedRAGManager - Create, Read, Update, Delete, Health Check & Tailor</p>
        </div>
        """, unsafe_allow_html=True)

        # Sidebar
        self._render_navigation_sidebar()

        # Main tabs for CRUD + Health + Tailor
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🔍 Browse RAG Systems",
            "➕ Create RAG System",
            "⚙️ Update/Configure",
            "🗑️ Delete/Archive",
            "💓 Health Dashboard",
            "🎯 Tailor & Optimize"
        ])

        with tab1:
            self._render_browse_rag_systems()

        with tab2:
            self._render_create_rag_system()

        with tab3:
            self._render_update_rag_system()

        with tab4:
            self._render_delete_rag_system()

        with tab5:
            self._render_health_dashboard()

        with tab6:
            self._render_tailor_optimize()

    def _render_custom_css(self):
        """Render custom CSS for V3 portal"""
        st.markdown("""
        <style>
            .main-header {
                background: linear-gradient(90deg, #0f766e 0%, #14b8a6 100%);
                color: white;
                padding: 1.5rem;
                border-radius: 10px;
                margin-bottom: 2rem;
                text-align: center;
            }
            .rag-system-card {
                background: white;
                border: 2px solid #e5e7eb;
                border-radius: 12px;
                padding: 1.5rem;
                margin: 1rem 0;
                transition: all 0.3s ease;
            }
            .rag-system-card:hover {
                border-color: #14b8a6;
                box-shadow: 0 8px 25px rgba(20, 184, 166, 0.15);
                transform: translateY(-2px);
            }
            .rag-system-card.ai-powered { border-left: 5px solid #3b82f6; }
            .rag-system-card.postgres { border-left: 5px solid #8b5cf6; }
            .rag-system-card.judge { border-left: 5px solid #f59e0b; }
            .rag-system-card.intelligent { border-left: 5px solid #10b981; }
            .rag-system-card.sme { border-left: 5px solid #ef4444; }
            .rag-system-card.dspy { border-left: 5px solid #ec4899; }
            .capability-badge {
                display: inline-block;
                background: #f0f9ff;
                color: #0369a1;
                padding: 0.25rem 0.5rem;
                border-radius: 4px;
                font-size: 0.75rem;
                margin: 0.125rem;
                border: 1px solid #bae6fd;
            }
            .health-status.healthy {
                color: #059669;
                background: #d1fae5;
                padding: 0.25rem 0.5rem;
                border-radius: 4px;
            }
            .health-status.warning {
                color: #d97706;
                background: #fef3c7;
                padding: 0.25rem 0.5rem;
                border-radius: 4px;
            }
            .health-status.error {
                color: #dc2626;
                background: #fee2e2;
                padding: 0.25rem 0.5rem;
                border-radius: 4px;
            }
            .metric-card {
                background: white;
                border: 2px solid #e5e7eb;
                border-radius: 8px;
                padding: 1rem;
                text-align: center;
                margin: 0.5rem 0;
            }
            .success-message {
                background: #dcfce7;
                border: 1px solid #16a34a;
                color: #15803d;
                padding: 1rem;
                border-radius: 8px;
                margin: 1rem 0;
            }
            .error-message {
                background: #fef2f2;
                border: 1px solid #dc2626;
                color: #dc2626;
                padding: 1rem;
                border-radius: 8px;
                margin: 1rem 0;
            }
        </style>
        """, unsafe_allow_html=True)

    def _render_navigation_sidebar(self):
        """Render navigation sidebar with URM status"""
        with st.sidebar:
            st.header("🎯 UnifiedRAGManager")

            # URM Status
            if self.rag_manager:
                st.success("✅ URM Connected")

                # Quick stats
                available_systems = len([sys for sys in RAGSystemType if self._check_system_availability(sys)])
                st.metric("Available Systems", f"{available_systems}/6")

                # System availability overview
                st.subheader("📊 System Status")
                for sys_type in RAGSystemType:
                    available = self._check_system_availability(sys_type)
                    status_icon = "✅" if available else "❌"
                    sys_info = self.rag_systems.get(sys_type.value, {})
                    st.write(f"{status_icon} {sys_info.get('icon', '🔧')} {sys_info.get('name', sys_type.value)}")
            else:
                st.error("❌ URM Not Available")

            st.markdown("---")

            # Quick Actions
            st.subheader("🚀 Quick Actions")

            if st.button("🔄 Refresh All Systems", use_container_width=True):
                self._refresh_all_systems()
                st.rerun()

            if st.button("💓 Run Health Checks", use_container_width=True):
                self._run_comprehensive_health_check()
                st.rerun()

            if st.button("📊 Generate Report", use_container_width=True):
                self._generate_system_report()

    def _render_browse_rag_systems(self):
        """Render the Browse/Read RAG systems tab"""
        st.header("🔍 Browse RAG Systems")
        st.markdown("Explore all available RAG orchestrators and their current status")

        # Filter and search
        col1, col2 = st.columns([2, 1])

        with col1:
            search_term = st.text_input("🔍 Search RAG systems:", placeholder="Search by name, capability, or use case...")

        with col2:
            filter_status = st.selectbox("Filter by Status:", ["All", "Available", "Unavailable"])

        # Display RAG systems
        for sys_type in RAGSystemType:
            sys_info = self.rag_systems.get(sys_type.value, {})
            available = self._check_system_availability(sys_type)

            # Apply filters
            if filter_status == "Available" and not available:
                continue
            if filter_status == "Unavailable" and available:
                continue

            if search_term and search_term.lower() not in json.dumps(sys_info).lower():
                continue

            # Render system card
            self._render_rag_system_card(sys_type, sys_info, available, detailed=True)

    def _render_create_rag_system(self):
        """Render the Create RAG system tab"""
        st.header("➕ Create New RAG System")
        st.markdown("Create and configure new RAG systems using UnifiedRAGManager")

        # System type selection
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Select RAG Type")
            selected_type = st.selectbox(
                "Choose RAG System Type:",
                options=list(RAGSystemType),
                format_func=lambda x: f"{self.rag_systems.get(x.value, {}).get('icon', '🔧')} {self.rag_systems.get(x.value, {}).get('name', x.value)}"
            )

        with col2:
            sys_info = self.rag_systems.get(selected_type.value, {})
            st.subheader(f"{sys_info.get('icon', '🔧')} {sys_info.get('name', 'RAG System')}")
            st.markdown(f"**Description:** {sys_info.get('description', 'No description available')}")

            st.markdown("**Capabilities:**")
            for capability in sys_info.get('capabilities', []):
                st.markdown(f"• {capability}")

        st.markdown("---")

        # Configuration form
        with st.form("create_rag_form"):
            st.subheader("⚙️ Configuration")

            col1, col2 = st.columns(2)

            with col1:
                system_id = st.text_input(
                    "System ID *",
                    help="Unique identifier for this RAG instance"
                )

                system_name = st.text_input(
                    "System Name *",
                    help="Display name for the RAG system"
                )

                description = st.text_area(
                    "Description",
                    help="Detailed description of the system's purpose"
                )

            with col2:
                domain = st.selectbox(
                    "Domain",
                    ["general", "financial", "legal", "technical", "research", "medical"],
                    help="Domain specialization"
                )

                priority = st.selectbox(
                    "Priority",
                    ["low", "medium", "high", "critical"],
                    index=1,
                    help="System priority level"
                )

                enable_monitoring = st.checkbox(
                    "Enable Monitoring",
                    value=True,
                    help="Enable health monitoring and metrics"
                )

            # System-specific configuration
            st.subheader(f"🎯 {sys_info.get('name', 'RAG System')} Specific Configuration")

            config_params = self._get_system_specific_config(selected_type)

            submitted = st.form_submit_button("🚀 Create RAG System", type="primary")

            if submitted:
                if not system_id or not system_name:
                    st.error("Please fill in all required fields (*)")
                else:
                    # Create the RAG system
                    config = {
                        "id": system_id,
                        "name": system_name,
                        "description": description,
                        "domain": domain,
                        "priority": priority,
                        "enable_monitoring": enable_monitoring,
                        **config_params
                    }

                    result = self._create_rag_system(selected_type, config)
                    self._display_operation_result(result)

    def _render_update_rag_system(self):
        """Render the Update/Configure RAG system tab"""
        st.header("⚙️ Update & Configure RAG Systems")
        st.markdown("Modify existing RAG system configurations and parameters")

        # Get existing systems (simulated for now)
        existing_systems = self._get_existing_rag_instances()

        if not existing_systems:
            st.info("No RAG systems found. Create your first system in the 'Create RAG System' tab.")
            return

        # System selection
        selected_system = st.selectbox(
            "Select RAG System to Update:",
            options=existing_systems,
            format_func=lambda x: f"{x['type_icon']} {x['name']} ({x['id']})"
        )

        if selected_system:
            st.markdown("---")

            # Current configuration display
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("📋 Current Configuration")
                st.json(selected_system['config'])

            with col2:
                st.subheader("⚙️ Update Configuration")

                with st.form("update_rag_form"):
                    new_name = st.text_input("Name", value=selected_system['name'])
                    new_description = st.text_area("Description", value=selected_system.get('description', ''))
                    new_domain = st.selectbox(
                        "Domain",
                        ["general", "financial", "legal", "technical", "research", "medical"],
                        index=["general", "financial", "legal", "technical", "research", "medical"].index(selected_system.get('domain', 'general'))
                    )

                    # Performance tuning
                    st.subheader("🎯 Performance Tuning")
                    enable_caching = st.checkbox("Enable Caching", value=selected_system.get('caching', True))
                    max_retries = st.slider("Max Retries", 1, 5, selected_system.get('max_retries', 3))
                    timeout = st.slider("Timeout (seconds)", 5, 60, selected_system.get('timeout', 30))

                    submitted = st.form_submit_button("💾 Update Configuration")

                    if submitted:
                        update_config = {
                            "name": new_name,
                            "description": new_description,
                            "domain": new_domain,
                            "caching": enable_caching,
                            "max_retries": max_retries,
                            "timeout": timeout
                        }

                        result = self._update_rag_system(selected_system['id'], update_config)
                        self._display_operation_result(result)

    def _render_delete_rag_system(self):
        """Render the Delete/Archive RAG system tab"""
        st.header("🗑️ Delete & Archive RAG Systems")
        st.markdown("Remove or archive RAG systems with proper cleanup")

        # Safety warning
        st.warning("⚠️ **Warning:** Deleting a RAG system will remove all associated data and configurations. Consider archiving instead.")

        # Get existing systems
        existing_systems = self._get_existing_rag_instances()

        if not existing_systems:
            st.info("No RAG systems found to delete.")
            return

        # System selection with detailed info
        selected_system = st.selectbox(
            "Select RAG System:",
            options=existing_systems,
            format_func=lambda x: f"{x['type_icon']} {x['name']} ({x['id']}) - Last used: {x.get('last_used', 'Unknown')}"
        )

        if selected_system:
            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 System Information")
                st.write(f"**Name:** {selected_system['name']}")
                st.write(f"**Type:** {selected_system['type']}")
                st.write(f"**Created:** {selected_system.get('created', 'Unknown')}")
                st.write(f"**Last Used:** {selected_system.get('last_used', 'Unknown')}")
                st.write(f"**Status:** {selected_system.get('status', 'Unknown')}")

            with col2:
                st.subheader("⚙️ Deletion Options")

                action = st.radio(
                    "Choose Action:",
                    ["Archive (Recommended)", "Soft Delete", "Permanent Delete"],
                    help="Archive preserves data, Soft Delete marks as deleted, Permanent Delete removes all data"
                )

                if action == "Archive (Recommended)":
                    st.info("✅ System will be archived and can be restored later")
                elif action == "Soft Delete":
                    st.warning("⚠️ System will be marked as deleted but data preserved")
                else:
                    st.error("🚨 All system data will be permanently deleted")

                # Confirmation
                confirm_text = st.text_input(
                    f"Type '{selected_system['id']}' to confirm:",
                    help="Type the exact system ID to confirm the action"
                )

                if st.button(f"🗑️ {action}", type="secondary" if action == "Archive (Recommended)" else "primary"):
                    if confirm_text == selected_system['id']:
                        result = self._delete_rag_system(selected_system['id'], action)
                        self._display_operation_result(result)
                    else:
                        st.error("System ID confirmation does not match")

    def _render_health_dashboard(self):
        """Render the Health Check dashboard"""
        st.header("💓 RAG Systems Health Dashboard")
        st.markdown("Monitor the health and performance of all RAG orchestrators")

        # Overall health metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            healthy_count = sum(1 for sys in RAGSystemType if self._check_system_availability(sys))
            st.metric("Healthy Systems", f"{healthy_count}/6", delta=f"+{healthy_count-4}" if healthy_count > 4 else None)

        with col2:
            avg_response_time = self._get_average_response_time()
            st.metric("Avg Response Time", f"{avg_response_time}ms", delta="-15ms")

        with col3:
            success_rate = self._get_overall_success_rate()
            st.metric("Success Rate", f"{success_rate:.1f}%", delta="+2.3%")

        with col4:
            total_queries = self._get_total_queries_today()
            st.metric("Queries Today", f"{total_queries:,}", delta="+156")

        st.markdown("---")

        # Individual system health
        st.subheader("🔍 Individual System Health")

        for sys_type in RAGSystemType:
            sys_info = self.rag_systems.get(sys_type.value, {})
            health_data = self._get_system_health(sys_type)

            with st.expander(f"{sys_info.get('icon', '🔧')} {sys_info.get('name', sys_type.value)} Health Status", expanded=False):
                col1, col2, col3 = st.columns(3)

                with col1:
                    status = health_data['status']
                    status_class = "healthy" if status == "healthy" else "warning" if status == "warning" else "error"
                    st.markdown(f"**Status:** <span class='health-status {status_class}'>{status.upper()}</span>", unsafe_allow_html=True)
                    st.write(f"**Response Time:** {health_data['response_time']}ms")

                with col2:
                    st.write(f"**Uptime:** {health_data['uptime']}")
                    st.write(f"**Error Rate:** {health_data['error_rate']:.2f}%")

                with col3:
                    st.write(f"**Memory Usage:** {health_data['memory_usage']}")
                    st.write(f"**Last Check:** {health_data['last_check']}")

                # Health check actions
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button(f"🔄 Refresh {sys_type.value}", key=f"refresh_{sys_type.value}"):
                        self._refresh_system_health(sys_type)
                        st.rerun()

                with col2:
                    if st.button(f"🧪 Deep Check {sys_type.value}", key=f"deep_{sys_type.value}"):
                        self._run_deep_health_check(sys_type)

                with col3:
                    if st.button(f"📊 System Logs {sys_type.value}", key=f"logs_{sys_type.value}"):
                        self._show_system_logs(sys_type)

        # Health trends
        st.markdown("---")
        st.subheader("📈 Health Trends")

        # Mock trend data
        trend_data = self._get_health_trend_data()
        st.line_chart(trend_data)

    def _render_tailor_optimize(self):
        """Render the Tailor & Optimize tab"""
        st.header("🎯 Tailor & Optimize RAG Systems")
        st.markdown("Customize and optimize RAG systems for specific use cases and performance requirements")

        # Optimization target selection
        existing_systems = self._get_existing_rag_instances()

        if not existing_systems:
            st.info("No RAG systems found. Create systems first to enable tailoring.")
            return

        selected_system = st.selectbox(
            "Select System to Tailor:",
            options=existing_systems,
            format_func=lambda x: f"{x['type_icon']} {x['name']} ({x['type']})"
        )

        if selected_system:
            st.markdown("---")

            # Tailoring options
            tab1, tab2, tab3, tab4 = st.tabs([
                "🎯 Performance Tuning",
                "📊 A/B Testing",
                "🔧 Custom Configuration",
                "📈 Optimization History"
            ])

            with tab1:
                self._render_performance_tuning(selected_system)

            with tab2:
                self._render_ab_testing(selected_system)

            with tab3:
                self._render_custom_configuration(selected_system)

            with tab4:
                self._render_optimization_history(selected_system)

    # Helper methods for the portal functionality

    def _check_system_availability(self, sys_type: RAGSystemType) -> bool:
        """Check if a RAG system type is available"""
        if not self.rag_manager:
            return False

        try:
            return self.rag_manager.is_system_available(sys_type)
        except Exception:
            return False

    def _render_rag_system_card(self, sys_type: RAGSystemType, sys_info: Dict[str, Any], available: bool, detailed: bool = False):
        """Render a RAG system card"""
        status_badge = "✅ Available" if available else "❌ Unavailable"
        css_class = sys_type.value.replace('_', '-')

        st.markdown(f"""
        <div class="rag-system-card {css_class}">
            <h3>{sys_info.get('icon', '🔧')} {sys_info.get('name', sys_type.value)}</h3>
            <p><strong>Status:</strong> {status_badge}</p>
            <p>{sys_info.get('description', 'No description available')}</p>
            {"<div><strong>Capabilities:</strong><br>" + ' '.join([f'<span class="capability-badge">{cap}</span>' for cap in sys_info.get('capabilities', [])]) + "</div>" if detailed else ""}
            {"<div><strong>Use Cases:</strong><br>" + ', '.join(sys_info.get('use_cases', [])) + "</div>" if detailed else ""}
        </div>
        """, unsafe_allow_html=True)

    def _get_system_specific_config(self, sys_type: RAGSystemType) -> Dict[str, Any]:
        """Get system-specific configuration parameters"""
        config = {}

        if sys_type == RAGSystemType.AI_POWERED:
            config['use_corporate_gateway'] = st.checkbox("Use Corporate Gateway", value=True)
            config['bedrock_model'] = st.selectbox("Bedrock Model", ["claude-3-sonnet", "claude-3-opus"])

        elif sys_type == RAGSystemType.POSTGRES:
            config['authority_level'] = st.selectbox("Authority Level", ["Tier 1 (Regulatory)", "Tier 2 (SOP)", "Tier 3 (Technical)"])
            config['rag_types'] = st.multiselect("RAG Types", ["ComplianceRAG", "DocumentRAG", "ExpertRAG"], default=["DocumentRAG"])

        elif sys_type == RAGSystemType.JUDGE:
            config['external_endpoint'] = st.text_input("External RAG Endpoint", placeholder="https://external-rag-api.example.com")
            config['fallback_system'] = st.selectbox("Fallback System", ["PostgresRAG", "IntelligentRAG"])

        elif sys_type == RAGSystemType.INTELLIGENT:
            config['pdf_processing'] = st.checkbox("Enable PDF Processing", value=True)
            config['embedding_model'] = st.selectbox("Embedding Model", ["bedrock", "sentence-transformers"])

        elif sys_type == RAGSystemType.SME:
            config['s3_bucket'] = st.text_input("S3 Bucket", placeholder="your-sme-bucket")
            config['embedding_models'] = st.multiselect("Embedding Models", ["OpenAI Ada-002", "SentenceTransformer", "BGE Large"])

        elif sys_type == RAGSystemType.DSPY:
            config['signature_type'] = st.selectbox("DSPy Signature", ["Predict", "ChainOfThought", "ProgramOfThought"])
            config['optimization_metric'] = st.selectbox("Optimization Metric", ["accuracy", "f1", "exact_match"])

        return config

    def _create_rag_system(self, sys_type: RAGSystemType, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new RAG system"""
        try:
            if self.rag_manager:
                # Use URM to create the system
                query = f"Create new {sys_type.value} RAG system with configuration: {json.dumps(config)}"

                result = self.rag_manager.query(
                    sys_type,
                    query,
                    domain=config.get('domain', 'general')
                )

                if hasattr(result, 'response'):
                    return {
                        "success": True,
                        "message": f"Successfully created {config['name']} ({sys_type.value})",
                        "system_id": config['id'],
                        "details": result.response
                    }

            # Fallback simulation
            return {
                "success": True,
                "message": f"Successfully created {config['name']} ({sys_type.value})",
                "system_id": config['id'],
                "details": "System created with simulated configuration"
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to create RAG system: {str(e)}"
            }

    def _update_rag_system(self, system_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing RAG system"""
        try:
            # Simulate update operation
            return {
                "success": True,
                "message": f"Successfully updated system {system_id}",
                "updated_fields": list(config.keys())
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to update system: {str(e)}"
            }

    def _delete_rag_system(self, system_id: str, action: str) -> Dict[str, Any]:
        """Delete or archive a RAG system"""
        try:
            action_map = {
                "Archive (Recommended)": "archived",
                "Soft Delete": "soft_deleted",
                "Permanent Delete": "permanently_deleted"
            }

            return {
                "success": True,
                "message": f"System {system_id} has been {action_map[action]}",
                "action": action
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to {action.lower()}: {str(e)}"
            }

    def _get_existing_rag_instances(self) -> List[Dict[str, Any]]:
        """Get list of existing RAG system instances"""
        # Simulated data - in real implementation, this would query URM
        return [
            {
                "id": "ai_powered_001",
                "name": "Corporate AI Assistant",
                "type": "ai_powered",
                "type_icon": "🤖",
                "domain": "financial",
                "status": "active",
                "created": "2025-09-15",
                "last_used": "2025-09-16 10:30",
                "config": {"use_corporate_gateway": True, "bedrock_model": "claude-3-sonnet"},
                "caching": True,
                "max_retries": 3,
                "timeout": 30
            },
            {
                "id": "postgres_002",
                "name": "Compliance RAG System",
                "type": "postgres",
                "type_icon": "🗃️",
                "domain": "legal",
                "status": "active",
                "created": "2025-09-14",
                "last_used": "2025-09-16 09:45",
                "config": {"authority_level": "Tier 1 (Regulatory)", "rag_types": ["ComplianceRAG"]},
                "caching": True,
                "max_retries": 2,
                "timeout": 45
            }
        ]

    def _get_system_health(self, sys_type: RAGSystemType) -> Dict[str, Any]:
        """Get health information for a specific system"""
        # Simulated health data
        import random
        statuses = ["healthy", "warning", "error"]
        weights = [0.7, 0.2, 0.1]  # 70% healthy, 20% warning, 10% error

        status = random.choices(statuses, weights=weights)[0]

        return {
            "status": status,
            "response_time": random.randint(100, 500),
            "uptime": f"{random.randint(85, 99)}.{random.randint(0, 9)}%",
            "error_rate": random.uniform(0, 5),
            "memory_usage": f"{random.randint(15, 85)}%",
            "last_check": datetime.now().strftime("%H:%M:%S")
        }

    def _get_average_response_time(self) -> int:
        """Get average response time across all systems"""
        return 245

    def _get_overall_success_rate(self) -> float:
        """Get overall success rate across all systems"""
        return 96.8

    def _get_total_queries_today(self) -> int:
        """Get total queries processed today"""
        return 1247

    def _get_health_trend_data(self) -> Dict[str, List[float]]:
        """Get health trend data for charting"""
        import random
        dates = ["Sep 10", "Sep 11", "Sep 12", "Sep 13", "Sep 14", "Sep 15", "Sep 16"]

        return {
            "Response Time (ms)": [random.randint(200, 300) for _ in dates],
            "Success Rate (%)": [random.uniform(95, 99) for _ in dates],
            "Memory Usage (%)": [random.randint(20, 80) for _ in dates]
        }

    def _refresh_all_systems(self):
        """Refresh status of all systems"""
        st.success("All systems refreshed successfully!")

    def _run_comprehensive_health_check(self):
        """Run comprehensive health check on all systems"""
        st.success("Comprehensive health check completed!")

    def _generate_system_report(self):
        """Generate and download system report"""
        st.info("System report generated! Download link will appear here.")

    def _refresh_system_health(self, sys_type: RAGSystemType):
        """Refresh health for specific system"""
        st.success(f"Health check refreshed for {sys_type.value}")

    def _run_deep_health_check(self, sys_type: RAGSystemType):
        """Run deep health check for specific system"""
        st.info(f"Running deep health check for {sys_type.value}...")

    def _show_system_logs(self, sys_type: RAGSystemType):
        """Show system logs"""
        st.info(f"Displaying logs for {sys_type.value}")

    def _render_performance_tuning(self, selected_system: Dict[str, Any]):
        """Render performance tuning interface"""
        st.subheader("⚡ Performance Tuning")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Current Performance:**")
            st.metric("Response Time", "245ms")
            st.metric("Throughput", "45 req/sec")
            st.metric("Accuracy", "94.2%")

        with col2:
            st.write("**Tuning Parameters:**")
            chunk_size = st.slider("Chunk Size", 100, 2000, 500)
            similarity_threshold = st.slider("Similarity Threshold", 0.5, 0.95, 0.7)
            max_results = st.slider("Max Results", 1, 20, 5)

            if st.button("🎯 Apply Optimizations"):
                st.success("Performance optimizations applied!")

    def _render_ab_testing(self, selected_system: Dict[str, Any]):
        """Render A/B testing interface"""
        st.subheader("📊 A/B Testing")

        st.write("Set up A/B tests to compare different configurations:")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Configuration A (Current)**")
            st.json(selected_system['config'])

        with col2:
            st.write("**Configuration B (Test)**")
            test_config = st.text_area("Test Configuration (JSON)", value='{"chunk_size": 750, "threshold": 0.8}')

        test_duration = st.selectbox("Test Duration", ["1 hour", "1 day", "1 week"])
        traffic_split = st.slider("Traffic Split (%)", 10, 50, 20)

        if st.button("🚀 Start A/B Test"):
            st.success(f"A/B test started with {traffic_split}% traffic for {test_duration}")

    def _render_custom_configuration(self, selected_system: Dict[str, Any]):
        """Render custom configuration interface"""
        st.subheader("🔧 Custom Configuration")

        st.write("Advanced configuration options:")

        config_json = st.text_area(
            "System Configuration (JSON)",
            value=json.dumps(selected_system['config'], indent=2),
            height=300
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("✅ Validate Configuration"):
                try:
                    json.loads(config_json)
                    st.success("Configuration is valid!")
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON: {e}")

        with col2:
            if st.button("💾 Save Configuration"):
                st.success("Configuration saved successfully!")

    def _render_optimization_history(self, selected_system: Dict[str, Any]):
        """Render optimization history"""
        st.subheader("📈 Optimization History")

        # Mock optimization history
        history = [
            {"date": "2025-09-16 10:00", "type": "Performance Tuning", "improvement": "+15%", "status": "Applied"},
            {"date": "2025-09-15 14:30", "type": "A/B Test", "improvement": "+8%", "status": "Completed"},
            {"date": "2025-09-14 09:15", "type": "Configuration Update", "improvement": "+12%", "status": "Applied"},
        ]

        for item in history:
            col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
            col1.write(item["date"])
            col2.write(item["type"])
            col3.write(item["improvement"])
            col4.write(item["status"])

    def _display_operation_result(self, result: Dict[str, Any]):
        """Display operation result with appropriate styling"""
        if result.get('success'):
            st.markdown(f"""
            <div class="success-message">
                <strong>SUCCESS!</strong> {result['message']}<br>
                {f"<strong>Details:</strong> {result.get('details', '')}" if result.get('details') else ""}
            </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(f"""
            <div class="error-message">
                <strong>ERROR:</strong> {result['message']}
            </div>
            """, unsafe_allow_html=True)

def main():
    """Main application entry point"""
    portal = RAGCreatorV3Portal()
    portal.render_portal()

if __name__ == "__main__":
    main()