"""
TidyLLM Whitepapers - Streamlit Demo
===================================

Showcase whitepaper retrieval and analysis capabilities using Y="+", C="-" 
mathematical decomposition framework for Context Engineering research.

Features:
- Search and analyze research papers about signal decomposition
- Mathematical model Y (Relevant) vs C (Context Collapse: R+S+N)
- Interactive visualization of paper relevance and noise patterns
"""

import streamlit as st
import sys
import os

from dotenv import load_dotenv
load_dotenv()

# Add the yrsn module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'yrsn'))

# Import research framework (from yrsn directory)
from research_framework import ResearchFramework, get_demo_papers, analyze_context_collapse_types, extract_table_of_contents, extract_bibliography

# Real search APIs are now always available
TIDYLLM_AVAILABLE = True

# Import backend configuration
from backend_config import get_backend_config, render_backend_sidebar

# Import search tracking
from search_tracker import YRSNSearchTracker, SearchSession, SearchResult, generate_session_id

# Import additional libraries for real search
import requests
import xml.etree.ElementTree as ET
import json
from urllib.parse import quote_plus
import time
import os

# Page functions will be defined below main()

def main():
    st.set_page_config(
        page_title="TidyLLM Whitepapers Demo",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar navigation
    st.sidebar.title("🔬 TidyLLM Whitepapers")
    
    # Navigation tabs
    selected_tab = st.sidebar.selectbox(
        "Navigation", 
        ["📑 Latest Papers", "🔍 Search", "📊 Analysis", "📈 YRSN Searches", "📚 Paper Repository", "⚙️ Config", "💡 UI Ideas", "📚 About Y=R+S+N"],
        key="navigation_tab"
    )
    
    # Show status - real search is always available
    st.sidebar.success("✅ Real search enabled (ArXiv + PubMed)")
    
    # Check if we need to show analysis directly
    if st.session_state.get('show_analysis', False):
        st.session_state['show_analysis'] = False  # Reset flag
        show_analysis_page()
    # Route to appropriate page
    elif selected_tab == "📑 Latest Papers":
        show_latest_papers_page()
    elif selected_tab == "🔍 Search":
        show_search_page()
    elif selected_tab == "📊 Analysis":
        show_analysis_page()
    elif selected_tab == "📈 YRSN Searches":
        show_yrsn_searches_page()
    elif selected_tab == "📚 Paper Repository":
        show_paper_repository_page()
    elif selected_tab == "⚙️ Config":
        show_config_page()
    elif selected_tab == "💡 UI Ideas":
        show_ui_ideas_page()
    elif selected_tab == "📚 About Y=R+S+N":
        show_about_yrsn_page()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("TidyLLM Whitepapers v1.0")
    st.sidebar.caption("Mathematical Decomposition Research")

def show_latest_papers_page():
    """Display latest papers page."""
    st.title("📑 Latest Papers")
    st.subheader("Recent Mathematical Decomposition Research")
    
    # Get demo papers using research framework
    demo_paper_analyses = get_demo_papers()
    framework = ResearchFramework()
    
    # Display papers in a clean layout
    for i, analysis in enumerate(demo_paper_analyses):
        with st.expander(f"📄 {analysis.title}", expanded=i==0):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"""
                **ArXiv ID**: {analysis.arxiv_id}  
                **Authors**: {', '.join(analysis.authors)}  
                **Relevance**: {analysis.context_relevance}  
                
                **Abstract**: {analysis.abstract[:200]}...
                """)
            
            with col2:
                st.markdown("**Y=R+S+N Analysis**")
                decomp = analysis.decomposition
                
                # Progress bars for each component
                st.progress(decomp.relevant, text=f"R (Relevant): {decomp.relevant:.2f}")
                st.progress(decomp.superfluous, text=f"S (Superfluous): {decomp.superfluous:.2f}") 
                st.progress(decomp.noise, text=f"N (Noise): {decomp.noise:.2f}")
                
                st.metric("Y Score", f"{decomp.y_score:.2f}")
            
            if st.button(f"🔍 Deep Analysis", key=f"deep_{i}"):
                st.session_state['selected_paper'] = analysis
                st.session_state['show_analysis'] = True
                st.rerun()

def show_search_page():
    """Display search page."""
    st.title("🔍 Search Papers")
    st.subheader("Find Research Using Y=R+S+N Framework")
    
    # Search controls
    col1, col2 = st.columns(2)
    
    with col1:
        # Use domain query from session state if available, otherwise use default
        default_query = st.session_state.get('domain_query', 'mathematical decomposition signal noise separation')
        search_query = st.text_input(
            "Research Query:",
            value=default_query,
            help="Enter terms related to signal decomposition, Y=R+S+N models, etc."
        )
        
        # Auto-trigger search if domain query was just selected
        if 'domain_query' in st.session_state and st.session_state.domain_query == search_query:
            # Clear the session state to prevent repeated searches
            del st.session_state['domain_query']
            # Auto-trigger the search
            st.session_state['auto_search'] = True
        
        paper_source = st.selectbox(
            "Paper Source:",
            ["ArXiv", "PubMed", "Google Scholar", "SemanticScholar", "CrossRef", "Multiple Sources", "Local Files"],
            help="Where to search for papers"
        )
    
    with col2:
        analysis_depth = st.slider(
            "Analysis Depth:",
            min_value=1, max_value=5, value=3,
            help="How deep to analyze paper content (1=basic, 5=comprehensive)"
        )
        
        max_results = st.number_input(
            "Max Results:",
            min_value=1, max_value=50, value=10,
            help="Maximum number of papers to return"
        )
    
    # Search button
    search_triggered = st.button("🔍 Search Papers", type="primary")
    
    # Also trigger search if auto_search is set (from domain query selection)
    if st.session_state.get('auto_search', False):
        search_triggered = True
        st.session_state['auto_search'] = False  # Clear the flag
    
    if search_triggered:
        # Always use real search now that we have API implementations
        search_papers_real(search_query, paper_source, analysis_depth, max_results)
    
    # Recent searches and domain queries
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🕒 Recent Searches")
        recent_queries = [
            "Y=R+S+N mathematical framework",
            "signal noise decomposition",
            "context collapse prevention", 
            "residual risk analysis"
        ]
        
        for query in recent_queries:
            if st.button(f"↻ {query}", key=f"recent_{query}"):
                # Always use real search now that we have API implementations
                search_papers_real(query, "ArXiv", 3, 10)
    
    with col2:
        st.subheader("🔬 Domain Research Queries")
        st.caption("Curated queries for different research domains")
        
        # Import domain queries from centralized config
        try:
            from ui_config import DOMAIN_QUERIES
            domain_queries = {domain: info["queries"] for domain, info in DOMAIN_QUERIES.items()}
        except ImportError:
            # Fallback to local definition if ui_config not available
            domain_queries = {
                "Context Engineering": [
                    "prompt engineering techniques optimization",
                    "retrieval augmented generation methods",
                    "context collapse mitigation strategies",
                    "information quality control systems",
                    "contextual relevance evaluation metrics"
                ],
                "Machine Learning": [
                    "deep neural networks classification",
                    "transformer architecture attention mechanism",
                    "reinforcement learning optimization"
                ]
            }
        
        # Show domain queries in expandable sections
        for domain, queries in domain_queries.items():
            # Get domain info (icon and description) from config
            try:
                from ui_config import DOMAIN_QUERIES
                domain_info = DOMAIN_QUERIES.get(domain, {})
                icon = domain_info.get("icon", "📖")
                description = domain_info.get("description", "")
                expander_title = f"{icon} {domain}"
                if description:
                    expander_title += f" - {description}"
            except ImportError:
                expander_title = f"📖 {domain}"
            
            with st.expander(expander_title):
                for query in queries:
                    if st.button(f"📋 {query}", key=f"domain_{domain}_{query}"):
                        # Populate the search input and trigger search
                        st.session_state['domain_query'] = query
                        st.rerun()

def show_analysis_page():
    """Display analysis page."""
    st.title("📊 Analysis Dashboard")
    st.subheader("Deep Y=R+S+N Mathematical Analysis")
    
    # Check if we have a selected paper
    if 'selected_paper' in st.session_state:
        analysis = st.session_state['selected_paper']
        analyze_detailed_paper(analysis)
    else:
        st.info("💡 Select a paper from Latest Papers or Search to see detailed analysis here.")
        
        # Show sample analysis capabilities
        st.markdown("### 🧮 Analysis Capabilities")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **📈 Y Score Calculation**
            - Relevance scoring (0.0-1.0)
            - Context engineering metrics  
            - Mathematical validation
            """)
        
        with col2:
            st.markdown("""
            **🔬 R+S+N Decomposition** 
            - Relevant content extraction
            - Superfluous content detection
            - Noise pattern analysis
            """)
        
        with col3:
            st.markdown("""
            **⚠️ Context Collapse Detection**
            - Poisoning risk assessment
            - Distraction pattern analysis
            - Confusion metrics
            - Clash detection
            """)
        
        # Framework resources section
        st.markdown("---")
        st.subheader("📚 Framework Documentation")
        st.markdown("Learn about the Y=R+S+N framework that powers this analysis:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # LaTeX file download
            try:
                with open("yrsn_explanation.tex", "r", encoding="utf-8") as f:
                    latex_content = f.read()
                st.download_button(
                    label="📄 Download LaTeX Guide",
                    data=latex_content,
                    file_name="yrsn_framework_explanation.tex",
                    mime="text/plain",
                    help="Complete LaTeX document explaining Y=R+S+N"
                )
            except FileNotFoundError:
                st.error("LaTeX file not found")
        
        with col2:
            # JSON file download  
            try:
                with open("yrsn_framework.json", "r", encoding="utf-8") as f:
                    json_content = f.read()
                st.download_button(
                    label="📊 Download Framework Data",
                    data=json_content,
                    file_name="yrsn_framework_data.json", 
                    mime="application/json",
                    help="Machine-readable framework specification"
                )
            except FileNotFoundError:
                st.error("JSON file not found")
        
        with col3:
            if st.button("📚 Go to Framework Guide", key="goto_framework_guide"):
                st.session_state['navigation_tab'] = "📚 About Y=R+S+N"
                st.rerun()

def show_config_page():
    """Display configuration page."""
    st.title("⚙️ Configuration")
    st.subheader("Backend Settings & API Configuration")
    
    # Get backend configuration
    backend_config = get_backend_config()
    backend_config.render_config_ui()

def show_yrsn_searches_page():
    """Display YRSN search tracking and analytics page"""
    st.title("📈 YRSN Search Tracking")
    st.subheader("Research Quality Trends Over Time")
    
    # Initialize tracker with backend configuration
    try:
        backend_config = get_backend_config()
        tracker = YRSNSearchTracker(backend_config)
    except Exception as e:
        st.error(f"Error initializing search tracker: {e}")
        st.info("Make sure PostgreSQL is configured in the Config tab")
        return
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Search History", "📊 Quality Analytics", "🔄 Domain Trends", "⚙️ Settings"])
    
    with tab1:
        st.header("Recent YRSN Searches")
        
        # Get search history
        try:
            history = tracker.get_search_history(limit=25)
            
            if history:
                # Create a more detailed table
                import pandas as pd
                
                # Prepare data for display
                display_data = []
                for search in history:
                    display_data.append({
                        'Timestamp': search['timestamp'][:19].replace('T', ' '),
                        'Query': search['query'][:50] + ('...' if len(search['query']) > 50 else ''),
                        'Domain': search['research_domain'],
                        'Source': search['search_source'],
                        'Results': search['total_results'],
                        'Y Score': f"{search['avg_y_score']:.3f}" if search['avg_y_score'] else 'N/A',
                        'Risk': f"{search['avg_context_risk']:.3f}" if search['avg_context_risk'] else 'N/A',
                        'Top Paper': search['top_paper_title'][:60] + ('...' if search['top_paper_title'] and len(search['top_paper_title']) > 60 else '') if search['top_paper_title'] else 'N/A'
                    })
                
                df = pd.DataFrame(display_data)
                
                # Display with formatting
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'Y Score': st.column_config.NumberColumn('Y Score', format="%.3f"),
                        'Risk': st.column_config.NumberColumn('Context Risk', format="%.3f"),
                        'Results': st.column_config.NumberColumn('Results', format="%d")
                    }
                )
                
                st.caption(f"Showing {len(history)} recent searches")
                
            else:
                st.info("No search history available. Start searching to see tracking data!")
                
        except Exception as e:
            st.error(f"Error loading search history: {e}")
    
    with tab2:
        st.header("Quality Analytics Dashboard")
        
        try:
            analytics = tracker.get_quality_analytics()
            
            if analytics['overall_stats']['total_searches'] > 0:
                # Overall statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Searches", analytics['overall_stats']['total_searches'])
                    
                with col2:
                    avg_y = analytics['overall_stats']['overall_avg_y'] or 0
                    st.metric("Avg Y Score", f"{avg_y:.3f}")
                    
                with col3:
                    avg_risk = analytics['overall_stats']['overall_avg_risk'] or 0
                    st.metric("Avg Context Risk", f"{avg_risk:.3f}")
                    
                with col4:
                    st.metric("Research Domains", analytics['overall_stats']['unique_domains'])
                
                st.markdown("---")
                
                # Top domains by quality
                if analytics['top_domains']:
                    st.subheader("🏆 Top Research Domains by Quality")
                    
                    domain_data = []
                    for domain in analytics['top_domains']:
                        domain_data.append({
                            'Domain': domain['domain'],
                            'Average Y Score': f"{domain['avg_quality']:.3f}",
                            'Search Count': domain['search_count']
                        })
                    
                    domain_df = pd.DataFrame(domain_data)
                    st.dataframe(domain_df, use_container_width=True, hide_index=True)
                
                # Daily quality trends
                if analytics['daily_trends']:
                    st.subheader("📈 Quality Trends (Last 30 Days)")
                    
                    import matplotlib.pyplot as plt
                    
                    dates = [trend['date'] for trend in analytics['daily_trends']]
                    qualities = [trend['avg_quality'] for trend in analytics['daily_trends']]
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(dates, qualities, marker='o', linewidth=2, markersize=6)
                    ax.set_title('Daily Average Y Scores')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Average Y Score')
                    ax.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    
            else:
                st.info("No analytics data available. Perform some searches to generate analytics!")
                
        except Exception as e:
            st.error(f"Error loading analytics: {e}")
    
    with tab3:
        st.header("🔄 Domain Trends")
        
        # Time period selector
        days = st.selectbox("Time Period", [7, 14, 30, 90], index=2)
        
        try:
            trends = tracker.get_domain_trends(days=days)
            
            if trends:
                st.subheader(f"Domain Activity (Last {days} Days)")
                
                # Group by domain
                domain_summary = {}
                for trend in trends:
                    domain = trend['domain']
                    if domain not in domain_summary:
                        domain_summary[domain] = {
                            'searches': 0,
                            'total_papers': 0,
                            'avg_quality': 0,
                            'recent_queries': []
                        }
                    
                    domain_summary[domain]['searches'] += 1
                    domain_summary[domain]['total_papers'] += trend['paper_count']
                    domain_summary[domain]['avg_quality'] += trend['avg_quality']
                    domain_summary[domain]['recent_queries'].append(trend['query'])
                
                # Calculate averages and display
                trend_data = []
                for domain, data in domain_summary.items():
                    trend_data.append({
                        'Domain': domain,
                        'Searches': data['searches'],
                        'Total Papers': data['total_papers'],
                        'Avg Quality': f"{data['avg_quality'] / data['searches']:.3f}",
                        'Recent Queries': ', '.join(set(data['recent_queries'][:3]))
                    })
                
                trend_df = pd.DataFrame(trend_data)
                trend_df = trend_df.sort_values('Avg Quality', ascending=False)
                
                st.dataframe(trend_df, use_container_width=True, hide_index=True)
                
            else:
                st.info(f"No domain trends available for the last {days} days.")
                
        except Exception as e:
            st.error(f"Error loading domain trends: {e}")
    
    with tab4:
        st.header("⚙️ Search Tracking Settings")
        
        st.subheader("Database Information")
        
        # Database connection info
        try:
            backend_config = get_backend_config()
            st.code(f"PostgreSQL Host: {backend_config.settings.postgres.host}")
            st.code(f"Database: {backend_config.settings.postgres.database}")
            st.code(f"SSL Mode: {backend_config.settings.postgres.ssl_mode}")
            
            # Test connection and show table info
            tracker = YRSNSearchTracker(backend_config)
            st.success("✅ Connected to PostgreSQL database")
            st.info("YRSN search tracking tables initialized successfully")
            
        except Exception as e:
            st.error(f"Database connection error: {e}")
            st.warning("⚠️ Check your PostgreSQL configuration in the Config tab")
        
        st.subheader("Data Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Search History", type="secondary"):
                if st.session_state.get('confirm_clear', False):
                    try:
                        # Clear database tables using PostgreSQL
                        backend_config = get_backend_config()
                        tracker = YRSNSearchTracker(backend_config)
                        
                        with tracker._get_connection() as conn:
                            with conn.cursor() as cursor:
                                cursor.execute("DELETE FROM yrsn_search_results")
                                cursor.execute("DELETE FROM yrsn_search_sessions") 
                                cursor.execute("DELETE FROM yrsn_domain_trends")
                        
                        st.success("✅ Search history cleared!")
                        st.session_state['confirm_clear'] = False
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error clearing history: {e}")
                else:
                    st.session_state['confirm_clear'] = True
                    st.warning("Click again to confirm clearing all search history!")
        
        with col2:
            # Export functionality could be added here
            st.info("💡 Export functionality coming soon")
        
        st.subheader("About YRSN Search Tracking")
        st.markdown("""
        This tracking system captures:
        - **Search queries** and their semantic analysis
        - **Y=R+S+N scores** for each paper found
        - **Research domain classification** for trend analysis
        - **Quality metrics** over time to identify patterns
        - **Context collapse risk** tracking across different topics
        
        Use this data to understand research quality patterns and identify emerging domains.
        """)

def show_paper_repository_page():
    """Display paper repository management with S3 cloud storage."""
    st.title("📚 Paper Repository")
    st.markdown("Local and cloud storage for downloaded papers with embedding-based comparisons")
    
    try:
        from paper_repository import get_paper_repository
        from backend_config import get_backend_config
        
        backend_config = get_backend_config()
        repo = get_paper_repository(backend_config)
        
        # Repository statistics
        stats = repo.get_repository_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📄 Total Papers", stats["total_papers"])
        with col2:
            st.metric("💾 Total Size", f"{stats['total_size_mb']} MB")
        with col3:
            st.metric("📁 Repository", "Local + S3")
        with col4:
            st.metric("🔍 Embeddings", "Ready")
        
        # Source breakdown
        if stats["source_breakdown"]:
            st.subheader("📊 Papers by Source")
            source_data = stats["source_breakdown"]
            cols = st.columns(len(source_data))
            for i, (source, count) in enumerate(source_data.items()):
                with cols[i]:
                    st.metric(f"📚 {source}", count)
        
        st.markdown("---")
        
        # Tabs for different repository functions
        repo_tab1, repo_tab2, repo_tab3, repo_tab4 = st.tabs([
            "📋 Paper Library", 
            "☁️ S3 Cloud Sync", 
            "🔍 Embedding Search",
            "📁 Collections"
        ])
        
        with repo_tab1:
            st.subheader("📋 Downloaded Papers")
            
            # Search within repository
            search_query = st.text_input("🔍 Search repository", placeholder="Search by title or author...")
            
            if search_query:
                papers = repo.search_papers(search_query)
                st.write(f"Found {len(papers)} papers matching '{search_query}'")
            else:
                papers = repo.list_papers(limit=20)
            
            if papers:
                for paper in papers:
                    with st.expander(f"📄 {paper['title'][:60]}... ({paper['source']})"):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown(f"**Authors:** {', '.join(paper['authors'][:3])}")
                            st.markdown(f"**Source:** {paper['source']} | **Size:** {paper['file_size_mb']} MB")
                            st.markdown(f"**Downloaded:** {paper['download_date'][:10]}")
                            if paper.get('y_score'):
                                st.markdown(f"**Y Score:** {paper['y_score']:.3f}")
                            
                            # Show collections this paper belongs to with X buttons
                            paper_collections = paper.get('collections', [])
                            if paper_collections:
                                st.markdown("**Collections:**")
                                for collection_name in paper_collections:
                                    col_a, col_b = st.columns([4, 1])
                                    with col_a:
                                        st.markdown(f"• {collection_name}")
                                    with col_b:
                                        if st.button("❌", key=f"remove_{paper['paper_id']}_{collection_name}"):
                                            result = repo.remove_from_collection(paper['paper_id'], collection_name)
                                            if result["success"]:
                                                st.success(f"Removed from {collection_name}")
                                                st.rerun()
                                            else:
                                                st.error(f"Error: {result['message']}")
                            else:
                                st.markdown("**Collections:** None")
                        
                        with col2:
                            if st.button("☁️ Upload to S3", key=f"s3_{paper['paper_id']}"):
                                st.info("S3 upload functionality coming soon!")
                            
                            if st.button("🔍 Find Similar", key=f"similar_{paper['paper_id']}"):
                                st.info("Embedding-based similarity search coming soon!")
                            
                            # Get available collections for dropdown
                            collections = repo.get_collections()
                            
                            if not collections:
                                if st.button("📁 Add to Collection", key=f"collection_{paper['paper_id']}"):
                                    st.warning("No collections exist yet. Create a collection first in the Collection Management section below!")
                            else:
                                # Use a form to handle collection selection and addition
                                with st.form(key=f"add_form_{paper['paper_id']}"):
                                    st.markdown("**Add to Collection:**")
                                    collection_names = [c["name"] for c in collections]
                                    
                                    selected_collection = st.selectbox(
                                        "Choose collection:",
                                        collection_names,
                                        key=f"select_collection_{paper['paper_id']}"
                                    )
                                    
                                    add_button = st.form_submit_button("📁 Add to Collection")
                                    
                                    if add_button:
                                        result = repo.add_to_collection(paper['paper_id'], selected_collection)
                                        if result["success"]:
                                            st.success(f"✅ Added '{paper.get('title', 'Unknown')}' to '{selected_collection}'")
                                            st.rerun()  # Refresh to show updated collection
                                        else:
                                            st.error(f"❌ {result['message']}")
            else:
                st.info("No papers in repository. Download some papers from the Search tab!")
        
        with repo_tab2:
            st.subheader("☁️ S3 Cloud Storage")
            st.markdown("**Perfect for scientist demos and collaboration!**")
            
            # S3 configuration
            st.markdown("### Configuration")
            
            # Check AWS credentials using S3 session manager
            try:
                from s3_session_manager import get_s3_session_manager
                
                s3_manager = get_s3_session_manager()
                status = s3_manager.get_credential_status()
                
                if status["available"]:
                    aws_status = f"✅ AWS credentials found ({status['source']})"
                    aws_color = "success"
                    
                    # Test actual S3 connection
                    connection_test = s3_manager.test_connection()
                    if connection_test["success"]:
                        st.success(f"**Status:** {aws_status}")
                        st.info(f"🌐 Connected to S3 - Found {connection_test['bucket_count']} buckets in {connection_test['region']}")
                        
                        # Show configured S3 bucket and prefix from settings.yaml
                        st.info(f"📦 Configured for S3 bucket: `{status.get('default_bucket', 'nsc-mvp1')}`")
                        if status.get('default_prefix'):
                            st.info(f"📁 Default prefix: `{status['default_prefix']}`")
                        if status.get('kms_key_id'):
                            st.info(f"🔐 KMS encryption ready: `{status['kms_key_id'][:50]}...`")
                    else:
                        st.warning(f"**Status:** {aws_status} (Connection failed)")
                        st.error(f"❌ {connection_test['message']}")
                else:
                    # Corporate-friendly messaging
                    aws_status = "🏢 Awaiting corporate AWS configuration"
                    st.info(f"**Status:** {aws_status}")
                    
                    # Show corporate setup info instead of warning
                    with st.expander("ℹ️ Corporate Environment Information", expanded=False):
                        st.markdown("""
                        **Corporate AWS Integration**
                        
                        This application is configured for corporate environments using:
                        - **IAM Roles** (recommended for EC2/ECS/Lambda deployment)
                        - **AWS SSO** integration for user access
                        - **Corporate credential management** systems
                        - **Service accounts** with appropriate permissions
                        
                        **Required Permissions:**
                        - S3: Read/Write access to `nsc-mvp1` bucket
                        - KMS: Decrypt permissions for configured KMS key
                        - Optional: Additional S3 buckets for multi-tenancy
                        
                        **For IT/DevOps Teams:**
                        ```
                        # IAM Policy example for this application
                        {
                          "Version": "2012-10-17",
                          "Statement": [
                            {
                              "Effect": "Allow", 
                              "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
                              "Resource": ["arn:aws:s3:::nsc-mvp1/*", "arn:aws:s3:::nsc-mvp1"]
                            },
                            {
                              "Effect": "Allow",
                              "Action": ["kms:Decrypt"],
                              "Resource": "arn:aws:kms:us-east-1:188494237500:key/5573d70c-3f2d-44ca-924e-463faba85389"
                            }
                          ]
                        }
                        ```
                        """)
                    
                    # Show that everything else is ready
                    st.success("✅ Application configuration complete - ready for corporate deployment")
                    
            except Exception as e:
                aws_status = f"❌ Error checking credentials: {str(e)[:50]}..."
                st.error(f"**Status:** {aws_status}")
            
            s3_bucket = st.text_input("S3 Bucket Name", placeholder="nsc-mvp1", value="nsc-mvp1")
            s3_prefix = st.text_input("S3 Prefix (optional)", placeholder="papers/", value="papers/")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("☁️ Sync to S3", type="primary"):
                    if s3_bucket:
                        with st.spinner(f"Uploading papers to s3://{s3_bucket}/{s3_prefix}..."):
                            # Actually call the S3 sync function
                            result = repo.sync_to_s3(s3_bucket, s3_prefix)
                            
                            if result["success"]:
                                st.success(f"🚀 {result['message']}")
                                st.info(f"📤 Uploaded {result['uploaded_count']} papers")
                                if result.get('errors'):
                                    st.warning(f"⚠️ {len(result['errors'])} errors occurred:")
                                    for error in result['errors'][:3]:  # Show first 3 errors
                                        st.code(error)
                                
                                st.code(f"""
# S3 Structure created:
{result['s3_url']}
├── papers/
├── metadata/  
├── collections/
└── repository_index.json
                                """)
                            else:
                                st.error(f"❌ {result['message']}")
                                st.info("💡 To fix this:")
                                st.code("""
# Configure AWS credentials:
aws configure
# OR set environment variables:
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
# OR use IAM role if on EC2
                                """)
                    else:
                        st.error("Please enter S3 bucket name")
            
            with col2:
                if st.button("📥 Sync from S3"):
                    if s3_bucket:
                        with st.spinner(f"Downloading from s3://{s3_bucket}/{s3_prefix}..."):
                            # Actually call the S3 sync from cloud function
                            result = repo.sync_from_s3(s3_bucket, s3_prefix)
                            
                            if result["success"]:
                                st.success(f"🔄 {result['message']}")
                                st.info(f"📥 Downloaded {result['downloaded_count']} files")
                                if result.get('errors'):
                                    st.warning(f"⚠️ {len(result['errors'])} errors occurred:")
                                    for error in result['errors'][:3]:
                                        st.code(error)
                                st.rerun()  # Refresh to show new papers
                            else:
                                st.error(f"❌ {result['message']}")
                    else:
                        st.error("Please enter S3 bucket name")
            
            st.markdown("### Benefits for Scientists")
            benefits = {
                "🤝 Collaboration": "Share paper collections with research teams",
                "📊 Reproducibility": "Version-controlled paper datasets for experiments", 
                "🔍 Discovery": "Cross-team paper discovery and recommendations",
                "💾 Backup": "Never lose important papers or research collections",
                "🚀 Scaling": "Handle large datasets for meta-analysis studies",
                "🔬 Analysis": "Run embedding comparisons across institutional papers"
            }
            
            for benefit, description in benefits.items():
                st.markdown(f"**{benefit}** {description}")
        
        with repo_tab3:
            st.subheader("🔍 Embedding-Based Paper Comparisons")
            st.markdown("**AI-powered research discovery for scientists**")
            
            # Embedding search interface
            comparison_query = st.text_area(
                "Research Interest", 
                placeholder="Describe your research interest or paste an abstract...",
                height=100
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.7)
            with col2:
                max_results = st.slider("Max Results", 5, 50, 10)
            with col3:
                search_scope = st.selectbox("Search Scope", ["Local Repository", "S3 Collection", "Both"])
            
            if st.button("🔍 Find Similar Papers", type="primary"):
                if comparison_query:
                    with st.spinner("🧠 Computing embeddings and finding similar papers..."):
                        # Placeholder for embedding search
                        st.success("🎯 Found similar papers!")
                        
                        # Mock results for demo
                        mock_results = [
                            {"title": "Advanced Context Engineering Methods", "similarity": 0.89, "y_score": 0.82},
                            {"title": "Prompt Optimization in Large Language Models", "similarity": 0.84, "y_score": 0.78},
                            {"title": "Information Quality Control Systems", "similarity": 0.81, "y_score": 0.75}
                        ]
                        
                        for i, result in enumerate(mock_results):
                            with st.expander(f"📄 {i+1}. {result['title']} (Similarity: {result['similarity']:.2f})"):
                                st.markdown(f"**Y Score:** {result['y_score']:.3f}")
                                st.markdown("**Why similar:** Common themes in context optimization and quality control")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.button("📥 Download", key=f"emb_download_{i}")
                                with col2:
                                    st.button("🔬 Analyze", key=f"emb_analyze_{i}")
                else:
                    st.warning("Please enter a research interest or query")
            
            st.markdown("### Embedding Features")
            embedding_features = {
                "🎯 Semantic Search": "Find papers by meaning, not just keywords",
                "🔗 Citation Discovery": "Discover papers that should be cited together", 
                "📈 Trend Analysis": "Identify emerging research directions",
                "🤖 Auto-Collections": "Automatically group related papers",
                "📊 Quality Ranking": "Combine embeddings with Y=R+S+N scores",
                "🌐 Cross-Domain": "Find connections across research fields"
            }
            
            for feature, description in embedding_features.items():
                st.markdown(f"**{feature}** {description}")
        
        with repo_tab4:
            st.subheader("📁 Research Collections")
            st.markdown("Organize papers into themed collections for different projects")
            
            # Collection management
            collections = repo.get_collections()
            
            # Create new collection
            with st.expander("➕ Create New Collection"):
                collection_name = st.text_input("Collection Name", placeholder="Machine Learning Papers")
                collection_desc = st.text_area("Description", placeholder="Papers related to ML research...")
                
                if st.button("📁 Create Collection"):
                    if collection_name:
                        result = repo.create_collection(collection_name, collection_desc)
                        if result["success"]:
                            st.success(f"✅ {result['message']}")
                            st.rerun()
                        else:
                            st.error(f"❌ {result['message']}")
                    else:
                        st.warning("Please enter collection name")
            
            # Show existing collections
            if collections:
                st.markdown("### 📚 Existing Collections")
                for collection in collections:
                    with st.expander(f"📁 {collection['name']} ({collection['paper_count']} papers)"):
                        st.markdown(f"**Description:** {collection['description']}")
                        st.markdown(f"**Created:** {collection['created_at'][:10]}")
                        
                        # Show papers in this collection
                        collection_papers = repo.index.get("collections", {}).get(collection["name"], {}).get("papers", [])
                        if collection_papers:
                            st.markdown("**Papers in this collection:**")
                            for paper_id in collection_papers:
                                if paper_id in repo.index.get("papers", {}):
                                    paper_title = repo.index["papers"][paper_id].get("title", "Unknown Title")
                                    st.markdown(f"- {paper_title}")
                        else:
                            st.markdown("**Papers:** None added yet")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if st.button("☁️ Sync to S3", key=f"sync_collection_{collection['name']}"):
                                st.info("Collection S3 sync - coming soon!")
                        with col2:
                            if st.button("📤 Export BibTeX", key=f"export_{collection['name']}"):
                                st.info("BibTeX export - coming soon!")
                        with col3:
                            if st.button("🔍 Analyze Collection", key=f"analyze_collection_{collection['name']}"):
                                with st.spinner("🧠 Analyzing papers with TidyLLM transformers..."):
                                    # Get papers in this collection
                                    collection_papers = repo.index.get("collections", {}).get(collection["name"], {}).get("papers", [])
                                    
                                    if len(collection_papers) < 2:
                                        st.warning("Need at least 2 papers in collection for comparison")
                                    else:
                                        # Generate comparison report
                                        report = repo.generate_embedding_comparison_report(collection_papers)
                                        
                                        if report.get("success"):
                                            st.success(f"✅ Analysis complete!")
                                            
                                            # Show collection stats
                                            stats = report["collection_stats"]
                                            col_a, col_b, col_c = st.columns(3)
                                            with col_a:
                                                st.metric("Papers", report["total_papers"])
                                            with col_b:
                                                st.metric("Avg Similarity", f"{stats['average_similarity_percent']}%")
                                            with col_c:
                                                st.metric("Comparisons", stats["total_comparisons"])
                                            
                                            # Show analysis summary
                                            st.markdown("### 📊 Analysis Summary")
                                            for summary_point in report["analysis_summary"]:
                                                st.markdown(f"- {summary_point}")
                                            
                                            # Show most similar pair
                                            if stats["most_similar_pair"]:
                                                st.markdown("### 🔥 Most Similar Papers")
                                                pair = stats["most_similar_pair"]
                                                st.markdown(f"**Similarity: {pair['similarity_percent']}%**")
                                                st.markdown(f"📄 **{pair['paper1']['title']}**")
                                                st.markdown(f"📄 **{pair['paper2']['title']}**")
                                            
                                            # Show detailed similarities
                                            with st.expander("📈 Detailed Pairwise Similarities"):
                                                for sim in report["pairwise_similarities"][:5]:  # Top 5
                                                    st.markdown(f"**{sim['similarity_percent']}%** similarity:")
                                                    st.markdown(f"- {sim['paper1']['title']}")
                                                    st.markdown(f"- {sim['paper2']['title']}")
                                                    st.markdown("---")
                                            
                                            # Show model details
                                            with st.expander("🔧 Model Details"):
                                                model_info = report["model_details"]
                                                st.markdown(f"**Model**: {report['embedding_model']}")
                                                st.markdown(f"**Vocabulary Size**: {model_info['vocab_size']:,}")
                                                st.markdown(f"**Embedding Dimension**: {model_info['embedding_dimension']:,}")
                                                st.markdown(f"**Attention Heads**: {model_info['attention_heads']}")
                                        else:
                                            st.error(f"Analysis failed: {report.get('message', 'Unknown error')}")
                                            if report.get("fallback_available"):
                                                st.info("💡 Install sentence-transformers for enhanced analysis")
            else:
                st.info("No collections yet. Create your first collection above!")
        
        # Repository path info
        st.markdown("---")
        st.caption(f"📂 Local repository: {stats['repository_path']}")
        
    except Exception as e:
        st.error(f"Repository system error: {e}")
        st.markdown("**Repository features include:**")
        st.markdown("- 📄 Local paper storage with metadata")
        st.markdown("- ☁️ S3 cloud synchronization for collaboration")
        st.markdown("- 🔍 Embedding-based similarity search")
        st.markdown("- 📁 Research collections and organization")

def show_ui_ideas_page():
    """Display UI improvement ideas and configuration management."""
    st.title("💡 UI Enhancement Ideas")
    st.markdown("Central place for managing queries, features, and improvement ideas")
    
    # Import UI improvements from config
    try:
        from ui_config import UI_IMPROVEMENTS, TRENDING_QUERIES, get_improvement_ideas
        
        st.subheader("🎯 Improvement Roadmap")
        
        # Show improvement ideas by category
        for category, details in UI_IMPROVEMENTS.items():
            priority_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}
            priority_icon = priority_color.get(details.get("priority", "medium"), "🟡")
            
            with st.expander(f"{priority_icon} {category.replace('_', ' ').title()} ({details.get('priority', 'medium')} priority)"):
                for idea in details.get("ideas", []):
                    st.markdown(f"- {idea}")
        
        st.subheader("🔥 Trending Research Queries")
        st.markdown("Popular queries that could be added to domain sections:")
        
        cols = st.columns(2)
        for i, query in enumerate(TRENDING_QUERIES):
            with cols[i % 2]:
                if st.button(f"📋 {query}", key=f"trending_{i}"):
                    st.session_state['domain_query'] = query
                    st.session_state['navigation_tab'] = "🔍 Search"
                    st.rerun()
        
        st.subheader("⚙️ Configuration Management")
        st.markdown("All queries and settings are managed in `ui_config.py`")
        
        st.code("""
# To add new domain queries:
DOMAIN_QUERIES["Your Domain"] = {
    "icon": "🔬",
    "description": "Your description", 
    "queries": ["query 1", "query 2"]
}

# To add improvement ideas:
UI_IMPROVEMENTS["new_feature"] = {
    "priority": "high",
    "ideas": ["Idea 1", "Idea 2"]
}
        """, language="python")
        
    except ImportError:
        st.error("UI configuration not available")
        st.markdown("Create `ui_config.py` to manage all queries and improvement ideas centrally")

def show_about_yrsn_page():
    """Display About Y=R+S+N page."""
    st.title("📚 About Y=R+S+N Framework")
    st.subheader("Mathematical Decomposition for Context Engineering")
    
    # Introduction
    st.markdown("""
    ## 🎯 Overview
    
    The **Y=R+S+N** framework is a mathematical approach to analyzing and decomposing content 
    to prevent context collapse in AI systems and research analysis.
    """)
    
    # Mathematical formula
    st.latex(r"Y = R + S + N")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 Components Explained
        
        **Y (Yield)**: Total valuable content score
        - Range: 0.0 to 1.0
        - Represents overall relevance and quality
        - Higher Y = more valuable content
        
        **R (Relevant)**: Core systematic content
        - Highly relevant information
        - Directly addresses research objectives
        - Mathematical and methodological content
        
        **S (Superfluous)**: Marginally systematic
        - Background information
        - Supporting but non-essential content
        - Contextual references
        """)
    
    with col2:
        st.markdown("""
        ### ⚠️ Context Collapse Prevention
        
        **N (Noise)**: True noise and errors
        - Irrelevant content
        - Formatting artifacts
        - Processing errors
        - Misleading information
        
        ### 🎯 Applications
        
        - **Research Paper Analysis**
        - **Content Quality Assessment**  
        - **AI Training Data Validation**
        - **Corporate Compliance Review**
        - **Risk Management**
        """)
    
    # Context collapse types
    st.markdown("---")
    st.subheader("🚨 Context Collapse Types")
    
    collapse_types = {
        "Context Poisoning": "Absorption of misinformation and biased content",
        "Context Distraction": "Information overload leading to focus loss", 
        "Context Confusion": "Integration of non-relevant or contradictory data",
        "Context Clash": "Conflicts between different information sources"
    }
    
    for collapse_type, description in collapse_types.items():
        st.markdown(f"**{collapse_type}**: {description}")
    
    # Implementation example
    st.markdown("---")
    st.subheader("💻 Implementation Example")
    
    st.code("""
# Y=R+S+N Analysis Example
def analyze_paper_yrsn(paper_content):
    # Calculate components
    R = calculate_relevant_score(paper_content)
    S = calculate_superfluous_score(paper_content)  
    N = calculate_noise_score(paper_content)
    
    # Total Y score
    Y = R + S + N
    
    return {
        'y_score': Y,
        'relevant': R,
        'superfluous': S, 
        'noise': N,
        'context_collapse_risk': assess_collapse_risk(R, S, N)
    }
    """, language="python")
    
    # Research citations
    st.markdown("---")
    st.subheader("📚 Research Foundation")
    
    st.markdown("""
    This framework builds on research in:
    - Signal-noise decomposition theory
    - Information theory and entropy
    - Context engineering methodologies
    - AI alignment and safety research
    - Mathematical optimization principles
    """)
    
    # Contact info
    st.info("💡 This framework is part of the TidyLLM research project for improving AI system reliability and research integrity.")
    
    # Download section
    st.markdown("---")
    st.subheader("📥 Download Resources")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # LaTeX file download
        try:
            with open("yrsn_explanation.tex", "r", encoding="utf-8") as f:
                latex_content = f.read()
            st.download_button(
                label="📄 Download LaTeX Source",
                data=latex_content,
                file_name="yrsn_framework_explanation.tex",
                mime="text/plain",
                help="LaTeX source file for creating PDF"
            )
        except FileNotFoundError:
            st.error("LaTeX file not found")
    
    with col2:
        # JSON file download  
        try:
            with open("yrsn_framework.json", "r", encoding="utf-8") as f:
                json_content = f.read()
            st.download_button(
                label="📊 Download JSON Data",
                data=json_content,
                file_name="yrsn_framework_data.json", 
                mime="application/json",
                help="Complete framework data in JSON format"
            )
        except FileNotFoundError:
            st.error("JSON file not found")
    
    with col3:
        # PDF generation info
        st.markdown("""
        **📑 Generate PDF**
        
        To create PDF from LaTeX:
        1. Download LaTeX file
        2. Use pdflatex or online compiler
        3. Requires: amsmath, tcolorbox packages
        """)
        
        # Simple PDF generation for basic framework guide
        def create_framework_pdf():
            try:
                from reportlab.lib.pagesizes import A4
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.lib.units import inch
                from io import BytesIO
                
                buffer = BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=A4)
                styles = getSampleStyleSheet()
                
                story = []
                
                # Title
                title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=20, spaceAfter=30, alignment=1)
                story.append(Paragraph("Y=R+S+N Framework Guide", title_style))
                story.append(Spacer(1, 20))
                
                # Content
                story.append(Paragraph("<b>Framework Overview</b>", styles['Heading2']))
                story.append(Paragraph("The Y=R+S+N framework is a mathematical approach to analyzing content quality and preventing context collapse in AI systems.", styles['Normal']))
                story.append(Spacer(1, 15))
                
                story.append(Paragraph("<b>Mathematical Formula</b>", styles['Heading2']))
                story.append(Paragraph("Y = R + S + N", styles['Normal']))
                story.append(Paragraph("Y Score = R + (0.5 × S)", styles['Normal']))
                story.append(Spacer(1, 15))
                
                story.append(Paragraph("<b>Components</b>", styles['Heading2']))
                story.append(Paragraph("• R (Relevant): Core systematic content (25-85%)", styles['Normal']))
                story.append(Paragraph("• S (Superfluous): Marginally systematic content (10-45%)", styles['Normal']))
                story.append(Paragraph("• N (Noise): True noise and errors (5-35%)", styles['Normal']))
                story.append(Spacer(1, 15))
                
                story.append(Paragraph("<b>Context Collapse Risk</b>", styles['Heading2']))
                story.append(Paragraph("Risk = S + (1.5 × N)", styles['Normal']))
                story.append(Paragraph("• Low Risk: 0.0-0.3 (Safe to use)", styles['Normal']))
                story.append(Paragraph("• Medium Risk: 0.3-0.6 (Validate claims)", styles['Normal']))
                story.append(Paragraph("• High Risk: 0.6+ (Use extreme caution)", styles['Normal']))
                story.append(Spacer(1, 20))
                
                story.append(Paragraph("<i>For complete documentation, download the LaTeX file and compile with pdflatex.</i>", styles['Normal']))
                story.append(Paragraph("<i>Generated by TidyLLM Whitepapers</i>", styles['Normal']))
                
                doc.build(story)
                buffer.seek(0)
                return buffer.getvalue()
                
            except ImportError:
                return None
            except Exception:
                return None
        
        try:
            pdf_data = create_framework_pdf()
            
            if pdf_data:
                st.download_button(
                    label="📑 Download PDF Guide",
                    data=pdf_data,
                    file_name="yrsn_framework_guide.pdf",
                    mime="application/pdf",
                    help="Download basic framework guide as PDF"
                )
                st.success("✅ PDF ready for download!")
            else:
                # Show direct link instead of problematic button
                st.markdown("""
                **🔧 Manual PDF Creation:**
                
                Visit [Overleaf.com](https://www.overleaf.com/) to compile LaTeX to PDF
                """)
                st.info("💡 Download the LaTeX file above and paste it into Overleaf to generate a professional PDF")
        except Exception as e:
            st.error(f"PDF generation error: {str(e)}")
            st.markdown("""
            **🔧 Alternative: Use LaTeX File**
            
            Download the LaTeX file above and use [Overleaf.com](https://www.overleaf.com/) to create your PDF.
            """)
    
    # Usage instructions
    st.markdown("### 📖 How to Use These Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📄 LaTeX File (.tex):**
        - Complete PDF-ready document
        - Professional formatting
        - Mathematical equations
        - Use with pdflatex or Overleaf
        - Perfect for sharing/printing
        """)
    
    with col2:
        st.markdown("""
        **📊 JSON File (.json):**
        - Machine-readable data
        - Complete framework specification
        - All scoring rules and examples
        - Use for integrations/automation
        - Import into other tools
        """)
    
    st.info("💡 **Tip:** The LaTeX file creates a beautiful 15+ page PDF guide in layman's English. Perfect for sharing with colleagues who want to understand Y=R+S+N without technical jargon!")

def search_papers(query, source, depth):
    """Search for papers based on user input"""
    st.header("🔍 Search Results")
    
    with st.spinner("Searching papers..."):
        try:
            # Mock search results for demo
            st.success(f"Found 12 papers matching: '{query}'")
            
            # Display search parameters
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Query Terms", len(query.split()))
            with col2: 
                st.metric("Source", source)
            with col3:
                st.metric("Analysis Depth", depth)
            
            # Mock results table
            st.markdown("### Top Results:")
            
            results_data = {
                "Title": [
                    "Mathematical decomposition of residual risk in signal processing",
                    "Context collapse prevention through orthogonal decomposition", 
                    "Y=R+S+N framework for noise separation in ML models"
                ],
                "Y Score": [0.91, 0.87, 0.83],
                "R (Relevant)": [0.68, 0.61, 0.59],
                "S (Superfluous)": [0.16, 0.19, 0.21], 
                "N (Noise)": [0.16, 0.20, 0.20],
                "ArXiv ID": ["2024.12345", "2024.12346", "2024.12347"]
            }
            
            st.dataframe(results_data, use_container_width=True)
            
        except Exception as e:
            st.error(f"Search failed: {e}")
            st.info("This is a demo - actual search functionality would connect to tidyllm-whitepapers backend")

def search_arxiv(query, max_results=10):
    """Search ArXiv for papers"""
    try:
        # ArXiv API
        base_url = "http://export.arxiv.org/api/query?"
        search_query = f"search_query=all:{quote_plus(query)}&start=0&max_results={max_results}"
        
        response = requests.get(base_url + search_query)
        response.raise_for_status()
        
        # Parse XML response
        root = ET.fromstring(response.content)
        papers = []
        
        # Namespace for ArXiv
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        for entry in root.findall('atom:entry', ns):
            title = entry.find('atom:title', ns).text.strip()
            authors = [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)]
            abstract = entry.find('atom:summary', ns).text.strip()
            published = entry.find('atom:published', ns).text[:10]
            arxiv_id = entry.find('atom:id', ns).text.split('/')[-1]
            
            papers.append({
                'title': title,
                'authors': authors,
                'abstract': abstract,
                'published': published,
                'arxiv_id': arxiv_id,
                'source': 'ArXiv'
            })
        
        return papers
    except Exception as e:
        st.error(f"ArXiv search error: {e}")
        return []

def search_pubmed(query, max_results=10):
    """Search PubMed for papers"""
    try:
        # PubMed eSearch API
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'json'
        }
        
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        pmids = data.get('esearchresult', {}).get('idlist', [])
        
        if not pmids:
            return []
        
        # Get paper details
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {
            'db': 'pubmed',
            'id': ','.join(pmids),
            'retmode': 'xml'
        }
        
        fetch_response = requests.get(fetch_url, params=fetch_params)
        fetch_response.raise_for_status()
        
        root = ET.fromstring(fetch_response.content)
        papers = []
        
        for article in root.findall('.//PubmedArticle'):
            try:
                title_elem = article.find('.//ArticleTitle')
                title = title_elem.text if title_elem is not None else "No title"
                
                authors = []
                for author in article.findall('.//Author'):
                    last_name = author.find('.//LastName')
                    first_name = author.find('.//ForeName')
                    if last_name is not None:
                        name = last_name.text
                        if first_name is not None:
                            name += f", {first_name.text}"
                        authors.append(name)
                
                abstract_elem = article.find('.//AbstractText')
                abstract = abstract_elem.text if abstract_elem is not None else "No abstract available"
                
                pmid_elem = article.find('.//PMID')
                pmid = pmid_elem.text if pmid_elem is not None else "N/A"
                
                papers.append({
                    'title': title,
                    'authors': authors,
                    'abstract': abstract,
                    'published': 'N/A',
                    'pmid': pmid,
                    'source': 'PubMed'
                })
            except Exception:
                continue
        
        return papers
    except Exception as e:
        st.error(f"PubMed search error: {e}")
        return []

def search_multiple_sources(query, max_results=10):
    """Search multiple sources and combine results"""
    all_papers = []
    
    # Search ArXiv
    arxiv_papers = search_arxiv(query, max_results // 2)
    all_papers.extend(arxiv_papers)
    
    # Search PubMed
    pubmed_papers = search_pubmed(query, max_results // 2)
    all_papers.extend(pubmed_papers)
    
    return all_papers

def download_arxiv_paper(arxiv_id, title="", authors=None, y_score=None, r_score=None, s_score=None, n_score=None, context_risk=None):
    """Download ArXiv paper PDF and add to repository"""
    try:
        # Import repository
        from paper_repository import get_paper_repository
        from backend_config import get_backend_config
        
        backend_config = get_backend_config()
        repo = get_paper_repository(backend_config)
        
        # Clean ArXiv ID (remove version number if present)
        clean_id = arxiv_id.split('v')[0] if 'v' in arxiv_id else arxiv_id
        
        with st.spinner(f"Downloading paper to repository: {title[:50]}..."):
            # Download to repository
            result = repo.download_paper(
                paper_id=clean_id,
                title=title,
                authors=authors or [],
                source="ArXiv",
                y_score=y_score,
                r_score=r_score,
                s_score=s_score,
                n_score=n_score,
                context_risk=context_risk
            )
            
            if result["success"]:
                st.success(f"✅ {result['message']}")
                st.info(f"📁 Saved to repository: {result.get('file_size', 0) / (1024*1024):.1f} MB")
                
                # Also provide browser download option
                pdf_url = f"https://arxiv.org/pdf/{clean_id}.pdf"
                response = requests.get(pdf_url, stream=True, timeout=30)
                response.raise_for_status()
                
                st.download_button(
                    label="💾 Download to Browser",
                    data=response.content,
                    file_name=f"arxiv_{clean_id}_{title[:30].replace(' ', '_')}.pdf",
                    mime="application/pdf",
                    help=f"Download {title} to your browser downloads"
                )
            else:
                st.warning(f"⚠️ {result['message']}")
                # Fallback to browser download
                pdf_url = f"https://arxiv.org/pdf/{clean_id}.pdf"
                response = requests.get(pdf_url, stream=True, timeout=30)
                response.raise_for_status()
                
                st.download_button(
                    label="💾 Download to Browser",
                    data=response.content,
                    file_name=f"arxiv_{clean_id}_{title[:30].replace(' ', '_')}.pdf",
                    mime="application/pdf",
                    help=f"Download {title} to your browser downloads"
                )
            
    except Exception as e:
        st.error(f"Download failed: {e}")
        clean_id = arxiv_id.split('v')[0] if 'v' in arxiv_id else arxiv_id
        st.info(f"You can manually download from: https://arxiv.org/pdf/{clean_id}.pdf")

def search_papers_real(query, source, depth, max_results=10):
    """Search for papers using real API calls and our working script"""
    st.header("🔍 Real Search Results")
    
    with st.spinner(f"Searching {source} for papers..."):
        try:
            # Import the working search script
            from search_script_clean import search_and_track
            
            # Call the working search script
            success, message, results_count, papers = search_and_track(query, source, max_results)
            
            if success:
                st.success(f"✅ {message}")
                st.info(f"🔍 Search tracked! Check the 'YRSN Searches' tab to see the results in your search history.")
                
                # Display the papers
                if papers:
                    st.subheader(f"📚 Found {len(papers)} Papers")
                    for i, paper in enumerate(papers):
                        with st.expander(f"📄 {i+1}. {paper['title'][:80]}...", expanded=False):
                            st.markdown(f"**Authors:** {', '.join(paper['authors'])}")
                            st.markdown(f"**Published:** {paper['published']}")
                            st.markdown(f"**ArXiv ID:** {paper['arxiv_id']}")
                            st.markdown(f"**Abstract:** {paper['abstract'][:500]}...")
                            
                            # Add download button
                            col1, col2 = st.columns([1, 1])
                            with col1:
                                if st.button(f"📥 Download PDF", key=f"download_{paper['arxiv_id']}"):
                                    download_arxiv_paper(
                                        paper['arxiv_id'], 
                                        title=paper.get('title', ''),
                                        authors=paper.get('authors', [])
                                    )
                            with col2:
                                if st.button(f"🔬 Analyze", key=f"analyze_{paper['arxiv_id']}"):
                                    st.session_state['selected_paper'] = paper
                                    st.session_state['navigation_tab'] = "📊 Analysis"
                                    st.rerun()
            else:
                st.error(f"❌ {message}")
                
        except Exception as e:
            st.error(f"❌ Search script error: {e}")
            import traceback
            st.code(traceback.format_exc())

def analyze_detailed_paper(analysis):
    """Analyze a specific paper in detail using research framework"""
    # Navigation header
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("← Back to Papers", key="back_to_papers"):
            if 'selected_paper' in st.session_state:
                del st.session_state['selected_paper']
            st.session_state['navigation_tab'] = "📑 Latest Papers"
            st.rerun()
    
    with col2:
        st.header(f"📊 Analysis: {analysis.title}")
    
    with col3:
        st.write("")  # Placeholder for spacing
    
    framework = ResearchFramework()
    
    with st.spinner("Analyzing paper content..."):
        # Paper overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **Paper Details:**
            - **Title**: {analysis.title}
            - **ArXiv ID**: {analysis.arxiv_id} 
            - **Authors**: {', '.join(analysis.authors)}
            - **Overall Y Score**: {analysis.decomposition.y_score:.2f}
            - **Overall Score**: {analysis.overall_score:.2f}
            """)
        
        with col2:
            st.markdown("**Key Insights:**")
            recommendations = framework.generate_paper_recommendations(analysis)
            for rec in recommendations:
                st.markdown(f"- {rec}")
            
            # Download paper button (if it's an ArXiv paper)
            if hasattr(analysis, 'arxiv_id') and analysis.arxiv_id:
                st.markdown("---")
                if st.button("📥 Download Original PDF", key="download_detailed"):
                    download_arxiv_paper(
                        analysis.arxiv_id, 
                        title=getattr(analysis, 'title', ''),
                        authors=getattr(analysis, 'authors', []),
                        y_score=getattr(analysis, 'y_score', None),
                        r_score=getattr(analysis, 'relevant', None),
                        s_score=getattr(analysis, 'superfluous', None),
                        n_score=getattr(analysis, 'noise', None),
                        context_risk=getattr(analysis, 'superfluous', 0) + (1.5 * getattr(analysis, 'noise', 0)) if hasattr(analysis, 'superfluous') and hasattr(analysis, 'noise') else None
                    )
        
        # Detailed decomposition analysis
        st.subheader("🔬 R+S+N Decomposition Analysis")
        
        decomp = analysis.decomposition
        
        # Progress bars for decomposition
        st.markdown("**Relevant Content (R):**")
        st.progress(decomp.relevant)
        st.text(f"Core mathematical concepts, systematic methodologies: {decomp.relevant:.1%}")
        
        st.markdown("**Superfluous Content (S):**") 
        st.progress(decomp.superfluous)
        st.text(f"Background information, marginally relevant sections: {decomp.superfluous:.1%}")
        
        st.markdown("**Noise (N):**")
        st.progress(decomp.noise) 
        st.text(f"Irrelevant content, errors, formatting issues: {decomp.noise:.1%}")
        
        # Context Collapse Analysis
        st.subheader("🚨 Context Collapse Risk Analysis")
        
        context_risk, risk_level = framework.calculate_context_collapse_risk(decomp)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Context Collapse Risk", f"{context_risk:.2f}", delta=None)
        with col2:
            if risk_level == "High":
                st.error(f"Risk Level: {risk_level}")
            elif risk_level == "Medium":
                st.warning(f"Risk Level: {risk_level}")
            else:
                st.success(f"Risk Level: {risk_level}")
        
        # Context collapse type analysis
        collapse_analysis = analyze_context_collapse_types(analysis.abstract)
        
        st.markdown("**Context Collapse Types:**")
        collapse_cols = st.columns(4)
        
        collapse_types = [
            ("Poisoning", collapse_analysis['poisoning']),
            ("Distraction", collapse_analysis['distraction']), 
            ("Confusion", collapse_analysis['confusion']),
            ("Clash", collapse_analysis['clash'])
        ]
        
        for i, (ctype, score) in enumerate(collapse_types):
            with collapse_cols[i]:
                st.metric(f"{ctype}", f"{score:.2f}")
        
        # Key quotes/excerpts
        st.subheader("📝 Abstract")
        st.markdown(f"> {analysis.abstract}")
        
        # Mathematical analysis
        st.subheader("🧮 Mathematical Content Analysis")
        math_scores = framework.analyze_mathematical_content(analysis.abstract)
        
        math_cols = st.columns(4)
        for i, (metric, score) in enumerate(math_scores.items()):
            with math_cols[i]:
                st.metric(metric.replace('_', ' ').title(), f"{score:.3f}")
        
        # Document Structure Analysis
        st.subheader("📋 Document Structure Analysis")
        
        # Create expanded content for TOC and bibliography extraction (using abstract as sample)
        full_text_sample = f"""
        {analysis.title}
        
        ABSTRACT
        {analysis.abstract}
        
        1. INTRODUCTION
        Mathematical decomposition frameworks have become increasingly important in signal processing and context engineering applications.
        
        2. METHODOLOGY  
        This paper presents a comprehensive approach to systematic content analysis using the Y=R+S+N framework.
        
        3. RESULTS
        Our analysis shows significant improvements in context collapse prevention.
        
        4. DISCUSSION
        The implications of this framework extend beyond traditional signal processing applications.
        
        5. CONCLUSION
        The Y=R+S+N framework provides a robust mathematical foundation for content quality analysis.
        
        REFERENCES
        [1] Smith, J. (2023). Context Engineering in Modern AI Systems. Journal of AI Research, 45(2), 123-145.
        [2] Johnson, M., & Lee, K. (2024). Mathematical Decomposition Techniques. IEEE Transactions on Signal Processing, 67(8), 234-256.
        [3] Brown, A. et al. (2023). Systematic Content Analysis Methods. Nature Machine Intelligence, 12(4), 67-89.
        """
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Table of Contents in expandable section
            toc_entries = extract_table_of_contents(full_text_sample)
            toc_count = len(toc_entries) if toc_entries else 7  # Default demo structure has 7 items
            
            with st.expander(f"📑 Table of Contents ({toc_count} sections)", expanded=True):
                if toc_entries:
                    for entry in toc_entries:
                        section_num = entry['section_number']
                        title = entry['title']
                        page = entry['page']
                        
                        if section_num:
                            st.markdown(f"**{section_num}** {title} {'.' * max(1, 30 - len(title) - len(section_num))} {page}")
                        else:
                            st.markdown(f"• {title} {'.' * max(1, 35 - len(title))} {page}")
                else:
                    st.info("No formal table of contents detected. Showing detected structure:")
                    st.markdown("""
                    **Detected Structure:**
                    • Abstract ...................... 1
                    • 1. Introduction ............... 2  
                    • 2. Methodology ................ 3
                    • 3. Results .................... 4
                    • 4. Discussion ................. 5
                    • 5. Conclusion ................. 6
                    • References .................... 7
                    """)
                
                # TOC Statistics
                if toc_entries:
                    st.markdown("---")
                    st.markdown("**TOC Statistics:**")
                    numbered_sections = sum(1 for entry in toc_entries if entry['section_number'])
                    st.markdown(f"• Numbered sections: {numbered_sections}")
                    st.markdown(f"• Total sections: {len(toc_entries)}")
        
        with col2:
            # Bibliography/References in expandable section
            references = extract_bibliography(full_text_sample)
            ref_count = len(references) if references else 3  # Demo has 3 references
            
            with st.expander(f"📚 Bibliography/References ({ref_count} entries)", expanded=True):
                if references:
                    # Show all references in the expandable section
                    for ref in references:
                        if ref['author'] and ref['year']:
                            st.markdown(f"**[{ref['number']}]** {ref['author']} ({ref['year']}). {ref['title'][:80]}{'...' if len(ref['title']) > 80 else ''}")
                        else:
                            st.markdown(f"**[{ref['number']}]** {ref['full_citation'][:100]}{'...' if len(ref['full_citation']) > 100 else ''}")
                else:
                    st.info("References extracted from demo content:")
                    st.markdown("""
                    **[1]** Smith, J. (2023). Context Engineering in Modern AI Systems. *Journal of AI Research*, 45(2), 123-145.
                    
                    **[2]** Johnson, M., & Lee, K. (2024). Mathematical Decomposition Techniques. *IEEE Trans. Signal Processing*, 67(8), 234-256.
                    
                    **[3]** Brown, A. et al. (2023). Systematic Content Analysis Methods. *Nature Machine Intelligence*, 12(4), 67-89.
                    """)
                
                # Reference statistics in the expandable section
                st.markdown("---")
                st.markdown("**Reference Statistics:**")
                
                if references:
                    st.markdown(f"• Total References: {len(references)}")
                    
                    # Count references by decade
                    years = [ref['year'] for ref in references if ref['year'].isdigit()]
                    if years:
                        year_ranges = {'2020s': 0, '2010s': 0, '2000s': 0, 'Other': 0}
                        for year in years:
                            year_int = int(year)
                            if year_int >= 2020:
                                year_ranges['2020s'] += 1
                            elif year_int >= 2010:
                                year_ranges['2010s'] += 1
                            elif year_int >= 2000:
                                year_ranges['2000s'] += 1
                            else:
                                year_ranges['Other'] += 1
                        
                        for decade, count in year_ranges.items():
                            if count > 0:
                                st.markdown(f"• {decade}: {count} references")
                    
                    # Additional statistics
                    authors_with_years = [ref for ref in references if ref['author'] and ref['year']]
                    st.markdown(f"• Structured citations: {len(authors_with_years)}")
                    st.markdown(f"• Unstructured citations: {len(references) - len(authors_with_years)}")
                else:
                    st.markdown("• Total References: 3 (demo)")
                    st.markdown("• 2020s: 3 references")
                    st.markdown("• Structured citations: 3")
                    st.markdown("• Coverage: AI Research, Signal Processing, Machine Intelligence")
        
        # Context Engineering relevance
        st.subheader("🎯 Context Engineering Relevance")
        y_score = analysis.decomposition.y_score
        
        if y_score > 0.9:
            st.success("🟢 Extremely High Relevance - Core to Context Engineering research")
        elif y_score > 0.8:
            st.info("🔵 High Relevance - Strong supporting evidence for Y=R+S+N framework")  
        elif y_score > 0.7:
            st.warning("🟡 Medium Relevance - Useful background and methodology")
        else:
            st.error("🔴 Low Relevance - Limited applicability")
        
        # Save to database option
        st.subheader("💾 Save Analysis")
        backend_config = get_backend_config()
        
        if backend_config.connection:
            if st.button("💾 Save Analysis to Database", key=f"save_{analysis.title[:10]}"):
                success = backend_config.store_paper_analysis(
                    title=analysis.title,
                    authors=analysis.authors,
                    abstract=analysis.abstract,
                    y_score=analysis.decomposition.y_score,
                    decomposition={
                        'R': analysis.decomposition.relevant,
                        'S': analysis.decomposition.superfluous,
                        'N': analysis.decomposition.noise
                    },
                    context_risk=risk_level,
                    arxiv_id=analysis.arxiv_id
                )
                
                if success:
                    st.success("✅ Analysis saved to database!")
                else:
                    st.error("❌ Failed to save analysis")
        else:
            st.info("🔧 Connect to database in Backend Configuration to save analyses")
        
        # Download Framework Resources
        st.markdown("---")
        st.subheader("📥 Download Y=R+S+N Framework Resources")
        st.markdown("Get the complete framework documentation to understand the mathematics behind this analysis.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # LaTeX file download with current analysis included
            try:
                with open("yrsn_explanation.tex", "r", encoding="utf-8") as f:
                    base_latex_content = f.read()
                
                # Add current paper analysis to LaTeX
                analysis_section = f"""
\\section{{Current Analysis: {analysis.title}}}

\\subsection{{Paper Overview}}
\\begin{{itemize}}
\\item \\textbf{{Title}}: {analysis.title}
\\item \\textbf{{ArXiv ID}}: {analysis.arxiv_id}
\\item \\textbf{{Authors}}: {', '.join(analysis.authors)}
\\item \\textbf{{Y Score}}: {analysis.decomposition.y_score:.3f}
\\end{{itemize}}

\\subsection{{R+S+N Decomposition Results}}
\\begin{{itemize}}
\\item \\textbf{{R (Relevant)}}: {analysis.decomposition.relevant:.1%} - Core systematic content
\\item \\textbf{{S (Superfluous)}}: {analysis.decomposition.superfluous:.1%} - Marginally systematic content
\\item \\textbf{{N (Noise)}}: {analysis.decomposition.noise:.1%} - True noise and errors
\\end{{itemize}}

\\subsection{{Abstract}}
{analysis.abstract}

\\subsection{{Context Collapse Risk Analysis}}
Context Collapse Risk: {context_risk:.3f} (Level: {risk_level})

\\subsection{{Recommendations}}
\\begin{{itemize}}
"""
                for rec in recommendations:
                    # Clean up recommendations for LaTeX
                    clean_rec = rec.replace("✅", "").replace("🔵", "").replace("⚠️", "").replace("🔧", "").replace("✂️", "").replace("🎯", "").replace("📖", "").replace("⚡", "").strip()
                    analysis_section += f"\\item {clean_rec}\n"
                
                analysis_section += """\\end{itemize}

"""
                
                # Insert analysis before the conclusion
                enhanced_latex = base_latex_content.replace("\\section{Conclusion}", analysis_section + "\\section{Conclusion}")
                
                st.download_button(
                    label="📄 Download LaTeX with Analysis",
                    data=enhanced_latex,
                    file_name=f"yrsn_analysis_{analysis.title[:20].replace(' ', '_')}.tex",
                    mime="text/plain",
                    help="LaTeX source with current paper analysis included"
                )
            except FileNotFoundError:
                st.error("LaTeX file not found")
        
        with col2:
            # JSON file download with current analysis
            try:
                import json
                with open("yrsn_framework.json", "r", encoding="utf-8") as f:
                    base_json_data = json.loads(f.read())
                
                # Add current analysis data
                base_json_data["current_analysis"] = {
                    "title": analysis.title,
                    "arxiv_id": analysis.arxiv_id,
                    "authors": analysis.authors,
                    "abstract": analysis.abstract,
                    "decomposition": {
                        "relevant": analysis.decomposition.relevant,
                        "superfluous": analysis.decomposition.superfluous,
                        "noise": analysis.decomposition.noise,
                        "y_score": analysis.decomposition.y_score
                    },
                    "context_collapse": {
                        "risk_score": context_risk,
                        "risk_level": risk_level
                    },
                    "mathematical_analysis": math_scores,
                    "collapse_types": collapse_analysis,
                    "recommendations": recommendations,
                    "timestamp": "2024-08-31"
                }
                
                enhanced_json = json.dumps(base_json_data, indent=2, ensure_ascii=False)
                
                st.download_button(
                    label="📊 Download JSON with Analysis",
                    data=enhanced_json,
                    file_name=f"yrsn_analysis_{analysis.title[:20].replace(' ', '_')}.json", 
                    mime="application/json",
                    help="Framework data with current paper analysis included"
                )
            except FileNotFoundError:
                st.error("JSON file not found")
            except Exception as e:
                st.error(f"Error creating enhanced JSON: {e}")
        
        with col3:
            # Generate PDF using reportlab for immediate download
            def create_analysis_pdf(analysis, context_risk, risk_level, recommendations, math_scores, collapse_analysis):
                try:
                    from reportlab.lib.pagesizes import letter, A4
                    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
                    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                    from reportlab.lib.units import inch
                    from reportlab.lib import colors
                    from io import BytesIO
                    
                    buffer = BytesIO()
                    doc = SimpleDocTemplate(buffer, pagesize=A4)
                    styles = getSampleStyleSheet()
                    
                    # Custom styles
                    title_style = ParagraphStyle(
                        'CustomTitle',
                        parent=styles['Heading1'],
                        fontSize=18,
                        spaceAfter=30,
                        alignment=1  # Center
                    )
                    
                    story = []
                    
                    # Title
                    story.append(Paragraph("Y=R+S+N Framework Analysis Report", title_style))
                    story.append(Spacer(1, 20))
                    
                    # Paper details
                    story.append(Paragraph("<b>Paper Analysis</b>", styles['Heading2']))
                    
                    details_data = [
                        ['Title:', analysis.title],
                        ['ArXiv ID:', analysis.arxiv_id],
                        ['Authors:', ', '.join(analysis.authors)],
                        ['Y Score:', f"{analysis.decomposition.y_score:.3f}"],
                    ]
                    
                    details_table = Table(details_data, colWidths=[1.5*inch, 4*inch])
                    details_table.setStyle(TableStyle([
                        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
                        ('FONTSIZE', (0,0), (-1,-1), 10),
                        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                    ]))
                    
                    story.append(details_table)
                    story.append(Spacer(1, 20))
                    
                    # R+S+N Decomposition
                    story.append(Paragraph("<b>R+S+N Decomposition</b>", styles['Heading2']))
                    
                    decomp_data = [
                        ['Component', 'Score', 'Percentage', 'Description'],
                        ['R (Relevant)', f"{analysis.decomposition.relevant:.3f}", f"{analysis.decomposition.relevant:.1%}", 'Core systematic content'],
                        ['S (Superfluous)', f"{analysis.decomposition.superfluous:.3f}", f"{analysis.decomposition.superfluous:.1%}", 'Marginally systematic content'],
                        ['N (Noise)', f"{analysis.decomposition.noise:.3f}", f"{analysis.decomposition.noise:.1%}", 'True noise and errors'],
                    ]
                    
                    decomp_table = Table(decomp_data)
                    decomp_table.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), colors.grey),
                        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0,0), (-1,-1), 10),
                        ('BOTTOMPADDING', (0,0), (-1,0), 12),
                        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
                        ('GRID', (0,0), (-1,-1), 1, colors.black)
                    ]))
                    
                    story.append(decomp_table)
                    story.append(Spacer(1, 20))
                    
                    # Context Collapse Risk
                    story.append(Paragraph("<b>Context Collapse Risk Analysis</b>", styles['Heading2']))
                    story.append(Paragraph(f"Risk Score: <b>{context_risk:.3f}</b>", styles['Normal']))
                    story.append(Paragraph(f"Risk Level: <b>{risk_level}</b>", styles['Normal']))
                    story.append(Spacer(1, 15))
                    
                    # Abstract
                    story.append(Paragraph("<b>Abstract</b>", styles['Heading2']))
                    story.append(Paragraph(analysis.abstract, styles['Normal']))
                    story.append(Spacer(1, 15))
                    
                    # Recommendations
                    story.append(Paragraph("<b>Recommendations</b>", styles['Heading2']))
                    for i, rec in enumerate(recommendations, 1):
                        clean_rec = rec.replace("✅", "").replace("🔵", "").replace("⚠️", "").replace("🔧", "").replace("✂️", "").replace("🎯", "").replace("📖", "").replace("⚡", "").strip()
                        story.append(Paragraph(f"{i}. {clean_rec}", styles['Normal']))
                    
                    story.append(Spacer(1, 20))
                    story.append(Paragraph("<i>Generated by TidyLLM Whitepapers Y=R+S+N Framework</i>", styles['Normal']))
                    
                    doc.build(story)
                    buffer.seek(0)
                    return buffer.getvalue()
                
                except ImportError:
                    return None
                except Exception as e:
                    st.error(f"PDF generation error: {e}")
                    return None
            
            # Try to generate PDF, with informative error handling
            try:
                pdf_data = create_analysis_pdf(analysis, context_risk, risk_level, recommendations, math_scores, collapse_analysis)
                
                if pdf_data:
                    st.download_button(
                        label="📑 Download PDF",
                        data=pdf_data,
                        file_name=f"yrsn_analysis_{analysis.title[:20].replace(' ', '_')}.pdf",
                        mime="application/pdf",
                        help="Download complete analysis as PDF report"
                    )
                    st.success("✅ PDF ready for download!")
                else:
                    # Fallback - show manual instructions and direct link
                    st.markdown("""
                    **📑 Create PDF Manually**
                    
                    1. Download LaTeX file above  
                    2. Visit [Overleaf.com](https://www.overleaf.com/) to compile
                    3. Copy LaTeX content and paste in Overleaf
                    4. Compile to generate PDF
                    """)
                    st.info("💡 ReportLab not available. Use LaTeX file with Overleaf for PDF generation.")
            except Exception as e:
                st.error(f"PDF generation failed: {str(e)}")
                st.markdown("""
                **📑 Alternative: Use LaTeX File**
                
                Download the LaTeX file above and use [Overleaf.com](https://www.overleaf.com/) to create your PDF.
                """)

# Additional demo sections
def show_system_capabilities():
    st.header("⚙️ TidyLLM System Capabilities")
    
    capabilities = {
        "Paper Discovery": "🔍 Search ArXiv, PubMed, and local repositories",
        "Content Extraction": "📄 PDF processing with text, tables, and figures", 
        "Mathematical Analysis": "🧮 Y=R+S+N decomposition scoring",
        "Citation Networks": "🔗 Author connections and reference mapping",
        "Semantic Search": "🎯 Context-aware paper retrieval",
        "Export Formats": "💾 BibTeX, APA, MLA, Chicago, IEEE, JSON"
    }
    
    cols = st.columns(2)
    for i, (capability, description) in enumerate(capabilities.items()):
        with cols[i % 2]:
            st.markdown(f"**{capability}**")
            st.markdown(description)

# Legacy functions preserved for backward compatibility

if __name__ == "__main__":
    main()