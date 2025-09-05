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

# Add the tidyllm-whitepapers module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tidyllm-whitepapers'))

try:
    # Import actual tidyllm-whitepapers classes
    from core import Paper, PaperCollection, papers
    from discovery import DiscoveryOperations  
    from analysis import AnalysisOperations
    from citations import CitationOperations
    from research_framework import ResearchFramework, get_demo_papers, analyze_context_collapse_types, extract_table_of_contents, extract_bibliography
    TIDYLLM_AVAILABLE = True
except ImportError as e:
    # For demo purposes, create placeholder functionality (don't show warning)
    from research_framework import ResearchFramework, get_demo_papers, analyze_context_collapse_types, extract_table_of_contents, extract_bibliography
    TIDYLLM_AVAILABLE = False

# Import backend configuration
from backend_config import get_backend_config, render_backend_sidebar

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
        ["📑 Latest Papers", "🔍 Search", "📊 Analysis", "⚙️ Config", "📚 About Y=R+S+N"],
        key="navigation_tab"
    )
    
    # Show status when modules are available
    if TIDYLLM_AVAILABLE:
        st.sidebar.success("✅ TidyLLM modules loaded")
    else:
        st.sidebar.warning("⚠️ Demo mode - limited functionality")
    
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
    elif selected_tab == "⚙️ Config":
        show_config_page()
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
        search_query = st.text_input(
            "Research Query:",
            value="mathematical decomposition signal noise separation",
            help="Enter terms related to signal decomposition, Y=R+S+N models, etc."
        )
        
        paper_source = st.selectbox(
            "Paper Source:",
            ["ArXiv", "Local Files", "Both"],
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
    if st.button("🔍 Search Papers", type="primary"):
        if TIDYLLM_AVAILABLE:
            search_papers_real(search_query, paper_source, analysis_depth)
        else:
            search_papers(search_query, paper_source, analysis_depth)
    
    # Recent searches
    st.markdown("---")
    st.subheader("🕒 Recent Searches")
    recent_queries = [
        "Y=R+S+N mathematical framework",
        "signal noise decomposition",
        "context collapse prevention", 
        "residual risk analysis"
    ]
    
    for query in recent_queries:
        if st.button(f"↻ {query}", key=f"recent_{query}"):
            if TIDYLLM_AVAILABLE:
                search_papers_real(query, "ArXiv", 3)
            else:
                search_papers(query, "ArXiv", 3)

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

def search_papers_real(query, source, depth):
    """Search for papers using actual tidyllm-whitepapers functionality"""
    st.header("🔍 Real TidyLLM Search Results")
    
    with st.spinner("Searching papers using tidyllm-whitepapers..."):
        try:
            # Use actual tidyllm-whitepapers functionality
            discovery = DiscoveryOperations()
            
            # Create a paper collection with the query
            collection = papers(query)
            
            # Apply discovery operations
            if source in ["ArXiv", "Both"]:
                collection = collection | discovery.arxiv(limit=min(10, depth * 2))
            
            st.success(f"✅ Found {len(collection.papers)} papers using real tidyllm-whitepapers")
            
            if collection.papers:
                st.markdown("### Results:")
                for i, paper in enumerate(collection.papers[:5]):
                    with st.expander(f"📄 {paper.title[:50]}..."):
                        st.markdown(f"""
                        **Authors**: {', '.join(paper.authors)}
                        **ArXiv ID**: {paper.arxiv_id}
                        **Categories**: {', '.join(paper.categories)}
                        **Published**: {paper.published_date}
                        
                        **Abstract**: {paper.abstract[:300]}...
                        """)
            else:
                st.info("No papers found. Try a different query.")
                
        except Exception as e:
            st.error(f"Real search failed: {e}")
            st.info("Falling back to demo mode")
            search_papers(query, source, depth)

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