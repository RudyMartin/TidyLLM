#!/usr/bin/env python3
"""
TidyLLM-{Domain} Streamlit Demo

Business-friendly interface for {domain} analysis using TidyLLM utilities.
Demonstrates practical application of educational ML with real business value.
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import the application
try:
    from tidyllm_{domain} import {MainClass}, {BusinessAnalyzer}
    from tidyllm_{domain}.analysis.business_intelligence import generate_portfolio_summary
    PACKAGE_AVAILABLE = True
except ImportError as e:
    st.error(f"Package import failed: {e}")
    PACKAGE_AVAILABLE = False

def main():
    """Main Streamlit application."""
    
    st.set_page_config(
        page_title="TidyLLM-{Domain} Demo",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 TidyLLM-{Domain} Demo")
    st.markdown("### Educational ML with Business Intelligence")
    
    if not PACKAGE_AVAILABLE:
        st.error("Package not available. Please install tidyllm-{domain} first.")
        return
    
    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
        st.session_state.results = []
    
    # Sidebar controls
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Configuration options
        quality_threshold = st.slider("Quality Threshold", 0.0, 1.0, 0.7, 0.1)
        educational_mode = st.checkbox("Educational Mode", value=True, 
                                     help="Show algorithm explanations and transparency")
        business_focus = st.checkbox("Business Focus", value=True,
                                   help="Emphasize business insights over technical details")
        
        # Initialize analyzer
        if st.button("🚀 Initialize Analyzer", type="primary"):
            with st.spinner("Initializing {domain} analyzer..."):
                try:
                    st.session_state.analyzer = {MainClass}(
                        quality_threshold=quality_threshold,
                        educational_mode=educational_mode,
                        business_focus=business_focus
                    )
                    
                    # Show capabilities
                    capabilities = st.session_state.analyzer.get_capabilities()
                    st.success("✅ Analyzer ready!")
                    
                    with st.expander("📋 Analyzer Capabilities"):
                        for key, value in capabilities.items():
                            if key == 'utilities_available':
                                st.write(f"**{key}:**")
                                for util, available in value.items():
                                    status = "✅" if available else "❌"
                                    st.write(f"  {status} {util}")
                            else:
                                st.write(f"**{key}:** {value}")
                
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        
        # Show current configuration
        if st.session_state.analyzer:
            st.subheader("📊 Current Settings")
            st.metric("Quality Threshold", f"{quality_threshold:.1f}")
            st.metric("Educational Mode", "On" if educational_mode else "Off")
            st.metric("Business Focus", "On" if business_focus else "Off")
    
    # Main interface
    if not st.session_state.analyzer:
        st.info("👈 Click 'Initialize Analyzer' in the sidebar to start")
        
        # Show example of what the application does
        st.subheader("🎯 What This Application Does")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Core Functionality:**
            - {Domain}-specific analysis using TidyLLM utilities
            - Educational transparency in all algorithms
            - Business-friendly insights and recommendations
            - Production-ready processing capabilities
            """)
        
        with col2:
            st.markdown("""
            **Business Intelligence:**
            - Quality scoring and assessment
            - ROI and efficiency metrics  
            - Risk identification and recommendations
            - Portfolio-level analysis and reporting
            """)
        
        # Example use cases
        st.subheader("📝 Example Use Cases")
        examples = [
            "Analyze {domain} data for quality and business value",
            "Generate business recommendations based on {domain} patterns",
            "Process multiple items efficiently with batch analysis",
            "Create portfolio summaries for stakeholder reporting"
        ]
        
        for i, example in enumerate(examples, 1):
            st.write(f"{i}. {example}")
        
        return
    
    # Analysis interface
    st.subheader("🔍 {Domain} Analysis")
    
    # Input methods
    input_method = st.radio(
        "Choose input method:",
        ["Single Item", "Multiple Items", "File Upload"],
        horizontal=True
    )
    
    if input_method == "Single Item":
        # Single item analysis
        user_input = st.text_area(
            "Enter data to analyze:",
            placeholder="Enter your {domain} data here...",
            height=100
        )
        
        if user_input and st.button("🔍 Analyze", type="primary"):
            with st.spinner("Analyzing..."):
                try:
                    result = st.session_state.analyzer.analyze(user_input)
                    st.session_state.results = [result]
                    
                    # Display results
                    display_single_result(result, educational_mode)
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {e}")
    
    elif input_method == "Multiple Items":
        # Batch analysis
        items_text = st.text_area(
            "Enter multiple items (one per line):",
            placeholder="Item 1\nItem 2\nItem 3...",
            height=150
        )
        
        if items_text:
            items = [item.strip() for item in items_text.split('\n') if item.strip()]
            st.write(f"Found {len(items)} items to analyze")
            
            if st.button("🔍 Analyze All", type="primary"):
                with st.spinner(f"Analyzing {len(items)} items..."):
                    try:
                        results = st.session_state.analyzer.batch_analyze(items)
                        st.session_state.results = results
                        
                        # Display batch results
                        display_batch_results(results, educational_mode)
                        
                    except Exception as e:
                        st.error(f"❌ Batch analysis failed: {e}")
    
    else:  # File Upload
        st.info("File upload functionality - implement based on your {domain} requirements")
        uploaded_file = st.file_uploader("Choose a file", type=['txt', 'csv', 'json'])
        
        if uploaded_file:
            st.write(f"File uploaded: {uploaded_file.name}")
            # Implement file processing based on your domain needs
    
    # Portfolio analysis
    if st.session_state.results:
        st.subheader("📊 Portfolio Analysis")
        
        if st.button("📈 Generate Portfolio Summary"):
            with st.spinner("Generating portfolio analysis..."):
                try:
                    business_analyzer = {BusinessAnalyzer}()
                    summary = business_analyzer.generate_portfolio_summary(st.session_state.results)
                    
                    st.markdown(summary)
                    
                except Exception as e:
                    st.error(f"❌ Portfolio analysis failed: {e}")

def display_single_result(result, educational_mode=True):
    """Display results for single item analysis."""
    
    st.subheader("📄 Analysis Results")
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Quality Score", f"{result.quality_score}/100")
    
    with col2:
        primary_metric = list(result.primary_metrics.values())[0] if result.primary_metrics else 0
        st.metric("Primary Metric", f"{primary_metric:.3f}")
    
    with col3:
        st.metric("Recommendations", len(result.recommendations))
    
    # Business assessment
    st.subheader("💼 Business Assessment")
    st.info(result.business_assessment)
    
    # Recommendations
    if result.recommendations:
        st.subheader("💡 Recommendations")
        for i, rec in enumerate(result.recommendations, 1):
            st.write(f"{i}. {rec}")
    
    # Educational details
    if educational_mode:
        with st.expander("🎓 Educational Details"):
            st.subheader("Primary Metrics")
            for key, value in result.primary_metrics.items():
                st.write(f"**{key}:** {value}")
            
            st.subheader("Processing Metadata")
            for key, value in result.processing_metadata.items():
                st.write(f"**{key}:** {value}")

def display_batch_results(results, educational_mode=True):
    """Display results for batch analysis."""
    
    st.subheader("📊 Batch Analysis Results")
    
    # Summary statistics
    quality_scores = [r.quality_score for r in results]
    avg_quality = sum(quality_scores) / len(quality_scores)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Items", len(results))
    
    with col2:
        st.metric("Average Quality", f"{avg_quality:.1f}")
    
    with col3:
        high_quality = sum(1 for score in quality_scores if score >= 80)
        st.metric("High Quality Items", f"{high_quality}/{len(results)}")
    
    with col4:
        total_recommendations = sum(len(r.recommendations) for r in results)
        st.metric("Total Recommendations", total_recommendations)
    
    # Quality distribution
    st.subheader("📈 Quality Distribution")
    
    # Create simple distribution display
    quality_ranges = {"Excellent (90-100)": 0, "Good (70-89)": 0, "Fair (50-69)": 0, "Poor (<50)": 0}
    
    for score in quality_scores:
        if score >= 90:
            quality_ranges["Excellent (90-100)"] += 1
        elif score >= 70:
            quality_ranges["Good (70-89)"] += 1
        elif score >= 50:
            quality_ranges["Fair (50-69)"] += 1
        else:
            quality_ranges["Poor (<50)"] += 1
    
    for range_name, count in quality_ranges.items():
        percentage = (count / len(results)) * 100
        st.write(f"**{range_name}:** {count} items ({percentage:.1f}%)")
    
    # Individual results
    with st.expander("📋 Individual Results"):
        for i, result in enumerate(results, 1):
            st.write(f"**Item {i}** - Quality: {result.quality_score}/100 - {result.business_assessment}")
            if result.recommendations:
                st.write(f"  Recommendations: {'; '.join(result.recommendations)}")
            st.divider()

if __name__ == "__main__":
    main()