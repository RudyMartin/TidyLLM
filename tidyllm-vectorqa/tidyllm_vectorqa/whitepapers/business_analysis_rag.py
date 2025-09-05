#!/usr/bin/env python3
"""
Business Analysis RAG - Enhanced RAG with Business Intelligence
==============================================================

Extends the existing RAG system with business-friendly analysis capabilities:
- Section Length Analysis (where authors invested effort)
- Signal-to-Noise Analysis (content quality vs fluff)

These analyses help business stakeholders quickly assess research value and ROI.
"""

import sys
from pathlib import Path

# Add analysis modules to path
analysis_dir = Path(__file__).parent / "advanced_analysis"
sys.path.append(str(analysis_dir))

# Import existing RAG system
from fixed_rag_app import FixedRAG

# Import business analysis tools
try:
    from section_length_analyzer import SectionLengthAnalyzer
    from signal_noise_analyzer import SignalNoiseAnalyzer
    from links_quality_checker import LinksQualityChecker
    ANALYSIS_AVAILABLE = True
except ImportError:
    print("WARNING: Business analysis tools not available")
    ANALYSIS_AVAILABLE = False

import streamlit as st
import json
from typing import Dict, Any, Optional

class BusinessAnalysisRAG(FixedRAG):
    """Enhanced RAG system with business intelligence capabilities."""
    
    def __init__(self):
        super().__init__()
        
        if ANALYSIS_AVAILABLE:
            self.section_analyzer = SectionLengthAnalyzer()
            self.signal_analyzer = SignalNoiseAnalyzer()
            self.links_checker = LinksQualityChecker()
        else:
            self.section_analyzer = None
            self.signal_analyzer = None
            self.links_checker = None
        
        # Business analysis cache
        self.business_analyses = {}
    
    def analyze_paper_business_value(self, paper_path: str, paper_id: str) -> Dict[str, Any]:
        """Analyze business value of a research paper."""
        if not ANALYSIS_AVAILABLE:
            return {"error": "Business analysis tools not available"}
        
        # Check cache first
        if paper_id in self.business_analyses:
            return self.business_analyses[paper_id]
        
        try:
            # Section analysis (effort investment)
            section_analysis = self.section_analyzer.analyze_document(paper_path)
            
            # Signal-noise analysis (content efficiency)
            efficiency_analysis = self.signal_analyzer.analyze_document(paper_path)
            
            # Links quality analysis (reliability assessment)
            links_analysis = self.links_checker.analyze_document(paper_path, max_link_checks=15)
            
            # Combine into business metrics
            business_metrics = {
                "effort_analysis": {
                    "primary_focus": section_analysis.research_focus,
                    "thoroughness_score": section_analysis.thoroughness_score,
                    "methodology_investment": section_analysis.methodology_depth,
                    "validation_investment": section_analysis.validation_investment,
                    "total_words": section_analysis.total_words,
                    "estimated_pages": section_analysis.total_words / 250
                },
                "efficiency_analysis": {
                    "signal_to_noise_ratio": efficiency_analysis.signal_to_noise_ratio,
                    "content_efficiency": efficiency_analysis.content_efficiency,
                    "information_density": efficiency_analysis.information_density,
                    "quality_assessment": efficiency_analysis.quality_assessment,
                    "reading_time_minutes": efficiency_analysis.total_words / 250,
                    "roi_assessment": "High" if efficiency_analysis.content_efficiency > 70 else "Medium" if efficiency_analysis.content_efficiency > 50 else "Low"
                },
                "reliability_analysis": {
                    "total_links": links_analysis.total_links,
                    "link_quality_score": links_analysis.link_quality_score,
                    "broken_links": links_analysis.bad_links,
                    "authority_score": links_analysis.authority_score,
                    "maintenance_indicator": links_analysis.maintenance_indicator,
                    "reliability_assessment": links_analysis.reliability_assessment
                },
                "business_recommendation": self._generate_business_recommendation(section_analysis, efficiency_analysis, links_analysis)
            }
            
            # Cache results
            self.business_analyses[paper_id] = business_metrics
            return business_metrics
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _generate_business_recommendation(self, section_analysis, efficiency_analysis, links_analysis) -> str:
        """Generate business recommendation based on analyses."""
        recommendations = []
        
        # Thoroughness assessment
        if section_analysis.thoroughness_score > 70:
            recommendations.append("COMPREHENSIVE: Thorough study with extensive analysis")
        elif section_analysis.thoroughness_score < 40:
            recommendations.append("BRIEF: Quick study suitable for rapid insights")
        
        # Methodology depth
        if section_analysis.methodology_depth > 30:
            recommendations.append("RIGOROUS: Strong methodological foundation")
        elif section_analysis.methodology_depth < 15:
            recommendations.append("LIMITED METHODS: May lack technical depth")
        
        # Validation strength
        if section_analysis.validation_investment > 25:
            recommendations.append("WELL-VALIDATED: Strong evidence and testing")
        elif section_analysis.validation_investment < 15:
            recommendations.append("WEAK VALIDATION: Limited experimental evidence")
        
        # Efficiency assessment
        if efficiency_analysis.signal_to_noise_ratio > 2.5:
            recommendations.append("HIGH VALUE: Dense with valuable information")
        elif efficiency_analysis.signal_to_noise_ratio < 1.0:
            recommendations.append("HIGH FILLER: Contains substantial bloat")
        
        # Reading time ROI
        reading_time = efficiency_analysis.total_words / 250
        if reading_time > 45 and efficiency_analysis.content_efficiency < 60:
            recommendations.append("TIME RISK: Long read with moderate information density")
        elif reading_time < 20 and efficiency_analysis.content_efficiency > 80:
            recommendations.append("QUICK WIN: Short, high-value read")
        
        # Links reliability assessment
        if links_analysis.total_links > 0:
            if links_analysis.bad_links > links_analysis.total_links * 0.5:
                recommendations.append("RELIABILITY RISK: High broken link rate indicates poor maintenance")
            elif links_analysis.link_quality_score >= 90 and links_analysis.authority_score >= 70:
                recommendations.append("WELL-MAINTAINED: High quality references with good maintenance")
            elif links_analysis.authority_score < 40:
                recommendations.append("LOW AUTHORITY: Limited authoritative source references")
        
        return " | ".join(recommendations) if recommendations else "STANDARD: Typical research paper"
    
    def enhanced_rag_qa(self, query: str, include_business_analysis: bool = True) -> str:
        """Enhanced RAG with business intelligence."""
        # Get standard RAG response
        standard_response = super().rag_qa(query)
        
        if not include_business_analysis or not ANALYSIS_AVAILABLE:
            return standard_response
        
        # Get the relevant papers from the standard response
        retrieved_chunks = self.retrieve(query, top_k=3)
        
        if not retrieved_chunks:
            return standard_response
        
        # Add business analysis section
        enhanced_response = standard_response + "\n\n---\n\n"
        enhanced_response += "## BUSINESS INTELLIGENCE ANALYSIS\n\n"
        
        for i, chunk in enumerate(retrieved_chunks, 1):
            paper = chunk["paper"]
            paper_path = paper.get("file_path", "")
            
            if paper_path:
                business_metrics = self.analyze_paper_business_value(paper_path, paper["id"])
                
                if "error" not in business_metrics:
                    effort = business_metrics["effort_analysis"]
                    efficiency = business_metrics["efficiency_analysis"]
                    reliability = business_metrics["reliability_analysis"]
                    
                    enhanced_response += f"**Paper {i}: {paper['title'][:50]}...**\n"
                    enhanced_response += f"- **Investment Focus:** {effort['primary_focus']} ({effort['methodology_investment']:.1f}% methodology, {effort['validation_investment']:.1f}% validation)\n"
                    enhanced_response += f"- **Content Quality:** {efficiency['quality_assessment']} (S/N: {efficiency['signal_to_noise_ratio']:.1f})\n"
                    enhanced_response += f"- **Links Quality:** {reliability['maintenance_indicator']} ({reliability['link_quality_score']:.1f}% working, {reliability['broken_links']} broken)\n"
                    enhanced_response += f"- **Reading Time:** {efficiency['reading_time_minutes']:.1f} min | **ROI:** {efficiency['roi_assessment']}\n"
                    enhanced_response += f"- **Business Rec:** {business_metrics['business_recommendation']}\n\n"
        
        return enhanced_response
    
    def get_portfolio_business_summary(self) -> str:
        """Generate business summary of the entire paper portfolio."""
        if not ANALYSIS_AVAILABLE:
            return "Business analysis tools not available."
        
        if not self.papers:
            return "No papers loaded for portfolio analysis."
        
        summary = ["# RESEARCH PORTFOLIO BUSINESS SUMMARY", ""]
        
        # Analyze all papers if not cached
        business_metrics_list = []
        for paper in self.papers:
            paper_path = paper.get("file_path", "")
            if paper_path:
                metrics = self.analyze_paper_business_value(paper_path, paper["id"])
                if "error" not in metrics:
                    business_metrics_list.append(metrics)
        
        if not business_metrics_list:
            return "No papers could be analyzed for business metrics."
        
        # Portfolio statistics
        total_papers = len(business_metrics_list)
        avg_thoroughness = sum(m["effort_analysis"]["thoroughness_score"] for m in business_metrics_list) / total_papers
        avg_efficiency = sum(m["efficiency_analysis"]["content_efficiency"] for m in business_metrics_list) / total_papers
        avg_snr = sum(m["efficiency_analysis"]["signal_to_noise_ratio"] for m in business_metrics_list) / total_papers
        total_reading_time = sum(m["efficiency_analysis"]["reading_time_minutes"] for m in business_metrics_list)
        
        summary.extend([
            f"**Portfolio Size:** {total_papers} research papers",
            f"**Average Thoroughness:** {avg_thoroughness:.1f}% (research depth)",
            f"**Average Content Efficiency:** {avg_efficiency:.1f}% (signal vs noise)",
            f"**Average Signal-to-Noise:** {avg_snr:.1f} (information density)",
            f"**Total Reading Time:** {total_reading_time:.0f} minutes ({total_reading_time/60:.1f} hours)",
            ""
        ])
        
        # Focus area distribution
        focus_areas = {}
        for metrics in business_metrics_list:
            focus = metrics["effort_analysis"]["primary_focus"]
            focus_areas[focus] = focus_areas.get(focus, 0) + 1
        
        summary.extend(["## RESEARCH FOCUS DISTRIBUTION"])
        for focus, count in sorted(focus_areas.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_papers) * 100
            summary.append(f"- **{focus}:** {count} papers ({percentage:.1f}%)")
        summary.append("")
        
        # Quality assessment
        high_quality = sum(1 for m in business_metrics_list if m["efficiency_analysis"]["signal_to_noise_ratio"] > 2.5)
        high_rigor = sum(1 for m in business_metrics_list if m["effort_analysis"]["methodology_investment"] > 30)
        well_validated = sum(1 for m in business_metrics_list if m["effort_analysis"]["validation_investment"] > 25)
        
        summary.extend([
            "## PORTFOLIO QUALITY METRICS",
            f"- **High Information Density:** {high_quality}/{total_papers} papers ({(high_quality/total_papers)*100:.1f}%)",
            f"- **Rigorous Methodology:** {high_rigor}/{total_papers} papers ({(high_rigor/total_papers)*100:.1f}%)",
            f"- **Well Validated:** {well_validated}/{total_papers} papers ({(well_validated/total_papers)*100:.1f}%)",
            ""
        ])
        
        # ROI Assessment
        portfolio_roi = "High" if avg_efficiency > 70 else "Medium" if avg_efficiency > 50 else "Low"
        time_efficiency = "Excellent" if avg_snr > 15 else "Good" if avg_snr > 5 else "Moderate"
        
        summary.extend([
            "## BUSINESS ASSESSMENT",
            f"- **Portfolio ROI:** {portfolio_roi} (based on content efficiency)",
            f"- **Time Efficiency:** {time_efficiency} (based on information density)",
            f"- **Recommended Reading Strategy:** {'Full portfolio' if portfolio_roi == 'High' else 'Selective reading' if portfolio_roi == 'Medium' else 'Abstract/summary focus'}",
            ""
        ])
        
        return "\n".join(summary)

def main():
    """Demo enhanced business RAG."""
    st.set_page_config(
        page_title="Business Analysis RAG",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Business Analysis RAG")
    st.markdown("### RAG with Business Intelligence - Investment & Efficiency Analysis")
    
    # Initialize session state
    if 'business_rag_system' not in st.session_state:
        st.session_state.business_rag_system = None
        st.session_state.papers_loaded = False
    
    # Sidebar controls
    with st.sidebar:
        st.header("🏢 Business Controls")
        
        if st.button("🚀 Initialize Business RAG", type="primary"):
            with st.spinner("Initializing business RAG system..."):
                try:
                    st.session_state.business_rag_system = BusinessAnalysisRAG()
                    st.success("✅ Business RAG system ready!")
                    if not ANALYSIS_AVAILABLE:
                        st.warning("⚠️ Business analysis limited - advanced tools not available")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        
        if st.session_state.business_rag_system and st.button("📚 Load Papers"):
            with st.spinner("Loading papers..."):
                try:
                    if st.session_state.business_rag_system.load_papers_for_rag():
                        st.session_state.papers_loaded = True
                        st.success("✅ Papers loaded!")
                        st.metric("Papers", len(st.session_state.business_rag_system.papers))
                        st.metric("Chunks", len(st.session_state.business_rag_system.paper_chunks))
                    else:
                        st.error("❌ Failed to load papers")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        
        # Portfolio summary
        if st.session_state.business_rag_system and st.session_state.papers_loaded:
            if st.button("📈 Portfolio Business Summary"):
                with st.spinner("Generating business portfolio analysis..."):
                    summary = st.session_state.business_rag_system.get_portfolio_business_summary()
                    st.session_state.portfolio_summary = summary
    
    # Main interface
    if not st.session_state.business_rag_system:
        st.info("👈 Click 'Initialize Business RAG' to start")
        
        # Show what business analysis provides
        st.subheader("🎯 Business Intelligence Features")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Section Length Analysis:**
            - Where authors invested their effort
            - Research thoroughness assessment  
            - Methodology vs validation focus
            - Reading time estimation
            """)
        
        with col2:
            st.markdown("""
            **Signal-to-Noise Analysis:**
            - Content quality vs filler ratio
            - Information density measurement
            - Time investment ROI
            - Reading efficiency guidance
            """)
        
        return
    
    if not st.session_state.papers_loaded:
        st.info("👈 Click 'Load Papers' to continue")
        return
    
    # Portfolio summary display
    if hasattr(st.session_state, 'portfolio_summary'):
        with st.expander("📊 Portfolio Business Summary", expanded=False):
            st.markdown(st.session_state.portfolio_summary)
    
    # Query interface with business analysis
    st.subheader("❓ Ask Questions with Business Intelligence")
    
    # Business analysis toggle
    include_business = st.checkbox("Include Business Analysis", value=True, 
                                  help="Add investment and efficiency analysis to responses")
    
    # Quick business questions
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Which papers are most worth reading?"):
            st.session_state.current_query = "Which papers provide the best value and highest information density?"
        if st.button("What's the research methodology focus?"):
            st.session_state.current_query = "What methodological approaches do these papers emphasize?"
    
    with col2:
        if st.button("How thorough is the validation?"):
            st.session_state.current_query = "How well validated are the findings across these papers?"
        if st.button("What's the time investment needed?"):
            st.session_state.current_query = "Which papers require the most reading time and provide the best ROI?"
    
    # Text input
    user_query = st.text_input(
        "Or enter your business question:",
        value=st.session_state.get('current_query', ''),
        placeholder="Ask about research value, methodology depth, reading efficiency, etc."
    )
    
    if user_query and st.button("🔍 Analyze with Business Intelligence", type="primary"):
        with st.spinner("Processing with business intelligence..."):
            try:
                if include_business:
                    response = st.session_state.business_rag_system.enhanced_rag_qa(user_query, include_business_analysis=True)
                else:
                    response = st.session_state.business_rag_system.rag_qa(user_query)
                
                st.subheader("💡 Enhanced Analysis")
                st.markdown(response)
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.exception(e)

if __name__ == "__main__":
    main()