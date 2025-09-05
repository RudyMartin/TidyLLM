"""
Dashboard Tab for Topic Accuracy

This tab provides a comprehensive dashboard for viewing topic accuracy and evaluation results.
"""

import streamlit as st
import sys
import os

# Add backend to path for imports
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

try:
    from backend.core.normalize_labels import normalize_label
except ImportError:
    # Fallback for deployment environment
    from core.normalize_labels import normalize_label

try:
    from backend.core.report_export import export_dashboard_to_pdf
except ImportError:
    # Fallback for deployment environment
    from core.report_export import export_dashboard_to_pdf

def tab_evaluation_dashboard():
    st.header("📊 Dashboard (Topic Accuracy)")
    
    # Upload evaluation results
    file = st.file_uploader("Upload Evaluation Results", type="json")
    
    if file:
        try:
            import json
            import pandas as pd
            
            data = json.load(file)
            df = pd.DataFrame(data)
            
            # Display basic statistics
            st.subheader("📈 Evaluation Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Examples", len(df))
            
            with col2:
                if "validation_result" in df.columns:
                    accuracy = (df["validation_result"] == "Correct").mean()
                    st.metric("Accuracy", f"{accuracy:.2%}")
            
            with col3:
                if "topic" in df.columns:
                    unique_topics = df["topic"].nunique()
                    st.metric("Unique Topics", unique_topics)
            
            # Display topic-wise accuracy
            if "topic" in df.columns and "validation_result" in df.columns:
                st.subheader("🎯 Topic-wise Accuracy")
                topic_accuracy = df.groupby("topic")["validation_result"].apply(
                    lambda x: (x == "Correct").mean()
                ).sort_values(ascending=False)
                
                st.bar_chart(topic_accuracy)
                
                # Display detailed topic breakdown
                st.write("Detailed Topic Breakdown:")
                st.dataframe(topic_accuracy.reset_index().rename(
                    columns={"validation_result": "accuracy"}
                ))
            
            # Export functionality
            if st.button("Export Dashboard to PDF"):
                try:
                    pdf_path = export_dashboard_to_pdf(df, "topic_accuracy_dashboard.pdf")
                    st.success(f"Dashboard exported to {pdf_path}")
                except Exception as e:
                    st.error(f"Export failed: {str(e)}")
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure the file is a valid JSON with evaluation results.")
