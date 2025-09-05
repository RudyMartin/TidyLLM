"""
Upload and Normalize Labeled Examples Tab

This tab handles the upload and normalization of labeled QA examples,
ensuring consistent label formats across different data sources.
"""

import streamlit as st
import pandas as pd
import sys
import os

# Add backend to path for imports
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from backend.core.normalize_labels import normalize_label

def tab_upload_normalize():
    st.header("📤 Upload and Normalize Labeled Examples")
    file = st.file_uploader("Upload Labeled Examples (JSON)", type="json")
    if file:
        try:
            df = pd.read_json(file)
            st.write("Raw Uploaded Data", df)
            
            # Check for required columns
            required_columns = ["topic", "report_chunk", "retrieved_context", "validation_result"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {missing_columns}")
                return
            
            if "validation_result" in df.columns:
                df["normalized_label"] = df["validation_result"].apply(normalize_label)
                st.success("Labels normalized.")
                st.write(df)
                st.download_button("Download Normalized", df.to_json(orient="records"), file_name="normalized_examples.json")
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure the file is a valid JSON with the required structure.")
