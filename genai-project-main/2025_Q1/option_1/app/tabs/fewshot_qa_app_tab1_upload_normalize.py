
import streamlit as st
import pandas as pd
from core.normalize_labels import normalize_label

def tab_upload_normalize():
    st.header("📤 Upload and Normalize Labeled Examples")
    file = st.file_uploader("Upload Labeled Examples (JSON)", type="json")
    if file:
        df = pd.read_json(file)
        st.write("Raw Uploaded Data", df)
        if "validation_result" in df.columns:
            df["normalized_label"] = df["validation_result"].apply(normalize_label)
            st.success("Labels normalized.")
            st.write(df)
            st.download_button("Download Normalized", df.to_json(orient="records"), file_name="normalized_examples.json")
