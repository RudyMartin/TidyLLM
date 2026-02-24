# 🚀 st_app_keydrivers.py - Streamlit App for Key Drivers Report Processing

import streamlit as st
import numpy as np
import pandas as pd
import json
import time
import datetime
import faiss
import boto3
import os
from embedding_helper import EmbeddingVectorizer
from configuration import CONFIG, MODEL_OPTIONS, s3_client
from key_helpers import (
    list_objects_s3,
    save_page_chunks_as_json_s3,
    smart_lexical_chunking
)

# **🔹 Initialize FAISS Index**
dimension = CONFIG["embedding_dimension"]
index = faiss.IndexFlatL2(dimension)

# **🔹 Sidebar - Model Selection & Rollback**
st.sidebar.title("Model & Embedding Options")
model_choice = st.sidebar.selectbox("Choose an Embedding Model:", list(MODEL_OPTIONS.keys()))
vectorizer = EmbeddingVectorizer(model_choice=model_choice)

# **🔹 Load Model History for Rollback**
model_log_file = "model_versions.csv"
try:
    model_history = pd.read_csv(model_log_file)
    model_versions = model_history["model_version"].unique().tolist()
except FileNotFoundError:
    model_versions = []

# **🔹 Allow Reverting to Previous Model Versions**
st.sidebar.subheader("Revert Model Version")
if model_versions:
    selected_version = st.sidebar.selectbox("Select a model version:", model_versions)
    if st.sidebar.button("Revert to Selected Version"):
        model_choice = selected_version.split("_")[0]  # Extract base model name
        vectorizer = EmbeddingVectorizer(model_choice=model_choice)
        st.session_state.session_history.append(f"Reverted to model: {selected_version}")
        st.success(f"✅ Reverted to {selected_version}")
else:
    st.sidebar.write("No previous models found.")

# **🔹 Main App Tabs**
tabs = st.tabs(["Upload & Process", "Query Search", "Model Retraining", "Instructions", "Contact Us"])

# **🔹 Upload & Process PDF Tab**
with tabs[0]:
    st.header("Upload & Process PDF")

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file:
        st.success("✅ PDF uploaded successfully! Extracting text...")
        log_to_s3("Uploaded a PDF")

# **🔹 Query Search Tab**
with tabs[1]:
    st.header("Search Queries in Processed Data")
    query_text = st.text_input("Enter a query:")

    if st.button("Search"):
        query_start = time.time()
        query_embedding = np.array([vectorizer.get_embedding(query_text)], dtype=np.float32)
        query_latency = time.time() - query_start

        st.session_state.analytics_data = pd.concat([
            st.session_state.analytics_data,
            pd.DataFrame({
                "timestamp": [datetime.datetime.utcnow()],
                "queries": [1],
                "latency": [query_latency]
            })
        ])

        st.success(f"✅ Query Processed in {query_latency:.3f} seconds")

# **🔹 Instructions Tab**
with tabs[3]:
    st.header("App Instructions")

    try:
        with open("instructions.json", "r") as f:
            instructions = json.load(f)
        for section in instructions["sections"]:
            st.subheader(section["title"])
            st.write(section["content"])
    except FileNotFoundError:
        st.warning("Instructions file missing.")

# **🔹 Contact Us Tab**
with tabs[4]:
    st.header("Contact Us")
    st.write("For any questions or support, reach out to:")
    st.write("📞 **Rudy Alvarez Martin**")
    st.write("📱 617-869-4992")
    st.write("🔗 [LinkedIn](https://linkedin.com/in/RudyMartin)")
    st.write("🏢 **Next Shift Consulting**")
