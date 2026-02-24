# 🚀 st_app_keydrivers.py - Multi-Model FAISS-Based Report Processing with DSPy & S3 Persistence

import streamlit as st
import numpy as np
import pandas as pd
import json
import time
import datetime
import faiss
import boto3
import os
import io
from embedding_helper import EmbeddingVectorizer
from dspy import BootstrapFewShot, Compiler
from configuration import CONFIG, MODEL_OPTIONS, s3_client
from key_helpers import (
    list_objects_s3,
    save_page_chunks_as_json_s3,
    smart_lexical_chunking
)

# **🔹 Sidebar - Model Selection & Rollback**
st.sidebar.title("Model & Embedding Options")
model_choice = st.sidebar.selectbox("Choose an Embedding Model:", list(MODEL_OPTIONS.keys()))
selected_model_id = MODEL_OPTIONS[model_choice]["id"]
selected_dimension = MODEL_OPTIONS[model_choice]["dimensions"] or CONFIG["embedding_dimension"]
vectorizer = EmbeddingVectorizer(model_choice=model_choice)

# **🔹 Define FAISS Index Storage Key in S3**
faiss_s3_key = f"{CONFIG['index_folder']}/{model_choice}_{selected_dimension}_faiss_index.bin"

# **🔹 Initialize FAISS Index**
index = faiss.IndexFlatL2(selected_dimension)

# **🔹 Load FAISS Index from S3 (If Available)**
def load_faiss_from_s3():
    """Loads FAISS index from S3 for the selected embedding model and dimension."""
    try:
        response = s3_client.get_object(Bucket=CONFIG["bucket_name"], Key=faiss_s3_key)
        index_data = response["Body"].read()
        global index
        index = faiss.deserialize_index(index, index_data)
        st.sidebar.success(f"✅ FAISS index for {model_choice} ({selected_dimension}D) loaded from S3!")
    except s3_client.exceptions.NoSuchKey:
        st.sidebar.warning(f"⚠️ No FAISS index found for {model_choice} ({selected_dimension}D). Starting fresh.")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading FAISS index: {e}")

# **🔹 Save FAISS Index to S3**
def save_faiss_to_s3():
    """Serializes and uploads FAISS index to S3 for the selected embedding model and dimension."""
    try:
        index_data = faiss.serialize_index(index)
        s3_client.put_object(Bucket=CONFIG["bucket_name"], Key=faiss_s3_key, Body=index_data)
        st.sidebar.success(f"✅ FAISS index for {model_choice} ({selected_dimension}D) saved to S3!")
    except Exception as e:
        st.sidebar.error(f"❌ Error saving FAISS index: {e}")

# **🔹 Initialize DSPy Components**
few_shot_optimizer = BootstrapFewShot(max_examples=3)
dspy_compiler = Compiler()

# **🔹 Load FAISS Index from S3 on Startup**
load_faiss_from_s3()

# **🔹 Upload & Process PDFs**
st.header("Upload & Process PDFs")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file:
    st.success("✅ PDF uploaded successfully! Extracting text...")

    extracted_text = "Extracted text from PDF..."
    embedding = vectorizer.get_embedding(extracted_text)

    # Store in FAISS Index
    index.add(np.array([embedding], dtype=np.float32))
    st.success(f"✅ Text indexed in FAISS for {model_choice} ({selected_dimension}D)!")

    # **Persist FAISS Index to S3**
    save_faiss_to_s3()

# **🔹 Query Search Using FAISS + DSPy**
st.header("Search Queries in Processed Data")
query_text = st.text_input("Enter a query:")

if st.button("Search"):
    query_start = time.time()

    # **DSPy Optimized Query Expansion**
    optimized_query = few_shot_optimizer.optimize(query_text)
    query_embedding = np.array([vectorizer.get_embedding(optimized_query)], dtype=np.float32)

    # **Search FAISS Index**
    _, indices = index.search(query_embedding, 3)
    retrieved_docs = [f"Document {i+1}" for i in indices[0] if i < len(index)]

    query_latency = time.time() - query_start
    st.success(f"✅ Query Processed in {query_latency:.3f} seconds")
    st.write("Retrieved Documents:", retrieved_docs)

# **🔹 Generate Improved Reports with DSPy**
st.header("Generate Improved Reports")
if st.button("Generate Report"):
    improved_report = dspy_compiler.compile(extracted_text)
    st.success("✅ Report Generated Successfully!")
    st.write(improved_report)
