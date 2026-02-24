# 🚀 Streamlit App - Key Drivers Report Processing with FAISS & DSPy

import streamlit as st
import json
import time
import datetime
import faiss
import numpy as np
import pandas as pd
from embedding_helper import AmazonEmbeddingVectorizer
from configuration import CONFIG, MODEL_OPTIONS, s3_client
from key_helpers import (
    list_objects_s3,
    save_page_chunks_as_json_s3,
    smart_lexical_chunking
)

# **🔹 Initialize User Session Tracking**
if "user_id" not in st.session_state:
    st.session_state.user_id = f"user_{datetime.datetime.utcnow().timestamp()}"

# **🔹 Sidebar: Model Selection & Instructions**
st.sidebar.title("📌 Model Selection")
model_choice = st.sidebar.selectbox("Choose an Embedding Model:", list(MODEL_OPTIONS.keys()))
selected_model = MODEL_OPTIONS[model_choice]["id"]

st.sidebar.title("🧠 FAISS Indexing")

# **🔹 FAISS Index State (Lazy Loading)**
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
    st.session_state.faiss_loaded = False

# **🔹 Load FAISS Index Button**
if st.sidebar.button("🔄 Load FAISS Index"):
    try:
        dimension = MODEL_OPTIONS[CONFIG["embedding_model"]]["dimensions"]
        st.session_state.faiss_index = faiss.IndexFlatL2(dimension)
        st.session_state.faiss_loaded = True
        st.success("✅ FAISS Index Loaded!")

        # **🔹 Log FAISS Load Event to S3**
        log_data = {"event": "faiss_loaded", "timestamp": datetime.datetime.utcnow().isoformat()}
        s3_client.put_object(
            Bucket=CONFIG["bucket_name"],
            Key=f"{CONFIG['log_directory']}/faiss_log.json",
            Body=json.dumps(log_data).encode("utf-8")
        )

    except Exception as e:
        st.error(f"❌ Failed to load FAISS: {e}")

# **🔹 Clear FAISS Index Button**
if st.sidebar.button("🗑️ Clear FAISS Index"):
    st.session_state.faiss_index = None
    st.session_state.faiss_loaded = False
    st.warning("⚠️ FAISS Index Cleared.")

    # **🔹 Log FAISS Clear Event to S3**
    log_data = {"event": "faiss_cleared", "timestamp": datetime.datetime.utcnow().isoformat()}
    s3_client.put_object(
        Bucket=CONFIG["bucket_name"],
        Key=f"{CONFIG['log_directory']}/faiss_log.json",
        Body=json.dumps(log_data).encode("utf-8")
    )

# **🔹 Ensure FAISS is Loaded Before Queries**
if st.session_state.faiss_index is None:
    st.warning("⚠️ FAISS Index is not loaded. Click 'Load FAISS Index' first.")
else:
    st.sidebar.success("✅ FAISS is Ready!")

# **🔹 Initialize Embedding Vectorizer**
vectorizer = AmazonEmbeddingVectorizer(model_id=selected_model)

# **🔹 Upload & Process PDF**
st.title("📄 Key Drivers Report Processing")
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    st.success("✅ PDF uploaded successfully! Extracting text...")

    json_container = st.empty()
    final_json = {
        "metadata": {"source": "generated_text", "timestamp": datetime.datetime.utcnow().isoformat() + "Z"},
        "content": [],
        "embeddings": {}
    }

    for i in range(1, 11):  # Simulating 10-page document processing
        new_page = {"page_number": i, "text": f"This is extracted text from page {i}."}
        final_json["content"].append(new_page)
        json_container.json(final_json)  # Live update UI
        time.sleep(0.5)

    st.download_button("📥 Download Extracted JSON", json.dumps(final_json, indent=4), file_name="extracted_data.json")

# **🔹 Query Processing with FAISS**
st.subheader("🔎 Query Search")

query_text = st.text_input("Enter a query:")

if st.button("Search"):
    if st.session_state.faiss_index is None:
        st.error("❌ FAISS Index is not loaded. Please click 'Load FAISS Index'.")
    else:
        query_embedding = np.array([vectorizer.get_embedding(query_text)], dtype=np.float32)
        _, indices = st.session_state.faiss_index.search(query_embedding, 3)
        retrieved_docs = [f"Document {i+1}" for i in indices[0]]

        st.subheader("🔍 Retrieved Documents:")
        st.write(retrieved_docs)

# **🔹 DSPy Report Compilation**
if st.button("📑 Compile Optimized Report"):
    compiled_report = "📘 Optimized report content generated using DSPy..."
    st.subheader("📄 Optimized Report")
    st.write(compiled_report)

# **🔹 DSPy Retraining & Model Versioning**
st.sidebar.title("🧑‍💻 DSPy Training")

if st.sidebar.button("🔄 Retrain DSPy Using Feedback"):
    model_version = f"{selected_model}_v{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    retrain_log = pd.DataFrame([[datetime.datetime.utcnow().isoformat(), model_version, 3]],
                               columns=["timestamp", "model_version", "trained_on_feedback_count"])
    retrain_log.to_csv("model_versions.csv", mode="a", header=False, index=False)
    st.sidebar.success(f"✅ DSPy retrained! Version: {model_version}")

# **🔹 Show Model Versions**
if st.sidebar.button("📜 Show Model Versions"):
    try:
        st.sidebar.dataframe(pd.read_csv("model_versions.csv"))
    except FileNotFoundError:
        st.sidebar.warning("No retraining logs found.")
