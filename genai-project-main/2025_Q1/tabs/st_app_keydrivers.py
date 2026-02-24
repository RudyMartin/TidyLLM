import streamlit as st
import fitz  # PyMuPDF for PDF text extraction
import faiss
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import os

# Import necessary configurations and helpers
from embedding_helper import AmazonEmbeddingVectorizer
from configuration import CONFIG, MODEL_OPTIONS, s3_client
from s3_pdf_helper import list_pdf_files_s3, list_folders_s3

# Import DSPy Modules
from dsp import BootstrapFewShot, DataAgent, Compiler

# Initialize DSPy Optimizers
query_optimizer = BootstrapFewShot(model="gpt-4")
compiler = Compiler()

# Initialize Embedding Vectorizer
vectorizer = AmazonEmbeddingVectorizer()

# Initialize FAISS index
dimension = MODEL_OPTIONS[CONFIG["DEFAULT_MODEL"]]["dimensions"]
index = faiss.IndexFlatL2(dimension)

# Log file
LOG_FILE = "query_logs.csv"

# Ensure log file exists
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["query", "retrieved_docs", "latency"]).to_csv(LOG_FILE, index=False)

# DSPy Query Optimization Agent
query_refinement_agent = DataAgent(
    name="DSPyQueryRefiner",
    description="Refines search queries to enhance retrieval from FAISS.",
    examples=[
        {"input": "What are the revenue drivers?", "output": "Retrieve financial performance data."},
        {"input": "Summarize cost reduction trends", "output": "Find reports on expense reductions."}
    ],
    optimizer=query_optimizer
)

# DSPy Report Improvement Agent
report_refinement_agent = DataAgent(
    name="DSPyReportRefiner",
    description="Refines extracted project reports using DSPy-enhanced retrieval.",
    examples=[
        {"input": "Fix missing budget details", "output": "Updated report with financial data included."},
        {"input": "Correct project timeline inconsistencies", "output": "Revised deadlines to align with records."}
    ],
    optimizer=query_optimizer
)

# Function: Upload PDF to S3 using CONFIG["bucket_name"] and CONFIG["prefix"]
def upload_pdf_to_s3(uploaded_file):
    bucket_name = CONFIG["bucket_name"]
    prefix = CONFIG["prefix"]
    file_path = f"{prefix}/{uploaded_file.name}"
    
    s3_client.upload_fileobj(uploaded_file, bucket_name, file_path)
    return file_path

# Function: Extract text from a PDF
def extract_text_from_pdf(pdf_stream):
    doc = fitz.open(stream=pdf_stream.read(), filetype="pdf")
    text = "\n".join([page.get_text() for page in doc])
    return text

# Function: Generate Embeddings using the prebuilt vectorizer
def generate_embedding(text):
    return vectorizer.get_embedding(text)

# Function: Store embeddings in FAISS
def add_to_faiss(embedding):
    embedding_array = np.array([embedding], dtype=np.float32)
    index.add(embedding_array)

# Function: Retrieve documents from FAISS
def retrieve_similar_documents(query_text, top_k=3):
    refined_query = query_refinement_agent.run(input=query_text)  # DSPy Query Optimization
    query_embedding = np.array([generate_embedding(refined_query)], dtype=np.float32)
    
    start = time.time()
    _, indices = index.search(query_embedding, top_k)
    end = time.time()

    retrieved_docs = [f"Document {i+1}" for i in indices[0]]
    log_query_performance(query_text, retrieved_docs, end - start)

    return retrieved_docs

# Function: Log query performance
def log_query_performance(query_text, retrieved_docs, latency):
    log_df = pd.read_csv(LOG_FILE)
    new_entry = pd.DataFrame([{"query": query_text, "retrieved_docs": retrieved_docs, "latency": latency}])
    log_df = pd.concat([log_df, new_entry], ignore_index=True)
    log_df.to_csv(LOG_FILE, index=False)

# Function: Plot query latency
def plot_query_times():
    log_df = pd.read_csv(LOG_FILE)
    plt.figure(figsize=(8, 5))
    plt.plot(log_df["query"], log_df["latency"], marker='o', linestyle='-', color='b')
    plt.xlabel("Query")
    plt.ylabel("Latency (s)")
    plt.title("Query Latency Over Time")
    plt.xticks(rotation=45)
    plt.grid()
    st.pyplot(plt)

# Streamlit UI
st.title("FAISS + DSPy AI-Powered PDF Query System")

# **Option 1: Upload a New PDF**
uploaded_file = st.file_uploader("Upload a new PDF", type="pdf")
if uploaded_file:
    st.success("PDF uploaded successfully!")
    file_path = upload_pdf_to_s3(uploaded_file)
    st.write(f"File saved in S3: `{file_path}`")

    extracted_text = extract_text_from_pdf(uploaded_file)
    st.text_area("Extracted Text", extracted_text[:500] + "...", height=150)

    if st.button("Generate Embeddings"):
        embedding = generate_embedding(extracted_text)
        add_to_faiss(embedding)
        st.success("Embeddings stored in FAISS!")

# **Option 2: Select a PDF from S3**
st.subheader("Select an Existing PDF from S3")
folders = list_folders_s3(CONFIG["bucket_name"], CONFIG["prefix"])
selected_folder = st.selectbox("Choose a folder:", folders) if folders else None

if selected_folder:
    pdf_files = list_pdf_files_s3(CONFIG["bucket_name"], selected_folder)
    selected_pdf = st.selectbox("Choose a PDF:", pdf_files) if pdf_files else None

    if selected_pdf and st.button("Process Selected PDF"):
        st.write(f"Processing `{selected_pdf}` from S3...")
        
        extracted_text = f"Extracted content of {selected_pdf} (simulation)"
        st.text_area("Extracted Text", extracted_text[:500] + "...", height=150)

        if st.button("Generate Embeddings for Selected PDF"):
            embedding = generate_embedding(extracted_text)
            add_to_faiss(embedding)
            st.success("Embeddings stored in FAISS!")

# **Query Input**
query_text = st.text_input("Enter a query:")
if st.button("Search"):
    retrieved_docs = retrieve_similar_documents(query_text)
    refined_report = report_refinement_agent.run(input="Refine report using: " + " ".join(retrieved_docs))  # DSPy Report Refinement
    st.subheader("Retrieved Documents:")
    st.write(retrieved_docs)
    st.subheader("Refined Report:")
    st.write(refined_report)

# **Show Query Performance Logs**
if st.button("Show Query Performance"):
    st.dataframe(pd.read_csv(LOG_FILE))

# **Show Latency Chart**
if st.button("Show Latency Chart"):
    plot_query_times()
