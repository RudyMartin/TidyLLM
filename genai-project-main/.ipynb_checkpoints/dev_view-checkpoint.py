##  streamlit app dev_view.py

!pip uninstall -y sparkmagic

!pip install -r require_dev_view_20250317.txt -q

import boto3
import faiss
import numpy as np
import pandas as pd
import os
import logging
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.llms import Bedrock
import dspy

# Configuration Settings
BUCKET_NAME = "sagemaker-us-east-1-188494237500"
SET_R_DIRECTORY = "dev/json"
FAISS_INDEX_R_PATH = "dev/idx"
INDEX_DIM = 384  # Adjusted to match 'all-MiniLM-L6-v2' embedding size

# AWS Config
s3_client = boto3.client('s3')
bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

# Load Titan Model
llm = Bedrock(model_id="amazon.titan-text-v3", client=bedrock)

# Load Embedding Model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# DSPy Configuration
pipeline = dspy.Pipeline()

# Define DSPy Signatures
class RationaleSignature(dspy.Signature):
    query: str = dspy.InputField()
    sources: list = dspy.InputField()
    ranked_response: str = dspy.OutputField()
    rationale: str = dspy.OutputField()

class ChainOfThoughtQASignature(dspy.Signature):
    query: str = dspy.InputField()
    context: str = dspy.InputField()
    reasoning_steps: str = dspy.OutputField()
    final_answer: str = dspy.OutputField()

rationale_model = dspy.Predict(RationaleSignature)
cot_qa_model = dspy.Predict(ChainOfThoughtQASignature)

# Function to Load PDFs from S3
def load_pdfs_from_s3(bucket_name, prefix=''):
    """Loads PDFs from S3 and extracts text."""
    pdf_texts = []
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        if 'Contents' not in response:
            logging.warning(f"No files found in {prefix}")
            return []

        for obj in response['Contents']:
            file_key = obj['Key']
            pdf_obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
            pdf_reader = PdfReader(pdf_obj['Body'])
            text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
            if text:
                pdf_texts.append(text)

    except Exception as e:
        logging.error(f"Error loading PDFs: {e}")

    return pdf_texts

# Load PDFs from Set R
def load_all_pdfs():
    return load_pdfs_from_s3(BUCKET_NAME, prefix=SET_R_DIRECTORY)

# Vectorization & FAISS Indexing
def create_faiss_index(texts):
    """Creates a FAISS index for the provided texts."""
    if not texts:
        logging.warning("No texts provided for indexing.")
        return None, None, None

    embeddings = embedding_model.encode(texts, convert_to_numpy=True)
    index = faiss.IndexFlatL2(INDEX_DIM)
    index.add(embeddings)
    return index, embeddings, texts

# Retrieve Relevant Chunks
def retrieve_text(query, index, texts, k=3):
    """Retrieves the top-k most relevant text chunks based on FAISS similarity search."""
    if index is None or not texts:
        logging.error("Index is not initialized.")
        return []

    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [{"text": texts[i], "source": f"Document {i}"} for i in indices[0] if i < len(texts)]

# Load Documents & Create Index
st.sidebar.header("📁 Load Documents")
if st.sidebar.button("Load Set R PDFs"):
    st.sidebar.write("Loading and indexing documents...")
    texts = load_all_pdfs()
    faiss_index, embeddings, texts = create_faiss_index(texts)
    st.sidebar.success("✅ Documents loaded and indexed.")

# Chatbot UI
st.title("📚 AI-Powered Document Chatbot")

user_input = st.text_input("Ask a question:")
category = st.selectbox("Select document set:", ["Reports (Set R)", "All"])

if user_input:
    if category == "Reports (Set R)":
        retrieved_chunks = retrieve_text(user_input, faiss_index, texts, k=3)
        sources = [chunk["text"] for chunk in retrieved_chunks]

        if sources:
            response = rationale_model(query=user_input, sources=sources)
            st.write(f"**AI Response:** {response.ranked_response}")
            st.write(f"**Rationale:** {response.rationale}")

            st.write("📄 **Top Retrieved Sources:**")
            for chunk in retrieved_chunks:
                st.write(f"- {chunk['source']}: {chunk['text'][:300]}...")  # Truncate for display

        else:
            st.warning("No relevant documents found.")
