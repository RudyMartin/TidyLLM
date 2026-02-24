import streamlit as st
import boto3
import faiss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.llms import Bedrock
import dspy
import os

# Configuration Settings
bucket_name = "your-s3"
set_r_directory = "development/set_r"
set_c_directory = "development/set-c"
faiss_index_r_path = "development/faiss_index/reports.index"
faiss_index_c_path = "development/faiss_index/content_reviews.index"
index_dim = 1536

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
    pdf_texts = []
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    if 'Contents' in response:
        for obj in response['Contents']:
            file_key = obj['Key']
            pdf_obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
            pdf_reader = PdfReader(pdf_obj['Body'])
            text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
            pdf_texts.append(text)
    return pdf_texts

# Load PDFs from two different sets
def load_all_pdfs():
    set_r_texts = load_pdfs_from_s3(bucket_name, prefix=set_r_directory)
    set_c_texts = load_pdfs_from_s3(bucket_name, prefix=set_c_directory)
    return set_r_texts, set_c_texts

# Vectorization & FAISS Indexing
def create_faiss_index(texts):
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)
    index = faiss.IndexFlatL2(index_dim)
    index.add(embeddings)
    return index, embeddings, texts

# Retrieve Relevant Chunks
def retrieve_text(query, index, texts, k=3):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [{"text": texts[i], "source": f"Document {i}"} for i in indices[0] if i < len(texts)]

# Streamlit Interface
st.title("AI Chatbot & Historical Metrics Dashboard")

# Chatbot UI
user_input = st.text_input("Ask a question:")
category = st.selectbox("Select document set:", ["Reports (Set R)", "Content & Reviews (Set C)", "All"])

if user_input:
    if category == "Reports (Set R)":
        relevant_texts = retrieve_text(user_input, st.session_state.index_r, st.session_state.texts_r)
    elif category == "Content & Reviews (Set C)":
        relevant_texts = retrieve_text(user_input, st.session_state.index_c, st.session_state.texts_c)
    else:
        relevant_texts_r = retrieve_text(user_input, st.session_state.index_r, st.session_state.texts_r)
        relevant_texts_c = retrieve_text(user_input, st.session_state.index_c, st.session_state.texts_c)
        relevant_texts = relevant_texts_r + relevant_texts_c
    
    sources = [res["text"] for res in relevant_texts]
    rationale_output = rationale_model(query=user_input, sources=sources)
    cot_qa_output = cot_qa_model(query=user_input, context=rationale_output.ranked_response)
    
    st.write("**Rationale for Best Response Selection:**", rationale_output.rationale)
    st.write("**Chain of Thought Reasoning Steps:**", cot_qa_output.reasoning_steps)
    st.write("**Final AI Answer:**", cot_qa_output.final_answer)

# Historical Metrics Section
st.sidebar.header("Historical Metrics")
if st.sidebar.button("Show Metrics Dashboard"):
    synthetic_data = pd.DataFrame({
        "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        "Findings": np.random.randint(5, 20, size=12),
        "Severity High": np.random.randint(1, 10, size=12),
        "Severity Medium": np.random.randint(2, 15, size=12),
        "Severity Low": np.random.randint(5, 25, size=12)
    })

    st.subheader("Findings Over Time")
    fig, ax = plt.subplots()
    ax.plot(synthetic_data["Month"], synthetic_data["Findings"], marker='o', linestyle='-')
    ax.set_xlabel("Month")
    ax.set_ylabel("Findings Count")
    ax.set_title("Number of Findings Reported Each Month")
    st.pyplot(fig)
    
    st.subheader("Severity Distribution")
    fig, ax = plt.subplots()
    synthetic_data.set_index("Month")[["Severity High", "Severity Medium", "Severity Low"]].plot(kind='bar', stacked=True, ax=ax)
    ax.set_ylabel("Count")
    ax.set_title("Severity Distribution of Findings")
    st.pyplot(fig)
    
    st.subheader("Overall Severity Breakdown")
    fig, ax = plt.subplots()
    pie_data = synthetic_data[["Severity High", "Severity Medium", "Severity Low"]].sum()
    ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
    ax.set_title("Severity Breakdown of All Findings")
    st.pyplot(fig)
