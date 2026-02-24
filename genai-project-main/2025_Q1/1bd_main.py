# 🚀 Key Drivers Report Processing (Basic Version - No Streamlit)

import json
import time
import datetime
import faiss
import numpy as np
import pandas as pd
import re
import io
import boto3
import logging
import unicodedata
import os
import nltk
from typing import Dict, List
from PyPDF2 import PdfReader
from dspy import BootstrapFewShot, Compiler

# **🔹 Configuration**
CONFIG = {
    "bucket_name": "sagemaker-us-east-1-188494237500",
    "pdf_folder": "dev/pdf/arxiv_wellsfargo",
    "pages_folder": "dev/page",
    "json_folder": "dev/json",
    "index_folder": "dev/idx",
    "log_directory": "dev/logs",
    "embedding_model": "amazon.titan-embed-text-v1",
    "embedding_dimension": 1536,
    "chunk_size": 200
}

MODEL_OPTIONS = {
    "titan_v1": {"id": "amazon.titan-embed-text-v1", "dimensions": 768},
    "titan_v2": {"id": "amazon.titan-embed-text-v2:0", "dimensions": 1024},
    "cohere": {"id": "cohere.embed-english-v3", "dimensions": None},
    "anthropic": {"id": "anthropic.claude-v2", "dimensions": None}
}

# **🔹 S3 Client**
s3_client = boto3.client("s3")

# **🔹 Ensure NLTK Dependencies**
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# **🔹 Initialize Logging**
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# **🔹 Smart Lexical Chunking**
def smart_lexical_chunking(text, max_words=200):
    sentences = nltk.sent_tokenize(text)
    chunks, current_chunk, current_length = [], [], 0

    for sentence in sentences:
        sub_sentences = re.split(r"[;,:]|\b(and|but|or|which|that)\b", sentence)
        sub_sentences = [s.strip() for s in sub_sentences if s]

        for sub_sentence in sub_sentences:
            sub_length = len(sub_sentence.split())
            if current_length + sub_length > max_words:
                chunks.append(" ".join(current_chunk))
                current_chunk, current_length = [], 0

            current_chunk.append(sub_sentence)
            current_length += sub_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# **🔹 List Objects in S3**
def list_objects_s3(bucket: str, prefix: str) -> List[str]:
    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        return [obj["Key"] for obj in response.get("Contents", [])]
    except Exception as e:
        logging.error(f"⚠️ Error listing S3 objects: {e}")
        return []

# **🔹 Save JSON to S3**
def save_json_to_s3(data: Dict, file_name: str):
    json_data = json.dumps(data).encode("utf-8")
    s3_client.put_object(Bucket=CONFIG["bucket_name"], Key=f"{CONFIG['json_folder']}/{file_name}.json", Body=json_data)

# **🔹 FAISS Index Management**
class FAISSHandler:
    def __init__(self, embedding_dimension):
        self.index = None
        self.embedding_dimension = embedding_dimension

    def load_index(self):
        """Loads FAISS index dynamically."""
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.embedding_dimension)
            logging.info("✅ FAISS Index Loaded.")

    def clear_index(self):
        """Clears FAISS index."""
        self.index = None
        logging.warning("⚠️ FAISS Index Cleared.")

    def search(self, query_embedding):
        """Performs a FAISS search."""
        if self.index is None:
            logging.error("❌ FAISS Index is not loaded.")
            return []
        _, indices = self.index.search(query_embedding, 3)
        return indices[0].tolist()

# **🔹 Process a PDF (Text Extraction)**
def process_pdf(file_path):
    """Extracts text from a PDF file and chunks it intelligently."""
    try:
        reader = PdfReader(file_path)
        text_data = []
        
        for page_num in range(len(reader.pages)):
            text = reader.pages[page_num].extract_text() or ""
            text = " ".join(text.split())  # Clean excessive spaces
            text_data.append({"page_number": page_num + 1, "text": text})

        return text_data

    except Exception as e:
        logging.error(f"❌ Error processing PDF: {e}")
        return []

# **🔹 DSPy Query Expansion & Report Generation**
def dspy_refine_query(query_text):
    """Expands a query using DSPy BootstrapFewShot."""
    return BootstrapFewShot.expand_query(query_text)

def dspy_generate_report(retrieved_docs):
    """Uses DSPy Compiler to optimize report generation."""
    return Compiler.compile(retrieved_docs)

# **🔹 TESTING THE SCRIPT WITHOUT STREAMLIT **

if __name__ == "__main__":
    logging.info("🚀 Running Key Drivers Report Processing (Basic Mode)")

    # **FAISS Indexing Test**
    faiss_handler = FAISSHandler(CONFIG["embedding_dimension"])

    # Load FAISS Index
    logging.info("🔄 Loading FAISS Index...")
    faiss_handler.load_index()

    # **Process a Sample PDF**
    sample_pdf_path = "sample.pdf"  # Replace with your test file
    logging.info(f"📄 Processing PDF: {sample_pdf_path}")
    extracted_text = process_pdf(sample_pdf_path)

    if extracted_text:
        logging.info("✅ PDF Processed Successfully.")
        logging.info(f"📄 Extracted Text from {len(extracted_text)} pages.")

        # **Chunk the extracted text**
        chunked_texts = []
        for page in extracted_text:
            chunked_texts.extend(smart_lexical_chunking(page["text"], CONFIG["chunk_size"]))

        logging.info(f"📑 Chunked into {len(chunked_texts)} segments.")

        # **Simulate Saving to S3**
        save_json_to_s3({"chunks": chunked_texts}, "sample_processed")

    # **Simulate Query Search**
    test_query = "Find key drivers of market trends"
    refined_query = dspy_refine_query(test_query)
    logging.info(f"🔍 Refined Query: {refined_query}")

    # Simulate FAISS search
    test_embedding = np.random.rand(1, CONFIG["embedding_dimension"]).astype(np.float32)
    results = faiss_handler.search(test_embedding)

    logging.info(f"🔎 Retrieved Documents: {results}")

    # **Simulate DSPy Report Generation**
    if results:
        report = dspy_generate_report(results)
        logging.info("📘 Generated Optimized Report:")
        logging.info(report)

    # **Clear FAISS Index**
    logging.info("🗑️ Clearing FAISS Index...")
    faiss_handler.clear_index()
