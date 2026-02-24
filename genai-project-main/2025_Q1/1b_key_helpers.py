# 🚀 key_helpers.py - Helper Functions for Key Drivers Report Processing

import re
import io
import json
import boto3
import logging
import datetime
import nltk
import os
from typing import Dict, List
from PyPDF2 import PdfReader, PdfWriter

# **🔹 Initialize S3 Client**
s3 = boto3.client("s3")

# **🔹 Configuration Dictionary**
CONFIG = {
    "bucket_name": "sagemaker-us-east-1-188494237500",
    "pdf_folder": "dev/pdf/arxiv_wellsfargo",
    "pages_folder": "dev/page",
    "json_folder": "dev/json",
    "index_folder": "dev/idx",
    "log_directory": "dev/logs",
    "chunk_size": 200,
    "pages_per_file": 1,
    "pdf_password": None
}

# **🔹 Ensure NLTK Dependencies are Available**
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# **🔹 Logging Setup**
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# -------------------------------------
# **🔹 Text Processing Functions**
# -------------------------------------

def clean_text(text: str) -> str:
    """Cleans extracted text by normalizing characters and fixing common PDF artifacts."""
    text = re.sub(r"\n{2,}", "\n", text)  # Keeps paragraph breaks
    text = text.replace("0.00-10", "0.00")  # Remove range artifacts
    text = re.sub(r"\s+", " ", text).strip()  # Normalize spaces
    text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)  # Fix hyphenated line breaks
    return text.strip()


def smart_lexical_chunking(text, max_words=200):
    """Splits academic text into semantically meaningful chunks."""
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

    if current_chunk and len(" ".join(current_chunk).split()) > 5:
        chunks.append(" ".join(current_chunk))
    return chunks


# -------------------------------------
# **🔹 S3 Helper Functions**
# -------------------------------------

def list_objects_s3(bucket: str, prefix: str, extensions: List[str] = None) -> List[str]:
    """Lists objects in an S3 bucket under a given prefix, with optional filtering by extension."""
    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        files = [obj["Key"] for obj in response.get("Contents", [])]

        # Apply extension filter if provided
        if extensions:
            files = [file for file in files if file.lower().endswith(tuple(extensions))]

        return files

    except Exception as e:
        logging.error(f"⚠️ Error listing S3 objects: {e}")
        return []


def save_page_chunks_as_json_s3(original_key: str, chunks: List[Dict], bucket: str = None):
    """Saves extracted text chunks into a structured JSON format on S3."""
    bucket = bucket or CONFIG["bucket_name"]
    json_folder = CONFIG["json_folder"]
    document_name = os.path.splitext(os.path.basename(original_key))[0]

    chunk_data = {
        "document_name": document_name,
        "text_date": datetime.datetime.utcnow().isoformat() + "Z",
        "chunks": chunks,
        "embeddings": {}
    }

    chunk_key = f"{json_folder}/{document_name}.json"
    s3.put_object(Bucket=bucket, Key=chunk_key, Body=json.dumps(chunk_data).encode("utf-8"))
    logging.info(f"✅ Saved extracted chunks for {document_name} in {chunk_key}")


# -------------------------------------
# **🔹 Error Logging**
# -------------------------------------

def log_error_to_s3(error_message: str):
    """Logs errors to a dedicated error log in S3."""
    bucket = CONFIG["bucket_name"]
    log_key = f"{CONFIG['log_directory']}/error_logs.json"

    error_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "error": error_message
    }

    try:
        response = s3.get_object(Bucket=bucket, Key=log_key)
        logs = json.loads(response["Body"].read().decode("utf-8"))
    except s3.exceptions.NoSuchKey:
        logs = []

    logs.append(error_entry)

    s3.put_object(
        Bucket=bucket,
        Key=log_key,
        Body=json.dumps(logs, indent=4).encode("utf-8")
    )
    logging.error(f"❌ Error logged: {error_message}")


# -------------------------------------
# **🔹 S3 Access Validation**
# -------------------------------------

def check_s3_access():
    """Verifies S3 access and prompts user to update credentials if necessary."""
    try:
        s3.list_objects_v2(Bucket=CONFIG["bucket_name"], MaxKeys=1)
        logging.info("✅ S3 Access Verified.")
        return True
    except Exception:
        logging.warning("❌ S3 Access Denied. Please update credentials.")
        return False


# -------------------------------------
# **🔹 Admin Functions**
# -------------------------------------

def upload_instructions_to_s3(file_content: str):
    """Uploads new instructions to S3."""
    bucket = CONFIG["bucket_name"]
    instructions_key = f"{CONFIG['log_directory']}/instructions.json"

    s3.put_object(
        Bucket=bucket,
        Key=instructions_key,
        Body=file_content
    )
    logging.info("✅ Instructions updated successfully in S3.")
