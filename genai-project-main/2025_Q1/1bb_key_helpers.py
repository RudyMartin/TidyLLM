import json
import boto3
import logging
import datetime
import os
import faiss
from typing import Dict, List

# **🔹 Initialize S3 Client**
s3 = boto3.client("s3")

# **🔹 Logging Setup**
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# **🔹 S3 Helper Functions**
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
