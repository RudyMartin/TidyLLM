import dspy
import ipywidgets as widgets
from IPython.display import display
import faiss
import numpy as np # Import numpy
from typing import List, Dict
from datetime import datetime
import re
import unicodedata
import boto3
import io
import os
import json
from typing import List, Dict
from PyPDF2 import PdfReader, PdfWriter
import logging


# Define Configuration Dictionary
#  import config #prod version

CONFIG = {
    "bucket_name": "sagemaker-us-east-1-188494237500",
    "pdf_folder": "dev/pdf/arxiv_wellsfargo",
    "pages_folder": "dev/page",
    "json_folder": "dev/json",
    "index_folder": "dev/idx",
    "embedding_model": "amazon.titan-embed-text-v1",  # Bedrock model ID
    "embedding_dimension": 1536,  # Specify the dimension for the Bedrock model you are using
    "chunk_size": 200,
    "nlist": 512,  # Number of Voronoi cells for IndexIVFFlat
    "nprobe": 16,   # Number of Voronoi cells to search during query
    "index_file": "faiss_index.faiss", #Local name.
    "num_training_samples": 1000 # training size
}

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize S3 client
s3 = boto3.client('s3')

# Configuration Widgets
bucket_name_widget = widgets.Text(description="Bucket Name:", placeholder=CONFIG["bucket_name"])
pdf_folder_widget = widgets.Text(description="PDF Folder:", placeholder=CONFIG["pdf_folder"])
pages_folder_widget = widgets.Text(description="Pages Folder:", placeholder=CONFIG["pages_folder"])
json_folder_widget = widgets.Text(description="JSON Folder:", placeholder=CONFIG["json_folder"])
index_folder_widget = widgets.Text(description="Index Folder:", placeholder=CONFIG["index_folder"])
chunk_size_widget = widgets.IntText(description="Chunk Size:", value=CONFIG["chunk_size"])

# Set Default Display Configuration Widget Values
display(bucket_name_widget, pdf_folder_widget, pages_folder_widget, json_folder_widget,
        chunk_size_widget)


def check_s3_object_tag(bucket: str, key: str, tag_key: str) -> bool:
    """Checks if an object in S3 has a specific tag."""
    try:
        response = s3.get_object_tagging(Bucket=bucket, Key=key)
        tags = {tag["Key"]: tag["Value"] for tag in response.get("TagSet", [])}
        return tag_key in tags
    except s3.exceptions.ClientError as e:
        if e.response['Error']['Code'] == 'AccessDenied':
            logging.warning(f"Skipping tag check for {key} due to insufficient permissions.")
            return False  # Assume the object has no tag and continue
        else:
            logging.error(f"Error retrieving tags for {key}: {e}")
            return False

def list_objects_s3(bucket: str, prefix: str, tag_key: str = "doc_processed") -> List[str]:
    """Lists all objects in an S3 bucket with a prefix that do NOT have a specific tag."""
    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        objects = response.get('Contents', [])
        logging.info(f"Found {len(objects)} objects in {bucket}/{prefix}")

        object_keys = [obj['Key'] for obj in objects if not check_s3_object_tag(bucket, obj['Key'], tag_key)]
        
        logging.info(f"Completed listing objects in {bucket}/{prefix}. {len(object_keys)} files found.")  # ✅ STATUS MESSAGE
        return object_keys
    except Exception as e:
        logging.error(f"Error listing objects in S3: {e}")
        return []

def split_pdf_into_pages_s3(pdf_key: str, config: Dict):
    """Splits a PDF into individual pages and uploads them to S3."""
    try:
        bucket = config["bucket_name"]
        response = s3.get_object(Bucket=bucket, Key=pdf_key)
        pdf_file = response["Body"].read()

        reader = PdfReader(io.BytesIO(pdf_file))
        base_name = os.path.splitext(os.path.basename(pdf_key))[0]

        for page_num, page in enumerate(reader.pages):
            output_pdf = io.BytesIO()
            writer = PdfWriter()
            writer.add_page(page)
            writer.write(output_pdf)
            output_pdf.seek(0)

            page_key = f"{config['pages_folder']}/{base_name}_page_{page_num + 1}.pdf"
            s3.put_object(Bucket=bucket, Key=page_key, Body=output_pdf.getvalue())
            logging.info(f"Uploaded page {page_num + 1} to {page_key}")

        logging.info(f"Completed splitting {pdf_key} into pages and uploading to S3.")  # ✅ STATUS MESSAGE
    except Exception as e:
        logging.error(f"Error splitting PDF {pdf_key}: {e}")

def extract_text_from_pdf_page_s3(page_key: str, config: Dict) -> Dict:
    """Extracts text from a single page PDF stored in S3."""
    try:
        bucket = config["bucket_name"]
        response = s3.get_object(Bucket=bucket, Key=page_key)
        pdf_file = response["Body"].read()

        reader = PdfReader(io.BytesIO(pdf_file))
        text = reader.pages[0].extract_text() or ""
        
        logging.info(f"Completed text extraction for {page_key}. Extracted {len(text)} characters.")  # ✅ STATUS MESSAGE
        return {"page_number": int(page_key.split("_")[-1].split(".")[0]), "text": text.strip()}
    except Exception as e:
        logging.error(f"Error extracting text from {page_key}: {e}")
        return {"page_number": None, "text": ""}

def chunk_text_into_segments(pdf_page_data: Dict, config: Dict) -> List[str]:
    """Chunks text from PDF page data into fixed-size segments."""
    try:
        text = pdf_page_data.get('text', '')
        if not text.strip():
            logging.warning(f"Skipping empty text for page {pdf_page_data.get('page_number', 'Unknown')}")
            return []

        words = text.split()
        chunks = [" ".join(words[i:i + config["chunk_size"]]) for i in range(0, len(words), config["chunk_size"])]

        logging.info(f"Completed text chunking for page {pdf_page_data.get('page_number', 'Unknown')}. {len(chunks)} chunks created.")  # ✅ STATUS MESSAGE
        return chunks
    except Exception as e:
        logging.error(f"Error chunking text: {e}")
        return []

def save_json_to_s3(data: Dict, json_key: str, config: Dict):
    """Saves JSON data to an S3 bucket."""
    try:
        bucket = config["bucket_name"]
        s3.put_object(Bucket=bucket, Key=json_key, Body=json.dumps(data))
        logging.info(f"Successfully saved JSON to S3: {json_key}.")  # ✅ STATUS MESSAGE
    except Exception as e:
        logging.error(f"Error saving JSON to S3: {e}")

def process_documents():
    """Processes all PDFs, extracts pages, text, and saves JSON."""
    try:
        pdf_keys = list_objects_s3(CONFIG["bucket_name"], CONFIG["pdf_folder"])
        logging.info(f"Processing {len(pdf_keys)} PDFs...")

        for pdf_key in pdf_keys:
            logging.info(f"Processing PDF: {pdf_key}")
            split_pdf_into_pages_s3(pdf_key, CONFIG)

            # Process each page
            page_keys = list_objects_s3(CONFIG["bucket_name"], CONFIG["pages_folder"])
            all_chunks = []

            for page_key in page_keys:
                pdf_page_data = extract_text_from_pdf_page_s3(page_key, CONFIG)
                chunks = chunk_text_into_segments(pdf_page_data, CONFIG)
                all_chunks.extend(chunks)

            # Save processed data as JSON
            base_name = os.path.splitext(os.path.basename(pdf_key))[0]
            json_key = f"{CONFIG['json_folder']}/{base_name}.json"
            save_json_to_s3({"chunks": all_chunks}, json_key, CONFIG)

            logging.info(f"Finished processing PDF: {pdf_key}. JSON saved successfully.")  # ✅ STATUS MESSAGE

    except Exception as e:
        logging.error(f"Error processing documents: {e}")

if __name__ == "__main__":
    process_documents()
