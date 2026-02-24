# version  with one json per page - easier to keep track of
## latest working version  Sunday morning

import dspy
import ipywidgets as widgets
from IPython.display import display
import faiss
import numpy as np
from typing import List, Dict
from datetime import datetime
import re
import unicodedata
import boto3
import io
import os
import json
from PyPDF2 import PdfReader, PdfWriter
import logging

# Define Configuration Dictionary
CONFIG = {
    "bucket_name": "sagemaker-us-east-1-188494237500",
    "pdf_folder": "dev/pdf/arxiv_wellsfargo",
    "pages_folder": "dev/page",
    "json_folder": "dev/json",
    "chunk_size": 200
}

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize S3 client
s3 = boto3.client('s3')

# Configuration Widgets
bucket_name_widget = widgets.Text(description="Bucket Name:", placeholder=CONFIG["bucket_name"])
pdf_folder_widget = widgets.Text(description="PDF Folder:", placeholder=CONFIG["pdf_folder"])
pages_folder_widget = widgets.Text(description="Pages Folder:", placeholder=CONFIG["pages_folder"])
json_folder_widget = widgets.Text(description="JSON Folder:", placeholder=CONFIG["json_folder"])
chunk_size_widget = widgets.IntText(description="Chunk Size:", value=CONFIG["chunk_size"])

# Set Default Display Configuration Widget Values
display(bucket_name_widget, pdf_folder_widget, pages_folder_widget, json_folder_widget, chunk_size_widget)


# Function to list objects in S3
def list_objects_s3(bucket: str, prefix: str) -> List[str]:
    """Lists all objects in an S3 bucket under a given prefix."""
    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        return [obj["Key"] for obj in response.get("Contents", [])]
    except Exception as e:
        logging.error(f"⚠️ Error listing S3 objects: {e}")
        return []


# Split PDF into pages and store in S3
def split_pdf_into_pages_s3(pdf_key: str):
    """Splits a PDF into individual pages and uploads them to S3."""
    try:
        bucket = CONFIG["bucket_name"]
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

            page_key = f"{CONFIG['pages_folder']}/{base_name}_page_{page_num + 1}.pdf"
            s3.put_object(Bucket=bucket, Key=page_key, Body=output_pdf.getvalue())
            logging.info(f"✅ Uploaded page {page_num + 1} to {page_key}")

    except Exception as e:
        logging.error(f"❌ Error splitting PDF {pdf_key}: {e}")

## this  is  the function that needs to be modified to fit s3 processing
def split_pdf(file_path, pages_per_file, output_folder, password=None):
    try:
        pdf_reader = PyPDF2.PdfReader(file_path)

        # Attempt to decrypt the PDF if it is encrypted
        if pdf_reader.is_encrypted:
            if password is None:
                print(f"PDF {file_path} is encrypted. Skipping.")
                return
            pdf_reader.decrypt(password)

        total_pages = len(pdf_reader.pages)

        for start_page in range(0, total_pages, pages_per_file):
            pdf_writer = PyPDF2.PdfWriter()

            for page in range(start_page, min(start_page + pages_per_file, total_pages)):
                pdf_writer.add_page(pdf_reader.pages[page])

            output_filename = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(file_path))[0]}_split_{start_page+1}_to_{min(start_page + pages_per_file, total_pages)}.pdf")

            with open(output_filename, 'wb') as out:
                pdf_writer.write(out)

            print(f"Created: {output_filename}")

    except PdfReadError as e:
        print(f"Error reading {file_path}: {e}")

# Extract text from a single PDF page in S3
def extract_text_from_pdf_page_s3(page_key: str) -> Dict:
    """Extracts text from a single-page PDF stored in S3."""
    try:
        bucket = CONFIG["bucket_name"]
        response = s3.get_object(Bucket=bucket, Key=page_key)
        pdf_file = response["Body"].read()

        reader = PdfReader(io.BytesIO(pdf_file))
        text = reader.pages[0].extract_text() or ""
        page_number = int(page_key.split("_")[-1].split(".")[0])

        return {"page_number": page_number, "text": text.strip()}

    except Exception as e:
        logging.error(f"❌ Error extracting text from {page_key}: {e}")
        return {"page_number": None, "text": ""}


# Chunk text into segments
def chunk_text_into_segments(pdf_page_data: Dict) -> List[str]:
    """Chunks extracted text into fixed-size segments."""
    try:
        text = pdf_page_data.get('text', '')
        if not text.strip():
            return []

        words = text.split()
        return [" ".join(words[i:i + CONFIG["chunk_size"]]) for i in range(0, len(words), CONFIG["chunk_size"])]

    except Exception as e:
        logging.error(f"❌ Error chunking text: {e}")
        return []


# Save extracted text chunks from a page into a single JSON file
def save_page_chunks_as_json_s3(original_key: str, chunks: List[str]):
    """Saves all text chunks from a single page into one JSON file."""
    try:
        document_name = os.path.basename(original_key)
        page_number = original_key.split("_")[-1].split(".")[0]
        text_date = datetime.now().isoformat()

        # Store all chunks as a list inside a single JSON file
        chunk_data = {
            "document_name": document_name,
            "page_number": page_number,
            "text_date": text_date,
            "chunks": chunks
        }
        
        chunk_key = f"{CONFIG['json_folder']}/page_{page_number}.json"
        s3.put_object(Bucket=CONFIG["bucket_name"], Key=chunk_key, Body=json.dumps(chunk_data).encode('utf-8'))

        logging.info(f"✅ Saved all chunks for page {page_number} in {chunk_key}")
    except Exception as e:
        logging.error(f"❌ Error saving page chunks to JSON: {e}")


# Process all PDFs in the S3 bucket
def process_documents():
    """Processes all PDFs, extracts pages, text, and saves JSON chunks."""
    try:
        pdf_keys = list_objects_s3(CONFIG["bucket_name"], CONFIG["pdf_folder"])
        logging.info(f"📄 Processing {len(pdf_keys)} PDFs...")

        for pdf_key in pdf_keys:
            logging.info(f"🔍 Processing PDF: {pdf_key}")
            split_pdf_into_pages_s3(pdf_key)

            page_keys = list_objects_s3(CONFIG["bucket_name"], CONFIG["pages_folder"])

            for page_key in page_keys:
                pdf_page_data = extract_text_from_pdf_page_s3(page_key)
                chunks = chunk_text_into_segments(pdf_page_data)

                if chunks:
                    save_page_chunks_as_json_s3(page_key, chunks)

            logging.info(f"✅ Finished processing PDF: {pdf_key}")

    except Exception as e:
        logging.error(f"❌ Error processing documents: {e}")


# UI Widgets for Execution
action_selector = widgets.Dropdown(
    options=[("Process PDFs", "process")],
    description="Action:",
)

execute_button = widgets.Button(description="Execute")

def execute_action(_):
    if action_selector.value == "process":
        process_documents()

execute_button.on_click(execute_action)

# Display UI
display(widgets.HTML("<h2 style='color: navy; background-color: lightgray; padding: 10px; border-radius: 8px;'>Document Processing</h2>"))
display(action_selector, execute_button)
