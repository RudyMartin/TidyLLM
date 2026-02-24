import os
import re
import json
import boto3
import tempfile
import extract_msg
import PyPDF2
import docx
from datetime import datetime
from botocore.exceptions import NoCredentialsError

# AWS Configuration
BUCKET_NAME = "soasis"
REGION = "us-east-1"
S3_PREFIX = "home/genai/issue_c/"

# Initialize S3 client
s3 = boto3.client("s3", region_name=REGION)

def list_s3_files(bucket, prefix):
    """Recursively list all files in an S3 bucket folder."""
    file_list = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            file_list.append(obj["Key"])
    return file_list

def download_s3_file(bucket, key):
    """Download a file from S3 to a temporary local path."""
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    s3.download_file(Bucket=bucket, Key=key, Filename=temp_file.name)
    return temp_file.name

def extract_text_from_doc(file_path):
    """Extract structured text from .docx files with fallback."""
    try:
        doc = docx.Document(file_path)
        return {
            "paragraphs": [para.text for para in doc.paragraphs],
            "tables": [[cell.text for cell in row.cells] for table in doc.tables for row in table.rows]
        }
    except Exception as e:
        print(f"Error reading DOCX file: {e}")
        return None  # Will trigger fallback

def extract_text_from_pdf(file_path):
    """Extract structured text from PDFs, keeping sections when possible."""
    try:
        structured_data = {"pages": []}
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                structured_data["pages"].append(page.extract_text() or "")
        return structured_data
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return None  # Will trigger fallback

def extract_text_from_msg(file_path):
    """Extract email metadata, text, and attachments from .msg files."""
    try:
        msg = extract_msg.Message(file_path)
        email_data = {
            "subject": msg.subject,
            "sender": msg.sender,
            "recipients": msg.recipients,
            "date": msg.date,
            "body": msg.body,
            "attachments": []
        }

        # Process attachments (DOCX & PDF)
        for attachment in msg.attachments:
            if attachment.longFilename.endswith(".pdf") or attachment.longFilename.endswith(".docx"):
                attachment_path = os.path.join(tempfile.gettempdir(), attachment.longFilename)
                with open(attachment_path, "wb") as f:
                    f.write(attachment.data)
                
                # Extract based on type
                if attachment.longFilename.endswith(".pdf"):
                    email_data["attachments"].append({
                        "filename": attachment.longFilename,
                        "content": extract_text_from_pdf(attachment_path) or {"raw_text": extract_fallback_text(attachment_path)}
                    })
                elif attachment.longFilename.endswith(".docx"):
                    email_data["attachments"].append({
                        "filename": attachment.longFilename,
                        "content": extract_text_from_doc(attachment_path) or {"raw_text": extract_fallback_text(attachment_path)}
                    })
        
        return email_data
    except Exception as e:
        print(f"Error reading MSG file: {e}")
        return None  # Will trigger fallback

def extract_fallback_text(file_path):
    """Fallback extraction if structured extraction fails."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            return file.read()
    except Exception as e:
        print(f"Error in fallback text extraction: {e}")
        return ""

def generate_json_structure(file_key, extracted_content, document_type):
    """
    Generates a semi-structured JSON format (Option 2).
    Falls back to dynamic JSON (Option 1) if structured extraction fails.
    """
    metadata = {
        "document_type": document_type,
        "source": "S3",
        "uploaded_at": datetime.utcnow().isoformat(),
        "file_key": file_key
    }

    if extracted_content:
        return {
            "metadata": metadata,
            "content": extracted_content
        }
    else:  # Fallback to dynamic JSON (Option 1) with metadata
        return {
            "metadata": metadata,
            "raw_text": extract_fallback_text(file_key)
        }

def process_s3_documents():
    """Process all .docx, .pdf, and .msg files from S3 and store in JSON format."""
    all_data = []
    
    file_list = list_s3_files(BUCKET_NAME, S3_PREFIX)
    
    for file_key in file_list:
        print(f"Processing: {file_key}")

        extracted_content = None
        document_type = "unknown"

        if file_key.endswith(".docx"):
            document_type = "docx"
            local_path = download_s3_file(BUCKET_NAME, file_key)
            extracted_content = extract_text_from_doc(local_path)

        elif file_key.endswith(".pdf"):
            document_type = "pdf"
            local_path = download_s3_file(BUCKET_NAME, file_key)
            extracted_content = extract_text_from_pdf(local_path)

        elif file_key.endswith(".msg"):
            document_type = "msg"
            local_path = download_s3_file(BUCKET_NAME, file_key)
            extracted_content = extract_text_from_msg(local_path)

        # Generate JSON structure with fallback
        structured_json = generate_json_structure(file_key, extracted_content, document_type)

        # Store structured data
        all_data.append(structured_json)

    # Save JSON
    json_output = "extracted_documents.json"
    with open(json_output, "w", encoding="utf-8") as json_file:
        json.dump(all_data, json_file, indent=4, ensure_ascii=False)

    # Upload JSON to S3
    s3.upload_file(json_output, BUCKET_NAME, "home/genai/processed/extracted_documents.json")
    print(f"Processed data uploaded to: s3://{BUCKET_NAME}/home/genai/processed/extracted_documents.json")

# Run processing
process_s3_documents()

---
def write_control_log(action, details):
    """
    Appends a log entry to the control log stored in S3.

    Args:
    - action (str): The action being logged (e.g., 'file_processed', 'error', 'upload_success').
    - details (dict): Additional details about the event (e.g., file_key, status).

    Example:
    write_control_log('file_processed', {'file_key': 'home/genai/issue_c/doc1.pdf', 'status': 'success'})
    """
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "action": action,
        "details": details
    }

    try:
        # Try to fetch existing log from S3
        existing_logs = []
        try:
            obj = s3.get_object(Bucket=BUCKET_NAME, Key=CONTROL_LOG_PATH)
            existing_logs = json.loads(obj['Body'].read().decode('utf-8'))
        except s3.exceptions.NoSuchKey:
            pass  # No log exists yet, will create a new one

        # Append new entry
        existing_logs.append(log_entry)

        # Upload updated log
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=CONTROL_LOG_PATH,
            Body=json.dumps(existing_logs, indent=4),
            ContentType="application/json"
        )

        print(f"Log entry added: {log_entry}")

    except Exception as e:
        print(f"Error writing to control log: {e}")
        
write_control_log("file_processed", {"file_key": "home/genai/issue_c/report.pdf", "status": "success"})
write_control_log("error", {"file_key": "home/genai/issue_c/broken.docx", "error_message": "Failed to extract text"})
write_control_log("upload_success", {"file_key": "home/genai/processed/extracted_documents.json", "status": "uploaded"})



        
