
"""
extraction_helper.py

PDF text extraction, S3 interaction, smart chunking, and text cleanup utilities.
Rewritten to use PyPDF2 instead of fitz/pdfplumber.

"""

import os
import re
import io
import json
import boto3
import nltk
import unicodedata
import logging
from datetime import datetime
from PyPDF2 import PdfReader


# ---------- TEXT CLEANING ----------

def clean_text(text: str) -> str:
    """
    Cleans extracted text by normalizing Unicode characters, removing newlines,
    tabs, and excessive whitespace. This helps to reduce artifacts from PDF extraction.
    """
    # Normalize unicode characters to a consistent form (NFKC)
    #text = unicodedata.normalize("NFKC", text)
    # Remove multiple newlines but **keep paragraph breaks**
    text = re.sub(r'\n{2,}', '\n', text)  # Keeps a single \n for meaningful breaks
    # Remove common PDF artifacts (e.g., incorrect range values)
    text = text.replace('0.00-10', '0.00')
    # Remove multiple spaces, newlines, and excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Fix hyphenated line breaks (common in PDF text extraction)
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    return text.strip()

def clean_text_backup(text):
    """
    Cleans extracted text by applying common text preprocessing steps.
    - Fixes common OCR artifacts
    - Standardizes formatting
    - Removes unwanted characters and excessive newlines
    """
    # Normalize Unicode characters (e.g., convert fancy quotes to ASCII)
    text = unicodedata.normalize("NFKD", text)
    
    # Encode and decode to normalize text before embedding
    text = text.encode('utf-8').decode('utf-8')

    # Remove common PDF artifacts (e.g., incorrect range values)
    text = text.replace('0.00-10', '0.00')

    # Fix hyphenated line breaks (common in PDF text extraction)
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)

    # Remove multiple spaces, newlines, and excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove multiple newlines but **keep paragraph breaks**
    text = re.sub(r'\n{2,}', '\n', text)  # Keeps a single \n for meaningful breaks

    # Optional: Remove all `\n` if not needed at all
    text = text.replace("\n", " ")

    # Remove special characters that don’t contribute to meaning
    text = re.sub(r'[^\w\s.,;!?%$-]', '', text)  # Keeps punctuation but removes junk

    return text


# ---------- S3 UTILITIES ----------

# this function requires ObjectTagging Read/Write permissions
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


def list_objects_s3(bucket: str, prefix: str) -> List[str]:
    """Lists all objects in an S3 bucket under a given prefix."""
    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        return [obj["Key"] for obj in response.get("Contents", [])]
    except Exception as e:
        logging.error(f"⚠️ Error listing S3 objects: {e}")
        return []

# ---------- TEXT EXTRACTION ----------

def extract_text_from_pdf_page_s3(page_key: str, next_page_key: str = None, bucket: str = None) -> Dict:
    """Extracts text from a single-page PDF and optionally appends the first sentence of the next page."""
    try:
        bucket = bucket or CONFIG["bucket_name"]
        response = s3_client.get_object(Bucket=bucket, Key=page_key)
        pdf_file = response["Body"].read()

        reader = PdfReader(io.BytesIO(pdf_file))
        text = reader.pages[0].extract_text() or ""
        text = clean_text(text)  # Apply text cleaning before proceeding
        
        page_number = int(page_key.split("_")[-1].split(".")[0])

        # Try to capture page continuity if a next page exists
        if next_page_key:
            try:
                next_response = s3_client.get_object(Bucket=bucket, Key=next_page_key)
                next_pdf_file = next_response["Body"].read()
                next_reader = PdfReader(io.BytesIO(next_pdf_file))
                next_text = next_reader.pages[0].extract_text() or ""

                # ✅ Fix: Actually append the first sentence from the next page
                if validate_page_continuity(text, next_text):
                    next_sentences = nltk.sent_tokenize(next_text)
                    if next_sentences:
                        text += " " + next_sentences[0]  # Append first sentence from the next page
                        logging.info(f"🔗 Appended first sentence from {next_page_key} to {page_key}")

            except Exception as e:
                logging.warning(f"⚠️ Could not retrieve next page {next_page_key}: {e}")

        return {"document_name": os.path.basename(page_key), "page_number": page_number, "text": text.strip()}

    except Exception as e:
        logging.error(f"❌ Error extracting text from {page_key}: {e}")
        return {"page_number": None, "text": ""}


# ---------- CHUNKING + STRUCTURE ----------


def chunk_text_into_segments(pdf_page_data: Dict, max_words: int = None) -> List[Dict]:
    """Splits academic text into semantically meaningful chunks before embedding."""
    try:
        max_words = max_words or CONFIG["chunk_size"]
        text = pdf_page_data.get('text', '')

        if not text.strip():
            return []

        page_number = pdf_page_data.get("page_number", 1)
        document_name = pdf_page_data.get("document_name", "unknown.pdf").replace(".pdf", "")

        chunks = []
        chunked_texts = smart_chunking(text, max_words)  # Use smart chunking

        for i, chunk_text in enumerate(chunked_texts):
            chunk_id = f"{document_name}_{page_number}_{i+1:03d}"
            chunks.append({"chunk_id": chunk_id, "text": chunk_text})

        return chunks

    except Exception as e:
        logging.error(f"❌ Error chunking text: {e}")
        return []

def smart_chunking(text, max_words=200):
    """Splits academic text into semantically meaningful chunks using sentence + sub-sentence splitting."""
    sentences = nltk.sent_tokenize(text)  # Split text into sentences
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())

        # If sentence is too long, attempt secondary splitting at logical points
        if sentence_length > max_words:
            sub_sentences = re.split(r'[;,:]|\b(and|but|or|which|that)\b', sentence)  # Break on logical points
            sub_sentences = [s.strip() for s in sub_sentences if s]  # Remove empty splits
        else:
            sub_sentences = [sentence]

        for sub_sentence in sub_sentences:
            sub_length = len(sub_sentence.split())

            # Start a new chunk if adding this sub-sentence exceeds max_words
            if current_length + sub_length > max_words:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0

            current_chunk.append(sub_sentence)
            current_length += sub_length

    # Add the last chunk if it contains enough words (Fixed)
    if current_chunk and len(" ".join(current_chunk).split()) > 5:  # Avoid adding chunks < 5 words
        chunks.append(" ".join(current_chunk))

    return chunks

def validate_page_continuity(text_current: str, text_next: str) -> bool:
    """Checks if the current page ends mid-sentence and suggests merging the next page's first sentence."""
    sentences = nltk.sent_tokenize(text_current)

    # If the last sentence does not end with '.', '!', or '?', assume it's incomplete
    if sentences and not sentences[-1].endswith(('.', '!', '?')):
        logging.warning("⚠️ Possible broken sentence at page boundary. Consider appending the next page's first sentence.")
        return True  # Suggest merging with the next page

    return False  # No continuity issues detected

def save_page_chunks_as_json_s3(original_key: str, chunks: List[Dict], bucket: str = None):
    """Saves all text chunks from a single page into one JSON file."""
    try:
        bucket = bucket or CONFIG["bucket_name"]
        json_folder = CONFIG["json_folder"]

        # Extract the base document name without extension
        document_name = os.path.splitext(os.path.basename(original_key))[0]

        # Extract the correct page sequence number (Fixed Regex)
        match = re.search(r'_page_(\d+)', original_key)
        page_number = match.group(1) if match else "unknown"

        text_date = datetime.now().isoformat()

        chunk_data = {
            "document_name": document_name,
            "page_number": page_number,
            "text_date": text_date,
            "chunks": chunks,
            "embeddings":{}
        }

        # ✅ FIX: Avoid double `_page_X` if it's already in `document_name`
        if not re.search(r'_page_\d+', document_name):
            document_name = f"{document_name}_page_{page_number}"

        chunk_key = f"{json_folder}/{document_name}.json"

        s3.put_object(Bucket=bucket, Key=chunk_key, Body=json.dumps(chunk_data).encode('utf-8'))

        logging.info(f"✅ Saved all chunks for {document_name} in {chunk_key}")

    except Exception as e:
        logging.error(f"❌ Error saving page chunks to JSON: {e}")

def prepare_pdf_splits():
    """Splits all PDFs in the configured S3 folder into pages and uploads them."""
    try:
        pdf_keys = list_objects_s3(CONFIG["bucket_name"], CONFIG["pdf_folder"])
        logging.info(f"📄 Found {len(pdf_keys)} PDFs to split.")

        for pdf_key in pdf_keys:
            logging.info(f"🔍 Splitting PDF: {pdf_key}")
            split_pdf_into_pages_s3(pdf_key)

        logging.info("✅ Finished splitting all PDFs.")

        # ✅ Display success message in the UI
        status_output.clear_output()
        with status_output:
            display(HTML("<b style='color:green;'>✅ Finished splitting all PDFs into pages.</b>"))

    except Exception as e:
        logging.error(f"❌ Error splitting PDFs: {e}")
        with status_output:
            display(HTML(f"<b style='color:red;'>❌ Error: {e}</b>"))


def extract_text_from_pdf_page_s3(page_key: str, next_page_key: str = None, bucket: str = None) -> Dict:
    """Extracts text from a single-page PDF and optionally appends the first sentence of the next page."""
    try:
        bucket = bucket or CONFIG["bucket_name"]
        response = s3_client.get_object(Bucket=bucket, Key=page_key)
        pdf_file = response["Body"].read()

        reader = PdfReader(io.BytesIO(pdf_file))
        text = reader.pages[0].extract_text() or ""
        text = clean_text(text)  # Apply text cleaning before proceeding
        
        page_number = int(page_key.split("_")[-1].split(".")[0])

        # Try to capture page continuity if a next page exists
        if next_page_key:
            try:
                next_response = s3.get_object(Bucket=bucket, Key=next_page_key)
                next_pdf_file = next_response["Body"].read()
                next_reader = PdfReader(io.BytesIO(next_pdf_file))
                next_text = next_reader.pages[0].extract_text() or ""

                # ✅ Fix: Actually append the first sentence from the next page
                if validate_page_continuity(text, next_text):
                    next_sentences = nltk.sent_tokenize(next_text)
                    if next_sentences:
                        text += " " + next_sentences[0]  # Append first sentence from the next page
                        logging.info(f"🔗 Appended first sentence from {next_page_key} to {page_key}")

            except Exception as e:
                logging.warning(f"⚠️ Could not retrieve next page {next_page_key}: {e}")

        return {"document_name": os.path.basename(page_key), "page_number": page_number, "text": text.strip()}

    except Exception as e:
        logging.error(f"❌ Error extracting text from {page_key}: {e}")
        return {"page_number": None, "text": ""}


def extract_text_to_json():
    """Runs the text extraction process from PDFs to structured JSON, ensuring all functions are used."""
    try:
        page_keys = list_objects_s3(CONFIG["bucket_name"], CONFIG["pages_folder"])
        logging.info(f"📄 Processing {len(page_keys)} PDF pages...")

        for i, page_key in enumerate(page_keys):
            next_page_key = page_keys[i+1] if i+1 < len(page_keys) else None

            # Extract text from the current page (and optionally check next page for continuity)
            pdf_page_data = extract_text_from_pdf_page_s3(page_key, next_page_key)

            # Process extracted text into smart chunks
            chunks = chunk_text_into_segments(pdf_page_data, CONFIG["chunk_size"])

            # Save processed chunks as JSON in S3
            if chunks:
                save_page_chunks_as_json_s3(page_key, chunks)

        logging.info("✅ Finished processing all PDF pages into JSON format.")


def split_pdf_into_pages_s3(pdf_key: str, pages_per_file: int = None, pdf_password: str = None):
    """
    Splits a PDF from S3 into chunks and uploads them back to S3.
    Allows overriding `pages_per_file` and `pdf_password` while using default settings from `CONFIG`.
    """
    try:
        # Load config settings
        bucket = CONFIG["bucket_name"]
        pages_folder = CONFIG["pages_folder"]

        # Allow overriding specific values
        pages_per_file = pages_per_file if pages_per_file is not None else CONFIG.get("pages_per_file", 1)
        pdf_password = pdf_password if pdf_password is not None else CONFIG.get("pdf_password")

        # Read PDF from S3
        response = s3.get_object(Bucket=bucket, Key=pdf_key)
        pdf_file = response["Body"].read()

        # Load PDF
        reader = PdfReader(io.BytesIO(pdf_file))

        # Handle encrypted PDFs
        if reader.is_encrypted:
            if pdf_password:
                reader.decrypt(pdf_password)
                logging.info(f"🔑 Decrypted PDF {pdf_key} successfully.")
            else:
                logging.warning(f"❌ PDF {pdf_key} is encrypted and no password was provided. Skipping.")
                return

        total_pages = len(reader.pages)
        base_name = os.path.splitext(os.path.basename(pdf_key))[0]

        # Split PDF into chunks
        for start_page in range(0, total_pages, pages_per_file):
            writer = PdfWriter()
            end_page = min(start_page + pages_per_file, total_pages)

            # Add pages to new PDF
            for page in range(start_page, end_page):
                writer.add_page(reader.pages[page])

            # **Fix: Adjust filename format**
            if pages_per_file == 1:  # Single page case
                page_key = f"{pages_folder}/{base_name}_page_{start_page+1}.pdf"
            else:  # Multi-page range case
                page_key = f"{pages_folder}/{base_name}_page_{start_page+1}_to_{end_page}.pdf"

            # Save split PDF to memory
            output_pdf = io.BytesIO()
            writer.write(output_pdf)
            output_pdf.seek(0)

            # Upload back to S3
            s3.put_object(Bucket=bucket, Key=page_key, Body=output_pdf.getvalue())

            logging.info(f"✅ Uploaded pages {start_page+1} to {end_page} as {page_key}")

    except Exception as e:
        logging.error(f"❌ Error processing PDF {pdf_key}: {e}")

# ---------- S3 FOLDER & FILE LISTING HELPERS ----------

def list_folders_s3(bucket, prefix=""):
    """List all folders (common prefixes) under a given S3 prefix."""
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/")
    return [cp["Prefix"] for cp in response.get("CommonPrefixes", [])]

def list_pdf_files_s3(bucket, prefix=""):
    """List all PDF files under a given S3 prefix."""
    return list_objects_s3(bucket, prefix, extension=".pdf")

def list_json_files_s3(bucket, prefix=""):
    """List all JSON files under a given S3 prefix."""
    return list_objects_s3(bucket, prefix, extension=".json")
