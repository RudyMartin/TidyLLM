import boto3
from datetime import datetime
import re

# S3 Configuration
BUCKET_NAME = "your-s3-bucket-name"  # Replace with your bucket name
FULL_PDF_PREFIX = "full_pdf/"
CHUNK_PDF_PREFIX = "chunk_pdf/"
EMBED_JSON_PREFIX = "embed_json/"
FAISS_INDEX_PREFIX = "faiss_index/"

# Metadata Key for Tracking Processing
DATE_PROCESSED_KEY = "date_processed" #constant naming
Utility Functions:

Let's create some helper functions to make the code more readable and reusable.

def get_s3_object_metadata(bucket_name, object_key):
    """Gets the metadata of an S3 object."""
    s3_client = boto3.client('s3')
    try:
        response = s3_client.head_object(Bucket=bucket_name, Key=object_key)
        return response.get('Metadata', {})
    except Exception as e:
        print(f"Error getting metadata for {object_key}: {e}")
        return {}

def update_s3_object_metadata(bucket_name, object_key, metadata):
    """Updates the metadata of an S3 object."""
    s3_client = boto3.client('s3')
    try:
        # 1. Get the object's current metadata (if any)
        existing_metadata = get_s3_object_metadata(bucket_name, object_key)

        # 2. Merge existing and new metadata
        merged_metadata = existing_metadata.copy()  # Start with existing
        merged_metadata.update(metadata)       # Add/override with new

        # 3. Copy the object to itself with the new metadata
        copy_source = {'Bucket': bucket_name, 'Key': object_key}
        s3_client.copy_object(
            CopySource=copy_source,
            Bucket=bucket_name,
            Key=object_key,
            Metadata=merged_metadata,
            MetadataDirective='REPLACE'  # Important!  Tells S3 to replace existing metadata
        )

        print(f"Successfully updated metadata for s3://{bucket_name}/{object_key}")
        return True # success

    except Exception as e:
        print(f"Error updating metadata for {object_key}: {e}")
        return False #failure

def list_s3_objects(bucket_name, prefix):
    """Lists all objects in an S3 bucket with the given prefix."""
    s3_client = boto3.client('s3')
    objects = []
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    objects.append(obj['Key'])
    except Exception as e:
        print(f"Error listing objects in {prefix}: {e}")
    return objects
1. Full PDF Processing:

def process_full_pdfs(bucket_name, full_pdf_prefix, chunk_pdf_prefix):
    """Processes full PDFs: chunks them and tags the source."""
    print("Processing Full PDFs...")
    full_pdf_files = list_s3_objects(bucket_name, full_pdf_prefix)

    for pdf_key in full_pdf_files:
        # 1. Check if it needs processing
        metadata = get_s3_object_metadata(bucket_name, pdf_key)
        if DATE_PROCESSED_KEY in metadata:
            retag = input(f"Full PDF {pdf_key} already processed. Retag? (y/n): ").lower() == 'y'
            if not retag:
                print(f"Skipping {pdf_key} - already processed.")
                continue

        # 2. Chunk the PDF (Your chunking logic here)
        #  Assumes you have a function create_chunks_from_pdf that takes the bucket and PDF key and puts the chunks into chunk_pdf_prefix
        success = create_chunks_from_pdf(bucket_name, pdf_key, chunk_pdf_prefix) #Dummy function - implement your own.
        if not success:
            print(f"Failed to create chunks for {pdf_key}. Skipping metadata update.")
            continue

        # 3. Tag the source PDF
        metadata = {DATE_PROCESSED_KEY: datetime.now().isoformat()}
        if update_s3_object_metadata(bucket_name, pdf_key, metadata):
            print(f"Successfully tagged full PDF {pdf_key}")
        else:
            print(f"Failed to tag full PDF {pdf_key}")

# Dummy chunking function - replace with your actual implementation.
def create_chunks_from_pdf(bucket_name, pdf_key, chunk_pdf_prefix):
    """Placeholder for the function to create chunks from a PDF and store in S3."""
    print(f"Creating chunks for {pdf_key} (Placeholder)")
    # Your chunking logic here
    # Replace with your actual logic

    # For demonstration purposes, assume it creates a dummy chunk
    # and stores it in S3 with a modified key.  Adapt to your chunking logic.
    s3_client = boto3.client("s3")
    try:
        #Extract filename from Key
        original_filename = pdf_key.split('/')[-1]

        # Create a dummy chunk file for demonstration
        chunk_content = b"This is a dummy chunked PDF content."
        chunk_filename = original_filename.replace(".pdf", "_chunk1.pdf")  # Adjust as needed

        # Upload the dummy chunk to S3
        chunk_key = f"{chunk_pdf_prefix}{chunk_filename}" #prefix with path name
        s3_client.put_object(Bucket=bucket_name, Key=chunk_key, Body=chunk_content)
        print(f"Dummy chunk created and uploaded to S3 as {chunk_key}")
        return True
    except Exception as e:
        print(f"Error creating and uploading dummy chunk: {e}")
        return False

2. Chunk PDF Processing (Embedding):

def process_chunk_pdfs(bucket_name, chunk_pdf_prefix, embed_json_prefix):
    """Processes chunk PDFs: embeds them and tags the source."""
    print("Processing Chunk PDFs...")
    chunk_pdf_files = list_s3_objects(bucket_name, chunk_pdf_prefix)

    for chunk_key in chunk_pdf_files:
        # 1. Check if it needs processing
        metadata = get_s3_object_metadata(bucket_name, chunk_key)
        if DATE_PROCESSED_KEY in metadata:
            retag = input(f"Chunk PDF {chunk_key} already processed. Retag? (y/n): ").lower() == 'y'
            if not retag:
                print(f"Skipping {chunk_key} - already processed.")
                continue

        # 2. Embed the chunk (Your embedding logic here)
        # Assumes you have a function embed_chunk_pdf that takes the bucket, PDF key, and puts the JSON result into embed_json_prefix
        success = embed_chunk_pdf(bucket_name, chunk_key, embed_json_prefix)
        if not success:
            print(f"Failed to embed {chunk_key}. Skipping metadata update.")
            continue

        # 3. Tag the source chunk PDF
        metadata = {DATE_PROCESSED_KEY: datetime.now().isoformat()}
        if update_s3_object_metadata(bucket_name, chunk_key, metadata):
            print(f"Successfully tagged chunk PDF {chunk_key}")
        else:
            print(f"Failed to tag chunk PDF {chunk_key}")
# Dummy Embed Json function - replace with your actual implementation.
def embed_chunk_pdf(bucket_name, chunk_key, embed_json_prefix):
    """Placeholder for the function to embed a chunk PDF and store JSON in S3."""
    print(f"Embedding {chunk_key} (Placeholder)")
    # Your embedding logic here
    # Replace with your actual logic

    # For demonstration purposes, assume it creates dummy JSON
    s3_client = boto3.client("s3")
    try:
        #Extract filename from Key
        filename = chunk_key.split('/')[-1]

        # Create a dummy JSON object for demonstration
        json_content = b'{"embedding": [0.1, 0.2, 0.3]}'
        json_filename = filename.replace(".pdf", ".json")  # Adjust as needed

        # Upload the dummy chunk to S3
        json_key = f"{embed_json_prefix}{json_filename}" #add path
        s3_client.put_object(Bucket=bucket_name, Key=json_key, Body=json_content)
        print(f"Dummy Json created and uploaded to S3 as {json_key}")
        return True
    except Exception as e:
        print(f"Error creating and uploading dummy Json: {e}")
        return False
3. Embed JSON Processing (Indexing):

def process_embed_jsons(bucket_name, embed_json_prefix, faiss_index_prefix):
    """Processes embed JSONs: indexes them and tags the source."""
    print("Processing Embed JSONs...")
    embed_json_files = list_s3_objects(bucket_name, embed_json_prefix)

    for json_key in embed_json_files:
        # 1. Check if it needs processing
        metadata = get_s3_object_metadata(bucket_name, json_key)
        if DATE_PROCESSED_KEY in metadata:
            retag = input(f"Embed JSON {json_key} already processed. Retag? (y/n): ").lower() == 'y'
            if not retag:
                print(f"Skipping {json_key} - already processed.")
                continue

        # 2. Index the JSON (Your indexing logic here)
        #  Assumes you have a function index_embed_json that takes the bucket, JSON key, and interacts with the FAISS index. You might not store a new object.
        success = index_embed_json(bucket_name, json_key, faiss_index_prefix)
        if not success:
            print(f"Failed to index {json_key}. Skipping metadata update.")
            continue

        # 3. Tag the source JSON
        metadata = {DATE_PROCESSED_KEY: datetime.now().isoformat()}
        if update_s3_object_metadata(bucket_name, json_key, metadata):
            print(f"Successfully tagged embed JSON {json_key}")
        else:
            print(f"Failed to tag embed JSON {json_key}")
# Dummy Index Json function - replace with your actual implementation.
def index_embed_json(bucket_name, json_key, faiss_index_prefix):
    """Placeholder for the function to index an embed JSON."""
    print(f"Indexing {json_key} (Placeholder)")
    # Your indexing logic here
    # Replace with your actual logic
    return True
Main Execution:

if __name__ == "__main__":
    process_full_pdfs(BUCKET_NAME, FULL_PDF_PREFIX, CHUNK_PDF_PREFIX)
    process_chunk_pdfs(BUCKET_NAME, CHUNK_PDF_PREFIX, EMBED_JSON_PREFIX)
    process_embed_jsons(BUCKET_NAME, EMBED_JSON_PREFIX, FAISS_INDEX_PREFIX)

    print("Finished processing pipeline.")
How to Use:

Configure: Set the BUCKET_NAME, prefixes, and DATE_PROCESSED_KEY at the beginning of the script.
Implement Processing Logic: Replace the create_chunks_from_pdf, embed_chunk_pdf, and index_embed_json placeholder functions with your actual logic. These functions are the heart of your pipeline.
Run: Execute the script. It will walk through each stage, check if files have already been processed, and prompt you to retag them if desired.
Key Improvements & Explanations:

Clear Pipeline Stages: The code is organized into functions that represent the distinct stages of your pipeline.
Date Tracking: The DATE_PROCESSED_KEY is used to track whether a file has been processed at each stage.
Yes/No Retagging Choice: The code prompts you to confirm whether you want to retag files that have already been processed. This gives you control over the pipeline.
Modular Structure: The utility functions (get_s3_object_metadata, update_s3_object_metadata, list_s3_objects) make the code more reusable and easier to understand.
Error Handling: Basic error handling is included in the utility functions and stage processing functions.
Next Steps and Considerations:

Replace Dummy Functions: The most important step: implement your actual PDF chunking, embedding, and indexing logic in the placeholder functions.
Logging: Add more detailed logging to track the progress of the pipeline and to debug any issues.
Exception Handling: Implement more robust exception handling to catch and handle errors gracefully.
Configuration: Consider using a configuration file to store the bucket name, prefixes, and other configuration options.
Scheduling: If you want to run the pipeline on a regular schedule, consider using AWS Lambda or AWS Step Functions.
Concurrency: If you have a large number of files, consider using threading or multiprocessing to process multiple files in parallel.
Triggering: You can set up S3 event triggers to automatically start the pipeline when new files are added to the full_pdf folder. This would make the pipeline fully automated.
This revised solution provides a clear and manageable framework for building your PDF processing pipeline with metadata tracking. Remember to adapt the code to your specific needs and to test it thoroughly. Good luck!

