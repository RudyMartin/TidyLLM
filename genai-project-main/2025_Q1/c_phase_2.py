import dspy
import ipywidgets as widgets
from IPython.display import display
import boto3
import PyPDF2
import json
import os
import faiss
import io  # Import io
import numpy as np # Import numpy
from typing import List, Dict
from datetime import datetime
import re
import unicodedata

# Configuration Dictionary

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


# Initialize S3 client
s3 = boto3.client('s3')

# Initialize Bedrock client
bedrock = boto3.client('bedrock-runtime')

# Configuration Widgets

bucket_name_widget = widgets.Text(description="Bucket Name:", placeholder=CONFIG["bucket_name"])
pdf_folder_widget = widgets.Text(description="PDF Folder:", placeholder=CONFIG["pdf_folder"])
pages_folder_widget = widgets.Text(description="Pages Folder:", placeholder=CONFIG["pages_folder"])
json_folder_widget = widgets.Text(description="JSON Folder:", placeholder=CONFIG["json_folder"])
index_folder_widget = widgets.Text(description="Index Folder:", placeholder=CONFIG["index_folder"])
chunk_size_widget = widgets.IntText(description="Chunk Size:", placeholder=CONFIG["chunk_size"])


embedding_model_widget = widgets.Text(description="Embedding Model:", placeholder=CONFIG["embedding_model"])
embedding_dimension_widget = widgets.IntText(description="Embedding Dim:", placeholder=CONFIG["embedding_dimension"])

index_file_widget = widgets.IntText(description="FAISS file", placeholder=CONFIG["index_file"])
nlist_widget = widgets.IntText(description="nlist:", placeholder=CONFIG["nlist"])
nprobe_widget = widgets.IntText(description="nprobe:", placeholder=CONFIG["nprobe"])
num_training_samples_widget = widgets.IntText(description="Training size:", placeholder=CONFIG["num_training_samples"])

# Display Configuration Widgets
display(bucket_name_widget, json_folder_widget, index_folder_widget, embedding_model_widget, embedding_dimension_widget,
        nlist_widget, nprobe_widget, num_training_samples_widget)

def generate_embedding(text):
    """Generate embedding for the given text using Bedrock."""
    response = bedrock.invoke_model(
        modelId=CONFIG["embedding_model"],
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"inputText": text})
    )
    embedding = json.loads(response['body'].read())['embedding']
    return embedding

def generate_and_store_embeddings_s3(config: Dict, json_chunk_key: str) -> None:
    """Generates embeddings for text chunks in S3 using Bedrock and stores them back in S3."""
    try:
        response = s3.get_object(Bucket=config["bucket_name"], Key=json_chunk_key)
        json_data = json.loads(response['Body'].read().decode('utf-8'))
        text = json_data['text']

        # Generate embedding using Bedrock
        embedding = generate_embedding(text)

        # Store embedding in S3 (example: as .npy file)
        embedding_key = f"{config['index_folder']}/embeddings/{os.path.splitext(os.path.basename(json_chunk_key))[0]}.npy" #embeddings/page_1_chunk_1.npy
        s3.put_object(Bucket=config["bucket_name"], Key=embedding_key, Body=np.array(embedding).tobytes())  # Convert to numpy array for consistent format

        # You could also store the embedding in metadata
        # s3.put_object_tagging(Bucket=bucket, Key=json_chunk_key, Tagging={'TagSet': [{'Key': 'embedding', 'Value': str(embedding.tolist())}]})

    except Exception as e:
        print(f"Error generating embeddings: {e}")

def save_faiss_index_s3(config: Dict, index: faiss.Index, filename: str = "faiss_index.faiss") -> None:
    """Saves a FAISS index to a local file and uploads it to S3."""
    index_path = filename

    try:
        # Save the index locally
        faiss.write_index(index, index_path)
        print(f"FAISS index saved locally to {index_path}")

        # Upload the index to S3
        s3_key = f"{config['index_folder']}/{filename}"  # Key for the S3 object
        s3.upload_file(index_path, config["bucket_name"], s3_key)
        print(f"FAISS index uploaded to S3: s3://{config['bucket_name']}/{s3_key}")

    except Exception as e:
        print(f"Error saving and uploading FAISS index: {e}")

def load_faiss_index_s3(config: Dict, filename: str = "faiss_index.faiss") -> faiss.Index:
    """Downloads a FAISS index from S3 to a local file and loads it into memory."""
    index_path = filename

    try:
        # Download the index from S3
        s3_key = f"{config['index_folder']}/{filename}"
        s3.download_file(config["bucket_name"], s3_key, index_path)
        print(f"FAISS index downloaded from S3: s3://{config['bucket_name']}/{s3_key} to {index_path}")

        # Load the index
        index = faiss.read_index(index_path)
        print("FAISS index loaded into memory.")
        return index

    except Exception as e:
        print(f"Error downloading and loading FAISS index: {e}")
        return None


def list_objects_s3(bucket: str, prefix: str, tag_key: str = "doc_processed") -> List[str]:
    """Lists all objects in an S3 bucket with a prefix that do NOT have a specific tag."""
    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        objects = response.get('Contents', [])
        object_keys = []

        for obj in objects:
            key = obj['Key']
            if not check_s3_object_tag(bucket, key, tag_key):
                object_keys.append(key)

        return object_keys
    except Exception as e:
        print(f"Error listing objects in S3: {e}")
        return []

def check_s3_object_tag(bucket: str, key: str, tag_key: str) -> bool:
    """Checks if an S3 object has a specific tag."""
    try:
        response = s3.get_object_tagging(Bucket=bucket, Key=key)
        tag_set = response.get('TagSet', [])
        for tag in tag_set:
            if tag['Key'] == tag_key:
                return True
        return False
    except Exception as e:
        # Handle exceptions, like object not found or no tagging permissions.
        # Depending on your needs, you might want to log this or raise an exception.
        print(f"Error checking tag for object {key}: {e}")
        return False  # Assume no tag if there's an error

def get_training_embeddings(config: Dict, num_samples: int) -> np.ndarray:
    """Retrieves a sample of embeddings from S3 for training the FAISS index."""
    # Get list of JSON chunks from S3 that have NOT been processed
    json_keys = list_objects_s3(CONFIG["bucket_name"], CONFIG["json_folder"])
    print(f"Found {len(json_keys)} JSON chunks total.")

    #Shuffle keys for randoms sampling
    np.random.shuffle(json_keys)

    #Determine how many samples to use
    num_samples = min(num_samples, len(json_keys))

    #Create list for training data
    training_data = []
    #Create the training
    print(f"Selecting training samples for {num_samples} samples")
    for json_key in json_keys[:num_samples]:
        embedding_key = f"{config['index_folder']}/embeddings/{os.path.splitext(os.path.basename(json_key))[0]}.npy"
        try:
            response = s3.get_object(Bucket=CONFIG["bucket_name"], Key=embedding_key)
            embedding_bytes = response['Body'].read()
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            training_data.append(embedding)
        except Exception as e:
             print(f"Error processing {embedding_key}")

    # Convert to numpy array
    training_data = np.array(training_data)

    #Check for dimension
    embedding_dimension = CONFIG["embedding_dimension"]
    training_data = training_data.reshape(-1, embedding_dimension)

    print(f"Generated training data of size {training_data.shape}")
    return training_data

def process_embeddings(button):
     # Update the config
    CONFIG["bucket_name"] = bucket_name_widget.value
    CONFIG["json_folder"] = json_folder_widget.value
    CONFIG["index_folder"] = index_folder_widget.value
    CONFIG["embedding_model"] = embedding_model_widget.value
    CONFIG["embedding_dimension"] = int(embedding_dimension_widget.value)
    CONFIG["nlist"] = int(nlist_widget.value)
    CONFIG["nprobe"] = int(nprobe_widget.value)
    CONFIG["num_training_samples"] = int(num_training_samples_widget.value)

    # Load the FAISS index, or create a new one if it doesn't exist
    index_s3_key = f"{CONFIG['index_folder']}/{CONFIG['index_file']}"  # Key for the S3 object
    try:
        s3.head_object(Bucket=CONFIG["bucket_name"], Key=index_s3_key)
        print("Found pre-existing index.")
        index = load_faiss_index_s3(CONFIG)
    except:

        # Initialize FAISS index
        print ("Did not find pre-existing index, creating a new one.")
        embedding_dimension = CONFIG["embedding_dimension"]
        nlist = CONFIG["nlist"]  # Get nlist from config
        quantizer = faiss.IndexFlatL2(embedding_dimension)
        index = faiss.IndexIVFFlat(quantizer, embedding_dimension, nlist, faiss.METRIC_L2)

        # Train the index
        print("Fetching training data")
        train_data = get_training_embeddings(CONFIG, CONFIG["num_training_samples"])

        #Check size and dimensions
        print("Training the IndexIVFFlat index...")
        print(f"The training size was found {train_data.shape}")

        # In a real application, you'd use a representative sample of your embeddings for training
        # Here, we'll generate some random vectors for demonstration purposes
        index.train(train_data)
        print("Index training complete.")

        index.nprobe = CONFIG["nprobe"]  # Tune this parameter based on your dataset

    # Get list of JSON chunks from S3 that have NOT been processed
    json_keys = list_objects_s3(CONFIG["bucket_name"], CONFIG["json_folder"])
    print(f"Found {len(json_keys)} JSON chunks to process.")

    # Process each JSON chunk
    for json_key in json_keys:
        print(f"Processing JSON chunk: {json_key}")

        # 5. Generate and store embeddings for each chunk
        generate_and_store_embeddings_s3(CONFIG, json_key)

        # 6. Load embeddings and add to the FAISS index
        embedding_key = f"{CONFIG['index_folder']}/embeddings/{os.path.splitext(os.path.basename(json_key))[0]}.npy"

        # Load and reshape embedding
        response = s3.get_object(Bucket=CONFIG["bucket_name"], Key=embedding_key)
        embedding_bytes = response['Body'].read()
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)

        # Ensure embedding is the correct dimension and shape
        embedding = embedding.reshape(1, CONFIG["embedding_dimension"])  # Reshape to (1, embedding_dimension)

        # Add to FAISS index
        index.add(embedding)

    # Save the FAISS index to S3
    save_faiss_index_s3(CONFIG, index, filename = CONFIG["index_file"])

process_button = widgets.Button(description="Process Embeddings")
process_button.on_click(process_embeddings)
display(process_button)
