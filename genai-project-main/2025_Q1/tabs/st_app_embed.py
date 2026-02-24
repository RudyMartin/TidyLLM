import streamlit as st
from embedding_helper import AmazonEmbeddingVectorizer
from s3_helper import list_json_files_s3, list_folders_s3
import boto3

# Default Configuration
CONFIG = {
    "bucket_name": "my-s3-bucket",
    "json_folder": "json_files/",  # Default folder
}

# Initialize Streamlit App
st.title("📄 S3 Batch Embedding Processor")
st.markdown("Select an S3 folder and embedding model to process all JSON files in that folder.")

# Fetch available folders in S3
s3_folders = list_folders_s3(CONFIG["bucket_name"])
if not s3_folders:
    st.warning("No folders found in S3.")
    st.stop()

# User selects a folder (prefix)
selected_folder = st.sidebar.selectbox("📂 Choose a folder:", s3_folders)

# Fetch supported models with dimensions
vectorizer_models = AmazonEmbeddingVectorizer.MODEL_OPTIONS
model_options = [f"{v['id']} ({v['dimensions']}D)" for k, v in vectorizer_models.items()]

# User selects a model (Model & Dimensions as a set)
selected_model_option = st.sidebar.selectbox("🤖 Choose an embedding model & dimensions:", model_options)

# Extract selected model_id and dimensions
selected_model = next(v["id"] for k, v in vectorizer_models.items() if f"{v['id']} ({v['dimensions']}D)" == selected_model_option)
selected_dimensions = next(v["dimensions"] for k, v in vectorizer_models.items() if f"{v['id']} ({v['dimensions']}D)" == selected_model_option)

# Button to start batch processing
if st.sidebar.button("⚡ Start Batch Processing"):
    with st.spinner("Processing all JSON files..."):
        # Get JSON files in the selected folder
        json_files = list_json_files_s3(CONFIG["bucket_name"], selected_folder)

        if not json_files:
            st.error(f"No JSON files found in '{selected_folder}'.")
        else:
            # Initialize vectorizer with selected model & dimensions
            vectorizer = AmazonEmbeddingVectorizer(model_id=selected_model, dimensions=selected_dimensions)

            # Process each JSON file in S3
            for s3_key in json_files:
                vectorizer.update_json_with_embeddings_s3(CONFIG["bucket_name"], s3_key, exclude_keys=["embeddings"])

            st.success(f"✅ Successfully processed {len(json_files)} JSON files in '{selected_folder}' using {selected_model} ({selected_dimensions}D)!")
