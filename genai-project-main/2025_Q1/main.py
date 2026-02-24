from embedding_helper import AmazonEmbeddingVectorizer
from file_helper import list_json_files_s3

# Directory containing JSON files
json_folder = "json_files"

# Get all JSON files in the folder
json_files = list_json_files(json_folder)

# Initialize vectorizer (Titan V2, default 1024 dimensions)
vectorizer = AmazonEmbeddingVectorizer(model_id="amazon.titan-embed-text-v2:0")

# Process each JSON file and add embeddings
for json_file in json_files:
    vectorizer.update_json_with_embeddings(json_file, exclude_keys=["embeddings"])

print("✅ All JSON files have been processed!")
