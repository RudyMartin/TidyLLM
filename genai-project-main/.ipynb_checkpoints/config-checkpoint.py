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
    "nprobe": 16   # Number of Voronoi cells to search during query
}