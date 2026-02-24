
CONFIG = {
    "bucket_name": "your-s3-bucket-name",
    "json_folder": "data/json",
    "index_folder": "data/idx",
    "compiled_folder": "compiled_modules",
    "nlist": 512,
    "nprobe": 16,
    "num_training_samples": 1000,
    "embedding_models": {
        "amazon.titan-embed-text-v2:0": 1536,
        "cohere.embed-english-v3": None,
        "anthropic.claude-v2": None
    },
    "default_model": "amazon.titan-embed-text-v2:0"
}
