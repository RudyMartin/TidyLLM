# 🚀 configuration.py - Centralized Config for Key Drivers App

import os
import re
import json
import boto3
import logging
import datetime
import faiss
import nltk
from typing import Dict, List

# **🔹 Logging Setup (Applies to All Scripts)**
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")  # Allows dynamic logging levels
logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
                    format="%(asctime)s - %(levelname)s - %(message)s")

# **🔹 Ensure NLTK Dependencies Are Available**
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# **🔹 S3 Client**
s3_client = boto3.client("s3")

# **🔹 Global Configuration Dictionary**
CONFIG = {
    "bucket_name": "sagemaker-us-east-1-188494237500",
    "pdf_folder": "dev/pdf/arxiv_wellsfargo",
    "pages_folder": "dev/page",
    "json_folder": "dev/json",
    "index_folder": "dev/idx",
    "log_directory": "dev/logs",
    "chunk_size": 200,
    "pages_per_file": 1,
    "pdf_password": None
}

# **🔹 Embedding Model Configurations**
MODEL_OPTIONS = {
    "titan_v1": {"id": "amazon.titan-embed-text-v1", "dimensions": 768},
    "titan_v2": {"id": "amazon.titan-embed-text-v2:0", "dimensions": 1024},
    "cohere": {"id": "cohere.embed-english-v3", "dimensions": None},
    "anthropic": {"id": "anthropic.claude-v2", "dimensions": None}
}
