# 🚀 key_helpers.py - Helper Functions for S3 & FAISS Management

from configuration import *  # ✅ Loads all settings, logging, and imports

# --------------------------------------------------
# **🔹 Smart Lexical Chunking**
# --------------------------------------------------
def smart_lexical_chunking(text, max_words=200):
    """Splits text into semantically meaningful chunks using sentence structure and logical breakpoints."""
    sentences = nltk.sent_tokenize(text)
    chunks, current_chunk, current_length = [], [], 0

    for sentence in sentences:
        sub_sentences = re.split(r"[;,:]|\b(and|but|or|which|that)\b", sentence) if len(sentence.split()) > max_words else [sentence]
        sub_sentences = [s.strip() for s in sub_sentences if s]

        for sub_sentence in sub_sentences:
            sub_length = len(sub_sentence.split())

            if current_length + sub_length > max_words:
                chunks.append(" ".join(current_chunk))
                current_chunk, current_length = [], 0

            current_chunk.append(sub_sentence)
            current_length += sub_length

    if current_chunk and len(" ".join(current_chunk).split()) > 5:
        chunks.append(" ".join(current_chunk))

    return chunks

# --------------------------------------------------
# **🔹 S3 Helper Functions**
# --------------------------------------------------
def list_objects_s3(bucket: str, prefix: str, extensions: List[str] = None) -> List[str]:
    """Lists objects in an S3 bucket under a given prefix, with optional filtering by extension."""
    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        files = [obj["Key"] for obj in response.get("Contents", [])]

        if extensions:
            files = [file for file in files if file.lower().endswith(tuple(extensions))]

        return files
    except Exception as e:
        logging.error(f"⚠️ Error listing S3 objects: {e}")
        return []

def save_page_chunks_as_json_s3(original_key: str, chunks: List[Dict], bucket: str = None):
    """Saves extracted text chunks into a structured JSON format on S3."""
    bucket = bucket or CONFIG["bucket_name"]
    json_folder = CONFIG["json_folder"]
    document_name = os.path.splitext(os.path.basename(original_key))[0]

    chunk_data = {
        "document_name": document_name,
        "text_date": datetime.datetime.utcnow().isoformat() + "Z",
        "chunks": chunks,
        "embeddings": {}
    }

    chunk_key = f"{json_folder}/{document_name}.json"
    s3_client.put_object(Bucket=bucket, Key=chunk_key, Body=json.dumps(chunk_data).encode("utf-8"))
    logging.info(f"✅ Saved extracted chunks for {document_name} in {chunk_key}")

# --------------------------------------------------
# **🔹 FAISS Index Management**
# --------------------------------------------------
def save_faiss_to_s3(index, model_choice, dimension):
    """Serializes and uploads FAISS index to S3 for the selected embedding model and dimension."""
    try:
        faiss_s3_key = f"{CONFIG['index_folder']}/{model_choice}_{dimension}_faiss_index.bin"
        index_data = faiss.serialize_index(index)
        s3_client.put_object(Bucket=CONFIG["bucket_name"], Key=faiss_s3_key, Body=index_data)
        logging.info(f"✅ FAISS index for {model_choice} ({dimension}D) saved to S3!")
    except Exception as e:
        logging.error(f"❌ Error saving FAISS index: {e}")

def load_faiss_from_s3(model_choice, dimension):
    """Loads FAISS index from S3 for the selected embedding model and dimension."""
    try:
        faiss_s3_key = f"{CONFIG['index_folder']}/{model_choice}_{dimension}_faiss_index.bin"
        response = s3_client.get_object(Bucket=CONFIG["bucket_name"], Key=faiss_s3_key)
        index_data = response["Body"].read()
        index = faiss.deserialize_index(faiss.IndexFlatL2(dimension), index_data)
        logging.info(f"✅ FAISS index for {model_choice} ({dimension}D) loaded from S3!")
        return index
    except s3_client.exceptions.NoSuchKey:
        logging.warning(f"⚠️ No FAISS index found for {model_choice} ({dimension}D). Starting fresh.")
        return faiss.IndexFlatL2(dimension)
    except Exception as e:
        logging.error(f"❌ Error loading FAISS index: {e}")
        return None

def delete_faiss_from_s3(model_choice, dimension):
    """Deletes FAISS index from S3 for the selected embedding model and dimension."""
    try:
        faiss_s3_key = f"{CONFIG['index_folder']}/{model_choice}_{dimension}_faiss_index.bin"
        s3_client.delete_object(Bucket=CONFIG["bucket_name"], Key=faiss_s3_key)
        logging.info(f"🗑️ Deleted FAISS index for {model_choice} ({dimension}D) from S3!")
    except Exception as e:
        logging.error(f"❌ Error deleting FAISS index: {e}")
