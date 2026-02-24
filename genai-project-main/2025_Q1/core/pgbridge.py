import os
import uuid
import json
import psycopg2
from core.config_helper import get_config, get_clients
from core.s3_utils import list_s3_files_with_extension, load_json_from_s3
from datetime import datetime


def structured_pgvector_index(model_key: str, db_instance: str = "genai"):
    config = get_config()
    clients = get_clients()
    bucket = config["bucket_name"]
    json_folder = config["json_folder"]

    pg_settings = config["pgvector"][db_instance]
    conn = psycopg2.connect(
        dbname=pg_settings["database"],
        user=pg_settings["user"],
        password=pg_settings["password"],
        host=pg_settings["host"],
        port=pg_settings.get("port", 5432)
    )

    json_files = list_s3_files_with_extension(bucket, json_folder, "json")

    with conn.cursor() as cur:
        for file_key in json_files:
            data = load_json_from_s3(clients.s3, bucket, file_key)
            meta = data.get("metadata", {})
            chunks = data.get("chunks", [])
            model_embeds = data.get("embeddings", {}).get(model_key, {})

            for i, chunk_text in enumerate(chunks):
                raw = model_embeds.get(str(i))
                if not raw:
                    continue  # Skip if missing

                # Support both list or dict format
                if isinstance(raw, dict):
                    emb = raw.get("vector")
                    embed_time = raw.get("timestamp")
                else:
                    emb = raw
                    embed_time = datetime.utcnow().isoformat()

                if not emb:
                    continue

                record_id = uuid.uuid4()

                # Insert into pg_vec
                cur.execute("""
                    INSERT INTO pg_vec (id, embedding, embed_time)
                    VALUES (%s, %s, %s);
                """, (record_id, emb, embed_time))

                # Insert into pg_details
                cur.execute("""
                    INSERT INTO pg_details (id, model_key, created_at, document_name, page_number, tags)
                    VALUES (%s, %s, %s, %s, %s, %s);
                """, (
                    record_id,
                    model_key,
                    meta.get("text_date", datetime.utcnow().isoformat()),
                    meta.get("document_name"),
                    meta.get("page_number"),
                    meta.get("tags")
                ))

    conn.commit()
    conn.close()
    print(f"✅ Indexed {len(json_files)} files for model: {model_key}")
