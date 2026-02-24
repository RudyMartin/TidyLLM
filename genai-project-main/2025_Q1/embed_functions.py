

def update_json_with_embedding(self, json_path, text):
    """
    Reads or creates a JSON file, generates an embedding, and stores it uniquely.

    Args:
        json_path (str): The path to the JSON file.
        text (str): The input text to embed.

    Returns:
        dict: Updated JSON data.
    """
    # Load existing JSON if it exists
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        # Create new JSON structure
        data = {
            "metadata": {
                "source": "generated_text",
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
            },
            "content": text,
            "embeddings": {}
        }

    # Generate embedding
    embedding_vector = self.generate_embedding(text)

    # ✅ Fix: If dimensions is None, use only `model_id`
    model_key = f"{self.model_id}_{self.dimensions}" if self.dimensions else self.model_id

    # ✅ Fix: Avoid storing None values explicitly in JSON
    embedding_entry = {
        "embed_date": datetime.datetime.utcnow().isoformat() + "Z",
        "vector": embedding_vector
    }
    if self.dimensions:
        embedding_entry["dimensions"] = self.dimensions  # Store only if not None

    # Store embedding under its unique key, replacing existing one if present
    data["embeddings"][model_key] = embedding_entry

    # Write back to JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    logger.info(f"✅ Successfully stored embedding from {self.model_id} ({'No dimensions' if not self.dimensions else f'{self.dimensions}D'}) in {json_path}")
    return data
