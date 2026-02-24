Perfect — this is a great call. Doing a **JSON cleanup and refresh pass** before embedding ensures:

- ✅ Consistent structure (`metadata`, `chunks`, `embeddings`)
- ✅ Clean text for embedding
- ✅ All embeddings follow the **short `model_key` format** like `"titan_v2"`

---

### ✅ Cleanup Plan Overview

Here’s a step-by-step outline of what we’ll do:

---

#### **Step 1: Define a Cleanup Script**

We’ll scan each JSON file in your S3 folder (or local test folder), and for each file:

- Ensure `metadata`, `chunks`, and `embeddings` keys are present.
- Clean up or rebuild `chunks` from the full `text` field if needed.
- Remove legacy or malformed `embeddings`.
- Optionally preserve tags or mark them for blank page inspection.
- Write back the cleaned file to S3 or local disk.

---

#### **Step 2: Rebuild with Short `model_key`**

We'll define the embedding format like this:
```json
"embeddings": {
  "titan_v2": {
    "id": "amazon.titan-embed-text-v2:0",
    "dimensions": 1024,
    "embed_date": "2025-04-04T01:23:45.000Z",
    "vector": [...]
  }
}
```

---

#### **Step 3: Re-Embed**

After cleanup, we’ll run:
```python
vector_manager.index_batch(["titan_v2"], force=True)
```
This will embed clean content and save it properly using the `titan_v2` key.

---

### ✅ Ready-to-Run Cleanup Script?

Would you like me to generate a full cleanup script that:

- Iterates through all your S3 JSONs
- Validates `chunks`, `metadata`, and resets `embeddings`
- Stores the updated version back to S3?

Let me know:
- ✅ Use **S3** or **local** folder?
- ✅ Should we generate chunks from `text` if missing?
- ✅ Skip blank/empty files?
- ✅ Preserve or reset tags?

Once you confirm those, I’ll generate the script for your environment!
