
### The JSON Structure (as of 4-8-24)

Each **page** of a document is saved as **one JSON object**, with **three clearly separated sections**:

```json
{
  "metadata": {
    "document_name": "1806.00663_page_5.pdf",
    "page_number": 5,
    "text_date": "2025-03-27T01:21:27.782025",
    "tags": ["#blankpage"]
  },
  "chunks": [
    "chunk_0 text",
    "chunk_1 text",
    "chunk_2 text"
  ],
  "embeddings": {
    "titan_v2": {
      "0": [ ...chunk_0_embedding... ],
      "1": [ ...chunk_1_embedding... ],
      "2": [ ...chunk_2_embedding... ]
    }
  }
}
```

### 🔥 Key Principles

1. ✅ **Do not flatten** the file. The structure above is canonical.
2. ✅ **Embedding is a separate step**, and gets added later to the `embeddings` section.
3. ✅ You might have multiple models (`titan_v2`, `cohere`, etc.) each with embeddings per chunk.
4. ✅ `metadata` is page-level; `chunks` and `embeddings` are list-mapped.

---

### 🧠 What We'll Do Instead

When indexing to `pgVector`, we will:
- **Load** this JSON file.
- For a given `model_key`:
  - Iterate over `chunks[i]` and `embeddings[model_key][i]`
  - Build per-chunk entries on the fly
  - Write `id`, `embedding` into `pg_vec`
  - Write `id`, `model_key`, and `timestamp` into `pg_details`

This keeps your existing pipeline untouched.

---

For more information go to **structured-indexing bridge** function next.
