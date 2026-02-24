

# 🔍 FAISS Index vs Metadata Persistence in VectorQA

In the VectorQA system, **FAISS** is used for fast vector search, while **S3** serves as the persistent ground truth store. There are two key types of persistence involved:

---

## 🧠 1. FAISS Index Persistence (`.faiss` binary)

### What is it?
- Serialized FAISS vector index (ANN structure)
- Saved via `faiss.serialize_index(...)`
- Restored via `faiss.read_index(...)`

### Pros:
✅ Very fast for query-time  
✅ Lightweight to load into memory  
✅ Optimized for inference

### Cons:
❌ Does **not** include original `text`, `metadata`, `signature`  
❌ Cannot reconstruct logs without external mapping

### Used In:
- `save_faiss_index_to_s3(bucket, prefix, index, model_key)`
- *(Planned)* `load_faiss_index_from_s3(...)`

---

## 📋 2. Metadata Persistence (JSON log files on S3)

### What is it?
Structured logs containing:

```json
{
  "text": "Customer flagged anomaly",
  "metadata": {
    "source": "audit-log",
    "event": "flagged"
  },
  "timestamp": "2025-04-06T14:33:12Z",
  "signature": "9a2bce32d..."
}
```

### Pros:
✅ Fully auditable and traceable  
✅ Allows FAISS index to be rebuilt at any time  
✅ Stores all contextual information  

### Cons:
⚠️ Slower to load for direct inference  
⚠️ Requires embedding and indexing on retrieval

### Used In:
- `write_audit_log_entry(...)`
- `load_all_logs_from_s3(...)`
- `semantic_search_logs_s3(...)`

---

## ✅ Best Practice: Use Both

| Use Case             | Save FAISS Index | Save Metadata Logs |
|----------------------|------------------|---------------------|
| 🔍 Fast semantic search | ✅ Yes           | ✅ Yes              |
| 🔁 Rebuilding FAISS index | Optional        | ✅ Required         |
| 🧾 Audit trail storage | ❌ No            | ✅ Yes              |
| 🔀 Model versioning    | ✅ Yes (per model_key) | ✅ Yes (per model_key) |

---

## 🔄 What’s Still To Add

| Function | Purpose |
|----------|---------|
| `load_faiss_index_from_s3()` | Load FAISS index for a model_key from S3 |
| `check_faiss_index_exists()` | Determine if FAISS index is already stored |
| `reindex_faiss_if_missing()` | Rebuild index from metadata if binary not found |
| `index_health_check()` | Scan index for drift, missing embeddings, stale versions |

---

## 🧠 Summary

> **FAISS is for fast retrieval.**  
> **S3 logs are the ground truth.**

Together, they power a system that is:
- 🔒 Audit-safe  
- ⚡ Search-efficient  
- 🛠️ Rebuild-capable  
- 📈 Scalable across versions



Would you like this markdown copied into a README for your `core/` folder as a developer reference too?
