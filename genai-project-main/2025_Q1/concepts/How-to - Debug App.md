### ✅ Typical Components in `demo_vector_sanity_app`:
- **PDF Upload** (to user-specific S3 folder)
- **Text Extraction** (using `extraction_helper.py`)
- **Embedding** (via `embedding_helper.py`)
- **FAISS + pgVector Indexing** (via `vector_manager`)
- **Side-by-Side Retrieval + Comparison**

---

### 👀 Common Debug Areas:
- [ ] **S3 paths or file not found errors**
- [ ] **Mismatch in model key vs. embedding ID**
- [ ] **Index not updating or being overwritten**
- [ ] **pgVector not syncing or missing entries**
- [ ] **Text extraction producing blank or malformed output**
- [ ] **Embedding outputs missing from JSON or stored incorrectly**
- [ ] **UI not updating / Streamlit stale states**

---
