
## 🧱 Architecture Overview

```
+---------------------------+
|  VectorManager (Selector)|
+---------------------------+
            ⬇
+---------------------------+
|   VectorBase (Abstract)   |  <-- Defines the shared contract
+---------------------------+
        ⬇             ⬇
+----------------+   +------------------+
|  FaissStore    |   | PgVectorStore    |
+----------------+   +------------------+
```

---

### 🔁 What Each File Does

---

### 1. `vector_manager.py`  
**🧠 Brain / Router / Orchestrator**

- Picks between FAISS and pgVector
- Delegates to correct store based on config:
```python
manager = VectorManager("pgvector", config, clients)
```
- Returns the right store object to handle:
  - `.validate()`
  - `.index()`
  - `.query()`
  - `.delete()`

✅ Ideal for Streamlit tabs and CLI tools that don’t want to care about the backend.

---

### 2. `vector_base.py`  
**📐 Interface / Template / Blueprint**

- Defines **abstract class**: `VectorStoreBase`
- Declares methods like:
```python
def validate_index(self): ...
def index_documents(self): ...
def query(self, vector): ...
```

- FAISS and pgVector **must implement** these methods.

✅ Gives your app a consistent contract — works with any store as long as it follows this base.

---

### 3. `faiss_store.py`  
**🚀 FAISS Implementation**

- Handles:
  - loading JSON chunks
  - building FAISS indexes
  - saving them to S3
  - querying with top-k results
- Uses `faiss` CPU engine

✅ Fast, stateless search engine stored on S3

---

### 4. `pgvector_store.py`  
**📊 pgVector Implementation**

- Stores vectors in PostgreSQL (with pgvector extension)
- Implements `.validate()`, `.index()`, `.query()` using raw SQL
- Tracks metadata in secondary table (`pg_details`)

✅ Persistent, SQL-friendly vector storage

---

## 🧠 Summary Flow

| Layer             | Responsibility                             | Lives In                  |
|------------------|---------------------------------------------|---------------------------|
| `VectorManager`  | Chooses which backend to use                | `vector_manager.py`       |
| `VectorBase`     | Defines the abstract interface              | `vector_base.py`          |
| `FaissStore`     | Implements indexing/querying FAISS          | `faiss_store.py`          |
| `PgVectorStore`  | Implements indexing/querying pgVector       | `pgvector_store.py`       |

---
