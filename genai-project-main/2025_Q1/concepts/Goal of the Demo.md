## 🎯 Goal of the Demo

You upload or pick a PDF, and the app:

1. Extracts + chunks the text
2. Embeds the chunks
3. Sends vectors to **FAISS** and/or **pgVector**
4. Runs a query and compares the results

---

## 💡 So Where Do the Core Modules Kick In?

Let’s map it to your app:

```python
from core.vector_manager import VectorManager
from core.embedding_helper import get_cached_vectorizer
```

---

### Step-by-Step 🔍

#### ✅ 1. **VectorManager Setup**
```python
clients, config, vectorizer, manager = get_tab_context()
```

- `get_tab_context()` calls:
  ```python
  manager = VectorManager(store_type, config, clients)
  ```

- You’re telling the system:
  > “Use FAISS or pgVector based on my dropdown / default”

✅ Behind the scenes, `VectorManager` chooses `FaissStore` or `PgVectorStore`.

---

#### ✅ 2. **Embedding**
```python
vectorizer = get_cached_vectorizer(model_key)
embedding = vectorizer.generate_embedding(query_text)
```

- Based on dropdown: `"titan_v1"` or `"pgmock_v1"` etc.
- This handles **real vs simulated embedding** using Bedrock or DB vectors

✅ Clean interface, LLM-agnostic

---

#### ✅ 3. **Query FAISS**
```python
faiss_results = manager.query("faiss", embedding)
```
- This will run:
  - `FaissStore.query(embedding)` → reads FAISS index from S3 and runs top-k

#### ✅ 4. **Query pgVector**
```python
pg_results = manager.query("pgvector", embedding)
```
- This will run:
  - `PgVectorStore.query(embedding)` → runs SQL `ORDER BY cosine_distance`

---

## 📂 Files That Work Together in the Demo

| File                      | Role                                |
|---------------------------|-------------------------------------|
| `demo_vector_sanity.py`   | UI logic                            |
| `vector_manager.py`       | Backend selector (FAISS or pgVector)|
| `vector_base.py`          | Shared interface                    |
| `faiss_store.py`          | FAISS backend                       |
| `pgvector_store.py`       | pgVector backend                    |
| `embedding_helper.py`     | Loads embedding models              |
| `embedding_manager.py`    | (Optional: bulk embedding pipeline) |

---

## ✅ Benefits of Using Core Architecture in Demo

- You can **switch vector stores without rewriting logic**
- Easy to compare results side by side
- Works across CLI, Streamlit, API

---

Want a test tab that runs `manager.validate()` or `manager.index_documents()` interactively next?
