# 📘 Vector Store Method Overview

### ✅ Supported Operations (`CRUDV` + extended helpers)

---

## 🔄 `index_create`

| Layer              | Function Signature                             | Purpose                              |
|-------------------|--------------------------------------------------|--------------------------------------|
| `vector_interface`| N/A – internal call in `vector_index()`          | Calls `manager.index_documents()`    |
| `VectorManager`   | `index_documents(model_key)`                     | Calls `.index_create(...)` internally|
| `BaseVectorStore` | `index_create(embeddings, model_key, metadata)` | Create a new vector index            |
| `PGVectorStore`   | ✅ Implemented                                   | INSERTs into `model_embeddings`      |
| `FaissStore`      | ✅ Implemented                                   | Builds FAISS and saves to S3         |

📎 **Example:**
```python
vector_index("titan_v2")
```

---

## 📥 `collect_embeddings_and_metadata`

| Layer              | Function Signature                             | Purpose                                      |
|-------------------|--------------------------------------------------|----------------------------------------------|
| `BaseVectorStore` | `collect_embeddings_and_metadata(model_key)`    | Pull data from JSON (via `data_collect`)     |
| `PGVectorStore`   | ✅ Implemented (calls shared helper)            | Collects from S3 before INSERT               |
| `FaissStore`      | ✅ Implemented (calls shared helper)            | Collects from S3 before training             |

📎 **Example:**
```python
embeds, meta = store.collect_embeddings_and_metadata("titan_v2")
```

---

## 🧪 `validate`

| Layer              | Function Signature                | Purpose                              |
|-------------------|------------------------------------|--------------------------------------|
| `vector_interface`| `vector_validate(model_key)`       | Public API call                      |
| `VectorManager`   | `validate(model_key)`              | Delegates to store                   |
| `BaseVectorStore` | `validate(model_key)`              | Abstract check for index validity    |
| `PGVectorStore`   | ✅ Check row count                 | SELECT COUNT(*) WHERE model_key=... |
| `FaissStore`      | ✅ Check for .faiss file on S3     | List S3 keys and verify file match   |

📎 **Example:**
```python
if vector_validate("titan_v2"):
    print("Index exists!")
```

---

## 🔁 `reindex`

| Layer              | Function Signature                | Purpose                              |
|-------------------|------------------------------------|--------------------------------------|
| `vector_interface`| `vector_reindex()`                 | Top-level trigger                    |
| `VectorManager`   | `reindex()`                        | Calls store.reindex()                |
| `BaseVectorStore` | `reindex()`                        | Abstract                              |
| `PGVectorStore`   | ✅ Chooses structured or default   | S3/JSON or structured insert mode     |
| `FaissStore`      | ✅ Calls embeddings → index_create | Full pipeline to rebuild index        |

📎 **Example:**
```python
vector_reindex()
```

---

## 🔍 `query`

| Layer              | Function Signature                | Purpose                              |
|-------------------|------------------------------------|--------------------------------------|
| `vector_interface`| `vector_query(query_embedding)`    | Query embedding and get top_k        |
| `VectorManager`   | `query(query_embedding, top_k)`    | Forwards to store                    |
| `BaseVectorStore` | `query(query_embedding, top_k)`    | Abstract method                      |
| `PGVectorStore`   | ✅ Uses `<#>` L2 operator          | Returns top-k rows from DB           |
| `FaissStore`      | ✅ Uses FAISS index                | Returns top-k from in-memory search  |

📎 **Example:**
```python
results = vector_query([0.01]*1024, top_k=5)
```

---

## 🧹 `delete_index`

| Layer              | Function Signature                | Purpose                              |
|-------------------|------------------------------------|--------------------------------------|
| `vector_interface`| `vector_delete()`                  | Public delete call                   |
| `VectorManager`   | `delete_index()`                   | Delegates to store                   |
| `BaseVectorStore` | `delete_index()`                   | Abstract                             |
| `PGVectorStore`   | ✅ Deletes all rows in table       | `DELETE FROM model_embeddings`       |
| `FaissStore`      | ✅ Not implemented yet             | (Stub exists)                        |

📎 **Example:**
```python
vector_delete()
```

---

## 📦 `load_index`

| Layer              | Function Signature                | Purpose                              |
|-------------------|------------------------------------|--------------------------------------|
| `vector_interface`| `vector_load()`                    | Loads index for interactive use      |
| `BaseVectorStore` | `load_index()`                     | Abstract                             |
| `PGVectorStore`   | ✅ No-op (uses live DB)            | Returns `True`                       |
| `FaissStore`      | ✅ Loads FAISS from S3             | `deserialize_index()`                |

📎 **Example:**
```python
vector_load()
```

---

## 📃 `get_index_status`

| Layer              | Function Signature                | Purpose                              |
|-------------------|------------------------------------|--------------------------------------|
| `BaseVectorStore` | `get_index_status()`               | Abstract                             |
| `PGVectorStore`   | ✅ Count rows + model_keys         | Uses `GROUP BY` query                |
| `FaissStore`      | ✅ S3 inspection and deserialization | Returns badge status + last update |

📎 **Example:**
```python
status = manager.get_index_status()
```

---

Would you like this exported as a Markdown `.md` file or turned into a PDF for doc sharing?
