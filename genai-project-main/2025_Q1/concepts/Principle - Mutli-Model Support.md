To scale toward multi-model, multi-store support — we need to:

1. **Lock down the assumptions**
2. **Prevent collisions**
3. **Document model-awareness at every layer**

---

## 🧱 Where We Must Lock or Declare Model Scope

### ✅ 1. `get_cached_vectorizer(model_key)`
🔐 **Assumption**: Each vectorizer is tied to a unique `model_key`  
🧠 **Fix**: Cache by `model_key`, not globally

```python
_vectorizer_cache = {}

def get_cached_vectorizer(model_key=None):
    model_key = model_key or get_config()["default_model"]
    if model_key not in _vectorizer_cache:
        _vectorizer_cache[model_key] = AmazonEmbeddingVectorizer.from_model_key(model_key)
    return _vectorizer_cache[model_key]
```

---

### ✅ 2. `VectorManager(store_type, config, clients)`
🔐 **Assumption**: Store manager is initialized **per store and per model**  
🧠 **Fix**: Optionally track `model_key` internally if needed by store

```python
self.model_key = model_key
```

---

### ✅ 3. `init_context.py → get_tab_context(store_type, model_key)`
🔐 **Assumption**: Config is shared, but vectorizer and manager are **per model_key**
🧠 **Fix**: Pass `model_key` to:
- `get_cached_vectorizer(model_key)`
- `VectorManager(..., model_key=model_key)`

---

### ✅ 4. `vector_interface.py` helpers
🔐 **Assumption**: You’re always using the current `model_key`
🧠 **Fix**: Let all functions accept `model_key`:
```python
def vector_index(model_key="titan_v1"):
    _, _, _, manager = get_tab_context(model_key=model_key)
    return manager.index_documents()
```

---

## 📄 Where to Document These Assumptions

### ✅ `embedding_helper.py`
- Cache must be per-model
- Normalize dimensions per model

### ✅ `init_context.py`
- All downstream functions tied to `model_key`
- Default fallback = `config["default_model"]`

### ✅ `vector_manager.py`
- Should route FAISS/pgVector by model-specific logic
- May hash indexes per (model_key, dimension)

---

## 🧠 Optional Enhancements

| Feature                          | Benefit                              |
|----------------------------------|---------------------------------------|
| `model_key_context` global       | Share across tabs in Streamlit       |
| `model_key` suffix in file keys  | Avoid overwrite on multi-model index |
| Vectorizer hash checker          | Detect dimension mismatches early    |

---

Other TODOs on our list:

- Add per-model caching in `embedding_helper.py`
- Patch `get_tab_context()` to handle `model_key`
- And update `vector_interface.py` to expose model_key-safe aliases?

This would lock the whole stack down to support multi-model, multi-store workflows with zero collisions.
