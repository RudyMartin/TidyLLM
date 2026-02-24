###  Transparent Runtime Toggle

To meet your goal of **smart, seamless toggling** between CLI and Streamlit, here's what we'll do behind the scenes:

The updated `init_context.py` actually works under the hood — and here's why it's so effective in your CLI + Streamlit environment.

---

### 🔁 High-Level Flow

1. **Detect Runtime:**
   ```python
   def is_streamlit():
       return any("streamlit" in arg for arg in sys.argv)
   ```
   - Scans `sys.argv` to detect whether you're running under Streamlit (`streamlit run`).
   - CLI tools or tests won’t match — so it falls back safely.

2. **Load Context Accordingly:**
   ```python
   config = get_config() if is_streamlit() else load_config()
   ```
   - `get_config()` → for **Streamlit**, reuses the global singleton.
   - `load_config()` → for **CLI**, reads fresh from `config.json`.

3. **Initialize Dependencies:**
   ```python
   clients = get_clients(config)
   vectorizer = get_cached_vectorizer(model_key or config["default_model"])
   manager = VectorManager(store_type, config, clients)
   ```

4. **Return a Unified NamedTuple:**
   ```python
   return TabContext(...)
   ```
   - This gives you `.config`, `.s3`, `.vectorizer`, `.bucket_name`, etc. all in one object (`ctx`).

---

### 🧠 Why It’s Powerful

| Feature | Benefit |
|--------|---------|
| `is_streamlit()` | Auto-switching logic – no user flag needed |
| `TabContext` | One object with all necessary pieces |
| `__getattr__` override | Access `.bucket_name` instead of `ctx.config["bucket_name"]` |
| Works with both `streamlit run` and `python script.py` | ✅ Bulletproof environment handling |

---

### ✅ Example Usage

In any tab or CLI:
```python
from core.init_context import get_tab_context

ctx = get_tab_context(model_key="titan_v2")

# Access like this
print(ctx.bucket_name)
ctx.s3.list_objects_v2(Bucket=ctx.bucket_name, Prefix="dev/json/")
embedding = ctx.vectorizer.generate_embedding("Hello world")
```

---

Let me know if you want a CLI example or a test harness that calls this next.

🔄 Now it switches automatically between:
- **Singleton config for Streamlit** ✅
- **Fresh load for CLI** ✅

---

### 🧠 Your Benefit:
- No need to pass a mode flag
- No Streamlit-specific failures during CLI runs
- Cleaner demo/testing/dev workflows

---

