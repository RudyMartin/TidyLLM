# 📘 Module: `proposed_init_context.py`

## 🔧 Purpose
The `proposed_init_context.py` module provides a **flexible and consistent way to initialize application context**, usable in both **Streamlit UI** and **CLI/scripts**. It returns configuration, AWS clients, and optionally model-aware tools (vector and embedding managers).

---

## 🚀 What’s New in This Version

### ✅ Key Enhancements
| Feature                         | Description |
|--------------------------------|-------------|
| `with_tools` flag              | Toggle for loading `vector_manager` and `embed_manager` |
| Supports multiple `model_key`s | Embeds are dynamically selected per model |
| Safe for both CLI and Streamlit| Avoids UI/session dependencies |
| NamedTuple: `TabContext`       | Unified object for accessing config, clients, tools |
| Dynamic access via `__getattr__` | Access `config[key]` or `clients.s3` transparently |

---

## 🆚 Difference from Old `get_tab_context()`

| Old Version                        | New Version                         |
|-----------------------------------|-------------------------------------|
| Always returns `clients`, `config` only | Returns optional `vector_manager` and `embed_manager` |
| No awareness of model key         | Accepts `model_key` override        |
| Used only in Streamlit tabs       | Now safe for CLI + tabs             |
| Hard to extend                   | Easily expandable with logging/agents |

---

## 🧱 Structure

```python
class TabContext(NamedTuple):
    clients: ClientBundle
    config: dict
    vector_manager: Optional[object]
    embed_manager: Optional[object]
```

You can access properties like:
```python
context.config["bucket_name"]
context.clients.s3
context.vector_manager
context.embed_manager
```

---

## ✅ Example Use Cases

### 📊 In Streamlit Tabs
```python
clients, config, vector_manager, embed_manager = get_tab_context(with_tools=True)
```

### 🔧 In CLI scripts
```python
clients, config, *_ = get_tab_context()  # No model tools needed
```

---

## ✅ Pros

| Benefit                         | Why It Matters |
|--------------------------------|----------------|
| 🔄 Reusable in any context     | Prevents duplicate code in tabs vs scripts |
| 🧠 Model-aware & flexible      | Works with `multi-model_key` setups |
| 🔌 Easily testable             | Independent of Streamlit session state |
| 💡 Extensible later            | Add audit loggers, tracing, user tokens, etc. |

---

## ❌ Cons / Considerations

| Limitation                       | Notes |
|----------------------------------|-------|
| Requires manual `with_tools=True` | Default context won’t auto-load embed tools |
| Can still be misused without type hints | Consider enforcing `model_key` logic in manager |
| Slightly more verbose for simple use | Use destructuring to reduce clutter |

---

## 🧩 Pages / Modules Affected (Once Integrated)

| File / Tab                         | How It Would Change |
|-----------------------------------|---------------------|
| `doc_pgvector_chat_tab.py`        | Replace `get_clients/get_config` with `get_tab_context(with_tools=True)` |
| `upload_embed_tab.py`             | Same replacement pattern |
| `pgvector_roundtrip_test.py`      | Can drop in `get_tab_context()` for lightweight config/clients |
| `create_pg_tables.py`             | Keep using only `get_config()` or `get_tab_context()` minimal |
| `embedding_helper.py` (deprecated) | Fully replaced by `vector_manager.get_embedder(...)` |

---
