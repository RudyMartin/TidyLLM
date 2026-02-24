
## 🧠 Why Not Use `get_tab_context()` in Back-End Scripts?

### 1. `get_tab_context()` is **UI-aware**
It’s designed for Streamlit tabs or interactive app environments where:
- `st.session_state` might exist
- Context objects like `embed_manager` are **pre-wrapped for caching or logging**
- It pulls from global singletons or the user session

### 2. These scripts (like `create_pg_tables.py`) are **CLI scripts**
They:
- Run from terminal or cron
- Do not require or benefit from session-dependent logic
- Should only depend on **.env**, `get_config()`, and basic utility functions

### 3. Separation of concerns
- Use `get_tab_context()` in `Streamlit app tabs`
- Use `get_config()` and direct imports in **backend scripts**, batch jobs, and CLI tools

---

## ✅ When to Use `get_tab_context()` vs. `get_config()`

| Use case | Use `get_tab_context()` | Use `get_config()` |
|----------|--------------------------|---------------------|
| Streamlit tab (user chat, upload, search) | ✅ Yes | ✅ Yes |
| CLI scripts (table creation, indexing) | ❌ No | ✅ Yes |
| Python modules (embedding, logging) | ❌ No | ✅ Yes |
| Auto-injected vector manager / embed manager | ✅ Yes (UI) | ❌ No |

---

Let me know if you want me to:
- Patch in `.env` fallback to your roundtrip test
- Inject a clean version of `get_clients()` if needed
- Build a hybrid `get_context(is_ui=True)` wrapper for dual-mode flexibility
