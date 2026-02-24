

## 🧩 Configuration Management Overview

This explains how configuration is handled consistently across your codebase using a layered approach.

---

### 🔧 Levels of Code and Config Usage

| **Level**         | **Examples**                         | **Config Source**         | **Notes**                                                                 |
|------------------|--------------------------------------|----------------------------|----------------------------------------------------------------------------|
| `UI`             | `admin_config_tab.py`, Streamlit tabs | `from config_manager import get_config()` | Uses global app config automatically from session or disk |
| `CLI Scripts`    | `debug_*.py`, batch runners           | `from config_manager import get_config()` | CLI-safe and centralized loading |
| `Pipelines`      | Vector embedding, agent orchestration | `from config_manager import get_config()` | Easy to switch or validate environment config |
| `Core Utils`     | `s3_utils.py`, `faiss_helper.py`      | `config` passed explicitly | Avoids circular imports, stays lightweight |
| `Model Modules`  | Vectorizers, chunkers, etc.           | `config` passed explicitly | Stay reusable, testable, decoupled from system context |

---

### 🧱 Config Loading Functions

| **Function**            | **Where Defined**         | **Purpose**                                 | **Who Should Use It**             |
|-------------------------|---------------------------|----------------------------------------------|-----------------------------------|
| `initialize_config()`   | `core/config_helper.py`   | Loads and patches `session_state["CONFIG"]` | Used internally everywhere        |
| `get_config()`          | `core/config_manager.py`  | Calls `initialize_config()` and returns config | UI, CLI, and pipelines            |
| `save_config()`         | `core/config_helper.py`   | Writes config to disk                        | UI/admin tools only               |
| `reload_config()`       | (Deprecated)              | ✅ Replaced by `initialize_config()`         | —                                 |

---

### 🧭 Naming Convention for Clarity

| **Name Pattern**        | **Used For**              | **Comment**                                    |
|-------------------------|---------------------------|------------------------------------------------|
| `get_config()`          | High-level config entry   | Safe for pipelines, CLI, and UI                |
| `initialize_config()`   | Low-level setup           | Internal bootstrapping; called only once       |
| `config`                | Local variable name       | Holds the final loaded dictionary              |

---

### 🔄 Flowchart of a Typical Load

```plaintext
UI/CLI Entry
   │
   └──> get_config()          ← Recommended
         │
         └──> initialize_config()
                 │
                 └──> st.session_state["CONFIG"]
```

---

### 🧪 Debugging Config Issues

| **Symptom**                             | **Possible Cause**                            | **Fix**                                      |
|-----------------------------------------|-----------------------------------------------|----------------------------------------------|
| `KeyError: 'CONFIG'`                    | `initialize_config()` not called              | Always use `get_config()` in UI/CLI          |
| `ImportError from config_helper`        | Using deprecated `get_config()` or `reload_config()` | Replace with `initialize_config()` or `get_config()` |
| Config not updating after UI changes    | `session_state["CONFIG"]` not refreshed       | Call `initialize_config()` again             |

---

Would you like a diagram or PDF version of this for onboarding or slide decks?
