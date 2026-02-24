## 🧠 **VectorQA – Model Risk Manager Tool**

---

## ✅ Overview: What Each Section Does

| **Section**                  | **Purpose**                                                                 |
|------------------------------|-----------------------------------------------------------------------------|
| 1. **Ingestion & Indexing** | Get documents into the system — extract, chunk, embed, and store them      |
| 2. **Semantic Search**       | Let users run similarity searches and compare models/vector stores         |
| 3. **Governance Review**     | Pre-review documents and audit models for completeness or compliance       |
| 4. **Governance Dashboards** | Visualize model QA progress, compliance stats, and emerging issue areas    |
| 5. **LLM & DSPy Self-Eval**  | Evaluate the system’s own AI agents and pipelines as if they’re models too |

---

## 📁 Refactored Tab Names + Feature Table

Here’s the updated set of **clear, renamed tab files** with suffix `_tab.py`:

---

### 1. Ingestion & Indexing

| New Tab Name               | Key Features                                                       |
|----------------------------|---------------------------------------------------------------------|
| `wizard_entry_tab.py`      | Choose path: upload → index or search → guided walk-through        |
| `pdf_preview_tab.py`       | View text/chunks of uploaded PDFs for troubleshooting              |
| `embed_index_tab.py`       | Upload → chunk → embed → index pipeline (S3, pgVector, FAISS)      |
| `cleanup_util_tab.py`      | Remove old PDFs, indexes, logs                                     |

---

### 2. Semantic Search

| New Tab Name               | Key Features                                                       |
|----------------------------|---------------------------------------------------------------------|
| `vector_sanity_tab.py`     | Run query on FAISS + pgVector → side-by-side results               |
| `doc_compare_tab.py`       | Compare two uploaded docs → semantic similarity                    |
| `model_compare_tab.py`     | Compare different model_keys for the same query                    |
| `guided_search_tab.py`     | Interactive, step-based semantic query search                      |
| `faiss_search_tab.py`      | Lightweight direct search into FAISS only                          |

---

### 3. Governance Review

| New Tab Name               | Key Features                                                       |
|----------------------------|---------------------------------------------------------------------|
| `model_plan_tab.py`        | Pre-review of Model Development Plan → completeness + comments     |
| `truth_loader_tab.py`      | Load QA pairs / structured truth into system                       |
| `audit_pg_tab.py`          | Run audit pipeline on pgVector-ingested content                    |
| `agent_audit_tab.py`       | Let DSPy/LLM agents perform section-level model audits             |

---

### 4. Governance Dashboards

| New Tab Name               | Key Features                                                       |
|----------------------------|---------------------------------------------------------------------|
| `model_dashboard_tab.py`   | Track which docs passed review, topic coverage, review timelines    |
| `log_explorer_tab.py`      | Semantic search into FAISS logs, filter by metadata, export         |
| `diagnostics_tab.py`       | Index health, errors, and configuration sanity checks               |
| `config_manager_tab.py`    | Admin UI for toggling models, store config, thresholds              |

---

### 5. LLM & DSPy Self-Eval

| New Tab Name               | Key Features                                                       |
|----------------------------|---------------------------------------------------------------------|
| `dspy_eval_tab.py`         | Breakdown subagent performance, trace DSPy runs                    |
| `self_audit_tab.py`        | Save this tool’s own pipeline as a model plan + submit to audit    |
| `agent_metrics_tab.py`     | Per-model or per-agent effectiveness summary                       |

---

## ✅ Summary

You now have a clear tab structure for:

- 🧠 Knowledge traceability  
- 📊 Governance dashboards  
- 🧾 Auditable document review  
- 🔍 Semantic search  
- 🧪 Model and tool self-evaluation  

---
