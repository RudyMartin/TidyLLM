# Project File Overview

## **Root Directory**

- **README.md**  
  General information and instructions about the project.
- **requirements.txt**  
  Python package dependencies.  
- **require_stable.txt**  
  Alternate or more stable requirements file (if you use it).
- **un_install.txt**  
  Notes/instructions related to uninstall or teardown steps.
- **setup_env.py**  
  A script for configuring or bootstrapping the environment.
- **main.py**  
  Potentially a primary entry point for running or demonstrating the app.
- **demo_main.py**  
  Demonstration app (Streamlit) showcasing dspy-based functionality.
- **check_files.py**  
  Script (possibly a validator) for scanning or checking file integrity/config.
- **__init__.py**  
  Marks this folder as a Python package (often empty).

> **Note**: Files with a `z` prefix (e.g., `za_demo_main.py`, `zb_demo_main.py`, etc.) or “untitled” in their name were removed by your **sweeper** script (or are duplicates/demos).

---

## **config/**

- **config.json**  
  Main application config (paths, credentials, environment variables).
- **default_config.json**  
  Fallback or reference configuration if `config.json` is missing.
- **old_config.txt**  
  Possibly an outdated or backup config file.

---

## **core/**

- **__init__.py**  
  Marks `core/` as a Python package.
- **config_helper.py**  
  Functions for reading/writing or validating config values.
- **extraction_helper.py**  
  Logic for extracting data (e.g., from PDFs, JSON, or text) before processing.
- **logging_helper.py**  
  Functions/wrappers to standardize logging across the app.
- **dspy_initializer.py**  
  Sets up or initializes older dspy components.
- **init_context.py**  
  Prepares a shared context for the application (e.g., global state, credentials).
- **embedding_helper.py / embedding_utils.py**  
  Embedding generation or vector-related utilities (integration with external models).
- **embedding_manager.py**  
  Manager that orchestrates embedding creation, caching, or retrieval.
- **faiss_store.py / pgvector_store.py**  
  Code to interact with Faiss (local vector DB) or PGVector (Postgres) for embeddings.
- **vector_manager.py**  
  Higher-level manager for vector searches, indexing, or retrieval.
- **client_bundle.py**  
  Potentially organizes external client connections (e.g., AWS or other APIs).
- **config_editor.py / config_manager.py**  
  Tools for editing or merging application config files.
- **model_key_utils.py**  
  Helpers for storing or retrieving model access keys/credentials.
- **faiss_log_writer.py / faiss_log_reader.py**  
  Specialized code to log or read Faiss indexing activity.
- **s3_utils.py**  
  AWS S3 read/write utilities.

---

## **tabs/**

- **\_\_init\_\_.py**  
  Marks `tabs/` as a package.
- **pdf_tab.py**  
  Streamlit tab for PDF ingestion or viewing.
- **logs_tab.py**  
  UI for viewing logs (if logs are rendered in the app).
- **batch_tab.py**  
  Possibly handles batch uploads or batch processes in the Streamlit UI.
- **config_tab.py**  
  UI for editing or displaying config settings.
- **faiss_diagnostics_tab.py**  
  Tools for diagnosing issues in Faiss-based vector storage.
- **guided_search_tab.py**  
  Streamlit tab for a “guided” approach to searching or querying embeddings.
- **plan_audit_tab.py**  
  A specialized tab for auditing or reviewing “plans” (model or pipeline docs).
- **playground_tab.py / truth_synth_tab.py / review_tab.py**  
  Various demonstration or development tabs (e.g., generating synthetic data, reviewing content).
- **admin_*_tab.py**  
  Admin-specific versions of the tabs with elevated control or debugging options (e.g., `admin_logs_tab.py`, `admin_pdf_tab.py`, etc.).
- **demo_*.py**  
  Demo or test variations of tabs (e.g., `demo_dashboard_tab.py`, `demo_guided_search_tab.py`).

> **Note**: Many “z” or “untitled” files were removed by the sweeper. If you see references to them, they are likely gone now.

---

## **test/**

- **test_embedding_helper.py**  
  Unit tests for embedding logic.
- **test_config_helper.py**  
  Unit tests for config-related functionality.
- **json/**  
  Folder containing sample or test JSON files (e.g., partial doc pages).

---

## **dsai/**

- **\_\_init\_\_.py**  
  Package marker.
- **\_dspy_keys.py / \_dspy_query_processing.py**  
  Possibly older or lower-level dspy code for key management or query flows.
- **dspy_pipeline_controller.py**  
  Main pipeline orchestrator in this older dspy structure.
- **dspy_query_processing.py**  
  Logic for processing user queries, possibly hooking into embeddings or LLM calls.
- **dspy_signature_helper.py / dspy_signature_s3_reader.py / dspy_signature_local_reader.py**  
  Tools and readers for handling “signatures” (some domain-specific concept) from S3 or local files.
- **dspy_report_correction_helper.py**  
  Could handle text corrections or formatting in reports.
- **dspy_trainset_loader.py**  
  Loads training data sets from local or remote sources.
- **dspy_section_auditor.py**  
  May handle compliance or auditing tasks for data “sections” in a pipeline.

### **dsai/modules/**

- **\_\_init\_\_.py**  
  Package marker for module subfolder.
- **base_module.py / risk_module.py / signatures.py**  
  Classes or logic for specialized domain tasks (risk scoring, signature analysis, etc.).

### **dsai/utils/**

- **\_\_init\_\_.py**  
  Package marker.
- **similarity_utils.py**  
  Utility functions to measure similarity or perform distance metrics on embeddings.

---

## **docs/**

- **verbose/**  
  Contains detailed `.docx` documents, possibly describing advanced model plans or internal architecture.  
- **standard/**  
  Common model documents (`.docx`) describing standard procedures or model plans.
- **high/**  
  High-level or advanced PDF docs describing specialized model workflows.
- **general/**  
  Additional PDF docs for various models (`Model_X_dev_plan.pdf` etc.).

---

## **groundtruth/**

- **strategic.json / operational.json / technical.json / model_development.json**  
  Ground truth data or reference sets for different aspects of the project.
- **high.json / standard.json / essential.json**  
  Additional ground truth or classification guidelines.

---

## Noteworthy Removals (By Sweeper)

- **\_\_pycache\_\_ Folders, .ipynb_checkpoints, logs/**  
  Auto-generated or debug folders that were removed.
- **All .log files**  
  Cleaned up by the sweeper to reduce clutter.
- **Many “z”\* or “untitled”\* files**  
  (e.g., `z__admin_pdf_tab.py`, `untitled2.py`) were also removed.

---

## Conclusion

This list gives you an **at-a-glance** view of the project’s core structure and purpose for each file. For a more in-depth explanation (e.g., how each module interacts, or how the Streamlit tabs are linked), you can expand on these descriptions in a larger “Developer Guide.”
