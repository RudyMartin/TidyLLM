**Option 1 Demo**

## 📁 **Application Structure – VectorQA Sage**

### 🧠 Core App & Tabs

| File | Purpose |
|------|---------|
| `app/fewshot_qa_app.py` | Main Streamlit Mega App |
| `app/tabs/fewshot_qa_app_tab1_upload_normalize.py` | Tab 1: Upload & Normalize |
| `app/tabs/fewshot_qa_app_tab2_split.py` | Tab 2: Split Dataset |
| `app/tabs/fewshot_qa_app_tab3_edit_examples.py` | Tab 3: Edit Examples |
| `app/tabs/fewshot_qa_app_tab4_dspy_prompt_config.py` | Tab 4: Prompt Strategy (LLM vs DSPy) |
| `app/tabs/fewshot_qa_app_tab5_evaluate.py` | Tab 5: Evaluate Models |
| `app/tabs/fewshot_qa_app_tab6_faiss_status.py` | Tab 6: FAISS Index/Model Status |
| `app/tabs/fewshot_qa_app_tab7_compile_dspy.py` | Tab 7: Compile DSPy Pipeline |

---

### 🔧 Utilities

| File | Purpose |
|------|---------|
| `core/config.py` | Centralized CONFIG dictionary (S3, models, index settings) |
| `core/faiss_model_mapper.py` | Mapping of FAISS index to model ID |
| `core/normalize_labels.py` | Label cleaner for validation results |
| `core/qa_log_utils.py` | Logging utility |
| `core/dspy_prompt_config.py` | Renders DSPy-friendly prompt formats |

---

### 📦 Scripts

| File | Purpose |
|------|---------|
| `scripts/admin_setup.py` | Initializes folders and FAISS mappings |
| `scripts/test_dspy_simulation.py` | Simulates prompt generation for each strategy |
| `compiled_modules/*.dspy` | Folder where compiled DSPy modules are saved |

---

### 📊 Evaluation + Documentation

| File | Purpose |
|------|---------|
| `data/examples_topic.json` | Sample labeled ground truth |
| `data/test_examples.json` | (Split version) |
| `data/validated_titan_v1.json` | Sample prediction file |
| `data/validated_cohere.json` | Sample prediction file |
| `data/faiss_model_map.json` | FAISS + model config |
| `docs/vectorqa_sage_quick_start_guide.json` | Structured quick start guide |
| `docs/vectorqa_sage_quick_start_guide.pdf` | Printable version |
| `docs/vectorqa_sage_quick_start_guide.docx` | Editable Word version |

---

### 📄 Project Meta

| File | Purpose |
|------|---------|
| `README.md` | Full usage instructions |
| `requirements.txt` | All Python dependencies |
| `setup.py` | Installable Python package template |
| `.gitignore` | Dev environment cleanup rules |

---

### 📦 Final Distribution ZIP

📥 [fewshot_qa_github_repo_final.zip](sandbox:/mnt/data/fewshot_qa_github_repo_final.zip)  
Includes everything above — organized for GitHub release or internal deployment.


