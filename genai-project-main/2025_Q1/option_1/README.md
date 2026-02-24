
# VectorQA Sage (Option 2)

VectorQA Sage is a Streamlit-based application for evaluating LLM outputs using DSPy pipelines, FAISS indexing, and topic-based validation. It supports prompt strategy tuning, compiled modules, and performance visualization.

## 🧠 Features
- Upload and normalize labeled QA examples
- Train/test split and manual editing
- Prompt strategy configuration with DSPy/LLM toggle
- Compile DSPy modules with few-shot optimization
- Evaluate results (accuracy, confusion matrix, topic winners)
- View FAISS/model status
- Save/load DSPy pipelines

## 🚀 Quick Start

```bash
pip install -r requirements.txt
streamlit run app/fewshot_qa_app.py
```

## 📁 App Tabs
1. Upload & Normalize
2. Split Dataset
3. Edit Examples
4. Prompt Config (DSPy)
5. Evaluate Models
6. FAISS & Model Status
7. Compile DSPy Module

## 🧰 Requirements
See `requirements.txt` for full dependencies.

## 📄 License
MIT – Provided by Next Shift Consulting
