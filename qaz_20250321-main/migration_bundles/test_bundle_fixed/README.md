
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
- MCP (Model Context Protocol) Hierarchical LLM system

## 🚀 Quick Start

### Option 1: Using the Launcher Script (Recommended)
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python run_app.py
```

### Option 2: Direct Streamlit Command
```bash
# Install dependencies
pip install -r requirements.txt

# Run from project root
streamlit run src/main.py
```

### Option 3: From src directory
```bash
# Navigate to src directory
cd src

# Run streamlit
streamlit run main.py
```

## 📁 Project Structure
```
qaz_20250321/
├── src/                    # Main application source
│   ├── main.py            # Streamlit app entry point
│   ├── backend/           # Backend services and core logic
│   ├── static/            # Static assets
│   └── *.py               # Streamlit tab modules
├── docs/                  # Documentation
│   └── walkthrough/       # 📚 Comprehensive documentation walkthrough
├── tests/                 # Test suite
├── deploy/                # Deployment configurations
├── database/              # Database layer
├── input/                 # Input files
├── run_app.py             # Application launcher
└── requirements.txt       # Dependencies
```

## 📚 Documentation
- **[📖 Documentation Walkthrough](docs/walkthrough/README.md)** - Complete guide to all documentation
- **[🚀 Quick Start Guide](QUICKSTART_GUIDE.md)** - Simple guide to get started
- **[📋 QA Demo README](QA_DEMO_README.md)** - Detailed demo documentation

## 📁 App Tabs
1. Upload & Normalize
2. Split Dataset
3. Edit Examples
4. Prompt Config (DSPy)
5. Evaluate Models
6. FAISS & Model Status
7. Compile DSPy Module
8. Dashboard (Topic Accuracy)

## 🧰 Requirements
See `requirements.txt` for full dependencies.

## 📄 License
MIT – Provided by Next Shift Consulting
