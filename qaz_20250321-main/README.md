
# VectorQA Sage

A Streamlit-based RAG application for document analysis and AI-powered chat.

## 🚀 Quick Start

Choose your demo:

```bash
# Simple demo with favorites prompt
python start_simple.py

# Enhanced QA demo  
python start_enhanced.py

# Advanced multi-tab interface
python start_advanced.py
```

## 📋 What It Does

- **Upload Documents**: PDF and TXT files (up to 5)
- **AI Chat**: Ask questions about your documents
- **Smart Processing**: Advanced PDF extraction and analysis
- **Local AI**: Powered by ZLLM gateway

## 📁 Project Structure

```
├── start_simple.py              # Simple demo with favorites prompt (Port 8555)
├── start_enhanced.py            # Enhanced QA demo (Port 8502)
├── start_advanced.py            # Advanced multi-tab interface (Port 8501)
├── docs/                        # Detailed documentation
├── src/                         # Source code
└── database/                    # Database schemas
```

## 🌐 Port Assignments

Each demo runs on a different port to avoid conflicts:

- **Simple Demo**: http://localhost:8555
- **Enhanced Demo**: http://localhost:8502  
- **Advanced Demo**: http://localhost:8501

## 📚 Documentation

- **[User Guide](docs/user-guide/README.md)** - Complete user instructions
- **[Technical Docs](docs/technical/README.md)** - Developer documentation
- **[Architecture](docs/architecture/README.md)** - System architecture
- **[Database Setup](docs/database/README.md)** - Database setup and schemas
- **[Simple Demo](README_SIMPLE_DEMO.md)** - Quick demo guide

## 🧰 Requirements

- Python 3.7+
- All dependencies installed automatically by the launcher

## 📄 License

MIT – Provided by Next Shift Consulting
