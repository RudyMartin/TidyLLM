# TidyLLM Installation Guide

## Overview
TidyLLM is now a complete Python package that automatically handles all its dependencies when you import it.

## 🚀 **Quick Installation**

### **Method 1: Local Development Installation (Recommended)**
```bash
# From the tidyllm directory
pip install -e .
```

This installs TidyLLM in "editable" mode with all core dependencies.

### **Method 2: Full Installation with All Features**
```bash
# Install with all optional dependencies
pip install -e .[all]
```

This includes web interfaces, document processing, advanced AI features, and development tools.

### **Method 3: Selective Feature Installation**
```bash
# Install with specific feature sets
pip install -e .[web]          # Web interface and visualization
pip install -e .[data]         # Advanced data processing  
pip install -e .[documents]    # Document processing capabilities
pip install -e .[ai]           # Extended AI capabilities
pip install -e .[dev]          # Development and testing tools
```

## 📦 **What Gets Installed Automatically**

### **Core Dependencies (Always Installed):**
- `pyyaml` - YAML configuration processing
- `requests` - HTTP client for API calls
- `pandas` - Data processing and Excel handling
- `openpyxl` - Excel file processing
- `boto3` - AWS services integration  
- `psycopg2-binary` - PostgreSQL database connectivity
- `mlflow` - Experiment tracking and ML lifecycle
- `dspy-ai` - AI programming framework
- `openai` - OpenAI API client
- `anthropic` - Anthropic API client
- `psutil` - System monitoring

### **Optional Dependencies (Install with [feature]):**
- **Web features**: Streamlit, Plotly, Rich console
- **Data features**: NumPy, Polars, PyArrow
- **Document features**: PyPDF2, pdfplumber, python-docx
- **AI features**: Transformers, PyTorch, sentence-transformers
- **Dev features**: pytest, black, flake8, mypy

## 🎯 **Usage After Installation**

### **Simple Import:**
```python
import tidyllm

# All dependencies are now available
processor = tidyllm.init_gateways()
```

### **QA Processor:**
```bash
# After installation, just run:
python qa_processor.py --setup
python qa_processor.py --chat-pdf document.pdf
```

### **QA Test Runner:**
```bash
# All 5 test scenarios:
python qa_test_runner.py --create-samples
python qa_test_runner.py --all
```

### **Command Line Tools:**
After installation, these commands are available:
```bash
tidyllm --help              # Main TidyLLM CLI
tidyllm-demo               # Launch demo interface  
tidyllm-workflow           # Workflow management
qa-processor --help        # QA processing tool
```

## 🔧 **Verification**

### **Test Installation:**
```python
import tidyllm
print(f"TidyLLM version: {tidyllm.__version__}")
print(f"Gateways available: {tidyllm.GATEWAYS_AVAILABLE}")
print(f"Knowledge systems: {tidyllm.KNOWLEDGE_SYSTEMS_AVAILABLE}")
```

### **Test QA Processor:**
```bash
python qa_processor.py --debug-config
python qa_processor.py --chat-test  
```

### **Test Full System:**
```bash
python qa_test_runner.py --create-samples --verbose
python qa_test_runner.py --test 1
```

## 🚨 **Troubleshooting**

### **Missing Dependencies Error:**
```
[ERROR] Missing core dependency (should be installed with tidyllm)
```
**Solution:**
```bash
pip install -e .[all]
```

### **Import Error:**
```
[ERROR] TidyLLM not available. Please install: pip install -e .
```
**Solution:**
```bash
# Ensure you're in the right directory
cd /path/to/tidyllm
pip install -e .
```

### **Module Not Found:**
```bash
# Reinstall in development mode
pip uninstall tidyllm
pip install -e .[all]
```

### **AWS/Bedrock Issues:**
```bash
# Check your AWS credentials
python qa_processor.py --debug-config
python qa_processor.py --chat-test
```

## 🎨 **Development Installation**

### **For Contributors:**
```bash
# Clone and install in development mode
git clone https://github.com/tidyllm/tidyllm.git
cd tidyllm
pip install -e .[dev,all]

# Install pre-commit hooks
pre-commit install

# Run tests
pytest
```

### **Build Package:**
```bash
# Build distribution packages
python -m build

# Install from built package
pip install dist/tidyllm-*.whl
```

## 📋 **System Requirements**

- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, Linux  
- **Memory**: 4GB+ recommended
- **Disk**: 2GB+ for full installation with AI features
- **AWS Account**: For Bedrock functionality (optional)
- **PostgreSQL**: For database features (optional)

## 🎯 **Quick Start After Installation**

```bash
# 1. Install TidyLLM
pip install -e .[all]

# 2. Set up QA processor  
python qa_processor.py --setup

# 3. Test connection
python qa_processor.py --chat-test

# 4. Process your first PDF
python qa_processor.py --chat-pdf your_document.pdf

# 5. Run comprehensive tests
python qa_test_runner.py --all
```

## ✅ **What This Achieves**

After installation:
- ✅ **Single Import**: `import tidyllm` gives you everything
- ✅ **Automatic Dependencies**: All required packages installed
- ✅ **Command Line Tools**: Ready-to-use CLI commands
- ✅ **No Manual Setup**: Package handles configuration
- ✅ **Optional Features**: Install only what you need
- ✅ **Development Ready**: Editable installation for development

**Result**: `import tidyllm` now automatically provides all dependencies and functionality! 🚀