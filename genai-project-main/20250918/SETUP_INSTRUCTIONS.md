# QA-Scoring Environment Setup Instructions

## Quick Start

### 1. Run the Setup Script
```bash
# Navigate to qa-scoring directory
cd /c/Users/marti/qa-scoring

# Minimal installation (core only)
python qa-scoring-setup.py --minimal

# Standard installation (core + web)
python qa-scoring-setup.py

# Full installation (everything)
python qa-scoring-setup.py --full

# Development installation (includes dev tools)
python qa-scoring-setup.py --dev
```

### 2. Manual Installation (Alternative)
```bash
# Install core requirements
pip install -r requirements.txt

# Install local packages in development mode
cd tlm && pip install -e . && cd ..
cd tidyllm-sentence && pip install -e . && cd ..
cd tidyllm && pip install -e . && cd ..
```

## Installation Options

### Minimal (--minimal)
**Core dependencies only - fastest install**
- TLM, tidyllm-sentence, tidyllm (local)
- requests, pyyaml, boto3, psycopg2, sqlalchemy
- polars, mlflow, dspy-ai

### Standard (default)
**Core + Web packages**
- Everything in minimal
- streamlit, flask, fastapi, uvicorn, plotly

### Full (--full)
**Everything including document processing**
- Everything in standard
- PyMuPDF, sentence-transformers, langchain
- reportlab, openai, psutil, schedule
- Development tools (pytest, black, flake8)

### Development (--dev)
**Standard + development tools**
- Everything in standard
- pytest, black, flake8

## Package Categories

### Local Development Packages
```
tlm                 # NumPy substitute (local dev install)
tidyllm-sentence    # Sentence processing (local dev install)
tidyllm            # Main framework (local dev install)
```

### Core Dependencies (Always Required)
```
requests           # HTTP requests
pyyaml            # Configuration files
boto3             # AWS SDK
botocore          # AWS SDK core
psycopg2-binary   # PostgreSQL adapter
sqlalchemy        # Database ORM
polars            # Fast DataFrame processing (replaces pandas)
mlflow            # ML lifecycle management
dspy-ai           # Language model programming
```

### Web & Visualization (Optional)
```
streamlit         # Dashboard framework
flask             # Web framework
fastapi           # Modern API framework
uvicorn           # ASGI server
plotly            # Interactive plots
```

### Document Processing (Optional)
```
PyMuPDF           # PDF processing (fitz)
sentence-transformers  # Text embeddings
langchain         # LLM applications
reportlab         # PDF generation
openai            # OpenAI API client
```

### System & Performance (Optional)
```
psutil            # System monitoring
schedule          # Task scheduling
```

### Development Tools (Optional)
```
pytest            # Testing framework
black             # Code formatter
flake8            # Linting
```

## Verification Commands

### Test Core Installation
```bash
cd tidyllm
python -c "import tlm; print(f'TLM v{tlm.__version__}')"
python -c "import polars as pl; print(f'Polars v{pl.__version__}')"
python -c "import tidyllm; print('TidyLLM ready!')"
```

### Test Web Features
```bash
python -c "import streamlit; print('Streamlit ready')"
python -c "import flask; print('Flask ready')"
```

### Test Document Processing
```bash
python -c "import fitz; print('PyMuPDF ready')"
python -c "import sentence_transformers; print('Sentence Transformers ready')"
```

## Key Changes from Previous Versions

### ✅ Pandas → Polars Migration Complete
- **Before:** `import pandas as pd`
- **After:** `import polars as pl  # Fast DataFrame processing`
- **Benefit:** 2-30x faster, 2-10x less memory

### ✅ NumPy → TLM Migration Complete
- **Before:** `import numpy as np`
- **After:** `import tidyllm.tlm as np  # TLM as numpy substitute`
- **Benefit:** No external NumPy dependency

### ✅ No Version Constraints
- **TLM:** No version requirements (uses local 1.2.0)
- **tidyllm-sentence:** No version requirements (uses local)
- **Benefit:** No dependency resolution warnings

## Directory Structure After Setup
```
qa-scoring/
├── requirements.txt           # Comprehensive package list
├── qa-scoring-setup.py       # Installation script
├── SETUP_INSTRUCTIONS.md     # This file
├── tlm/                      # Local TLM package
├── tidyllm-sentence/         # Local sentence processing
└── tidyllm/                  # Main TidyLLM framework
    ├── admin/
    │   ├── settings.yaml     # QA environment config
    │   └── aws_settings.yaml # AWS configuration
    ├── data/                 # Created by setup
    ├── logs/                 # Created by setup
    ├── cache/                # Created by setup
    └── mlruns/              # Created by setup
```

## Troubleshooting

### Common Issues

**1. Python Version Error**
```bash
# Ensure Python 3.8+
python --version
```

**2. Permission Errors**
```bash
# Use --user flag if needed
pip install --user -r requirements.txt
```

**3. Local Package Import Errors**
```bash
# Reinstall in development mode
cd tlm && pip install -e . --force-reinstall
```

**4. AWS/Database Connection Issues**
- Check `admin/settings.yaml` configuration
- Verify AWS credentials in `admin/aws_settings.yaml`
- Test database connection with provided credentials

### Getting Help
```bash
# Show setup script help
python qa-scoring-setup.py --help

# Test basic functionality
cd tidyllm
python simple_tests.py
```

## Next Steps After Installation

1. **Configure Environment**
   ```bash
   cd tidyllm
   # Check settings
   cat admin/settings.yaml
   ```

2. **Test Core Functionality**
   ```bash
   # Run basic tests
   python -c "import tidyllm; print('Success!')"
   ```

3. **Start Development**
   ```bash
   # Launch Streamlit dashboard
   streamlit run portals/flow/flow_creator_v3.py
   ```

Environment: **qa-scoring**
Root Path: **C:\Users\marti\qa-scoring\tidyllm**