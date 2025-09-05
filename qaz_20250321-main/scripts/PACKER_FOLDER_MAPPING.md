# Manual Packer - Folder Mapping Guide

## 🎯 **Quick Reference**

This guide maps the manual deployment packer script (`pack_files.py`) to the project folder structure, explaining what gets included in each package.

## 📦 **Package Contents Mapping**

### **1. Database Schemas Package (`database_schemas_*.zip`)**
```
📁 Source: database/
├── infra/                          # Infrastructure scripts
│   ├── 01_extensions.sql           # PostgreSQL extensions
│   ├── 02_review_system.sql        # Review system tables
│   ├── 03_embeddings_system.sql    # Embeddings and vectors
│   ├── 04_event_tracking.sql       # Event tracking tables
│   └── 05_mlflow_integration.sql   # MLflow integration tables
└── prod_scripts/                   # Production scripts
    └── 00_complete_setup.sql       # Complete setup wrapper
```

**Purpose**: Database setup and migration
**Deployment**: Run SQL scripts to set up PostgreSQL with pgvector

---

### **2. Credentials & Config Package (`credentials_config_*.zip`)**
```
📁 Source: environ_settings/
├── config.local.yaml               # Local environment config
├── config.staging.yaml             # Staging environment config
├── config.production.yaml          # Production environment config
├── place_settings_here.txt         # Settings template
├── unpack_environ.py               # Environment setup script
└── create_gitignore.py             # Git ignore generator

📁 Source: dev_configs/
├── qa_criteria_simplified.yaml     # Simplified QA criteria
└── qa_criteria_full.yaml           # Full QA criteria
```

**Purpose**: Environment setup and configuration
**Deployment**: Configure database connections, MLflow settings, API keys

---

### **3. Input Assets Package (`input_assets_*.zip`)**
```
📁 Source: input/
├── *.pdf                           # Demo documents (filtered)
└── omnibus/                        # Selected demo files only
    ├── Readme Rag Demo.pdf         # Essential demo files
    ├── Robot Presentation.pdf      # (when --include-demos)
    ├── Smart Fruit Ripeness System.pdf
    └── helper_functions.txt
```

**Purpose**: Demo and testing assets
**Deployment**: Documents for testing and demonstration
**Filtering**: Excludes large omnibus collections unless `--include-demos`

---

### **4. Core Application Package (`core_application_*.zip`)**
```
📁 Source: src/
├── main.py                         # Streamlit app entry point
├── backend/                        # Backend services and core logic
│   ├── core/                       # Core system components
│   ├── llm/                        # LLM gateway and integration
│   ├── mcp/                        # Model Context Protocol
│   └── processing/                 # Document processing workers
├── static/                         # Web application static assets
└── assets/                         # Report generation assets
    ├── prompts/                    # QA prompt templates
    └── *.tex                       # LaTeX report templates

📁 Source: scripts/
├── pack_files.py                   # This packer script
├── unpack_and_deploy.py            # Deployment script
└── [other utility scripts]

📄 Source: Root files
├── simple_demo.py                  # Main Streamlit demo
├── start_*.py                      # Application launchers
├── requirements*.txt               # Python dependencies
├── README*.md                      # Documentation
└── IMPORTANT_START_HERE.md         # Getting started guide

📁 Source: docs/
└── [organized documentation]
```

**Purpose**: Core application functionality
**Deployment**: Complete application with source code and documentation

---

## 🚫 **Excluded Folders**

### **Runtime Data (Regenerated)**
```
📁 data/                            # Runtime data, cache, experiments
📁 output/                          # Generated outputs
📁 logs/                            # Application logs
📁 test_outputs/                    # Test execution results
📁 rag_output/                      # RAG processing outputs
```

### **System & Maintenance**
```
📁 _archive/                        # Archived files
📁 optional/                        # Optional components
📁 .venv/                           # Python virtual environment
📁 .git/                            # Git repository
📁 notebooks/                       # Jupyter notebooks
📁 tests/                           # Test suite
```

### **Build Artifacts**
```
📁 __pycache__/                     # Python cache files
📁 *.pyc                            # Compiled Python files
📁 *.log                            # Log files
📁 .DS_Store                        # macOS system files
```

---

## 🔧 **Packer Script Options**

### **Basic Usage**
```bash
# Create all packages with default settings
python scripts/pack_files.py

# Clean site before packaging
python scripts/pack_files.py --clean

# Include demo files in input package
python scripts/pack_files.py --include-demos

# Verbose logging
python scripts/pack_files.py --verbose
```

### **Package Filtering Logic**

#### **Input Assets Filtering**
- **Default**: Excludes large omnibus collections
- **With `--include-demos`**: Includes essential demo files
- **Always excluded**: System files, cache, temporary files

#### **Core Application Filtering**
- **Always included**: Source code, requirements, documentation
- **Always excluded**: Runtime data, build artifacts, system files

---

## 📋 **Deployment Order**

1. **Database Schemas** → Set up PostgreSQL with pgvector
2. **Credentials & Config** → Configure environment settings
3. **Input Assets** → Add demo documents for testing
4. **Core Application** → Deploy application code and dependencies

---

## 🎯 **Migration Bundles vs Manual Packer**

| Aspect | Migration Bundles | Manual Packer |
|--------|------------------|---------------|
| **Contents** | Pre-built, tested packages | Current development state |
| **Structure** | Self-contained bundles | 4 separate packages |
| **Use Case** | Production deployments | Development/testing |
| **Filtering** | Pre-determined | Configurable via options |
| **Documentation** | Bundle-specific guides | Generic deployment guide |

---

## 📚 **Related Documentation**

- **[PROJECT_FOLDER_STRUCTURE.md](../docs/PROJECT_FOLDER_STRUCTURE.md)** - Complete folder structure guide
- **[NEW_GUY_S3_QA_GUIDE.md](../docs/NEW_GUY_S3_QA_GUIDE.md)** - New team member guide
- **[AWS_S3_NATIVE_QA_PLAN.md](../docs/planning/AWS_S3_NATIVE_QA_PLAN.md)** - Deployment strategy

---

**This mapping ensures the manual packer creates deployment packages that match the project's organized folder structure!** 🚀
