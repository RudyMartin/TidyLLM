# TidyLLM Streamlit Demo Consolidation Guide

## 🚨 **PROBLEM SOLVED: No More Going in Circles!**

This document explains how we solved the scattered Streamlit demo chaos you identified:
- Multiple session management implementations (S3, PostgreSQL, Bedrock)
- Inconsistent demo launchers with different names
- Missing run-on-save functionality causing manual refresh hassles
- Confusing file organization with demos scattered everywhere

## ✅ **UNIFIED SOLUTION ARCHITECTURE**

### **1. Unified Session Management (`scripts/start_unified_sessions.py`)**
**ONE SESSION MANAGER TO RULE THEM ALL**

```python
# Before: Scattered session management
tidyllm-vectorqa/.../s3_session_manager.py         # S3 implementation #1
tidyllm/knowledge_systems/core/s3_manager.py       # S3 implementation #2  
Multiple files with different psycopg2 patterns    # PostgreSQL chaos
Various Bedrock credential approaches               # Bedrock confusion

# After: Single unified manager
from start_unified_sessions import get_global_session_manager

# Get any service client
session_mgr = get_global_session_manager()
s3_client = session_mgr.get_s3_client()
bedrock_client = session_mgr.get_bedrock_client()
postgres_conn = session_mgr.get_postgres_connection()
```

**Key Features:**
- **Single Credential Discovery**: Automatic detection from environment, AWS profiles, IAM roles
- **Connection Pooling**: PostgreSQL connection pool, S3/Bedrock client reuse
- **Health Monitoring**: Real-time health checks for all services
- **Error Handling**: Graceful degradation when services unavailable
- **Thread Safety**: Safe for concurrent Streamlit usage

### **2. Universal Streamlit Launcher (`scripts/start_streamlit_universal.py`)**
**STANDARDIZED NAMING AND BEHAVIOR**

```bash
# Before: Confusing launcher names
start_heiros.py
heiros_streamlit_demo.py  
launch_demo.py
run_demo.py
# ... different patterns everywhere

# After: Standardized pattern
start_whitepapers_demo.py
start_heiros_demo.py
start_vectorqa_demo.py
start_rag_demo.py
start_gateway_demo.py
start_ticker_demo.py
start_settings_demo.py
```

**Universal Features:**
- `--server.runOnSave=true` - **AUTO-RELOAD ON FILE CHANGES (No more manual refresh!)**
- Dynamic port discovery (no conflicts)
- Cross-platform process management
- Unified session integration
- Health monitoring for required services
- Consistent error handling and logging

### **3. Demo Registry System**
**SINGLE SOURCE OF TRUTH FOR ALL DEMOS**

```python
DEMO_REGISTRY = {
    "whitepapers": {
        "file": "tidyllm-whitepapers/streamlit_demo/app.py",
        "name": "TidyLLM Whitepapers Research", 
        "description": "Mathematical decomposition research papers",
        "services": [ServiceType.S3, ServiceType.POSTGRESQL]
    },
    "heiros": {
        "file": "heiros_streamlit_demo.py",
        "name": "TidyLLM-HeirOS Workflow Dashboard",
        "description": "Hierarchical workflow management", 
        "services": [ServiceType.POSTGRESQL]
    },
    # ... all demos registered here
}
```

## 📁 **PREFERRED DEMO VERSIONS (Consolidated)**

### **PRIMARY DEMOS (Keep These)**
1. **`start_whitepapers_demo`** - Mathematical research papers
   - **File**: `tidyllm-whitepapers/streamlit_demo/app.py`
   - **Services**: S3, PostgreSQL
   - **Status**: ✅ PREFERRED VERSION

2. **`start_heiros_demo`** - Hierarchical workflow management  
   - **File**: `heiros_streamlit_demo.py`
   - **Services**: PostgreSQL
   - **Status**: ✅ PREFERRED VERSION

3. **`start_vectorqa_demo`** - Vector-based question answering
   - **File**: `tidyllm-vectorqa/tidyllm_vectorqa/whitepapers/app.py` 
   - **Services**: S3, PostgreSQL
   - **Status**: ✅ PREFERRED VERSION

4. **`start_rag_demo`** - Retrieval-augmented generation
   - **File**: `tidyllm-vectorqa/tidyllm_vectorqa/whitepapers/streamlit_rag_demo.py`
   - **Services**: S3, PostgreSQL, Bedrock
   - **Status**: ✅ PREFERRED VERSION

### **SECONDARY DEMOS (Specialized Use Cases)**
5. **`start_gateway_demo`** - Gateway management dashboard
6. **`start_ticker_demo`** - Live data ticker dashboard  
7. **`start_settings_demo`** - System settings configuration

### **DEPRECATED DEMOS (Can Remove)**
```
tidyllm-vectorqa/tidyllm_vectorqa/whitepapers/business_analysis_rag.py
tidyllm-vectorqa/tidyllm_vectorqa/whitepapers/fixed_rag_app.py
tidyllm-vectorqa/tidyllm_vectorqa/whitepapers/simple_rag_demo.py
tidyllm-vectorqa/tidyllm_vectorqa/whitepapers/validation_qa_demo_app.py
tidyllm-vectorqa/tidyllm_vectorqa/whitepapers/app_fixed.py
tidyllm/run_demo.py
tidyllm/visual_demo.py
tidyllm/demo-standalone/run_demo.py
tidyllm/demo-standalone/visual_demo.py
# ... and other duplicates
```

## 🚀 **USAGE EXAMPLES**

### **Launch Any Demo with Auto-Reload**
```bash
# Navigate to root directory
cd C:\Users\marti\github

# Launch whitepapers demo with auto-reload
scripts/start_whitepapers_demo.py

# Launch with dependency installation
scripts/start_heiros_demo.py --install-deps

# Launch without logs (quiet mode)  
scripts/start_rag_demo.py --no-logs
```

### **Universal Launcher (Advanced)**
```bash
# Launch any demo by name
python scripts/start_streamlit_universal.py whitepapers
python scripts/start_streamlit_universal.py heiros --install-deps

# List all available demos
python scripts/start_streamlit_universal.py --help
```

### **Session Management in Your Code**
```python
# In any Streamlit demo, use unified sessions
from scripts.start_unified_sessions import get_global_session_manager

@st.cache_resource
def get_session_manager():
    return get_global_session_manager()

# Use in your demo
session_mgr = get_session_manager()

# S3 operations
s3_client = session_mgr.get_s3_client()
if s3_client:
    buckets = s3_client.list_buckets()

# PostgreSQL operations  
conn = session_mgr.get_postgres_connection()
if conn:
    # Use connection
    session_mgr.return_postgres_connection(conn)

# Bedrock operations
bedrock_client = session_mgr.get_bedrock_client() 
if bedrock_client:
    # Use Bedrock client
    pass
```

## 🔧 **CONFIGURATION**

### **Environment Variables**
```bash
# AWS Credentials (optional - auto-discovered)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_PROFILE=default

# PostgreSQL (recommended)
POSTGRES_PASSWORD=your_password
DATABASE_URL=postgresql://user:pass@host:port/db

# TidyLLM Settings (optional)
TIDYLLM_S3_BUCKET=your-default-bucket
TIDYLLM_AWS_REGION=us-east-1
```

### **Settings File Support**
```yaml
# tidyllm/admin/embeddings_settings.yaml
aws:
  default_bucket: "tidyllm-documents"  
  region: "us-east-1"

database:
  host: "localhost"
  port: 5432
  database: "tidyllm_db" 
  username: "postgres"
  # password loaded from environment
```

## ✅ **BENEFITS ACHIEVED**

### **For Users:**
- **No More Confusion**: Standard `start_{demo_name}_demo.py` pattern
- **Auto-Reload**: File changes automatically refresh demo (no manual F5!)
- **Consistent Behavior**: Same launch experience across all demos
- **Health Monitoring**: Clear status of all required services

### **For Developers:**
- **Single Session Management**: One place for all service connections
- **Unified Error Handling**: Consistent patterns across demos  
- **Easy Extension**: Add new demos by registering in DEMO_REGISTRY
- **Clean Architecture**: Clear separation of concerns

### **For Operations:**
- **Predictable Deployment**: Standard launcher patterns
- **Health Monitoring**: Built-in service health checks
- **Environment Integration**: Automatic credential discovery
- **Process Management**: Clean process lifecycle handling

## 🎯 **NEXT STEPS**

1. **Test New Launchers**: Try the new standardized launchers
2. **Remove Deprecated Demos**: Clean up old/duplicate demo files
3. **Update Documentation**: Point to new launcher patterns
4. **Team Training**: Share new standardized approach

## 🚨 **MIGRATION GUIDE**

### **Old Pattern → New Pattern**
```bash
# Old (scattered approaches)
python heiros_streamlit_demo.py
streamlit run tidyllm-vectorqa/.../app.py --port 8501
python tidyllm/run_demo.py

# New (standardized)  
scripts/start_heiros_demo.py
scripts/start_vectorqa_demo.py  
scripts/start_whitepapers_demo.py
```

### **Old Session Management → New Unified**
```python
# Old (scattered imports)
from tidyllm_vectorqa.whitepapers.s3_session_manager import S3SessionManager
from tidyllm.knowledge_systems.core.s3_manager import S3Manager
import psycopg2  # Direct usage

# New (unified)
from scripts.start_unified_sessions import get_global_session_manager

session_mgr = get_global_session_manager()
s3_client = session_mgr.get_s3_client()
conn = session_mgr.get_postgres_connection()
```

---

**🎉 PROBLEM SOLVED: No more going in circles with scattered sessions and inconsistent demos!**

**🔄 STREAMLIT RUN-ON-SAVE: Enabled across ALL demos - no more manual refresh hassles!**

**📝 STANDARDIZED NAMING: All demos follow `start_{demo_name}_demo.py` pattern**

**🔗 UNIFIED SESSIONS: One session manager for S3, PostgreSQL, and Bedrock across ALL demos**