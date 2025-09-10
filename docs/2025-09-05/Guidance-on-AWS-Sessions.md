# TidyLLM AWS Sessions Management Guide

**Status**: ✅ **OPERATIONAL** - Unified session management architecture established  
**Last Updated**: September 7, 2025  
**Architecture**: UnifiedSessionManager pattern

---

## 🚨 **CRITICAL: ADMIN FOLDER FIRST - COMPLETE CREDENTIAL SOLUTION EXISTS**

### **⚠️ BEFORE WRITING ANY NEW CREDENTIAL CODE - CHECK ADMIN FOLDER ⚠️**

The `tidyllm/admin/` folder contains a **complete, tested, cross-platform credential management system**:

#### **✅ Complete Credential Infrastructure:**
- **`credential_loader.py`** - Python API for loading credentials from multiple sources
- **`set_aws_env.bat`** - Windows batch script with `set` commands
- **`set_aws_env.sh`** - Linux shell script with `export` commands  
- **`restart_aws_session.py`** - Full session restart and verification utility
- **`settings.yaml`** - YAML configuration with real AWS endpoints
- **`test_config.py`** - Comprehensive configuration testing

#### **✅ One-Line Solution for Unit Tests:**
```python
# ✅ CORRECT - Use existing credential loader
from tidyllm.admin.credential_loader import set_aws_environment
set_aws_environment()  # Automatically loads and sets environment variables

# ✅ CORRECT - Test credentials and connectivity
from tidyllm.admin.credential_loader import test_credentials
test_credentials()  # Tests loading, S3 access, and bucket connectivity
```

#### **✅ Cross-Platform Environment Persistence:**
```bash
# Windows
call tidyllm\admin\set_aws_env.bat

# Linux
source tidyllm/admin/set_aws_env.sh

# Python (works on both platforms)
python -c "from tidyllm.admin.credential_loader import set_aws_environment; set_aws_environment()"
```

#### **🎯 Why Unit Tests Failed (And How to Fix):**
Unit tests failed because they didn't load credentials into their process environment. The solution was already built - just import `credential_loader.py` at the top of test files.

**DO NOT CREATE NEW CREDENTIAL SOLUTIONS - USE THE EXISTING ADMIN INFRASTRUCTURE!**

---

## 🎯 **OVERVIEW**

This guide establishes the **official AWS session management architecture** for TidyLLM. All AWS operations, database connections, and MLflow tracking MUST use the UnifiedSessionManager pattern to ensure consistent credential management and prevent configuration conflicts.

---

## 🚨 **CRITICAL: One Session Manager Rule**

### **✅ THE OFFICIAL SESSION MANAGER**
```python
# ✅ CORRECT - The one and only session manager
from scripts.start_unified_sessions import UnifiedSessionManager

session_mgr = UnifiedSessionManager()
```

**Location**: `scripts/start_unified_sessions.py`  
**Purpose**: Consolidate ALL session management across TidyLLM  
**Status**: Official architecture pattern  

### **🚫 FORBIDDEN SESSION PATTERNS**
```python
# ❌ NEVER USE - Scattered session managers
from tidyllm.vectorqa.whitepapers.s3_session_manager import S3SessionManager
from tidyllm.knowledge_systems.core.s3_manager import SomeS3Manager
from tidyllm.admin.session_utils import DatabaseManager

# ❌ NEVER USE - Direct client creation
import boto3
s3_client = boto3.client('s3')  # Always use session_mgr.get_s3_client()

import psycopg2
conn = psycopg2.connect(...)  # Always use session_mgr.get_postgres_connection()
```

---

## 🏗️ **UNIFIED ARCHITECTURE COMPONENTS**

### **1. AWS S3 Operations**
```python
# ✅ CORRECT - S3 through UnifiedSessionManager
session_mgr = UnifiedSessionManager()

# Get S3 client
s3_client = session_mgr.get_s3_client()

# High-level operations
session_mgr.upload_to_s3("bucket-name", "key", data)
content = session_mgr.download_from_s3("bucket-name", "key")
files = session_mgr.list_s3_objects("bucket-name", "prefix/")

# Direct client operations
response = s3_client.put_object(
    Bucket="nsc-mvp1",
    Key="dropzones/document.pdf", 
    Body=file_data,
    ServerSideEncryption='AES256'
)
```

### **2. PostgreSQL Database Operations**
```python
# ✅ CORRECT - Database through UnifiedSessionManager
session_mgr = UnifiedSessionManager()

# Get database connection
conn = session_mgr.get_postgres_connection()

# Execute queries
results = session_mgr.execute_postgres_query(
    "SELECT * FROM document_embeddings WHERE created_date > %s",
    (datetime.now() - timedelta(days=7),)
)

# Bulk operations
session_mgr.bulk_insert_embeddings(embeddings_data)
```

### **3. MLflow Experiment Tracking**
```python
# ✅ CORRECT - MLflow through UnifiedSessionManager
session_mgr = UnifiedSessionManager()

# Get MLflow client
mlflow_client = session_mgr.get_mlflow_client()

# Log experiments
session_mgr.log_mlflow_experiment({
    "document_count": 150,
    "embedding_model": "tidyllm-sentence-tfidf",
    "processing_time": 45.2
})

# Track artifacts in S3
session_mgr.log_mlflow_artifact("model.pkl", "s3://nsc-mvp1/mlflow-artifacts/")
```

---

## ⚙️ **CONFIGURATION MANAGEMENT**

### **Environment Variables (Required)**
```bash
# AWS Credentials
export AWS_ACCESS_KEY_ID=***REMOVED***
export AWS_SECRET_ACCESS_KEY=***REMOVED***
export AWS_DEFAULT_REGION=us-east-1

# Database Configuration
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=mlflow
export POSTGRES_USER=mlflowuser
export POSTGRES_PASSWORD=mlflowpass

# MLflow Configuration  
export MLFLOW_TRACKING_URI=postgresql://mlflowuser:mlflowpass@localhost:5432/mlflowdb
export MLFLOW_S3_ENDPOINT_URL=https://s3.amazonaws.com
```

### **Unified Configuration Pattern**
```python
# ✅ CORRECT - Centralized configuration
UNIFIED_CONFIG = {
    "aws": {
        "region": "us-east-1",
        "default_bucket": "nsc-mvp1",
        "credentials_source": "environment",
        "encryption": "AES256"
    },
    "postgres": {
        "host": "localhost",
        "port": 5432,
        "database": "mlflow",
        "ssl_mode": "require",
        "connection_timeout": 30
    },
    "mlflow": {
        "tracking_uri": "postgresql://mlflowuser:mlflowpass@localhost:5432/mlflowdb",
        "artifact_root": "s3://nsc-mvp1/mlflow-artifacts/",
        "experiment_name": "tidyllm-processing"
    }
}
```

---

## 🚀 **COMMON USAGE PATTERNS**

### **Drop Zone File Processing**
```python
def process_dropzone_file(file_path, doc_type):
    """Process a dropped file through unified sessions"""
    
    session_mgr = UnifiedSessionManager()
    
    # 1. Upload to S3
    s3_key = f"dropzones/{datetime.now().strftime('%Y%m%d_%H%M%S')}/{doc_type}/{os.path.basename(file_path)}"
    session_mgr.upload_to_s3("nsc-mvp1", s3_key, file_path)
    
    # 2. Generate embeddings (using TidyLLM native stack)
    import tidyllm_sentence as tls
    content = session_mgr.download_from_s3("nsc-mvp1", s3_key)
    embeddings, model = tls.tfidf_fit_transform([content])
    
    # 3. Store in database
    session_mgr.execute_postgres_query(
        "INSERT INTO document_embeddings (s3_key, doc_type, embedding, created_date) VALUES (%s, %s, %s, %s)",
        (s3_key, doc_type, embeddings[0], datetime.now())
    )
    
    # 4. Track in MLflow
    session_mgr.log_mlflow_experiment({
        "s3_key": s3_key,
        "doc_type": doc_type,
        "embedding_dimensions": len(embeddings[0]),
        "processing_status": "completed"
    })
    
    return s3_key
```

### **Research Paper Processing**
```python
def process_research_paper(paper_content, metadata):
    """Process research paper with comprehensive tracking"""
    
    session_mgr = UnifiedSessionManager()
    
    # S3 storage with metadata
    paper_key = f"papers/{metadata['year']}/{metadata['title'].replace(' ', '_')}.pdf"
    session_mgr.upload_to_s3("nsc-mvp1", paper_key, paper_content)
    
    # Extract and embed mathematical formulations
    import tidyllm.tlm as np
    formulations = extract_math_content(paper_content)  # Custom function
    
    # Store research data
    session_mgr.execute_postgres_query(
        "INSERT INTO research_papers (s3_key, title, authors, year, math_formulations) VALUES (%s, %s, %s, %s, %s)",
        (paper_key, metadata['title'], metadata['authors'], metadata['year'], formulations)
    )
    
    # MLflow research tracking
    session_mgr.log_mlflow_experiment({
        "paper_title": metadata['title'],
        "formulation_count": len(formulations),
        "processing_framework": "Y=R+S+N",
        "storage_location": f"s3://nsc-mvp1/{paper_key}"
    })
```

---

## 🔧 **SETUP AND DEPLOYMENT**

### **Quick Setup Script**
```python
#!/usr/bin/env python3
"""setup_unified_sessions.py - Initialize unified session management"""

import os
import sys
from scripts.start_unified_sessions import UnifiedSessionManager

def setup_unified_sessions():
    """Setup and validate unified session management"""
    
    print("🚀 Setting up TidyLLM Unified Session Management...")
    
    # 1. Validate environment variables
    required_vars = [
        'AWS_ACCESS_KEY_ID',
        'AWS_SECRET_ACCESS_KEY', 
        'AWS_DEFAULT_REGION',
        'POSTGRES_HOST',
        'POSTGRES_USER',
        'POSTGRES_PASSWORD'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        print("Run: source tidyllm/admin/set_aws_env.sh")
        return False
    
    # 2. Test unified session manager
    try:
        session_mgr = UnifiedSessionManager()
        
        # Test S3
        s3_client = session_mgr.get_s3_client()
        buckets = s3_client.list_buckets()
        print(f"✅ S3 Connection: {len(buckets['Buckets'])} buckets accessible")
        
        # Test PostgreSQL
        conn = session_mgr.get_postgres_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT version()")
        db_version = cursor.fetchone()[0]
        print(f"✅ PostgreSQL Connection: {db_version[:50]}...")
        cursor.close()
        conn.close()
        
        # Test MLflow
        mlflow_client = session_mgr.get_mlflow_client()
        experiments = mlflow_client.list_experiments()
        print(f"✅ MLflow Connection: {len(experiments)} experiments found")
        
        print("🎉 Unified Session Management: FULLY OPERATIONAL")
        return True
        
    except Exception as e:
        print(f"❌ Session setup failed: {e}")
        return False

if __name__ == "__main__":
    success = setup_unified_sessions()
    sys.exit(0 if success else 1)
```

### **Integration Testing**
```python
def test_unified_session_integration():
    """Comprehensive integration test for unified sessions"""
    
    session_mgr = UnifiedSessionManager()
    
    # Test data
    test_data = {"test": "unified_session_integration", "timestamp": datetime.now().isoformat()}
    test_key = f"test/integration_{int(time.time())}.json"
    
    # 1. Test S3 upload/download cycle
    session_mgr.upload_to_s3("nsc-mvp1", test_key, json.dumps(test_data))
    downloaded = session_mgr.download_from_s3("nsc-mvp1", test_key)
    assert json.loads(downloaded) == test_data
    
    # 2. Test database operations
    session_mgr.execute_postgres_query(
        "INSERT INTO test_table (data, created_at) VALUES (%s, %s)",
        (json.dumps(test_data), datetime.now())
    )
    
    results = session_mgr.execute_postgres_query(
        "SELECT data FROM test_table WHERE data::json->>'test' = %s",
        (test_data['test'],)
    )
    assert len(results) > 0
    
    # 3. Test MLflow logging
    session_mgr.log_mlflow_experiment({
        "test_type": "unified_session_integration",
        "s3_operations": "successful",
        "database_operations": "successful"
    })
    
    print("✅ Unified session integration test: PASSED")
```

---

## 📋 **TROUBLESHOOTING**

### **Common Issues and Solutions**

#### **"AWS credentials not configured"**
```bash
# Solution 1: Set environment variables
source tidyllm/admin/set_aws_env.sh

# Solution 2: Verify credentials
echo $AWS_ACCESS_KEY_ID  # Should show AKIASXYZBZ...
python -c "import boto3; print(boto3.client('s3').list_buckets())"
```

#### **"PostgreSQL connection failed"**
```python
# Solution: Check database configuration
session_mgr = UnifiedSessionManager()
try:
    conn = session_mgr.get_postgres_connection()
    print("✅ Database connection successful")
except Exception as e:
    print(f"❌ Database connection failed: {e}")
    print("Check POSTGRES_* environment variables")
```

#### **"Multiple session managers detected"**
```bash
# Solution: Remove scattered session managers
find . -name "*session*manager*.py" -not -path "./scripts/start_unified_sessions.py"
# Review and consolidate any scattered implementations
```

### **Performance Monitoring**
```python
def monitor_session_performance():
    """Monitor unified session manager performance"""
    
    session_mgr = UnifiedSessionManager()
    
    # S3 operation timing
    start_time = time.time()
    session_mgr.upload_to_s3("nsc-mvp1", "test/performance.txt", "test data")
    s3_time = time.time() - start_time
    
    # Database operation timing  
    start_time = time.time()
    session_mgr.execute_postgres_query("SELECT 1")
    db_time = time.time() - start_time
    
    print(f"📊 Performance Metrics:")
    print(f"   S3 Upload: {s3_time:.3f}s")
    print(f"   DB Query:  {db_time:.3f}s")
    
    return {"s3_time": s3_time, "db_time": db_time}
```

---

## 🎯 **MIGRATION GUIDE**

### **Migrating from Scattered Session Managers**

#### **Before (Scattered Pattern)**
```python
# ❌ OLD - Multiple session managers
from tidyllm.vectorqa.whitepapers.s3_session_manager import S3SessionManager
from tidyllm.knowledge_systems.core.database_manager import DatabaseManager

s3_mgr = S3SessionManager()
db_mgr = DatabaseManager()

s3_mgr.upload_file("bucket", "key", data)
db_mgr.execute_query(sql)
```

#### **After (Unified Pattern)**
```python
# ✅ NEW - Single unified session manager
from scripts.start_unified_sessions import UnifiedSessionManager

session_mgr = UnifiedSessionManager()

session_mgr.upload_to_s3("bucket", "key", data)
session_mgr.execute_postgres_query(sql)
```

### **Migration Checklist**
- [ ] Replace all scattered session manager imports
- [ ] Update S3 operations to use `session_mgr.get_s3_client()`
- [ ] Update database operations to use `session_mgr.get_postgres_connection()`
- [ ] Update MLflow operations to use `session_mgr.get_mlflow_client()`
- [ ] Remove old session manager files
- [ ] Test unified session functionality
- [ ] Update documentation and examples

---

## 📚 **REFERENCES**

### **Key Files**
- **Main Implementation**: `scripts/start_unified_sessions.py`
- **Configuration Test**: `tidyllm/admin/test_config.py`
- **AWS Setup**: `AWS_CREDENTIALS_SETUP.md`
- **Architecture Constraints**: `IMPORTANT-CONSTRAINTS-FOR-THIS-CODEBASE.md`

### **Related Documentation**
- TidyLLM Drop Zones Architecture
- PostgreSQL MLflow Integration
- S3-First Processing Architecture
- TidyLLM Stack Constraints

### **Configuration Files**
- `tidyllm/admin/embeddings_settings.yaml`
- `drop_zones.yaml`
- Environment variable scripts in `tidyllm/admin/`

---

**🚨 CRITICAL REMINDER**: Always use UnifiedSessionManager. No exceptions. No additional session managers. This architecture prevents configuration conflicts and ensures consistent credential management across the entire TidyLLM ecosystem.

---

**Session Management Architecture**: ✅ **ESTABLISHED**  
**Integration Status**: ✅ **OPERATIONAL**  
**Migration Support**: ✅ **AVAILABLE**