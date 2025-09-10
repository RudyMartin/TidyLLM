# 🚨 CRITICAL: TidyLLM Stack Constraints

**⚠️ READ THIS FIRST - MANDATORY FOR ALL CONTRIBUTORS ⚠️**

This document defines the **non-negotiable architectural constraints** for TidyLLM. These are not suggestions - they are **fundamental principles** that must be followed in ALL code, examples, documentation, and implementations.

---

## 🤖 **ATTENTION AI AGENTS & DEVELOPERS - MUST READ ALL GUIDANCE**

### **🔴 CRITICAL: READ THESE DOCUMENTS BEFORE ANY WORK 🔴**

**AI AGENTS/ASSISTANTS**: You MUST read and understand ALL of these guidance documents before making ANY changes to this codebase. These are NOT optional - they define the architecture, constraints, and patterns that MUST be followed.

### **Priority 1: Core Constraints & Architecture (READ FIRST)**
- **[THIS DOCUMENT - IMPORTANT-CONSTRAINTS-FOR-THIS-CODEBASE.md](./IMPORTANT-CONSTRAINTS-FOR-THIS-CODEBASE.md)** - YOU ARE HERE - Stack constraints and forbidden dependencies
- **🔴 CRITICAL: ALWAYS CHECK `tidyllm/admin/` FOLDER FIRST** - Contains restart_aws_session.py, settings.yaml, and ALL connection utilities
- **[CRITICAL_DESIGN_DECISIONS.md](./CRITICAL_DESIGN_DECISIONS.md)** - Architectural conflicts and mandatory decision points
- **[Guidance-on-AWS-Sessions.md](./Guidance-on-AWS-Sessions.md)** - AWS session management patterns
- **[Session-Management-Migration-Map.md](./Session-Management-Migration-Map.md)** - UnifiedSessionManager migration guide

### **Priority 2: Gateway & System Architecture**
- **[/2025-09-04/GATEWAY_ARCHITECTURE_OVERVIEW.md](../2025-09-04/GATEWAY_ARCHITECTURE_OVERVIEW.md)** - Three-tier gateway hierarchy
- **[/2025-09-04/UNIFIED_GATEWAY_ARCHITECTURE.md](../2025-09-04/UNIFIED_GATEWAY_ARCHITECTURE.md)** - Unified gateway patterns
- **[/2025-09-04/S3_FIRST_ARCHITECTURE_GUIDANCE.md](../2025-09-04/S3_FIRST_ARCHITECTURE_GUIDANCE.md)** - S3-First processing requirements
- **[/2025-09-04/DROP_ZONES_ARCHITECTURE.md](../2025-09-04/DROP_ZONES_ARCHITECTURE.md)** - Drop zone architecture patterns

### **Priority 3: Implementation Guidance**
- **[/2025-09-04/GUIDANCE_ON_MODEL_SELECTION.md](../2025-09-04/GUIDANCE_ON_MODEL_SELECTION.md)** - Model selection criteria
- **[/2025-09-04/Guidance-on-Using-MCP.md](../2025-09-04/Guidance-on-Using-MCP.md)** - MCP protocol usage
- **[/2025-09-04/DOCUMENT_COLLECTIONS_SYSTEM.md](../2025-09-04/DOCUMENT_COLLECTIONS_SYSTEM.md)** - Document handling patterns
- **[/2025-09-04/RAG2DAG_DOCUMENTATION.md](../2025-09-04/RAG2DAG_DOCUMENTATION.md)** - RAG to DAG conversion patterns

### **Priority 4: System Components**
- **[/2025-09-03/TIDYLLM_ECOSYSTEM_ARCHITECTURE.md](../2025-09-01/TIDYLLM_ECOSYSTEM_ARCHITECTURE.md)** - Complete ecosystem overview
- **[/2025-09-03/COMPLETE_ARCHITECTURE_DIAGRAMS.md](../2025-09-03/COMPLETE_ARCHITECTURE_DIAGRAMS.md)** - Architecture diagrams
- **[/2025-09-03/ARCHITECTURE_INDEX.md](../2025-09-03/ARCHITECTURE_INDEX.md)** - Architecture component index
- **[/2025-09-04/TIDYLLM_VISITOR_EXPERIENCE_DESIGN.md](../2025-09-04/TIDYLLM_VISITOR_EXPERIENCE_DESIGN.md)** - User experience design

### **🛑 STOP: Verification Checklist for AI Agents**
Before making ANY changes, verify you have:
- [ ] Read ALL Priority 1 documents completely
- [ ] Understood the forbidden dependencies (numpy, pandas, sklearn, etc.)
- [ ] Understood the required substitutions (tidyllm.tlm, tidyllm_sentence, polars)
- [ ] Read the gateway architecture documents
- [ ] Understood S3-First processing requirements
- [ ] Reviewed UnifiedSessionManager patterns
- [ ] Checked file organization constraints (/scripts, /docs, never in root)

**AI AGENTS**: If you have NOT read these documents, STOP and read them NOW. Making changes without understanding these constraints will create technical debt and architectural violations.

---

## 🚫 **FORBIDDEN DEPENDENCIES** 

### **NEVER USE THESE:**
```python
# ❌ FORBIDDEN - Big Tech ML Dependencies
import numpy           # Use tidyllm.tlm instead
import pandas          # Use polars instead  
import sklearn         # Use tidyllm.tlm algorithms instead
import sentence_transformers  # Use tidyllm_sentence instead
import tensorflow      # Use tidyllm.tlm instead
import pytorch         # Use tidyllm.tlm instead
import transformers    # Use tidyllm_sentence instead
```

### **Why These Are Banned:**
- **Infrastructure Sovereignty**: Complete independence from Big Tech ML frameworks
- **Transparency**: Every operation must be readable and modifiable
- **Educational Mission**: Code teaches concepts, not black boxes
- **Memory Efficiency**: Avoid bloated corporate frameworks
- **Vendor Lock-in Avoidance**: Maximum portability and user control

---

## ✅ **REQUIRED STACK SUBSTITUTIONS**

### **Math & Linear Algebra:**
```python
# ✅ CORRECT - TidyLLM Native
import tidyllm.tlm as np  # Pure Python numpy substitute
# - 100% readable list-based operations
# - Zero external dependencies 
# - Educational transparency
# - Complete user ownership

# Example usage:
data = np.array([[1, 2], [3, 4]])           # Pure Python lists
normalized = np.l2_normalize(data)          # Transparent normalization
similarity = np.dot(vec1, vec2)             # Readable dot product
```

### **Embeddings & NLP:**
```python
# ✅ CORRECT - TidyLLM Sentence
import tidyllm_sentence as tls
# - 77% of sentence-transformers quality
# - 177x less memory usage
# - Zero external ML dependencies
# - Pure Python implementation
# - Complete algorithmic transparency

# Example usage:
embeddings, model = tls.tfidf_fit_transform(sentences)
similarity = tls.cosine_similarity(emb1, emb2)
results = tls.semantic_search(query_emb, corpus_embs)
```

### **Data Processing:**
```python
# ✅ CORRECT - Polars
import polars as pl
# - Faster than pandas
# - Memory efficient
# - Better API design
# - Rust-powered performance

# Example usage:
df = pl.DataFrame(data)
result = df.filter(pl.col("score") > 0.5)
```

---

## 🏗️ **ARCHITECTURE PRINCIPLES**

### **1. Infrastructure Sovereignty**
- **Goal**: Complete independence from corporate ML frameworks
- **Implementation**: Use only TidyLLM native stack
- **Benefit**: Users own their entire ML pipeline

### **2. Transparency First** 
- **Goal**: Every operation is readable and modifiable
- **Implementation**: Pure Python algorithms in `tlm`
- **Benefit**: Educational value and complete understanding

### **3. Memory Efficiency**
- **Goal**: Minimal resource footprint
- **Implementation**: Lightweight alternatives (tidyllm-sentence vs sentence-transformers)
- **Benefit**: Runs anywhere, edge deployment ready

### **4. Zero Vendor Lock-in**
- **Goal**: Maximum portability
- **Implementation**: Standard library + minimal dependencies
- **Benefit**: Code works anywhere, forever

---

## 🔧 **CRITICAL: ALWAYS CHECK `tidyllm/admin/` FOLDER FIRST**

**⚠️ MANDATORY FOR ALL DEVELOPERS & AI AGENTS ⚠️**

Before attempting ANY AWS connections, database connections, or system setup, **ALWAYS CHECK THE ADMIN FOLDER FIRST**:

### **🚨 Admin Folder Contains ALL Connection Utilities:**
- **`tidyllm/admin/restart_aws_session.py`** - AWS session restart and authentication utility
- **`tidyllm/admin/settings.yaml`** - REAL AWS RDS, S3, and Bedrock configuration 
- **`tidyllm/admin/credential_loader.py`** - Credential loading and management
- **`tidyllm/admin/config_manager.py`** - Configuration management utilities
- **`tidyllm/admin/gateway_control.py`** - Gateway control and monitoring
- **`tidyllm/admin/set_aws_*.py/bat/sh`** - AWS environment setup scripts
- **`tidyllm/admin/run_diagnostics_real.py`** - Cross-platform system diagnostic script (Windows + Linux/SageMaker)
- **`tidyllm/admin/test_cross_platform.py`** - Cross-platform compatibility verification

### **🔑 REAL CONFIGURATION DISCOVERED:**
From `tidyllm/admin/settings.yaml`:
```yaml
# REAL AWS RDS DATABASE (NOT localhost!)
postgres:
  host: "vectorqa-cluster.cluster-cu562e4m02nq.us-east-1.rds.amazonaws.com"
  db_name: "vectorqa"
  db_user: "vectorqa_user"
  ssl_mode: "require"

# REAL S3 BUCKET
s3:
  region: "us-east-1"
  bucket: "nsc-mvp1"  # Primary production bucket
```

### **⚡ STEP 1 FOR ANY SETUP:**
```bash
# ALWAYS START HERE - NOT with fake localhost configs!
cd tidyllm/admin/

# Run cross-platform diagnostic (works on Windows + Linux/SageMaker)
python run_diagnostics_real.py

# Alternative: Run existing admin tests
python restart_aws_session.py --verify  
python test_config.py
```

### **🚫 WHAT NOT TO DO:**
- ❌ Don't create fake settings.yaml in root
- ❌ Don't use localhost database connections
- ❌ Don't assume credentials need to be configured
- ❌ Don't create mock AWS services

### **✅ WHAT TO DO:**
- ✅ Use `tidyllm/admin/settings.yaml` for real config
- ✅ Run `restart_aws_session.py` to fix authentication
- ✅ Check `admin/` folder for existing utilities before writing new ones
- ✅ Use real RDS endpoint: `vectorqa-cluster.cluster-cu562e4m02nq.us-east-1.rds.amazonaws.com`
- ✅ Use real S3 bucket: `nsc-mvp1`

**🎯 BOTTOM LINE**: The admin folder contains PRODUCTION-READY connection utilities and REAL configuration. Use them!

---

## 📋 **IMPLEMENTATION CHECKLIST**

Before writing ANY code, verify:

### ✅ **Stack Compliance:**
- [ ] Using `tidyllm.tlm` instead of `numpy`
- [ ] All arrays are Python lists, not numpy arrays
- [ ] Linear algebra uses `tlm.dot()`, `tlm.matmul()`, etc.
- [ ] Using `tidyllm_sentence` instead of `sentence_transformers`
- [ ] TF-IDF, word averaging, n-gram methods preferred
- [ ] No transformer model downloads
- [ ] Using `polars` instead of `pandas`
- [ ] DataFrame operations use `pl.DataFrame()` API
- [ ] Using `tidyllm.tlm` algorithms instead of `sklearn`

### ✅ **S3-First Architecture:**
- [ ] No local file processing or storage
- [ ] All uploads go directly to S3 buckets
- [ ] Processing happens S3 → S3 (streaming)
- [ ] No app folders or temp directories
- [ ] MLflow connects directly to PostgreSQL
- [ ] All artifacts stored in S3, not locally
- [ ] Using `S3SessionManager` for all S3 operations
- [ ] Encrypted at rest and in transit

### ✅ **Security Compliance:**
- [ ] No sensitive data stored on local machine
- [ ] No local database files (SQLite, etc.)
- [ ] All processing is stateless and cloud-native
- [ ] Streaming operations for memory efficiency
- [ ] Proper S3 bucket segmentation
- [ ] PostgreSQL for centralized experiment tracking

---

## ☁️ **CRITICAL: S3-FIRST PROCESSING ARCHITECTURE**

### **🚫 FORBIDDEN: Local Machine Processing**
```python
# ❌ NEVER DO THIS - Local machine processing
with open("document.pdf", "rb") as f:
    embeddings = generate_embeddings(f.read())
    
df = pd.read_csv("data.csv")  # Processing on local machine
model.fit(df)  # Training locally

# ❌ NEVER DO THIS - App folder storage
app_folder = "./uploads/"
processed_data = "./processed/"
```

### **✅ REQUIRED: S3-First Cloud Processing**
```python
# ✅ CORRECT - Direct S3 upload and processing
from tidyllm.s3_session_manager import S3SessionManager, S3Utils

s3_manager = S3SessionManager()
s3_utils = S3Utils(s3_manager)

# 1. Upload directly to S3
s3_manager.upload_file("document.pdf", "processing-bucket", "docs/document.pdf")

# 2. Process IN S3 buckets (not locally)
documents = s3_utils.list_pdf_files_s3("processing-bucket", "docs/")
for doc_key in documents:
    # Process directly from S3 to S3
    embeddings = process_s3_document("processing-bucket", doc_key)
    s3_utils.save_json_to_s3("embeddings-bucket", f"embeddings/{doc_key}.json", embeddings)

# 3. MLflow connects directly to PostgreSQL (not local files)
mlflow.set_tracking_uri("postgresql://mlflowuser:pass@host:5432/mlflowdb")
```

### **🏗️ S3-First Architecture Principles**

#### **1. No Local File Processing**
- **Rule**: Files uploaded directly from machine → S3
- **Processing**: All operations happen S3 → S3 
- **Storage**: No app folders, no temp directories
- **Benefit**: Scalable, stateless, cloud-native

#### **2. S3 Bucket Segmentation**
```python
# ✅ CORRECT - Organized bucket strategy
BUCKETS = {
    "raw_uploads": "company-raw-documents",      # Direct uploads
    "processed": "company-processed-docs",       # Post-processing
    "embeddings": "company-embeddings-store",    # Vector storage
    "models": "company-trained-models",          # Model artifacts
    "metadata": "company-doc-metadata"           # Document metadata
}
```

#### **3. PostgreSQL-Direct MLflow**
```python
# ✅ CORRECT - Direct PostgreSQL connection
MLFLOW_CONFIG = {
    "backend_store_uri": "postgresql://mlflowuser:pass@postgres-host:5432/mlflowdb",
    "artifact_root": "s3://company-mlflow-artifacts/",  # Artifacts in S3
    "no_local_storage": True  # Force cloud storage
}

# All experiment tracking goes directly to PostgreSQL
# All artifacts stored directly in S3
# No local caching or storage
```

### **🔒 Security Constraints**

#### **Data Flow Security:**
```
User Machine → S3 Upload → Cloud Processing → S3 Storage → PostgreSQL Tracking
     ↓              ↓              ↓              ↓              ↓
  Encrypted     Encrypted      In-Transit     Encrypted    Encrypted
  in Transit    at Rest        Security       at Rest      at Rest
```

#### **🚨 SECURITY VIOLATIONS:**
```python
# ❌ FORBIDDEN - Local storage creates security risks
temp_file = "/tmp/uploaded_doc.pdf"  # Leaves traces
app_cache = "./cache/"               # Persistent local data
local_models = "./models/"           # Model files on disk

# ❌ FORBIDDEN - Local database/tracking
mlflow.set_tracking_uri("./mlruns")  # Local MLflow storage
sqlite_db = "local.db"               # Local database files

# ❌ FORBIDDEN - Account-level S3 operations in corporate environments
s3_client.list_buckets()             # Requires account-level permissions
s3_client.list_all_my_buckets()      # Corporate users don't have this access
s3_client.get_account_attributes()   # Account-level operations forbidden
```

#### **✅ SECURE PATTERNS:**
```python
# ✅ CORRECT - Corporate S3 bucket-level operations (NOT account-level)
def test_s3_corporate_access(bucket_name="nsc-mvp1"):
    """Test S3 access using bucket-level operations only"""
    
    session_mgr = UnifiedSessionManager()
    
    # ✅ CORRECT - Bucket-level operations (corporate users have these permissions)
    try:
        # Test file upload (bucket-level permission)
        test_data = {"test": "corporate_s3_access", "timestamp": datetime.now().isoformat()}
        session_mgr.upload_to_s3(bucket_name, "test/access_check.json", json.dumps(test_data))
        print(f"✅ S3 Upload access verified for bucket: {bucket_name}")
        
        # Test file download (bucket-level permission)
        downloaded = session_mgr.download_from_s3(bucket_name, "test/access_check.json")
        print(f"✅ S3 Download access verified for bucket: {bucket_name}")
        
        # Test list objects in bucket (bucket-level permission)
        objects = session_mgr.list_s3_objects(bucket_name, "test/")
        print(f"✅ S3 List objects verified: {len(objects)} objects in test/")
        
        return True
        
    except Exception as e:
        print(f"❌ S3 bucket access failed: {e}")
        return False

# ✅ CORRECT - Zero local persistence with corporate S3 patterns
def process_document_s3_to_s3(bucket_source, key_source, bucket_dest, key_dest):
    """Process document entirely in cloud - no local storage, no account-level operations"""
    
    # Stream directly from S3 → process → stream to S3
    session_mgr = UnifiedSessionManager()
    s3_client = session_mgr.get_s3_client()
    
    # Get document from S3 (streaming) - bucket-level operation
    response = s3_client.get_object(Bucket=bucket_source, Key=key_source)
    document_stream = response['Body']
    
    # Process in memory (no disk writes)
    processed_data = process_stream(document_stream)
    
    # Write directly to S3 (streaming) - bucket-level operation
    s3_client.put_object(
        Bucket=bucket_dest,
        Key=key_dest, 
        Body=processed_data,
        ServerSideEncryption='AES256'  # Encrypt at rest
    )
    
    # No local cleanup needed - nothing stored locally
    # No account-level operations used
```

### **🚀 Performance Benefits:**

| **Constraint** | **Security Benefit** | **Performance Benefit** |
|---------------|---------------------|------------------------|
| **S3-First Processing** | No local data exposure | Infinite scalability |
| **Direct PostgreSQL** | Centralized audit logs | No sync overhead |
| **No App Folders** | Zero persistent traces | Stateless deployment |
| **Streaming Operations** | Minimal memory footprint | Real-time processing |

---

## 🚨 **VIOLATION EXAMPLES & FIXES**

### **❌ WRONG - Stack + Architecture Violations:**
```python
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# Local processing (SECURITY VIOLATION)
with open("./uploads/document.pdf", "rb") as f:
    content = f.read()

# Big Tech dependencies (STACK VIOLATION)
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts)
df = pd.DataFrame({'embeddings': embeddings})
kmeans = KMeans(n_clusters=3).fit(embeddings)

# Local storage (ARCHITECTURE VIOLATION)
df.to_csv("./processed/results.csv")
mlflow.set_tracking_uri("./mlruns")
```

### **✅ CORRECT - TidyLLM Native + S3-First:**
```python
import tidyllm.tlm as np
import tidyllm_sentence as tls
import polars as pl
from tidyllm.s3_session_manager import S3SessionManager, S3Utils

# S3-first processing (SECURE)
s3_manager = S3SessionManager()
s3_utils = S3Utils(s3_manager)

# Stream document from S3 → process → back to S3
documents = s3_utils.list_pdf_files_s3("source-bucket", "docs/")
for doc_key in documents:
    # TidyLLM native embeddings (NO sentence-transformers)
    doc_content = s3_utils.load_json_from_s3("source-bucket", doc_key)
    embeddings, model = tls.tfidf_fit_transform([doc_content])
    
    # TidyLLM native clustering (NO sklearn)
    centers, labels, _ = np.kmeans_fit(embeddings, k=3)
    
    # Save directly to S3 (NO local storage)
    results = {"embeddings": embeddings, "clusters": labels}
    s3_utils.save_json_to_s3("results-bucket", f"processed/{doc_key}", results)

# PostgreSQL MLflow (NO local tracking)
mlflow.set_tracking_uri("postgresql://mlflowuser:pass@host:5432/mlflowdb")
```

---

## 📖 **REFERENCE IMPLEMENTATION**

See `IMPROVED_USAGE_EXAMPLES.py` for complete examples following these constraints.

### **Gateway Configuration:**
```python
registry = init_gateways({
    "ai_processing": {
        "embedding_provider": "tidyllm_sentence",  # NOT sentence-transformers
        "math_backend": "tlm"  # NOT numpy
    },
    "workflow_optimizer": {
        "stack_constraints": {
            "data_processing": "polars",  # NOT pandas
            "embeddings": "tidyllm_sentence",
            "math": "tlm"
        }
    }
})
```

---

## 🎯 **PERFORMANCE BENCHMARKS**

Our native stack delivers:

| **Component** | **TidyLLM Native** | **Big Tech** | **Advantage** |
|---------------|-------------------|--------------|---------------|
| **Memory** | 0.5MB | 88.7MB | **177x less** |
| **Dependencies** | Zero | Many | **Pure Python** |
| **Transparency** | Complete | Black box | **Educational** |
| **Quality** | 77.9% | 100% | **Acceptable trade-off** |
| **Startup** | Instant | 10-30s | **177x faster** |

---

## 📁 **CRITICAL: FILE ORGANIZATION CONSTRAINTS**

### **🚫 FORBIDDEN: Root Directory Code Files**
```
❌ NEVER PUT WORKING CODE IN ROOT:
/my_script.py           # Wrong - clutters root
/analysis_code.py       # Wrong - confusing location
/temp_solution.py       # Wrong - unclear purpose
/SOME_FEATURE.py        # Wrong - not organized
```

### **✅ REQUIRED: Organized Directory Structure**
```
✅ CORRECT ORGANIZATION:
/scripts/               # All executable scripts
  ├── universal_bracket_flows.py
  ├── s3_flow_processor.py
  ├── workflow_executor.py
  └── deployment/
      ├── setup_lambda.py
      └── configure_s3.py

/docs/                  # All documentation
  ├── universal_bracket_flows.md
  ├── api_reference.md
  ├── deployment_guide.md
  └── examples/
      ├── cli_usage.md
      └── s3_integration.md

/prompts/              # Workflow prompt templates
  ├── analyst_report_prompts.md
  ├── section_view_prompts.md
  ├── peer_review_prompts.md
  └── workflow_templates/

/tidyllm/              # Core library code
  ├── universal_flow_parser.py
  ├── s3_flow_parser.py
  ├── workflows/        # YAML workflow definitions
  ├── flow_agreements/  # Flow configuration files  
  ├── knowledge_systems/ # Domain RAG builders
  └── gateways/

/tidyllm-compliance/   # SOP and compliance templates
  ├── sop_golden_answers/
  ├── model_risk/
  └── consistency/

/tests/                # All test files
  ├── test_bracket_flows.py
  └── test_s3_integration.py

/sql/                  # All SQL files and database scripts
  ├── heiros_queries_analytics.sql
  ├── heiros_queries_workflows.sql
  ├── database_setup.sql
  └── migration_scripts/
      ├── v1_initial_schema.sql
      └── v2_add_indexes.sql
```

### **📋 File Placement Rules:**

#### **Scripts Directory (`/scripts/`):**
- **Executable files** that users run directly
- **Deployment scripts** for setup/configuration  
- **Utility scripts** for maintenance
- **Demo scripts** showing features
- **CLI tools** and command wrappers

#### **Docs Directory (`/docs/`):**
- **User guides** and tutorials
- **API documentation** and references
- **Implementation plans** and specifications
- **Architecture diagrams** and examples
- **Deployment guides** and troubleshooting

#### **Prompts Directory (`/prompts/`):**
- **Workflow prompt templates** for AI processing
- **Stage-specific prompts** for document analysis
- **Reusable prompt patterns** and examples
- **Template libraries** for different workflows

#### **Library Directory (`/tidyllm/`):**
- **Core implementation** code
- **Importable modules** and classes
- **Reusable components** and utilities
- **NO executable scripts** (those go in `/scripts/`)

#### **Tests Directory (`/tests/`):**
- **Unit tests** and integration tests
- **Test fixtures** and data
- **Test utilities** and helpers

#### **SQL Directory (`/sql/`):**
- **Database schemas** and table definitions
- **Query files** and stored procedures
- **Migration scripts** and database updates
- **Analytics queries** and reports
- **Database setup** and initialization scripts

### **🚨 ENFORCEMENT RULES:**

#### **Mandatory File Organization:**
1. **Scripts** → `/scripts/` directory ONLY
2. **Documentation** → `/docs/` directory ONLY  
3. **Prompt templates** → `/prompts/` directory ONLY
4. **Working code** → Appropriate `/tidyllm/` subdirectory
5. **Tests** → `/tests/` directory ONLY
6. **SQL files** → `/sql/` directory ONLY
7. **Workflow definitions** → `/tidyllm/workflows/` ONLY
8. **Flow agreements** → `/tidyllm/flow_agreements/` ONLY
9. **Domain RAG builders** → `/tidyllm/knowledge_systems/` ONLY
10. **Compliance templates** → `/tidyllm-compliance/` ONLY
11. **NO working code files in root directory**

#### **File Naming Convention:**
- **Scripts**: `action_description.py` (e.g., `deploy_bracket_flows.py`)
- **Docs**: `topic_type.md` (e.g., `bracket_flows_user_guide.md`)
- **Code**: `module_purpose.py` (e.g., `universal_flow_parser.py`)
- **SQL**: `system_purpose.sql` (e.g., `heiros_queries_analytics.sql`)

#### **PR Requirements:**
- [ ] All scripts moved to `/scripts/`
- [ ] All documentation moved to `/docs/`
- [ ] All prompt templates moved to `/prompts/`
- [ ] All SQL files moved to `/sql/`
- [ ] Workflow definitions in `/tidyllm/workflows/`
- [ ] Flow configurations in `/tidyllm/flow_agreements/`
- [ ] Domain RAG builders in `/tidyllm/knowledge_systems/`
- [ ] Compliance templates in `/tidyllm-compliance/`
- [ ] No working code files in root directory
- [ ] Proper subdirectory organization
- [ ] Clear file naming conventions

### **Example Violation → Fix:**
```
❌ WRONG:
/UNIVERSAL_BRACKET_IMPLEMENTATION_PLAN.md  # Root clutter
/s3_flow_parser.py                        # Library in root
/universal_bracket_examples.py            # Demo in root
/heiros_queries_analytics.sql             # SQL in root

✅ CORRECT:
/docs/universal_bracket_implementation_plan.md
/tidyllm/s3_flow_parser.py  
/scripts/universal_bracket_examples.py
/sql/heiros_queries_analytics.sql
```

---

## 🚨 **CRITICAL: UNIFIED SESSION MANAGEMENT ARCHITECTURE**

### **🚫 FORBIDDEN: Multiple Session Manager Implementations**
```python
# ❌ FORBIDDEN - Scattered session management patterns
from tidyllm.vectorqa.whitepapers.s3_session_manager import S3SessionManager  # Old pattern
from tidyllm.knowledge_systems.core.s3_manager import SomeS3Manager             # Fragmented
from tidyllm.gateway.s3_utils import S3Utils                                    # Scattered
```

### **✅ REQUIRED: UnifiedSessionManager Only**
```python
# ✅ CORRECT - One central session manager
from scripts.start_unified_sessions import UnifiedSessionManager

# Initialize unified session manager
session_mgr = UnifiedSessionManager()

# All operations go through unified interface
s3_client = session_mgr.get_s3_client()
postgres_conn = session_mgr.get_postgres_connection()
mlflow_client = session_mgr.get_mlflow_client()

# Unified operations
session_mgr.upload_to_s3(bucket, key, data)
session_mgr.execute_postgres_query(sql)
session_mgr.log_mlflow_experiment(params, metrics)
```

### **🏗️ Session Management Principles**

#### **1. Single Source of Truth**
- **Rule**: ALL session management goes through `UnifiedSessionManager`
- **Location**: `scripts/start_unified_sessions.py`
- **Benefit**: No configuration conflicts, centralized credential management

#### **2. No Additional Session Managers**
- **Forbidden**: Creating new S3SessionManager, DatabaseManager, etc.
- **Required**: Extend UnifiedSessionManager if new functionality needed
- **Reason**: Prevents scattered credential patterns and configuration conflicts

#### **3. Consolidated Configuration**
```python
# ✅ CORRECT - Unified configuration
UNIFIED_CONFIG = {
    "aws": {
        "region": "us-east-1",
        "s3_bucket": "nsc-mvp1",
        "credentials_source": "environment"
    },
    "postgres": {
        "host": "localhost",
        "port": 5432,
        "database": "mlflow",
        "ssl_mode": "require"
    },
    "mlflow": {
        "tracking_uri": "postgresql://mlflowuser:pass@localhost:5432/mlflowdb",
        "artifact_root": "s3://nsc-mvp1/mlflow-artifacts/"
    }
}
```

### **🚨 SESSION MANAGER VIOLATIONS:**
```python
# ❌ FORBIDDEN - Creating additional session managers
class MyS3SessionManager:  # Don't create new session managers
    pass

class CustomDatabaseManager:  # Don't fragment session management
    pass

# ❌ FORBIDDEN - Direct boto3/psycopg2 usage without UnifiedSessionManager
import boto3
s3_client = boto3.client('s3')  # Use session_mgr.get_s3_client() instead

import psycopg2
conn = psycopg2.connect(...)  # Use session_mgr.get_postgres_connection() instead
```

### **✅ APPROVED SESSION PATTERNS:**
```python
# ✅ CORRECT - All operations through UnifiedSessionManager
def process_document_with_unified_sessions(doc_path):
    """Process document using unified session management"""
    
    session_mgr = UnifiedSessionManager()
    
    # S3 operations
    session_mgr.upload_to_s3("source-bucket", "docs/document.pdf", doc_path)
    
    # Database operations  
    embeddings = session_mgr.execute_postgres_query(
        "SELECT embedding FROM document_embeddings WHERE doc_id = %s", 
        (doc_id,)
    )
    
    # MLflow tracking
    session_mgr.log_mlflow_experiment({
        "document": doc_path,
        "embedding_count": len(embeddings)
    })
    
    # All credentials managed centrally
    # All configurations unified
    # All connections pooled efficiently
```

### **📋 Session Management Checklist:**
- [ ] Using `UnifiedSessionManager` from `scripts/start_unified_sessions.py`
- [ ] NOT creating additional session manager classes
- [ ] NOT using direct boto3/psycopg2/mlflow clients
- [ ] All AWS operations through `session_mgr.get_s3_client()`
- [ ] All database operations through `session_mgr.get_postgres_connection()`
- [ ] All MLflow operations through `session_mgr.get_mlflow_client()`
- [ ] Credentials managed through unified configuration
- [ ] No scattered session patterns in tidyllm-vectorqa or other submodules

---

## 🔥 **ENFORCEMENT**

### **Code Review Requirements:**
1. **Mandatory**: All PRs must pass stack constraint validation
2. **Mandatory**: All files must be in correct directories
3. **Automatic**: CI checks for forbidden imports AND file organization
4. **Documentation**: All examples must use native stack AND proper file locations
5. **Testing**: Test cases must not import forbidden dependencies

### **Exception Policy:**
**NO EXCEPTIONS.** If you need functionality not available in the native stack:
1. Implement it in `tlm` or `tidyllm-sentence`
2. Create a lightweight pure Python alternative
3. Document the implementation for educational value
4. **Place all files in correct directories**

---

## 💡 **WHY THIS MATTERS**

TidyLLM isn't just another ML library - it's a **paradigm shift** toward:

- **Democratic ML**: Accessible to everyone, not just Big Tech
- **Educational Excellence**: Learn by reading every line
- **Infrastructure Independence**: Own your entire stack
- **Sustainable Computing**: Efficient, not wasteful
- **Transparent AI**: Understand every algorithm step

**This is bigger than code - it's about the future of ML infrastructure.**

---

## 📞 **QUESTIONS?**

If you're unsure about any constraint:
1. Read this document again
2. Check `IMPROVED_USAGE_EXAMPLES.py` 
3. Look at existing `tlm` and `tidyllm-sentence` implementations
4. Ask: "Does this increase or decrease user sovereignty?"

**Remember: Every dependency we avoid is a victory for user freedom.**

---

## 🤖 **FINAL REMINDER FOR AI AGENTS**

### **BEFORE YOU WRITE ANY CODE:**
1. **GO BACK** to the top of this document
2. **READ** the "ATTENTION AI AGENTS & DEVELOPERS" section
3. **FOLLOW** every link to the guidance documents
4. **UNDERSTAND** all constraints and patterns
5. **ONLY THEN** proceed with implementation

### **AI AGENT COMPLIANCE STATEMENT:**
By making changes to this codebase, you acknowledge that you have:
- ✅ Read ALL Priority 1-4 guidance documents listed at the top
- ✅ Understood the forbidden dependencies and required substitutions
- ✅ Will follow the S3-First architecture
- ✅ Will use UnifiedSessionManager for all sessions
- ✅ Will place files in correct directories (/scripts, /docs, etc.)
- ✅ Will NOT use numpy, pandas, sklearn, or sentence-transformers
- ✅ Will use tidyllm.tlm, tidyllm_sentence, and polars instead

**If you have NOT read the guidance documents, STOP NOW and read them.**

---

**🚨 THIS DOCUMENT IS THE LAW. FOLLOW IT. 🚨**

**🔴 AI AGENTS: THE GUIDANCE DOCUMENTS LINKED AT THE TOP ARE MANDATORY READING 🔴**