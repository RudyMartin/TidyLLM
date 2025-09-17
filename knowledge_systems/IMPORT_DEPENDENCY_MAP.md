# Knowledge Systems Import Dependency Map

**Created**: 2025-01-16 (Phase 3.1)
**Purpose**: Document all import dependencies before migration execution
**Backup Branch**: `knowledge_systems_consolidation_backup_20250916_071035`

---

## 🎯 **CRITICAL IMPORT PATHS - PRODUCTION SYSTEMS**

### **1. Core Domain RAG** (`tidyllm/knowledge_systems/core/domain_rag.py`)
**18KB - ENTERPRISE SYSTEM**
```python
# CRITICAL IMPORTS - THESE MUST WORK
from ..facades.vector_storage import VectorStorageFacade
from ..core.s3_document_manager import S3DocumentManager
from tidyllm.infrastructure.standards import TidyLLMStandardRequest
from tidyllm.admin.credential_loader import get_s3_config, get_postgresql_config
```

### **2. Knowledge Interface** (`tidyllm/knowledge_systems/interfaces/knowledge_interface.py`)
**455 lines - PUBLIC API**
```python
# CRITICAL IMPORTS - PORTAL USES THIS
from ..core.knowledge_manager import KnowledgeManager, KnowledgeSystemConfig
from ..core.domain_rag import RAGQuery, RAGResponse
from ...admin.credential_loader import get_s3_config, build_s3_path
```

### **3. Vector Manager** (`tidyllm/knowledge_systems/core/vector_manager.py`)
**27KB - POSTGRESQL INTEGRATION**
```python
# CRITICAL IMPORTS - DATABASE OPERATIONS
import psycopg2
from tidyllm.admin.settings import get_database_config
```

---

## 🔗 **ACTIVE INTEGRATIONS**

### **Portal Integration** ✅ **UPDATED**
**File**: `tidyllm/portals/rag/rag_creator_v2.py`
```python
# NEW IMPORT - PHASE 2.3 COMPLETE
from tidyllm.knowledge_systems.interfaces.knowledge_interface import get_knowledge_interface

@st.cache_resource
def get_knowledge_service():
    return get_knowledge_interface()
```

### **S3 Integration**
```python
# S3 DOCUMENT PROCESSING
from tidyllm.admin.credential_loader import get_s3_config
from tidyllm.knowledge_systems.core.s3_document_manager import S3DocumentManager
```

### **PostgreSQL Integration**
```python
# VECTOR STORAGE
from tidyllm.knowledge_systems.core.vector_manager import VectorManager
from tidyllm.admin.settings import get_postgresql_config
```

---

## ⚠️ **IMPORT RISKS & DEPENDENCIES**

### **High Risk Paths**
1. **TidyLLM Admin System**
   - `tidyllm.admin.credential_loader` - Credential management
   - `tidyllm.admin.settings` - Configuration system
   - **RISK**: If admin system changes, knowledge_systems breaks

2. **TidyLLM Infrastructure**
   - `tidyllm.infrastructure.standards` - Standard request/response patterns
   - `tidyllm.gateways.file_storage_gateway` - File storage operations
   - **RISK**: Infrastructure changes could break core functionality

3. **External Dependencies**
   - `psycopg2` - PostgreSQL connection (pgvector operations)
   - `boto3` - AWS S3 operations
   - `streamlit` - Portal caching (@st.cache_resource)

### **Import Circular Dependency Risks**
```
❌ AVOID THESE PATTERNS:
tidyllm.knowledge_systems → tidyllm.compliance → tidyllm.knowledge_systems
tidyllm.portals → tidyllm.knowledge_systems → tidyllm.portals
```

---

## 📦 **MIGRATED SYSTEMS IMPORT STATUS**

### **Compliance Systems** (`migrated/compliance/`)
- ✅ **Safe to integrate**: All use standard PostgreSQL + S3 patterns
- ⚠️ **Watch for**: Circular imports with tidyllm.compliance package

### **Scattered RAG Systems** (`migrated/scattered_rag/`)
- ✅ **Safe to integrate**: Most use basic imports (psycopg2, yaml, json)
- ⚠️ **Watch for**: Some may have hardcoded paths or deprecated imports

### **RAG Adapters** (`migrated/scattered_rag/adapters/`)
- ✅ **postgres_rag_adapter.py**: Uses standard PostgreSQL patterns
- ✅ **intelligent_rag_adapter.py**: Clean import structure
- ✅ **ai_powered_rag_adapter.py**: Uses TidyLLM infrastructure properly
- ✅ **judge_rag_adapter.py**: External integration patterns
- ✅ **sme_rag_system.py**: Standard import patterns

---

## 🚫 **UNCERTAIN SYSTEMS - IMPORT ANALYSIS**

### **VectorQA Service** (`uncertain/duplicate_systems/vectorqa_service/`)
**37 files - NEW UNTESTED CODE**
```python
# THESE IMPORTS MAY CONFLICT
from tidyllm.knowledge_systems.core.domain_rag import DomainRAG  # ❌ CIRCULAR?
from tidyllm.infrastructure.standards import *  # ❌ STAR IMPORTS
from .domain.entities import *  # ❌ STAR IMPORTS
```
**RECOMMENDATION**: ❌ **DELETE** - Too much untested infrastructure

### **VectorRAG Wrapper** (`uncertain/duplicate_systems/vectorRAG_wrapper.py`)
```python
# RELATIVELY SAFE IMPORTS
from tidyllm.compliance.domain_rag.authoritative_rag import AuthoritativeRAG
from tidyllm.knowledge_systems.core.domain_rag import DomainRAG
```
**RECOMMENDATION**: 🔄 **EVALUATE** - Extract useful patterns only

---

## 🛡️ **SAFETY PROTOCOLS**

### **Before Any Import Changes**
1. ✅ **Backup branch created**: `knowledge_systems_consolidation_backup_20250916_071035`
2. ✅ **Portal updated safely**: Uses KnowledgeInterface (tested pattern)
3. ✅ **Core systems preserved**: No changes to production domain_rag.py

### **Import Testing Protocol**
1. **Test core imports**: `from tidyllm.knowledge_systems.interfaces import get_knowledge_interface`
2. **Test portal integration**: Launch RAG Creator portal
3. **Test database connections**: VectorManager PostgreSQL operations
4. **Test S3 operations**: S3DocumentManager file processing

### **Rollback Strategy**
```bash
# IF ANYTHING BREAKS:
git checkout knowledge_systems_consolidation_backup_20250916_071035
# RESTORE WORKING STATE IMMEDIATELY
```

---

## 🎯 **PHASE 3.2 INTEGRATION PLAN**

### **Safe Integration Order**
1. **Test current system**: Verify all portals work with KnowledgeInterface
2. **Integrate compliance systems**: Add domain configs to KnowledgeInterface
3. **Integrate scattered adapters**: One by one with testing
4. **Extract useful patterns**: From migrated systems into core
5. **Delete uncertain systems**: After extracting any valuable code

### **Import Validation Checklist**
- [ ] KnowledgeInterface imports work
- [ ] Portal loads without errors
- [ ] Database connections successful
- [ ] S3 operations functional
- [ ] No circular import warnings
- [ ] All core functionality intact

---

**🔒 SAFETY FIRST**: Every import change will be tested immediately with rollback ready.