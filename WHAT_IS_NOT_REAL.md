# What Is NOT Real in TidyLLM System

**⚠️ WARNING:** These method calls, imports, and files DO NOT exist or are broken. Do not use them!

---

## 1. Non-Existent API Methods

| Method Call | Where Referenced | Status | Why It Fails |
|-------------|------------------|---------|--------------|
| `session_mgr.validate_session()` | Multiple files | ❌ **NOT REAL** | UnifiedSessionManager has no `validate_session` method |
| `session_mgr.test_postgres_connection()` | Onboarding app (previously) | ❌ **NOT REAL** | Use `get_postgres_connection()` instead |

---

## 2. Deleted Demo Files (Still Referenced)

| File Path | Status | Impact |
|-----------|---------|--------|
| `tidyllm/api_demo.py` | 🗑️ **DELETED** | Any imports will fail |
| `tidyllm/api_examples.py` | 🗑️ **DELETED** | Contains non-working `list_models()` calls |
| `tidyllm/api_examples_2.py` | 🗑️ **DELETED** | Contains non-working `list_models()` calls |
| `tidyllm/api_examples_3.py` | 🗑️ **DELETED** | Contains non-working `list_models()` calls |
| `tidyllm/executive_api_demo.py` | 🗑️ **DELETED** | Contains non-working `list_models()` calls |
| `tidyllm/executive_demo.py` | 🗑️ **DELETED** | Contains non-working `list_models()` calls |
| `tidyllm/final_real_demo.py` | 🗑️ **DELETED** | Demo file no longer exists |
| `tidyllm/real_api_demo.py` | 🗑️ **DELETED** | Demo file no longer exists |
| `tidyllm/real_demo_alternative.py` | 🗑️ **DELETED** | Demo file no longer exists |
| `tidyllm/real_demo_bypass.py` | 🗑️ **DELETED** | Demo file no longer exists |
| `tidyllm/simple_api_test.py` | 🗑️ **DELETED** | Contains non-working `list_models()` calls |
| `tidyllm/simple_quality_test.py` | 🗑️ **DELETED** | Test file no longer exists |
| `tidyllm/backend_quality_test.py` | 🗑️ **DELETED** | Test file no longer exists |

---

## 3. Methods That Look Real But Are Problematic

| Method Call | Status | Issue | Workaround |
|-------------|--------|-------|------------|
| `gateway.health_check()` | ⚠️ **UNRELIABLE** | Called in `gateway_registry.py` but not all gateways implement it | Use `gateway.validate_config()` instead |
| `worker.health_check()` | ⚠️ **UNRELIABLE** | Called in `worker_integration.py` but async pattern inconsistent | Check worker status directly |
| `embedding_standardizer.get_model_info()` | ❓ **UNKNOWN** | Called in `vector_manager.py` but may not exist on all standardizers | Check if method exists first |

---

## 4. Deprecated Patterns (Don't Use)

| Pattern | Why Not To Use | Use Instead |
|---------|----------------|-------------|
| Direct `boto3.client()` calls | Bypasses credential management | `session_mgr.get_s3_client()` or `session_mgr.get_bedrock_client()` |
| Manual connection handling | No connection pooling | `session_mgr.get_postgres_connection()` + `return_postgres_connection()` |
| Mock validation in tests | Hides real issues | Real connection tests with proper error messages |
| Hardcoded credentials | Security risk | Use `settings.yaml` with UnifiedSessionManager |
| Direct file imports from deleted files | Import errors | Use working imports from `CRITICAL_CALLS_THAT_WORK.md` |

---

## 5. Common Error Patterns to Avoid

### ❌ **Import Errors**
```python
# These WILL fail:
from tidyllm.api_demo import something           # File deleted
from tidyllm.simple_api_test import test_func   # File deleted  
import tidyllm; models = tidyllm.list_models()  # Method doesn't exist
```

### ❌ **Method Call Errors** 
```python
# These methods DO NOT exist:
session_mgr.test_connection()           # No such method
session_mgr.validate_session()         # No such method  
session_mgr.test_postgres_connection()  # No such method
```

### ❌ **Unreliable Patterns**
```python
# These may fail unpredictably:
gateway.health_check()                  # Not implemented on all gateways
worker.health_check()                   # Async pattern inconsistent
```

---

## 6. Files That Exist But Have Broken Code

| File Path | Issue | Impact |
|-----------|-------|--------|
| `tidyllm/examples/api_examples.py` | Contains `tidyllm.list_models()` call | Will throw AttributeError |
| `tidyllm/examples/executive_demo.py` | Contains `tidyllm.list_models()` call | Will throw AttributeError |
| `tidyllm/knowledge_systems/s3_first_domain_rag.py` | Contains `s3_manager.test_connection()` call | Will throw AttributeError |

---

## 7. MLflow Trash Files (Being Cleaned Up)

| Pattern | Status | Safe to Ignore |
|---------|---------|----------------|
| `${MLFLOW_TRACKING_URI}/.trash/**` | 🗑️ **DELETED** | ✅ Yes - MLflow cleanup artifacts |

---

## 8. What This Means for New Developers

### ✅ **DO:**
- Only use methods from `CRITICAL_CALLS_THAT_WORK.md`
- Use UnifiedSessionManager for all connections
- Import from working module paths only
- Test connections with real validation (no mocks)

### ❌ **DON'T:**
- Try to use `tidyllm.list_models()` - it doesn't exist
- Import from deleted demo files
- Use `test_connection()` methods - they don't exist  
- Rely on `health_check()` methods - they're unreliable
- Use direct `boto3.client()` calls - bypasses credential management

### 🔍 **How to Verify a Method Exists:**
```python
# Always check if a method exists before using it:
if hasattr(object, 'method_name'):
    result = object.method_name()
else:
    print("Method doesn't exist!")
```

---

## 9. Red Flags That Indicate Non-Real Code

| Red Flag | What It Means | Action |
|----------|---------------|---------|
| `AttributeError: ... has no attribute '...'` | Method doesn't exist | Check `CRITICAL_CALLS_THAT_WORK.md` |
| `ModuleNotFoundError: No module named '...'` | File/module deleted | Update import path |
| `ImportError: cannot import name '...'` | Function doesn't exist in module | Use different function |
| Code referencing `api_demo`, `simple_api_test`, etc. | Using deleted files | Use working examples |
| Calls to `list_models()`, `test_connection()`, etc. | Using non-existent methods | Use real methods from working list |

---

**🚨 CRITICAL RULE:** If it's not in `CRITICAL_CALLS_THAT_WORK.md`, assume it might not exist and verify first!

**Last Updated:** 2025-09-09  
**System Version:** TidyLLM v1.0.4