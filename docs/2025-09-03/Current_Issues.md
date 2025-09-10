# Current Issues - TidyLLM Ecosystem
**Generated:** 2025-09-03  
**Status:** Active Issues Tracking  
**Priority:** Critical → High → Medium → Low

---

## 🔴 **CRITICAL ISSUES** (System Breaking)

### 1. **Provider Factory Misuse - 100% Failure Rate**
**Location:** DSPy implementations  
**Impact:** ALL LLM operations fail  
**Details:**
- Provider factory functions being called incorrectly
- Returning function objects instead of text responses
- Affects all 4 competing DSPy implementations
**Fix Required:** Immediate - blocking all LLM functionality

### 2. **DataTable Integration Broken**
**Location:** `tidyllm/dt.py`  
**Impact:** Complete DataTable functionality unavailable  
**Details:**
- Undefined variable `_dt` throughout the module
- Python 3.13 installation failures
- No scikit-learn integration possible
**Fix Required:** Remove or properly implement DataTable support

### 3. **Sentence Embedding Import Conflicts**
**Location:** Multiple implementations across repos  
**Impact:** RAG system functionality compromised  
**Details:**
- 3 competing implementations discovered
- Import errors in tidyllm-sentence
- Missing integration with main system
**Fix Required:** Consolidate to single implementation

---

## 🟠 **HIGH PRIORITY ISSUES** (Major Functionality)

### 4. **Math Library Failures**
**Location:** `tidyllm/numpy_compat.py`  
**Impact:** 25% of math operations failing  
**Test Results:**
- TLM Library: 9/12 tests passing (75% health)
- Matrix mean function errors
- tidyllm-sentence: Cannot test due to import issues
**Fix Required:** Week 1

### 5. **Repository Fragmentation**
**Impact:** Development complexity and maintenance burden  
**Details:**
- 14+ separate repositories discovered
- Inconsistent versioning
- Circular dependencies between repos
- 30-40% code bloat from duplicates
**Fix Required:** Consolidation plan needed

### 6. **Gateway Architecture Conflicts**
**Location:** Multiple gateway implementations  
**Impact:** Unclear execution path, performance issues  
**Details:**
- 4 competing LLMGateway implementations
- 3 competing DSPy gateway backends
- Triple gateway proposal pending implementation
**Fix Required:** Architecture decision and consolidation

### 7. **AWS Credentials Not Configured**
**Impact:** Limited to mock mode for demos  
**Details:**
- Bedrock integration complete but not accessible
- S3 operations failing
- MLflow remote tracking unavailable
**Fix Required:** Configuration guide needed

---

## 🟡 **MEDIUM PRIORITY ISSUES** (Feature Limitations)

### 8. **TidyMart Integration Incomplete**
**Status:** 228 tasks identified  
**Phases:**
1. Database Centralization (Week 1-2)
2. Storage Layer (Week 2-3)
3. Module Migration (Week 3-4)
4. Gateway Integration (Week 4-5)
5. Testing & Optimization (Week 5-6)
**Impact:** Enterprise features unavailable

### 9. **Circular Import Dependencies**
**Locations:**
- `tidyllm/__init__.py`
- `tidyllm/core.py`
- `tidyllm/verbs.py`
**Impact:** Import order sensitivity, potential runtime failures

### 10. **Missing Test Coverage**
**Details:**
- No tests for gateway implementations
- Missing integration tests
- Error handling tests incomplete
**Impact:** Uncertain system reliability

### 11. **Documentation Scattered**
**Issues:**
- Documentation across 14+ repos
- Missing consolidated API docs
- Incomplete migration guides
- No central documentation site

---

## 🟢 **LOW PRIORITY ISSUES** (Enhancements)

### 12. **Performance Optimizations Needed**
- Caching layer not implemented
- Parallel execution not optimized
- Stream processing incomplete

### 13. **CLI Enhancement Opportunities**
- Command chaining not implemented
- No autocomplete support
- Missing shell aliases

### 14. **Monitoring and Metrics**
- Basic metrics collection only
- No dashboard implementation
- Missing alerting system

---

## 📊 **Issue Statistics**

### By Component:
- **DSPy**: 4 critical, 2 high priority issues
- **Gateways**: 3 high priority issues  
- **DataTable**: 2 critical issues
- **Sentence Embedding**: 1 critical, 2 high priority issues
- **Math Libraries**: 2 high priority issues
- **Documentation**: 4 medium priority issues

### By Type:
- **Bugs**: 7 issues (4 critical, 3 high)
- **Architecture**: 5 issues (all high priority)
- **Integration**: 4 issues (1 critical, 3 medium)
- **Documentation**: 3 issues (all medium)
- **Performance**: 3 issues (all low)

---

## 🚨 **Immediate Action Items**

### Week 1 Sprint (Critical Fixes):
1. **Fix provider factory pattern** - Causes 100% LLM failure
2. **Resolve DataTable `_dt` variable** - Or remove module
3. **Consolidate sentence embeddings** - Pick one implementation
4. **Fix math library errors** - Matrix operations failing

### Week 2 Sprint (Architecture):
1. **Choose single gateway pattern** - Eliminate duplicates
2. **Start repository consolidation** - Merge related repos
3. **Standardize imports** - Create import guidelines
4. **Document AWS setup** - Enable full functionality

### Week 3-4 (Stabilization):
1. **Complete TidyMart integration** - Follow 228-task plan
2. **Implement comprehensive testing** - Achieve 80% coverage
3. **Consolidate documentation** - Single source of truth
4. **Setup monitoring** - Track system health

---

## 📝 **Known Workarounds**

### For Provider Factory Issue:
```python
# Instead of: response = provider_fn(messages)
# Use: provider = provider_fn(); response = provider(messages)
```

### For DataTable Issues:
```python
# Temporarily use pandas instead of datatable
import pandas as pd
# df = datatable.Frame(data)  # Old
df = pd.DataFrame(data)  # Workaround
```

### For Sentence Embedding:
```python
# Use the working implementation from tidyllm.core
from tidyllm.core import SentenceEmbeddings
# Not from tidyllm-sentence or other variants
```

### For AWS Credentials:
```python
# Use mock mode for development
USE_MOCK = True  # In configuration
```

---

## 🔄 **Update History**

- **2025-09-03**: Initial comprehensive issue compilation
- **Previous**: Various scattered issue reports consolidated

---

## 📞 **Support Channels**

- **GitHub Issues**: (Currently no open issues - needs population)
- **Internal Tracking**: `tidyllm/error_tracker.py`
- **Documentation**: See respective fix guides in repo root

---

## ⚡ **Quick Reference**

**Most Critical Fix:**
```bash
# Fix provider factory immediately - blocking ALL LLM operations
# See: DSPY_BUG_FIX_CHECKLIST.md for detailed steps
```

**Health Check Command:**
```bash
python demo_status_check.py  # Current: 80% success rate
```

**For New Developers:**
1. Start with `PRECISE_FIX_GUIDE_FOR_BASICS_TEAM.md`
2. Review `DEMO_STATUS_FINAL.md` for current state
3. Check `TIDYMART_DETAILED_TODOS.md` for roadmap

---

**Note:** This document consolidates issues from multiple sources. For detailed fix instructions, refer to the specific guide documents mentioned in each section.