# 🔗 DEPENDENCY-AWARE EXECUTION PLAN

**Critical Issue**: The TODO list has interdependencies that could cause cascading failures if executed in wrong order.

---

## 🚨 **INTERDEPENDENCY ANALYSIS**

### **Critical Dependencies Identified:**

1. **Fix #1 (Circular Dependencies) BLOCKS Everything**
   - EnhancedQAOrchestrator can't be tested until circular imports are resolved
   - Advanced QA Orchestrator won't work until DataMart is extracted
   - Demo files may fail if they import broken orchestrators

2. **Fix #2 (Enhanced QA) DEPENDS ON Fix #1**  
   - Enhanced QA imports may fail due to circular dependencies
   - Testing Enhanced QA requires working DataMart imports

3. **Fix #3 (Demo File) DEPENDS ON Fix #1 + Fix #2**
   - Demo file imports EnhancedQAOrchestrator 
   - Will fail if orchestrator imports are broken

4. **Fix #4 (Consolidation) DEPENDS ON ALL PREVIOUS FIXES**
   - Requires stable base to build upon
   - Can't test consolidation if basic imports don't work

---

## 🎯 **DEPENDENCY-SAFE EXECUTION STRATEGY**

### **PHASE 1: FOUNDATION (Fix #1 Only)**
**Goal**: Create stable import foundation before attempting anything else

#### **Critical Path Dependencies:**
```
DataMart Extraction → Import Updates → Validation → CHECKPOINT
```

#### **Safe Execution Steps:**
1. **ISOLATE FIX #1**: Complete ALL of Fix #1 before touching anything else
2. **CHECKPOINT VALIDATION**: Ensure basic imports work before proceeding
3. **NO PARALLEL WORK**: Don't attempt other fixes until Fix #1 is 100% complete

### **PHASE 2: ORCHESTRATOR FIXES (Fix #2)**
**Goal**: Fix enhanced orchestrator now that foundation is stable

### **PHASE 3: USER INTERFACE (Fix #3)**
**Goal**: Fix demo file now that orchestrators work

### **PHASE 4: OPTIMIZATION (Fix #4)**
**Goal**: Consolidate architecture now that everything is stable

---

## 🛡️ **RISK MITIGATION STRATEGIES**

### **Strategy 1: Incremental Validation**
```bash
# After each major step, validate the system still works
python -c "
import sys; sys.path.append('src')
from backend.mcp.orchestrators.simple_qa_orchestrator import SimpleQAOrchestrator
print('✅ Simple QA still works')
"
```

### **Strategy 2: Rollback Points**
- Create git commit after each completed phase
- Never start next phase until current phase is validated
- Keep backup branch until all fixes are complete

### **Strategy 3: Dependency Checking**
Before each fix, run dependency check:
```bash
# Check what might break
python -c "
import sys
sys.path.append('src')

try:
    from backend.mcp.orchestrators.enhanced_qa_orchestrator import EnhancedQAOrchestrator
    print('Enhanced QA: ✅ WORKING')
except Exception as e:
    print(f'Enhanced QA: ❌ BROKEN - {e}')

try:
    from backend.mcp.orchestrators.advanced_qa_orchestrator import AdvancedQAOrchestrator  
    print('Advanced QA: ✅ WORKING')
except Exception as e:
    print(f'Advanced QA: ❌ BROKEN - {e}')
"
```

---

## 📋 **UPDATED DEPENDENCY-SAFE TODO LIST**

### **🚨 PHASE 1: FIX #1 - CIRCULAR DEPENDENCIES (CRITICAL PATH)**

#### **Pre-Phase 1 Safety Checks**
- [ ] **SAFETY-1.1**: `git status` - ensure clean working directory
- [ ] **SAFETY-1.2**: `git checkout -b backup-before-any-changes` - create safety backup
- [ ] **SAFETY-1.3**: Run baseline system check:
  ```bash
  python -c "
  import sys; sys.path.append('src')
  
  # Test what currently works
  try:
      from backend.mcp.orchestrators.simple_qa_orchestrator import SimpleQAOrchestrator
      print('Simple QA: ✅ BASELINE WORKING')
  except Exception as e:
      print(f'Simple QA: ❌ BASELINE BROKEN - {e}')
      exit(1)  # Don't proceed if even simple doesn't work
  "
  ```
- [ ] **SAFETY-1.4**: Document current broken state for comparison

#### **Phase 1A: Setup New DataMart Package**
- [ ] **1A.1**: `git checkout -b fix-circular-dependencies-only` 
- [ ] **1A.2**: `mkdir -p src/backend/datamart`
- [ ] **1A.3**: `touch src/backend/datamart/__init__.py`
- [ ] **1A.4**: Test new package can be imported: `python -c "import src.backend.datamart"`

#### **Phase 1B: Extract DataMartManager (NO OTHER CHANGES)**
- [ ] **1B.1**: Read `advanced_qa_orchestrator.py` lines 232-588 carefully
- [ ] **1B.2**: Create `src/backend/datamart/core_manager.py` with ONLY DataMartManager
- [ ] **1B.3**: Test new manager imports in isolation:
  ```python
  from src.backend.datamart.core_manager import DataMartManager, DataMartMode
  dm = DataMartManager(DataMartMode.SIMPLE)
  assert dm.initialize_datamart()
  print("✅ New DataMart works in isolation")
  ```
- [ ] **1B.4**: **CHECKPOINT**: Don't modify original file yet - just ensure extraction works

#### **Phase 1C: Update Advanced QA Orchestrator (SINGLE FILE)**
- [ ] **1C.1**: Remove lines 232-588 from `advanced_qa_orchestrator.py`
- [ ] **1C.2**: Add import: `from ...datamart.core_manager import DataMartManager, DataMartMode`
- [ ] **1C.3**: Test ONLY Advanced QA Orchestrator:
  ```python
  from src.backend.mcp.orchestrators.advanced_qa_orchestrator import AdvancedQAOrchestrator
  qa = AdvancedQAOrchestrator()
  print("✅ Advanced QA works with new DataMart")
  ```
- [ ] **1C.4**: **CHECKPOINT**: Commit this single change before proceeding

#### **Phase 1D: Extract NumPy Substitute (CAREFUL SEQUENCING)**
- [ ] **1D.1**: Create `src/backend/datamart/numpy_substitute.py`
- [ ] **1D.2**: Copy NumPy code from `datamart_numpy_substitution.py` (DON'T DELETE ORIGINAL YET)
- [ ] **1D.3**: Update new file import: `from .core_manager import DataMartManager, DataMartMode`
- [ ] **1D.4**: Test NumPy substitute in isolation:
  ```python
  from src.backend.datamart.numpy_substitute import np
  arr = np.array([1,2,3])  
  assert arr == [1,2,3]
  print("✅ NumPy substitute works")
  ```

#### **Phase 1E: Update Import References (ONE BY ONE)**
**Critical**: Update ONE file at a time, test after each change

- [ ] **1E.1**: Update `enhanced_datamart_manager.py` ONLY:
  - Change: `from .datamart_numpy_substitution import DataMartManager, DataMartMode`
  - To: `from .datamart.core_manager import DataMartManager, DataMartMode`
  - Test: `python -c "from src.backend.core.enhanced_datamart_manager import EnhancedDataMartManager"`
  - **CHECKPOINT**: Commit if this works

- [ ] **1E.2**: Update `rag_qa_orchestrator.py` ONLY:
  - Change: `from ...core.datamart_numpy_substitution import np`
  - To: `from ...datamart.numpy_substitute import np`
  - Test: `python -c "from src.backend.mcp.orchestrators.rag_qa_orchestrator import RAGQAOrchestrator"`
  - **CHECKPOINT**: Commit if this works

- [ ] **1E.3**: Update `embedding_helper.py` ONLY:
  - Change: `from .datamart_numpy_substitution import np`
  - To: `from ..datamart.numpy_substitute import np`  
  - Test: `python -c "from src.backend.core.embedding_helper import [whatever class exists]"`
  - **CHECKPOINT**: Commit if this works

- [ ] **1E.4**: Update `vst_mvr_comparison_worker.py` ONLY:
  - Change: `from ...core.datamart_numpy_substitution import np`
  - To: `from ...datamart.numpy_substitute import np`
  - Test: `python -c "from src.backend.mcp.workers.vst_mvr_comparison_worker import VSTMVRComparisonWorker"`
  - **CHECKPOINT**: Commit if this works

#### **Phase 1F: Final Validation (COMPREHENSIVE)**
- [ ] **1F.1**: Delete old `datamart_numpy_substitution.py` file ONLY after all references updated
- [ ] **1F.2**: Run comprehensive system test:
  ```python
  import sys
  sys.path.append('src')
  
  # Test all orchestrators work
  from backend.mcp.orchestrators.simple_qa_orchestrator import SimpleQAOrchestrator
  from backend.mcp.orchestrators.advanced_qa_orchestrator import AdvancedQAOrchestrator
  from backend.datamart.core_manager import DataMartManager
  from backend.datamart.numpy_substitute import np
  
  print("✅ All imports successful - no circular dependencies")
  ```
- [ ] **1F.3**: Test functionality still works:
  ```python
  # Test Simple QA still works
  simple_qa = SimpleQAOrchestrator()
  test_doc = {'content': 'test', 'file_path': 'test.txt'}
  result = simple_qa.process_document(test_doc)
  assert result['status'] == 'success'
  print("✅ Simple QA functionality preserved")
  
  # Test Advanced QA still works  
  advanced_qa = AdvancedQAOrchestrator()
  result = advanced_qa.process_document(test_doc)
  assert result['status'] == 'success' 
  print("✅ Advanced QA functionality preserved")
  ```

- [ ] **PHASE 1 COMPLETE**: `git commit -m "PHASE 1 COMPLETE: Circular dependencies resolved"`

---

### **🔴 PHASE 2: FIX #2 - ENHANCED QA ORCHESTRATOR (DEPENDS ON PHASE 1)**

#### **Pre-Phase 2 Dependency Check**
- [ ] **SAFETY-2.1**: Verify Phase 1 worked:
  ```bash
  python -c "
  import sys; sys.path.append('src')
  from backend.mcp.orchestrators.simple_qa_orchestrator import SimpleQAOrchestrator
  from backend.mcp.orchestrators.advanced_qa_orchestrator import AdvancedQAOrchestrator
  print('✅ Phase 1 dependencies satisfied')
  "
  ```

#### **Phase 2A: Test Current Enhanced QA State**  
- [ ] **2A.1**: Test current Enhanced QA import (expect it to work better now):
  ```bash
  python -c "from src.backend.mcp.orchestrators.enhanced_qa_orchestrator import EnhancedQAOrchestrator"
  ```
- [ ] **2A.2**: If it now works → Skip to Phase 3
- [ ] **2A.3**: If still fails → Continue with coordinator creation

#### **Phase 2B: Create Missing Coordinators (Only if needed)**
- [ ] **2B.1**: Create coordinator directory structure
- [ ] **2B.2**: Implement minimal coordinators  
- [ ] **2B.3**: Test Enhanced QA works
- [ ] **2B.4**: **CHECKPOINT**: `git commit -m "PHASE 2 COMPLETE: Enhanced QA fixed"`

---

### **🔴 PHASE 3: FIX #3 - DEMO FILE (DEPENDS ON PHASE 1+2)**

#### **Pre-Phase 3 Dependency Check**
- [ ] **SAFETY-3.1**: Verify Phase 2 worked:
  ```bash
  python -c "
  import sys; sys.path.append('src')
  from backend.mcp.orchestrators.enhanced_qa_orchestrator import EnhancedQAOrchestrator
  qa = EnhancedQAOrchestrator()
  print('✅ Phase 2 dependencies satisfied')
  "
  ```

#### **Phase 3A: Create Demo File**
- [ ] **3A.1**: Create `src/demo.py` that imports EnhancedQAOrchestrator
- [ ] **3A.2**: Test demo file works in isolation
- [ ] **3A.3**: Test enhanced launcher finds demo file
- [ ] **3A.4**: **CHECKPOINT**: `git commit -m "PHASE 3 COMPLETE: Demo file added"`

---

### **🟡 PHASE 4: FIX #4 - CONSOLIDATION (DEPENDS ON ALL PREVIOUS)**

#### **Pre-Phase 4 Dependency Check**
- [ ] **SAFETY-4.1**: Verify all previous phases work:
  ```bash
  # Run comprehensive system test
  bash tests/run_all_fix_tests.sh
  ```
- [ ] **4A.1**: Only proceed if ALL tests pass

#### **Phase 4A: Architecture Consolidation**
- [ ] **4A.1**: Analyze enhanced_datamart_manager.py features
- [ ] **4A.2**: Plan modular extraction
- [ ] **4A.3**: Implement consolidation
- [ ] **4A.4**: **CHECKPOINT**: `git commit -m "PHASE 4 COMPLETE: Architecture consolidated"`

---

## 🚨 **DEPENDENCY FAILURE PROTOCOLS**

### **If Phase 1 Fails:**
```bash
# Immediate rollback - don't proceed
git checkout backup-before-any-changes
git branch -D fix-circular-dependencies-only

# Analyze what went wrong
echo "Phase 1 failed - investigating specific step that broke"
```

### **If Phase 2 Fails:**
```bash
# Rollback only Phase 2 changes
git reset --hard HEAD~1  # Undo last Phase 2 commit
# Phase 1 remains intact - can retry Phase 2
```

### **If Phase 3 Fails:**
```bash  
# Rollback only Phase 3 changes
git reset --hard HEAD~1  # Undo last Phase 3 commit
# Phases 1+2 remain intact
```

### **If Phase 4 Fails:**
```bash
# Rollback only Phase 4 changes  
git reset --hard HEAD~1  # Undo last Phase 4 commit
# Core system (Phases 1-3) remains working
```

---

## ✅ **SUCCESS GATES**

Each phase has a SUCCESS GATE that must be passed:

### **Phase 1 Success Gate:**
```bash
python -c "
import sys; sys.path.append('src')
from backend.mcp.orchestrators.simple_qa_orchestrator import SimpleQAOrchestrator
from backend.mcp.orchestrators.advanced_qa_orchestrator import AdvancedQAOrchestrator  
from backend.datamart.core_manager import DataMartManager
print('✅ PHASE 1 SUCCESS - Proceed to Phase 2')
"
```

### **Phase 2 Success Gate:**
```bash
python -c "
import sys; sys.path.append('src')
from backend.mcp.orchestrators.enhanced_qa_orchestrator import EnhancedQAOrchestrator
qa = EnhancedQAOrchestrator()
result = qa.process_document({'content': 'test', 'file_path': 'test.txt'})
assert result['status'] == 'success'
print('✅ PHASE 2 SUCCESS - Proceed to Phase 3')
"
```

### **Phase 3 Success Gate:**
```bash
python -c "
import sys; sys.path.append('src')
import demo
print('✅ PHASE 3 SUCCESS - Proceed to Phase 4')
"
```

### **Phase 4 Success Gate:**
```bash
# All tests pass
bash tests/run_all_fix_tests.sh
echo "✅ PHASE 4 SUCCESS - All fixes complete"
```

---

**🎯 KEY INSIGHT**: By respecting dependencies and using incremental validation with rollback points, we can safely navigate the complex interdependencies without breaking the system.