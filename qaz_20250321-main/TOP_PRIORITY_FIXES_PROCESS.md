# 🚨 TOP PRIORITY FIXES - IMPLEMENTATION PROCESS

**Created**: August 27, 2025  
**Status**: READY FOR EXECUTION  
**Priority**: CRITICAL - System Stability Issues  
**Estimated Time**: 2-3 days for complete resolution

---

## 🎯 **EXECUTIVE SUMMARY**

This document provides a **step-by-step process** to resolve the **4 critical issues** identified in the codebase analysis:

1. **🚨 CRITICAL**: Circular dependency crisis in DataMart ecosystem
2. **🔴 HIGH**: EnhancedQAOrchestrator import path errors  
3. **🔴 HIGH**: Missing demo file in enhanced launcher
4. **🟡 MEDIUM**: DataMart architecture consolidation

Each fix includes **detailed steps**, **validation tests**, and **rollback procedures**.

---

## 📋 **PRE-EXECUTION CHECKLIST**

### **Before Starting Any Fixes**
- [ ] **Backup entire codebase** to separate branch: `git checkout -b backup-before-fixes`
- [ ] **Document current system state**: Run all existing tests
- [ ] **Create fix tracking branch**: `git checkout -b fix-critical-dependencies`
- [ ] **Test current import behavior**: Run dependency validation tests
- [ ] **Verify MLflow integration status**: Ensure external systems work

### **Required Tools & Setup**
- [ ] Python environment with all dependencies
- [ ] Access to `src/` directory structure
- [ ] Text editor with find/replace capabilities
- [ ] Terminal access for running tests
- [ ] Git version control access

---

## 🚨 **FIX #1: RESOLVE CIRCULAR DEPENDENCIES** 

### **Status**: ❌ **CRITICAL - SYSTEM BREAKING**
### **Priority**: 🚨 **IMMEDIATE**
### **Estimated Time**: 8-12 hours

### **Problem Analysis**
```python
# Current Circular Chain:
datamart_numpy_substitution.py → advanced_qa_orchestrator.py → enhanced_datamart_manager.py → datamart_numpy_substitution.py

# Files Involved:
- src/backend/core/datamart_numpy_substitution.py
- src/backend/core/enhanced_datamart_manager.py  
- src/backend/mcp/orchestrators/advanced_qa_orchestrator.py
- Plus 5 downstream files importing from these
```

### **🎯 SOLUTION STRATEGY: Extract DataMart to Standalone Module**

#### **Step 1: Create New Canonical DataMart Module**
```bash
# Create new standalone module
mkdir -p src/backend/datamart/
touch src/backend/datamart/__init__.py
touch src/backend/datamart/core_manager.py
touch src/backend/datamart/numpy_substitute.py
```

#### **Step 2: Extract DataMartManager from advanced_qa_orchestrator.py**
```python
# NEW FILE: src/backend/datamart/core_manager.py
# Move lines 232-588 from advanced_qa_orchestrator.py
# This becomes the ONE canonical implementation

from enum import Enum
from typing import Dict, List, Any, Optional
import logging
import uuid
import json
from datetime import datetime

class DataMartMode(Enum):
    SIMPLE = "simple"
    ENHANCED = "enhanced"
    ADVANCED = "advanced"

class DataMartManager:
    # Move complete implementation here
    # 356 lines of sophisticated data handling
    # Progressive complexity (Simple/Enhanced/Advanced)
    # Uses datatable for high performance
```

#### **Step 3: Create NumPy Substitute Module**
```python
# NEW FILE: src/backend/datamart/numpy_substitute.py
# Move NumPy replacement functionality here
# Import DataMartManager from .core_manager (same package)

from .core_manager import DataMartManager, DataMartMode
from typing import Union, List, Tuple, Any, Optional, Dict
import random
import math
from datetime import datetime
import json

class NumpySubstitute:
    # Move all NumPy replacement code here
    # No circular imports - all in same package
```

#### **Step 4: Update All Import Statements**
```python
# BEFORE (circular):
from ..mcp.orchestrators.advanced_qa_orchestrator import DataMartManager, DataMartMode
from .datamart_numpy_substitution import DataMartManager, DataMartMode

# AFTER (clean):
from ..datamart.core_manager import DataMartManager, DataMartMode
from ..datamart.numpy_substitute import np, NumpySubstitute
```

### **🧪 VALIDATION TESTS FOR FIX #1**

#### **Test 1: Import Dependency Check**
```python
# FILE: tests/test_import_dependencies.py
import sys
import importlib

def test_no_circular_imports():
    """Test that all DataMart imports work without circular dependencies"""
    
    # Test individual module imports
    try:
        from src.backend.datamart.core_manager import DataMartManager, DataMartMode
        assert DataMartManager is not None
        print("✅ DataMartManager imports successfully")
    except ImportError as e:
        print(f"❌ DataMartManager import failed: {e}")
        return False
    
    try:
        from src.backend.datamart.numpy_substitute import np, NumpySubstitute
        assert np is not None
        print("✅ NumPy substitute imports successfully")  
    except ImportError as e:
        print(f"❌ NumPy substitute import failed: {e}")
        return False
    
    # Test orchestrator imports work
    try:
        from src.backend.mcp.orchestrators.advanced_qa_orchestrator import AdvancedQAOrchestrator
        orchestrator = AdvancedQAOrchestrator()
        print("✅ AdvancedQAOrchestrator imports successfully")
    except ImportError as e:
        print(f"❌ AdvancedQAOrchestrator import failed: {e}")
        return False
    
    print("✅ All imports work - no circular dependencies detected")
    return True

if __name__ == "__main__":
    test_no_circular_imports()
```

#### **Test 2: DataMart Functionality Test**
```python
# FILE: tests/test_datamart_functionality.py
def test_datamart_basic_operations():
    """Test that DataMart functionality works after refactoring"""
    
    from src.backend.datamart.core_manager import DataMartManager, DataMartMode
    
    # Test Simple mode
    simple_dm = DataMartManager(DataMartMode.SIMPLE)
    assert simple_dm.initialize_datamart() == True
    
    # Test Enhanced mode  
    enhanced_dm = DataMartManager(DataMartMode.ENHANCED)
    assert enhanced_dm.initialize_datamart() == True
    
    # Test Advanced mode
    advanced_dm = DataMartManager(DataMartMode.ADVANCED)
    assert advanced_dm.initialize_datamart() == True
    
    print("✅ All DataMart modes work correctly")
    return True

def test_numpy_substitute_functionality():
    """Test that NumPy substitute works after refactoring"""
    
    from src.backend.datamart.numpy_substitute import np, NumpySubstitute
    
    # Test basic array operations
    data = [1, 2, 3, 4, 5]
    arr = np.array(data)
    assert arr == data
    
    # Test mathematical operations
    mean_val = np.mean(data)
    assert abs(mean_val - 3.0) < 0.001
    
    # Test NumpySubstitute class
    numpy_sub = NumpySubstitute()
    zeros = numpy_sub.zeros(5)
    assert zeros == [0, 0, 0, 0, 0]
    
    print("✅ NumPy substitute works correctly")
    return True

if __name__ == "__main__":
    test_datamart_basic_operations()
    test_numpy_substitute_functionality()
```

#### **Test 3: System Integration Test**
```bash
# RUN: System-wide import test
python -c "
import sys
sys.path.append('src')

print('Testing system-wide imports...')

# Test all major components
from backend.datamart.core_manager import DataMartManager
from backend.mcp.orchestrators.simple_qa_orchestrator import SimpleQAOrchestrator
from backend.mcp.orchestrators.enhanced_qa_orchestrator import EnhancedQAOrchestrator  
from backend.mcp.orchestrators.advanced_qa_orchestrator import AdvancedQAOrchestrator
from backend.mcp.workers.file_classification_worker import FileClassificationWorker

print('✅ All major components import successfully')
print('✅ No circular dependency errors detected')
"
```

### **📝 STEP-BY-STEP EXECUTION FOR FIX #1**

#### **Phase 1: Preparation (30 minutes)**
1. **Create backup**: `git checkout -b backup-datamart-fix`
2. **Create new directories**: 
   ```bash
   mkdir -p src/backend/datamart/
   touch src/backend/datamart/__init__.py
   ```
3. **Run baseline tests**: Save current test results for comparison

#### **Phase 2: Extract DataMartManager (2 hours)**
1. **Create core_manager.py**:
   - Copy lines 232-588 from `advanced_qa_orchestrator.py`
   - Add proper imports and logging setup
   - Test imports work: `python -c "from src.backend.datamart.core_manager import DataMartManager"`

2. **Update advanced_qa_orchestrator.py**:
   - Remove DataMartManager definition (lines 232-588)
   - Add import: `from ...datamart.core_manager import DataMartManager, DataMartMode`
   - Test file still imports: `python -c "from src.backend.mcp.orchestrators.advanced_qa_orchestrator import AdvancedQAOrchestrator"`

#### **Phase 3: Extract NumPy Substitute (1 hour)**  
1. **Create numpy_substitute.py**:
   - Move all NumPy replacement code from `datamart_numpy_substitution.py`
   - Update imports to use local DataMartManager: `from .core_manager import DataMartManager`
   - Test functionality: Run numpy substitute test

#### **Phase 4: Update All Import Statements (2 hours)**
1. **Find all files importing DataMart**:
   ```bash
   grep -r "from.*datamart" src/
   grep -r "import.*DataMart" src/
   ```

2. **Update each file systematically**:
   - `enhanced_datamart_manager.py`: Update import path
   - `rag_qa_orchestrator.py`: Update import path  
   - `embedding_helper.py`: Update import path
   - `vst_mvr_comparison_worker.py`: Update import path
   - All test files: Update import paths

#### **Phase 5: Validation & Testing (2 hours)**
1. **Run import dependency test**: Should pass without circular import errors
2. **Run DataMart functionality test**: Should maintain all features
3. **Run system integration test**: Should import all components
4. **Run existing test suite**: Should not break existing functionality

#### **Phase 6: Cleanup (30 minutes)**
1. **Remove old files**: 
   - Delete `datamart_numpy_substitution.py` (functionality moved to datamart package)
   - Keep `enhanced_datamart_manager.py` but update imports
2. **Update documentation**: Note the new import paths
3. **Commit changes**: `git commit -m "Fix: Resolve circular dependencies in DataMart"`

### **🔄 ROLLBACK PROCEDURE FOR FIX #1**
If anything breaks during the fix:
```bash
# Immediate rollback
git checkout backup-datamart-fix
git branch -D fix-critical-dependencies

# Alternative: Revert specific changes
git revert <commit-hash>
```

---

## 🔴 **FIX #2: ENHANCED QA ORCHESTRATOR IMPORT ERRORS**

### **Status**: 🔴 **HIGH - FEATURE BREAKING**
### **Priority**: 🔴 **URGENT** 
### **Estimated Time**: 2-4 hours

### **Problem Analysis**
```python
# Current Import Errors in enhanced_qa_orchestrator.py:
from ...coordinators.document_inspector_coordinator import DocumentInspectorCoordinator  # ❌ PATH ERROR
from ...coordinators.caption_inspector_coordinator import CaptionInspectorCoordinator    # ❌ PATH ERROR
```

### **🎯 SOLUTION STRATEGY: Fix Import Paths or Create Missing Coordinators**

#### **Step 1: Investigate Current Coordinator Status**
```bash
# Find existing coordinators
find src/ -name "*coordinator*" -type f
find src/ -name "*inspector*" -type f

# Check if coordinators exist elsewhere
grep -r "DocumentInspectorCoordinator" src/
grep -r "CaptionInspectorCoordinator" src/
```

#### **Step 2A: If Coordinators Exist (Fix Imports)**
```python
# Update import paths in enhanced_qa_orchestrator.py
# BEFORE:
from ...coordinators.document_inspector_coordinator import DocumentInspectorCoordinator
from ...coordinators.caption_inspector_coordinator import CaptionInspectorCoordinator

# AFTER (example - actual path depends on discovery):
from ...core.coordinators.document_inspector_coordinator import DocumentInspectorCoordinator  
from ...core.coordinators.caption_inspector_coordinator import CaptionInspectorCoordinator
```

#### **Step 2B: If Coordinators Don't Exist (Create Minimal Implementations)**
```python
# CREATE: src/backend/core/coordinators/document_inspector_coordinator.py
class DocumentInspectorCoordinator:
    """Document Inspector Coordinator for Enhanced QA"""
    
    def __init__(self):
        self.coordinator_id = "doc_inspector_001"
        
    def inspect_document(self, document):
        """Basic document inspection functionality"""
        return {
            'toc_extracted': False,
            'bibliography_found': False,
            'links_validated': False,
            'structure_analysis': 'basic',
            'confidence_score': 0.7
        }

# CREATE: src/backend/core/coordinators/caption_inspector_coordinator.py  
class CaptionInspectorCoordinator:
    """Caption Inspector Coordinator for Enhanced QA"""
    
    def __init__(self):
        self.coordinator_id = "caption_inspector_001"
        
    def inspect_captions(self, document):
        """Basic caption inspection functionality"""
        return {
            'captions_found': 0,
            'caption_quality': 'unknown',
            'image_references': [],
            'table_captions': [],
            'confidence_score': 0.6
        }
```

### **🧪 VALIDATION TESTS FOR FIX #2**

#### **Test 1: EnhancedQAOrchestrator Import Test**
```python
# FILE: tests/test_enhanced_qa_imports.py
def test_enhanced_qa_orchestrator_imports():
    """Test that EnhancedQAOrchestrator imports without errors"""
    
    try:
        from src.backend.mcp.orchestrators.enhanced_qa_orchestrator import EnhancedQAOrchestrator
        orchestrator = EnhancedQAOrchestrator()
        print("✅ EnhancedQAOrchestrator imports and initializes successfully")
        return True
    except ImportError as e:
        print(f"❌ EnhancedQAOrchestrator import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ EnhancedQAOrchestrator initialization failed: {e}")
        return False

def test_coordinator_functionality():
    """Test that coordinators work correctly"""
    
    from src.backend.mcp.orchestrators.enhanced_qa_orchestrator import EnhancedQAOrchestrator
    
    orchestrator = EnhancedQAOrchestrator()
    
    # Test with sample document
    test_document = {
        'content': 'This is a test document for validation.',
        'file_path': 'test.txt',
        'metadata': {'type': 'text', 'title': 'Test Document'}
    }
    
    try:
        result = orchestrator.process_document(test_document)
        assert result['status'] == 'success'
        print("✅ EnhancedQAOrchestrator processes documents successfully")
        return True
    except Exception as e:
        print(f"❌ EnhancedQAOrchestrator processing failed: {e}")
        return False

if __name__ == "__main__":
    test_enhanced_qa_orchestrator_imports()
    test_coordinator_functionality()
```

### **📝 STEP-BY-STEP EXECUTION FOR FIX #2**

#### **Phase 1: Investigate Current State (30 minutes)**
1. **Find coordinator files**: `find src/ -name "*coordinator*" -type f`
2. **Check import references**: `grep -r "coordinator" src/backend/mcp/orchestrators/enhanced_qa_orchestrator.py`
3. **Determine if coordinators exist elsewhere**: Search entire codebase

#### **Phase 2: Implement Solution (1-2 hours)**
**Option A**: Fix import paths (if coordinators exist)
**Option B**: Create minimal coordinator implementations (if they don't exist)

#### **Phase 3: Test & Validate (1 hour)**  
1. **Run import test**: Verify EnhancedQAOrchestrator imports
2. **Run functionality test**: Verify document processing works
3. **Integration test**: Verify with SimpleQAOrchestrator inheritance

### **🔄 ROLLBACK PROCEDURE FOR FIX #2**
```bash
# If imports were changed, revert the file
git checkout HEAD -- src/backend/mcp/orchestrators/enhanced_qa_orchestrator.py

# If coordinators were created, remove them
rm -rf src/backend/core/coordinators/ 
```

---

## 🔴 **FIX #3: MISSING DEMO FILE IN ENHANCED LAUNCHER**

### **Status**: 🔴 **HIGH - FEATURE BREAKING**
### **Priority**: 🔴 **URGENT**
### **Estimated Time**: 1-2 hours

### **Problem Analysis**
```python
# enhanced.py line 71:
demo_file = project_root / "src" / "demo.py"  # ❌ FILE DOES NOT EXIST
```

### **🎯 SOLUTION STRATEGY: Create Missing Demo File or Update Path**

#### **Option A: Create Missing demo.py File**
```python
# CREATE: src/demo.py
#!/usr/bin/env python3
"""
Enhanced Demo - Entry point for enhanced QA system demos
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def main():
    st.set_page_config(
        page_title="Enhanced QA Demo",
        page_icon="⚡",
        layout="wide"
    )
    
    st.markdown("# ⚡ Enhanced QA Demo")
    st.markdown("Welcome to the Enhanced Document QA System!")
    
    # Import and run enhanced functionality
    try:
        from backend.mcp.orchestrators.enhanced_qa_orchestrator import EnhancedQAOrchestrator
        
        if st.button("Test Enhanced QA"):
            orchestrator = EnhancedQAOrchestrator()
            
            test_doc = {
                'content': 'This is a test document for the enhanced QA system.',
                'file_path': 'test.txt',
                'metadata': {'type': 'text'}
            }
            
            result = orchestrator.process_document(test_doc)
            st.success("✅ Enhanced QA system working!")
            st.json(result)
            
    except Exception as e:
        st.error(f"❌ Enhanced QA system error: {e}")

if __name__ == "__main__":
    main()
```

#### **Option B: Update Enhanced Launcher Path**
```python
# UPDATE: enhanced.py line 71
# BEFORE:
demo_file = project_root / "src" / "demo.py"

# AFTER (point to existing streamlit demo):
demo_file = project_root / "streamlit_mvr_demo.py"
# OR point to simple demo:
demo_file = project_root / "simple_demo.py"
```

### **🧪 VALIDATION TESTS FOR FIX #3**

#### **Test 1: Enhanced Launcher Test**
```bash
# FILE: tests/test_enhanced_launcher.sh
#!/bin/bash

echo "Testing Enhanced Launcher..."

# Test that enhanced.py runs without errors
python enhanced.py &
LAUNCHER_PID=$!

# Wait a moment for startup
sleep 3

# Check if process is still running (no immediate crash)
if ps -p $LAUNCHER_PID > /dev/null; then
    echo "✅ Enhanced launcher started successfully"
    kill $LAUNCHER_PID
    exit 0
else
    echo "❌ Enhanced launcher failed to start"
    exit 1
fi
```

#### **Test 2: Demo File Functionality Test**  
```python
# FILE: tests/test_demo_functionality.py
def test_demo_file_exists_and_works():
    """Test that demo file exists and can be imported"""
    
    from pathlib import Path
    
    # Test that demo file exists
    demo_path = Path("src/demo.py")
    assert demo_path.exists(), f"Demo file not found at {demo_path}"
    
    # Test that demo file can be imported
    import sys
    sys.path.insert(0, "src")
    
    try:
        import demo
        print("✅ Demo file imports successfully")
        return True
    except ImportError as e:
        print(f"❌ Demo file import failed: {e}")
        return False

if __name__ == "__main__":
    test_demo_file_exists_and_works()
```

### **📝 STEP-BY-STEP EXECUTION FOR FIX #3**

#### **Phase 1: Choose Solution (15 minutes)**
1. **Assess existing demos**: Look at `simple_demo.py`, `streamlit_mvr_demo.py`
2. **Decide approach**: Create new demo.py OR update launcher path
3. **Recommended**: Create new demo.py for consistency

#### **Phase 2: Implement Solution (30 minutes)**
1. **Create src/demo.py** with enhanced QA functionality
2. **Test import**: `python -c "import sys; sys.path.append('src'); import demo"`
3. **Test functionality**: Run demo file directly

#### **Phase 3: Validate Launcher (30 minutes)**
1. **Test enhanced launcher**: `python enhanced.py`
2. **Verify no errors**: Should start without file not found errors
3. **Test integration**: Verify launcher finds and uses demo correctly

### **🔄 ROLLBACK PROCEDURE FOR FIX #3**
```bash
# If demo.py was created:
rm src/demo.py

# If enhanced.py was modified:
git checkout HEAD -- enhanced.py
```

---

## 🟡 **FIX #4: DATAMART ARCHITECTURE CONSOLIDATION**

### **Status**: 🟡 **MEDIUM - ARCHITECTURAL CLEANUP**
### **Priority**: 🟡 **POST-CRITICAL-FIXES**
### **Estimated Time**: 4-6 hours

### **Problem Analysis**
Three competing DataMart implementations without clear hierarchy:
1. `advanced_qa_orchestrator.py` (lines 232-588) - Most complete  
2. `datamart_numpy_substitution.py` - NumPy replacement system
3. `enhanced_datamart_manager.py` - Live/remote integration

### **🎯 SOLUTION STRATEGY: Consolidate to Single Architecture**

**Note**: This should be done AFTER Fix #1 (circular dependencies) is completed.

#### **Recommended Architecture**:
```
src/backend/datamart/
├── __init__.py
├── core_manager.py          # Main DataMart (from Fix #1)  
├── numpy_substitute.py      # NumPy replacement (from Fix #1)
├── remote_manager.py        # AWS/PostgreSQL features
└── connection_manager.py    # Connection handling
```

### **📝 STEP-BY-STEP EXECUTION FOR FIX #4**

This fix builds on Fix #1 and should be executed after circular dependencies are resolved.

#### **Phase 1: Analyze Enhanced DataMart Manager (1 hour)**
1. **Review enhanced_datamart_manager.py**: Identify unique features
2. **Identify overlap**: What's duplicated from core DataMart?
3. **Plan integration**: How to merge unique features into core module

#### **Phase 2: Extract Unique Features (2 hours)**  
1. **Create remote_manager.py**: Move AWS/PostgreSQL/Redis features
2. **Create connection_manager.py**: Move connection handling
3. **Update core_manager.py**: Add hooks for remote features

#### **Phase 3: Update References (2 hours)**
1. **Find all enhanced_datamart_manager imports**
2. **Update to use new modular architecture**  
3. **Test all functionality maintained**

#### **Phase 4: Validation & Cleanup (1 hour)**
1. **Run comprehensive DataMart tests**
2. **Remove old enhanced_datamart_manager.py**
3. **Update documentation**

---

## 🧪 **COMPREHENSIVE TEST SUITE**

### **Master Test Runner**
```bash
# FILE: tests/run_all_fix_tests.sh
#!/bin/bash

echo "🧪 Running Comprehensive Fix Validation Tests"
echo "=============================================="

# Test 1: Import Dependencies  
echo "1. Testing Import Dependencies..."
python tests/test_import_dependencies.py
if [ $? -ne 0 ]; then
    echo "❌ Import dependency test FAILED"
    exit 1
fi

# Test 2: DataMart Functionality
echo "2. Testing DataMart Functionality..."  
python tests/test_datamart_functionality.py
if [ $? -ne 0 ]; then
    echo "❌ DataMart functionality test FAILED"
    exit 1
fi

# Test 3: Enhanced QA Orchestrator
echo "3. Testing Enhanced QA Orchestrator..."
python tests/test_enhanced_qa_imports.py
if [ $? -ne 0 ]; then
    echo "❌ Enhanced QA test FAILED"
    exit 1
fi

# Test 4: Demo File
echo "4. Testing Demo File..."
python tests/test_demo_functionality.py
if [ $? -ne 0 ]; then
    echo "❌ Demo file test FAILED"
    exit 1
fi

# Test 5: Enhanced Launcher
echo "5. Testing Enhanced Launcher..."
bash tests/test_enhanced_launcher.sh
if [ $? -ne 0 ]; then
    echo "❌ Enhanced launcher test FAILED"
    exit 1
fi

# Test 6: System Integration
echo "6. Testing System Integration..."
python -c "
import sys
sys.path.append('src')

# Test all major components import without errors
from backend.datamart.core_manager import DataMartManager
from backend.mcp.orchestrators.simple_qa_orchestrator import SimpleQAOrchestrator
from backend.mcp.orchestrators.enhanced_qa_orchestrator import EnhancedQAOrchestrator
from backend.mcp.orchestrators.advanced_qa_orchestrator import AdvancedQAOrchestrator

print('✅ All system components import successfully')
"

if [ $? -ne 0 ]; then
    echo "❌ System integration test FAILED"
    exit 1
fi

echo ""
echo "🎉 ALL TESTS PASSED!"
echo "✅ System is ready for production"
echo "=============================================="
```

---

## 📊 **EXECUTION TIMELINE & TRACKING**

### **Day 1 (8 hours)**
- [ ] **Morning (4 hours)**: Fix #1 - Circular Dependencies
  - [ ] Setup & preparation (1 hour)
  - [ ] Extract DataMartManager (2 hours) 
  - [ ] Initial testing (1 hour)
- [ ] **Afternoon (4 hours)**: Complete Fix #1
  - [ ] Update all import statements (2 hours)
  - [ ] Comprehensive testing (1 hour)
  - [ ] Documentation & cleanup (1 hour)

### **Day 2 (6 hours)**  
- [ ] **Morning (3 hours)**: Fix #2 - Enhanced QA Orchestrator
  - [ ] Investigate coordinator status (1 hour)
  - [ ] Implement solution (1 hour)
  - [ ] Testing & validation (1 hour)
- [ ] **Afternoon (3 hours)**: Fix #3 - Demo File + Start Fix #4
  - [ ] Create missing demo file (1 hour)
  - [ ] Test enhanced launcher (1 hour)  
  - [ ] Begin DataMart consolidation analysis (1 hour)

### **Day 3 (4 hours) - Optional**
- [ ] **Complete Fix #4**: DataMart Architecture Consolidation
  - [ ] Extract unique features (2 hours)
  - [ ] Update references (1 hour) 
  - [ ] Final validation (1 hour)

### **Progress Tracking Checklist**
- [ ] Fix #1 Phase 1: Preparation ✓
- [ ] Fix #1 Phase 2: Extract DataMartManager ✓  
- [ ] Fix #1 Phase 3: Extract NumPy Substitute ✓
- [ ] Fix #1 Phase 4: Update Imports ✓
- [ ] Fix #1 Phase 5: Validation ✓
- [ ] Fix #1 Phase 6: Cleanup ✓
- [ ] Fix #2 Complete ✓
- [ ] Fix #3 Complete ✓
- [ ] Fix #4 Complete ✓ (Optional)
- [ ] All Tests Passing ✓
- [ ] Documentation Updated ✓

---

## 🚨 **EMERGENCY PROCEDURES**

### **If System Breaks During Fixes**
1. **Immediate Response**: `git checkout backup-before-fixes`
2. **Assess damage**: What specific fix caused the issue?
3. **Partial rollback**: Revert only the problematic changes
4. **Analyze & retry**: Fix the issue before continuing

### **If Tests Fail**
1. **Don't proceed** to next fix until current fix passes all tests
2. **Debug systematically**: Use test output to identify specific problems
3. **Check import paths**: Most common issue is incorrect relative imports
4. **Verify file locations**: Ensure all files are in expected locations

### **Communication Protocol**
- **Document all changes** in git commits with descriptive messages
- **Update progress tracking** checklist after each completed phase
- **Note any deviations** from the plan in commit messages or comments
- **Save test results** for comparison and debugging

---

## 🎯 **SUCCESS CRITERIA**

### **Fix #1 Success Criteria**
- [ ] All DataMart-related imports work without circular dependency errors
- [ ] All existing DataMart functionality preserved
- [ ] Import dependency test passes
- [ ] DataMart functionality test passes  
- [ ] System integration test passes

### **Fix #2 Success Criteria**  
- [ ] EnhancedQAOrchestrator imports without errors
- [ ] Document processing works in enhanced mode
- [ ] Enhanced QA test passes
- [ ] No regression in SimpleQAOrchestrator functionality

### **Fix #3 Success Criteria**
- [ ] Enhanced launcher starts without "file not found" errors
- [ ] Demo file exists and is functional
- [ ] Demo functionality test passes
- [ ] Enhanced launcher test passes

### **Fix #4 Success Criteria** (Optional)
- [ ] Single, clear DataMart architecture
- [ ] All unique features preserved
- [ ] Performance maintained or improved
- [ ] Clear documentation of architecture decisions

### **Overall Success Criteria**
- [ ] **No circular import errors** in entire system
- [ ] **All orchestrators work** (Simple, Enhanced, Advanced)
- [ ] **All demos launch** without errors
- [ ] **All existing functionality preserved**
- [ ] **Master test suite passes** 100%
- [ ] **System is production-ready**

---

**🎉 END RESULT**: A stable, well-architected system with no circular dependencies, working enhanced features, and clear modular design ready for production use and future development.