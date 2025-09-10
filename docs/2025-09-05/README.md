# TidyLLM Strategic Test Suite

## Overview
This directory contains the consolidated strategic test suite for TidyLLM, reducing test complexity from 52+ individual test files to 8 strategic test suites.

## Test Consolidation Summary
- **Before**: 52 test files for 790 code files (15:1 ratio)
- **After**: 8 strategic test suites + 1 test runner (9 files total)
- **Reduction**: 83% fewer test files
- **Focus**: Business value over excessive coverage

## Strategic Test Suites

### Critical Tests (Must Pass)
1. **0_test_smoke.py** - Critical Path Verification
   - Basic imports and initialization
   - System availability checks
   - Performance sanity checks
   
2. **1_test_gateways.py** - Gateway System Tests  
   - Gateway initialization and registration
   - Service discovery and health checks
   - Inter-gateway communication

3. **7_test_security.py** - Security & Authentication Tests
   - Credential protection
   - Input validation
   - Access control mechanisms

### Optional Tests (Nice to Have)
4. **2_test_knowledge_server.py** - Knowledge MCP Server Tests
   - MCP tool functionality
   - Knowledge resource management
   - Search and retrieval operations

5. **3_test_s3_aws.py** - S3 & AWS Connectivity Tests
   - AWS authentication
   - S3 bucket operations
   - Cloud service integration

6. **4_test_integrations.py** - Cross-System Integration Tests
   - Component interaction
   - Data flow validation
   - Error propagation

7. **5_test_config.py** - Configuration Management Tests
   - Settings file loading
   - Environment variable handling
   - Configuration validation

8. **6_test_performance.py** - Performance & Load Tests
   - Response time benchmarks
   - Concurrent access handling
   - Memory usage patterns

## Running Tests

### Run All Tests
```bash
python tests/run_all_tests.py
```

### Run Specific Test Suite
```bash
python tests/run_all_tests.py --suite smoke
python tests/run_all_tests.py --suite gateways
```

### List Available Suites
```bash
python tests/run_all_tests.py --list
```

### Fail Fast Mode (Stop on First Critical Failure)
```bash
python tests/run_all_tests.py --fail-fast
```

## Test Numbering Convention
Tests are numbered 0-7 to ensure execution order:
- Lower numbers = higher priority
- 0-1, 7 = Critical tests
- 2-6 = Optional tests

## Windows Compatibility
All Unicode emojis have been replaced with text equivalents for Windows compatibility:
- ✅ → [OK]
- ❌ → [FAIL]  
- ⚠️ → [WARN]
- 🚀 → [START]
- 🚨 → [CRITICAL]
- 🎉 → [SUCCESS]

## Maintenance Philosophy
- **Quality over Quantity**: Focus on tests that provide business value
- **Strategic Coverage**: Test critical paths and integration points
- **Automation First**: Generate tests programmatically when possible
- **Consolidation**: Combine related tests into comprehensive suites
- **Performance**: Fast test execution for rapid feedback

## Success Criteria
The system is considered operational if:
1. All critical tests pass (smoke, gateways, security)
2. At least 50% of optional tests pass
3. No memory leaks or performance regressions detected

## Legacy Test Archive

### **Test #27: Gateway Dependencies System**
**File**: `27_test_gateway_dependencies.py`  
**Purpose**: Validates the automatic gateway dependency enabling system

**What It Tests**:
- ✅ **DSPyGateway** automatically enables **LLMGateway** 
- ✅ **HeirOSGateway** automatically enables both **DSPyGateway + LLMGateway**
- ✅ **LLMGateway** has no dependencies (foundation layer)
- ✅ Safety placeholders configured for future alternate states
- ✅ Complete dependency chain validation

**Test Functions**:
- `test_gateway_imports()` - Consolidated gateway imports
- `test_llm_gateway_dependencies()` - LLM foundation layer validation
- `test_dspy_gateway_dependencies()` - DSPy→LLM dependency
- `test_heiros_gateway_dependencies()` - HeirOS→DSPy+LLM dependencies
- `test_dependency_chain_logic()` - Complete chain validation
- `test_safety_placeholders()` - Alternate state placeholders

### **Test #28: Gateway Reorganization** 
**File**: `28_test_gateway_reorganization.py`  
**Purpose**: Validates the consolidated gateway architecture improvements

**What It Tests**:
- ✅ All gateways consolidated in `tidyllm/gateways/` for clarity
- ✅ Consistent **BaseGateway** interface implementation
- ✅ Gateway registry for dynamic loading (`get_gateway()`)
- ✅ All gateways can process requests successfully
- ✅ Backward compatibility with legacy naming

**Test Functions**:
- `test_consolidated_gateway_structure()` - Unified location validation
- `test_base_gateway_interface_compliance()` - BaseGateway implementation
- `test_gateway_registry_functionality()` - Dynamic loading system
- `test_gateway_processing_capabilities()` - Actual processing tests
- `test_backward_compatibility()` - Legacy support validation

## 📊 Standard Test Template

All tests follow this consistent pattern:

### **Header Format**:
```python
#!/usr/bin/env python3
"""
TidyLLM Standard Test #XX: Test Name
====================================

Description of what this test covers and why it's important.

IMPORTANT FOR AGENTS/LLMs:
- Specific testing guidance for AI agents
- Evidence collection requirements  
- Real vs simulated testing instructions
- Expected outcomes and validation criteria
"""
```

### **Evidence Collection**:
Every test includes:
```python
def save_evidence(test_name: str, evidence_data: dict) -> str:
    """Save test evidence to EVIDENCE folder."""
    evidence_dir = Path(__file__).parent / "EVIDENCE"
    evidence_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"evidence_{test_name}_{timestamp}.json"
    filepath = evidence_dir / filename
    
    with open(filepath, 'w') as f:
        json.dump(evidence_data, f, indent=2, default=str)
    
    return str(filepath)
```

### **Test Structure**:
- **Clear function names** with descriptive docstrings
- **Detailed progress reporting** with PASS/FAIL status
- **Comprehensive assertions** with helpful error messages  
- **Exception handling** with evidence saving
- **Project path setup** for proper imports

### **Result Reporting**:
```python
def main():
    """Run all tests with standardized reporting."""
    print("Test Suite Name")
    print("=" * 80)
    
    # Run tests and collect results
    passed = 0
    failed = 0
    
    # Final summary
    print(f"Tests Passed: {passed}")
    print(f"Tests Failed: {failed}")
    print(f"Evidence saved to: tests/EVIDENCE/")
    
    return failed == 0
```

## 🗂️ Evidence Collection System

### **Evidence Files**:
All tests create timestamped evidence files in `tests/EVIDENCE/`:
- **Format**: `evidence_{test_name}_{YYYYMMDD_HHMMSS}.json`
- **Content**: Complete test configuration, results, and metadata
- **Purpose**: Debugging, validation, and audit trails

### **Evidence Data Structure**:
```json
{
  "test_name": "gateway_dependencies",
  "timestamp": "2024-01-01T12:00:00",
  "expected_dependencies": ["llm"],
  "results": {
    "dependency_test": "PASS",
    "actual_dependencies": ["llm"],
    "dependency_object": {
      "use_dspy_gateway": false,
      "use_llm_gateway": true,
      "use_heiros_gateway": false
    }
  }
}
```

## 🚀 Running Tests

### **Individual Tests**:
```bash
# Run specific test
python tests/27_test_gateway_dependencies.py
python tests/28_test_gateway_reorganization.py

# View evidence
ls tests/EVIDENCE/evidence_*
```

### **Expected Results**:
```
TidyLLM Gateway Dependencies Test Suite
================================================================================
Tests Passed: 6
Tests Failed: 0
Evidence saved to: tests/EVIDENCE/

ALL GATEWAY DEPENDENCY TESTS PASSED!
SUCCESS: DSPyGateway automatically enables LLMGateway
SUCCESS: HeirOSGateway automatically enables DSPyGateway + LLMGateway
SUCCESS: LLMGateway is self-contained (foundation layer)
Gateway dependency system is working correctly!
```

## 💡 Gateway Architecture Overview

### **Dependency Chain**:
```
                Corporate Foundation
                       ↓
                 LLMGateway
                (self-contained)
                       ↑
                       │ requires
                       │
                 DSPyGateway ← requires ← HeirOSGateway
                 (needs LLM)              (needs both)
```

### **Automatic Enabling**:
- **Use DSPyGateway** → Automatically gets LLM governance
- **Use HeirOSGateway** → Automatically gets DSPy + LLM (full stack)  
- **Use LLMGateway** → Pure corporate control layer

### **Registry System**:
```python
from tidyllm.gateways import get_gateway

# Dynamic gateway creation
dspy_gateway = get_gateway("dspy", backend="mock")
llm_gateway = get_gateway("llm")
heiros_gateway = get_gateway("heiros")

# All implement BaseGateway interface
response = gateway.process_sync(input_data)
```

## 📋 Test Requirements

### **For New Tests**:
1. **Follow template pattern** exactly
2. **Include evidence collection** with timestamped files
3. **Use descriptive test function names** with docstrings
4. **Add IMPORTANT section** for AI agent guidance
5. **Number sequentially** (next available number)
6. **Test both success and failure cases**
7. **Include comprehensive assertions**

### **For AI Agents/LLMs**:
- **DO NOT** use mock/simulated functionality when real systems are available
- **ALWAYS** attempt real operations first before falling back
- **SAVE** complete evidence with metadata and results
- **INCLUDE** configuration objects, processing results, and error details
- **USE** timestamp-based filenames for evidence
- **VALIDATE** actual functionality vs expected behavior

## 🔍 Debugging Failed Tests

1. **Check evidence files** in `tests/EVIDENCE/` for detailed error information
2. **Review test output** for specific assertion failures
3. **Verify dependencies** are installed and configured correctly
4. **Check import paths** and project structure
5. **Validate configuration files** and environment setup

## ✅ Status

- ✅ **Gateway Dependencies**: Fully tested and validated
- ✅ **Gateway Reorganization**: Architecture improvements confirmed  
- ✅ **Evidence Collection**: Complete audit trails available
- ✅ **Template Compliance**: All tests follow standard pattern
- ✅ **Integration**: Seamlessly fits existing test framework

**Result**: TidyLLM's gateway system is comprehensively tested with full evidence collection and standardized reporting!