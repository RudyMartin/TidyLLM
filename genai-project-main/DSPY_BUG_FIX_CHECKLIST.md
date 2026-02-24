# DSPy Team Bug Fix Checklist
## Critical Issues Found in TidyLLM Codebase - Detailed Resolution Guide

This checklist addresses critical bugs discovered in the TidyLLM codebase that are likely present in similar DSPy implementations. Follow each step carefully to avoid missing critical fixes.

---

## 🚨 CRITICAL BUG #1: Provider Factory Function Misuse
### Issue Description
**Error:** `claude() got multiple values for argument 'model'`
**Cause:** Provider factory functions being called directly instead of using proper pipeline pattern
**Impact:** 100% failure rate in all LLM operations

### Step-by-Step Fix Process

#### Step 1: Identify Incorrect Usage Patterns
Search your codebase for these **INCORRECT** patterns:
```python
# ❌ WRONG - These will fail
response = claude(prompt, model="claude-3-haiku")
response = openai(prompt, model="gpt-4")
response = bedrock(prompt, model="anthropic.claude-v2")
```

#### Step 2: Replace with Correct Pipeline Pattern
Replace with these **CORRECT** patterns:
```python
# ✅ CORRECT - Provider factory + chat pipeline
from your_llm_module import chat, claude, openai, bedrock, llm_message

# For Claude
message = llm_message(prompt)
provider = claude("claude-3-haiku")
response = chat(provider)(message)

# For OpenAI
message = llm_message(prompt)
provider = openai("gpt-4")
response = chat(provider)(message)

# For Bedrock
message = llm_message(prompt)
provider = bedrock("anthropic.claude-v2")
response = chat(provider)(message)
```

#### Step 3: Files to Check (High Priority)
Search these file types for the incorrect patterns:
- [ ] `load_testing.py` or equivalent load test files
- [ ] `model_comparison.py` or benchmarking files
- [ ] Any files with `concurrent` or `parallel` in the name
- [ ] Demo scripts and example files
- [ ] Integration test files

#### Step 4: Specific Code Locations to Fix

**In Load Testing Files:**
```python
# ❌ WRONG (Line ~197 in our case)
response = claude(prompt, model="claude-3-haiku")

# ✅ CORRECT REPLACEMENT
message = llm_message(prompt)
provider = claude("claude-3-haiku")
response = chat(provider)(message)
```

**In Model Comparison Files:**
```python
# ❌ WRONG (ClaudeProvider class ~line 118)
response = claude(prompt, model=model)

# ✅ CORRECT REPLACEMENT
message = llm_message(prompt)
provider = claude(model)
response = chat(provider)(message)

# ❌ WRONG (BedrockProvider class ~line 166)  
response = bedrock(prompt, model=model)

# ✅ CORRECT REPLACEMENT
message = llm_message(prompt)
provider = bedrock(model)
response = chat(provider)(message)
```

#### Step 5: Verification Steps
After making changes, verify:
- [ ] Load tests show success_rate > 0.0 (not 0.0)
- [ ] Response objects contain actual text, not function references
- [ ] No errors containing "got multiple values for argument"
- [ ] Evidence files show successful_requests > 0

---

## 🚨 CRITICAL BUG #2: Function Objects Instead of Text Responses
### Issue Description
**Error:** Response contains `'<function chat.<locals>._chat at 0x000001A46B546DE0>'` instead of actual text
**Cause:** Incorrect function call order in chat pipeline
**Impact:** All responses are function objects, not text

### Step-by-Step Fix Process

#### Step 1: Identify Function Object Responses
Look for evidence files or logs showing:
```json
{
    "response": "'<function chat.<locals>._chat at 0x000001A46B546DE0>'",
    "response_type": "function"
}
```

#### Step 2: Check Call Order
Search for these **INCORRECT** call patterns:
```python
# ❌ WRONG - Returns function object
response = chat(message, provider)
response = chat(prompt, claude("model"))
```

#### Step 3: Fix Call Order
Replace with **CORRECT** patterns:
```python
# ✅ CORRECT - Returns actual text
response = chat(provider)(message)
response = chat(claude("model"))(llm_message(prompt))
```

#### Step 4: Verification
- [ ] Response should be string type, not function type
- [ ] Response should contain actual AI-generated text
- [ ] Evidence files should show meaningful response content

---

## 🚨 CRITICAL BUG #3: Evidence Data Truncation
### Issue Description
**Problem:** Evidence files only contain 9-10 lines instead of complete data
**Cause:** Manual evidence creation instead of preserving complete result objects
**Impact:** Loss of critical debugging information

### Step-by-Step Fix Process

#### Step 1: Identify Truncated Evidence Files
Look for evidence files with:
- Very short length (9-10 lines)
- Missing cost analysis
- Missing memory monitoring
- Missing detailed error information

#### Step 2: Replace Manual Evidence Creation
**WRONG Pattern:**
```python
# ❌ Manual evidence creation loses data
evidence = {
    "test_type": "load_test",
    "success": True,
    "summary": "Load test completed"
}
```

**CORRECT Pattern:**
```python
# ✅ Preserve complete result data
evidence = result.__dict__
# OR
evidence = {
    **result.__dict__,
    "additional_metadata": "value"
}
```

#### Step 3: Update Evidence Generation Code
- [ ] Replace manual dictionary creation with `result.__dict__`
- [ ] Ensure all test results preserve complete data
- [ ] Include memory monitoring data
- [ ] Include cost tracking information
- [ ] Include detailed error traces

---

## 🚨 ARCHITECTURAL BUG #4: Competing Implementations
### Issue Description
**Problem:** Multiple implementations of the same functionality causing import confusion
**Common Examples:** Sentence embeddings, database integrations, utility functions
**Impact:** Import errors, inconsistent behavior, maintenance burden

### Step-by-Step Resolution Process

#### Step 1: Identify Competing Implementations
Search for duplicate folders/files:
- [ ] Multiple `sentence` or `embeddings` directories
- [ ] Multiple `database` or `db` modules
- [ ] Multiple `utils` or `helpers` modules
- [ ] Multiple `integration` modules

#### Step 2: Choose Canonical Implementation
For each competing implementation:
- [ ] Identify which has the most features
- [ ] Identify which is actively maintained
- [ ] Identify which is used in tests
- [ ] Choose ONE as canonical

#### Step 3: Remove Redundant Implementations
- [ ] Delete redundant directories/files
- [ ] Update all imports to use canonical implementation
- [ ] Test that all functionality still works

#### Step 4: Example Fix Pattern (Sentence Embeddings)
```python
# ❌ WRONG - Internal competing implementations
from .sentence.embeddings import fit_transform
from ..vectorqa.sentence.tfidf import cosine_similarity

# ✅ CORRECT - Single canonical external package
try:
    import your_sentence_package as tls
    fit_transform = tls.tfidf_fit_transform
    cosine_similarity = tls.cosine_similarity
except ImportError:
    fit_transform = cosine_similarity = None
```

---

## 🚨 CRITICAL BUG #5: Undefined Variable References
### Issue Description
**Problem:** Variables referenced but never imported or defined
**Common Example:** `_dt` variable causing crashes
**Impact:** Runtime crashes in production

### Step-by-Step Fix Process

#### Step 1: Search for Undefined Variables
Look for variables that are:
- [ ] Referenced but not imported
- [ ] Referenced but not defined in current scope
- [ ] Private variables (starting with `_`) from other modules

#### Step 2: Common Problematic Patterns
```python
# ❌ WRONG - _dt is undefined
if _dt is not None:
    return _dt.something()

# ❌ WRONG - Missing imports
result = some_external_function()  # Where is this imported from?
```

#### Step 3: Fix with Proper Imports
```python
# ✅ CORRECT - Proper import pattern
try:
    import datatable as _dt
except ImportError:
    _dt = None

if _dt is not None:
    return _dt.something()
else:
    # Fallback implementation
    import polars as pl
    return pl.something()
```

---

## 📋 COMPREHENSIVE VERIFICATION CHECKLIST

### Before Starting Fixes
- [ ] Run full test suite and record baseline results
- [ ] Create backup of current codebase
- [ ] Document current evidence file locations
- [ ] Identify all provider types used (Claude, OpenAI, Bedrock, etc.)

### During Fixes - File-by-File Checklist
For each file you modify:
- [ ] Search for `claude(.*,.*model` patterns
- [ ] Search for `openai(.*,.*model` patterns  
- [ ] Search for `bedrock(.*,.*model` patterns
- [ ] Search for `chat(message,` patterns
- [ ] Replace with correct `chat(provider)(message)` pattern
- [ ] Verify imports are present for `llm_message`, `chat`, providers
- [ ] Test the specific file if possible

### After Each Fix
- [ ] Run affected tests to verify fix works
- [ ] Check evidence files for successful responses
- [ ] Verify response_type is string, not function
- [ ] Check success_rate > 0.0 in load tests

### Final Verification Steps
- [ ] Run complete test suite
- [ ] All load tests show success_rate > 0.0
- [ ] All evidence files contain complete data (not 9-10 lines)
- [ ] No responses containing function object strings
- [ ] No "multiple values for argument" errors
- [ ] All imports resolve correctly
- [ ] No undefined variable errors

### Evidence File Quality Check
Each evidence file should contain:
- [ ] Complete result data (20+ lines minimum)
- [ ] Memory monitoring information
- [ ] Cost tracking data
- [ ] Detailed timing information
- [ ] Complete error traces (if any failures)
- [ ] Success/failure statistics
- [ ] Response content samples

---

## 🔍 SPECIFIC SEARCH PATTERNS

Use these search patterns in your IDE/grep to find issues:

### Pattern 1: Incorrect Provider Usage
```bash
grep -r "claude(" . | grep "model.*="
grep -r "openai(" . | grep "model.*="
grep -r "bedrock(" . | grep "model.*="
```

### Pattern 2: Wrong Chat Call Order
```bash
grep -r "chat(" . | grep "message.*,.*provider"
grep -r "chat(" . | grep "prompt.*,.*claude"
```

### Pattern 3: Function Object Responses
```bash
grep -r "function.*locals" evidence/
grep -r "<function chat" evidence/
```

### Pattern 4: Truncated Evidence
```bash
find evidence/ -name "*.json" -exec wc -l {} \; | sort -n
# Files with < 15 lines are likely truncated
```

---

## ⚠️ CRITICAL SUCCESS INDICATORS

After completing all fixes, you MUST see:
1. **Load test success_rate > 0.0** (was 0.0 with bugs)
2. **Evidence files > 20 lines** (were 9-10 lines)
3. **Responses contain text** (not function objects)
4. **Zero "multiple values" errors**
5. **All imports resolve without errors**

If any of these indicators are not met, the fixes are incomplete.

---

## 📞 EMERGENCY DEBUGGING

If you still see issues after following this checklist:

1. **Check the exact error message** - compare with examples in this document
2. **Verify the call pattern** - ensure `chat(provider)(message)` not `chat(message, provider)`
3. **Check imports** - ensure all required functions are imported
4. **Test one file at a time** - isolate the problem
5. **Review evidence files** - they contain the actual error details

Remember: These bugs caused **100% failure rates** in production systems. Every single item in this checklist is critical for proper functionality.