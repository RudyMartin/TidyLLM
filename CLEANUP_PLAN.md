# TidyLLM Legacy Cleanup Plan

## 🗑️ SAFE TO REMOVE (Build Artifacts)

### Python Cache & Build Files
```bash
# These are generated files that can be safely removed:
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name ".pytest_cache" -type d -exec rm -rf {} +
find . -name "*.egg-info" -type d -exec rm -rf {} +
find . -name "dist" -type d -exec rm -rf {} +
find . -name "build" -type d -exec rm -rf {} +
```

**Total Space Saved**: ~2-3MB of cache files

## 📁 REORGANIZE (Root Level Test/Demo Files)

### Move to Proper Directories
These files are in the wrong location:
```
ROOT LEVEL (should be moved):
├── demo_api_examples.py      → tidyllm/examples/
├── flow_demo.py             → tidyllm/examples/
├── qa_test_runner.py        → tests/
├── test_ai_dropzone_manager.py → tests/
├── test_ai_dropzone_triggers.py → tests/
├── test_gateway_data_corruption.py → tests/
├── test_import.py           → tests/
└── test_s3_backup_system.py → tests/
```

## 🏗️ DUPLICATE PROJECTS (Consider Archiving)

### Separate Packages That Could Be Archived
```
./tidyllm-sentence/  - Sentence similarity package (separate from main TidyLLM)
./tlm/              - Time series/ML package (separate from main TidyLLM)
```

**Questions:**
- Are these still actively used?
- Could they be separate repositories?
- Should they be archived?

## ⚠️ DO NOT REMOVE

### Keep These Important Directories
```
./tests/            - Main test suite (recently standardized)
./tidyllm/tests/    - Package-level tests  
./tidyllm/examples/ - Code examples
./drop_zones/       - Active drop zone data
./workflows/        - Workflow definitions
```

## 🧹 CLEANUP COMMANDS

### Phase 1: Remove Build Artifacts (SAFE)
```bash
# Remove all Python cache files
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null  
find . -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null
find . -name "dist" -type d -exec rm -rf {} + 2>/dev/null

# Remove root level cache
rm -rf __pycache__ .pytest_cache tidyllm.egg-info
```

### Phase 2: Reorganize Files (CAREFUL)
```bash
# Move demo files to examples
mkdir -p tidyllm/examples
mv demo_api_examples.py tidyllm/examples/
mv flow_demo.py tidyllm/examples/

# Move test files to tests
mv test_*.py tests/
mv qa_test_runner.py tests/
```

## 📊 CLEANUP IMPACT

### Before Cleanup:
- **Build artifacts**: ~2-3MB
- **Scattered files**: 8 misplaced files
- **Cache pollution**: 40+ __pycache__ directories

### After Cleanup:
- **Space saved**: 2-3MB
- **Organized structure**: All files in proper locations  
- **Cleaner development**: No cache pollution in git status

## 🎯 RECOMMENDATION

1. **Start with Phase 1** (build artifacts) - completely safe
2. **Review Phase 2** (file moves) - verify files aren't actively used
3. **Evaluate separate packages** - tidyllm-sentence, tlm projects
4. **Add to .gitignore**: Ensure these artifacts don't return

Would you like me to execute Phase 1 cleanup (removing build artifacts)?