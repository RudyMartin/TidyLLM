# TidyLLM Package Dependency Issues

⚠️ **CRITICAL**: This package currently has architecture violations that prevent it from being truly standalone.

## Package Status
- ✅ **Installs**: Package installs successfully with `pip install -e .`
- ✅ **Imports**: Basic import works: `import tidyllm`
- ❌ **Standalone**: Many functions fail when used due to external dependencies
- ❌ **GitHub Ready**: Cannot be deployed as independent repo without refactoring

## Major Dependency Violations

### 1. Parent Directory Imports
**Files**: `cli.py`
```python
# Lines 274, 288
from .. import qa_processor      # VIOLATION: Imports from parent package
from .. import qa_test_runner    # VIOLATION: Imports from parent package
```
**Impact**: CLI functionality breaks when package is standalone

### 2. Sys.Path Manipulations
**Files**: Multiple flow examples, session management
```python
# flow/examples/*.py (multiple files)
sys.path.append(str(Path(__file__).parent.parent.parent.parent))  # VIOLATION: Goes 4 levels up

# infrastructure/session/s3.py
sys.path.append(str(Path(__file__).parent.parent / 'tidyllm' / 'admin'))  # VIOLATION: Assumes parent structure

# infrastructure/session_management.py
sys.path.insert(0, str(admin_dir.parent.parent))  # VIOLATION: Adds parent.parent to path
```
**Impact**: Runtime failures when external directories don't exist

### 3. Hardcoded External Paths
**Files**: `infrastructure/connection_pool.py`
```python
# Line 286
Path("../tidyllm/admin/settings.yaml")  # VIOLATION: Hardcoded relative path outside package
```
**Impact**: Configuration loading fails in standalone deployment

### 4. Admin Folder Hunting
**Files**: Multiple session management files
```python
# infrastructure/session_management.py, diagnostics.py
for path in [current_dir] + list(current_dir.parents):  # VIOLATION: Searches up directory tree
```
**Impact**: Package assumes specific external folder structure exists

### 5. Workflow Config External Paths
**Files**: `workflow_configs/*.yaml` (multiple files)
```yaml
# All workflow configs have patterns like:
destination: "../mvr_qa/"        # VIOLATION: External path reference
destination: "../completed/"     # VIOLATION: External path reference
```
**Impact**: Workflow processing fails without expected external directories

## Affected Functionality

### Works in Standalone Mode:
- ✅ Basic package import
- ✅ Core class definitions
- ✅ Gateway registry (if properly initialized)
- ✅ Domain models and value objects

### Fails in Standalone Mode:
- ❌ CLI commands (`tidyllm` command line interface)
- ❌ Session management (AWS, DB connections)
- ❌ Workflow processing
- ❌ Admin/diagnostic functions
- ❌ Flow examples
- ❌ Configuration loading from external files

## Refactoring Required for GitHub Deployment

### Priority 1: Critical Import Fixes
- Remove all `from ..` imports that escape package
- Internalize or stub missing external dependencies

### Priority 2: Path Independence
- Remove all `sys.path` manipulations
- Use proper package-relative imports
- Internalize configuration files

### Priority 3: Self-Contained Configuration
- Move critical config files into package
- Use environment variables for external dependencies
- Provide sensible defaults for missing external resources

### Priority 4: Workflow Cleanup
- Fix hardcoded paths in YAML configs
- Use package-relative or environment-based paths

## Deployment Options

### Option A: Full Refactor (Recommended for Production)
- Fix all violations above
- Results in clean, standalone package
- ~50+ files need modification
- Estimated effort: 2-3 days

### Option B: Minimal Viable Package
- Fix only import violations that break installation
- Add graceful degradation for missing dependencies
- Document limitations clearly
- Estimated effort: 4-6 hours

### Option C: Current State (Development Only)
- Keep in qa-shipping parent structure
- Use for development and testing
- Not suitable for independent deployment

## Future Feature Markers

All files with dependency violations have been marked with:
```python
# future_feature: remove external dependency on [specific issue]
```

## Recommendation

For immediate GitHub deployment: **Use Option B** - create minimal viable package with graceful degradation and clear documentation of limitations.

For production deployment: **Plan Option A** - full refactor to create truly independent package.

---
*This document generated automatically during dependency analysis - update as refactoring progresses*