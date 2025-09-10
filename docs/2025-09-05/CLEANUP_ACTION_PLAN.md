# TidyLLM Cleanup & Fresh Repository Action Plan

**Created**: September 5, 2025  
**Purpose**: Step-by-step execution plan for cleaning TidyLLM and creating fresh repository  
**Timeline**: 2-3 days estimated

---

## 📋 **PHASE 1: PREPARATION & BACKUP** (Day 1 Morning)

### **Step 1.1: Create Full Backup**
```bash
# Create timestamped backup
cd ..
cp -r github github_backup_20250905
# or zip it
zip -r TidyLLM_FULL_BACKUP_20250905.zip github/

# Verify backup
ls -la github_backup_20250905/
```

### **Step 1.2: Create Working Branch**
```bash
cd github
git checkout -b cleanup-for-fresh-repo
git push --set-upstream origin cleanup-for-fresh-repo
```

### **Step 1.3: Document Current State**
```bash
# Generate final reports
python comprehensive_python_audit.py
python submodule_analyzer.py
python uncertain_file_investigator.py

# Save reports to archive
mkdir cleanup_reports_20250905
mv *.csv *.json *report*.txt cleanup_reports_20250905/
```

---

## 🔧 **PHASE 2: REMOVE OBVIOUS DEAD CODE** (Day 1 Afternoon)

### **Step 2.1: Remove Inactive Submodules**
```bash
# Remove submodules that aren't git repos
rm -rf tidyllm-cross-integration
rm -rf tidyllm-enterprise  
rm -rf tidyllm-heiros
rm -rf tidyllm-whitepapers
rm -rf tidyllm-x-template

# Verify removal
ls -la | grep tidyllm-
# Should only show: tidyllm-compliance, tidyllm-demos, tidyllm-documents, 
#                   tidyllm-gateway, tidyllm-sentence, tidyllm-vectorqa
```

### **Step 2.2: Remove Empty/Template Files**
```python
# Create and run cleanup script
cat > remove_empty_files.py << 'EOF'
import os
from pathlib import Path

# List of empty files to remove (from investigation)
empty_files = [
    "tidyllm/knowledge_resource_server/__init__.py",
    "tidyllm/deprecated/old-tidyllm/compliance/consistency/__init__.py",
    "tidyllm/deprecated/old-tidyllm/compliance/evidence/__init__.py",
    "tidyllm/deprecated/old-tidyllm/compliance/model_risk/__init__.py",
    "tidyllm/deprecated/old-tidyllm/compliance/__init__.py",
    "tidyllm/deprecated/old-tidyllm/documents/classification/__init__.py",
    "tidyllm/deprecated/old-tidyllm/documents/extraction/__init__.py",
    "tidyllm/deprecated/old-tidyllm/documents/templates/__init__.py",
    "tidyllm/deprecated/old-tidyllm/documents/__init__.py",
    "tidyllm/deprecated/old-tidyllm/enterprise/analysis/model_risk/__init__.py",
    # Add all 70 files from audit
]

removed_count = 0
for file_path in empty_files:
    try:
        if Path(file_path).exists():
            os.remove(file_path)
            print(f"Removed: {file_path}")
            removed_count += 1
    except Exception as e:
        print(f"Error removing {file_path}: {e}")

print(f"\nTotal removed: {removed_count} files")
EOF

python remove_empty_files.py
```

### **Step 2.3: Clean Archive/Old Directories**
```bash
# Remove old archives that are duplicated in deprecated
rm -rf archive/old_src
rm -rf archive/old_tidyllm-vectorqa_sentence_backup_20250903
rm -rf old_archives/
rm -rf old_boss_demo_evidence/
rm -rf old_integration/
rm -rf old_src/
rm -rf old_tidyllm-vectorqa_sentence_backup_20250903/
rm -rf old-docs/
rm -rf old-tidyllm-docs/

# Keep only essential archives
ls -la archive/
```

### **Step 2.4: Commit Phase 2**
```bash
git add -A
git commit -m "Phase 2: Remove inactive submodules and empty files"
git push
```

---

## 🛠️ **PHASE 3: MIGRATE ARCHITECTURE VIOLATIONS** (Day 1 Evening - Day 2 Morning)

### **Step 3.1: Create Migration Script**
```python
# Create migration_fixer.py
cat > migration_fixer.py << 'EOF'
#!/usr/bin/env python3
"""
Architecture Migration Fixer
Fixes forbidden patterns in Python files
"""

import re
from pathlib import Path

class ArchitectureMigrator:
    def __init__(self):
        self.files_to_migrate = [
            "scripts/unified_credential_setup.py",
            "rudy_test_embeddings.py",
            "tests/2_test_s3_aws.py",
            "tidyllm/gateways/database_gateway.py",
            # Add all 89 files from investigation
        ]
        
    def fix_boto3_patterns(self, content: str) -> str:
        """Replace direct boto3 calls with UnifiedSessionManager"""
        # Replace boto3.client('s3')
        content = re.sub(
            r"boto3\.client\(['\"]s3['\"]\)",
            "self.session_mgr.get_s3_client()",
            content
        )
        
        # Add import if needed
        if "UnifiedSessionManager" not in content and "get_s3_client" in content:
            import_line = "from scripts.start_unified_sessions import UnifiedSessionManager\n"
            content = import_line + content
            
        return content
    
    def fix_psycopg2_patterns(self, content: str) -> str:
        """Replace direct psycopg2 with UnifiedSessionManager"""
        # Replace psycopg2.connect
        content = re.sub(
            r"psycopg2\.connect\([^)]+\)",
            "self.session_mgr.get_postgres_connection()",
            content
        )
        
        # Replace execute patterns
        content = re.sub(
            r"cursor\.execute\(([^)]+)\)",
            r"self.session_mgr.execute_postgres_query(\1)",
            content
        )
        
        return content
    
    def fix_numpy_patterns(self, content: str) -> str:
        """Replace numpy with tidyllm.tlm"""
        content = re.sub(r"import numpy as np", "from tidyllm.tlm.pure import ops as np", content)
        content = re.sub(r"from numpy import", "from tidyllm.tlm.pure.ops import", content)
        return content
    
    def fix_sklearn_patterns(self, content: str) -> str:
        """Replace sklearn with tidyllm.tlm implementations"""
        # Map sklearn to tidyllm equivalents
        sklearn_map = {
            "from sklearn.feature_extraction.text import TfidfVectorizer": 
                "from tidyllm_sentence import TfidfVectorizer",
            "from sklearn.cluster import KMeans":
                "from tidyllm.tlm.cluster import KMeans",
            "from sklearn.decomposition import PCA":
                "from tidyllm.tlm.decomp import PCA",
        }
        
        for old, new in sklearn_map.items():
            content = content.replace(old, new)
            
        return content
        
    def migrate_file(self, file_path: str):
        """Migrate a single file"""
        try:
            path = Path(file_path)
            if not path.exists():
                print(f"Skip (not found): {file_path}")
                return
                
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            original = content
            
            # Apply migrations
            content = self.fix_boto3_patterns(content)
            content = self.fix_psycopg2_patterns(content)
            content = self.fix_numpy_patterns(content)
            content = self.fix_sklearn_patterns(content)
            
            if content != original:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"Migrated: {file_path}")
            else:
                print(f"No changes: {file_path}")
                
        except Exception as e:
            print(f"Error migrating {file_path}: {e}")

    def run(self):
        """Run all migrations"""
        print("Starting architecture migration...")
        for file_path in self.files_to_migrate:
            self.migrate_file(file_path)
        print("Migration complete!")

if __name__ == "__main__":
    migrator = ArchitectureMigrator()
    migrator.run()
EOF

python migration_fixer.py
```

### **Step 3.2: Test Migrated Files**
```bash
# Run tests to verify migrations work
python tests/run_all_tests.py

# Check specific migrated files
python rudy_test_embeddings.py --test
python scripts/unified_credential_setup.py --verify
```

### **Step 3.3: Commit Phase 3**
```bash
git add -A
git commit -m "Phase 3: Migrate architecture violations to UnifiedSessionManager"
git push
```

---

## 🧹 **PHASE 4: REORGANIZE STRUCTURE** (Day 2 Afternoon)

### **Step 4.1: Create Clean Directory Structure**
```bash
# Create new clean structure
mkdir -p clean_tidyllm/{core,scripts,tests,docs,configs}

# Core functionality
cp -r tidyllm/knowledge_systems clean_tidyllm/core/
cp -r tidyllm/admin clean_tidyllm/core/
cp -r tidyllm/gateways clean_tidyllm/core/  # After migration
cp -r tidyllm/workflows clean_tidyllm/core/

# Scripts
cp scripts/start_unified_sessions.py clean_tidyllm/scripts/
cp scripts/unified_drop_zones.py clean_tidyllm/scripts/
cp scripts/production_tracking_drop_zones.py clean_tidyllm/scripts/
cp rudy_test_embeddings.py clean_tidyllm/scripts/  # After migration

# Tests
cp -r tests/*.py clean_tidyllm/tests/
cp -r tests/EVIDENCE clean_tidyllm/tests/

# Documentation (today's only)
cp CLEANUP_COMPLETED.md clean_tidyllm/docs/
cp IMPORTANT-CONSTRAINTS-FOR-THIS-CODEBASE.md clean_tidyllm/docs/
cp AWS_CREDENTIALS_SETUP.md clean_tidyllm/docs/
cp docs/Guidance-on-AWS-Sessions.md clean_tidyllm/docs/
cp docs/Session-Management-Migration-Map.md clean_tidyllm/docs/

# Configs
cp drop_zones/config.yaml clean_tidyllm/configs/drop_zones.yaml
cp tidyllm/admin/embeddings_settings.yaml clean_tidyllm/configs/
```

### **Step 4.2: Copy Active Submodules**
```bash
# Copy only active submodules
cp -r tidyllm-compliance clean_tidyllm/
cp -r tidyllm-documents clean_tidyllm/
cp -r tidyllm-gateway clean_tidyllm/
cp -r tidyllm-sentence clean_tidyllm/
cp -r tidyllm-vectorqa clean_tidyllm/
cp -r tlm clean_tidyllm/

# Skip tidyllm-demos until reviewed
```

### **Step 4.3: Create Setup Files**
```python
# Create clean_tidyllm/setup.py
cat > clean_tidyllm/setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="tidyllm",
    version="2.0.0",
    description="Clean TidyLLM with Unified Architecture",
    packages=find_packages(),
    install_requires=[
        "polars>=0.18.0",
        "dspy-ai>=2.0.0",
        "mlflow>=2.0.0",
        "psycopg2-binary>=2.9.0",
        "boto3>=1.26.0",
        "pyyaml>=6.0",
        "streamlit>=1.25.0",
    ],
    python_requires=">=3.8",
)
EOF

# Create clean_tidyllm/requirements.txt
cat > clean_tidyllm/requirements.txt << 'EOF'
polars>=0.18.0
dspy-ai>=2.0.0
mlflow>=2.0.0
psycopg2-binary>=2.9.0
boto3>=1.26.0
pyyaml>=6.0
streamlit>=1.25.0
pytest>=7.0.0
python-dotenv>=1.0.0
EOF

# Create clean_tidyllm/README.md
cat > clean_tidyllm/README.md << 'EOF'
# TidyLLM - Clean Architecture Edition

Unified session management architecture with TidyLLM native stack.

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure credentials:
   ```bash
   python scripts/unified_credential_setup.py
   ```

3. Run tests:
   ```bash
   python tests/run_all_tests.py
   ```

4. Start drop zones:
   ```bash
   python scripts/unified_drop_zones.py
   ```

## Architecture

- **UnifiedSessionManager**: Single source of truth for all external services
- **TidyLLM Native Stack**: tidyllm-sentence, tidyllm.tlm, polars
- **No forbidden dependencies**: No numpy, no sentence-transformers, no direct boto3/psycopg2
EOF
```

### **Step 4.4: Commit Phase 4**
```bash
git add clean_tidyllm/
git commit -m "Phase 4: Create clean repository structure"
git push
```

---

## ✅ **PHASE 5: VALIDATION & TESTING** (Day 2 Evening)

### **Step 5.1: Test Clean Installation**
```bash
# Create virtual environment
cd clean_tidyllm
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install clean version
pip install -e .

# Run all tests
python tests/run_all_tests.py
```

### **Step 5.2: Verify Architecture Compliance**
```python
# Create compliance_checker.py
cat > compliance_checker.py << 'EOF'
import os
import re
from pathlib import Path

def check_forbidden_patterns(root_dir="."):
    """Check for any remaining forbidden patterns"""
    forbidden = {
        'boto3.client': 0,
        'psycopg2.connect': 0,
        'import numpy': 0,
        'from sklearn': 0,
        'sentence_transformers': 0,
    }
    
    for py_file in Path(root_dir).rglob("*.py"):
        if 'deprecated' in str(py_file):
            continue
            
        with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        for pattern in forbidden:
            if pattern in content:
                forbidden[pattern] += 1
                print(f"VIOLATION: {pattern} found in {py_file}")
    
    print("\nCompliance Summary:")
    for pattern, count in forbidden.items():
        status = "✅ PASS" if count == 0 else f"❌ FAIL ({count} files)"
        print(f"{pattern}: {status}")

if __name__ == "__main__":
    check_forbidden_patterns()
EOF

python compliance_checker.py
```

### **Step 5.3: Size Comparison**
```bash
# Check size reduction
du -sh ../github  # Original
du -sh .         # Clean version

# Should see ~80% reduction
```

### **Step 5.4: Commit Phase 5**
```bash
git add compliance_checker.py
git commit -m "Phase 5: Validation and compliance verification"
git push
```

---

## 🚀 **PHASE 6: CREATE FRESH REPOSITORY** (Day 3)

### **Step 6.1: Initialize New Repository**
```bash
# Create fresh repo on GitHub (via web interface)
# Name: TidyLLM-Clean or TidyLLM-v2

# Clone and setup
cd ../
git clone https://github.com/RudyMartin/TidyLLM-Clean.git
cd TidyLLM-Clean

# Copy clean structure
cp -r ../github/clean_tidyllm/* .
```

### **Step 6.2: Initialize Git**
```bash
git add .
git commit -m "Initial commit: Clean TidyLLM with unified architecture"
git push origin main
```

### **Step 6.3: Setup CI/CD**
```yaml
# Create .github/workflows/test.yml
mkdir -p .github/workflows
cat > .github/workflows/test.yml << 'EOF'
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - uses: actions/setup-python@v2
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    - name: Run tests
      run: |
        python tests/run_all_tests.py
    - name: Check compliance
      run: |
        python compliance_checker.py
EOF
```

### **Step 6.4: Documentation & Tags**
```bash
# Tag the clean version
git tag -a v2.0.0 -m "Clean architecture with UnifiedSessionManager"
git push --tags

# Update README with badges
echo "![Tests](https://github.com/RudyMartin/TidyLLM-Clean/workflows/Tests/badge.svg)" >> README.md
```

---

## 📊 **SUCCESS METRICS**

### **Quantitative Goals**
- [ ] Reduce codebase from 773 to ~300 Python files
- [ ] Achieve 100% architecture compliance (0 forbidden patterns)
- [ ] Reduce repository size by >75%
- [ ] Maintain 100% test pass rate
- [ ] Remove all 5 inactive submodules

### **Qualitative Goals**
- [ ] Single source of truth for session management
- [ ] Clear, understandable directory structure
- [ ] No scattered configurations
- [ ] Consistent coding patterns
- [ ] Educational value maintained

---

## 🔄 **ROLLBACK PLAN**

If issues arise at any phase:

```bash
# Rollback to backup
cd ..
rm -rf github
cp -r github_backup_20250905 github
cd github

# Or reset git
git reset --hard origin/master
git clean -fd
```

---

## 📅 **TIMELINE**

| Phase | Duration | Status |
|-------|----------|---------|
| **Phase 1: Preparation** | 2 hours | ⏳ Ready |
| **Phase 2: Remove Dead Code** | 3 hours | ⏳ Ready |
| **Phase 3: Migration** | 4 hours | ⏳ Ready |
| **Phase 4: Reorganize** | 3 hours | ⏳ Ready |
| **Phase 5: Validation** | 2 hours | ⏳ Ready |
| **Phase 6: Fresh Repo** | 2 hours | ⏳ Ready |
| **Total** | ~16 hours | ⏳ Ready to Execute |

---

## ✅ **CHECKLIST BEFORE STARTING**

- [ ] Full backup created
- [ ] Team notified of cleanup
- [ ] GitHub access ready for new repo
- [ ] Test environment prepared
- [ ] Time blocked for execution
- [ ] Rollback plan understood

---

**Ready to Execute**: This plan provides step-by-step commands for complete cleanup and fresh repository creation.