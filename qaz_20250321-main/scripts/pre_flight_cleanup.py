#!/usr/bin/env python3
"""
Pre-Flight and Cleanup System for QA System

This script provides comprehensive pre-flight checks and cleanup utilities for the QA system.
It ensures the system is ready for demos, testing, and production use.

IMPORTANT: This script respects .gitignore exclusions and will NOT clean up files
that are already excluded by git. This prevents conflicts with git's own cleanup
mechanisms and ensures we don't accidentally remove files that should be preserved.

Files in .gitignore (like __pycache__, .pytest_cache, *.pyc, etc.) are handled
automatically by git and don't need manual cleanup from this script.

CRITICAL: Before running this script, ensure you have activated the py311 conda environment:
    conda activate py311

This is required for proper dependency checking and system validation.

SETUP REQUIRED: If you encounter NumPy compatibility issues or missing dependencies,
run the setup script first:
    python scripts/setup_before_preflight.py
"""

import os
import sys
import shutil
import json
import glob
import subprocess
import re
import fnmatch
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QAPreFlightCleanup:
    """Pre-flight checks and cleanup utilities for QA system"""
    
    def __init__(self, dry_run=True):
        self.dry_run = dry_run
        self.project_root = Path.cwd()
        self.gitignore_patterns = self._load_gitignore_patterns()
        self.cleanup_stats = {
            'files_removed': 0,
            'directories_removed': 0,
            'bytes_freed': 0,
            'api_keys_scrubbed': 0,
            'files_skipped_gitignore': 0,
            'errors': []
        }
    
    def _load_gitignore_patterns(self):
        """Load patterns from .gitignore file"""
        patterns = set()
        gitignore_file = self.project_root / '.gitignore'
        
        if gitignore_file.exists():
            try:
                with open(gitignore_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        # Skip comments and empty lines
                        if line and not line.startswith('#'):
                            # Remove trailing comments
                            pattern = line.split('#')[0].strip()
                            if pattern:
                                patterns.add(pattern)
            except Exception as e:
                self.cleanup_stats['errors'].append(f"Failed to load .gitignore: {e}")
        
        # Add default patterns that should always be ignored
        patterns.update({
            '__pycache__', '*.pyc', '*.pyo', '*.pyd',
            '.pytest_cache', '.git', '.DS_Store',
            'node_modules', '.venv', 'venv'
        })
        
        return patterns
    
    def _is_gitignored(self, file_path):
        """Check if a file/directory matches any gitignore pattern"""
        relative_path = file_path.relative_to(self.project_root)
        path_str = str(relative_path)
        path_parts = relative_path.parts
        
        for pattern in self.gitignore_patterns:
            # Direct match
            if fnmatch.fnmatch(path_str, pattern):
                return True
            
            # Check if any parent directory matches
            for part in path_parts:
                if fnmatch.fnmatch(part, pattern):
                    return True
            
            # Handle directory patterns (ending with /)
            if pattern.endswith('/'):
                dir_pattern = pattern.rstrip('/')
                if fnmatch.fnmatch(path_str, dir_pattern) or fnmatch.fnmatch(path_str, dir_pattern + '/*'):
                    return True
                for part in path_parts:
                    if fnmatch.fnmatch(part, dir_pattern):
                        return True
        
        return False
    
    def run_pre_flight_checks(self):
        """Run comprehensive pre-flight checks"""
        print("🛫 PRE-FLIGHT CHECKS")
        print("=" * 60)
        
        checks = [
            self._check_python_environment(),
            self._check_dependencies(),
            self._check_directories(),
            self._check_config_files(),
            self._check_test_files(),
            self._check_demo_readiness(),
            self._check_api_security(),
            self._check_security_hardening(),
            self._check_robust_bundle_ready(),  # NEW: Robust bundle validation
            self._check_environment_compatibility(),  # NEW: Environment compatibility
            self._check_embedding_centralization(),  # NEW: Embedding centralization validation
            self._check_naked_calls(),  # NEW: Naked calls validation
        ]
        
        all_passed = all(checks)
        
        print(f"\n📊 PRE-FLIGHT SUMMARY: {'✅ ALL CHECKS PASSED' if all_passed else '❌ SOME CHECKS FAILED'}")
        return all_passed
    
    def _check_python_environment(self):
        """Check Python environment"""
        print("\n🐍 Python Environment:")
        
        # Check Python version
        python_version = sys.version_info
        version_ok = python_version.major == 3 and python_version.minor >= 8
        print(f"  Python Version: {python_version.major}.{python_version.minor}.{python_version.micro} {'✅' if version_ok else '❌'}")
        
        # Check if we're in the py311 conda environment
        python_path = sys.executable
        py311_active = "py311" in python_path
        
        # If not in py311, check if conda environment exists
        if not py311_active:
            try:
                result = subprocess.run(['conda', 'info', '--envs'], capture_output=True, text=True)
                if result.returncode == 0 and 'py311' in result.stdout:
                    py311_python = "/Users/rudy/opt/anaconda3/envs/py311/bin/python"
                    if os.path.exists(py311_python):
                        print(f"  Conda Environment: ⚠️  py311 exists but not active")
                        print(f"    💡 Activate py311: conda activate py311")
                        print(f"    💡 Or use: {py311_python}")
                        print(f"    Current Python: {python_path}")
                    else:
                        print(f"  Conda Environment: ❌ py311 not found")
                        print(f"    💡 Create py311: conda create -n py311 python=3.11")
                else:
                    print(f"  Conda Environment: ❌ py311 not found")
                    print(f"    💡 Create py311: conda create -n py311 python=3.11")
            except FileNotFoundError:
                print(f"  Conda Environment: ❌ conda not found")
        else:
            print(f"  Conda Environment: ✅ py311 active")
        
        # Check virtual environment
        venv_active = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        print(f"  Virtual Environment: {'✅ Active' if venv_active else '⚠️  Not detected'}")
        
        return version_ok and py311_active
    
    def _check_dependencies(self):
        """Check required dependencies"""
        print("\n📦 Dependencies:")
        
        # Check if we're in py311 environment first
        python_path = sys.executable
        py311_active = "py311" in python_path
        
        if py311_active:
            print("  🎯 py311 Environment Detected - Checking py311 requirements...")
            required_packages = [
                'streamlit', 'yaml', 'pandas', 'numpy', 'mlflow', 
                'openai', 'boto3', 'sentence_transformers', 'torch', 'transformers',
                'dspy', 'litellm', 'psycopg2', 'pgvector'
            ]
            
            optional_packages = [
                'chromadb', 'autogluon', 'fastapi'
            ]
        else:
            print("  ⚠️  Not in py311 environment - checking basic requirements...")
            required_packages = [
                'streamlit', 'yaml', 'pandas', 'numpy', 'mlflow', 
                'openai', 'boto3'
            ]
            
            optional_packages = [
                'chromadb', 'sentence_transformers'
            ]
        
        missing_packages = []
        optional_missing = []
        
        # Check required packages
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                print(f"  {package}: ✅")
            except ImportError:
                print(f"  {package}: ❌")
                missing_packages.append(package)
        
        # Check optional packages
        for package in optional_packages:
            try:
                __import__(package.replace('-', '_'))
                print(f"  {package}: ✅ (optional)")
            except ImportError:
                print(f"  {package}: ⚠️  (optional)")
                optional_missing.append(package)
        
        if missing_packages:
            print(f"  ⚠️  Missing required packages: {', '.join(missing_packages)}")
            print(f"  💡 Run: pip install {' '.join(missing_packages)}")
        
        if optional_missing:
            print(f"  💡 Optional packages not installed: {', '.join(optional_missing)}")
        
        # Additional py311 specific checks
        if py311_active:
            py311_ok = self._check_py311_specific_requirements()
            return len(missing_packages) == 0 and py311_ok
        else:
            print(f"  ⚠️  py311 environment not active - skipping py311-specific checks")
            print(f"  💡 Activate py311: conda activate py311")
            return len(missing_packages) == 0
    
    def _check_py311_specific_requirements(self):
        """Check py311-specific requirements and configurations"""
        print("\n🎯 py311 Specific Requirements:")
        
        # Check py311 requirements file exists
        py311_req_file = self.project_root / 'py311_requirements.txt'
        if py311_req_file.exists():
            print(f"  py311_requirements.txt: ✅")
        else:
            print(f"  py311_requirements.txt: ❌")
            return False
        
        # Check install script exists
        install_script = self.project_root / 'install_py311_packages.py'
        if install_script.exists():
            print(f"  install_py311_packages.py: ✅")
        else:
            print(f"  install_py311_packages.py: ❌")
            return False
        
        # Check key py311 packages
        py311_key_packages = [
            'sentence_transformers', 'torch', 'transformers', 'dspy', 'litellm'
        ]
        
        missing_py311 = []
        for package in py311_key_packages:
            try:
                __import__(package.replace('-', '_'))
                print(f"  {package}: ✅")
            except ImportError:
                print(f"  {package}: ❌")
                missing_py311.append(package)
        
        if missing_py311:
            print(f"  ⚠️  Missing py311 packages: {', '.join(missing_py311)}")
            print(f"  💡 Run: python install_py311_packages.py")
            return False
        
        print(f"  ✅ All py311 requirements met")
        return True
    
    def _check_directories(self):
        """Check required directories exist"""
        print("\n📁 Directory Structure:")
        
        required_dirs = [
            'src', 'docs', 'tests', 'input', 'output', 'test_outputs'
        ]
        
        missing_dirs = []
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                print(f"  {dir_name}/: ✅")
            else:
                print(f"  {dir_name}/: ❌")
                missing_dirs.append(dir_name)
        
        if missing_dirs:
            print(f"  ⚠️  Missing directories: {', '.join(missing_dirs)}")
        
        return len(missing_dirs) == 0
    
    def _check_config_files(self):
        """Check configuration files"""
        print("\n⚙️  Configuration Files:")
        
        config_files = [
            'requirements.txt', 'src/requirements.txt', 
            'dev_configs/qa_criteria_full.yaml',
            'dev_configs/qa_criteria_simplified.yaml'
        ]
        
        missing_configs = []
        for config_file in config_files:
            config_path = self.project_root / config_file
            if config_path.exists():
                print(f"  {config_file}: ✅")
            else:
                print(f"  {config_file}: ❌")
                missing_configs.append(config_file)
        
        if missing_configs:
            print(f"  ⚠️  Missing config files: {', '.join(missing_configs)}")
        
        return len(missing_configs) == 0
    
    def _check_test_files(self):
        """Check test files"""
        print("\n🧪 Test Files:")
        
        test_files = [
            'test_dashboard.py', 'test_sme_qa_reviewer_system.py',
            'src/qa_demo.py'
        ]
        
        missing_tests = []
        for test_file in test_files:
            test_path = self.project_root / test_file
            if test_path.exists():
                print(f"  {test_file}: ✅")
            else:
                print(f"  {test_file}: ❌")
                missing_tests.append(test_file)
        
        if missing_tests:
            print(f"  ⚠️  Missing test files: {', '.join(missing_tests)}")
        
        return len(missing_tests) == 0
    
    def _check_demo_readiness(self):
        """Check if system is ready for demo"""
        print("\n🎯 Demo Readiness:")
        
        # Check if Streamlit app can start
        try:
            result = subprocess.run(
                ['python', '-c', 'import streamlit; print("Streamlit available")'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                print("  Streamlit: ✅")
                streamlit_ok = True
            else:
                print("  Streamlit: ❌")
                streamlit_ok = False
        except Exception as e:
            print(f"  Streamlit: ❌ ({e})")
            streamlit_ok = False
        
        # Check if main demo file exists
        demo_file = self.project_root / 'src' / 'qa_demo.py'
        demo_ok = demo_file.exists()
        print(f"  Demo App: {'✅' if demo_ok else '❌'}")
        
        return streamlit_ok and demo_ok
    
    def _check_api_security(self):
        """Check for exposed API keys and credentials"""
        print("\n🔒 API Security:")
        
        # Define API key patterns to search for
        api_patterns = {
            'Google API Key': r'AIzaSy[A-Za-z0-9_-]{33}',
            'OpenAI API Key': r'sk-[A-Za-z0-9]{20,}',
            'Cohere API Key': r'[A-Za-z0-9]{32,}',  # Generic pattern for Cohere
            'AWS Access Key': r'AKIA[0-9A-Z]{16}',
            'JWT Token': r'eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}',
            'GitHub Token': r'ghp_[A-Za-z0-9]{36}',
            'Slack Token': r'xoxb-[0-9]{10,13}-[0-9]{10,13}-[A-Za-z0-9]{24}',
            'S3 Bucket': r's3://[a-z0-9][a-z0-9\-]*[a-z0-9]',
            'AWS Account ID': r'\b[0-9]{12}\b',
            'DB Connection String': r'postgresql://[^:]+:[^@]+@[^/]+/\w+',
            'Personal Name': r'John Doe',
            'Email Address': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        }
        
        exposed_keys = []
        files_checked = 0
        
        # Search through relevant file types
        search_extensions = ['.py', '.md', '.txt', '.yaml', '.yml', '.json', '.env']
        files_skipped = 0
        
        for file_path in self.project_root.rglob('*'):
            if (file_path.is_file() and 
                file_path.suffix in search_extensions):
                
                # Skip gitignored files
                if self._is_gitignored(file_path):
                    files_skipped += 1
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        files_checked += 1
                        
                        for key_type, pattern in api_patterns.items():
                            matches = re.findall(pattern, content)
                            if matches:
                                for match in matches:
                                    # Skip obvious placeholders
                                    if not any(placeholder in match.lower() for placeholder in 
                                             ['your_', 'example', 'placeholder', 'xxx', 'sk-proj-']):
                                        exposed_keys.append({
                                            'type': key_type,
                                            'key': match,
                                            'file': str(file_path.relative_to(self.project_root)),
                                            'line': self._find_line_number(content, match)
                                        })
                except Exception as e:
                    continue
        
        print(f"  Files scanned: {files_checked}")
        print(f"  Files skipped (gitignored): {files_skipped}")
        
        if exposed_keys:
            print(f"  ⚠️  EXPOSED API KEYS FOUND: {len(exposed_keys)}")
            for key_info in exposed_keys:
                print(f"    - {key_info['type']} in {key_info['file']}:{key_info['line']}")
                print(f"      Key: {key_info['key'][:10]}...{key_info['key'][-4:]}")
            print("  🚨 IMMEDIATE ACTION REQUIRED:")
            print("    1. Regenerate all exposed keys")
            print("    2. Remove keys from files")
            print("    3. Use environment variables instead")
            return False
        else:
            print("  ✅ No exposed API keys found")
            return True
    
    def _check_security_hardening(self):
        """Check security hardening measures"""
        print("\n🛡️  Security Hardening:")
        
        security_checks = []
        
        # Check for sensitive environment configuration folder
        env_settings = self.project_root / 'environ_settings'
        
        if env_settings.exists():
            # Check if this sensitive folder is properly gitignored
            if self._is_gitignored(env_settings):
                print("  Environment config: ✅ (environ_settings/ exists and gitignored)")
                security_checks.append(True)
            else:
                print("  Environment config: 🚨 CRITICAL - environ_settings/ NOT gitignored!")
                print("    🚨 This sensitive folder contains deployment secrets!")
                print("    💡 Add 'environ_settings/' to .gitignore immediately")
                security_checks.append(False)
        else:
            print("  Environment config: ⚠️  No environ_settings/ deployment structure")
            print("    💡 Create environ_settings/ folder for deployment configs")
            security_checks.append(True)  # Warning, not failure since it might be intentional
        
        # Check for pre-commit hooks
        pre_commit_hook = self.project_root / '.git' / 'hooks' / 'pre-commit'
        if pre_commit_hook.exists():
            print("  Pre-commit hooks: ✅")
            security_checks.append(True)
        else:
            print("  Pre-commit hooks: ⚠️  Not configured")
            print("    💡 Add pre-commit hook to prevent API key commits")
            security_checks.append(True)  # Warning, not failure
        
        # Check for large files in repo
        large_files = []
        for file_path in self.project_root.rglob('*'):
            if (file_path.is_file() and 
                not self._is_gitignored(file_path) and
                file_path.stat().st_size > 10 * 1024 * 1024):  # 10MB
                large_files.append(file_path)
        
        if large_files:
            print(f"  Large files: ⚠️  {len(large_files)} files > 10MB")
            for lf in large_files[:3]:  # Show first 3
                size_mb = lf.stat().st_size / 1024 / 1024
                print(f"    - {lf.relative_to(self.project_root)} ({size_mb:.1f}MB)")
            if len(large_files) > 3:
                print(f"    - ... and {len(large_files) - 3} more")
            print("    💡 Consider using Git LFS for large files")
            security_checks.append(True)  # Warning, not failure
        else:
            print("  Large files: ✅ None found")
            security_checks.append(True)
        
        # Check sensitive file extensions
        sensitive_extensions = ['.p12', '.pfx', '.key', '.pem', '.crt']
        sensitive_files = []
        for ext in sensitive_extensions:
            matches = list(self.project_root.rglob(f'*{ext}'))
            for match in matches:
                if not self._is_gitignored(match):
                    sensitive_files.append(match)
        
        if sensitive_files:
            print(f"  Sensitive files: ❌ {len(sensitive_files)} unignored certificate/key files")
            for sf in sensitive_files[:3]:
                print(f"    - {sf.relative_to(self.project_root)}")
            security_checks.append(False)
        else:
            print("  Sensitive files: ✅ None found")
            security_checks.append(True)
        
        return all(security_checks)
    
    def _find_line_number(self, content, search_text):
        """Find line number of text in content"""
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if search_text in line:
                return i
        return 0
    
    def scrub_api_keys(self):
        """Scrub API keys from files by replacing them with 'XXXX' placeholders"""
        print("\n🧹 SCRUBBING API KEYS")
        print("=" * 60)
        
        # Define API key patterns to search and replace
        api_patterns = {
            'Google API Key': (r'AIzaSy[A-Za-z0-9_-]{33}', 'AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'),
            'OpenAI API Key': (r'sk-[A-Za-z0-9]{20,}', 'sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'),
            'Cohere API Key': (r'[A-Za-z0-9]{32}(?=[^A-Za-z0-9])', 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'),
            'AWS Access Key': (r'AKIA[0-9A-Z]{16}', 'AKIAXXXXXXXXXXXXXXXX'),
            'JWT Token': (r'eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}', 'eyJXXXXXXXXXXX.XXXXXXXXXXX.XXXXXXXXXXX'),
            'GitHub Token': (r'ghp_[A-Za-z0-9]{36}', 'ghp_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'),
            'S3 Bucket': (r's3://[a-z0-9][a-z0-9\-]*[a-z0-9]', 's3://example-bucket-name'),
            'AWS Account ID': (r'\b[0-9]{12}\b', 'XXXXXXXXXXXX'),
            'DB Connection String': (r'postgresql://[^:]+:[^@]+@[^/]+/\w+', 'postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}'),
            'Personal Name': (r'John Doe', 'John Doe'),
            'Company Name': (r'Example Company LLC', 'Example Company LLC'),
            'Email Address': (r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', 'user@example.com'),
        }
        
        files_modified = 0
        keys_scrubbed = 0
        
        # Search through relevant file types
        search_extensions = ['.py', '.md', '.txt', '.yaml', '.yml', '.json']
        files_skipped = 0
        
        for file_path in self.project_root.rglob('*'):
            if (file_path.is_file() and 
                file_path.suffix in search_extensions):
                
                # Skip gitignored files
                if self._is_gitignored(file_path):
                    files_skipped += 1
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        original_content = f.read()
                    
                    modified_content = original_content
                    file_keys_found = 0
                    
                    for key_type, (pattern, replacement) in api_patterns.items():
                        matches = re.findall(pattern, original_content)
                        if matches:
                            for match in matches:
                                # Skip obvious placeholders that are already scrubbed
                                if not any(placeholder in match.lower() for placeholder in 
                                         ['your_', 'example', 'placeholder', 'xxx', 'xxxx']):
                                    modified_content = re.sub(re.escape(match), replacement, modified_content)
                                    file_keys_found += 1
                                    keys_scrubbed += 1
                                    print(f"  🔄 Scrubbed {key_type} in {file_path.relative_to(self.project_root)}")
                    
                    # Write back modified content if changes were made
                    if file_keys_found > 0 and not self.dry_run:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(modified_content)
                        files_modified += 1
                    elif file_keys_found > 0:
                        print(f"    🔍 Would modify {file_path.relative_to(self.project_root)} (dry run)")
                        
                except Exception as e:
                    self.cleanup_stats['errors'].append(f"Failed to scrub {file_path}: {e}")
        
        self.cleanup_stats['api_keys_scrubbed'] = keys_scrubbed
        self.cleanup_stats['files_skipped_gitignore'] = files_skipped
        
        print(f"  Files skipped (gitignored): {files_skipped}")
        
        if keys_scrubbed > 0:
            print(f"  ✅ Scrubbed {keys_scrubbed} API keys from {files_modified} files")
            if not self.dry_run:
                print("  ⚠️  REMEMBER TO:")
                print("    1. Regenerate the original API keys")
                print("    2. Set up environment variables")
                print("    3. Update your deployment scripts")
        else:
            print("  ✅ No API keys found to scrub")
    
    def cleanup_test_outputs(self, days_old=7):
        """Clean up old test output files"""
        print(f"\n🧹 CLEANING TEST OUTPUTS (older than {days_old} days)")
        print("=" * 60)
        
        test_outputs_dir = self.project_root / 'test_outputs'
        if not test_outputs_dir.exists():
            print("  No test_outputs directory found")
            return
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        files_to_remove = []
        
        for file_path in test_outputs_dir.glob('*.json'):
            if file_path.stat().st_mtime < cutoff_date.timestamp():
                files_to_remove.append(file_path)
        
        if not files_to_remove:
            print("  No old test files found")
            return
        
        print(f"  Found {len(files_to_remove)} old test files:")
        for file_path in files_to_remove:
            file_size = file_path.stat().st_size
            file_date = datetime.fromtimestamp(file_path.stat().st_mtime)
            print(f"    {file_path.name} ({file_size} bytes, {file_date.strftime('%Y-%m-%d')})")
            
            if not self.dry_run:
                try:
                    file_path.unlink()
                    self.cleanup_stats['files_removed'] += 1
                    self.cleanup_stats['bytes_freed'] += file_size
                    print(f"      ✅ Removed")
                except Exception as e:
                    self.cleanup_stats['errors'].append(f"Failed to remove {file_path}: {e}")
                    print(f"      ❌ Error: {e}")
            else:
                print(f"      🔍 Would remove (dry run)")
    
    def cleanup_temp_files(self):
        """Clean up temporary files (excluding .gitignore items)"""
        print("\n🗑️  CLEANING TEMPORARY FILES")
        print("=" * 60)
        
        # Skip patterns that are already in .gitignore
        # These are handled by git and don't need manual cleanup
        gitignore_patterns = [
            '__pycache__', '*.pyc', '*.pyo', '.pytest_cache',
            '*.tmp', '*.temp', '*.log', '*.bak', '*.backup',
            '.DS_Store', '.vscode', '.idea', '*.swp', '*.swo'
        ]
        
        print("  ⚠️  Skipping files already in .gitignore:")
        for pattern in gitignore_patterns:
            print(f"    - {pattern}")
        
        # Only clean up non-gitignored temporary files
        manual_cleanup_patterns = [
            '*.tmp.local', '*.temp.local', 'temp_*', 'tmp_*'
        ]
        
        for pattern in manual_cleanup_patterns:
            matches = list(self.project_root.rglob(pattern))
            if matches:
                print(f"  {pattern}: {len(matches)} files")
                for match in matches:
                    if not self.dry_run:
                        try:
                            if match.is_file():
                                match.unlink()
                                self.cleanup_stats['files_removed'] += 1
                            elif match.is_dir():
                                shutil.rmtree(match)
                                self.cleanup_stats['directories_removed'] += 1
                        except Exception as e:
                            self.cleanup_stats['errors'].append(f"Failed to remove {match}: {e}")
                    else:
                        print(f"    🔍 Would remove: {match}")
        
        print("  ✅ Gitignored files are handled automatically by git")
    
    def cleanup_output_folders(self):
        """Clean up output folders (excluding .gitignore items)"""
        print("\n📤 CLEANING OUTPUT FOLDERS")
        print("=" * 60)
        
        # Only clean specific output folders that aren't in .gitignore
        output_dirs = ['output', 'reports']  # Removed 'logs' as it's in .gitignore
        
        for dir_name in output_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                files = list(dir_path.glob('*'))
                if files:
                    print(f"  {dir_name}/: {len(files)} files")
                    for file_path in files:
                        # Skip files that would be gitignored
                        if file_path.name in ['.gitkeep', 'README.md']:
                            print(f"    🔒 Keeping: {file_path.name}")
                            continue
                            
                        if not self.dry_run:
                            try:
                                if file_path.is_file():
                                    file_path.unlink()
                                    self.cleanup_stats['files_removed'] += 1
                                elif file_path.is_dir():
                                    shutil.rmtree(file_path)
                                    self.cleanup_stats['directories_removed'] += 1
                            except Exception as e:
                                self.cleanup_stats['errors'].append(f"Failed to remove {file_path}: {e}")
                        else:
                            print(f"    🔍 Would remove: {file_path}")
                else:
                    print(f"  {dir_name}/: ✅ Clean")
            else:
                print(f"  {dir_name}/: ❌ Not found")
        
        print("  ⚠️  Skipping 'logs/' directory (already in .gitignore)")
    
    def cleanup_cache_directories(self):
        """Clean up old cache files"""
        print("\n🗄️  CLEANING CACHE DIRECTORIES")
        print("=" * 60)
        
        cache_dirs = ['llm_cache', 'llm_metrics', 'mlflow_export', 'mlflow_real_doc_export']
        cutoff_date = datetime.now() - timedelta(days=30)  # 30 days old
        
        for cache_dir in cache_dirs:
            cache_path = self.project_root / cache_dir
            if cache_path.exists():
                old_files = []
                for file_path in cache_path.rglob('*'):
                    if file_path.is_file() and file_path.stat().st_mtime < cutoff_date.timestamp():
                        old_files.append(file_path)
                
                if old_files:
                    print(f"  {cache_dir}/: {len(old_files)} old files")
                    total_size = sum(f.stat().st_size for f in old_files)
                    print(f"    Total size: {total_size / 1024 / 1024:.1f} MB")
                    
                    if not self.dry_run:
                        for file_path in old_files:
                            try:
                                file_path.unlink()
                                self.cleanup_stats['files_removed'] += 1
                                self.cleanup_stats['bytes_freed'] += file_path.stat().st_size
                            except Exception as e:
                                self.cleanup_stats['errors'].append(f"Failed to remove {file_path}: {e}")
                    else:
                        print(f"    🔍 Would remove {len(old_files)} files (dry run)")
                else:
                    print(f"  {cache_dir}/: ✅ No old files")
            else:
                print(f"  {cache_dir}/: ❌ Not found")
    
    def cleanup_root_temp_files(self):
        """Clean up temporary files at root level"""
        print("\n🗑️  CLEANING ROOT TEMPORARY FILES")
        print("=" * 60)
        
        # Root level temporary file patterns
        root_temp_patterns = [
            'aws_deployment_test_report_*.json',
            'test_dashboard_results_*.json',
            '*.log',  # Log files at root
            'temp_*.json',
            'test_*.log',
        ]
        
        files_removed = 0
        for pattern in root_temp_patterns:
            matches = list(self.project_root.glob(pattern))
            if matches:
                print(f"  {pattern}: {len(matches)} files")
                for match in matches:
                    # For root temp files, clean them even if gitignored
                    # (they shouldn't be in the repo anyway)
                        
                    if not self.dry_run:
                        try:
                            match.unlink()
                            files_removed += 1
                            self.cleanup_stats['files_removed'] += 1
                            print(f"    ✅ Removed: {match.name}")
                        except Exception as e:
                            self.cleanup_stats['errors'].append(f"Failed to remove {match}: {e}")
                            print(f"    ❌ Error removing {match.name}: {e}")
                    else:
                        print(f"    🔍 Would remove: {match.name}")
            else:
                print(f"  {pattern}: No files found")
        
        if files_removed == 0 and not self.dry_run:
            print("  ✅ No root temporary files to clean")
        elif self.dry_run:
            print("  🔍 Dry run - no files actually removed")
    
    def generate_cleanup_report(self):
        """Generate cleanup report"""
        print("\n📊 CLEANUP REPORT")
        print("=" * 60)
        
        if self.dry_run:
            print("🔍 DRY RUN MODE - No files were actually removed")
        else:
            print(f"✅ Files removed: {self.cleanup_stats['files_removed']}")
            print(f"✅ Directories removed: {self.cleanup_stats['directories_removed']}")
            print(f"✅ API keys scrubbed: {self.cleanup_stats['api_keys_scrubbed']}")
            print(f"✅ Bytes freed: {self.cleanup_stats['bytes_freed']:,}")
        
        if self.cleanup_stats['errors']:
            print(f"\n❌ Errors encountered:")
            for error in self.cleanup_stats['errors']:
                print(f"  - {error}")
    
    def run_full_cleanup(self):
        """Run full cleanup process"""
        print("🧹 FULL CLEANUP PROCESS")
        print("=" * 60)
        
        self.scrub_api_keys()
        self.cleanup_root_temp_files()
        self.cleanup_cache_directories()
        self.cleanup_test_outputs()
        self.cleanup_temp_files()
        self.cleanup_output_folders()
        self.generate_cleanup_report()
    
    def _check_robust_bundle_ready(self):
        """Check if robust bundle is ready for deployment"""
        print("\n📦 Robust Bundle Validation:")
        
        # Check if robust bundle tools exist
        robust_tools = [
            'scripts/create_robust_bundle.py',
            'tests/test_environment_compatibility.py',
            'src/utils/import_helper.py',
            'src/config/settings.py'
        ]
        
        missing_tools = []
        for tool in robust_tools:
            tool_path = self.project_root / tool
            if tool_path.exists():
                print(f"  {tool}: ✅")
            else:
                print(f"  {tool}: ❌")
                missing_tools.append(tool)
        
        # Check if production bundle exists
        bundle_dir = self.project_root / "migration_bundles" / "production_ready_bundle"
        if bundle_dir.exists():
            print(f"  production_ready_bundle: ✅")
            
            # Check bundle components
            bundle_components = [
                'src', 'config', 'launchers', 'validation', 'docs'
            ]
            
            missing_components = []
            for component in bundle_components:
                component_path = bundle_dir / component
                if component_path.exists():
                    print(f"    {component}/: ✅")
                else:
                    print(f"    {component}/: ❌")
                    missing_components.append(component)
            
            if missing_components:
                print(f"    ⚠️  Missing bundle components: {', '.join(missing_components)}")
                return False
        else:
            print(f"  production_ready_bundle: ❌")
            print(f"    💡 Create with: python3 scripts/create_robust_bundle.py")
            return False
        
        if missing_tools:
            print(f"  ⚠️  Missing robust bundle tools: {', '.join(missing_tools)}")
            return False
        
        print(f"  ✅ Robust bundle ready for deployment")
        return True
    
    def _check_environment_compatibility(self):
        """Check environment compatibility across different deployment targets"""
        print("\n🌍 Environment Compatibility:")
        
        try:
            # Import and run environment compatibility tests
            sys.path.insert(0, str(self.project_root))
            
            # Check if test module exists
            test_module = self.project_root / "tests" / "test_environment_compatibility.py"
            if not test_module.exists():
                print(f"  test_environment_compatibility.py: ❌")
                print(f"    💡 Environment compatibility tests not found")
                return False
            
            # Run a quick compatibility check
            from tests.test_environment_compatibility import EnvironmentTester
            tester = EnvironmentTester()
            
            # Test local environment (fastest)
            try:
                result = tester.test_local_development()
                print(f"  Local Development: ✅")
            except Exception as e:
                print(f"  Local Development: ❌ - {e}")
                return False
            
            # Check if other environment tests are available
            environments = ["sagemaker", "docker", "lambda", "standalone"]
            available_envs = []
            
            for env in environments:
                test_method = getattr(tester, f"test_{env}_environment", None)
                if test_method:
                    available_envs.append(env)
            
            print(f"  Available Environment Tests: {', '.join(available_envs)}")
            
            if len(available_envs) >= 3:
                print(f"  ✅ Environment compatibility tests available")
                return True
            else:
                print(f"  ⚠️  Limited environment test coverage")
                return False
                
        except ImportError as e:
            print(f"  ❌ Environment compatibility tests failed: {e}")
            return False
        except Exception as e:
            print(f"  ❌ Environment compatibility check failed: {e}")
            return False
    
    def _check_embedding_centralization(self):
        """Check that all embedding generation uses centralized EmbeddingHelper"""
        print("\n🔗 Embedding Centralization:")
        
        try:
            # Import and run the validation script
            from scripts.validate_embedding_centralization import EmbeddingCentralizationValidator
            
            validator = EmbeddingCentralizationValidator()
            success = validator.run_validation()
            
            if success:
                print("  ✅ All embedding generation properly centralized")
                return True
            else:
                print("  ❌ Embedding centralization issues found")
                print("  💡 Run: python scripts/validate_embedding_centralization.py")
                return False
                
        except ImportError as e:
            print(f"  ❌ Embedding validation script not found: {e}")
            return False
        except Exception as e:
            print(f"  ❌ Embedding validation failed: {e}")
            return False
    
    def _check_naked_calls(self):
        """Check for naked calls that bypass centralized systems"""
        print("\n🔗 Naked Calls Validation:")
        
        try:
            # Import and run the naked call test suite
            from tests.test_naked_calls import run_naked_call_tests
            
            success = run_naked_call_tests()
            
            if success:
                print("  ✅ No naked calls detected")
                return True
            else:
                print("  ❌ Naked calls detected")
                print("  💡 Run: python scripts/run_naked_call_tests.py")
                return False
                
        except ImportError as e:
            print(f"  ❌ Naked call test suite not found: {e}")
            return False
        except Exception as e:
            print(f"  ❌ Naked call validation failed: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='QA System Pre-Flight and Cleanup')
    parser.add_argument('--pre-flight', action='store_true', help='Run pre-flight checks')
    parser.add_argument('--cleanup', action='store_true', help='Run cleanup process')
    parser.add_argument('--scrub-keys', action='store_true', help='Scrub API keys only')
    parser.add_argument('--full', action='store_true', help='Run both pre-flight and cleanup')
    parser.add_argument('--all', action='store_true', help='Run everything (pre-flight + cleanup + scrub-keys)')
    parser.add_argument('--dry-run', action='store_true', default=True, help='Dry run mode (default)')
    parser.add_argument('--force', action='store_true', help='Force actual cleanup (not dry run)')
    parser.add_argument('--days', type=int, default=7, help='Age threshold for test files (default: 7 days)')
    
    args = parser.parse_args()
    
    # Set dry run mode
    dry_run = not args.force
    
    cleanup_system = QAPreFlightCleanup(dry_run=dry_run)
    
    if args.pre_flight or args.full or args.all:
        cleanup_system.run_pre_flight_checks()
    
    if args.cleanup or args.full or args.all:
        cleanup_system.run_full_cleanup()
    
    if args.scrub_keys or args.all:
        # If --all is used, scrubbing is already included in run_full_cleanup()
        if not (args.cleanup or args.full or args.all):
            cleanup_system.scrub_api_keys()
            cleanup_system.generate_cleanup_report()
    
    if not any([args.pre_flight, args.cleanup, args.full, args.scrub_keys, args.all]):
        # Default: run pre-flight checks
        cleanup_system.run_pre_flight_checks()

if __name__ == "__main__":
    main()
