#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate Embedding Centralization

This script scans the codebase to ensure all embedding generation goes through
the centralized EmbeddingHelper instead of direct SentenceTransformer calls.
"""

import os
import re
import ast
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import fnmatch

class EmbeddingCentralizationValidator:
    """Validates that all embedding generation uses centralized EmbeddingHelper"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.gitignore_patterns = self._load_gitignore_patterns()
        self.issues = []
        self.allowed_files = {
            # Files that are allowed to import SentenceTransformer for testing/validation
            "scripts/validate_embedding_centralization.py",
            "src/backend/core/embedding_helper.py",  # The centralized helper itself
            "test_enhanced_embeddings.py",
            "scripts/setup_before_preflight.py",
            "scripts/pre_flight_cleanup.py",
        }
        
    def scan_for_naked_sentence_transformers(self) -> List[Dict]:
        """Scan for direct SentenceTransformer usage"""
        print("🔍 Scanning for naked SentenceTransformer calls...")
        
        issues = []
        
        # Find all Python files
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            relative_path = str(file_path.relative_to(self.project_root))
            
            # Skip gitignored files
            if self._is_gitignored(relative_path):
                continue
            
            # Skip allowed files
            if relative_path in self.allowed_files:
                continue
                
            # Skip test files and scripts for now (we'll handle them separately)
            if "test_" in relative_path or "scripts/" in relative_path:
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for direct SentenceTransformer imports
                if self._has_direct_sentence_transformer_import(content):
                    issues.append({
                        'file': relative_path,
                        'line': self._find_import_line(content, 'SentenceTransformer'),
                        'type': 'direct_import',
                        'message': 'Direct SentenceTransformer import detected'
                    })
                
                # Check for direct SentenceTransformer instantiation
                if self._has_direct_sentence_transformer_instantiation(content):
                    issues.append({
                        'file': relative_path,
                        'line': self._find_instantiation_line(content),
                        'type': 'direct_instantiation',
                        'message': 'Direct SentenceTransformer instantiation detected'
                    })
                
                # Check for direct .encode() calls
                if self._has_direct_encode_calls(content):
                    issues.append({
                        'file': relative_path,
                        'line': self._find_encode_line(content),
                        'type': 'direct_encode',
                        'message': 'Direct .encode() call detected - should use EmbeddingHelper'
                    })
                    
            except Exception as e:
                print(f"⚠️  Error scanning {relative_path}: {e}")
        
        return issues
    
    def _has_direct_sentence_transformer_import(self, content: str) -> bool:
        """Check if file has direct SentenceTransformer import"""
        patterns = [
            r'from sentence_transformers import SentenceTransformer',
            r'import sentence_transformers',
            r'from sentence_transformers\.SentenceTransformer import'
        ]
        
        for pattern in patterns:
            if re.search(pattern, content):
                return True
        return False
    
    def _has_direct_sentence_transformer_instantiation(self, content: str) -> bool:
        """Check if file has direct SentenceTransformer instantiation"""
        patterns = [
            r'SentenceTransformer\(',
            r'= SentenceTransformer\(',
            r'embedding_model = SentenceTransformer\(',
        ]
        
        for pattern in patterns:
            if re.search(pattern, content):
                return True
        return False
    
    def _has_direct_encode_calls(self, content: str) -> bool:
        """Check if file has direct .encode() calls related to embeddings"""
        # Look for embedding-related .encode() calls specifically
        embedding_encode_patterns = [
            r'model\.encode\(',
            r'embedding_model\.encode\(',
            r'\.encode\([^)]*\)',  # .encode() calls that are likely embedding-related
        ]
        
        # Exclude legitimate string encoding operations
        exclude_patterns = [
            r'\.encode\([\'"]utf-8[\'"]\)',  # UTF-8 encoding
            r'\.encode\(\)',  # Default encoding (usually for hashing)
            r'hashlib\.',  # Hash operations
            r'unicodedata\.',  # Unicode normalization
        ]
        
        # Check for embedding-related patterns
        for pattern in embedding_encode_patterns:
            if re.search(pattern, content):
                # Check if it's excluded
                is_excluded = False
                for exclude_pattern in exclude_patterns:
                    if re.search(exclude_pattern, content):
                        is_excluded = True
                        break
                
                if not is_excluded:
                    return True
        
        return False
    
    def _find_import_line(self, content: str, import_name: str) -> int:
        """Find the line number of an import"""
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if import_name in line and ('import' in line or 'from' in line):
                return i
        return 0
    
    def _find_instantiation_line(self, content: str) -> int:
        """Find the line number of SentenceTransformer instantiation"""
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if 'SentenceTransformer(' in line:
                return i
        return 0
    
    def _find_encode_line(self, content: str) -> int:
        """Find the line number of embedding-related .encode() call"""
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if '.encode(' in line:
                # Skip legitimate string encoding operations
                if any(exclude in line for exclude in ['hashlib', 'unicodedata', 'utf-8', 'encode()']):
                    continue
                # Look for embedding-related patterns
                if any(pattern in line for pattern in ['model.encode', 'embedding_model.encode']):
                    return i
        return 0
    
    def validate_embedding_helper_usage(self) -> List[Dict]:
        """Validate that EmbeddingHelper is being used correctly"""
        print("🔍 Validating EmbeddingHelper usage...")
        
        issues = []
        
        # Find all Python files
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            relative_path = str(file_path.relative_to(self.project_root))
            
            # Skip gitignored files
            if self._is_gitignored(relative_path):
                continue
            
            # Skip the helper itself and test files
            if relative_path in ["src/backend/core/embedding_helper.py"]:
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check if file uses embeddings but doesn't import EmbeddingHelper
                if self._uses_embeddings(content) and not self._imports_embedding_helper(content):
                    issues.append({
                        'file': relative_path,
                        'line': 0,
                        'type': 'missing_embedding_helper',
                        'message': 'Uses embeddings but doesn\'t import EmbeddingHelper'
                    })
                    
            except Exception as e:
                print(f"⚠️  Error validating {relative_path}: {e}")
        
        return issues
    
    def _uses_embeddings(self, content: str) -> bool:
        """Check if file uses embedding-related functionality"""
        embedding_keywords = [
            'embedding', 'embeddings', 'vector', 'similarity', 'cosine',
            'sentence_transformers', 'SentenceTransformer'
        ]
        
        return any(keyword in content.lower() for keyword in embedding_keywords)
    
    def _imports_embedding_helper(self, content: str) -> bool:
        """Check if file imports EmbeddingHelper"""
        patterns = [
            r'from.*embedding_helper import EmbeddingHelper',
            r'import.*embedding_helper',
            r'from.*core\.embedding_helper import'
        ]
        
        for pattern in patterns:
            if re.search(pattern, content):
                return True
        return False
    
    def _load_gitignore_patterns(self) -> List[str]:
        """Load patterns from .gitignore file"""
        gitignore_path = self.project_root / '.gitignore'
        patterns = []
        
        if gitignore_path.exists():
            try:
                with open(gitignore_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            # Convert gitignore patterns to fnmatch patterns
                            pattern = line
                            if pattern.startswith('/'):
                                pattern = pattern[1:]  # Remove leading slash
                            if pattern.endswith('/'):
                                pattern = pattern[:-1] + '/*'  # Convert directory to glob
                            patterns.append(pattern)
            except Exception as e:
                print(f"⚠️  Warning: Could not read .gitignore: {e}")
        
        return patterns
    
    def _is_gitignored(self, file_path: str) -> bool:
        """Check if file matches any .gitignore pattern"""
        for pattern in self.gitignore_patterns:
            if fnmatch.fnmatch(file_path, pattern) or fnmatch.fnmatch(file_path, f"*/{pattern}"):
                return True
        return False
    
    def generate_fix_suggestions(self, issues: List[Dict]) -> str:
        """Generate suggestions for fixing the issues"""
        suggestions = []
        
        for issue in issues:
            file_path = issue['file']
            issue_type = issue['type']
            
            if issue_type == 'direct_import':
                suggestions.append(f"""
📁 {file_path}:{issue['line']}
❌ {issue['message']}
💡 Fix: Replace with EmbeddingHelper import
   from backend.core.embedding_helper import EmbeddingHelper
""")
            
            elif issue_type == 'direct_instantiation':
                suggestions.append(f"""
📁 {file_path}:{issue['line']}
❌ {issue['message']}
💡 Fix: Use EmbeddingHelper instead
   embedding_helper = EmbeddingHelper(target_dimensions=1024)
   embedding_model = embedding_helper.embedding_model
""")
            
            elif issue_type == 'direct_encode':
                suggestions.append(f"""
📁 {file_path}:{issue['line']}
❌ {issue['message']}
💡 Fix: Use EmbeddingHelper.generate_embedding()
   embedding, metadata = embedding_helper.generate_embedding(text, content_id)
""")
            
            elif issue_type == 'missing_embedding_helper':
                suggestions.append(f"""
📁 {file_path}
❌ {issue['message']}
💡 Fix: Import and use EmbeddingHelper
   from backend.core.embedding_helper import EmbeddingHelper
   embedding_helper = EmbeddingHelper(target_dimensions=1024)
""")
        
        return '\n'.join(suggestions)
    
    def run_validation(self) -> bool:
        """Run the complete validation"""
        print("🚀 Embedding Centralization Validation")
        print("=" * 60)
        
        # Scan for naked SentenceTransformer calls
        naked_issues = self.scan_for_naked_sentence_transformers()
        
        # Validate EmbeddingHelper usage
        helper_issues = self.validate_embedding_helper_usage()
        
        # Combine all issues
        all_issues = naked_issues + helper_issues
        
        # Report results
        print(f"\n📊 Validation Results:")
        print(f"   Files scanned: {len(list(self.project_root.rglob('*.py')))}")
        print(f"   Issues found: {len(all_issues)}")
        
        if all_issues:
            print(f"\n❌ Issues Found:")
            for issue in all_issues:
                print(f"   {issue['file']}:{issue['line']} - {issue['message']}")
            
            print(f"\n🔧 Fix Suggestions:")
            print(self.generate_fix_suggestions(all_issues))
            
            return False
        else:
            print(f"\n✅ No issues found! All embedding generation is properly centralized.")
            return True

def main():
    """Main validation function"""
    validator = EmbeddingCentralizationValidator()
    success = validator.run_validation()
    
    if success:
        print("\n🎉 Validation passed! Embedding generation is properly centralized.")
        sys.exit(0)
    else:
        print("\n⚠️  Validation failed! Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
