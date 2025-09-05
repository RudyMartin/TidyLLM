#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Centralization Validator

This script validates that all external service calls go through centralized systems:
- Database connections through centralized connection manager
- LLM calls through unified LLM gateway
- MLflow tracking through centralized configuration
- AWS services through centralized AWS manager
- Embeddings through EmbeddingHelper
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import fnmatch

class CentralizationValidator:
    """Validates that all external calls go through centralized systems"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.gitignore_patterns = self._load_gitignore_patterns()
        self.issues = []
        
        # Define centralized systems and their patterns
        self.centralized_systems = {
            'database': {
                'name': 'Database Connection Manager',
                'allowed_patterns': [
                    r'from.*database.*connection',
                    r'from.*db.*manager',
                    r'DatabaseConnectionManager',
                    r'get_database_connection',
                ],
                'naked_patterns': [
                    r'psycopg2\.connect\(',
                    r'psycopg2\.connect\(',
                    r'psycopg2\.connect\s*\(',
                    r'mysql\.connector\.connect',
                ]
            },
            'llm': {
                'name': 'Unified LLM Gateway',
                'allowed_patterns': [
                    r'from.*llm.*gateway',
                    r'from.*unified_llm_gateway',
                    r'UnifiedLLMGateway',
                    r'LLMGateway',
                ],
                'naked_patterns': [
                    r'openai\.ChatCompletion\.create\(',
                    r'openai\.Completion\.create\(',
                    r'anthropic\.Anthropic\(',
                    r'litellm\.completion\(',
                    r'requests\.post.*api\.openai\.com',
                    r'requests\.post.*api\.anthropic\.com',
                ]
            },
            'mlflow': {
                'name': 'Centralized MLflow Config',
                'allowed_patterns': [
                    r'from.*mlflow_config',
                    r'from.*core\.mlflow_config',
                    r'MLflowConfig',
                    r'get_mlflow_config',
                    r'setup_mlflow_environment',
                    r'self\.mlflow_config\.',
                ],
                'naked_patterns': [
                    r'mlflow\.set_tracking_uri\(',
                    r'mlflow\.set_experiment\(',
                    r'mlflow\.start_run\(',
                    r'mlflow\.log_',
                ]
            },
            'aws': {
                'name': 'Centralized AWS Manager',
                'allowed_patterns': [
                    r'from.*aws.*manager',
                    r'from.*core\.aws',
                    r'AWSManager',
                    r'get_aws_client',
                ],
                'naked_patterns': [
                    r'boto3\.client\(',
                    r'boto3\.resource\(',
                    r'aws_access_key_id',
                    r'aws_secret_access_key',
                ]
            },
            'embeddings': {
                'name': 'EmbeddingHelper',
                'allowed_patterns': [
                    r'from.*embedding_helper',
                    r'from.*core\.embedding_helper',
                    r'EmbeddingHelper',
                ],
                'naked_patterns': [
                    r'from sentence_transformers import',
                    r'SentenceTransformer\(',
                    r'\.encode\(',
                ]
            }
        }
        
        # Files that are allowed to have naked calls (testing, setup, etc.)
        self.allowed_files = {
            'scripts/validate_centralization.py',
            'scripts/validate_embedding_centralization.py',
            'scripts/setup_*.py',
            'scripts/test_*.py',
            'test_*.py',
            'tests/*.py',
            'tests/**/*.py',
            'notebooks/*.py',
            'src/backend/core/embedding_helper.py',
            'src/backend/core/mlflow_config.py',
            'src/backend/core/aws_llm_manager.py',
            'src/backend/llm/unified_llm_gateway.py',
            'src/backend/core/database_connection_manager.py',
            'src/backend/mcp/orchestrators/rag_qa_orchestrator.py',
        }
    
    def validate_centralization(self) -> Dict[str, List[Dict]]:
        """Validate all centralized systems"""
        print("🔍 Comprehensive Centralization Validation")
        print("=" * 60)
        
        all_issues = {}
        
        for system_name, system_config in self.centralized_systems.items():
            print(f"\n🔗 Validating {system_config['name']}...")
            issues = self._validate_system(system_name, system_config)
            if issues:
                all_issues[system_name] = issues
                print(f"   ❌ Found {len(issues)} issues")
            else:
                print(f"   ✅ No issues found")
        
        return all_issues
    
    def _validate_system(self, system_name: str, system_config: Dict) -> List[Dict]:
        """Validate a specific centralized system"""
        issues = []
        
        # Find all Python files
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            relative_path = str(file_path.relative_to(self.project_root))
            
            # Skip gitignored files
            if self._is_gitignored(relative_path):
                continue
            
            # Skip allowed files
            if self._is_allowed_file(relative_path):
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for naked calls
                for pattern in system_config['naked_patterns']:
                    if re.search(pattern, content):
                        line_num = self._find_pattern_line(content, pattern)
                        issues.append({
                            'file': relative_path,
                            'line': line_num,
                            'pattern': pattern,
                            'type': 'naked_call',
                            'message': f'Direct {system_name} call detected'
                        })
                
                # Check if file uses the system but doesn't import centralized version
                if self._uses_system(content, system_config['naked_patterns']):
                    if not self._imports_centralized_system(content, system_config['allowed_patterns']):
                        issues.append({
                            'file': relative_path,
                            'line': 0,
                            'pattern': '',
                            'type': 'missing_centralization',
                            'message': f'Uses {system_name} but doesn\'t import centralized {system_config["name"]}'
                        })
                        
            except Exception as e:
                print(f"⚠️  Error scanning {relative_path}: {e}")
        
        return issues
    
    def _is_allowed_file(self, file_path: str) -> bool:
        """Check if file is allowed to have naked calls"""
        for pattern in self.allowed_files:
            if fnmatch.fnmatch(file_path, pattern):
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
    
    def _uses_system(self, content: str, patterns: List[str]) -> bool:
        """Check if content uses the system"""
        return any(re.search(pattern, content) for pattern in patterns)
    
    def _imports_centralized_system(self, content: str, patterns: List[str]) -> bool:
        """Check if content imports centralized system"""
        return any(re.search(pattern, content) for pattern in patterns)
    
    def _find_pattern_line(self, content: str, pattern: str) -> int:
        """Find the line number of a pattern"""
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if re.search(pattern, line):
                return i
        return 0
    
    def generate_fix_suggestions(self, all_issues: Dict[str, List[Dict]]) -> str:
        """Generate suggestions for fixing the issues"""
        suggestions = []
        
        for system_name, issues in all_issues.items():
            system_config = self.centralized_systems[system_name]
            suggestions.append(f"\n🔧 {system_config['name']} Issues:")
            
            for issue in issues:
                file_path = issue['file']
                line_num = issue['line']
                issue_type = issue['type']
                
                if issue_type == 'naked_call':
                    suggestions.append(f"""
📁 {file_path}:{line_num}
❌ {issue['message']}
💡 Fix: Use centralized {system_config['name']}
   Example: from backend.core.{system_name}_manager import {system_config['name']}
""")
                
                elif issue_type == 'missing_centralization':
                    suggestions.append(f"""
📁 {file_path}
❌ {issue['message']}
💡 Fix: Import and use centralized {system_config['name']}
   Example: from backend.core.{system_name}_manager import {system_config['name']}
""")
        
        return '\n'.join(suggestions)
    
    def run_validation(self) -> bool:
        """Run the complete validation"""
        all_issues = self.validate_centralization()
        
        # Report results
        total_issues = sum(len(issues) for issues in all_issues.values())
        print(f"\n📊 Validation Results:")
        print(f"   Files scanned: {len(list(self.project_root.rglob('*.py')))}")
        print(f"   Systems checked: {len(self.centralized_systems)}")
        print(f"   Total issues found: {total_issues}")
        
        if all_issues:
            print(f"\n❌ Issues Found:")
            for system_name, issues in all_issues.items():
                system_config = self.centralized_systems[system_name]
                print(f"   {system_config['name']}: {len(issues)} issues")
            
            print(f"\n🔧 Fix Suggestions:")
            print(self.generate_fix_suggestions(all_issues))
            
            return False
        else:
            print(f"\n✅ No issues found! All external calls are properly centralized.")
            return True

def main():
    """Main validation function"""
    validator = CentralizationValidator()
    success = validator.run_validation()
    
    if success:
        print("\n🎉 Validation passed! All external calls are properly centralized.")
        sys.exit(0)
    else:
        print("\n⚠️  Validation failed! Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
