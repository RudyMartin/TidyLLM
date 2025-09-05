#!/usr/bin/env python3
"""
TidyLLM Internal Domain RAG Builder
==================================

Creates hierarchical domainRAG for TidyLLM's own documentation to resolve
code conflicts and documentation inconsistencies.

NAMING CONVENTION (to avoid confusion):
- external_domain_rag = Model validation compliance docs (knowledge_base/)
- internal_domain_rag = TidyLLM's own architecture docs (docs/ + git-tracked files)
- tidyllm_self_rag = This system (self-referential)

HIERARCHY for TidyLLM Internal Docs:
1. CRITICAL (precedence 1.0) - CRITICAL_DESIGN_DECISIONS.md, IMPORTANT-CONSTRAINTS
2. ARCHITECTURE (precedence 0.9) - Integration roadmaps, system architecture
3. CURRENT (precedence 0.8) - Latest date folder docs (2025-09-05)
4. HISTORICAL (precedence 0.7) - Earlier date folders
5. EXAMPLES (precedence 0.6) - Example code and templates
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import subprocess

class TidyLLMInternalRAG:
    """Build hierarchical domainRAG for TidyLLM's own documentation"""
    
    def __init__(self):
        self.base_path = Path(".")
        self.output_dir = Path("tidyllm_internal_domain_rag")
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        print("=" * 80)
        print("TIDYLLM INTERNAL DOMAIN RAG BUILDER")
        print("Building self-referential domainRAG for conflict resolution")
        print("=" * 80)
    
    def build_internal_domain_rag(self) -> Dict[str, Any]:
        """Build hierarchical domainRAG from TidyLLM's own docs"""
        
        # Step 1: Get only git-tracked files (not ignored)
        tracked_files = self._get_git_tracked_docs()
        
        # Step 2: Organize by hierarchy
        hierarchy = self._organize_by_hierarchy(tracked_files)
        
        # Step 3: Build manifest
        manifest = self._create_internal_manifest(hierarchy)
        
        # Step 4: Generate conflict detection queries
        conflict_queries = self._generate_internal_conflict_queries()
        
        # Step 5: Create demo system
        self._create_internal_demo(manifest, conflict_queries)
        
        print(f"\n[SUCCESS] TidyLLM Internal Domain RAG created")
        print(f"Output: {self.output_dir}/")
        print(f"Total documents: {sum(len(level['files']) for level in hierarchy.values())}")
        
        return {
            'system_type': 'TidyLLM Internal Domain RAG',
            'purpose': 'Self-referential conflict resolution',
            'hierarchy': hierarchy,
            'manifest_path': str(self.output_dir / 'manifest.json'),
            'demo_path': str(self.output_dir / 'demo.py')
        }
    
    def _get_git_tracked_docs(self) -> List[str]:
        """Get only git-tracked documentation files"""
        
        try:
            # Get all tracked markdown and text files
            result = subprocess.run(
                ['git', 'ls-files', '*.md', '*.txt', '*.rst'],
                capture_output=True, text=True, check=True
            )
            
            files = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            # Filter for actual documentation (not generated reports)
            doc_files = []
            excluded_patterns = [
                'sop_conflict_reports/',
                'domain_reports/', 
                'cross_domain_comparisons/',
                'boss_demo_evidence/',
                '.github/',
                'archive/'
            ]
            
            for file in files:
                if file and not any(pattern in file for pattern in excluded_patterns):
                    if Path(file).exists():  # Make sure file actually exists
                        doc_files.append(file)
            
            print(f"[TRACKED] Found {len(doc_files)} git-tracked documentation files")
            return doc_files
            
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Git command failed: {e}")
            return []
    
    def _organize_by_hierarchy(self, files: List[str]) -> Dict[str, Any]:
        """Organize files by hierarchical precedence"""
        
        hierarchy = {
            'critical': {
                'precedence': 1.0,
                'level': 'CRITICAL_DECISIONS',
                'files': [],
                'description': 'Critical design decisions and constraints'
            },
            'architecture': {
                'precedence': 0.9,
                'level': 'ARCHITECTURE',
                'files': [],
                'description': 'System architecture and integration'
            },
            'current': {
                'precedence': 0.8,
                'level': 'CURRENT',
                'files': [],
                'description': 'Latest documentation (2025-09-05)'
            },
            'recent': {
                'precedence': 0.7,
                'level': 'RECENT',
                'files': [],
                'description': 'Recent documentation (2025-09-04, 2025-09-03)'
            },
            'historical': {
                'precedence': 0.6,
                'level': 'HISTORICAL', 
                'files': [],
                'description': 'Historical documentation (2025-09-01)'
            },
            'examples': {
                'precedence': 0.5,
                'level': 'EXAMPLES',
                'files': [],
                'description': 'Examples and templates'
            }
        }
        
        for file in files:
            file_path = Path(file)
            file_name = file_path.name.lower()
            file_dir = str(file_path.parent)
            
            # Categorize by content and location
            if any(keyword in file_name for keyword in ['critical', 'constraint', 'important']):
                hierarchy['critical']['files'].append(file)
            elif any(keyword in file_name for keyword in ['architecture', 'integration', 'roadmap', 'system']):
                hierarchy['architecture']['files'].append(file)
            elif '2025-09-05' in file_dir:
                hierarchy['current']['files'].append(file)
            elif any(date in file_dir for date in ['2025-09-04', '2025-09-03']):
                hierarchy['recent']['files'].append(file)
            elif '2025-09-01' in file_dir:
                hierarchy['historical']['files'].append(file)
            elif 'example' in file_dir.lower() or 'template' in file_name:
                hierarchy['examples']['files'].append(file)
            else:
                # Default to architecture level for root docs
                hierarchy['architecture']['files'].append(file)
        
        # Print hierarchy summary
        print(f"\n[HIERARCHY] TidyLLM Internal Documentation:")
        for level_name, level_data in hierarchy.items():
            count = len(level_data['files'])
            if count > 0:
                print(f"  {level_data['level']:<15} ({level_data['precedence']:<3}): {count:>3} files")
        
        return hierarchy
    
    def _create_internal_manifest(self, hierarchy: Dict[str, Any]) -> Dict[str, Any]:
        """Create manifest for internal domain RAG"""
        
        manifest = {
            'system_type': 'TidyLLM Internal Domain RAG',
            'purpose': 'Self-referential conflict resolution for TidyLLM codebase',
            'created_at': datetime.now().isoformat(),
            'total_documents': sum(len(level['files']) for level in hierarchy.values()),
            'hierarchy': hierarchy,
            'conflict_resolution': {
                'rule': 'Higher precedence level wins in conflicts',
                'precedence_order': ['CRITICAL_DECISIONS', 'ARCHITECTURE', 'CURRENT', 'RECENT', 'HISTORICAL', 'EXAMPLES']
            },
            'naming_convention': {
                'external_domain_rag': 'Model validation compliance (knowledge_base/)',
                'internal_domain_rag': 'TidyLLM self-documentation (docs/)',
                'tidyllm_self_rag': 'This system (conflict resolution)'
            }
        }
        
        # Save manifest
        manifest_path = self.output_dir / 'manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
        
        print(f"[SAVED] Internal manifest: {manifest_path}")
        return manifest
    
    def _generate_internal_conflict_queries(self) -> List[str]:
        """Generate conflict detection queries for TidyLLM internal docs"""
        
        queries = [
            # Architecture conflicts
            "What is the official session management pattern?",
            "Should we use UnifiedSessionManager or Gateway pattern?",
            "Which embedding system is primary: tidyllm-sentence or tidyllm-vectorqa?",
            
            # Integration conflicts  
            "How should MLflow be integrated with TidyLLM?",
            "What is the approved workflow system: RAG2DAG or HeirOS?",
            "Which database should be used: PostgreSQL or SQLite?",
            
            # Development conflicts
            "What are the current constraints for this codebase?",
            "How should AWS credentials be managed?",
            "What is the proper drop zones architecture?",
            
            # Demo and deployment conflicts
            "Which demos are currently functional?",
            "What is the deployment strategy?",
            "How should dependencies be managed?",
            
            # Documentation conflicts
            "What is the current architecture documentation standard?",
            "How should new features be documented?",
            "What are the deprecated patterns to avoid?"
        ]
        
        return queries
    
    def _create_internal_demo(self, manifest: Dict[str, Any], queries: List[str]) -> None:
        """Create demo system for internal domain RAG"""
        
        demo_script = f'''#!/usr/bin/env python3
"""
TidyLLM Internal Domain RAG Demo
===============================
Self-referential conflict resolution for TidyLLM documentation.

Generated: {datetime.now().isoformat()}
Total Documents: {manifest['total_documents']}
"""

from pathlib import Path
import json

class TidyLLMInternalRAG:
    def __init__(self):
        print("TidyLLM Internal Domain RAG System Initialized")
        print("Purpose: Resolve conflicts in TidyLLM's own documentation")
        
        # Load hierarchy
        with open("manifest.json") as f:
            self.manifest = json.load(f)
        
        hierarchy = self.manifest['hierarchy']
        for level_name, level_data in hierarchy.items():
            count = len(level_data['files'])
            if count > 0:
                print(f"{{level_data['level']}}: {{count}} docs")
    
    def query(self, question):
        """Query internal documentation with hierarchical precedence"""
        print(f"\\nQuery: {{question}}")
        
        # Simple relevance matching (would use real embeddings)
        results = []
        
        # Check each hierarchy level
        hierarchy = self.manifest['hierarchy']
        for level_name, level_data in hierarchy.items():
            for doc_path in level_data['files']:
                if self._is_relevant(question, doc_path):
                    results.append({{
                        'file': Path(doc_path).name,
                        'level': level_data['level'],
                        'precedence': level_data['precedence'],
                        'path': doc_path
                    }})
        
        # Sort by precedence (highest first)
        results.sort(key=lambda x: x['precedence'], reverse=True)
        
        print(f"Results: {{len(results)}} documents found")
        for i, result in enumerate(results[:3], 1):
            print(f"  {{i}}. [{{result['level']}}] {{result['file']}}")
        
        return results[:5]
    
    def _is_relevant(self, question, doc_path):
        """Check if document is relevant to question"""
        question_lower = question.lower()
        path_lower = doc_path.lower()
        
        # Simple keyword matching
        keywords = question_lower.split()
        return any(keyword in path_lower for keyword in keywords if len(keyword) > 3)

def main():
    """Demo the internal RAG system"""
    
    print("TIDYLLM INTERNAL DOMAIN RAG DEMO")
    print("="*50)
    
    rag = TidyLLMInternalRAG()
    
    # Test queries for conflict resolution
    test_queries = {queries}
    
    for query in test_queries[:5]:  # Test first 5 queries
        results = rag.query(query)
        print()
    
    print("DEMO COMPLETE")
    print("="*50)
    print("Hierarchy: CRITICAL > ARCHITECTURE > CURRENT > RECENT > HISTORICAL > EXAMPLES")

if __name__ == "__main__":
    main()
'''
        
        demo_path = self.output_dir / 'demo.py'
        with open(demo_path, 'w') as f:
            f.write(demo_script)
        
        print(f"[SAVED] Internal demo: {demo_path}")
        
        # Create README
        readme_content = f"""# TidyLLM Internal Domain RAG

Self-referential domain RAG for resolving conflicts in TidyLLM's own documentation.

## Purpose
- Resolve conflicts between different documentation sources
- Establish precedence hierarchy for architectural decisions
- Provide authoritative answers about TidyLLM implementation

## Hierarchy (Precedence Order)
1. **CRITICAL_DECISIONS** (1.0) - Critical design decisions and constraints
2. **ARCHITECTURE** (0.9) - System architecture and integration docs
3. **CURRENT** (0.8) - Latest documentation (2025-09-05)
4. **RECENT** (0.7) - Recent documentation (2025-09-04, 2025-09-03)
5. **HISTORICAL** (0.6) - Historical documentation (2025-09-01)
6. **EXAMPLES** (0.5) - Examples and templates

## Usage
```bash
python demo.py
```

## Naming Convention
- **external_domain_rag**: Model validation compliance (knowledge_base/)
- **internal_domain_rag**: TidyLLM self-documentation (docs/)
- **tidyllm_self_rag**: This system (conflict resolution)

Generated: {datetime.now().isoformat()}
Total Documents: {manifest['total_documents']}
"""
        
        readme_path = self.output_dir / 'README.md'
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        print(f"[SAVED] Internal README: {readme_path}")

def main():
    """Build TidyLLM internal domain RAG"""
    
    builder = TidyLLMInternalRAG()
    result = builder.build_internal_domain_rag()
    
    print(f"\n" + "="*80)
    print("TIDYLLM INTERNAL DOMAIN RAG COMPLETE")
    print("="*80)
    print(f"System Type: {result['system_type']}")
    print(f"Purpose: {result['purpose']}")
    print(f"Demo: python {result['demo_path']}")

if __name__ == "__main__":
    main()