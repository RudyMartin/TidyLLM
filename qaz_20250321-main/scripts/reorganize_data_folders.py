#!/usr/bin/env python3
"""
Data Folder Reorganization Script
=================================

Reorganizes scattered data folders into a clean hierarchical structure:

Current messy structure:
- llm_cache/, llm_metrics/, llm_enhanced_output/, etc. (at root)
- mlflow_*, mlruns/ (at root)  
- output/, test_outputs/, reports/ (at root)
- input/ (at root)

New clean structure:
data/
├── cache/
│   ├── llm/          # llm_cache, llm_metrics, llm_utilization_reports
│   └── mlflow/       # mlflow_export, mlflow_real_doc_export
├── input/            # input folder contents
├── output/           # output, reports, test_outputs  
├── experiments/      # mlruns, mlflow_real_document_output
└── logs/             # logs folder
"""

import os
import shutil
from pathlib import Path
import argparse
from datetime import datetime

class DataReorganizer:
    def __init__(self, dry_run=True):
        self.dry_run = dry_run
        self.project_root = Path.cwd()
        self.data_root = self.project_root / 'data'
        self.moves_made = []
        self.errors = []
        
    def create_data_structure(self):
        """Create the new data folder structure"""
        structure = {
            'data': {
                'cache': {
                    'llm': {},
                    'mlflow': {}
                },
                'input': {},
                'output': {},
                'experiments': {},
                'logs': {}
            }
        }
        
        def create_dirs(base_path, structure):
            for name, subdirs in structure.items():
                dir_path = base_path / name
                if not self.dry_run:
                    dir_path.mkdir(parents=True, exist_ok=True)
                print(f"  📁 {'Would create' if self.dry_run else 'Created'}: {dir_path.relative_to(self.project_root)}/")
                if subdirs:
                    create_dirs(dir_path, subdirs)
        
        print("🏗️  CREATING DATA FOLDER STRUCTURE")
        print("=" * 60)
        create_dirs(self.project_root, structure)
    
    def move_folder(self, source, destination):
        """Move a folder from source to destination"""
        source_path = self.project_root / source
        dest_path = self.data_root / destination
        
        if not source_path.exists():
            print(f"  ⚠️  Skip {source} (not found)")
            return
            
        try:
            if not self.dry_run:
                if dest_path.exists():
                    print(f"  🔄 Merging {source} → {destination}")
                    # Merge directories
                    for item in source_path.iterdir():
                        item_dest = dest_path / item.name
                        if item.is_dir():
                            shutil.copytree(item, item_dest, dirs_exist_ok=True)
                        else:
                            shutil.copy2(item, item_dest)
                    shutil.rmtree(source_path)
                else:
                    shutil.move(str(source_path), str(dest_path))
                    print(f"  ✅ Moved {source} → {destination}")
            else:
                print(f"  🔍 Would move {source} → {destination}")
                
            self.moves_made.append((source, destination))
            
        except Exception as e:
            error_msg = f"Failed to move {source}: {e}"
            self.errors.append(error_msg)
            print(f"  ❌ {error_msg}")
    
    def reorganize_folders(self):
        """Execute the folder reorganization"""
        print("\\n📦 REORGANIZING DATA FOLDERS")
        print("=" * 60)
        
        # Define the reorganization mapping
        folder_moves = [
            # LLM Cache folders
            ('llm_cache', 'cache/llm/llm_cache'),
            ('llm_metrics', 'cache/llm/llm_metrics'), 
            ('llm_enhanced_output', 'cache/llm/llm_enhanced_output'),
            ('llm_utilization_reports', 'cache/llm/llm_utilization_reports'),
            
            # MLflow Cache folders
            ('mlflow_export', 'cache/mlflow/mlflow_export'),
            ('mlflow_real_doc_export', 'cache/mlflow/mlflow_real_doc_export'),
            
            # Input data
            ('input', 'input'),
            
            # Output folders
            ('output', 'output/reports'),
            ('reports', 'output/artifacts'),
            ('test_outputs', 'output/test_results'),
            
            # Experiment data
            ('mlruns', 'experiments/mlruns'),
            ('mlflow_real_document_output', 'experiments/document_outputs'),
            
            # Logs
            ('logs', 'logs'),
        ]
        
        for source, destination in folder_moves:
            self.move_folder(source, destination)
    
    def create_readme_files(self):
        """Create README files explaining the new structure"""
        print("\\n📝 CREATING DOCUMENTATION")
        print("=" * 60)
        
        readmes = {
            'data/README.md': '''# Data Directory Structure

This directory contains all data-related folders organized by purpose:

## 📁 Directory Structure

### `cache/`
Temporary cache and intermediate processing files:
- `llm/` - LLM response caches, metrics, utilization reports
- `mlflow/` - MLflow export data and real document exports

### `input/`
Input documents and sample data files for processing

### `output/`  
Generated outputs and reports:
- `reports/` - Generated QA reports and analysis
- `artifacts/` - Report artifacts and assets
- `test_results/` - Test execution results

### `experiments/`
Experiment tracking and results:
- `mlruns/` - MLflow experiment tracking data
- `document_outputs/` - Real document processing outputs

### `logs/`
Application logs and execution traces

## 🧹 Cleanup
Cache and output folders can be safely cleaned up:
```bash
# Clean old cache files (30+ days)
python scripts/pre_flight_cleanup.py --cleanup --force

# Or manually clean specific folders
rm -rf data/cache/llm/*
rm -rf data/output/test_results/*
```

## 🔒 Security
- This entire `data/` directory is in .gitignore
- Contains no secrets or credentials
- Safe to clean up during deployment
''',
            
            'data/cache/README.md': '''# Cache Directory

Contains temporary cached data that can be safely deleted:

- **llm/** - LLM API response caches, metrics, usage reports
- **mlflow/** - MLflow export files and processed document caches

All cache files are automatically cleaned after 30 days by the cleanup script.
''',
            
            'data/experiments/README.md': '''# Experiments Directory

Contains experiment tracking and research outputs:

- **mlruns/** - MLflow experiment database and artifacts
- **document_outputs/** - Real document processing results for analysis

This data helps track model performance and document processing results.
'''
        }
        
        for file_path, content in readmes.items():
            full_path = self.project_root / file_path
            if not self.dry_run:
                full_path.parent.mkdir(parents=True, exist_ok=True)
                with open(full_path, 'w') as f:
                    f.write(content)
                print(f"  ✅ Created: {file_path}")
            else:
                print(f"  🔍 Would create: {file_path}")
    
    def update_gitignore(self):
        """Update .gitignore for the new data structure"""
        print("\\n🔒 UPDATING .GITIGNORE")
        print("=" * 60)
        
        gitignore_path = self.project_root / '.gitignore'
        
        # New gitignore entries for data folder
        data_entries = '''
# =============================================================================
# DATA DIRECTORY (auto-generated content)
# =============================================================================
# All data folders - contains caches, outputs, experiments
data/

# Legacy data folder patterns (in case any are missed)
llm_cache/
llm_metrics/
llm_enhanced_output/
llm_utilization_reports/
mlflow_export/
mlflow_real_doc_export/
mlflow_real_document_output/
mlruns/
'''
        
        if not self.dry_run:
            with open(gitignore_path, 'a') as f:
                f.write(data_entries)
            print("  ✅ Updated .gitignore with data/ patterns")
        else:
            print("  🔍 Would update .gitignore with data/ patterns")
    
    def generate_report(self):
        """Generate reorganization report"""
        print("\\n📊 REORGANIZATION REPORT")
        print("=" * 60)
        
        if self.moves_made:
            print(f"✅ Successfully {'would move' if self.dry_run else 'moved'} {len(self.moves_made)} folders:")
            for source, dest in self.moves_made:
                print(f"  {source} → data/{dest}")
        
        if self.errors:
            print(f"\\n❌ Errors encountered: {len(self.errors)}")
            for error in self.errors:
                print(f"  - {error}")
        
        print(f"\\n{'🔍 DRY RUN COMPLETE' if self.dry_run else '✅ REORGANIZATION COMPLETE'}")
        
        if self.dry_run:
            print("\\n💡 To execute the reorganization:")
            print("python scripts/reorganize_data_folders.py --execute")

def main():
    parser = argparse.ArgumentParser(description='Reorganize data folders into clean structure')
    parser.add_argument('--execute', action='store_true', help='Actually perform the reorganization (default is dry-run)')
    parser.add_argument('--dry-run', action='store_true', default=True, help='Preview changes without executing (default)')
    
    args = parser.parse_args()
    
    # If --execute is specified, turn off dry-run
    dry_run = not args.execute
    
    reorganizer = DataReorganizer(dry_run=dry_run)
    
    print(f"🗂️  DATA FOLDER REORGANIZATION {'(DRY RUN)' if dry_run else '(EXECUTING)'}")
    print("=" * 80)
    print(f"Project: {reorganizer.project_root}")
    print(f"Target: {reorganizer.data_root}/")
    
    # Execute reorganization steps
    reorganizer.create_data_structure()
    reorganizer.reorganize_folders()
    reorganizer.create_readme_files()
    reorganizer.update_gitignore()
    reorganizer.generate_report()

if __name__ == "__main__":
    main()