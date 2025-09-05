#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze Uncategorized Files Script
==================================

This script analyzes the remaining uncategorized files in input/omnibus
and determines what should be discarded or further organized.

Usage:
    python scripts/analyze_uncategorized.py [--move-to-discard]
"""

import os
import sys
import shutil
import argparse
import logging
from pathlib import Path
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UncategorizedAnalyzer:
    """Analyze uncategorized files and determine disposition"""
    
    def __init__(self, project_root=None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.input_dir = self.project_root / "input" / "omnibus"
        self.discard_dir = self.project_root / "discard"
        
    def analyze_uncategorized_files(self):
        """Analyze all files in input/omnibus and categorize them"""
        logger.info("📊 Analyzing uncategorized files...")
        
        # Categories for uncategorized files
        categories = {
            'code_repositories': [],
            'wfc_files': [],
            'miscellaneous_files': [],
            'checkpoint_files': [],
            'large_files': [],
            'potential_keep': []
        }
        
        total_files = 0
        total_size = 0
        
        # Walk through all files
        for file_path in self.input_dir.rglob('*'):
            if file_path.is_file():
                total_files += 1
                file_size = file_path.stat().st_size
                total_size += file_size
                relative_path = file_path.relative_to(self.input_dir)
                
                # Categorize based on path and filename
                if self._is_code_repository(file_path):
                    categories['code_repositories'].append({
                        'path': relative_path,
                        'size': file_size,
                        'full_path': file_path
                    })
                elif self._is_wfc_file(file_path):
                    categories['wfc_files'].append({
                        'path': relative_path,
                        'size': file_size,
                        'full_path': file_path
                    })
                elif self._is_checkpoint_file(file_path):
                    categories['checkpoint_files'].append({
                        'path': relative_path,
                        'size': file_size,
                        'full_path': file_path
                    })
                elif file_size > 10 * 1024 * 1024:  # Files larger than 10MB
                    categories['large_files'].append({
                        'path': relative_path,
                        'size': file_size,
                        'full_path': file_path
                    })
                elif self._is_potential_keep(file_path):
                    categories['potential_keep'].append({
                        'path': relative_path,
                        'size': file_size,
                        'full_path': file_path
                    })
                else:
                    categories['miscellaneous_files'].append({
                        'path': relative_path,
                        'size': file_size,
                        'full_path': file_path
                    })
        
        # Print analysis
        logger.info(f"📊 Total files analyzed: {total_files}")
        logger.info(f"📊 Total size: {total_size / 1024 / 1024:.1f} MB")
        
        for category, files in categories.items():
            if files:
                category_size = sum(f['size'] for f in files)
                logger.info(f"📁 {category.upper()}: {len(files)} files, {category_size / 1024 / 1024:.1f} MB")
                
                # Show sample files for each category
                logger.info(f"   Sample files:")
                for i, file_info in enumerate(files[:3]):
                    logger.info(f"     - {file_info['path']} ({file_info['size'] / 1024 / 1024:.1f} MB)")
                if len(files) > 3:
                    logger.info(f"     ... and {len(files) - 3} more files")
                logger.info("")
        
        return categories, total_files, total_size
    
    def _is_code_repository(self, file_path):
        """Check if file is part of a code repository"""
        path_str = str(file_path)
        return any(repo in path_str for repo in [
            'bayesian-CNN-LSTM-Q-learning-main',
            'Data-Detective-Stats-Game',
            'SSRI Network Tutorial Materials'
        ])
    
    def _is_wfc_file(self, file_path):
        """Check if file is a WFC (Wells Fargo) file"""
        return 'wfc' in str(file_path).lower()
    
    def _is_checkpoint_file(self, file_path):
        """Check if file is a checkpoint file"""
        return '.ipynb_checkpoints' in str(file_path) or 'checkpoint' in file_path.name.lower()
    
    def _is_potential_keep(self, file_path):
        """Check if file might be worth keeping"""
        filename = file_path.name.lower()
        return any(keyword in filename for keyword in [
            'model', 'risk', 'financial', 'mlops', 'best_practices',
            'harvard', 'lecture', 'notes', 'teradata', 'mistral'
        ])
    
    def move_to_discard(self, categories):
        """Move appropriate files to discard folder"""
        logger.info("🗑️ Moving files to discard folder...")
        
        # Create discard subdirectories
        discard_dirs = {
            'code_repositories': self.discard_dir / 'code_repositories',
            'wfc_files': self.discard_dir / 'wfc_files',
            'checkpoint_files': self.discard_dir / 'checkpoint_files',
            'miscellaneous_files': self.discard_dir / 'miscellaneous_files',
            'large_files': self.discard_dir / 'large_files'
        }
        
        for dir_path in discard_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        moved_count = 0
        total_size = 0
        
        # Move files to appropriate discard subdirectories
        for category, files in categories.items():
            if category in discard_dirs and files:
                for file_info in files:
                    source_path = file_info['full_path']
                    dest_path = discard_dirs[category] / file_info['path'].name
                    
                    try:
                        # Create destination directory if needed
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Move file
                        shutil.move(str(source_path), str(dest_path))
                        moved_count += 1
                        total_size += file_info['size']
                        logger.debug(f"Moved: {file_info['path']} -> {dest_path}")
                        
                    except Exception as e:
                        logger.error(f"Failed to move {file_info['path']}: {e}")
        
        logger.info(f"✅ Moved {moved_count} files to discard ({total_size / 1024 / 1024:.1f} MB)")
        
        # Show what's left (potential keep files)
        potential_keep = categories.get('potential_keep', [])
        if potential_keep:
            logger.info(f"📋 {len(potential_keep)} files marked for potential keeping:")
            for file_info in potential_keep:
                logger.info(f"   - {file_info['path']} ({file_info['size'] / 1024 / 1024:.1f} MB)")
        
        return moved_count, total_size
    
    def create_discard_summary(self, categories, total_files, total_size):
        """Create a summary of the discard operation"""
        logger.info("📋 Creating discard summary...")
        
        summary_content = f"""# Uncategorized Files Analysis & Discard Summary

Generated on: {__import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 📊 Analysis Results

### Total Files Analyzed
- **Files**: {total_files}
- **Size**: {total_size / 1024 / 1024:.1f} MB

## 📁 File Categories

### Code Repositories
- **Files**: {len(categories['code_repositories'])}
- **Size**: {sum(f['size'] for f in categories['code_repositories']) / 1024 / 1024:.1f} MB
- **Examples**: bayesian-CNN-LSTM-Q-learning-main, Data-Detective-Stats-Game
- **Disposition**: Moved to discard/code_repositories/

### WFC Files (Wells Fargo)
- **Files**: {len(categories['wfc_files'])}
- **Size**: {sum(f['size'] for f in categories['wfc_files']) / 1024 / 1024:.1f} MB
- **Examples**: Quarterly reports, financial documents
- **Disposition**: Moved to discard/wfc_files/

### Checkpoint Files
- **Files**: {len(categories['checkpoint_files'])}
- **Size**: {sum(f['size'] for f in categories['checkpoint_files']) / 1024 / 1024:.1f} MB
- **Examples**: .ipynb_checkpoints, temporary files
- **Disposition**: Moved to discard/checkpoint_files/

### Large Files (>10MB)
- **Files**: {len(categories['large_files'])}
- **Size**: {sum(f['size'] for f in categories['large_files']) / 1024 / 1024:.1f} MB
- **Disposition**: Moved to discard/large_files/

### Miscellaneous Files
- **Files**: {len(categories['miscellaneous_files'])}
- **Size**: {sum(f['size'] for f in categories['miscellaneous_files']) / 1024 / 1024:.1f} MB
- **Disposition**: Moved to discard/miscellaneous_files/

### Potential Keep Files
- **Files**: {len(categories['potential_keep'])}
- **Size**: {sum(f['size'] for f in categories['potential_keep']) / 1024 / 1024:.1f} MB
- **Examples**: Model files, Harvard lectures, MLOps guides
- **Disposition**: Left in input/omnibus/ for manual review

## 🎯 Recommendations

### Files to Review Manually
The following files might be worth keeping for the knowledge base:

"""
        
        # Add potential keep files to summary
        for file_info in categories.get('potential_keep', []):
            summary_content += f"- {file_info['path']} ({file_info['size'] / 1024 / 1024:.1f} MB)\n"
        
        summary_content += f"""

### Next Steps
1. **Review potential keep files** and move relevant ones to knowledge base
2. **Clean up discard folder** if needed
3. **Update deployment packages** to exclude discarded files
4. **Test knowledge base** with remaining organized content

## 📁 Discard Structure

```
discard/
├── code_repositories/     # ML code projects
├── wfc_files/            # Wells Fargo documents
├── checkpoint_files/     # Jupyter checkpoints
├── large_files/          # Files >10MB
└── miscellaneous_files/  # Other uncategorized files
```

**Total files moved to discard: {sum(len(categories.get(cat, [])) for cat in ['code_repositories', 'wfc_files', 'checkpoint_files', 'large_files', 'miscellaneous_files'])}**
"""
        
        summary_path = self.project_root / "DISCARD_SUMMARY.md"
        with open(summary_path, 'w') as f:
            f.write(summary_content)
            
        logger.info(f"📋 Summary created: {summary_path}")
        return summary_path

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Analyze uncategorized files")
    parser.add_argument("--move-to-discard", action="store_true", help="Move files to discard folder")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    try:
        analyzer = UncategorizedAnalyzer()
        
        # Analyze files
        categories, total_files, total_size = analyzer.analyze_uncategorized_files()
        
        if args.move_to_discard:
            # Move files to discard
            moved_count, moved_size = analyzer.move_to_discard(categories)
            
            # Create summary
            summary_path = analyzer.create_discard_summary(categories, total_files, total_size)
            
            print(f"\n🎉 Discard operation complete!")
            print(f"📊 Files moved: {moved_count}")
            print(f"📊 Size moved: {moved_size / 1024 / 1024:.1f} MB")
            print(f"📋 Summary: {summary_path}")
        else:
            print(f"\n📊 Analysis complete!")
            print(f"📊 Total files: {total_files}")
            print(f"📊 Total size: {total_size / 1024 / 1024:.1f} MB")
            print(f"💡 Use --move-to-discard to move files to discard folder")
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
