#!/usr/bin/env python3
"""
Documentation Date Organizer
============================

Organizes all documentation by creation/modification date into docs/YYYY-MM-DD/ folders.
Prevents duplication and maintains single source of truth for each document.
"""

import os
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import json

class DocumentationOrganizer:
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path).resolve()
        self.docs_path = self.root_path / "docs"
        self.deprecated_docs_path = self.root_path / "deprecated" / "docs"
        self.duplicate_tracker = {}
        self.moved_files = []
        
    def get_file_hash(self, file_path: Path) -> str:
        """Generate hash of file content to detect duplicates"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return None
            
    def get_file_dates(self, file_path: Path) -> Tuple[datetime, datetime]:
        """Get creation and modification dates"""
        stat = file_path.stat()
        mod_time = datetime.fromtimestamp(stat.st_mtime)
        # On Windows, st_ctime is creation time; on Unix it's last metadata change
        create_time = datetime.fromtimestamp(stat.st_ctime)
        return create_time, mod_time
        
    def find_all_documentation(self) -> List[Path]:
        """Find all documentation files in the repository"""
        doc_extensions = ['.md', '.txt', '.rst', '.pdf', '.docx', '.html']
        doc_files = []
        
        # Common documentation locations
        search_paths = [
            self.root_path,
            self.root_path / 'docs',
            self.root_path / 'documentation',
            self.root_path / 'tidyllm',
            self.root_path / 'scripts',
            self.root_path / 'tests'
        ]
        
        for search_path in search_paths:
            if not search_path.exists():
                continue
                
            for ext in doc_extensions:
                doc_files.extend(search_path.rglob(f'*{ext}'))
                
        # Filter out already organized files and deprecated
        filtered = []
        for doc in doc_files:
            path_str = str(doc)
            # Skip if already in date folder
            if '/docs/20' in path_str or '\\docs\\20' in path_str:
                continue
            # Skip if in deprecated
            if '/deprecated/' in path_str or '\\deprecated\\' in path_str:
                continue
            # Skip if in .git
            if '/.git/' in path_str or '\\.git\\' in path_str:
                continue
            # Skip node_modules, venv, etc
            if any(skip in path_str for skip in ['node_modules', 'venv', '.venv', '__pycache__']):
                continue
                
            filtered.append(doc)
            
        return filtered
        
    def organize_by_date(self, dry_run: bool = True) -> Dict:
        """Organize documentation by date"""
        print(f"[SCAN] Finding all documentation files...")
        doc_files = self.find_all_documentation()
        print(f"[SCAN] Found {len(doc_files)} documentation files to organize")
        
        organization_plan = {
            'by_date': {},
            'duplicates': [],
            'errors': []
        }
        
        # Build organization plan
        for doc_path in doc_files:
            try:
                # Get file info
                file_hash = self.get_file_hash(doc_path)
                create_time, mod_time = self.get_file_dates(doc_path)
                
                # Use modification time as primary date
                file_date = mod_time.strftime('%Y-%m-%d')
                
                # Check for duplicates
                if file_hash and file_hash in self.duplicate_tracker:
                    existing = self.duplicate_tracker[file_hash]
                    organization_plan['duplicates'].append({
                        'file': str(doc_path),
                        'duplicate_of': existing['path'],
                        'action': 'skip'
                    })
                    continue
                    
                # Track this file
                if file_hash:
                    self.duplicate_tracker[file_hash] = {
                        'path': str(doc_path),
                        'date': file_date
                    }
                
                # Determine target path
                date_folder = self.docs_path / file_date
                target_path = date_folder / doc_path.name
                
                # Handle name conflicts
                if target_path.exists():
                    # Check if same content
                    target_hash = self.get_file_hash(target_path)
                    if target_hash == file_hash:
                        organization_plan['duplicates'].append({
                            'file': str(doc_path),
                            'duplicate_of': str(target_path),
                            'action': 'delete_source'
                        })
                        continue
                    else:
                        # Different content, rename
                        base = target_path.stem
                        ext = target_path.suffix
                        counter = 1
                        while target_path.exists():
                            target_path = date_folder / f"{base}_{counter}{ext}"
                            counter += 1
                
                # Add to plan
                if file_date not in organization_plan['by_date']:
                    organization_plan['by_date'][file_date] = []
                    
                organization_plan['by_date'][file_date].append({
                    'source': str(doc_path),
                    'target': str(target_path),
                    'size': doc_path.stat().st_size,
                    'created': create_time.isoformat(),
                    'modified': mod_time.isoformat()
                })
                
            except Exception as e:
                organization_plan['errors'].append({
                    'file': str(doc_path),
                    'error': str(e)
                })
        
        # Execute plan if not dry run
        if not dry_run:
            print(f"\n[EXECUTE] Organizing documentation...")
            for date, files in organization_plan['by_date'].items():
                date_folder = self.docs_path / date
                date_folder.mkdir(parents=True, exist_ok=True)
                
                for file_info in files:
                    source = Path(file_info['source'])
                    target = Path(file_info['target'])
                    
                    try:
                        print(f"[MOVE] {source.name} -> docs/{date}/")
                        shutil.move(str(source), str(target))
                        self.moved_files.append(file_info)
                    except Exception as e:
                        print(f"[ERROR] Failed to move {source}: {e}")
                        
            # Handle duplicates
            for dup in organization_plan['duplicates']:
                if dup['action'] == 'delete_source':
                    try:
                        os.remove(dup['file'])
                        print(f"[DELETE] Removed duplicate: {Path(dup['file']).name}")
                    except:
                        pass
        
        return organization_plan
        
    def generate_report(self, organization_plan: Dict) -> str:
        """Generate organization report"""
        report = f"""
=== DOCUMENTATION ORGANIZATION REPORT ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

[SUMMARY]
- Total dates with documentation: {len(organization_plan['by_date'])}
- Total files to organize: {sum(len(files) for files in organization_plan['by_date'].values())}
- Duplicate files found: {len(organization_plan['duplicates'])}
- Errors encountered: {len(organization_plan['errors'])}

[ORGANIZATION BY DATE]
"""
        
        # Sort dates descending (newest first)
        for date in sorted(organization_plan['by_date'].keys(), reverse=True):
            files = organization_plan['by_date'][date]
            report += f"\n[DATE] {date} ({len(files)} files):\n"
            for file_info in files[:10]:  # Show first 10
                source_path = Path(file_info['source'])
                report += f"  - {source_path.name} ({file_info['size']} bytes)\n"
            if len(files) > 10:
                report += f"  ... and {len(files) - 10} more files\n"
                
        if organization_plan['duplicates']:
            report += f"\n[DUPLICATES FOUND]\n"
            for dup in organization_plan['duplicates'][:20]:
                report += f"- {Path(dup['file']).name} = {Path(dup['duplicate_of']).name}\n"
                
        if organization_plan['errors']:
            report += f"\n[ERRORS]\n"
            for err in organization_plan['errors'][:10]:
                report += f"- {Path(err['file']).name}: {err['error']}\n"
                
        report += f"""
[PROPOSED STRUCTURE]
docs/
+-- 2025-09-05/     # Today's documentation
+-- 2025-09-04/     # Yesterday's updates
+-- 2025-09-03/     # Earlier this week
+-- 2025-09-02/
+-- 2025-09-01/
+-- 2025-08-31/
+-- 2025-08-30/
+-- ... (older dates)

[BENEFITS]
- Clear chronological organization
- No duplicate documentation
- Easy to find recent changes
- Historical record preserved
- Single source of truth per document
"""
        
        return report

def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Organize documentation by date')
    parser.add_argument('--execute', action='store_true', help='Execute the organization (default is dry run)')
    parser.add_argument('--output', default='doc_organization_report.txt', help='Output report file')
    args = parser.parse_args()
    
    organizer = DocumentationOrganizer()
    
    # Run organization
    dry_run = not args.execute
    if dry_run:
        print("[DRY RUN] Analyzing documentation organization (no files will be moved)...")
    else:
        print("[EXECUTE] Organizing documentation by date...")
        
    organization_plan = organizer.organize_by_date(dry_run=dry_run)
    
    # Generate report
    report = organizer.generate_report(organization_plan)
    
    # Save report
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(report)
        
    # Save detailed plan as JSON
    json_file = args.output.replace('.txt', '.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(organization_plan, f, indent=2, default=str)
    
    print(report)
    print(f"\n[COMPLETE] Report saved to: {args.output}")
    print(f"[COMPLETE] Detailed plan saved to: {json_file}")
    
    if dry_run:
        print("\n[INFO] This was a dry run. Use --execute to actually organize files.")

if __name__ == "__main__":
    main()