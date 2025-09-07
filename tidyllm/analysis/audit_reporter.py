#!/usr/bin/env python3
"""
Comprehensive Python File Audit Tool
=====================================

Analyzes all Python files in the codebase to determine:
- Active vs inactive status
- Last modification date
- Usage patterns and uncertainty factors
- Submodule organization

Generates CSV report for cleanup planning.
"""

import os
import csv
import stat
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import re
from typing import Dict, List, Tuple, Optional

class PythonFileAuditor:
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path).resolve()
        self.today = datetime.now().date()
        self.recent_threshold = self.today - timedelta(days=7)
        
        # Patterns for determining file status
        self.active_patterns = [
            r'if __name__ == ["\']__main__["\']',
            r'def main\(',
            r'from scripts\.',
            r'UnifiedSessionManager',
            r'import tidyllm',
            r'@app\.route',
            r'streamlit',
            r'def test_'
        ]
        
        self.inactive_patterns = [
            r'# TODO.*deprecated',
            r'# DEPRECATED',
            r'# OLD.*',
            r'# BROKEN',
            r'raise NotImplementedError',
            r'pass  # placeholder'
        ]
        
        self.uncertain_patterns = [
            r'boto3\.client',  # Should use UnifiedSessionManager
            r'psycopg2\.connect',  # Should use UnifiedSessionManager
            r'import numpy',  # Should use tidyllm.tlm
            r'from sklearn',  # Should use tidyllm.tlm
            r'sentence_transformers',  # Should use tidyllm-sentence
            r'# TODO',
            r'# FIXME',
            r'# HACK'
        ]

    def get_file_stats(self, file_path: Path) -> Dict:
        """Get file statistics and metadata"""
        try:
            stats = file_path.stat()
            mod_time = datetime.fromtimestamp(stats.st_mtime)
            
            return {
                'size': stats.st_size,
                'last_modified': mod_time,
                'is_recent': mod_time.date() >= self.recent_threshold,
                'is_today': mod_time.date() == self.today,
                'permissions': stat.filemode(stats.st_mode)
            }
        except Exception as e:
            return {
                'size': 0,
                'last_modified': datetime.min,
                'is_recent': False,
                'is_today': False,
                'permissions': 'unknown',
                'error': str(e)
            }

    def analyze_file_content(self, file_path: Path) -> Dict:
        """Analyze file content to determine status"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Count pattern matches
            active_score = sum(1 for pattern in self.active_patterns 
                             if re.search(pattern, content, re.IGNORECASE))
            
            inactive_score = sum(1 for pattern in self.inactive_patterns 
                                if re.search(pattern, content, re.IGNORECASE))
            
            uncertain_score = sum(1 for pattern in self.uncertain_patterns 
                                 if re.search(pattern, content, re.IGNORECASE))
            
            # Additional analysis
            import_count = len(re.findall(r'^import |^from .+ import', content, re.MULTILINE))
            function_count = len(re.findall(r'^def ', content, re.MULTILINE))
            class_count = len(re.findall(r'^class ', content, re.MULTILINE))
            comment_lines = len([line for line in content.split('\n') 
                               if line.strip().startswith('#')])
            
            # Determine status
            if inactive_score > 0 or 'deprecated' in file_path.name.lower():
                status = 'inactive'
            elif active_score >= 2 and uncertain_score == 0:
                status = 'active'
            elif uncertain_score > 0 or active_score == 1:
                status = 'uncertain'
            elif function_count == 0 and class_count == 0 and import_count == 0:
                status = 'inactive'
            else:
                status = 'uncertain'
                
            # Generate notes
            notes = []
            if 'test' in file_path.name.lower():
                notes.append('test_file')
            if 'demo' in file_path.name.lower():
                notes.append('demo')
            if uncertain_score > 0:
                notes.append(f'uncertain_patterns:{uncertain_score}')
            if inactive_score > 0:
                notes.append(f'inactive_patterns:{inactive_score}')
            if import_count == 0:
                notes.append('no_imports')
            if function_count == 0 and class_count == 0:
                notes.append('no_functions_or_classes')
                
            return {
                'status': status,
                'active_score': active_score,
                'inactive_score': inactive_score,
                'uncertain_score': uncertain_score,
                'import_count': import_count,
                'function_count': function_count,
                'class_count': class_count,
                'comment_lines': comment_lines,
                'line_count': len(content.split('\n')),
                'notes': ';'.join(notes) if notes else 'standard'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'notes': f'read_error:{str(e)}',
                'active_score': 0,
                'inactive_score': 0,
                'uncertain_score': 0,
                'import_count': 0,
                'function_count': 0,
                'class_count': 0,
                'comment_lines': 0,
                'line_count': 0
            }

    def get_submodule_info(self) -> Dict[str, Dict]:
        """Get information about git submodules"""
        submodules = {}
        try:
            # Check for .gitmodules file
            gitmodules_path = self.root_path / '.gitmodules'
            if gitmodules_path.exists():
                with open(gitmodules_path, 'r') as f:
                    content = f.read()
                    
                # Parse submodule entries
                current_submodule = None
                for line in content.split('\n'):
                    line = line.strip()
                    if line.startswith('[submodule '):
                        current_submodule = line.split('"')[1]
                        submodules[current_submodule] = {}
                    elif '=' in line and current_submodule:
                        key, value = line.split('=', 1)
                        submodules[current_submodule][key.strip()] = value.strip()
                        
        except Exception as e:
            print(f"Error reading submodules: {e}")
            
        return submodules

    def audit_directory(self, directory: Path, max_depth: int = 10) -> List[Dict]:
        """Recursively audit Python files in directory"""
        files_data = []
        
        if max_depth <= 0:
            return files_data
            
        try:
            for item in directory.iterdir():
                if item.is_file() and item.suffix == '.py':
                    # Get file statistics
                    file_stats = self.get_file_stats(item)
                    
                    # Analyze content
                    content_analysis = self.analyze_file_content(item)
                    
                    # Determine relative path
                    try:
                        rel_path = item.relative_to(self.root_path)
                    except ValueError:
                        rel_path = item
                    
                    # Combine data
                    file_data = {
                        'filename': item.name,
                        'relative_path': str(rel_path),
                        'directory': str(item.parent.relative_to(self.root_path)),
                        'size_bytes': file_stats['size'],
                        'last_modified': file_stats['last_modified'].strftime('%Y-%m-%d %H:%M:%S'),
                        'is_recent': file_stats['is_recent'],
                        'is_today': file_stats['is_today'],
                        'status': content_analysis['status'],
                        'notes': content_analysis['notes'],
                        'active_score': content_analysis['active_score'],
                        'inactive_score': content_analysis['inactive_score'],
                        'uncertain_score': content_analysis['uncertain_score'],
                        'import_count': content_analysis['import_count'],
                        'function_count': content_analysis['function_count'],
                        'class_count': content_analysis['class_count'],
                        'line_count': content_analysis['line_count']
                    }
                    
                    files_data.append(file_data)
                    
                elif item.is_dir() and not item.name.startswith('.'):
                    # Skip certain directories
                    skip_dirs = {'__pycache__', '.git', 'node_modules', '.env', 'venv', '.venv'}
                    if item.name not in skip_dirs:
                        files_data.extend(self.audit_directory(item, max_depth - 1))
                        
        except PermissionError:
            print(f"Permission denied: {directory}")
        except Exception as e:
            print(f"Error processing {directory}: {e}")
            
        return files_data

    def generate_report(self) -> str:
        """Generate comprehensive audit report"""
        print("[AUDIT] Starting comprehensive Python file audit...")
        
        # Audit all Python files
        all_files = self.audit_directory(self.root_path)
        
        # Get submodule information
        submodules = self.get_submodule_info()
        
        # Sort by status and last modified
        all_files.sort(key=lambda x: (x['status'], x['last_modified']), reverse=True)
        
        # Generate CSV
        csv_filename = f"python_audit_{self.today.strftime('%Y%m%d')}.csv"
        csv_path = self.root_path / csv_filename
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            if all_files:
                fieldnames = all_files[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_files)
        
        # Generate summary statistics
        total_files = len(all_files)
        active_files = len([f for f in all_files if f['status'] == 'active'])
        inactive_files = len([f for f in all_files if f['status'] == 'inactive'])
        uncertain_files = len([f for f in all_files if f['status'] == 'uncertain'])
        error_files = len([f for f in all_files if f['status'] == 'error'])
        recent_files = len([f for f in all_files if f['is_recent']])
        today_files = len([f for f in all_files if f['is_today']])
        
        # Generate report
        report = f"""
=== COMPREHENSIVE PYTHON FILE AUDIT REPORT ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Root Directory: {self.root_path}

[STATS] SUMMARY STATISTICS:
- Total Python files: {total_files}
- Active files: {active_files} ({active_files/total_files*100:.1f}%)
- Inactive files: {inactive_files} ({inactive_files/total_files*100:.1f}%)
- Uncertain files: {uncertain_files} ({uncertain_files/total_files*100:.1f}%)
- Error files: {error_files} ({error_files/total_files*100:.1f}%)
- Recent files (7 days): {recent_files} ({recent_files/total_files*100:.1f}%)
- Today's files: {today_files} ({today_files/total_files*100:.1f}%)

[UNCERTAIN] FILES REQUIRING INVESTIGATION:
"""
        
        uncertain_files_list = [f for f in all_files if f['status'] == 'uncertain']
        for file_data in uncertain_files_list[:20]:  # Show first 20 uncertain files
            report += f"- {file_data['relative_path']} | {file_data['notes']} | Modified: {file_data['last_modified']}\n"
            
        if len(uncertain_files_list) > 20:
            report += f"... and {len(uncertain_files_list) - 20} more uncertain files\n"
        
        report += f"""
[SUBMODULES] INFORMATION:
"""
        if submodules:
            for name, info in submodules.items():
                report += f"- {name}: {info.get('url', 'Unknown URL')}\n"
        else:
            report += "No submodules found in .gitmodules\n"
        
        report += f"""
[CSV] DETAILED REPORT SAVED: {csv_filename}

[NEXT] STEPS:
1. Review uncertain files to determine actual status
2. Clean up inactive files and move to deprecated
3. Analyze submodules for cleanup opportunities  
4. Prepare clean codebase for fresh repository
"""
        
        return report

def main():
    """Main execution function"""
    auditor = PythonFileAuditor()
    report = auditor.generate_report()
    print(report)
    
    # Save report to file
    report_filename = f"python_audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n[OK] Audit complete! Report saved as: {report_filename}")

if __name__ == "__main__":
    main()