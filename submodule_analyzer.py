#!/usr/bin/env python3
"""
Submodule Analyzer for TidyLLM Ecosystem
========================================

Analyzes all tidyllm-* directories to understand:
- Git status and commit history
- Python file organization
- Active vs inactive status
- Integration patterns
"""

import os
import subprocess
from pathlib import Path
import json
from datetime import datetime

class SubmoduleAnalyzer:
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path).resolve()
        self.analysis_results = {}
        
    def get_git_info(self, path: Path) -> dict:
        """Get git information for a directory"""
        try:
            old_cwd = os.getcwd()
            os.chdir(path)
            
            # Check if it's a git repo
            result = subprocess.run(['git', 'rev-parse', '--is-inside-work-tree'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                return {'is_git_repo': False}
            
            # Get basic git info
            git_info = {'is_git_repo': True}
            
            # Current branch
            result = subprocess.run(['git', 'branch', '--show-current'], 
                                  capture_output=True, text=True)
            git_info['current_branch'] = result.stdout.strip() if result.returncode == 0 else 'unknown'
            
            # Last commit
            result = subprocess.run(['git', 'log', '-1', '--format=%H|%s|%ad'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                commit_data = result.stdout.strip().split('|', 2)
                if len(commit_data) >= 3:
                    git_info['last_commit'] = {
                        'hash': commit_data[0],
                        'message': commit_data[1],
                        'date': commit_data[2]
                    }
            
            # Remote info
            result = subprocess.run(['git', 'remote', '-v'], 
                                  capture_output=True, text=True)
            git_info['remotes'] = result.stdout.strip() if result.returncode == 0 else 'none'
            
            # Status
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  capture_output=True, text=True)
            git_info['has_changes'] = len(result.stdout.strip()) > 0
            git_info['status_output'] = result.stdout.strip()
            
            return git_info
            
        except Exception as e:
            return {'is_git_repo': False, 'error': str(e)}
        finally:
            os.chdir(old_cwd)
    
    def analyze_python_files(self, path: Path) -> dict:
        """Analyze Python files in directory"""
        py_files = list(path.rglob("*.py"))
        
        analysis = {
            'total_py_files': len(py_files),
            'directories': set(),
            'main_files': [],
            'test_files': [],
            'demo_files': [],
            'has_setup_py': (path / 'setup.py').exists(),
            'has_pyproject_toml': (path / 'pyproject.toml').exists(),
            'has_requirements_txt': (path / 'requirements.txt').exists(),
            'has_readme': any((path / f"README.{ext}").exists() for ext in ['md', 'rst', 'txt'])
        }
        
        for py_file in py_files:
            rel_path = py_file.relative_to(path)
            analysis['directories'].add(str(rel_path.parent))
            
            # Categorize files
            filename = py_file.name.lower()
            if filename in ['main.py', '__main__.py'] or filename.startswith('app'):
                analysis['main_files'].append(str(rel_path))
            elif 'test' in filename:
                analysis['test_files'].append(str(rel_path))
            elif 'demo' in filename or 'example' in filename:
                analysis['demo_files'].append(str(rel_path))
        
        analysis['directories'] = sorted(list(analysis['directories']))
        return analysis
    
    def determine_activity_status(self, path: Path, git_info: dict, py_analysis: dict) -> str:
        """Determine if submodule is active, inactive, or uncertain"""
        
        # Check for clear inactive indicators
        if not py_analysis['total_py_files']:
            return 'inactive_no_python'
        
        if not git_info.get('is_git_repo', False):
            return 'inactive_not_git'
            
        # Check for active indicators
        active_score = 0
        
        if py_analysis['has_setup_py'] or py_analysis['has_pyproject_toml']:
            active_score += 2
            
        if py_analysis['main_files']:
            active_score += 2
            
        if py_analysis['has_readme']:
            active_score += 1
            
        if git_info.get('has_changes', False):
            active_score += 1
            
        # Check last commit date
        last_commit = git_info.get('last_commit', {})
        if last_commit:
            try:
                # Simple heuristic - if committed recently, likely active
                commit_date = last_commit.get('date', '')
                if '2025' in commit_date or '2024' in commit_date:
                    active_score += 1
            except:
                pass
        
        if active_score >= 4:
            return 'active'
        elif active_score >= 2:
            return 'uncertain'
        else:
            return 'inactive'
    
    def analyze_submodule(self, path: Path) -> dict:
        """Comprehensive analysis of a single submodule"""
        print(f"[ANALYZING] {path.name}")
        
        analysis = {
            'name': path.name,
            'path': str(path),
            'analyzed_at': datetime.now().isoformat(),
        }
        
        # Git analysis
        analysis['git'] = self.get_git_info(path)
        
        # Python file analysis
        analysis['python'] = self.analyze_python_files(path)
        
        # Activity status
        analysis['status'] = self.determine_activity_status(path, analysis['git'], analysis['python'])
        
        # Size analysis
        try:
            total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            analysis['size_bytes'] = total_size
            analysis['size_mb'] = round(total_size / (1024*1024), 2)
        except:
            analysis['size_bytes'] = 0
            analysis['size_mb'] = 0
        
        return analysis
    
    def analyze_all_submodules(self) -> dict:
        """Analyze all tidyllm-* submodules"""
        print("[SUBMODULE] Starting submodule analysis...")
        
        # Find all tidyllm-* directories
        submodule_dirs = [d for d in self.root_path.iterdir() 
                         if d.is_dir() and d.name.startswith('tidyllm-')]
        
        # Also check for tlm directory
        tlm_dir = self.root_path / 'tlm'
        if tlm_dir.exists() and tlm_dir.is_dir():
            submodule_dirs.append(tlm_dir)
            
        results = {
            'analysis_date': datetime.now().isoformat(),
            'root_path': str(self.root_path),
            'total_submodules': len(submodule_dirs),
            'submodules': {}
        }
        
        for submodule_dir in sorted(submodule_dirs):
            analysis = self.analyze_submodule(submodule_dir)
            results['submodules'][submodule_dir.name] = analysis
        
        return results
    
    def generate_report(self) -> str:
        """Generate comprehensive submodule report"""
        results = self.analyze_all_submodules()
        
        # Statistics
        total = results['total_submodules']
        active = len([s for s in results['submodules'].values() if s['status'] == 'active'])
        inactive = len([s for s in results['submodules'].values() if s['status'].startswith('inactive')])
        uncertain = len([s for s in results['submodules'].values() if s['status'] == 'uncertain'])
        
        total_size_mb = sum(s['size_mb'] for s in results['submodules'].values())
        
        report = f"""
=== TIDYLLM SUBMODULE ANALYSIS REPORT ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Root Directory: {results['root_path']}

[STATS] SUBMODULE SUMMARY:
- Total submodules: {total}
- Active: {active} ({active/total*100:.1f}%)
- Inactive: {inactive} ({inactive/total*100:.1f}%)
- Uncertain: {uncertain} ({uncertain/total*100:.1f}%)
- Total disk usage: {total_size_mb:.1f} MB

[DETAILED] SUBMODULE ANALYSIS:
"""
        
        for name, analysis in results['submodules'].items():
            status_indicator = {
                'active': '[ACTIVE]',
                'uncertain': '[UNCERTAIN]',
                'inactive': '[INACTIVE]',
                'inactive_no_python': '[INACTIVE-NO-PY]',
                'inactive_not_git': '[INACTIVE-NOT-GIT]'
            }.get(analysis['status'], '[UNKNOWN]')
            
            git_info = analysis.get('git', {})
            py_info = analysis.get('python', {})
            
            report += f"""
{status_indicator} {name}:
  - Size: {analysis['size_mb']:.1f} MB
  - Python files: {py_info.get('total_py_files', 0)}
  - Git repo: {git_info.get('is_git_repo', False)}
  - Has changes: {git_info.get('has_changes', False)}
  - Main files: {len(py_info.get('main_files', []))}
  - Test files: {len(py_info.get('test_files', []))}
  - Package files: setup.py={py_info.get('has_setup_py', False)}, pyproject.toml={py_info.get('has_pyproject_toml', False)}"""
            
            if git_info.get('last_commit'):
                commit = git_info['last_commit']
                report += f"\n  - Last commit: {commit.get('message', 'Unknown')[:50]}... ({commit.get('date', 'Unknown date')})"
        
        report += f"""

[CLEANUP] RECOMMENDATIONS:
"""
        
        # Generate cleanup recommendations
        for name, analysis in results['submodules'].items():
            if analysis['status'].startswith('inactive'):
                report += f"- REMOVE: {name} - {analysis['status']}\n"
            elif analysis['status'] == 'uncertain':
                report += f"- REVIEW: {name} - Needs investigation\n"
        
        report += f"""
[JSON] Full analysis saved to: submodule_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json
"""
        
        return report, results

def main():
    """Main execution function"""
    analyzer = SubmoduleAnalyzer()
    report_text, full_results = analyzer.generate_report()
    
    # Save JSON results
    json_filename = f"submodule_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    
    # Save text report
    report_filename = f"submodule_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n[OK] Submodule analysis complete! Reports saved as:")
    print(f"  - {json_filename}")
    print(f"  - {report_filename}")

if __name__ == "__main__":
    main()