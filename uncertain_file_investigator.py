#!/usr/bin/env python3
"""
Uncertain File Investigator
===========================

Analyzes uncertain Python files to determine actual status and reasons for uncertainty.
"""

import csv
import re
from pathlib import Path
from typing import Dict, List
from datetime import datetime

class UncertainFileInvestigator:
    def __init__(self, csv_file: str = "python_audit_20250905.csv"):
        self.csv_file = csv_file
        self.root_path = Path(".").resolve()
        
    def load_uncertain_files(self) -> List[Dict]:
        """Load uncertain files from CSV audit"""
        uncertain_files = []
        
        try:
            with open(self.csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['status'] == 'uncertain':
                        uncertain_files.append(row)
        except FileNotFoundError:
            print(f"[ERROR] CSV file not found: {self.csv_file}")
            return []
            
        return uncertain_files
    
    def investigate_file(self, file_data: Dict) -> Dict:
        """Deep investigation of a single uncertain file"""
        file_path = Path(file_data['relative_path'])
        investigation = {
            'filename': file_data['filename'],
            'path': file_data['relative_path'],
            'original_notes': file_data['notes'],
            'investigation_date': datetime.now().isoformat(),
            'recommendation': 'unknown',
            'confidence': 'low',
            'reasoning': []
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Enhanced analysis patterns
            analysis_results = self._analyze_content_patterns(content, file_path)
            investigation.update(analysis_results)
            
            # Make recommendation based on analysis
            recommendation = self._make_recommendation(investigation, file_path)
            investigation.update(recommendation)
            
        except Exception as e:
            investigation['error'] = str(e)
            investigation['recommendation'] = 'error_reading_file'
            
        return investigation
    
    def _analyze_content_patterns(self, content: str, file_path: Path) -> Dict:
        """Analyze file content for specific patterns"""
        analysis = {
            'has_main_function': bool(re.search(r'def main\(', content)),
            'has_if_main': bool(re.search(r'if __name__ == ["\']__main__["\']', content)),
            'has_imports': len(re.findall(r'^import |^from .+ import', content, re.MULTILINE)),
            'has_classes': len(re.findall(r'^class ', content, re.MULTILINE)),
            'has_functions': len(re.findall(r'^def ', content, re.MULTILINE)),
            'line_count': len(content.split('\n')),
            'has_docstrings': bool(re.search(r'"""[\s\S]*?"""', content)),
            
            # Problem patterns
            'forbidden_patterns': [],
            'architectural_violations': [],
            'deprecated_markers': [],
            'todo_markers': []
        }
        
        # Check for forbidden patterns (reasons for uncertainty)
        forbidden_checks = [
            ('boto3_direct', r'boto3\.client\('),
            ('psycopg2_direct', r'psycopg2\.connect\('),
            ('numpy_usage', r'import numpy|from numpy'),
            ('sklearn_usage', r'from sklearn|import sklearn'),
            ('sentence_transformers', r'sentence_transformers')
        ]
        
        for name, pattern in forbidden_checks:
            if re.search(pattern, content, re.IGNORECASE):
                analysis['forbidden_patterns'].append(name)
                
        # Check for architectural violations
        if 'session_manager' in content.lower() and 'unifiedsessionmanager' not in content:
            analysis['architectural_violations'].append('non_unified_session_manager')
            
        # Check for deprecated markers
        deprecated_patterns = [
            r'# ?deprecated',
            r'# ?todo.*deprecated',
            r'raise NotImplementedError',
            r'# ?fixme',
            r'# ?hack'
        ]
        
        for pattern in deprecated_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                analysis['deprecated_markers'].append(pattern)
                
        # Check for TODO markers
        todo_matches = re.findall(r'# ?todo.*', content, re.IGNORECASE)
        analysis['todo_markers'] = todo_matches[:5]  # First 5 TODOs
        
        return analysis
    
    def _make_recommendation(self, investigation: Dict, file_path: Path) -> Dict:
        """Make recommendation based on investigation"""
        reasoning = []
        confidence = 'medium'
        
        # Path-based analysis
        path_str = str(file_path).lower()
        
        if '/deprecated/' in path_str or '/old-' in path_str:
            recommendation = 'inactive_deprecated_location'
            reasoning.append('Located in deprecated directory')
            confidence = 'high'
        elif '/test' in path_str or path_str.endswith('test.py'):
            if investigation.get('has_functions', 0) > 0:
                recommendation = 'active_test_file'
                reasoning.append('Test file with functions')
            else:
                recommendation = 'inactive_empty_test'
                reasoning.append('Test file but no test functions')
        elif 'demo' in path_str:
            if investigation.get('has_main_function') and investigation.get('has_if_main'):
                recommendation = 'active_demo'
                reasoning.append('Demo with executable main')
            else:
                recommendation = 'uncertain_demo'
                reasoning.append('Demo but missing main execution')
                
        # Content-based analysis
        if investigation.get('forbidden_patterns'):
            recommendation = 'needs_migration'
            reasoning.append(f"Contains forbidden patterns: {', '.join(investigation['forbidden_patterns'])}")
            confidence = 'high'
        elif investigation.get('deprecated_markers'):
            recommendation = 'inactive_marked_deprecated'
            reasoning.append('Contains deprecation markers')
            confidence = 'high'
        elif investigation.get('has_functions', 0) == 0 and investigation.get('has_classes', 0) == 0:
            recommendation = 'inactive_no_functionality'
            reasoning.append('No functions or classes defined')
            confidence = 'high'
        elif investigation.get('has_main_function') and investigation.get('has_if_main'):
            recommendation = 'active_executable'
            reasoning.append('Has main function and execution guard')
            confidence = 'high'
        elif investigation.get('has_imports', 0) == 0:
            recommendation = 'inactive_no_imports'
            reasoning.append('No imports - likely empty or template')
            confidence = 'medium'
        elif investigation.get('todo_markers'):
            recommendation = 'uncertain_has_todos'
            reasoning.append(f"Has TODO items: {len(investigation['todo_markers'])} found")
            confidence = 'medium'
        else:
            recommendation = 'uncertain_needs_manual_review'
            reasoning.append('Mixed indicators - needs manual review')
            confidence = 'low'
            
        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'reasoning': reasoning
        }
    
    def generate_investigation_report(self) -> str:
        """Generate comprehensive investigation report"""
        print("[INVESTIGATING] Loading uncertain files from audit...")
        
        uncertain_files = self.load_uncertain_files()
        if not uncertain_files:
            return "No uncertain files found in audit CSV."
        
        print(f"[INVESTIGATING] Analyzing {len(uncertain_files)} uncertain files...")
        
        investigations = []
        for file_data in uncertain_files:
            investigation = self.investigate_file(file_data)
            investigations.append(investigation)
            
        # Generate statistics
        total = len(investigations)
        by_recommendation = {}
        by_confidence = {}
        
        for inv in investigations:
            rec = inv['recommendation']
            conf = inv['confidence']
            by_recommendation[rec] = by_recommendation.get(rec, 0) + 1
            by_confidence[conf] = by_confidence.get(conf, 0) + 1
            
        # Generate report
        report = f"""
=== UNCERTAIN FILES INVESTIGATION REPORT ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total uncertain files analyzed: {total}

[STATS] RECOMMENDATION BREAKDOWN:
"""
        for rec, count in sorted(by_recommendation.items()):
            report += f"- {rec}: {count} ({count/total*100:.1f}%)\n"
            
        report += f"""
[STATS] CONFIDENCE LEVELS:
"""
        for conf, count in sorted(by_confidence.items()):
            report += f"- {conf}: {count} ({count/total*100:.1f}%)\n"
            
        report += f"""
[DETAILED] INVESTIGATION RESULTS:
"""
        
        # Group by recommendation
        for recommendation in sorted(by_recommendation.keys()):
            matching_files = [inv for inv in investigations if inv['recommendation'] == recommendation]
            report += f"\n--- {recommendation.upper()} ({len(matching_files)} files) ---\n"
            
            for inv in matching_files[:10]:  # Show first 10 in each category
                report += f"- {inv['path']}\n"
                if inv['reasoning']:
                    report += f"  Reason: {'; '.join(inv['reasoning'])}\n"
                if inv.get('forbidden_patterns'):
                    report += f"  Violations: {', '.join(inv['forbidden_patterns'])}\n"
                    
            if len(matching_files) > 10:
                report += f"  ... and {len(matching_files) - 10} more files\n"
        
        report += f"""
[CLEANUP] ACTION ITEMS:

HIGH PRIORITY (Immediate Action):
"""
        high_priority = [inv for inv in investigations 
                        if inv['confidence'] == 'high' and 
                        ('inactive' in inv['recommendation'] or 'migration' in inv['recommendation'])]
        
        for inv in high_priority[:20]:
            action = 'MIGRATE' if 'migration' in inv['recommendation'] else 'DELETE'
            report += f"- {action}: {inv['path']} ({inv['recommendation']})\n"
            
        report += f"""
MEDIUM PRIORITY (Review Required):
"""
        medium_priority = [inv for inv in investigations 
                          if inv['confidence'] == 'medium' or 'uncertain' in inv['recommendation']]
        
        for inv in medium_priority[:20]:
            report += f"- REVIEW: {inv['path']} ({inv['recommendation']})\n"
        
        return report, investigations

def main():
    """Main execution function"""
    investigator = UncertainFileInvestigator()
    report_text, full_results = investigator.generate_investigation_report()
    
    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save CSV with investigation results
    csv_filename = f"uncertain_files_investigation_{timestamp}.csv"
    if full_results:
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['filename', 'path', 'recommendation', 'confidence', 'reasoning', 
                         'forbidden_patterns', 'has_main_function', 'has_functions', 'has_classes']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in full_results:
                row = {
                    'filename': result['filename'],
                    'path': result['path'],
                    'recommendation': result['recommendation'],
                    'confidence': result['confidence'],
                    'reasoning': '; '.join(result.get('reasoning', [])),
                    'forbidden_patterns': '; '.join(result.get('forbidden_patterns', [])),
                    'has_main_function': result.get('has_main_function', False),
                    'has_functions': result.get('has_functions', 0),
                    'has_classes': result.get('has_classes', 0)
                }
                writer.writerow(row)
    
    # Save text report
    report_filename = f"uncertain_investigation_report_{timestamp}.txt"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n[OK] Investigation complete! Reports saved as:")
    print(f"  - {csv_filename}")
    print(f"  - {report_filename}")

if __name__ == "__main__":
    main()