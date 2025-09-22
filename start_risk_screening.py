#!/usr/bin/env python3
"""
Start Risk Screening Process
===========================

This is the main script that starts the risk screening process.
Run this to generate screening_{date}.json files.
"""

import json
import os
from pathlib import Path
from datetime import datetime

# PathManager import with fallback
try:
    from core.utilities.path_manager import get_path_manager
except ImportError:
    try:
        from common.utilities.path_manager import get_path_manager
    except ImportError:
        def get_path_manager():
            class MockPathManager:
                @property
                def root_folder(self):
                    return os.getcwd()
            return MockPathManager()

def start_risk_screening():
    print('🚀 STARTING RISK SCREENING PROCESS')
    print('=' * 50)
    
    path_manager = get_path_manager()
    base_path = Path(path_manager.root_folder)
    results = {
        'screening_metadata': {
            'timestamp': datetime.now().isoformat(),
            'screening_date': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'base_path': str(base_path),
            'service_version': 'start-1.0.0'
        },
        'file_assessments': {},
        'directory_summaries': {},
        'risk_summary': {},
        'compliance_status': {}
    }
    
    total_files = 0
    high_risk_files = []
    untagged_files = []
    compliance_requirements = set()
    
    for dir_name in ['tidyllm', 'v2', 'onboarding', 'pending']:
        print('📁 Scanning ' + dir_name + '...')
        dir_path = base_path / dir_name
        
        if dir_path.exists():
            dir_summary = {
                'total_files': 0,
                'python_files': 0,
                'markdown_files': 0,
                'high_risk_files': [],
                'untagged_files': [],
                'compliance_requirements': []
            }
            
            try:
                # Scan Python files
                py_files = list(dir_path.rglob('*.py'))[:10]
                for py_file in py_files:
                    try:
                        content = py_file.read_text(encoding='utf-8', errors='ignore')[:1000]
                        content_lower = content.lower()
                        
                        # Risk assessment
                        if '@risk: high' in content_lower or 'risk level: high' in content_lower:
                            risk_level = 'HIGH'
                            high_risk_files.append(str(py_file))
                            dir_summary['high_risk_files'].append(str(py_file))
                        elif '@risk:' in content_lower or 'risk level:' in content_lower:
                            risk_level = 'TAGGED'
                        else:
                            risk_level = 'UNTAGGED'
                            untagged_files.append(str(py_file))
                            dir_summary['untagged_files'].append(str(py_file))
                        
                        # Compliance check
                        if 'sox' in content_lower:
                            compliance_requirements.add('SOX')
                            dir_summary['compliance_requirements'].append('SOX')
                        if 'pci' in content_lower:
                            compliance_requirements.add('PCI-DSS')
                            dir_summary['compliance_requirements'].append('PCI-DSS')
                        if 'gdpr' in content_lower:
                            compliance_requirements.add('GDPR')
                            dir_summary['compliance_requirements'].append('GDPR')
                        
                        results['file_assessments'][str(py_file)] = {
                            'file_type': 'python',
                            'risk_level': risk_level,
                            'directory': dir_name
                        }
                        
                        dir_summary['python_files'] += 1
                        dir_summary['total_files'] += 1
                        total_files += 1
                        
                    except Exception as e:
                        print('  ⚠️  Error reading ' + py_file.name)
                
                results['directory_summaries'][dir_name] = dir_summary
                print('  ✅ ' + str(dir_summary['total_files']) + ' files (Python: ' + str(dir_summary['python_files']) + ')')
                print('     High Risk: ' + str(len(dir_summary['high_risk_files'])) + ', Untagged: ' + str(len(dir_summary['untagged_files'])))
                
            except Exception as e:
                print('  ❌ Error scanning ' + dir_name)
        else:
            print('  ⏭️  Directory not found')
    
    # Generate summary
    results['risk_summary'] = {
        'total_files_assessed': total_files,
        'high_risk_files': len(high_risk_files),
        'untagged_files': len(untagged_files),
        'production_ready': len(high_risk_files) == 0
    }
    
    results['compliance_status'] = {
        'identified_frameworks': list(compliance_requirements),
        'framework_count': len(compliance_requirements)
    }
    
    # Save results
    output_file = 'screening_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print('\n📊 SUMMARY:')
    print('   Total Files: ' + str(total_files))
    print('   High Risk: ' + str(len(high_risk_files)))
    print('   Untagged: ' + str(len(untagged_files)))
    print('   Compliance Frameworks: ' + str(len(compliance_requirements)))
    print('   Production Ready: ' + ('YES' if len(high_risk_files) == 0 else 'NO'))
    
    print('\n✅ Risk screening results saved to: ' + output_file)
    return results

if __name__ == '__main__':
    start_risk_screening()
