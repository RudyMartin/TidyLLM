#!/usr/bin/env python3
import json
import time
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

def simple_scan():
    print("🚀 SIMPLE RISK SCREENING")
    print("=" * 40)
    
    path_manager = get_path_manager()
    base_path = Path(path_manager.root_folder)
    results = {
        'timestamp': datetime.now().isoformat(),
        'directories': {},
        'summary': {}
    }
    
    dirs = ['tidyllm', 'v2', 'onboarding', 'pending']
    
    for dir_name in dirs:
        print(f"📁 Scanning {dir_name}...")
        dir_path = base_path / dir_name
        
        if dir_path.exists():
            try:
                # Quick scan - just count files
                py_files = list(dir_path.rglob('*.py'))[:10]  # Limit to 10
                md_files = list(dir_path.rglob('*.md'))[:5]   # Limit to 5
                
                results['directories'][dir_name] = {
                    'python_files': len(py_files),
                    'markdown_files': len(md_files),
                    'sample_files': [str(f) for f in py_files[:3]]
                }
                
                print(f"  ✅ Found {len(py_files)} Python, {len(md_files)} Markdown")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                results['directories'][dir_name] = {'error': str(e)}
        else:
            print(f"  ⏭️  Not found")
            results['directories'][dir_name] = {'status': 'not_found'}
    
    # Save results
    output_file = f"simple_screening_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    return results

if __name__ == "__main__":
    simple_scan()
