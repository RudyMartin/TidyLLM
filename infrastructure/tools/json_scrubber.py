#!/usr/bin/env python3
"""
Batch JSON Scrubber for AI-Scoring Project
==========================================

Finds and scrubs all JSON files in the project to ensure ASCII compliance
and fix any JSON structure issues.
"""

import os
import json
from pathlib import Path
from tidyllm.utils.json_scrubber import safe_load_json_with_scrubbing

def find_all_json_files(directory="."):
    """Find all JSON files in the directory tree"""
    json_files = []
    for root, dirs, files in os.walk(directory):
        # Skip common irrelevant directories
        dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.pytest_cache',
                                               'node_modules', '.vscode', '.idea'}]

        for file in files:
            if file.endswith('.json'):
                json_files.append(Path(root) / file)

    return sorted(json_files)

def test_and_scrub_json_files():
    """Test and scrub all JSON files in the project"""
    print("[SEARCH] Finding all JSON files...")
    json_files = find_all_json_files()
    print(f"Found {len(json_files)} JSON files")

    results = {
        'total_files': len(json_files),
        'clean_files': 0,
        'scrubbed_files': 0,
        'failed_files': 0,
        'details': []
    }

    print("\n[TEST] Testing JSON files...")
    print("=" * 80)

    for i, file_path in enumerate(json_files, 1):
        print(f"[{i:2d}/{len(json_files)}] {file_path}")

        # Test loading with safe loader
        result = safe_load_json_with_scrubbing(str(file_path))

        file_result = {
            'file': str(file_path),
            'success': result['success']
        }

        if result['success']:
            if result.get('scrubbing_required'):
                print(f"    [WARNING] Required scrubbing")
                if result.get('structure_fixes_required'):
                    print(f"    [REPAIR] Structure fixes applied")

                # Write back the cleaned version
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(result['data'], f, indent=2, ensure_ascii=True)
                    print(f"    [OK] Cleaned and saved")
                    results['scrubbed_files'] += 1
                    file_result['scrubbed'] = True
                    file_result['fixes'] = result.get('fixes_applied', [])
                except Exception as e:
                    print(f"    [ERROR] Failed to save: {e}")
                    results['failed_files'] += 1
                    file_result['error'] = str(e)
            else:
                print(f"    [OK] Already clean")
                results['clean_files'] += 1
                file_result['clean'] = True
        else:
            print(f"    [ERROR] Failed: {result['error']}")
            results['failed_files'] += 1
            file_result['error'] = result['error']

        results['details'].append(file_result)
        print()

    return results

def main():
    """Main batch scrubbing function"""
    print("TidyLLM Batch JSON Scrubber")
    print("=" * 40)

    results = test_and_scrub_json_files()

    print("[SUMMARY] SUMMARY RESULTS")
    print("=" * 40)
    print(f"Total files processed: {results['total_files']}")
    print(f"Clean files (no changes): {results['clean_files']}")
    print(f"Scrubbed files (fixed): {results['scrubbed_files']}")
    print(f"Failed files (errors): {results['failed_files']}")

    success_rate = (results['clean_files'] + results['scrubbed_files']) / results['total_files'] * 100
    print(f"Success rate: {success_rate:.1f}%")

    if results['failed_files'] > 0:
        print(f"\n[ERRORS] Failed files:")
        for detail in results['details']:
            if 'error' in detail:
                print(f"  - {detail['file']}: {detail['error']}")

    if results['scrubbed_files'] > 0:
        print(f"\n[REPAIRS] Files that were scrubbed:")
        for detail in results['details']:
            if detail.get('scrubbed'):
                print(f"  - {detail['file']}")
                if 'fixes' in detail and detail['fixes']:
                    for fix in detail['fixes']:
                        print(f"    * {fix}")

    print(f"\n[COMPLETE] Batch scrubbing complete!")
    return results

if __name__ == "__main__":
    main()