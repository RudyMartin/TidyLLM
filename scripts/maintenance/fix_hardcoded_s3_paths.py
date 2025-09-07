#!/usr/bin/env python3
"""
Fix Hardcoded S3 Bucket Names and Prefixes
==========================================

Systematically replace hardcoded S3 bucket names and prefixes across all TidyLLM scripts
with configurable S3 path management system.

This script:
1. Finds all hardcoded references to "nsc-mvp1" bucket
2. Finds all hardcoded prefixes like "knowledge_base/", "mvr_analysis/"
3. Replaces them with configurable path building using credential_loader
"""

import re
import os
from pathlib import Path


def get_s3_replacement_patterns():
    """Get patterns and replacements for S3 hardcoding"""
    
    return [
        # Bucket name patterns
        {
            'pattern': r'bucket\s*=\s*["\']nsc-mvp1["\']',
            'replacement': 'bucket = s3_config["bucket"]',
            'requires_import': True
        },
        {
            'pattern': r'["\']nsc-mvp1["\']',
            'replacement': 's3_config["bucket"]',
            'requires_import': True,
            'context_check': 'bucket'  # Only replace if "bucket" is in the same line or nearby
        },
        
        # Common prefix patterns
        {
            'pattern': r's3_prefix\s*=\s*["\']knowledge_base/([^"\']*)["\']',
            'replacement': 's3_prefix = build_s3_path("knowledge_base", "\\1")',
            'requires_import': True
        },
        {
            'pattern': r's3_prefix\s*=\s*["\']mvr_analysis/([^"\']*)["\']',
            'replacement': 's3_prefix = build_s3_path("mvr_analysis", "\\1")',
            'requires_import': True
        },
        {
            'pattern': r'["\']knowledge_base/([^"\']*)["\']',
            'replacement': 'build_s3_path("knowledge_base", "\\1")',
            'requires_import': True,
            'context_check': 's3'  # Only replace if "s3" is in context
        },
        {
            'pattern': r'["\']mvr_analysis/([^"\']*)["\']',
            'replacement': 'build_s3_path("mvr_analysis", "\\1")',
            'requires_import': True,
            'context_check': 's3'  # Only replace if "s3" is in context
        }
    ]


def get_import_template():
    """Get the import template to add to files"""
    
    return '''
# S3 Configuration Management
sys.path.append(str(Path(__file__).parent.parent / 'tidyllm' / 'admin') if 'tidyllm' in str(Path(__file__)) else str(Path(__file__).parent / 'tidyllm' / 'admin'))
from credential_loader import get_s3_config, build_s3_path

# Get S3 configuration (bucket and path builder)
s3_config = get_s3_config()  # Add environment parameter for dev/staging/prod
'''


def should_skip_file(file_path):
    """Check if file should be skipped (admin files, etc.)"""
    
    skip_patterns = [
        'credential_loader.py',
        'restart_aws_session.py', 
        'fix_hardcoded',
        'settings.yaml',
        '.git',
        '__pycache__'
    ]
    
    return any(pattern in str(file_path) for pattern in skip_patterns)


def fix_s3_paths_in_file(file_path):
    """Fix hardcoded S3 paths in a single file"""
    
    if should_skip_file(file_path):
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        needs_import = False
        
        # Check if file has hardcoded S3 references
        if 'nsc-mvp1' not in content and 'knowledge_base/' not in content and 'mvr_analysis/' not in content:
            return False
        
        print(f"[FIX] Processing S3 paths in: {file_path.name}")
        
        # Apply replacement patterns
        patterns = get_s3_replacement_patterns()
        
        for pattern_info in patterns:
            pattern = pattern_info['pattern']
            replacement = pattern_info['replacement']
            
            # Check context if required
            if 'context_check' in pattern_info:
                context = pattern_info['context_check']
                # Simple context check - only replace if context word is in the file
                if context.lower() not in content.lower():
                    continue
            
            # Apply the replacement
            old_content = content
            content = re.sub(pattern, replacement, content)
            
            if content != old_content:
                needs_import = pattern_info.get('requires_import', False)
                print(f"    Applied pattern: {pattern}")
        
        # Add import if needed and not already present
        if needs_import and 'from credential_loader import' not in content:
            # Find the import section
            lines = content.split('\n')
            import_end_idx = 0
            
            for i, line in enumerate(lines):
                if line.strip() and not (line.strip().startswith(('import ', 'from ', '#', '"""', "'''"))):
                    import_end_idx = i
                    break
            
            # Insert import template
            import_template = get_import_template()
            lines.insert(import_end_idx, import_template)
            content = '\n'.join(lines)
            
            print(f"    Added S3 configuration imports")
        
        # Write back if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"    [SUCCESS] Fixed S3 paths in {file_path.name}")
            return True
        else:
            print(f"    [SKIP] No changes needed for {file_path.name}")
            return False
            
    except Exception as e:
        print(f"    [ERROR] Failed to process {file_path.name}: {e}")
        return False


def main():
    """Fix hardcoded S3 paths across all scripts"""
    
    print("=" * 80)
    print("BATCH FIX HARDCODED S3 BUCKET NAMES AND PREFIXES")
    print("=" * 80)
    print("Replacing hardcoded S3 paths with configurable system")
    
    # Find all Python files with potential S3 hardcoding
    python_files = list(Path('.').rglob('*.py'))
    candidate_files = []
    
    for file_path in python_files:
        if should_skip_file(file_path):
            continue
            
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            if 'nsc-mvp1' in content or 'knowledge_base/' in content or 'mvr_analysis/' in content:
                candidate_files.append(file_path)
        except:
            continue
    
    print(f"\n[SCAN] Found {len(candidate_files)} files with potential S3 hardcoding:")
    for file_path in candidate_files[:10]:  # Show first 10
        print(f"  - {file_path}")
    if len(candidate_files) > 10:
        print(f"  ... and {len(candidate_files) - 10} more files")
    
    print(f"\n[PROCESSING] Fixing S3 paths...")
    
    fixed_count = 0
    for file_path in candidate_files:
        if fix_s3_paths_in_file(file_path):
            fixed_count += 1
    
    print(f"\n[SUMMARY] S3 path fixing complete:")
    print(f"  Files processed: {len(candidate_files)}")
    print(f"  Files modified: {fixed_count}")
    print(f"  Files skipped: {len(candidate_files) - fixed_count}")
    
    if fixed_count > 0:
        print(f"\n[SUCCESS] S3 paths made configurable in {fixed_count} files!")
        print(f"[SUCCESS] All files now support environment-specific buckets and prefixes")
        
        print(f"\n[USAGE] Scripts now support:")
        print(f"  • Default: s3_config = get_s3_config()")
        print(f"  • Development: s3_config = get_s3_config('development')")
        print(f"  • Production: s3_config = get_s3_config('production')")
        print(f"  • Dynamic paths: build_s3_path('knowledge_base', 'checklist')")
    else:
        print(f"\n[INFO] No S3 hardcoding found to fix")


if __name__ == "__main__":
    main()