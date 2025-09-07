#!/usr/bin/env python3
"""
Final Credential Cleanup
=======================

Remove all remaining hardcoded credentials except from legitimate admin files.
"""

import re
from pathlib import Path


def fix_remaining_credentials():
    """Fix all remaining hardcoded credentials"""
    
    # Get all Python files with hardcoded credentials
    credential_pattern = r'REMOVED_AWS_KEY|xGMWglS3y6LbOk\+uDMrb6rFwRrCpJBgzMrhoWtYk'
    
    # Files that should keep credentials (legitimate admin files)
    keep_credentials = {
        'tidyllm/admin/set_aws_credentials.py',
        'tidyllm/admin/set_aws_env.bat',
        'tidyllm/admin/set_aws_env.sh', 
        'tidyllm/admin/set_aws_credentials.bat',
        'set_aws_credentials.bat'
    }
    
    python_files = Path('.').rglob('*.py')
    
    fixed_files = []
    
    for file_path in python_files:
        # Skip admin credential files
        if any(str(file_path).endswith(keep) for keep in keep_credentials):
            continue
            
        # Skip if file doesn't exist or can't be read
        if not file_path.exists():
            continue
            
        try:
            content = file_path.read_text(encoding='utf-8')
            original_content = content
            
            # Check if file has hardcoded credentials
            if not re.search(credential_pattern, content):
                continue
            
            print(f"[FIX] Processing: {file_path}")
            
            # Remove hardcoded credential assignments
            patterns_to_remove = [
                r"os\.environ\[['\"]AWS_ACCESS_KEY_ID['\"]\]\s*=\s*['\"][^'\"]+['\"]",
                r"os\.environ\[['\"]AWS_SECRET_ACCESS_KEY['\"]\]\s*=\s*['\"][^'\"]+['\"]",
                r"os\.environ\[['\"]AWS_DEFAULT_REGION['\"]\]\s*=\s*['\"][^'\"]+['\"]",
                r"['\"]AWS_ACCESS_KEY_ID['\"]\s*:\s*['\"][^'\"]+['\"]",
                r"['\"]AWS_SECRET_ACCESS_KEY['\"]\s*:\s*['\"][^'\"]+['\"]",
                r"['\"]AWS_DEFAULT_REGION['\"]\s*:\s*['\"][^'\"]+['\"]"
            ]
            
            for pattern in patterns_to_remove:
                content = re.sub(pattern, '# Credentials loaded by centralized system', content)
            
            # Clean up duplicate comments and empty lines
            content = re.sub(r'# Credentials loaded by centralized system\s*\n\s*# Credentials loaded by centralized system', 
                           '# Credentials loaded by centralized system', content)
            
            # Write back if changed
            if content != original_content:
                file_path.write_text(content, encoding='utf-8')
                print(f"  [SUCCESS] Cleaned credentials from {file_path}")
                fixed_files.append(str(file_path))
            
        except Exception as e:
            print(f"  [ERROR] Failed to process {file_path}: {e}")
    
    return fixed_files


def main():
    """Clean up all remaining hardcoded credentials"""
    
    print("=" * 60)
    print("FINAL CREDENTIAL CLEANUP")
    print("=" * 60)
    
    fixed_files = fix_remaining_credentials()
    
    print(f"\n[SUMMARY] Final cleanup complete:")
    print(f"  Files fixed: {len(fixed_files)}")
    
    if fixed_files:
        print(f"  Fixed files:")
        for file_path in fixed_files:
            print(f"    - {file_path}")
    
    print(f"\n[VERIFICATION] Checking remaining hardcoded credentials...")
    
    # Count remaining credentials
    remaining_files = []
    credential_pattern = r'REMOVED_AWS_KEY'
    
    for file_path in Path('.').rglob('*.py'):
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            if re.search(credential_pattern, content):
                # Check if it's a legitimate admin file
                is_admin_file = any(str(file_path).endswith(keep) for keep in [
                    'set_aws_credentials.py', 'set_aws_env.bat', 'set_aws_env.sh', 
                    'set_aws_credentials.bat', 'credential_loader.py', 'restart_aws_session.py'
                ])
                
                if not is_admin_file:
                    remaining_files.append(str(file_path))
        except:
            pass
    
    if remaining_files:
        print(f"  [WARNING] {len(remaining_files)} files still have hardcoded credentials:")
        for file_path in remaining_files:
            print(f"    - {file_path}")
    else:
        print(f"  [SUCCESS] No hardcoded credentials found in application files!")
        print(f"  [SUCCESS] Only legitimate admin credential files retain credentials")


if __name__ == "__main__":
    main()