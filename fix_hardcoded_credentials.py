#!/usr/bin/env python3
"""
Batch Fix Hardcoded Credentials
===============================

Systematically replace hardcoded AWS credentials across all TidyLLM scripts
with centralized credential management system.
"""

import os
import re
from pathlib import Path


def get_files_with_hardcoded_credentials():
    """Get list of files that need credential fixing"""
    
    files_to_fix = [
        # Root level scripts
        "flow_modeling_upload.py",
        "flow_sop_upload.py", 
        "flow_checklist_upload.py",
        "one_click_domain_rag_builder.py",
        "test_sop_docs_processing.py",
        "test_aws_connection.py",
        "risk_management_sop_drop_zone.py",
        "sop_domain_rag_tidyllm.py",
        "create_sop_domain_flow.py",
        "start_production_with_aws.py",
        
        # Scripts directory
        "scripts/unified_credential_setup.py",
        "scripts/demo_file_upload_app.py", 
        "scripts/check_s3_papers.py",
        
        # Knowledge systems
        "tidyllm/knowledge_systems/create_domain_workflow.py",
        "tidyllm/knowledge_systems/true_s3_first_domain_rag.py"
    ]
    
    # Filter to only existing files
    existing_files = []
    for file_path in files_to_fix:
        full_path = Path(__file__).parent / file_path
        if full_path.exists():
            existing_files.append(full_path)
    
    return existing_files


def create_credential_import_code():
    """Generate the standard credential import code"""
    
    return '''import sys
from pathlib import Path

# Add admin directory to path for credential loading
sys.path.append(str(Path(__file__).parent.parent / 'tidyllm' / 'admin') if 'tidyllm' in str(Path(__file__)) else str(Path(__file__).parent / 'tidyllm' / 'admin'))
from credential_loader import set_aws_environment

# Load AWS credentials using centralized system
set_aws_environment()'''


def fix_file_credentials(file_path):
    """Fix hardcoded credentials in a single file"""
    
    print(f"[FIX] Processing: {file_path}")
    
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Remove hardcoded credential assignments
        patterns_to_remove = [
            r"os\.environ\['AWS_ACCESS_KEY_ID'\]\s*=\s*['\"][^'\"]+['\"]",
            r"os\.environ\['AWS_SECRET_ACCESS_KEY'\]\s*=\s*['\"][^'\"]+['\"]", 
            r"os\.environ\['AWS_DEFAULT_REGION'\]\s*=\s*['\"][^'\"]+['\"]",
            r"os\.environ\[\"AWS_ACCESS_KEY_ID\"\]\s*=\s*['\"][^'\"]+['\"]",
            r"os\.environ\[\"AWS_SECRET_ACCESS_KEY\"\]\s*=\s*['\"][^'\"]+['\"]",
            r"os\.environ\[\"AWS_DEFAULT_REGION\"\]\s*=\s*['\"][^'\"]+['\"]"
        ]
        
        changes_made = False
        for pattern in patterns_to_remove:
            old_content = content
            content = re.sub(pattern, '', content)
            if content != old_content:
                changes_made = True
        
        # Look for import section to insert credential loading
        import_section_end = None
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            # Find the end of the import section
            if line.strip() and not (line.strip().startswith('import ') or 
                                   line.strip().startswith('from ') or
                                   line.strip().startswith('#') or
                                   line.strip().startswith('"""') or
                                   line.strip().startswith("'''")):
                import_section_end = i
                break
        
        # If we found where imports end and we made changes, add credential loading
        if changes_made and import_section_end is not None:
            # Insert credential loading code
            credential_code = create_credential_import_code()
            lines.insert(import_section_end, '')
            lines.insert(import_section_end + 1, '# Centralized AWS credential management')
            for line in reversed(credential_code.split('\n')):
                lines.insert(import_section_end + 2, line)
            
            content = '\n'.join(lines)
        
        # Write the file back if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  [SUCCESS] Fixed credentials in {file_path.name}")
            return True
        else:
            print(f"  [SKIP] No hardcoded credentials found in {file_path.name}")
            return False
            
    except Exception as e:
        print(f"  [ERROR] Failed to fix {file_path.name}: {e}")
        return False


def main():
    """Fix hardcoded credentials across all TidyLLM scripts"""
    
    print("=" * 80)
    print("BATCH FIX HARDCODED CREDENTIALS")
    print("=" * 80)
    print("Replacing hardcoded AWS credentials with centralized credential management")
    
    files_to_fix = get_files_with_hardcoded_credentials()
    
    print(f"\n[SCAN] Found {len(files_to_fix)} files to process:")
    for file_path in files_to_fix:
        print(f"  - {file_path.name}")
    
    print(f"\n[PROCESSING] Fixing credentials...")
    
    fixed_count = 0
    for file_path in files_to_fix:
        if fix_file_credentials(file_path):
            fixed_count += 1
    
    print(f"\n[SUMMARY] Processing complete:")
    print(f"  Files processed: {len(files_to_fix)}")
    print(f"  Files modified: {fixed_count}")
    print(f"  Files skipped: {len(files_to_fix) - fixed_count}")
    
    if fixed_count > 0:
        print(f"\n[SUCCESS] Hardcoded credentials removed from {fixed_count} files!")
        print(f"[SUCCESS] All files now use centralized credential management")
    else:
        print(f"\n[INFO] No hardcoded credentials found to fix")


if __name__ == "__main__":
    main()