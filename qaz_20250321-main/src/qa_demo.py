# -*- coding: utf-8 -*-
"""
QA Document Processing Demo

Streamlit interface for QA document processing using MCP framework.
"""

# Environment setup
from config.setup import setup_env
setup_env()

import streamlit as st
import zipfile
import tempfile
from typing import List, Dict, Any
from datetime import datetime

from backend.mcp.orchestrators.qa_orchestrator import QAOrchestrator
import re

# Use basic Streamlit file upload (simpler and more reliable)
ATTACHMENTS_AVAILABLE = False


def extract_zip_safely(zip_file, max_size_mb=50, max_files=100):
    """
    Safely extract ZIP file with comprehensive security checks
    
    Args:
        zip_file: UploadedFile object
        max_size_mb: Maximum total size in MB
        max_files: Maximum number of files to extract
    
    Returns:
        List of extracted file data or error message
    """
    # Define dangerous file extensions
    DANGEROUS_EXTENSIONS = {
        # Executables
        '.exe', '.bat', '.cmd', '.com', '.scr', '.pif', '.vbs', '.js', '.jar',
        # Scripts
        '.py', '.sh', '.bash', '.zsh', '.csh', '.ksh', '.pl', '.rb', '.php',
        # System files
        '.dll', '.so', '.dylib', '.sys', '.drv', '.ocx', '.cpl',
        # Archives that could contain more dangerous content
        '.tar', '.gz', '.bz2', '.7z', '.rar', '.cab',
        # Other potentially dangerous
        '.reg', '.inf', '.msi', '.msp', '.mst', '.chm', '.hta', '.wsf',
        # Network/communication files
        '.url', '.lnk', '.desktop', '.app', '.command'
    }
    
    # Define allowed file extensions
    ALLOWED_EXTENSIONS = {
        # Documents
        '.pdf', '.docx', '.txt', '.csv', '.xlsx', '.md', '.doc', '.rtf',
        # Data formats
        '.log',
        # Images
        '.gif', '.jpg', '.jpeg', '.png', '.svg', '.bmp', '.tiff', '.webp',
        # Code and Analysis
        '.py', '.ipynb', '.r', '.sql', '.sas', '.mat', '.m',
        # Statistical and Data
        '.sav', '.dta', '.rds', '.rdata', '.parquet', '.feather', '.h5', '.hdf5',
        # Configuration and Metadata
        '.toml', '.ini', '.cfg', '.conf',
        # Reports and Outputs
        '.tex', '.bib', '.ris', '.enw'
    }
    
    try:
        # Check file size
        zip_file.seek(0, 2)  # Seek to end
        file_size = zip_file.tell()
        zip_file.seek(0)  # Reset to beginning
        
        if file_size > max_size_mb * 1024 * 1024:
            return f"Error: ZIP file too large (max {max_size_mb}MB)"
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            extracted_files = []
            file_count = 0
            blocked_files = []
            
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                # Check for zip bomb (too many files)
                file_list = zip_ref.namelist()
                if len(file_list) > max_files:
                    return f"Error: Too many files in ZIP (max {max_files})"
                
                # Extract files
                for file_info in zip_ref.infolist():
                    if file_count >= max_files:
                        break
                    
                    # Skip directories
                    if file_info.is_dir():
                        continue
                    
                    # Check for path traversal attacks
                    file_path = file_info.filename
                    if '..' in file_path or file_path.startswith('/') or '\\' in file_path:
                        blocked_files.append(f"{file_path} (path traversal)")
                        continue
                    
                    # Check file extension
                    file_ext = os.path.splitext(file_path)[1].lower()
                    
                    # Block dangerous files
                    if file_ext in DANGEROUS_EXTENSIONS:
                        blocked_files.append(f"{file_path} (dangerous extension: {file_ext})")
                        continue
                    
                    # Only allow specific file types
                    if file_ext not in ALLOWED_EXTENSIONS:
                        blocked_files.append(f"{file_path} (unsupported extension: {file_ext})")
                        continue
                    
                    # Check file size (individual file limit)
                    if file_info.file_size > 10 * 1024 * 1024:  # 10MB per file
                        blocked_files.append(f"{file_path} (file too large: {file_info.file_size} bytes)")
                        continue
                    
                    # Extract file
                    try:
                        zip_ref.extract(file_info, temp_dir)
                        extracted_path = os.path.join(temp_dir, file_path)
                        
                        # Read file content
                        with open(extracted_path, 'rb') as f:
                            content = f.read()
                        
                        # Additional content checks
                        if len(content) > 10 * 1024 * 1024:  # 10MB content limit
                            blocked_files.append(f"{file_path} (content too large)")
                            continue
                        
                        # Check file signatures (magic bytes) for executable files
                        file_signatures = {
                            b'MZ': 'Windows executable',
                            b'\x7fELF': 'ELF executable',
                            b'#!/': 'Shell script',
                            b'<?php': 'PHP script',
                            b'<%': 'ASP script',
                            b'PK\x03\x04': 'ZIP archive',
                            b'Rar!': 'RAR archive',
                            b'\x1f\x8b\x08': 'GZIP archive',
                            b'\x37\x7A\x68': '7-Zip archive',
                        }
                        
                        # Check first few bytes for executable signatures
                        for signature, description in file_signatures.items():
                            if content.startswith(signature):
                                blocked_files.append(f"{file_path} (executable signature: {description})")
                                break
                        else:
                            # Check for suspicious content patterns
                            content_str = content.decode('utf-8', errors='ignore').lower()
                        suspicious_patterns = [
                            '#!/bin/bash', '#!/bin/sh', '#!/usr/bin/python',
                            'exec(', 'eval(', 'system(', 'subprocess.',
                            'os.system', 'shell_exec', 'passthru',
                            'cmd.exe', 'powershell', 'wscript',
                            'regsvr32', 'rundll32', 'certutil',
                            # Executable file signatures
                            'mz',  # DOS/Windows executable header
                            'pe\x00\x00',  # PE executable header
                            'elf',  # ELF executable header
                            '#!/',  # Shebang scripts
                            '<?php',  # PHP scripts
                            '<%',  # ASP scripts
                            'javascript:',  # JavaScript execution
                            'vbscript:',  # VBScript execution
                            'data:text/html',  # Data URLs
                            'data:application/x-javascript',  # JavaScript data URLs
                            # XML/HTML specific threats
                            '<!entity',  # XXE attack
                            '<!doctype',  # XXE attack
                            '<?xml-stylesheet',  # XXE attack
                            '<?xml version',  # XXE attack
                            '<script',  # XSS attack
                            'onclick=',  # XSS attack
                            'onload=',  # XSS attack
                            'onerror=',  # XSS attack
                            'javascript:',  # XSS attack
                            'vbscript:',  # XSS attack
                            'data:',  # Data URL attacks
                            # YAML specific threats
                            '!!python/object',  # YAML code execution
                            '!!python/name',  # YAML code execution
                            '!!python/module',  # YAML code execution
                            '!!python/function',  # YAML code execution
                            '!!python/apply',  # YAML code execution
                            '!!python/eval',  # YAML code execution
                            '!!python/exec',  # YAML code execution
                            '!!python/import',  # YAML code execution
                            '!!python/new',  # YAML code execution
                            '!!python/object/apply',  # YAML code execution
                            '!!python/object/new',  # YAML code execution
                            # JSON specific threats
                            '__proto__',  # JSON prototype pollution
                            'constructor',  # JSON prototype pollution
                            'prototype',  # JSON prototype pollution
                        ]
                        
                        for pattern in suspicious_patterns:
                            if pattern in content_str:
                                blocked_files.append(f"{file_path} (suspicious content: {pattern})")
                                break
                        else:
                            # Validate file extension matches content
                            is_valid, reason = validate_file_extension(file_path, content)
                            if not is_valid:
                                blocked_files.append(f"{file_path} ({reason})")
                                continue
                            
                            # File passed all checks
                            extracted_files.append({
                                'name': os.path.basename(file_path),
                                'content': content,
                                'type': file_ext[1:],  # Remove the dot
                                'size': len(content)
                            })
                            file_count += 1
                        
                    except Exception as e:
                        blocked_files.append(f"{file_path} (extraction error: {str(e)})")
                        continue
            
            # Report blocked files if any
            if blocked_files:
                st.warning(f"⚠️ **Security Alert**: {len(blocked_files)} files were blocked:")
                for blocked in blocked_files[:5]:  # Show first 5
                    st.warning(f"  - {blocked}")
                if len(blocked_files) > 5:
                    st.warning(f"  - ... and {len(blocked_files) - 5} more files")
            
            return extracted_files if extracted_files else "Error: No valid files found in ZIP"
            
    except Exception as e:
        return f"Error extracting ZIP file: {str(e)}"


def validate_file_content(file_path, content):
    """
    Comprehensive content validation - checks actual file type vs extension
    
    Args:
        file_path: File path/name
        content: File content bytes
    
    Returns:
        (is_valid, reason)
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # Comprehensive file signatures (magic bytes) for content validation
    content_signatures = {
        # Images
        '.jpg': [b'\xff\xd8\xff'],
        '.jpeg': [b'\xff\xd8\xff'],
        '.png': [b'\x89PNG\r\n\x1a\n'],
        '.gif': [b'GIF87a', b'GIF89a'],
        '.bmp': [b'BM'],
        '.tiff': [b'II*\x00', b'MM\x00*'],
        '.webp': [b'RIFF'],
        '.svg': [b'<svg', b'<?xml'],
        
        # Documents
        '.pdf': [b'%PDF'],
        '.zip': [b'PK\x03\x04'],
        '.docx': [b'PK\x03\x04'],  # DOCX is actually a ZIP
        '.xlsx': [b'PK\x03\x04'],  # XLSX is actually a ZIP
        '.doc': [b'\xd0\xcf\x11\xe0'],  # OLE2 compound document
        '.xls': [b'\xd0\xcf\x11\xe0'],  # OLE2 compound document
        
        # Text-based formats
        '.xml': [b'<?xml', b'<'],
        '.html': [b'<!DOCTYPE', b'<html', b'<HTML'],
        '.htm': [b'<!DOCTYPE', b'<html', b'<HTML'],
        '.json': [b'{', b'['],
        '.yaml': [b'---', b'#'],
        '.yml': [b'---', b'#'],
        '.toml': [b'#', b'['],
        '.ini': [b'[', b'#', b';'],
        '.cfg': [b'[', b'#', b';'],
        '.conf': [b'[', b'#', b';'],
        '.tex': [b'\\documentclass', b'\\begin{document}'],
        '.bib': [b'@'],
        
        # Code files
        '.py': [b'#!/usr/bin/python', b'#!/usr/bin/env python', b'import ', b'def ', b'class '],
        '.r': [b'#!/usr/bin/R', b'#!/usr/bin/env R', b'library(', b'function(', b'<-'],
        '.sql': [b'SELECT', b'INSERT', b'UPDATE', b'DELETE', b'CREATE', b'DROP', b'ALTER'],
        '.mat': [b'MATLAB'],  # MATLAB files
        '.m': [b'function', b'classdef', b'%'],  # MATLAB/Objective-C
        '.sas': [b'DATA', b'PROC', b'LIBNAME', b'OPTIONS'],
        
        # Statistical data
        '.sav': [b'$FL2'],  # SPSS files
        '.dta': [b'<stata_dta>'],  # Stata files
        '.rds': [b'RDA2'],  # R data files
        '.rdata': [b'RDA2'],  # R data files
        '.parquet': [b'PAR1'],  # Parquet files
        '.feather': [b'FEA1'],  # Feather files
        '.h5': [b'\x89HDF\r\n\x1a\n'],  # HDF5 files
        '.hdf5': [b'\x89HDF\r\n\x1a\n'],  # HDF5 files
        
        # Archives
        '.tar': [b'ustar', b'gzip'],
        '.gz': [b'\x1f\x8b\x08'],
        '.bz2': [b'BZ'],
        '.7z': [b'7z\xbc\xaf\x27\x1c'],
        '.rar': [b'Rar!'],
        '.cab': [b'MSCF'],
    }
    
    # Check if file extension has expected signatures
    if file_ext in content_signatures:
        expected_sigs = content_signatures[file_ext]
        if expected_sigs:  # If we have specific signatures to check
            content_start = content[:200].lower()  # Check first 200 bytes
            
            # Check if any expected signature matches
            signature_found = False
            for sig in expected_sigs:
                if sig.lower() in content_start:
                    signature_found = True
                    break
            
            if not signature_found:
                return False, f"File extension {file_ext} doesn't match content signature"
    
    # Additional content-specific validations
    if file_ext in ['.xml', '.html', '.htm']:
        # Check for XXE attacks in XML
        if b'<!entity' in content.lower() or b'<!doctype' in content.lower():
            return False, f"File contains XML external entity declarations (XXE attack)"
        
        # Check for script tags in HTML
        if b'<script' in content.lower():
            return False, f"File contains script tags (potential XSS)"
    
    elif file_ext in ['.json']:
        # Check for JSON prototype pollution
        if b'__proto__' in content.lower() or b'constructor' in content.lower():
            return False, f"File contains prototype pollution patterns"
    
    elif file_ext in ['.yaml', '.yml']:
        # Check for YAML code execution
        yaml_dangerous = [b'!!python/object', b'!!python/name', b'!!python/module', 
                         b'!!python/function', b'!!python/apply', b'!!python/eval']
        for pattern in yaml_dangerous:
            if pattern in content.lower():
                return False, f"File contains YAML code execution patterns"
    
    elif file_ext in ['.py']:
        # Check for dangerous Python imports
        dangerous_imports = [b'import os', b'import subprocess', b'import sys',
                           b'from os import', b'from subprocess import', b'from sys import']
        for imp in dangerous_imports:
            if imp in content.lower():
                return False, f"File contains potentially dangerous Python imports"
    
    return True, "Valid file content"


def calculate_file_hash(content):
    """
    Calculate SHA-256 hash of file content for duplicate detection
    
    Args:
        content: File content bytes
    
    Returns:
        SHA-256 hash string
    """
    import hashlib
    return hashlib.sha256(content).hexdigest()


def validate_file_extension(file_path, content):
    """
    Legacy function - now calls validate_file_content
    """
    return validate_file_content(file_path, content)


def validate_review_id(review_id):
    """Validate Review ID format (REVXXXXX)"""
    if not review_id:
        return False, "Review ID is required"
    
    pattern = r'^REV\d{5}$'
    if not re.match(pattern, review_id):
        return False, "Review ID must be in format REVXXXXX (example: REV00001)"
    
    return True, "Valid Review ID"


def main():
    """Main Streamlit application"""
    
    st.set_page_config(
        page_title="QA Document Processing Demo",
        page_icon="📋",
        layout="wide"
    )
    
    st.title("📋 QA Document Processing Demo")
    
    # Navigation links to other apps
    st.markdown("### 🧭 Navigation")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🏠 Main App (8-Tab Interface)", use_container_width=True):
            st.switch_page("main.py")
    
    with col2:
        if st.button("📊 MCP Dashboard", use_container_width=True):
            st.switch_page("mcp_dashboard.py")
    
    with col3:
        if st.button("📈 Model Evaluation Dashboard", use_container_width=True):
            st.switch_page("model_eval_dashboard.py")
    
    st.markdown("---")
    
    # Initialize session state
    if 'qa_orchestrator' not in st.session_state:
        st.session_state.qa_orchestrator = QAOrchestrator()
    
    if 'workflow_result' not in st.session_state:
        st.session_state.workflow_result = None
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Only show Review ID field initially
        review_id = st.text_input(
            "Review ID", 
            placeholder="REV00001",
            help="Enter review number in format REVXXXXX (example: REV00001)"
        )
        
        # Validate Review ID format
        if review_id:
            is_valid, message = validate_review_id(review_id)
            if not is_valid:
                st.error(message)
            else:
                st.success("✅ Valid Review ID format")
        
        # Hidden fields with default values (only shown after processing)
        team_num = "QA Team 1"
        process_name = "QA Validation Review"
        reviewer_name = "Alex"
        model_type = "Machine Learning"  # Default value
        risk_tier = "Medium"  # Default value
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📁 Document Upload")
        
        # Prompt input
        st.subheader("💬 Custom Prompt (Optional)")
        custom_prompt = st.text_area(
            "Enter a custom prompt for QA processing",
            placeholder="Enter your custom QA instructions here...",
            height=100,
            help="Provide specific instructions for how to process and analyze the documents"
        )
        
        st.markdown("---")
        
        # File upload (using reliable Streamlit file uploader)
        uploaded_files = st.file_uploader(
            "Upload QA Documents",
            type=['pdf', 'docx', 'txt', 'csv', 'xlsx', 'md', 'zip', 'gif', 'jpg', 'jpeg', 'png', 'svg', 'bmp', 'tiff', 'webp', 'py', 'ipynb', 'r', 'sql', 'sas', 'mat', 'm', 'sav', 'dta', 'rds', 'rdata', 'parquet', 'feather', 'h5', 'hdf5', 'yaml', 'yml', 'toml', 'ini', 'cfg', 'conf', 'tex', 'bib', 'ris', 'enw'],
            accept_multiple_files=True,
            help="Upload one or more documents for QA processing. .md files with 'prompt' in the title will be used as prompts. ZIP files will be automatically extracted. Supports documents, images, code files (Python, R, SQL, MATLAB), and data files."
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")
            
            # Process uploaded files and extract prompts
            prompt_files = []
            document_files = []
            all_files = []
            seen_files = set()  # Track duplicates by filename
            file_hashes = {}  # Track duplicates by content hash
            duplicate_stats = {'skipped': 0, 'overwritten': 0}
            
            for file in uploaded_files:
                # Handle ZIP files
                if file.name.lower().endswith('.zip'):
                    st.info(f"📦 **ZIP File Detected**: {file.name}")
                    extracted_result = extract_zip_safely(file)
                    
                    if isinstance(extracted_result, str) and extracted_result.startswith("Error"):
                        st.error(f"❌ {extracted_result}")
                        continue
                    
                    # Add extracted files
                    for extracted_file in extracted_result:
                        file_content = extracted_file['content']
                        file_hash = calculate_file_hash(file_content)
                        file_name = extracted_file['name']
                        
                        # Check for duplicate filename
                        if file_name in seen_files:
                            st.warning(f"⚠️ **Duplicate Filename Skipped**: {file_name}")
                            duplicate_stats['skipped'] += 1
                            continue
                        
                        # Check for duplicate content (different filename)
                        if file_hash in file_hashes:
                            existing_file = file_hashes[file_hash]
                            st.warning(f"⚠️ **Duplicate Content Skipped**: {file_name} (same as {existing_file})")
                            duplicate_stats['skipped'] += 1
                            continue
                        
                        # File is unique - add it
                        seen_files.add(file_name)
                        file_hashes[file_hash] = file_name
                        all_files.append(extracted_file)
                        
                        if file_name.lower().endswith('.md') and 'prompt' in file_name.lower():
                            prompt_files.append(extracted_file)
                            st.info(f"📝 **Prompt File Extracted**: {file_name}")
                        else:
                            document_files.append(extracted_file)
                
                else:
                    # Handle regular files
                    file_content = file.read()
                    file.seek(0)  # Reset file pointer for later use
                    file_hash = calculate_file_hash(file_content)
                    file_name = file.name
                    
                    # Check for duplicate filename
                    if file_name in seen_files:
                        st.warning(f"⚠️ **Duplicate Filename Skipped**: {file_name}")
                        duplicate_stats['skipped'] += 1
                        continue
                    
                    # Check for duplicate content (different filename)
                    if file_hash in file_hashes:
                        existing_file = file_hashes[file_hash]
                        st.warning(f"⚠️ **Duplicate Content Skipped**: {file_name} (same as {existing_file})")
                        duplicate_stats['skipped'] += 1
                        continue
                    
                    # File is unique - add it
                    seen_files.add(file_name)
                    file_hashes[file_hash] = file_name
                    all_files.append(file)
                    
                    if file_name.lower().endswith('.md') and 'prompt' in file_name.lower():
                        # This is a prompt file
                        prompt_files.append(file)
                        st.info(f"📝 **Prompt File Detected**: {file_name}")
                    else:
                        # This is a document file
                        document_files.append(file)
            
            # Display duplicate detection summary
            if duplicate_stats['skipped'] > 0:
                st.info(f"🔄 **Duplicate Detection**: {duplicate_stats['skipped']} duplicate files were automatically skipped to prevent overwrites.")
            
            # Display uploaded files
            st.subheader("📋 Uploaded Files")
            
            if prompt_files:
                st.write("**📝 Prompt Files:**")
                for i, file in enumerate(prompt_files):
                    if hasattr(file, 'size'):
                        st.write(f"  {i+1}. **{file.name}** - Prompt file ({file.size} bytes)")
                    else:
                        st.write(f"  {i+1}. **{file.name}** - Prompt file ({file['size']} bytes)")
            
            if document_files:
                st.write("**📄 Document Files:**")
                for i, file in enumerate(document_files):
                    if hasattr(file, 'size'):
                        st.write(f"  {i+1}. **{file.name}** ({file.type}) - {file.size} bytes")
                    else:
                        st.write(f"  {i+1}. **{file.name}** ({file['type']}) - {file['size']} bytes")
            
            # Extract prompt from .md files
            extracted_prompt = ""
            if prompt_files:
                st.subheader("📝 Extracted Prompts")
                for file in prompt_files:
                    if hasattr(file, 'read'):
                        # Regular file object
                        content = file.read().decode('utf-8')
                        file.seek(0)  # Reset file pointer for later use
                    else:
                        # Extracted file from ZIP
                        content = file['content'].decode('utf-8')
                    
                    st.text_area(f"Prompt from {file.name if hasattr(file, 'name') else file['name']}:", content, height=150, disabled=True)
                    extracted_prompt += f"\n\n--- Prompt from {file.name if hasattr(file, 'name') else file['name']} ---\n{content}"
        
        st.markdown("---")
        
        # Process button
        if uploaded_files and review_id and validate_review_id(review_id)[0]:
            if st.button("🚀 Process Documents", type="primary", use_container_width=True):
                with st.spinner("Processing documents..."):
                    # Prepare files for processing
                    files_data = []
                    for file in all_files:
                        if hasattr(file, 'read'):
                            # Regular file object
                            files_data.append({
                                "filename": file.name,
                                "content": file.read(),
                                "type": file.type,
                                "size": file.size
                            })
                        else:
                            # Extracted file from ZIP
                            files_data.append({
                                "filename": file['name'],
                                "content": file['content'],
                                "type": file['type'],
                                "size": file['size']
                            })
                    
                    # Combine prompts
                    final_prompt = ""
                    if custom_prompt:
                        final_prompt += f"Custom Prompt:\n{custom_prompt}\n\n"
                    if extracted_prompt:
                        final_prompt += f"Extracted Prompts:\n{extracted_prompt}\n\n"
                    
                    # Display final prompt if available
                    if final_prompt:
                        st.subheader("🎯 Final Processing Prompt")
                        st.text_area("Combined prompt for processing:", final_prompt, height=150, disabled=True)
                    
                    # Process documents using MCP framework
                    result = st.session_state.qa_orchestrator.process_qa_documents(
                        files=files_data,
                        team_num=team_num,
                        process_name=process_name,
                        reviewer_name=reviewer_name,
                        review_id=review_id,
                        model_type=model_type,
                        risk_tier=risk_tier,
                        custom_prompt=final_prompt if final_prompt else None
                    )
                    
                    st.session_state.workflow_result = result
                    st.rerun()
        
        # Display results
        if st.session_state.workflow_result:
            st.markdown("---")
            st.header("📊 Processing Results")
            
            result = st.session_state.workflow_result
            
            if result["status"] == "completed":
                st.success("✅ Document processing completed successfully!")
                
                # Display extracted fields
                if "extraction_result" in result:
                    st.subheader("🔍 Extracted Fields")
                    extracted_fields = result["extraction_result"].get("extracted_fields", {})
                    
                    # Create a form for field editing
                    with st.form("extracted_fields_form"):
                        st.subheader("📝 Edit Extracted Fields")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            review_id_extracted = st.text_input(
                                "Review ID",
                                value=extracted_fields.get("review_id", ""),
                                key="review_id_extracted"
                            )
                            model_type_extracted = st.text_input(
                                "Model Type",
                                value=extracted_fields.get("model_type", ""),
                                key="model_type_extracted"
                            )
                            risk_tier_extracted = st.selectbox(
                                "Risk Tier",
                                ["Low", "Medium", "High", "Critical"],
                                index=["Low", "Medium", "High", "Critical"].index(
                                    extracted_fields.get("risk_tier", "Medium")
                                ),
                                key="risk_tier_extracted"
                            )
                            model_id = st.text_input(
                                "Model ID",
                                value=extracted_fields.get("model_id", ""),
                                key="model_id"
                            )
                            model_name = st.text_input(
                                "Model Name",
                                value=extracted_fields.get("model_name", ""),
                                key="model_name"
                            )
                        
                        with col2:
                            version = st.text_input(
                                "Version",
                                value=extracted_fields.get("version", ""),
                                key="version"
                            )
                            authors = st.text_input(
                                "Authors",
                                value=extracted_fields.get("authors", ""),
                                key="authors"
                            )
                            date = st.text_input(
                                "Date (mm-dd-yyyy)",
                                value=extracted_fields.get("date", ""),
                                key="date"
                            )
                            validation_type = st.text_input(
                                "Validation Type",
                                value=extracted_fields.get("validation_type", ""),
                                key="validation_type"
                            )
                        
                        # Submit button - this MUST be inside the form
                        submitted = st.form_submit_button("📄 Generate QA HealthCheck Report", type="primary", use_container_width=True)
                        
                        if submitted:
                            st.success("📄 QA HealthCheck Report generated successfully!")
                            
                            # Display report information
                            if "report_result" in result:
                                report_info = result["report_result"]
                                st.info(f"📋 Report saved to: {report_info.get('report_path', 'S3 location')}")
                                st.info(f"📄 Report sections: {', '.join(report_info.get('report_sections', []))}")
            
            elif result["status"] == "failed":
                st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
    
    with col2:
        st.header("ℹ️ Information")
        
        st.info("""
        **Demo Instructions:**
        
        1. **Upload Documents**: Select one or more files (including ZIP files)
        2. **Fill Required Fields**: Review ID and Model Type
        3. **Click Process**: Start the QA workflow
        4. **Review Extracted Fields**: Verify/correct extracted information
        5. **Generate Report**: Create QA HealthCheck Report
        
        **Supported File Types:**
        - **Documents**: PDF, DOCX, TXT, CSV, XLSX, MD, ZIP
        - **Images**: GIF, JPG, PNG, SVG, BMP, TIFF, WEBP
        - **Code & Analysis**: Python (.py), Jupyter (.ipynb), R (.r), SQL (.sql), SAS (.sas), MATLAB (.mat, .m)
        - **Statistical Data**: SPSS (.sav), Stata (.dta), R Data (.rds, .rdata), Parquet (.parquet), Feather (.feather), HDF5 (.h5, .hdf5)
        - **Configuration**: YAML (.yaml, .yml), TOML (.toml), INI (.ini, .cfg, .conf)
        - **Reports**: LaTeX (.tex), Bibliography (.bib, .ris, .enw)
        - **Data**: JSON, XML, HTML, LOG
        
        **Markdown Prompt Files:**
        - Upload `.md` files with "prompt" in the filename
        - Example: `my_prompt.md`, `qa_prompt.md`, `prompt_instructions.md`
        - Content will be automatically extracted and used as processing instructions
        
        **ZIP File Support:**
        - Upload ZIP files containing multiple documents
        - Automatic extraction with security checks
        - Duplicate file detection and prevention
        - Maximum 50MB ZIP files, 100 files per ZIP
        
        **Duplicate Prevention:**
        - Same filename uploads are automatically skipped
        - Warning messages for duplicate files
        
        **S3 Storage Path:**
        `{s3_root}/usecase-qa/teams/{team_num}/{process_name}/{reviewer_name}/{review_id}/`
        """)
        
        st.markdown("---")
        
        # Display workflow status
        if st.session_state.workflow_result:
            st.subheader("🔄 Workflow Status")
            result = st.session_state.workflow_result
            
            if result["status"] == "completed":
                st.success("✅ Completed")
                st.write(f"**Workflow ID:** {result.get('workflow_id', 'N/A')}")
                st.write(f"**Timestamp:** {result.get('timestamp', 'N/A')}")
            else:
                st.error("❌ Failed")
                st.write(f"**Error:** {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
