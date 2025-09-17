#!/usr/bin/env python3
"""
Simple QA Processor for Boss - No TidyLLM Dependency
====================================================

Basic document processing without external LLM dependencies.
Focuses on file parsing and basic structure analysis.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from dataclasses import dataclass

@dataclass
class SimpleQAResult:
    """Simple QA processing result."""
    file_name: str
    file_type: str
    processing_status: str
    content_extracted: bool
    basic_stats: Dict[str, Any]
    processed_timestamp: str

class SimpleQAProcessor:
    """
    Simple QA processor without TidyLLM dependencies.
    
    Basic file processing and content extraction.
    """
    
    def __init__(self):
        print("[QA] Simple QA Processor initialized (No TidyLLM)")
    
    def process_files(self, file_path: str) -> SimpleQAResult:
        """Process a file and extract basic information."""
        file_path = Path(file_path)
        
        try:
            if file_path.suffix.lower() == '.json':
                return self._process_json(file_path)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                return self._process_excel(file_path)
            elif file_path.suffix.lower() == '.pdf':
                return self._process_pdf(file_path)
            else:
                return self._create_unsupported_result(file_path)
                
        except Exception as e:
            return self._create_error_result(file_path, str(e))
    
    def _process_json(self, file_path: Path) -> SimpleQAResult:
        """Process JSON file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Basic JSON analysis
            keys = list(data.keys()) if isinstance(data, dict) else []
            content_size = len(str(data))
            
            stats = {
                'format': 'JSON',
                'top_level_keys': len(keys),
                'sample_keys': keys[:5] if keys else [],
                'content_size_chars': content_size,
                'is_dict': isinstance(data, dict),
                'is_list': isinstance(data, list),
                'nested_levels': self._estimate_json_depth(data)
            }
            
            return SimpleQAResult(
                file_name=file_path.name,
                file_type='JSON',
                processing_status='SUCCESS',
                content_extracted=True,
                basic_stats=stats,
                processed_timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
            )
            
        except Exception as e:
            return self._create_error_result(file_path, f"JSON parsing error: {e}")
    
    def _process_excel(self, file_path: Path) -> SimpleQAResult:
        """Process Excel file."""
        try:
            # Basic Excel file analysis
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            
            # Read first sheet for basic stats
            if sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_names[0], nrows=100)  # Limited read
                stats = {
                    'format': 'Excel',
                    'sheet_count': len(sheet_names),
                    'sheet_names': sheet_names,
                    'first_sheet_rows': len(df),
                    'first_sheet_columns': len(df.columns),
                    'column_names': df.columns.tolist()[:10],  # First 10 columns
                    'has_data': not df.empty
                }
            else:
                stats = {
                    'format': 'Excel',
                    'sheet_count': 0,
                    'error': 'No sheets found'
                }
            
            return SimpleQAResult(
                file_name=file_path.name,
                file_type='Excel',
                processing_status='SUCCESS',
                content_extracted=True,
                basic_stats=stats,
                processed_timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
            )
            
        except Exception as e:
            return self._create_error_result(file_path, f"Excel processing error: {e}")
    
    def _process_pdf(self, file_path: Path) -> SimpleQAResult:
        """Process PDF file (basic metadata only)."""
        try:
            # Basic PDF file stats without complex dependencies
            file_size = file_path.stat().st_size
            
            stats = {
                'format': 'PDF',
                'file_size_bytes': file_size,
                'file_size_mb': round(file_size / (1024 * 1024), 2),
                'text_extraction': 'Not available (no PDF library)',
                'pages': 'Unknown (requires PDF library)',
                'note': 'Basic file info only - install PyPDF2 for full processing'
            }
            
            return SimpleQAResult(
                file_name=file_path.name,
                file_type='PDF',
                processing_status='LIMITED',
                content_extracted=False,
                basic_stats=stats,
                processed_timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
            )
            
        except Exception as e:
            return self._create_error_result(file_path, f"PDF processing error: {e}")
    
    def _estimate_json_depth(self, obj, current_depth=0) -> int:
        """Estimate JSON nesting depth."""
        max_depth = current_depth
        
        if isinstance(obj, dict):
            for value in obj.values():
                depth = self._estimate_json_depth(value, current_depth + 1)
                max_depth = max(max_depth, depth)
        elif isinstance(obj, list):
            for item in obj[:5]:  # Check first 5 items only
                depth = self._estimate_json_depth(item, current_depth + 1)
                max_depth = max(max_depth, depth)
                
        return max_depth
    
    def _create_unsupported_result(self, file_path: Path) -> SimpleQAResult:
        """Create result for unsupported file types."""
        stats = {
            'format': 'Unknown',
            'file_extension': file_path.suffix,
            'error': f'Unsupported file type: {file_path.suffix}'
        }
        
        return SimpleQAResult(
            file_name=file_path.name,
            file_type='Unsupported',
            processing_status='SKIPPED',
            content_extracted=False,
            basic_stats=stats,
            processed_timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )
    
    def _create_error_result(self, file_path: Path, error_message: str) -> SimpleQAResult:
        """Create error result."""
        stats = {
            'format': 'Error',
            'error_message': error_message
        }
        
        return SimpleQAResult(
            file_name=file_path.name,
            file_type='Error',
            processing_status='ERROR',
            content_extracted=False,
            basic_stats=stats,
            processed_timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )

def demo_simple_qa_processor():
    """Demo the simple QA processor."""
    print("Simple QA Processor Demo")
    print("=" * 30)
    
    processor = SimpleQAProcessor()
    
    # Test with various file types (if they exist)
    test_files = [
        "sample.json",
        "data.xlsx", 
        "document.pdf",
        "unknown.txt"
    ]
    
    for test_file in test_files:
        print(f"\n[PROCESSING] {test_file}")
        
        # Create a dummy file for testing if it doesn't exist
        if not Path(test_file).exists():
            if test_file.endswith('.json'):
                with open(test_file, 'w') as f:
                    json.dump({"sample": "data", "nested": {"key": "value"}}, f)
            print(f"   Created sample file: {test_file}")
        
        try:
            result = processor.process_files(test_file)
            print(f"   Status: {result.processing_status}")
            print(f"   Type: {result.file_type}")
            print(f"   Content Extracted: {result.content_extracted}")
            print(f"   Stats: {result.basic_stats}")
        except Exception as e:
            print(f"   Error: {e}")

if __name__ == "__main__":
    demo_simple_qa_processor()