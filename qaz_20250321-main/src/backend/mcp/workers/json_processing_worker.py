#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON Processing Worker

Dedicated worker for JSON file processing with validation and error handling.
This worker handles JSON files with proper error recovery and validation.
"""

import logging
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class JSONProcessingWorker:
    """Dedicated worker for JSON file processing"""
    
    def __init__(self):
        self.worker_id = f"json_worker_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.processing_stats = {
            'files_processed': 0,
            'successful_parses': 0,
            'failed_parses': 0,
            'validation_errors': 0
        }
        
        logger.info("JSON Processing Worker initialized")
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process JSON file with validation"""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"JSON file not found: {file_path}")
            
            # Validate file extension
            if not file_path.suffix.lower() == '.json':
                return {
                    'success': False,
                    'error': 'Not a JSON file',
                    'file_path': str(file_path),
                    'processing_time': datetime.now().isoformat(),
                    'worker_id': self.worker_id
                }
            
            # Read and parse JSON
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse JSON with error handling
            try:
                data = json.loads(content)
            except json.JSONDecodeError as e:
                self.processing_stats['validation_errors'] += 1
                self.processing_stats['failed_parses'] += 1
                self.processing_stats['files_processed'] += 1
                
                return {
                    'success': False,
                    'error': f'Invalid JSON format: {str(e)}',
                    'file_path': str(file_path),
                    'processing_time': datetime.now().isoformat(),
                    'worker_id': self.worker_id,
                    'json_error': {
                        'line': e.lineno,
                        'column': e.colno,
                        'message': e.msg
                    }
                }
            
            # Validate JSON structure
            validation_result = self._validate_json_structure(data)
            
            self.processing_stats['successful_parses'] += 1
            self.processing_stats['files_processed'] += 1
            
            logger.info(f"✅ Successfully parsed JSON file: {file_path}")
            
            return {
                'success': True,
                'data': data,
                'file_path': str(file_path),
                'processing_time': datetime.now().isoformat(),
                'worker_id': self.worker_id,
                'validation': validation_result,
                'file_size': file_path.stat().st_size,
                'content_length': len(content)
            }
            
        except Exception as e:
            self.processing_stats['failed_parses'] += 1
            self.processing_stats['files_processed'] += 1
            
            logger.error(f"❌ Failed to process JSON file {file_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'file_path': str(file_path),
                'processing_time': datetime.now().isoformat(),
                'worker_id': self.worker_id
            }
    
    def _validate_json_structure(self, data: Any) -> Dict[str, Any]:
        """Validate JSON structure and provide insights"""
        validation = {
            'is_valid': True,
            'type': type(data).__name__,
            'size': 0,
            'depth': 0,
            'has_nested_objects': False,
            'has_arrays': False,
            'key_count': 0
        }
        
        try:
            if isinstance(data, dict):
                validation['key_count'] = len(data.keys())
                validation['has_nested_objects'] = any(isinstance(v, dict) for v in data.values())
                validation['has_arrays'] = any(isinstance(v, list) for v in data.values())
                validation['depth'] = self._calculate_depth(data)
                validation['size'] = len(str(data))
            elif isinstance(data, list):
                validation['size'] = len(data)
                validation['has_nested_objects'] = any(isinstance(item, dict) for item in data)
                validation['has_arrays'] = any(isinstance(item, list) for item in data)
                validation['depth'] = self._calculate_depth(data)
            
        except Exception as e:
            validation['is_valid'] = False
            validation['error'] = str(e)
        
        return validation
    
    def _calculate_depth(self, obj: Any, current_depth: int = 0) -> int:
        """Calculate the maximum depth of nested structures"""
        if not isinstance(obj, (dict, list)):
            return current_depth
        
        max_depth = current_depth
        
        if isinstance(obj, dict):
            for value in obj.values():
                max_depth = max(max_depth, self._calculate_depth(value, current_depth + 1))
        elif isinstance(obj, list):
            for item in obj:
                max_depth = max(max_depth, self._calculate_depth(item, current_depth + 1))
        
        return max_depth
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            'worker_id': self.worker_id,
            'stats': self.processing_stats.copy(),
            'timestamp': datetime.now().isoformat()
        }
    
    def validate_json_file(self, file_path: str) -> Dict[str, Any]:
        """Validate JSON file without processing"""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return {
                    'valid': False,
                    'error': 'File not found',
                    'file_path': str(file_path)
                }
            
            # Check file extension
            if not file_path.suffix.lower() == '.json':
                return {
                    'valid': False,
                    'error': 'Not a JSON file',
                    'file_path': str(file_path)
                }
            
            # Check if file is readable and valid JSON
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Try to parse JSON
                json.loads(content)
                
                return {
                    'valid': True,
                    'file_path': str(file_path),
                    'file_size': file_path.stat().st_size,
                    'content_length': len(content),
                    'first_line': content.split('\n')[0][:50] + '...' if len(content) > 50 else content
                }
                
            except json.JSONDecodeError as e:
                return {
                    'valid': False,
                    'error': f'Invalid JSON: {str(e)}',
                    'file_path': str(file_path),
                    'json_error': {
                        'line': e.lineno,
                        'column': e.colno,
                        'message': e.msg
                    }
                }
                
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'file_path': str(file_path)
            }


# Convenience function for easy usage
def load_json_file(file_path: str) -> Dict[str, Any]:
    """Convenience function to load JSON file"""
    worker = JSONProcessingWorker()
    result = worker.process_document(file_path)
    
    if result['success']:
        return result['data']
    else:
        raise ValueError(f"Failed to load JSON file: {result['error']}")


if __name__ == "__main__":
    # Test the worker
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        worker = JSONProcessingWorker()
        result = worker.process_document(file_path)
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python json_processing_worker.py <json_file_path>")
