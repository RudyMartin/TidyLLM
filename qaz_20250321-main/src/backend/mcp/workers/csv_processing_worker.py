#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV Processing Worker

Dedicated worker for CSV file processing with pandas-free implementation.
This worker handles CSV files using pure Python for data operations.
"""

import logging
import csv
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class CSVProcessingWorker:
    """Dedicated worker for CSV file processing"""
    
    def __init__(self):
        self.worker_id = f"csv_worker_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.processing_stats = {
            'files_processed': 0,
            'successful_parses': 0,
            'failed_parses': 0,
            'total_rows': 0,
            'total_columns': 0
        }
        
        logger.info("CSV Processing Worker initialized")
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process CSV file with validation"""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"CSV file not found: {file_path}")
            
            # Validate file extension
            if not file_path.suffix.lower() == '.csv':
                return {
                    'success': False,
                    'error': 'Not a CSV file',
                    'file_path': str(file_path),
                    'processing_time': datetime.now().isoformat(),
                    'worker_id': self.worker_id
                }
            
            # Read CSV file
            data = []
            headers = []
            row_count = 0
            column_count = 0
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    csv_reader = csv.reader(f)
                    
                    # Read headers
                    headers = next(csv_reader, [])
                    column_count = len(headers)
                    
                    # Read data rows
                    for row in csv_reader:
                        data.append(row)
                        row_count += 1
                        
            except UnicodeDecodeError:
                # Try with different encoding
                with open(file_path, 'r', encoding='latin-1') as f:
                    csv_reader = csv.reader(f)
                    headers = next(csv_reader, [])
                    column_count = len(headers)
                    
                    for row in csv_reader:
                        data.append(row)
                        row_count += 1
            
            # Validate CSV structure
            validation_result = self._validate_csv_structure(data, headers)
            
            self.processing_stats['successful_parses'] += 1
            self.processing_stats['files_processed'] += 1
            self.processing_stats['total_rows'] += row_count
            self.processing_stats['total_columns'] += column_count
            
            logger.info(f"✅ Successfully parsed CSV file: {file_path} ({row_count} rows, {column_count} columns)")
            
            return {
                'success': True,
                'data': {
                    'headers': headers,
                    'rows': data,
                    'row_count': row_count,
                    'column_count': column_count
                },
                'file_path': str(file_path),
                'processing_time': datetime.now().isoformat(),
                'worker_id': self.worker_id,
                'validation': validation_result,
                'file_size': file_path.stat().st_size
            }
            
        except Exception as e:
            self.processing_stats['failed_parses'] += 1
            self.processing_stats['files_processed'] += 1
            
            logger.error(f"❌ Failed to process CSV file {file_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'file_path': str(file_path),
                'processing_time': datetime.now().isoformat(),
                'worker_id': self.worker_id
            }
    
    def _validate_csv_structure(self, data: List[List[str]], headers: List[str]) -> Dict[str, Any]:
        """Validate CSV structure and provide insights"""
        validation = {
            'is_valid': True,
            'row_count': len(data),
            'column_count': len(headers),
            'has_headers': len(headers) > 0,
            'consistent_columns': True,
            'empty_rows': 0,
            'max_row_length': 0,
            'min_row_length': 0,
            'data_types': {}
        }
        
        try:
            if not data:
                validation['is_valid'] = False
                validation['error'] = 'No data rows found'
                return validation
            
            # Check column consistency
            row_lengths = [len(row) for row in data]
            validation['max_row_length'] = max(row_lengths)
            validation['min_row_length'] = min(row_lengths)
            validation['consistent_columns'] = validation['max_row_length'] == validation['min_row_length']
            
            # Count empty rows
            validation['empty_rows'] = sum(1 for row in data if not any(cell.strip() for cell in row))
            
            # Analyze data types for each column
            if headers and data:
                for i, header in enumerate(headers):
                    column_data = [row[i] if i < len(row) else '' for row in data]
                    validation['data_types'][header] = self._analyze_column_type(column_data)
            
        except Exception as e:
            validation['is_valid'] = False
            validation['error'] = str(e)
        
        return validation
    
    def _analyze_column_type(self, column_data: List[str]) -> Dict[str, Any]:
        """Analyze the data type of a column"""
        analysis = {
            'type': 'string',
            'numeric_count': 0,
            'date_count': 0,
            'empty_count': 0,
            'unique_values': len(set(column_data))
        }
        
        for value in column_data:
            value = value.strip()
            
            if not value:
                analysis['empty_count'] += 1
                continue
            
            # Check if numeric
            try:
                float(value)
                analysis['numeric_count'] += 1
            except ValueError:
                pass
            
            # Check if date (simple check)
            if '/' in value or '-' in value:
                try:
                    datetime.strptime(value, '%Y-%m-%d')
                    analysis['date_count'] += 1
                except ValueError:
                    try:
                        datetime.strptime(value, '%m/%d/%Y')
                        analysis['date_count'] += 1
                    except ValueError:
                        pass
        
        # Determine primary type
        total_values = len(column_data) - analysis['empty_count']
        if total_values > 0:
            numeric_ratio = analysis['numeric_count'] / total_values
            date_ratio = analysis['date_count'] / total_values
            
            if numeric_ratio > 0.8:
                analysis['type'] = 'numeric'
            elif date_ratio > 0.8:
                analysis['type'] = 'date'
        
        return analysis
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            'worker_id': self.worker_id,
            'stats': self.processing_stats.copy(),
            'timestamp': datetime.now().isoformat()
        }
    
    def validate_csv_file(self, file_path: str) -> Dict[str, Any]:
        """Validate CSV file without processing"""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return {
                    'valid': False,
                    'error': 'File not found',
                    'file_path': str(file_path)
                }
            
            # Check file extension
            if not file_path.suffix.lower() == '.csv':
                return {
                    'valid': False,
                    'error': 'Not a CSV file',
                    'file_path': str(file_path)
                }
            
            # Check if file is readable
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline()
                
                return {
                    'valid': True,
                    'file_path': str(file_path),
                    'file_size': file_path.stat().st_size,
                    'first_line': first_line.strip()[:50] + '...' if len(first_line) > 50 else first_line.strip()
                }
                
            except UnicodeDecodeError:
                # Try with different encoding
                with open(file_path, 'r', encoding='latin-1') as f:
                    first_line = f.readline()
                
                return {
                    'valid': True,
                    'file_path': str(file_path),
                    'file_size': file_path.stat().st_size,
                    'first_line': first_line.strip()[:50] + '...' if len(first_line) > 50 else first_line.strip(),
                    'encoding': 'latin-1'
                }
                
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'file_path': str(file_path)
            }


# Convenience function for easy usage
def load_csv_file(file_path: str) -> Dict[str, Any]:
    """Convenience function to load CSV file"""
    worker = CSVProcessingWorker()
    result = worker.process_document(file_path)
    
    if result['success']:
        return result['data']
    else:
        raise ValueError(f"Failed to load CSV file: {result['error']}")


if __name__ == "__main__":
    # Test the worker
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        worker = CSVProcessingWorker()
        result = worker.process_document(file_path)
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python csv_processing_worker.py <csv_file_path>")
