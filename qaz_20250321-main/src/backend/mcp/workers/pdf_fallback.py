#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fallback PDF Processor

Provides alternative PDF processing when PyMuPDF is not available.
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import io

logger = logging.getLogger(__name__)

class FallbackPDFProcessor:
    """Fallback PDF processor using alternative libraries"""
    
    def __init__(self):
        self.available_methods = self._check_available_methods()
    
    def _check_available_methods(self) -> Dict[str, bool]:
        """Check which PDF processing methods are available"""
        methods = {}
        
        # Check for pdfplumber
        try:
            import pdfplumber
            methods['pdfplumber'] = True
        except ImportError:
            methods['pdfplumber'] = False
        
        # Check for pypdf (modern replacement for PyPDF2)
        try:
            import pypdf
            methods['pypdf'] = True
        except ImportError:
            methods['pypdf'] = False
        
        # Check for pdf2image
        try:
            import pdf2image
            methods['pdf2image'] = True
        except ImportError:
            methods['pdf2image'] = False
        
        return methods
    
    def process_pdf(self, file_path: str = None, file_content: bytes = None) -> Dict[str, Any]:
        """Process PDF using available fallback methods"""
        
        if not file_path and not file_content:
            raise ValueError("Either file_path or file_content must be provided")
        
        # Try pdfplumber first (best fallback)
        if self.available_methods.get('pdfplumber'):
            return self._process_with_pdfplumber(file_path, file_content)
        
        # Try pypdf as second option
        elif self.available_methods.get('pypdf'):
            return self._process_with_pypdf(file_path, file_content)
        
        else:
            return {
                'success': False,
                'confidence_score': 0.0,
                'error': "No PDF processing libraries available",
                'suggestion': "Install one of: pip install pdfplumber pypdf PyMuPDF"
            }
    
    def _process_with_pdfplumber(self, file_path: str = None, file_content: bytes = None) -> Dict[str, Any]:
        """Process PDF using pdfplumber"""
        try:
            import pdfplumber
            
            if file_content:
                pdf = pdfplumber.open(io.BytesIO(file_content))
            else:
                pdf = pdfplumber.open(file_path)
            
            text_content = ""
            page_count = len(pdf.pages)
            
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                text_content += f"\n--- Page {page_num + 1} ---\n{page_text}"
            
            # Extract tables
            tables = []
            for page_num, page in enumerate(pdf.pages):
                page_tables = page.extract_tables()
                for table_num, table in enumerate(page_tables):
                    tables.append({
                        'page': page_num + 1,
                        'table_index': table_num,
                        'data': table
                    })
            
            pdf.close()
            
            return {
                'success': True,
                'confidence_score': 0.85,
                'processing': {
                    'text_content': text_content,
                    'page_count': page_count,
                    'images': [],  # pdfplumber doesn't extract images easily
                    'tables': tables,
                    'file_size': len(file_content) if file_content else Path(file_path).stat().st_size,
                    'method': 'pdfplumber'
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing PDF with pdfplumber: {e}")
            return {
                'success': False,
                'confidence_score': 0.0,
                'error': f"pdfplumber processing failed: {str(e)}"
            }
    
    def _process_with_pypdf(self, file_path: str = None, file_content: bytes = None) -> Dict[str, Any]:
        """Process PDF using pypdf (modern replacement for PyPDF2)"""
        try:
            import pypdf
            
            if file_content:
                pdf_file = io.BytesIO(file_content)
            else:
                pdf_file = open(file_path, 'rb')
            
            pdf_reader = pypdf.PdfReader(pdf_file)
            
            text_content = ""
            page_count = len(pdf_reader.pages)
            
            for page_num in range(page_count):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text() or ""
                text_content += f"\n--- Page {page_num + 1} ---\n{page_text}"
            
            pdf_file.close()
            
            return {
                'success': True,
                'confidence_score': 0.80,
                'processing': {
                    'text_content': text_content,
                    'page_count': page_count,
                    'images': [],  # pypdf doesn't extract images
                    'tables': [],  # pypdf doesn't extract tables
                    'file_size': len(file_content) if file_content else Path(file_path).stat().st_size,
                    'method': 'pypdf'
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing PDF with pypdf: {e}")
            return {
                'success': False,
                'confidence_score': 0.0,
                'error': f"pypdf processing failed: {str(e)}"
            }
