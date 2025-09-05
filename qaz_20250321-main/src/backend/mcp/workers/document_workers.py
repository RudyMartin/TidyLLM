#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Document Processing Workers

Specialized workers for document processing tasks in the MCP hierarchy.
"""

# Modern PDF processing - no more fitz dependency!
try:
    from .modern_pdf_processor import ModernPDFProcessor
    MODERN_PDF_AVAILABLE = True
except ImportError as e:
    MODERN_PDF_AVAILABLE = False
    MODERN_PDF_ERROR = str(e)
    print(f"⚠️  Modern PDF processor not available: {e}")
    print("   Install with: pip install pdfplumber pypdfium2")

import re
from typing import Dict, Any, List
from pathlib import Path
import logging

from .base_worker import BaseWorker
from ..protocol.message_protocol import MCPMessage, TaskType


class PDFProcessorWorker(BaseWorker):
    """Worker for processing PDF documents"""
    
    def __init__(self):
        super().__init__("pdf_processor", "document_processing")
    
    def process_task(self, message: MCPMessage) -> Dict[str, Any]:
        """Process PDF document"""
        payload = message.payload
        file_path = payload.get('file_path')
        file_content = payload.get('file_content')
        
        if not file_path and not file_content:
            raise ValueError("Either file_path or file_content must be provided")
        
        # Check if modern PDF processor is available
        if not MODERN_PDF_AVAILABLE:
            # Try fallback PDF processor
            try:
                from .pdf_fallback import FallbackPDFProcessor
                fallback_processor = FallbackPDFProcessor()
                result = fallback_processor.process_pdf(file_path, file_content)
                if result['success']:
                    result['processing']['method'] = 'fallback_' + result['processing'].get('method', 'unknown')
                    return result
                else:
                    return {
                        'success': False,
                        'confidence_score': 0.0,
                        'error': f"Modern PDF processor not available: {MODERN_PDF_ERROR}",
                        'fallback_error': result.get('error', 'Unknown fallback error'),
                        'suggestion': "Install modern PDF libraries: pip install pdfplumber pypdfium2"
                    }
            except ImportError:
                return {
                    'success': False,
                    'confidence_score': 0.0,
                    'error': f"Modern PDF processor not available: {MODERN_PDF_ERROR}",
                    'suggestion': "Install modern PDF libraries: pip install pdfplumber pypdfium2"
                }
        
        try:
            # Use modern PDF processor instead of fitz
            modern_processor = ModernPDFProcessor()
            result = modern_processor.process_pdf(file_path, file_content)
            
            if result['success']:
                return result
            else:
                return {
                    'success': False,
                    'confidence_score': 0.0,
                    'error': result.get('error', 'Unknown error in modern PDF processor')
                }
            
        except Exception as e:
            return {
                'success': False,
                'confidence_score': 0.0,
                'error': str(e)
            }
    
    # Note: Image and table extraction methods moved to ModernPDFProcessor
    # This eliminates the fitz dependency completely


class TextCleanerWorker(BaseWorker):
    """Worker for cleaning and preprocessing text"""
    
    def __init__(self):
        super().__init__("text_cleaner", "text_processing")
    
    def process_task(self, message: MCPMessage) -> Dict[str, Any]:
        """Clean and preprocess text"""
        payload = message.payload
        text_content = payload.get('text_content', '')
        
        if not text_content:
            return {
                'success': False,
                'confidence_score': 0.0,
                'error': 'No text content provided'
            }
        
        try:
            # Clean text
            cleaned_text = self._clean_text(text_content)
            
            # Extract references
            references = self._extract_references(cleaned_text)
            
            # Extract claims
            claims = self._extract_claims(cleaned_text)
            
            # Extract evidence
            evidence = self._extract_evidence(cleaned_text)
            
            return {
                'success': True,
                'confidence_score': 0.9,
                'processing': {
                    'cleaned_text': cleaned_text,
                    'original_length': len(text_content),
                    'cleaned_length': len(cleaned_text),
                    'references': references,
                    'claims': claims,
                    'evidence': evidence
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'confidence_score': 0.0,
                'error': str(e)
            }
    
    def _clean_text(self, text: str) -> str:
        """Clean text content"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\"\']', '', text)
        
        # Fix common OCR issues
        text = re.sub(r'[|]', 'I', text)  # Fix OCR I/l confusion
        text = re.sub(r'[0]', 'O', text)  # Fix OCR O/0 confusion
        
        return text.strip()
    
    def _extract_references(self, text: str) -> List[Dict[str, Any]]:
        """Extract references from text"""
        references = []
        
        # Academic reference patterns
        patterns = [
            r'([A-Z][a-z]+,\s*[A-Z]\.\s*\d{4})',
            r'([A-Z][a-z]+\s+et\s+al\.\s*\d{4})',
            r'([A-Z][a-z]+,\s*[A-Z][a-z]+,\s*\d{4})',
            r'([A-Z][a-z]+\s+&\s+[A-Z][a-z]+,\s*\d{4})',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                ref = {
                    "text": match.group(1),
                    "position": match.start(),
                    "type": "academic_reference"
                }
                references.append(ref)
        
        # URL patterns
        url_pattern = r'https?://[^\s]+'
        url_matches = re.finditer(url_pattern, text)
        for match in url_matches:
            ref = {
                "text": match.group(0),
                "position": match.start(),
                "type": "url"
            }
            references.append(ref)
        
        return references
    
    def _extract_claims(self, text: str) -> List[Dict[str, Any]]:
        """Extract claims from text"""
        claims = []
        
        claim_patterns = [
            r'[A-Z][^.!?]*\s+(is|are|was|were|will be|should be|must be)[^.!?]*[.!?]',
            r'[A-Z][^.!?]*\s+(proves|demonstrates|shows|indicates|suggests)[^.!?]*[.!?]'
        ]
        
        for pattern in claim_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                claims.append({
                    "text": match.group(0),
                    "position": match.start()
                })
        
        return claims
    
    def _extract_evidence(self, text: str) -> List[Dict[str, Any]]:
        """Extract evidence from text"""
        evidence = []
        
        evidence_patterns = [
            r'\d+%',
            r'\d+\.\d+',
            r'study shows',
            r'research indicates',
            r'data suggests'
        ]
        
        for pattern in evidence_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                evidence.append({
                    "text": match.group(0),
                    "position": match.start()
                })
        
        return evidence


class EmbeddingGeneratorWorker(BaseWorker):
    """Worker for generating embeddings"""
    
    def __init__(self):
        super().__init__("embedding_generator", "embedding_processing")
        self.embedding_model = None
        self._load_embedding_model()
    
    def _load_embedding_model(self):
        """Load the embedding model using centralized EmbeddingHelper"""
        try:
            from ...core.embedding_helper import EmbeddingHelper
            # Use centralized EmbeddingHelper with target dimensions
            self.embedding_helper = EmbeddingHelper(target_dimensions=1024)
            self.embedding_model = self.embedding_helper.embedding_model
            self.logger.info(f"✅ Using centralized EmbeddingHelper with {self.embedding_helper.target_dimensions} dimensions")
        except ImportError:
            self.logger.warning("EmbeddingHelper not available, using fallback")
            self.embedding_model = None
    
    def process_task(self, message: MCPMessage) -> Dict[str, Any]:
        """Generate embeddings for text chunks"""
        payload = message.payload
        text_chunks = payload.get('text_chunks', [])
        
        if not text_chunks:
            return {
                'success': False,
                'confidence_score': 0.0,
                'error': 'No text chunks provided'
            }
        
        if not self.embedding_model:
            return {
                'success': False,
                'confidence_score': 0.0,
                'error': 'Embedding model not available'
            }
        
        try:
            embeddings = []
            
            for i, chunk in enumerate(text_chunks):
                # Generate embedding using centralized helper
                embedding, metadata = self.embedding_helper.generate_embedding(
                    chunk['text'], 
                    f"chunk_{chunk.get('chunk_id', i)}"
                )
                embedding_dim = len(embedding)
                
                embeddings.append({
                    'chunk_id': chunk.get('chunk_id', f'chunk_{i}'),
                    'embedding': embedding.tolist(),
                    'embedding_dim': embedding_dim,
                    'text_length': len(chunk['text'])
                })
            
            return {
                'success': True,
                'confidence_score': 0.95,
                'processing': {
                    'embeddings': embeddings,
                    'total_chunks': len(embeddings),
                    'embedding_dim': len(embeddings[0]['embedding']) if embeddings else 0
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'confidence_score': 0.0,
                'error': str(e)
            }


class TableExtractorWorker(BaseWorker):
    """Worker for extracting and processing tables"""
    
    def __init__(self):
        super().__init__("table_extractor", "table_processing")
    
    def process_task(self, message: MCPMessage) -> Dict[str, Any]:
        """Extract and process table data"""
        payload = message.payload
        tables = payload.get('tables', [])
        
        if not tables:
            return {
                'success': False,
                'confidence_score': 0.0,
                'error': 'No tables provided'
            }
        
        try:
            processed_tables = []
            
            for table in tables:
                processed_table = self._process_table(table)
                processed_tables.append(processed_table)
            
            return {
                'success': True,
                'confidence_score': 0.9,
                'processing': {
                    'processed_tables': processed_tables,
                    'total_tables': len(processed_tables)
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'confidence_score': 0.0,
                'error': str(e)
            }
    
    def _process_table(self, table: Dict[str, Any]) -> Dict[str, Any]:
        """Process individual table"""
        content = table.get('content', [])
        header = table.get('header', [])
        
        # Extract structured data
        structured_data = []
        for row in content:
            if len(row) == len(header):
                row_data = dict(zip(header, row))
                structured_data.append(row_data)
        
        # Analyze table content
        analysis = self._analyze_table_content(content, header)
        
        return {
            'table_id': table.get('table_id', 'unknown'),
            'page': table.get('page', 0),
            'rows': len(content),
            'columns': len(header),
            'header': header,
            'structured_data': structured_data,
            'analysis': analysis
        }
    
    def _analyze_table_content(self, content: List[List[str]], header: List[str]) -> Dict[str, Any]:
        """Analyze table content"""
        analysis = {
            'numeric_columns': [],
            'text_columns': [],
            'date_columns': [],
            'data_types': {}
        }
        
        if not content or not header:
            return analysis
        
        # Analyze each column
        for col_idx, col_name in enumerate(header):
            column_values = [row[col_idx] for row in content if len(row) > col_idx]
            
            # Determine column type
            numeric_count = 0
            date_count = 0
            
            for value in column_values:
                if re.search(r'\d+\.?\d*', value):
                    numeric_count += 1
                if re.search(r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}', value):
                    date_count += 1
            
            total_values = len(column_values)
            if total_values > 0:
                numeric_ratio = numeric_count / total_values
                date_ratio = date_count / total_values
                
                if date_ratio > 0.3:
                    analysis['date_columns'].append(col_name)
                    analysis['data_types'][col_name] = 'date'
                elif numeric_ratio > 0.5:
                    analysis['numeric_columns'].append(col_name)
                    analysis['data_types'][col_name] = 'numeric'
                else:
                    analysis['text_columns'].append(col_name)
                    analysis['data_types'][col_name] = 'text'
        
        return analysis
