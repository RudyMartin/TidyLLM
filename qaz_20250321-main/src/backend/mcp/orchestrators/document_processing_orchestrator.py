#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Document Processing Orchestrator

Unified orchestrator for processing multiple document types with dedicated workers.
This provides a centralized interface for handling PDF, YAML, JSON, CSV, Excel, Word, and XML files.
"""

import logging
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class DocumentType:
    """Supported document types"""
    YAML = "yaml"
    JSON = "json"
    PDF = "pdf"
    CSV = "csv"
    EXCEL = "excel"
    WORD = "word"
    XML = "xml"
    TEXT = "text"
    MARKDOWN = "markdown"


class ProcessingMode:
    """Processing modes for different complexity levels"""
    SIMPLE = "simple"      # Basic parsing
    ENHANCED = "enhanced"  # Advanced features
    ADVANCED = "advanced"  # Full capabilities


class DocumentProcessingOrchestrator:
    """Unified document processing orchestrator with dedicated workers"""
    
    def __init__(self, mode: ProcessingMode = ProcessingMode.SIMPLE):
        self.mode = mode
        self.orchestrator_id = f"doc_orchestrator_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.workers = {}
        self.processing_stats = {
            'files_processed': 0,
            'successful_parses': 0,
            'failed_parses': 0,
            'by_type': {}
        }
        
        # Initialize workers
        self._initialize_workers()
        
        logger.info(f"Document Processing Orchestrator initialized in {mode} mode")
        logger.info(f"Available document types: {list(self.workers.keys())}")
    
    def _initialize_workers(self):
        """Initialize all document processing workers"""
        try:
            # YAML Worker
            import sys
            sys.path.append('src/backend/mcp/workers')
            from yaml_processing_worker import YAMLProcessingWorker
            self.workers[DocumentType.YAML] = YAMLProcessingWorker()
            logger.info("✅ YAML worker initialized")
        except Exception as e:
            logger.warning(f"⚠️ YAML worker not available: {e}")
        
        try:
            # JSON Worker
            from json_processing_worker import JSONProcessingWorker
            self.workers[DocumentType.JSON] = JSONProcessingWorker()
            logger.info("✅ JSON worker initialized")
        except Exception as e:
            logger.warning(f"⚠️ JSON worker not available: {e}")
        
        try:
            # PDF Worker
            from pdf_processing_worker import PDFProcessingWorker
            self.workers[DocumentType.PDF] = PDFProcessingWorker()
            logger.info("✅ PDF worker initialized")
        except Exception as e:
            logger.warning(f"⚠️ PDF worker not available: {e}")
        
        try:
            # CSV Worker
            from csv_processing_worker import CSVProcessingWorker
            self.workers[DocumentType.CSV] = CSVProcessingWorker()
            logger.info("✅ CSV worker initialized")
        except Exception as e:
            logger.warning(f"⚠️ CSV worker not available: {e}")
        
        try:
            # Excel Worker
            from excel_processing_worker import ExcelProcessingWorker
            self.workers[DocumentType.EXCEL] = ExcelProcessingWorker()
            logger.info("✅ Excel worker initialized")
        except Exception as e:
            logger.warning(f"⚠️ Excel worker not available: {e}")
        
        try:
            # Word Worker
            from word_processing_worker import WordProcessingWorker
            self.workers[DocumentType.WORD] = WordProcessingWorker()
            logger.info("✅ Word worker initialized")
        except Exception as e:
            logger.warning(f"⚠️ Word worker not available: {e}")
        
        try:
            # XML Worker
            from xml_processing_worker import XMLProcessingWorker
            self.workers[DocumentType.XML] = XMLProcessingWorker()
            logger.info("✅ XML worker initialized")
        except Exception as e:
            logger.warning(f"⚠️ XML worker not available: {e}")
        
        try:
            # Text Worker
            from text_processing_worker import TextProcessingWorker
            self.workers[DocumentType.TEXT] = TextProcessingWorker()
            logger.info("✅ Text worker initialized")
        except Exception as e:
            logger.warning(f"⚠️ Text worker not available: {e}")
        
        try:
            # Markdown Worker
            from markdown_processing_worker import MarkdownProcessingWorker
            self.workers[DocumentType.MARKDOWN] = MarkdownProcessingWorker()
            logger.info("✅ Markdown worker initialized")
        except Exception as e:
            logger.warning(f"⚠️ Markdown worker not available: {e}")
    
    def detect_document_type(self, file_path: str) -> str:
        """Detect document type based on file extension"""
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        type_mapping = {
            '.yaml': DocumentType.YAML,
            '.yml': DocumentType.YAML,
            '.json': DocumentType.JSON,
            '.pdf': DocumentType.PDF,
            '.csv': DocumentType.CSV,
            '.xlsx': DocumentType.EXCEL,
            '.xls': DocumentType.EXCEL,
            '.docx': DocumentType.WORD,
            '.doc': DocumentType.WORD,
            '.xml': DocumentType.XML,
            '.txt': DocumentType.TEXT,
            '.md': DocumentType.MARKDOWN,
            '.markdown': DocumentType.MARKDOWN
        }
        
        return type_mapping.get(extension, DocumentType.TEXT)
    
    def process_document(self, file_path: str, document_type: Optional[str] = None) -> Dict[str, Any]:
        """Process document using appropriate worker"""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"Document not found: {file_path}")
            
            # Detect document type if not provided
            if not document_type:
                document_type = self.detect_document_type(str(file_path))
            
            # Check if worker is available
            if document_type not in self.workers:
                return {
                    'success': False,
                    'error': f"No worker available for document type: {document_type}",
                    'file_path': str(file_path),
                    'document_type': document_type,
                    'processing_time': datetime.now().isoformat(),
                    'orchestrator_id': self.orchestrator_id
                }
            
            # Process with appropriate worker
            worker = self.workers[document_type]
            result = worker.process_document(str(file_path))
            
            # Update statistics
            self.processing_stats['files_processed'] += 1
            if result.get('success', False):
                self.processing_stats['successful_parses'] += 1
            else:
                self.processing_stats['failed_parses'] += 1
            
            # Update type-specific statistics
            if document_type not in self.processing_stats['by_type']:
                self.processing_stats['by_type'][document_type] = {
                    'processed': 0,
                    'successful': 0,
                    'failed': 0
                }
            
            self.processing_stats['by_type'][document_type]['processed'] += 1
            if result.get('success', False):
                self.processing_stats['by_type'][document_type]['successful'] += 1
            else:
                self.processing_stats['by_type'][document_type]['failed'] += 1
            
            # Add orchestrator metadata
            result['orchestrator_id'] = self.orchestrator_id
            result['document_type'] = document_type
            result['worker_used'] = document_type
            
            logger.info(f"✅ Processed {file_path} as {document_type}")
            return result
            
        except Exception as e:
            self.processing_stats['files_processed'] += 1
            self.processing_stats['failed_parses'] += 1
            
            logger.error(f"❌ Failed to process document {file_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'file_path': str(file_path),
                'document_type': document_type,
                'processing_time': datetime.now().isoformat(),
                'orchestrator_id': self.orchestrator_id
            }
    
    def process_multiple_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """Process multiple documents"""
        results = []
        
        for file_path in file_paths:
            result = self.process_document(file_path)
            results.append(result)
        
        return {
            'success': all(r.get('success', False) for r in results),
            'results': results,
            'summary': {
                'total_files': len(file_paths),
                'successful': sum(1 for r in results if r.get('success', False)),
                'failed': sum(1 for r in results if not r.get('success', False)),
                'processing_time': datetime.now().isoformat(),
                'orchestrator_id': self.orchestrator_id
            }
        }
    
    def validate_document(self, file_path: str) -> Dict[str, Any]:
        """Validate document without processing"""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return {
                    'valid': False,
                    'error': 'File not found',
                    'file_path': str(file_path)
                }
            
            document_type = self.detect_document_type(str(file_path))
            
            return {
                'valid': True,
                'file_path': str(file_path),
                'document_type': document_type,
                'file_size': file_path.stat().st_size,
                'worker_available': document_type in self.workers,
                'processing_time': datetime.now().isoformat(),
                'orchestrator_id': self.orchestrator_id
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'file_path': str(file_path),
                'processing_time': datetime.now().isoformat(),
                'orchestrator_id': self.orchestrator_id
            }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            'orchestrator_id': self.orchestrator_id,
            'mode': self.mode,
            'available_workers': list(self.workers.keys()),
            'stats': self.processing_stats.copy(),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_supported_types(self) -> List[str]:
        """Get list of supported document types"""
        return list(self.workers.keys())
    
    def check_worker_availability(self) -> Dict[str, bool]:
        """Check availability of all workers"""
        return {
            doc_type: doc_type in self.workers
            for doc_type in [
                DocumentType.YAML, DocumentType.JSON, DocumentType.PDF,
                DocumentType.CSV, DocumentType.EXCEL, DocumentType.WORD,
                DocumentType.XML, DocumentType.TEXT, DocumentType.MARKDOWN
            ]
        }
    
    def get_workers(self) -> Dict[str, Any]:
        """Get all available workers"""
        return self.workers.copy()
    
    @property
    def _workers(self) -> Dict[str, Any]:
        """Property access to workers for backward compatibility"""
        return self.workers


# Convenience function for easy usage
def process_document(file_path: str) -> Dict[str, Any]:
    """Convenience function to process a document"""
    orchestrator = DocumentProcessingOrchestrator()
    result = orchestrator.process_document(file_path)
    
    if result['success']:
        return result['data']
    else:
        raise ValueError(f"Failed to process document: {result['error']}")


if __name__ == "__main__":
    # Test the orchestrator
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        orchestrator = DocumentProcessingOrchestrator()
        result = orchestrator.process_document(file_path)
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python document_processing_orchestrator.py <document_file_path>")
