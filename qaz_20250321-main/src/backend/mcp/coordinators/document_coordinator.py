#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Document Coordinator

Coordinates document processing tasks using specialized workers in the MCP hierarchy.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from ..protocol.message_protocol import (
    MCPMessage, TaskType, Priority, 
    create_coordinator_to_worker_message,
    create_coordinator_to_planner_message
)
from ..workers.document_workers import (
    PDFProcessorWorker, TextCleanerWorker, 
    EmbeddingGeneratorWorker, TableExtractorWorker
)
from ..workers.live_context_worker import LiveContextWorker


class DocumentCoordinator:
    """Coordinates document processing tasks"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Initialize workers
        self.pdf_processor = PDFProcessorWorker()
        self.text_cleaner = TextCleanerWorker()
        self.embedding_generator = EmbeddingGeneratorWorker()
        self.table_extractor = TableExtractorWorker()
        self.live_context_worker = LiveContextWorker()
        
        # Performance tracking
        self.performance_metrics = {
            'total_documents_processed': 0,
            'total_chunks_generated': 0,
            'total_embeddings_created': 0,
            'average_processing_time': 0.0,
            'success_rate': 0.0
        }
        
        # Audit trail
        self.audit_log = []
    
    def process_document(self, message: MCPMessage) -> MCPMessage:
        """Process a document through the complete pipeline"""
        start_time = datetime.now()
        
        # Add audit entry
        message.add_audit_entry(
            action="document_processing_started",
            decision_reasoning="Document coordinator started processing pipeline",
            confidence_score=1.0
        )
        
        try:
            # Step 1: Process PDF
            pdf_result = self._process_pdf(message)
            if not pdf_result['success']:
                return self._create_error_response(message, pdf_result['error'])
            
            # Step 2: Clean text
            text_result = self._clean_text(pdf_result['text_content'])
            if not text_result['success']:
                return self._create_error_response(message, text_result['error'])
            
            # Step 3: Generate chunks
            chunks_result = self._generate_chunks(text_result['cleaned_text'])
            if not chunks_result['success']:
                return self._create_error_response(message, chunks_result['error'])
            
            # Step 4: Generate embeddings
            embeddings_result = self._generate_embeddings(chunks_result['chunks'])
            if not embeddings_result['success']:
                return self._create_error_response(message, embeddings_result['error'])
            
            # Step 5: Process tables
            tables_result = self._process_tables(pdf_result['tables'])
            
            # Step 6: Integrate live context (optional)
            try:
                live_context_result = self._integrate_live_context(text_result)
                live_context_available = True
            except Exception as e:
                self.logger.warning(f"Live context integration failed (non-critical): {e}")
                live_context_result = {'success': False, 'live_context': {}}
                live_context_available = False
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Update performance metrics
            self._update_metrics(True, processing_time, len(chunks_result['chunks']))
            
            # Create comprehensive result
            result = {
                'success': True,
                'confidence_score': 0.95,
                'processing_time': processing_time,
                'document_info': {
                    'page_count': pdf_result['page_count'],
                    'file_size': pdf_result['file_size'],
                    'images_count': len(pdf_result['images']),
                    'tables_count': len(pdf_result['tables'])
                },
                'text_processing': {
                    'original_length': text_result['original_length'],
                    'cleaned_length': text_result['cleaned_length'],
                    'references_count': len(text_result['references']),
                    'claims_count': len(text_result['claims']),
                    'evidence_count': len(text_result['evidence'])
                },
                'chunks': {
                    'total_chunks': len(chunks_result['chunks']),
                    'chunks': chunks_result['chunks']
                },
                'embeddings': {
                    'total_embeddings': len(embeddings_result['embeddings']),
                    'embedding_dim': embeddings_result['embedding_dim']
                },
                'tables': tables_result.get('processed_tables', []),
                'live_context': live_context_result.get('live_context', {}),
                'live_context_available': live_context_available,
                'audit_trail': message.audit_trail
            }
            
            # Add completion audit entry
            message.add_audit_entry(
                action="document_processing_completed",
                decision_reasoning=f"Document processing pipeline completed successfully. Live context: {'available' if live_context_available else 'unavailable'}",
                confidence_score=0.95,
                performance_metrics={
                    'processing_time': processing_time,
                    'total_chunks': len(chunks_result['chunks']),
                    'total_embeddings': len(embeddings_result['embeddings']),
                    'live_context_available': live_context_available
                }
            )
            
            # Create response message
            response_message = create_coordinator_to_planner_message(
                task_type=TaskType.PROCESSING,
                payload=result,
                context=message.context
            )
            response_message.audit_trail = message.audit_trail.copy()
            
            self.logger.info(f"Document processing completed in {processing_time:.2f}s")
            return response_message
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_metrics(False, processing_time, 0)
            
            message.add_audit_entry(
                action="document_processing_failed",
                decision_reasoning=f"Document processing failed: {str(e)}",
                confidence_score=0.0,
                error_details=str(e)
            )
            
            return self._create_error_response(message, str(e))
    
    def _process_pdf(self, message: MCPMessage) -> Dict[str, Any]:
        """Process PDF document using PDF processor worker"""
        # Create worker message
        worker_message = create_coordinator_to_worker_message(
            worker="pdf_processor",
            task_type=TaskType.PROCESSING,
            payload=message.payload,
            context=message.context
        )
        
        # Execute worker
        response = self.pdf_processor.execute(worker_message)
        
        if response.payload['success']:
            return response.payload['result']['processing']
        else:
            return {'success': False, 'error': response.payload['result']['error']}
    
    def _clean_text(self, text_content: str) -> Dict[str, Any]:
        """Clean text using text cleaner worker"""
        # Create worker message
        worker_message = create_coordinator_to_worker_message(
            worker="text_cleaner",
            task_type=TaskType.PROCESSING,
            payload={'text_content': text_content},
            context={}
        )
        
        # Execute worker
        response = self.text_cleaner.execute(worker_message)
        
        if response.payload['success']:
            return response.payload['result']['processing']
        else:
            return {'success': False, 'error': response.payload['result']['error']}
    
    def _generate_chunks(self, text_content: str) -> Dict[str, Any]:
        """Generate text chunks"""
        # Simple chunking logic (can be enhanced)
        sentences = text_content.split('.')
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_index = 1
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > 200 and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    'chunk_id': f'chunk_{chunk_index:03d}',
                    'text': chunk_text,
                    'word_count': len(chunk_text.split()),
                    'char_count': len(chunk_text),
                    'chunk_number': chunk_index
                })
                
                current_chunk = []
                current_length = 0
                chunk_index += 1
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                'chunk_id': f'chunk_{chunk_index:03d}',
                'text': chunk_text,
                'word_count': len(chunk_text.split()),
                'char_count': len(chunk_text),
                'chunk_number': chunk_index
            })
        
        return {
            'success': True,
            'chunks': chunks
        }
    
    def _generate_embeddings(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate embeddings using embedding generator worker"""
        # Create worker message
        worker_message = create_coordinator_to_worker_message(
            worker="embedding_generator",
            task_type=TaskType.PROCESSING,
            payload={'text_chunks': chunks},
            context={}
        )
        
        # Execute worker
        response = self.embedding_generator.execute(worker_message)
        
        if response.payload['success']:
            return response.payload['result']['processing']
        else:
            return {'success': False, 'error': response.payload['result']['error']}
    
    def _process_tables(self, tables: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process tables using table extractor worker"""
        if not tables:
            return {'success': True, 'processed_tables': []}
        
        # Create worker message
        worker_message = create_coordinator_to_worker_message(
            worker="table_extractor",
            task_type=TaskType.PROCESSING,
            payload={'tables': tables},
            context={}
        )
        
        # Execute worker
        response = self.table_extractor.execute(worker_message)
        
        if response.payload['success']:
            return response.payload['result']['processing']
        else:
            return {'success': True, 'processed_tables': []}  # Non-critical failure
    
    def _integrate_live_context(self, text_result: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate live context using live context worker"""
        # Create worker message
        worker_message = create_coordinator_to_worker_message(
            worker="live_context_worker",
            task_type=TaskType.PROCESSING,
            payload={'document_analysis': text_result},
            context={}
        )
        
        # Execute worker
        response = self.live_context_worker.execute(worker_message)
        
        if response.payload['success']:
            return {'success': True, 'live_context': response.payload['result']['processing']}
        else:
            return {'success': True, 'live_context': {}}  # Non-critical failure
    
    def _create_error_response(self, original_message: MCPMessage, error: str) -> MCPMessage:
        """Create error response message"""
        error_result = {
            'success': False,
            'error': error,
            'confidence_score': 0.0
        }
        
        response_message = create_coordinator_to_planner_message(
            task_type=TaskType.PROCESSING,
            payload=error_result,
            context=original_message.context
        )
        response_message.audit_trail = original_message.audit_trail.copy()
        
        return response_message
    
    def _update_metrics(self, success: bool, processing_time: float, chunks_count: int):
        """Update performance metrics"""
        self.performance_metrics['total_documents_processed'] += 1
        self.performance_metrics['total_chunks_generated'] += chunks_count
        
        # Update success rate
        total_docs = self.performance_metrics['total_documents_processed']
        if success:
            successful_docs = self.performance_metrics.get('successful_documents', 0) + 1
            self.performance_metrics['successful_documents'] = successful_docs
            self.performance_metrics['success_rate'] = successful_docs / total_docs
        
        # Update average processing time
        total_time = self.performance_metrics.get('total_processing_time', 0) + processing_time
        self.performance_metrics['total_processing_time'] = total_time
        self.performance_metrics['average_processing_time'] = total_time / total_docs
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.performance_metrics.copy()
    
    def get_worker_metrics(self) -> Dict[str, Any]:
        """Get metrics from all workers"""
        return {
            'pdf_processor': self.pdf_processor.get_performance_metrics(),
            'text_cleaner': self.text_cleaner.get_performance_metrics(),
            'embedding_generator': self.embedding_generator.get_performance_metrics(),
            'table_extractor': self.table_extractor.get_performance_metrics(),
            'live_context_worker': self.live_context_worker.get_performance_metrics()
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on coordinator and workers"""
        return {
            'coordinator': 'document_coordinator',
            'status': 'healthy',
            'performance_metrics': self.get_performance_metrics(),
            'worker_metrics': self.get_worker_metrics(),
            'timestamp': datetime.now().isoformat()
        }
