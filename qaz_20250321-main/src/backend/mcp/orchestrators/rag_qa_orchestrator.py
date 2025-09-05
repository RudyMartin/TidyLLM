#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG QA Orchestrator with Database Integration

Specialized orchestrator for Research-Augmented Generation (RAG) workflows,
with full integration to Aurora PostgreSQL database for persistent vector storage.
"""

import logging
import time
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import uuid
from dataclasses import asdict, is_dataclass
import os

import dspy
from ...core.datamart_numpy_substitution import np
import pandas as pd
import hashlib

from ...core.document_processor import DocumentProcessor
from ...core.embedding_helper import EmbeddingHelper
from ...core.dspy_config import DSPyConfig
from ...core.database_connection_manager import get_database_manager

logger = logging.getLogger(__name__)


class RAGQAOrchestrator:
    """RAG QA orchestrator with database integration for research queries and whitepaper analysis"""
    
    def __init__(self, config_path: str = "config/qa_criteria_full.yaml"):
        self.config_path = Path(config_path)
        self.output_dir = Path("rag_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.embedding_helper = EmbeddingHelper()
        self.dspy_config = DSPyConfig(provider="openai")
        
        # Initialize enhanced embedding helper with model tracking
        self.embedding_helper = EmbeddingHelper(target_dimensions=1024)
        self.embedding_model = self.embedding_helper.embedding_model
        self.embedding_dimension = self.embedding_helper.target_dimensions
        
        # Log embedding model information
        model_info = self.embedding_helper.get_model_info()
        logger.info(f"Embedding model initialized: {model_info['model_metadata']['model_name']}")
        logger.info(f"Original dimensions: {model_info['model_metadata']['original_dimensions']}")
        logger.info(f"Target dimensions: {model_info['model_metadata']['target_dimensions']}")
        if model_info['model_metadata']['needs_padding']:
            logger.info(f"Padding method: {model_info['model_metadata']['padding_dimensions']} zeros")
        elif model_info['model_metadata']['needs_truncation']:
            logger.info(f"Truncation method: {model_info['model_metadata']['truncation_dimensions']} dimensions")
        
        # Database connection
        self.db_connection = None
        self._setup_database_connection()
        
        # Initialize DSPy RAG components
        self._setup_dspy_rag()
        
        logger.info("RAG QA Orchestrator with database integration initialized")
    
    def _setup_database_connection(self):
        """Setup database connection using centralized manager"""
        try:
            # Use centralized database connection manager
            self.db_manager = get_database_manager()
            
            if self.db_manager.test_connection():
                logger.info("✅ Database connection established via centralized manager")
                # Ensure pgvector extension is available
                self.db_manager.ensure_extension("vector")
                self.db_connection = True  # Flag to indicate connection is available
            else:
                logger.warning("Database connection not available. Using local file storage only.")
                self.db_connection = None
                
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            self.db_connection = None
    
    def _ensure_pgvector_extension(self):
        """Ensure pgvector extension is available"""
        if hasattr(self, 'db_manager') and self.db_manager:
            self.db_manager.ensure_extension("vector")
    
    def _setup_dspy_rag(self):
        """Setup DSPy RAG signatures and predictors"""
        
        # Define RAG signature for research Q&A
        class ResearchQA(dspy.Signature):
            """Research Q&A signature for whitepaper analysis"""
            question = dspy.InputField(desc="The research question to answer")
            context = dspy.InputField(desc="Relevant context from the whitepaper")
            answer = dspy.OutputField(desc="Comprehensive answer based on the context")
            reasoning = dspy.OutputField(desc="Step-by-step reasoning for the answer")
            confidence = dspy.OutputField(desc="Confidence level (0-1) in the answer")
            sources = dspy.OutputField(desc="Specific sources/citations from the context")
        
        # Define analysis signature
        class WhitepaperAnalysis(dspy.Signature):
            """Whitepaper analysis signature"""
            content = dspy.InputField(desc="Whitepaper content to analyze")
            key_findings = dspy.OutputField(desc="Key findings and insights")
            methodology = dspy.OutputField(desc="Research methodology used")
            conclusions = dspy.OutputField(desc="Main conclusions and implications")
            limitations = dspy.OutputField(desc="Limitations and caveats")
            future_work = dspy.OutputField(desc="Suggested future research directions")
        
        # Create DSPy predictors
        self.rag_predictor = dspy.ChainOfThought(ResearchQA)
        self.analysis_predictor = dspy.ChainOfThought(WhitepaperAnalysis)
        
        logger.info("DSPy RAG components initialized")
    
    def process_whitepaper_rag(self, 
                              files: List[Dict[str, Any]],
                              research_questions: List[str] = None,
                              analysis_depth: str = "comprehensive",
                              chunk_size: int = 512,
                              chunk_overlap: int = 50,
                              store_in_database: bool = True) -> Dict[str, Any]:
        """Process whitepaper with RAG capabilities and database storage"""
        
        batch_id = f"rag_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        try:
            logger.info(f"🚀 Starting RAG processing with database integration for batch: {batch_id}")
            
            # Step 1: Process and chunk documents
            logger.info("Step 1: Processing and chunking documents")
            chunked_documents = self._process_and_chunk_documents(files, chunk_size, chunk_overlap)
            
            # Step 2: Store documents in database (if available)
            if store_in_database and hasattr(self, 'db_manager') and self.db_manager:
                logger.info("Step 2: Storing documents in database")
                doc_ids = self._store_documents_in_database(files, chunked_documents)
            else:
                doc_ids = [f"local_{uuid.uuid4()}" for _ in files]
                logger.info("Step 2: Using local storage (database not available)")
            
            # Step 3: Generate embeddings and store in database
            logger.info("Step 3: Generating embeddings and storing in database")
            if hasattr(self, 'db_manager') and self.db_manager:
                vector_store_info = self._store_embeddings_in_database(chunked_documents, doc_ids)
            else:
                vector_store_info = self._build_local_vector_store(chunked_documents)
            
            # Step 4: Perform comprehensive whitepaper analysis
            logger.info("Step 4: Performing comprehensive whitepaper analysis")
            analysis_result = self._analyze_whitepaper(chunked_documents, analysis_depth)
            
            # Step 5: Answer research questions using RAG
            logger.info("Step 5: Answering research questions using RAG")
            qa_results = []
            if research_questions:
                qa_results = self._answer_research_questions_database(
                    research_questions, doc_ids
                )
            
            # Step 6: Generate comprehensive output
            logger.info("Step 6: Generating comprehensive output")
            comprehensive_output = self._generate_rag_output(
                batch_id, chunked_documents, analysis_result, qa_results, vector_store_info
            )
            
            # Step 7: Save outputs
            self._save_rag_outputs(batch_id, comprehensive_output)
            
            logger.info(f"✅ RAG processing with database integration completed for batch: {batch_id}")
            
            return {
                'batch_id': batch_id,
                'status': 'completed',
                'documents_processed': len(files),
                'chunks_created': len(chunked_documents),
                'questions_answered': len(qa_results),
                'analysis_completed': True,
                'database_used': hasattr(self, 'db_manager') and self.db_manager is not None,
                'doc_ids': doc_ids,
                'output_files': self._get_output_files(batch_id),
                'rag_results': comprehensive_output
            }
            
        except Exception as e:
            logger.error(f"RAG processing failed: {e}")
            return {
                'batch_id': batch_id,
                'status': 'failed',
                'error': str(e)
            }
    
    def _store_documents_in_database(self, 
                                   files: List[Dict[str, Any]], 
                                   chunked_documents: List[Dict[str, Any]]) -> List[str]:
        """Store documents and metadata in database"""
        
        doc_ids = []
        
        for file in files:
            try:
                filename = file.get('filename', 'Unknown')
                content = file.get('content', '')
                file_size = file.get('size', 0)
                
                # Generate document ID
                doc_id = f"doc_{uuid.uuid4().hex[:8]}"
                doc_ids.append(doc_id)
                
                # Calculate content hash
                if isinstance(content, bytes):
                    content_str = content.decode('utf-8', errors='ignore')
                else:
                    content_str = str(content)
                
                content_hash = hashlib.sha1(content_str.encode()).hexdigest()
                
                # Store document metadata
                with self.db_manager.get_cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO document_metadata 
                        (doc_id, title, doc_type, status, total_chunks, file_size_bytes, ingested_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (doc_id) DO UPDATE SET
                        last_processed = now()
                    """, (
                        doc_id,
                        filename,
                        'whitepaper',
                        'processed',
                        len([c for c in chunked_documents if c['filename'] == filename]),
                        file_size,
                        datetime.now()
                    ))
                
                # Store document chunks
                for chunk in chunked_documents:
                    if chunk['filename'] == filename:
                        with self.db_manager.get_cursor() as cursor:
                            cursor.execute("""
                                INSERT INTO document_chunks 
                                (doc_id, chunk_id, chunk_text, char_count, embedding_model, created_at)
                                VALUES (%s, %s, %s, %s, %s, %s)
                                ON CONFLICT (doc_id, chunk_id) DO UPDATE SET
                                chunk_text = EXCLUDED.chunk_text,
                                char_count = EXCLUDED.char_count,
                                embedding_model = EXCLUDED.embedding_model
                            """, (
                                doc_id,
                                chunk['chunk_id'],
                                chunk['content'],
                                len(chunk['content']),
                                'all-mpnet-base-v2',  # Set default embedding model
                                datetime.now()
                            ))
                logger.info(f"Stored document {filename} with ID {doc_id}")
                
            except Exception as e:
                logger.error(f"Failed to store document {filename}: {e}")
        
        return doc_ids
    
    def _store_embeddings_in_database(self, 
                                    chunked_documents: List[Dict[str, Any]], 
                                    doc_ids: List[str]) -> Dict[str, Any]:
        """Store embeddings in database using pgvector with enhanced tracking"""
        
        embedding_count = 0
        model_info = self.embedding_helper.get_model_info()
        
        for chunk_doc in chunked_documents:
            try:
                # Generate embedding with metadata tracking
                embedding, metadata = self.embedding_helper.generate_embedding(
                    chunk_doc['content'], 
                    chunk_doc['chunk_id']
                )
                
                # Store in database with comprehensive metadata
                with self.db_manager.get_cursor() as cursor:
                    cursor.execute("""
                        UPDATE document_chunks 
                        SET embedding = %s, 
                            embedding_model = %s,
                            content_hash = %s
                        WHERE doc_id = %s AND chunk_id = %s
                    """, (
                        embedding.tolist(),  # Already adjusted to 1024 dimensions
                        metadata.model_name,
                        metadata.content_hash,
                        chunk_doc['doc_id'] if 'doc_id' in chunk_doc else doc_ids[0],
                        chunk_doc['chunk_id']
                    ))
                
                embedding_count += 1
                
                # Log embedding generation details
                logger.debug(f"Generated embedding for {chunk_doc['chunk_id']}: "
                           f"{metadata.original_dimensions} -> {len(embedding)} dimensions "
                           f"({metadata.padding_method})")
                
            except Exception as e:
                logger.error(f"Failed to store embedding for {chunk_doc['chunk_id']}: {e}")
        

        
        logger.info(f"Stored {embedding_count} embeddings in database")
        logger.info(f"Model: {model_info['model_metadata']['model_name']}")
        logger.info(f"Dimensions: {model_info['model_metadata']['original_dimensions']} -> {model_info['model_metadata']['target_dimensions']}")
        
        return {
            'embedding_count': embedding_count,
            'embedding_model': model_info['model_metadata']['model_name'],
            'original_dimensions': model_info['model_metadata']['original_dimensions'],
            'target_dimensions': model_info['model_metadata']['target_dimensions'],
            'padding_method': model_info['model_metadata']['needs_padding'],
            'storage_type': 'database'
        }
    
    def _build_local_vector_store(self, chunked_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build local vector store (fallback when database not available)"""
        
        vector_store = {
            'embeddings': [],
            'chunk_ids': [],
            'metadata': []
        }
        
        for chunk_doc in chunked_documents:
            try:
                # Generate embedding using centralized helper
                embedding, metadata = self.embedding_helper.generate_embedding(
                    chunk_doc['content'], 
                    chunk_doc['chunk_id']
                )
                vector_store['embeddings'].append(embedding)
                vector_store['chunk_ids'].append(chunk_doc['chunk_id'])
                vector_store['metadata'].append(chunk_doc['metadata'])
            except Exception as e:
                logger.error(f"Failed to create embedding for {chunk_doc['chunk_id']}: {e}")
        
        # Convert embeddings to DataMart format
        vector_store['embeddings'] = np.array(vector_store['embeddings'])
        # Convert to list if it's a datatable Frame
        if hasattr(vector_store['embeddings'], 'to_list'):
            vector_store['embeddings'] = vector_store['embeddings'].to_list()
        
        return {
            'embedding_count': len(vector_store['embeddings']),
            'embedding_model': 'all-MiniLM-L6-v2',
            'storage_type': 'local_memory',
            'vector_store': vector_store
        }
    
    def _answer_research_questions_database(self, 
                                          questions: List[str],
                                          doc_ids: List[str]) -> List[Dict[str, Any]]:
        """Answer research questions using database-stored embeddings"""
        
        qa_results = []
        
        for question in questions:
            try:
                if hasattr(self, 'db_manager') and self.db_manager:
                    # Use database for retrieval
                    relevant_chunks = self._retrieve_from_database(question, doc_ids, top_k=5)
                else:
                    # Fallback to local retrieval
                    relevant_chunks = self._retrieve_relevant_chunks_local(question, top_k=5)
                
                if relevant_chunks:
                    # Create context from relevant chunks
                    context = "\n\n".join([chunk['content'] for chunk in relevant_chunks])
                    
                    # Use DSPy RAG to answer
                    rag_result = self.rag_predictor(
                        question=question,
                        context=context
                    )
                    
                    qa_result = {
                        'question': question,
                        'answer': rag_result.answer,
                        'reasoning': rag_result.reasoning,
                        'confidence': rag_result.confidence,
                        'sources': rag_result.sources,
                        'relevant_chunks': [chunk['chunk_id'] for chunk in relevant_chunks],
                        'context_length': len(context),
                        'retrieval_method': 'database' if hasattr(self, 'db_manager') and self.db_manager else 'local'
                    }
                else:
                    qa_result = {
                        'question': question,
                        'error': 'No relevant chunks found'
                    }
                
                qa_results.append(qa_result)
                
            except Exception as e:
                logger.error(f"Failed to answer question '{question}': {e}")
                qa_results.append({
                    'question': question,
                    'error': f'Failed to answer: {str(e)}'
                })
        
        return qa_results
    
    def _retrieve_from_database(self, 
                              query: str, 
                              doc_ids: List[str], 
                              top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks from database using pgvector"""
        
        try:
            # Generate query embedding using centralized helper
            query_embedding, metadata = self.embedding_helper.generate_embedding(query, "query")
            
            # Query database using pgvector similarity search
            with self.db_manager.get_cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT 
                        dc.chunk_id,
                        dc.chunk_text as content,
                        dc.embedding,
                        dc.doc_id,
                        1 - (dc.embedding <=> %s) as similarity_score
                    FROM document_chunks dc
                    WHERE dc.doc_id = ANY(%s)
                    AND dc.embedding IS NOT NULL
                    ORDER BY dc.embedding <=> %s
                    LIMIT %s
                """, (
                    query_embedding.tolist(),
                    doc_ids,
                    query_embedding.tolist(),
                    top_k
                ))
                
                results = cursor.fetchall()
                
                relevant_chunks = []
                for row in results:
                    relevant_chunks.append({
                        'chunk_id': row['chunk_id'],
                        'content': row['content'],
                        'similarity_score': float(row['similarity_score']),
                        'doc_id': row['doc_id']
                    })
                
                return relevant_chunks
                
        except Exception as e:
            logger.error(f"Database retrieval failed: {e}")
            return []
    
    def _retrieve_relevant_chunks_local(self, 
                                      query: str, 
                                      top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks from local vector store (fallback)"""
        
        # This would use the local vector store if database is not available
        # Implementation depends on how local storage is maintained
        return []
    
    def _process_and_chunk_documents(self, 
                                   files: List[Dict[str, Any]], 
                                   chunk_size: int = 512,
                                   chunk_overlap: int = 50) -> List[Dict[str, Any]]:
        """Process documents and create chunks for RAG"""
        
        chunked_documents = []
        
        for file in files:
            try:
                filename = file.get('filename', 'Unknown')
                content = file.get('content', '')
                
                if isinstance(content, bytes):
                    content = content.decode('utf-8', errors='ignore')
                
                # Create chunks
                chunks = self._create_text_chunks(content, chunk_size, chunk_overlap)
                
                for i, chunk in enumerate(chunks):
                    chunk_doc = {
                        'chunk_id': f"{filename}_chunk_{i}",
                        'filename': filename,
                        'chunk_index': i,
                        'content': chunk,
                        'chunk_size': len(chunk),
                        'metadata': {
                            'source_file': filename,
                            'chunk_number': i,
                            'total_chunks': len(chunks)
                        }
                    }
                    chunked_documents.append(chunk_doc)
                
                logger.info(f"Created {len(chunks)} chunks for {filename}")
                
            except Exception as e:
                logger.error(f"Failed to process {filename}: {e}")
        
        return chunked_documents
    
    def _create_text_chunks(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Create overlapping text chunks"""
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # If this isn't the last chunk, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start, end - 100), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def _analyze_whitepaper(self, 
                          chunked_documents: List[Dict[str, Any]], 
                          analysis_depth: str) -> Dict[str, Any]:
        """Perform comprehensive whitepaper analysis"""
        
        try:
            # Combine all chunks for analysis
            full_content = "\n\n".join([chunk['content'] for chunk in chunked_documents])
            
            # Use DSPy for analysis
            analysis_result = self.analysis_predictor(
                content=full_content
            )
            
            return {
                'key_findings': analysis_result.key_findings,
                'methodology': analysis_result.methodology,
                'conclusions': analysis_result.conclusions,
                'limitations': analysis_result.limitations,
                'future_work': analysis_result.future_work,
                'analysis_depth': analysis_depth,
                'chunks_analyzed': len(chunked_documents)
            }
            
        except Exception as e:
            logger.error(f"Whitepaper analysis failed: {e}")
            return {
                'error': f'Analysis failed: {str(e)}',
                'analysis_depth': analysis_depth
            }
    
    def _generate_rag_output(self, 
                           batch_id: str,
                           chunked_documents: List[Dict[str, Any]],
                           analysis_result: Dict[str, Any],
                           qa_results: List[Dict[str, Any]],
                           vector_store_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive RAG output"""
        
        # Calculate statistics
        total_chunks = len(chunked_documents)
        successful_qa = len([r for r in qa_results if 'error' not in r])
        
        rag_output = {
            'batch_id': batch_id,
            'processing_timestamp': datetime.now().isoformat(),
            'statistics': {
                'total_chunks': total_chunks,
                'questions_answered': len(qa_results),
                'successful_answers': successful_qa,
                'qa_success_rate': successful_qa / len(qa_results) if qa_results else 0
            },
            'whitepaper_analysis': analysis_result,
            'qa_results': qa_results,
            'vector_store_info': vector_store_info,
            'database_info': {
                'connected': hasattr(self, 'db_manager') and self.db_manager is not None,
                'storage_type': vector_store_info.get('storage_type', 'unknown'),
                'embedding_count': vector_store_info.get('embedding_count', 0)
            },
            'chunk_summary': {
                'total_chunks': total_chunks,
                'avg_chunk_size': sum(len(chunk['content']) for chunk in chunked_documents) / total_chunks if total_chunks > 0 else 0,
                'chunk_distribution': self._analyze_chunk_distribution(chunked_documents)
            },
            'metadata': {
                'generated_by': 'RAG QA Orchestrator with Database Integration',
                'version': '2.0',
                'processing_mode': 'rag_qa_with_database'
            }
        }
        
        return rag_output
    
    def _analyze_chunk_distribution(self, chunked_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze distribution of document chunks"""
        
        chunk_sizes = [len(chunk['content']) for chunk in chunked_documents]
        
        return {
            'min_size': min(chunk_sizes) if chunk_sizes else 0,
            'max_size': max(chunk_sizes) if chunk_sizes else 0,
            'avg_size': sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0,
            'total_size': sum(chunk_sizes)
        }
    
    def _save_rag_outputs(self, batch_id: str, comprehensive_output: Dict[str, Any]):
        """Save RAG outputs to files"""
        
        # Save comprehensive output
        output_file = self.output_dir / f"rag_output_{batch_id}.json"
        with open(output_file, 'w') as f:
            json.dump(comprehensive_output, f, indent=2, default=str)
        
        # Save Q&A results separately
        if comprehensive_output.get('qa_results'):
            qa_file = self.output_dir / f"qa_results_{batch_id}.json"
            with open(qa_file, 'w') as f:
                json.dump(comprehensive_output['qa_results'], f, indent=2, default=str)
        
        # Save analysis separately
        if comprehensive_output.get('whitepaper_analysis'):
            analysis_file = self.output_dir / f"analysis_{batch_id}.json"
            with open(analysis_file, 'w') as f:
                json.dump(comprehensive_output['whitepaper_analysis'], f, indent=2, default=str)
        
        logger.info(f"Saved RAG outputs to {self.output_dir}")
    
    def _get_output_files(self, batch_id: str) -> List[str]:
        """Get list of output files for this batch"""
        
        output_files = []
        for file in self.output_dir.glob(f"*{batch_id}*"):
            output_files.append(str(file))
        
        return output_files
    
    def interactive_qa_session(self, 
                             doc_ids: List[str] = None) -> Dict[str, Any]:
        """Start an interactive Q&A session"""
        
        session_id = str(uuid.uuid4())
        
        return {
            'session_id': session_id,
            'database_connected': hasattr(self, 'db_manager') and self.db_manager is not None,
            'doc_ids_available': doc_ids if doc_ids else [],
            'status': 'ready_for_questions',
            'retrieval_method': 'database' if hasattr(self, 'db_manager') and self.db_manager else 'local'
        }
    
    def close_database_connection(self):
        """Close database connection"""
        if hasattr(self, 'db_manager') and self.db_manager:
            self.db_manager.close_pool()
            logger.info("Database connection pool closed")
