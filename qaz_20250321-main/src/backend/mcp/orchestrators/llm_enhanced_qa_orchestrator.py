#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-Enhanced QA Orchestrator

This module provides an enhanced QA orchestrator that integrates
LLM capabilities through the centralized LLM Gateway for comprehensive
document processing, analysis, and report generation.
"""

import logging
import time
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import uuid
from dataclasses import asdict, is_dataclass

from ...llm.unified_llm_gateway import UnifiedLLMGateway
from ...llm.llm_enhanced_agents import (
    LLMEnhancedDocumentClassifier,
    LLMEnhancedStandardsLibrarian,
    LLMEnhancedDigestGenerator
)
from ...core.qa_report_generator import QAReportGenerator
from ...core.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)


def dataclass_json_encoder(obj):
    """Custom JSON encoder for dataclasses"""
    if is_dataclass(obj):
        return asdict(obj)
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    else:
        return str(obj)


class LLMEnhancedQAOrchestrator:
    """LLM-enhanced QA orchestrator with comprehensive document processing"""
    
    def __init__(self, config_path: str, output_dir: str = "llm_enhanced_output"):
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize Unified LLM Gateway
        self.llm_gateway = UnifiedLLMGateway(
            experiment_name="qa-document-processing",
            enable_tracking=True
        )
        
        # Initialize LLM-enhanced agents
        self.document_classifier = LLMEnhancedDocumentClassifier(self.llm_gateway)
        self.standards_librarian = LLMEnhancedStandardsLibrarian(self.llm_gateway)
        self.digest_generator = LLMEnhancedDigestGenerator(self.llm_gateway)
        
        # Initialize core components
        self.document_processor = DocumentProcessor()
        self.qa_generator = QAReportGenerator()
        
        # Load QA criteria
        self.qa_criteria = self._load_qa_criteria()
        
        logger.info("LLM-Enhanced QA Orchestrator initialized")
    
    def _load_qa_criteria(self) -> Dict[str, Any]:
        """Load QA criteria from YAML configuration"""
        try:
            import yaml
            with open(self.config_path, 'r') as f:
                criteria = yaml.safe_load(f)
            logger.info(f"Loaded QA criteria from {self.config_path}")
            return criteria
        except Exception as e:
            logger.error(f"Failed to load QA criteria: {e}")
            return {}
    
    def process_qa_documents_with_llm(self, input_dir: str = "input", 
                                    batch_id: Optional[str] = None,
                                    budget_limit: Optional[float] = None) -> Dict[str, Any]:
        """Process documents with LLM enhancement and generate comprehensive reports"""
        
        # Generate batch ID if not provided
        if not batch_id:
            batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        # Start LLM batch processing
        self.llm_gateway.start_batch(batch_id, budget_limit)
        
        try:
            logger.info(f"Starting LLM-enhanced QA processing for batch: {batch_id}")
            
            # Step 1: Process documents
            documents = self._process_documents(input_dir)
            
            # Step 2: LLM-enhanced classification and analysis
            enhanced_results = self._enhance_documents_with_llm(documents)
            
            # Step 3: Generate QA reports
            qa_reports = self._generate_qa_reports(enhanced_results)
            
            # Step 4: Generate comprehensive output
            final_output = self._generate_comprehensive_output(
                batch_id, documents, enhanced_results, qa_reports
            )
            
            # Step 5: Generate LLM utilization report
            llm_report = self.llm_gateway.generate_batch_report(batch_id)
            
            # Step 6: Save all outputs
            self._save_outputs(batch_id, final_output, llm_report)
            
            logger.info(f"Completed LLM-enhanced QA processing for batch: {batch_id}")
            
            return {
                'batch_id': batch_id,
                'status': 'completed',
                'documents_processed': len(documents),
                'llm_utilization_report': llm_report,
                'output_files': self._get_output_files(batch_id)
            }
            
        except Exception as e:
            logger.error(f"Failed to process documents: {e}")
            return {
                'batch_id': batch_id,
                'status': 'failed',
                'error': str(e)
            }
        finally:
            # End LLM batch processing
            self.llm_gateway.end_batch(batch_id)
    
    def _process_documents(self, input_dir: str) -> List[Dict[str, Any]]:
        """Process documents using document processor"""
        logger.info(f"Processing documents from: {input_dir}")
        
        try:
            # Set the input directory for the document processor
            self.document_processor.input_dir = Path(input_dir)
            
            # Process all documents from the specified input directory
            document_results = self.document_processor.process_all_documents()
            
            # Extract metadata for each document
            enhanced_documents = []
            for doc_result in document_results:
                metadata = self.document_processor.extract_metadata_fields([doc_result])
                
                enhanced_doc = {
                    'filename': doc_result.get('filename', 'Unknown'),
                    'content': doc_result.get('content', ''),
                    'file_path': doc_result.get('file_path', ''),
                    'file_size': doc_result.get('file_size', 0),
                    'extracted_metadata': metadata,
                    'processing_timestamp': datetime.now().isoformat()
                }
                
                enhanced_documents.append(enhanced_doc)
            
            logger.info(f"Processed {len(enhanced_documents)} documents")
            return enhanced_documents
            
        except Exception as e:
            logger.error(f"Failed to process documents: {e}")
            return []
    
    def _enhance_documents_with_llm(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance documents with LLM analysis"""
        logger.info("Enhancing documents with LLM analysis")
        
        enhanced_documents = []
        
        for document in documents:
            try:
                enhanced_doc = document.copy()
                
                # Step 1: LLM classification
                classification_result = self.document_classifier.classify_with_llm(document)
                enhanced_doc['llm_classification'] = classification_result
                
                # Step 2: LLM metadata extraction
                llm_metadata = self.document_classifier.extract_metadata_with_llm(document)
                enhanced_doc['llm_metadata'] = llm_metadata
                
                # Step 3: Document-specific processing based on classification
                doc_type = classification_result.get('classification', 'unknown')
                
                if doc_type == 'standards':
                    # Extract standards and best practices
                    standards_result = self.standards_librarian.extract_standards_with_llm(document)
                    enhanced_doc['llm_standards_analysis'] = standards_result
                    
                else:
                    # Generate digest for non-standards documents
                    digest_result = self.digest_generator.generate_digest_with_llm(document)
                    enhanced_doc['llm_digest'] = digest_result
                
                enhanced_documents.append(enhanced_doc)
                
            except Exception as e:
                logger.error(f"Failed to enhance document {document.get('filename', 'Unknown')}: {e}")
                # Add document with error information
                document['llm_enhancement_error'] = str(e)
                enhanced_documents.append(document)
        
        logger.info(f"Enhanced {len(enhanced_documents)} documents with LLM")
        return enhanced_documents
    
    def _generate_qa_reports(self, enhanced_documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate QA reports for enhanced documents"""
        logger.info("Generating QA reports")
        
        qa_reports = []
        
        for enhanced_doc in enhanced_documents:
            try:
                # Extract metadata for QA analysis
                metadata = enhanced_doc.get('extracted_metadata', {})
                llm_metadata = enhanced_doc.get('llm_metadata', {})
                
                # Combine metadata sources
                combined_metadata = {**metadata, **llm_metadata}
                
                # Generate QA report
                # Create document structure expected by QA generator
                document_for_qa = [{
                    'filename': enhanced_doc.get('filename', 'Unknown'),
                    'content': enhanced_doc.get('content', ''),
                    'file_path': enhanced_doc.get('file_path', ''),
                    'file_size': enhanced_doc.get('file_size', 0)
                }]
                
                qa_report = self.qa_generator.generate_report(
                    document_for_qa,
                    combined_metadata
                )
                
                # Add LLM enhancement information
                qa_report_dict = qa_report.__dict__.copy()
                qa_report_dict['llm_enhancement'] = {
                    'classification': enhanced_doc.get('llm_classification', {}),
                    'llm_metadata': enhanced_doc.get('llm_metadata', {}),
                    'standards_analysis': enhanced_doc.get('llm_standards_analysis', {}),
                    'digest': enhanced_doc.get('llm_digest', {})
                }
                
                qa_reports.append(qa_report_dict)
                
            except Exception as e:
                logger.error(f"Failed to generate QA report for {enhanced_doc.get('filename', 'Unknown')}: {e}")
                qa_reports.append({
                    'filename': enhanced_doc.get('filename', 'Unknown'),
                    'error': f'QA report generation failed: {str(e)}'
                })
        
        logger.info(f"Generated {len(qa_reports)} QA reports")
        return qa_reports
    
    def _generate_comprehensive_output(self, batch_id: str, 
                                     documents: List[Dict[str, Any]],
                                     enhanced_documents: List[Dict[str, Any]],
                                     qa_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive output package"""
        logger.info("Generating comprehensive output package")
        
        # Calculate processing statistics
        total_documents = len(documents)
        successful_enhancements = len([d for d in enhanced_documents if 'llm_enhancement_error' not in d])
        successful_qa_reports = len([r for r in qa_reports if 'error' not in r])
        
        # Generate batch summary
        batch_summary = {
            'batch_id': batch_id,
            'processing_timestamp': datetime.now().isoformat(),
            'statistics': {
                'total_documents': total_documents,
                'successful_enhancements': successful_enhancements,
                'successful_qa_reports': successful_qa_reports,
                'enhancement_success_rate': successful_enhancements / total_documents if total_documents > 0 else 0,
                'qa_report_success_rate': successful_qa_reports / total_documents if total_documents > 0 else 0
            },
            'document_types': self._analyze_document_types(enhanced_documents),
            'processing_summary': self._generate_processing_summary(enhanced_documents)
        }
        
        comprehensive_output = {
            'batch_summary': batch_summary,
            'documents': documents,
            'enhanced_documents': enhanced_documents,
            'qa_reports': qa_reports,
            'metadata': {
                'generated_by': 'LLM-Enhanced QA Orchestrator',
                'version': '1.0',
                'processing_mode': 'llm_enhanced'
            }
        }
        
        return comprehensive_output
    
    def _analyze_document_types(self, enhanced_documents: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze document types from LLM classification"""
        type_counts = {}
        
        for doc in enhanced_documents:
            classification = doc.get('llm_classification', {})
            doc_type = classification.get('classification', 'unknown')
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        return type_counts
    
    def _generate_processing_summary(self, enhanced_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate processing summary"""
        summary = {
            'total_documents': len(enhanced_documents),
            'average_confidence': 0,
            'key_themes': [],
            'organizations': [],
            'date_range': {'earliest': None, 'latest': None}
        }
        
        if not enhanced_documents:
            return summary
        
        # Calculate average confidence
        confidences = []
        all_themes = []
        organizations = []
        dates = []
        
        for doc in enhanced_documents:
            classification = doc.get('llm_classification', {})
            metadata = doc.get('llm_metadata', {})
            
            # Confidence
            confidence = classification.get('confidence', 0)
            confidences.append(confidence)
            
            # Themes
            themes = classification.get('key_themes', [])
            all_themes.extend(themes)
            
            # Organizations
            org = metadata.get('organization', '')
            if org and org != 'Unknown':
                organizations.append(org)
            
            # Dates
            date = metadata.get('date', '')
            if date and date != 'Unknown':
                dates.append(date)
        
        summary['average_confidence'] = sum(confidences) / len(confidences) if confidences else 0
        summary['key_themes'] = list(set(all_themes))[:10]  # Top 10 unique themes
        summary['organizations'] = list(set(organizations))[:10]  # Top 10 unique organizations
        
        return summary
    
    def _save_outputs(self, batch_id: str, comprehensive_output: Dict[str, Any], 
                     llm_report: Dict[str, Any]):
        """Save all output files"""
        logger.info(f"Saving outputs for batch: {batch_id}")
        
        batch_dir = self.output_dir / batch_id
        batch_dir.mkdir(exist_ok=True)
        
        # Save comprehensive output
        output_file = batch_dir / "comprehensive_output.json"
        with open(output_file, 'w') as f:
            json.dump(comprehensive_output, f, indent=2, default=dataclass_json_encoder)
        
        # Save LLM utilization report
        llm_report_file = batch_dir / "llm_utilization_report.json"
        with open(llm_report_file, 'w') as f:
            json.dump(llm_report, f, indent=2, default=dataclass_json_encoder)
        
        # Generate individual document reports
        self._generate_individual_reports(batch_dir, comprehensive_output)
        
        # Generate summary report
        self._generate_summary_report(batch_dir, comprehensive_output, llm_report)
        
        logger.info(f"Saved outputs to: {batch_dir}")
    
    def _generate_individual_reports(self, batch_dir: Path, comprehensive_output: Dict[str, Any]):
        """Generate individual reports for each document"""
        enhanced_docs = comprehensive_output.get('enhanced_documents', [])
        
        for doc in enhanced_docs:
            filename = doc.get('filename', 'Unknown')
            safe_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_')).rstrip()
            
            doc_report_file = batch_dir / f"document_report_{safe_filename}.json"
            with open(doc_report_file, 'w') as f:
                json.dump(doc, f, indent=2, default=dataclass_json_encoder)
    
    def _generate_summary_report(self, batch_dir: Path, comprehensive_output: Dict[str, Any], 
                               llm_report: Dict[str, Any]):
        """Generate human-readable summary report"""
        summary_file = batch_dir / "summary_report.md"
        
        batch_summary = comprehensive_output.get('batch_summary', {})
        statistics = batch_summary.get('statistics', {})
        document_types = batch_summary.get('document_types', {})
        
        with open(summary_file, 'w') as f:
            f.write(f"# LLM-Enhanced QA Processing Summary\n\n")
            f.write(f"**Batch ID:** {batch_summary.get('batch_id', 'Unknown')}\n")
            f.write(f"**Processing Date:** {batch_summary.get('processing_timestamp', 'Unknown')}\n\n")
            
            f.write("## Processing Statistics\n\n")
            f.write(f"- **Total Documents:** {statistics.get('total_documents', 0)}\n")
            f.write(f"- **Successful Enhancements:** {statistics.get('successful_enhancements', 0)}\n")
            f.write(f"- **Successful QA Reports:** {statistics.get('successful_qa_reports', 0)}\n")
            f.write(f"- **Enhancement Success Rate:** {statistics.get('enhancement_success_rate', 0):.1%}\n")
            f.write(f"- **QA Report Success Rate:** {statistics.get('qa_report_success_rate', 0):.1%}\n\n")
            
            f.write("## Document Types\n\n")
            for doc_type, count in document_types.items():
                f.write(f"- **{doc_type}:** {count} documents\n")
            
            f.write("\n## LLM Utilization Summary\n\n")
            llm_summary = llm_report.get('processing_summary', {})
            f.write(f"- **Total LLM Calls:** {llm_summary.get('total_calls', 0)}\n")
            f.write(f"- **Total Tokens:** {llm_summary.get('total_tokens', 0):,}\n")
            f.write(f"- **Total Cost:** ${llm_summary.get('total_cost', 0):.4f}\n")
            f.write(f"- **Success Rate:** {llm_summary.get('success_rate', 0):.1%}\n")
            f.write(f"- **Average Response Time:** {llm_summary.get('average_response_time', 0):.2f}s\n")
    
    def _get_output_files(self, batch_id: str) -> Dict[str, str]:
        """Get list of output files for the batch"""
        batch_dir = self.output_dir / batch_id
        
        if not batch_dir.exists():
            return {}
        
        output_files = {}
        for file_path in batch_dir.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(self.output_dir)
                output_files[file_path.name] = str(relative_path)
        
        return output_files
