#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced QA Orchestrator

Enhanced QA orchestrator with document inspection, caption analysis, 
database integration, and quality prediction capabilities.
This is the intermediate level for progressive complexity architecture.
"""

import logging
import uuid
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from .simple_qa_orchestrator import SimpleQAOrchestrator
from ...core.database_connection_manager import get_database_manager
from ...coordinators.document_inspector_coordinator import DocumentInspectorCoordinator
from ...coordinators.caption_inspector_coordinator import CaptionInspectorCoordinator

logger = logging.getLogger(__name__)


class DatabaseQualityAnalyzer:
    """Analyzes quality patterns using database data"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
    
    def predict_quality(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Predict quality score based on historical patterns"""
        
        try:
            # Extract document characteristics
            doc_features = self.extract_document_features(document)
            
            # Query historical patterns
            query = """
            SELECT 
                AVG(quality_score) as predicted_score,
                COUNT(*) as pattern_count,
                ARRAY_AGG(DISTINCT common_issues) FILTER (WHERE common_issues IS NOT NULL) as likely_issues
            FROM qa_quality_patterns 
            WHERE document_type = %s
            AND quality_score > 0.7
            """
            
            result = self.db_manager.execute_query(query, (doc_features['type'],))
            
            if result and result[0]['predicted_score']:
                predicted_score = float(result[0]['predicted_score'])
                pattern_count = result[0]['pattern_count']
                likely_issues = result[0]['likely_issues'] or []
            else:
                predicted_score = 0.5
                pattern_count = 0
                likely_issues = []
            
            confidence = self.calculate_prediction_confidence(doc_features, pattern_count)
            
            return {
                'predicted_score': predicted_score,
                'likely_issues': likely_issues,
                'confidence': confidence,
                'pattern_count': pattern_count,
                'doc_features': doc_features
            }
            
        except Exception as e:
            logger.error(f"Error predicting quality: {e}")
            return {
                'predicted_score': 0.5,
                'likely_issues': [],
                'confidence': 0.0,
                'pattern_count': 0,
                'doc_features': {}
            }
    
    def extract_document_features(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features for pattern matching"""
        
        content = document.get('content', '')
        metadata = document.get('metadata', {})
        
        return {
            'type': metadata.get('type', 'unknown'),
            'length': len(content),
            'has_tables': self.detect_tables(content),
            'has_numbers': self.detect_numerical_content(content),
            'complexity_score': self.calculate_complexity(content),
            'word_count': len(content.split()),
            'sentence_count': len([s for s in content.split('.') if s.strip()])
        }
    
    def detect_tables(self, content: str) -> bool:
        """Detect if content contains tables"""
        import re
        table_patterns = [
            r'\|\s*[^|]+\s*\|',  # Markdown tables
            r'<table',           # HTML tables
            r'\t+',              # Tab-separated
            r',\s*,',            # CSV-like
        ]
        return any(re.search(pattern, content) for pattern in table_patterns)
    
    def detect_numerical_content(self, content: str) -> bool:
        """Detect if content contains numerical data"""
        import re
        number_patterns = [
            r'\d+\.\d+',         # Decimals
            r'\d+%',             # Percentages
            r'\$\d+',            # Currency
            r'\d{4}-\d{2}-\d{2}' # Dates
        ]
        return any(re.search(pattern, content) for pattern in number_patterns)
    
    def calculate_complexity(self, content: str) -> float:
        """Calculate document complexity score"""
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        if not sentences:
            return 0.0
        
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        unique_words = len(set(content.lower().split()))
        total_words = len(content.split())
        
        # Normalize complexity score (0-1)
        complexity = min(1.0, (avg_sentence_length / 20.0 + unique_words / total_words) / 2.0)
        return complexity
    
    def calculate_prediction_confidence(self, doc_features: Dict[str, Any], pattern_count: int) -> float:
        """Calculate confidence in quality prediction"""
        # More patterns = higher confidence
        pattern_confidence = min(1.0, pattern_count / 10.0)
        
        # Feature completeness = higher confidence
        feature_confidence = sum([
            bool(doc_features.get('type')),
            bool(doc_features.get('length', 0) > 100),
            bool(doc_features.get('word_count', 0) > 10)
        ]) / 3.0
        
        return (pattern_confidence + feature_confidence) / 2.0


class EnhancedQAOrchestrator(SimpleQAOrchestrator):
    """Enhanced QA orchestrator with document inspection and database integration"""
    
    def __init__(self):
        super().__init__()
        
        # Add enhanced resources
        self.document_inspector = DocumentInspectorCoordinator()
        self.caption_inspector = CaptionInspectorCoordinator()
        
        # Database integration
        try:
            self.db_manager = get_database_manager()
            self.quality_analyzer = DatabaseQualityAnalyzer(self.db_manager)
            self.db_available = True
            logger.info("✅ Database connection established for Enhanced QA Orchestrator")
        except Exception as e:
            logger.warning(f"⚠️ Database not available for Enhanced QA Orchestrator: {e}")
            self.db_manager = None
            self.quality_analyzer = None
            self.db_available = False
        
        logger.info(f"Enhanced QA Orchestrator initialized with session ID: {self.session_id}")
    
    def process_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced document processing with inspections and database integration"""
        
        start_time = datetime.now()
        
        try:
            logger.info("Starting enhanced document processing...")
            
            # Handle None document
            if document is None:
                raise ValueError("Document cannot be None")
            
            # Get Simple QA results first
            simple_results = super().process_document(document)
            
            # Perform enhanced inspections
            document_inspection = self._perform_document_inspection(document)
            caption_analysis = self._perform_caption_analysis(document)
            
            # Get quality prediction from database
            quality_prediction = self._predict_quality(document)
            
            # Calculate enhanced quality score
            enhanced_quality_score = self._calculate_enhanced_quality_score(
                simple_results['quality_score'],
                document_inspection,
                caption_analysis,
                quality_prediction
            )
            
            # Generate enhanced report
            enhanced_report = self._generate_enhanced_report(
                simple_results,
                document_inspection,
                caption_analysis,
                quality_prediction
            )
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Store quality metrics in database if available
            if self.db_available:
                self._store_quality_metrics(document, enhanced_quality_score, {
                    'document_inspection': document_inspection,
                    'caption_analysis': caption_analysis,
                    'quality_prediction': quality_prediction
                })
            
            return {
                'document': document,
                'simple_results': simple_results,
                'document_inspection': document_inspection,
                'caption_analysis': caption_analysis,
                'quality_prediction': quality_prediction,
                'enhanced_quality_score': enhanced_quality_score,
                'enhanced_report': enhanced_report,
                'processing_time_ms': processing_time,
                'session_id': self.session_id,
                'orchestrator_type': 'enhanced',
                'db_available': self.db_available,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced document processing: {e}")
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                'document': document,
                'simple_results': {},
                'document_inspection': {},
                'caption_analysis': {},
                'quality_prediction': {},
                'enhanced_quality_score': 0.0,
                'enhanced_report': {'error': str(e)},
                'processing_time_ms': processing_time,
                'session_id': self.session_id,
                'orchestrator_type': 'enhanced',
                'db_available': self.db_available,
                'status': 'error',
                'error': str(e)
            }
    
    def _perform_document_inspection(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive document inspection"""
        try:
            logger.info("Performing document inspection...")
            
            # Use document inspector coordinator
            inspection_result = self.document_inspector.inspect_document(document)
            
            return {
                'toc_analysis': inspection_result.get('toc_analysis', {}),
                'bibliography_analysis': inspection_result.get('bibliography_analysis', {}),
                'link_analysis': inspection_result.get('link_analysis', {}),
                'structure_analysis': inspection_result.get('structure_analysis', {}),
                'inspection_score': inspection_result.get('overall_score', 0.0),
                'issues_found': inspection_result.get('issues', []),
                'recommendations': inspection_result.get('recommendations', [])
            }
            
        except Exception as e:
            logger.error(f"Error in document inspection: {e}")
            return {
                'toc_analysis': {},
                'bibliography_analysis': {},
                'link_analysis': {},
                'structure_analysis': {},
                'inspection_score': 0.0,
                'issues_found': [f"Inspection error: {str(e)}"],
                'recommendations': ['Check document format and accessibility']
            }
    
    def _perform_caption_analysis(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Perform caption analysis"""
        try:
            logger.info("Performing caption analysis...")
            
            # Use caption inspector coordinator
            caption_result = self.caption_inspector.analyze_document(document)
            
            return {
                'total_captions': caption_result.get('total_captions', 0),
                'captions_with_numbers': caption_result.get('captions_with_numbers', 0),
                'captions_with_issues': caption_result.get('captions_with_issues', 0),
                'caption_quality_score': caption_result.get('quality_score', 0.0),
                'caption_details': caption_result.get('caption_details', []),
                'caption_recommendations': caption_result.get('recommendations', [])
            }
            
        except Exception as e:
            logger.error(f"Error in caption analysis: {e}")
            return {
                'total_captions': 0,
                'captions_with_numbers': 0,
                'captions_with_issues': 0,
                'caption_quality_score': 0.0,
                'caption_details': [],
                'caption_recommendations': ['Check document for caption extraction issues']
            }
    
    def _predict_quality(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Predict quality using database patterns"""
        if not self.db_available or not self.quality_analyzer:
            return {
                'predicted_score': 0.5,
                'likely_issues': [],
                'confidence': 0.0,
                'pattern_count': 0,
                'doc_features': {}
            }
        
        try:
            logger.info("Predicting quality using database patterns...")
            return self.quality_analyzer.predict_quality(document)
            
        except Exception as e:
            logger.error(f"Error in quality prediction: {e}")
            return {
                'predicted_score': 0.5,
                'likely_issues': [],
                'confidence': 0.0,
                'pattern_count': 0,
                'doc_features': {}
            }
    
    def _calculate_enhanced_quality_score(self, simple_score: float, 
                                        document_inspection: Dict[str, Any],
                                        caption_analysis: Dict[str, Any],
                                        quality_prediction: Dict[str, Any]) -> float:
        """Calculate enhanced quality score combining all factors"""
        
        # Base score from simple QA
        base_score = simple_score
        
        # Document inspection score
        inspection_score = document_inspection.get('inspection_score', 0.0)
        
        # Caption analysis score
        caption_score = caption_analysis.get('caption_quality_score', 0.0)
        
        # Quality prediction score
        prediction_score = quality_prediction.get('predicted_score', 0.5)
        prediction_confidence = quality_prediction.get('confidence', 0.0)
        
        # Weighted combination
        enhanced_score = (
            base_score * 0.25 +
            inspection_score * 0.25 +
            caption_score * 0.20 +
            (prediction_score * prediction_confidence) * 0.30
        )
        
        return min(1.0, max(0.0, enhanced_score))
    
    def _generate_enhanced_report(self, simple_results: Dict[str, Any],
                                document_inspection: Dict[str, Any],
                                caption_analysis: Dict[str, Any],
                                quality_prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive enhanced report"""
        
        return {
            'summary': {
                'overall_quality_score': simple_results.get('quality_score', 0.0),
                'enhanced_quality_score': self._calculate_enhanced_quality_score(
                    simple_results.get('quality_score', 0.0),
                    document_inspection,
                    caption_analysis,
                    quality_prediction
                ),
                'document_inspection_score': document_inspection.get('inspection_score', 0.0),
                'caption_quality_score': caption_analysis.get('caption_quality_score', 0.0),
                'predicted_quality_score': quality_prediction.get('predicted_score', 0.0)
            },
            'document_inspection': {
                'toc_quality': document_inspection.get('toc_analysis', {}),
                'bibliography_quality': document_inspection.get('bibliography_analysis', {}),
                'link_quality': document_inspection.get('link_analysis', {}),
                'structure_quality': document_inspection.get('structure_analysis', {})
            },
            'caption_analysis': {
                'total_captions': caption_analysis.get('total_captions', 0),
                'numbered_captions': caption_analysis.get('captions_with_numbers', 0),
                'caption_issues': caption_analysis.get('captions_with_issues', 0),
                'caption_recommendations': caption_analysis.get('caption_recommendations', [])
            },
            'quality_prediction': {
                'predicted_score': quality_prediction.get('predicted_score', 0.0),
                'confidence': quality_prediction.get('confidence', 0.0),
                'likely_issues': quality_prediction.get('likely_issues', []),
                'pattern_count': quality_prediction.get('pattern_count', 0)
            },
            'recommendations': self._generate_recommendations(
                document_inspection,
                caption_analysis,
                quality_prediction
            )
        }
    
    def _generate_recommendations(self, document_inspection: Dict[str, Any],
                                caption_analysis: Dict[str, Any],
                                quality_prediction: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Document inspection recommendations
        inspection_issues = document_inspection.get('issues_found', [])
        if inspection_issues:
            recommendations.extend([f"Inspection: {issue}" for issue in inspection_issues])
        
        # Caption recommendations
        caption_recs = caption_analysis.get('caption_recommendations', [])
        if caption_recs:
            recommendations.extend([f"Caption: {rec}" for rec in caption_recs])
        
        # Quality prediction recommendations
        likely_issues = quality_prediction.get('likely_issues', [])
        if likely_issues:
            recommendations.extend([f"Prediction: {issue}" for issue in likely_issues])
        
        return recommendations
    
    def _store_quality_metrics(self, document: Dict[str, Any], 
                             quality_score: float, 
                             analysis_data: Dict[str, Any]) -> None:
        """Store quality metrics in database"""
        if not self.db_available:
            return
        
        try:
            # Store quality metrics
            metrics_data = {
                'session_id': self.session_id,
                'document_hash': str(hash(document.get('content', ''))),
                'quality_score': quality_score,
                'document_inspection_score': analysis_data.get('document_inspection', {}).get('inspection_score', 0.0),
                'caption_quality_score': analysis_data.get('caption_analysis', {}).get('caption_quality_score', 0.0),
                'predicted_quality_score': analysis_data.get('quality_prediction', {}).get('predicted_score', 0.0),
                'processing_time_ms': analysis_data.get('processing_time_ms', 0),
                'created_at': datetime.now()
            }
            
            # Insert into database
            with self.db_manager.get_cursor() as cursor:
                cursor.execute("""
                    INSERT INTO qa_quality_metrics 
                    (session_id, document_hash, quality_score, document_inspection_score, 
                     caption_quality_score, predicted_quality_score, processing_time_ms, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    metrics_data['session_id'],
                    metrics_data['document_hash'],
                    metrics_data['quality_score'],
                    metrics_data['document_inspection_score'],
                    metrics_data['caption_quality_score'],
                    metrics_data['predicted_quality_score'],
                    metrics_data['processing_time_ms'],
                    metrics_data['created_at']
                ))
            
            logger.info(f"Stored quality metrics for session {self.session_id}")
            
        except Exception as e:
            logger.error(f"Error storing quality metrics: {e}")
