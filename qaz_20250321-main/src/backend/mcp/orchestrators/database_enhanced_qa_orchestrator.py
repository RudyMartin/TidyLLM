#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database-Enhanced QA Orchestrator

This orchestrator integrates database-driven quality control mechanisms
to enhance QA accuracy and reliability through:
- Historical pattern analysis
- Numerical validation
- Contextual enhancement
- Real-time quality monitoring
"""

import logging
import uuid
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

from .qa_orchestrator import QAOrchestrator
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
        table_indicators = ['|', '\t', 'table', 'row', 'column']
        return any(indicator in content.lower() for indicator in table_indicators)
    
    def detect_numerical_content(self, content: str) -> bool:
        """Detect if content contains numerical data"""
        import re
        numbers = re.findall(r'\d+\.?\d*', content)
        return len(numbers) > 5  # Threshold for numerical content
    
    def calculate_complexity(self, content: str) -> float:
        """Calculate content complexity score"""
        # Simple complexity calculation based on sentence length and vocabulary
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        if not sentences:
            return 0.0
        
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        unique_words = len(set(content.lower().split()))
        total_words = len(content.split())
        
        if total_words == 0:
            return 0.0
        
        vocabulary_diversity = unique_words / total_words
        
        # Normalize to 0-1 scale
        complexity = min(1.0, (avg_sentence_length * 0.1 + vocabulary_diversity * 5) / 2)
        return complexity
    
    def calculate_prediction_confidence(self, features: Dict[str, Any], pattern_count: int) -> float:
        """Calculate confidence in prediction"""
        # Higher confidence with more patterns and better feature extraction
        base_confidence = min(1.0, pattern_count / 10.0)  # Max confidence with 10+ patterns
        
        # Adjust based on feature quality
        feature_quality = 0.0
        if features.get('type') != 'unknown':
            feature_quality += 0.2
        if features.get('has_numbers'):
            feature_quality += 0.2
        if features.get('has_tables'):
            feature_quality += 0.2
        if features.get('complexity_score', 0) > 0.3:
            feature_quality += 0.2
        
        return min(1.0, base_confidence + feature_quality)


class NumericalValidator:
    """Validates numerical data using database rules"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
    
    def validate_data(self, numerical_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate numerical data against database rules"""
        
        validation_results = []
        
        for data_point in numerical_data:
            try:
                # Get applicable validation rules
                rules = self.get_validation_rules(data_point.get('type', 'general'))
                
                for rule in rules:
                    validation_result = self.apply_validation_rule(data_point, rule)
                    validation_results.append(validation_result)
                    
            except Exception as e:
                logger.error(f"Error validating data point {data_point}: {e}")
                validation_results.append({
                    'field': data_point.get('field_name', 'unknown'),
                    'value': data_point.get('value'),
                    'valid': False,
                    'error': str(e),
                    'confidence': 0.0
                })
        
        return validation_results
    
    def get_validation_rules(self, data_type: str) -> List[Dict[str, Any]]:
        """Get validation rules for data type"""
        
        query = """
        SELECT rule_name, rule_type, rule_definition, confidence_threshold
        FROM numerical_validation_rules
        WHERE rule_definition->>'data_type' = %s
        OR rule_definition->>'data_type' = 'general'
        ORDER BY confidence_threshold DESC
        """
        
        try:
            result = self.db_manager.execute_query(query, (data_type,))
            return result or []
        except Exception as e:
            logger.error(f"Error getting validation rules: {e}")
            return []
    
    def apply_validation_rule(self, data_point: Dict[str, Any], rule: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a specific validation rule"""
        
        try:
            rule_type = rule.get('rule_type', 'unknown')
            
            if rule_type == 'range':
                return self.validate_range(data_point, rule)
            elif rule_type == 'formula':
                return self.validate_formula(data_point, rule)
            elif rule_type == 'consistency':
                return self.validate_consistency(data_point, rule)
            else:
                return {
                    'field': data_point.get('field_name', 'unknown'),
                    'value': data_point.get('value'),
                    'rule': rule.get('rule_name', 'unknown'),
                    'valid': False,
                    'error': f'Unknown rule type: {rule_type}',
                    'confidence': 0.0
                }
                
        except Exception as e:
            return {
                'field': data_point.get('field_name', 'unknown'),
                'value': data_point.get('value'),
                'rule': rule.get('rule_name', 'unknown'),
                'valid': False,
                'error': str(e),
                'confidence': 0.0
            }
    
    def validate_range(self, data_point: Dict[str, Any], rule: Dict[str, Any]) -> Dict[str, Any]:
        """Validate value falls within expected range"""
        
        rule_def = rule.get('rule_definition', {})
        min_val = rule_def.get('min')
        max_val = rule_def.get('max')
        value = data_point.get('value')
        
        if min_val is None or max_val is None or value is None:
            return {
                'field': data_point.get('field_name', 'unknown'),
                'value': value,
                'rule': rule.get('rule_name', 'unknown'),
                'valid': False,
                'error': 'Missing range or value',
                'confidence': 0.0
            }
        
        try:
            value = float(value)
            min_val = float(min_val)
            max_val = float(max_val)
            
            is_valid = min_val <= value <= max_val
            
            return {
                'field': data_point.get('field_name', 'unknown'),
                'value': value,
                'rule': rule.get('rule_name', 'unknown'),
                'valid': is_valid,
                'expected_range': f"{min_val} - {max_val}",
                'confidence': rule.get('confidence_threshold', 0.8),
                'rule_type': 'range'
            }
            
        except (ValueError, TypeError) as e:
            return {
                'field': data_point.get('field_name', 'unknown'),
                'value': value,
                'rule': rule.get('rule_name', 'unknown'),
                'valid': False,
                'error': f'Value conversion error: {e}',
                'confidence': 0.0
            }
    
    def validate_formula(self, data_point: Dict[str, Any], rule: Dict[str, Any]) -> Dict[str, Any]:
        """Validate mathematical formula consistency"""
        # Placeholder for formula validation
        return {
            'field': data_point.get('field_name', 'unknown'),
            'value': data_point.get('value'),
            'rule': rule.get('rule_name', 'unknown'),
            'valid': True,  # Placeholder
            'confidence': rule.get('confidence_threshold', 0.8),
            'rule_type': 'formula'
        }
    
    def validate_consistency(self, data_point: Dict[str, Any], rule: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data consistency with historical patterns"""
        # Placeholder for consistency validation
        return {
            'field': data_point.get('field_name', 'unknown'),
            'value': data_point.get('value'),
            'rule': rule.get('rule_name', 'unknown'),
            'valid': True,  # Placeholder
            'confidence': rule.get('confidence_threshold', 0.8),
            'rule_type': 'consistency'
        }


class ContextEnhancer:
    """Enhances QA context using database knowledge"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
    
    def enhance_context(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance document context with database knowledge"""
        
        try:
            # Extract domain and topics
            domain = self.extract_domain(document)
            topics = self.extract_topics(document)
            
            # Get relevant knowledge
            domain_knowledge = self.get_domain_knowledge(domain)
            topic_context = self.get_topic_context(topics)
            
            # Get temporal context
            temporal_context = self.get_temporal_context(document)
            
            enhancement_score = self.calculate_enhancement_score(
                domain_knowledge, topic_context, temporal_context
            )
            
            return {
                'domain_knowledge': domain_knowledge,
                'topic_context': topic_context,
                'temporal_context': temporal_context,
                'enhancement_score': enhancement_score,
                'domain': domain,
                'topics': topics
            }
            
        except Exception as e:
            logger.error(f"Error enhancing context: {e}")
            return {
                'domain_knowledge': [],
                'topic_context': [],
                'temporal_context': {},
                'enhancement_score': 0.0,
                'domain': 'unknown',
                'topics': []
            }
    
    def extract_domain(self, document: Dict[str, Any]) -> str:
        """Extract domain from document"""
        metadata = document.get('metadata', {})
        content = document.get('content', '').lower()
        
        # Check metadata first
        if 'domain' in metadata:
            return metadata['domain']
        
        # Infer from content
        domain_keywords = {
            'financial': ['revenue', 'profit', 'loss', 'earnings', 'financial'],
            'technical': ['code', 'algorithm', 'system', 'technical', 'implementation'],
            'scientific': ['research', 'study', 'experiment', 'scientific', 'methodology'],
            'legal': ['legal', 'law', 'regulation', 'compliance', 'contract']
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in content for keyword in keywords):
                return domain
        
        return 'general'
    
    def extract_topics(self, document: Dict[str, Any]) -> List[str]:
        """Extract topics from document"""
        content = document.get('content', '').lower()
        
        # Simple topic extraction (could be enhanced with NLP)
        topics = []
        
        # Common topic patterns
        topic_patterns = [
            'machine learning', 'artificial intelligence', 'data analysis',
            'financial reporting', 'technical documentation', 'research findings'
        ]
        
        for pattern in topic_patterns:
            if pattern in content:
                topics.append(pattern)
        
        return topics
    
    def get_domain_knowledge(self, domain: str) -> List[Dict[str, Any]]:
        """Retrieve domain-specific knowledge"""
        
        query = """
        SELECT content, confidence_score, knowledge_type
        FROM domain_knowledge_base
        WHERE domain = %s
        AND confidence_score > 0.8
        ORDER BY confidence_score DESC
        LIMIT 10
        """
        
        try:
            result = self.db_manager.execute_query(query, (domain,))
            return result or []
        except Exception as e:
            logger.error(f"Error getting domain knowledge: {e}")
            return []
    
    def get_topic_context(self, topics: List[str]) -> List[Dict[str, Any]]:
        """Get context for specific topics"""
        # Placeholder implementation
        return []
    
    def get_temporal_context(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Get temporal context for document"""
        # Placeholder implementation
        return {
            'current_date': datetime.now().isoformat(),
            'document_date': document.get('metadata', {}).get('date'),
            'temporal_relevance': 0.8
        }
    
    def calculate_enhancement_score(self, domain_knowledge: List, topic_context: List, temporal_context: Dict) -> float:
        """Calculate overall enhancement score"""
        
        # Simple scoring based on available context
        score = 0.0
        
        if domain_knowledge:
            score += 0.4
        
        if topic_context:
            score += 0.3
        
        if temporal_context:
            score += 0.3
        
        return min(1.0, score)


class DatabaseEnhancedQAOrchestrator(QAOrchestrator):
    """QA Orchestrator with database-driven quality control"""
    
    def __init__(self):
        super().__init__()
        self.db_manager = get_database_manager()
        self.quality_analyzer = DatabaseQualityAnalyzer(self.db_manager)
        self.numerical_validator = NumericalValidator(self.db_manager)
        self.context_enhancer = ContextEnhancer(self.db_manager)
        self.document_inspector = DocumentInspectorCoordinator()
        self.caption_inspector = CaptionInspectorCoordinator()
        self.session_id = str(uuid.uuid4())
    
    def process_document_with_quality_control(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced document processing with database-driven QC"""
        
        start_time = datetime.now()
        
        try:
            # 1. Historical Pattern Analysis
            logger.info("Starting historical pattern analysis...")
            quality_prediction = self.quality_analyzer.predict_quality(document)
            
            # 2. Numerical Data Extraction and Validation
            logger.info("Extracting and validating numerical data...")
            numerical_data = self.extract_numerical_data(document)
            validation_results = self.numerical_validator.validate_data(numerical_data)
            
            # 3. Contextual Enhancement
            logger.info("Enhancing context with database knowledge...")
            enhanced_context = self.context_enhancer.enhance_context(document)
            
            # 4. Document Inspection (TOC, Bibliography, Links)
            logger.info("Performing comprehensive document inspection...")
            document_inspection = self.perform_document_inspection(document)
            
            # 5. Caption Analysis
            logger.info("Analyzing document captions...")
            caption_analysis = self.perform_caption_analysis(document)
            
            # 6. Quality Assessment
            logger.info("Calculating quality score...")
            quality_score = self.calculate_quality_score(
                quality_prediction, validation_results, enhanced_context, document_inspection, caption_analysis
            )
            
            # 7. Store Results
            logger.info("Storing quality metrics...")
            self.store_quality_metrics(document, quality_score, validation_results, document_inspection, caption_analysis)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                'document': document,
                'quality_score': quality_score,
                'validation_results': validation_results,
                'enhanced_context': enhanced_context,
                'document_inspection': document_inspection,
                'caption_analysis': caption_analysis,
                'quality_prediction': quality_prediction,
                'processing_time_ms': processing_time,
                'session_id': self.session_id,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error in database-enhanced QA processing: {e}")
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                'document': document,
                'quality_score': 0.0,
                'validation_results': [],
                'enhanced_context': {},
                'document_inspection': None,
                'caption_analysis': None,
                'quality_prediction': {},
                'processing_time_ms': processing_time,
                'session_id': self.session_id,
                'status': 'error',
                'error': str(e)
            }
    
    def extract_numerical_data(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract numerical data from document"""
        
        import re
        
        content = document.get('content', '')
        numerical_data = []
        
        # Find numbers in content
        numbers = re.findall(r'(\d+\.?\d*)', content)
        
        for i, number in enumerate(numbers[:20]):  # Limit to first 20 numbers
            try:
                value = float(number)
                numerical_data.append({
                    'field_name': f'number_{i+1}',
                    'value': value,
                    'type': 'general',
                    'source': 'content_extraction'
                })
            except ValueError:
                continue
        
        return numerical_data
    
    def calculate_quality_score(self, quality_prediction: Dict, validation_results: List, enhanced_context: Dict, document_inspection: Optional[Dict[str, Any]] = None, caption_analysis: Optional[Dict[str, Any]] = None) -> float:
        """Calculate composite quality score including document inspection metrics"""
        
        # Base score from prediction
        base_score = quality_prediction.get('predicted_score', 0.5)
        
        # Validation score
        if validation_results:
            valid_count = sum(1 for result in validation_results if result.get('valid', False))
            validation_score = valid_count / len(validation_results)
        else:
            validation_score = 0.5
        
        # Context enhancement score
        context_score = enhanced_context.get('enhancement_score', 0.0)
        
        # Document inspection score (including broken links metric)
        document_score = 0.5  # Default score
        if document_inspection:
            # Base document quality score
            document_score = document_inspection.get('overall_quality_score', 0.5)
            
            # Adjust for broken links (penalty)
            broken_links = document_inspection.get('broken_links', 0)
            total_links = document_inspection.get('total_links', 0)
            
            if total_links > 0:
                broken_link_ratio = broken_links / total_links
                # Apply penalty for broken links (up to 20% penalty)
                link_penalty = broken_link_ratio * 0.2
                document_score = max(0.0, document_score - link_penalty)
        
        # Caption analysis score
        caption_score = 0.5  # Default score
        if caption_analysis:
            total_captions = caption_analysis.get('total_captions', 0)
            captions_with_numbers = caption_analysis.get('captions_with_numbers', 0)
            
            if total_captions > 0:
                # Score based on percentage of captions with numbers and overall quality
                numbered_ratio = captions_with_numbers / total_captions
                quality_issues = caption_analysis.get('captions_with_issues', 0)
                quality_ratio = 1.0 - (quality_issues / total_captions) if total_captions > 0 else 1.0
                caption_score = (numbered_ratio * 0.6 + quality_ratio * 0.4)
        
        # Weighted combination (adjusted weights to include caption analysis)
        final_score = (
            base_score * 0.25 +
            validation_score * 0.25 +
            context_score * 0.15 +
            document_score * 0.20 +  # Document inspection
            caption_score * 0.15     # Caption analysis
        )
        
        return min(1.0, max(0.0, final_score))
    
    def perform_document_inspection(self, document: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Perform comprehensive document inspection including link validation"""
        
        try:
            # Check if document has a file path
            document_path = document.get('file_path') or document.get('path')
            
            if not document_path:
                logger.warning("No document path found for inspection")
                return None
            
            # Perform document inspection
            inspection_result = self.document_inspector.inspect_document(
                document_path=document_path,
                document_id=document.get('document_id'),
                document_title=document.get('title') or document.get('metadata', {}).get('title'),
                validate_links=True,  # Enable link validation for broken links metric
                extract_toc=True,
                extract_bibliography=True,
                extract_links=True
            )
            
            # Extract key metrics for QA
            inspection_summary = {
                'document_id': inspection_result.document_id,
                'overall_quality_score': inspection_result.overall_quality_score,
                'total_sections': inspection_result.total_sections,
                'total_citations': inspection_result.total_citations,
                'total_links': inspection_result.total_links,
                'broken_links': inspection_result.broken_links,  # Key metric for Sneaky Boss
                'valid_links': inspection_result.valid_links,
                'link_health_ratio': (inspection_result.valid_links / inspection_result.total_links) if inspection_result.total_links > 0 else 1.0,
                'issues': inspection_result.issues,
                'recommendations': inspection_result.recommendations,
                'inspection_timestamp': inspection_result.inspection_timestamp
            }
            
            logger.info(f"Document inspection completed: {inspection_result.broken_links} broken links found")
            return inspection_summary
            
        except Exception as e:
            logger.error(f"Error performing document inspection: {e}")
            return None
    
    def store_quality_metrics(self, document: Dict[str, Any], quality_score: float, validation_results: List, document_inspection: Optional[Dict[str, Any]] = None, caption_analysis: Optional[Dict[str, Any]] = None) -> None:
        """Store quality metrics in database"""
        
        try:
            # Store session metrics
            metrics_data = {
                'quality_score': quality_score,
                'validation_count': len(validation_results),
                'valid_count': sum(1 for r in validation_results if r.get('valid', False)),
                'document_hash': str(hash(document.get('content', ''))),
                'document_type': document.get('metadata', {}).get('type', 'unknown')
            }
            
            # Add document inspection metrics if available
            if document_inspection:
                metrics_data.update({
                    'broken_links': document_inspection.get('broken_links', 0),
                    'total_links': document_inspection.get('total_links', 0),
                    'link_health_ratio': document_inspection.get('link_health_ratio', 1.0),
                    'total_sections': document_inspection.get('total_sections', 0),
                    'total_citations': document_inspection.get('total_citations', 0),
                    'document_quality_score': document_inspection.get('overall_quality_score', 0.0)
                })
            
            # Add caption analysis metrics if available
            if caption_analysis:
                metrics_data.update({
                    'total_captions': caption_analysis.get('total_captions', 0),
                    'captions_with_numbers': caption_analysis.get('captions_with_numbers', 0),
                    'captions_with_issues': caption_analysis.get('captions_with_issues', 0),
                    'caption_confidence_score': caption_analysis.get('confidence_score', 0.0)
                })
            
            query = """
            INSERT INTO qa_quality_metrics (session_id, metric_name, metric_value, metric_unit)
            VALUES (%s, %s, %s, %s)
            """
            
            # Store individual metrics
            for metric_name, metric_value in metrics_data.items():
                if isinstance(metric_value, (int, float)):
                    self.db_manager.execute_command(
                        query, 
                        (self.session_id, metric_name, metric_value, 'score')
                    )
            
            logger.info(f"Stored quality metrics for session {self.session_id}")
            
        except Exception as e:
            logger.error(f"Error storing quality metrics: {e}")
    
    def get_quality_report(self, session_id: str = None) -> Dict[str, Any]:
        """Get quality report for a session"""
        
        if not session_id:
            session_id = self.session_id
        
        try:
            query = """
            SELECT metric_name, metric_value, metric_unit, timestamp
            FROM qa_quality_metrics
            WHERE session_id = %s
            ORDER BY timestamp DESC
            """
            
            metrics = self.db_manager.execute_query(query, (session_id,))
            
            return {
                'session_id': session_id,
                'metrics': metrics or [],
                'summary': self.summarize_metrics(metrics or [])
            }
            
        except Exception as e:
            logger.error(f"Error getting quality report: {e}")
            return {
                'session_id': session_id,
                'metrics': [],
                'summary': {},
                'error': str(e)
            }
    
    def summarize_metrics(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize quality metrics"""
        
        summary = {}
        
        for metric in metrics:
            name = metric.get('metric_name')
            value = metric.get('metric_value')
            
            if name and value is not None:
                summary[name] = value
        
        return summary
    
    def get_broken_links_metric(self, document_id: str = None) -> Dict[str, Any]:
        """Get broken links metric specifically for Sneaky Boss reporting"""
        
        if not document_id:
            document_id = self.session_id
        
        try:
            # Get quality report
            quality_report = self.get_quality_report(document_id)
            
            # Extract broken links metrics
            broken_links_metric = {
                'document_id': document_id,
                'metric_name': 'broken_links_count',
                'total_links': 0,
                'broken_links': 0,
                'valid_links': 0,
                'link_health_ratio': 1.0,
                'severity': 'none',
                'recommendations': []
            }
            
            # Find broken links metric in the report
            for metric in quality_report.get('metrics', []):
                if metric.get('metric_name') == 'broken_links':
                    broken_links_metric['broken_links'] = metric.get('metric_value', 0)
                elif metric.get('metric_name') == 'total_links':
                    broken_links_metric['total_links'] = metric.get('metric_value', 0)
                elif metric.get('metric_name') == 'link_health_ratio':
                    broken_links_metric['link_health_ratio'] = metric.get('metric_value', 1.0)
            
            # Calculate valid links
            broken_links_metric['valid_links'] = (
                broken_links_metric['total_links'] - broken_links_metric['broken_links']
            )
            
            # Determine severity
            if broken_links_metric['total_links'] > 0:
                broken_ratio = broken_links_metric['broken_links'] / broken_links_metric['total_links']
                if broken_ratio > 0.3:
                    broken_links_metric['severity'] = 'high'
                    broken_links_metric['recommendations'].append('🔴 CRITICAL: High number of broken links detected')
                elif broken_ratio > 0.1:
                    broken_links_metric['severity'] = 'medium'
                    broken_links_metric['recommendations'].append('🟡 WARNING: Moderate number of broken links detected')
                elif broken_ratio > 0:
                    broken_links_metric['severity'] = 'low'
                    broken_links_metric['recommendations'].append('🟢 INFO: Some broken links detected')
                else:
                    broken_links_metric['severity'] = 'none'
                    broken_links_metric['recommendations'].append('✅ All links are working properly')
            else:
                broken_links_metric['recommendations'].append('ℹ️ No links found in document')
            
            return broken_links_metric
            
        except Exception as e:
            logger.error(f"Error getting broken links metric: {e}")
            return {
                'document_id': document_id,
                'metric_name': 'broken_links_count',
                'error': str(e),
                'total_links': 0,
                'broken_links': 0,
                'valid_links': 0,
                'link_health_ratio': 1.0,
                'severity': 'error',
                'recommendations': ['❌ Error retrieving broken links metric']
            }
    
    def perform_caption_analysis(self, document: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Perform caption analysis on document"""
        
        try:
            # Check if document has a file path
            document_path = document.get('file_path') or document.get('path')
            
            if not document_path:
                logger.warning("No document path found for caption analysis")
                return None
            
            # Perform caption analysis
            caption_analysis = self.caption_inspector.inspect_document_captions(
                document_path=document_path,
                document_id=document.get('document_id'),
                document_title=document.get('title') or document.get('metadata', {}).get('title')
            )
            
            # Convert to dictionary for easier handling
            if caption_analysis:
                return {
                    'total_captions': caption_analysis.total_captions,
                    'captions_by_type': caption_analysis.captions_by_type,
                    'captions_with_numbers': caption_analysis.captions_with_numbers,
                    'captions_without_numbers': caption_analysis.captions_without_numbers,
                    'captions_with_issues': len([a for a in caption_analysis.quality_assessments if a.issues]),
                    'confidence_score': caption_analysis.confidence_score,
                    'captions': [caption.to_dict() for caption in caption_analysis.captions],
                    'quality_assessments': [assessment.to_dict() for assessment in caption_analysis.quality_assessments],
                    'extraction_method': caption_analysis.extraction_method,
                    'caption_sections': caption_analysis.caption_sections,
                    'metadata': caption_analysis.metadata
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error performing caption analysis: {e}")
            return None
    
    def store_quality_metrics(self, document: Dict[str, Any], quality_score: float, validation_results: List, document_inspection: Optional[Dict[str, Any]] = None, caption_analysis: Optional[Dict[str, Any]] = None) -> None:
        """Store quality metrics in database"""
        
        try:
            # Store session metrics
            metrics_data = {
                'quality_score': quality_score,
                'validation_count': len(validation_results),
                'valid_count': sum(1 for r in validation_results if r.get('valid', False)),
                'document_hash': str(hash(document.get('content', ''))),
                'document_type': document.get('metadata', {}).get('type', 'unknown')
            }
            
            # Add document inspection metrics if available
            if document_inspection:
                metrics_data.update({
                    'broken_links': document_inspection.get('broken_links', 0),
                    'total_links': document_inspection.get('total_links', 0),
                    'link_health_ratio': document_inspection.get('link_health_ratio', 1.0),
                    'total_sections': document_inspection.get('total_sections', 0),
                    'total_citations': document_inspection.get('total_citations', 0),
                    'document_quality_score': document_inspection.get('overall_quality_score', 0.0)
                })
            
            # Add caption analysis metrics if available
            if caption_analysis:
                metrics_data.update({
                    'total_captions': caption_analysis.get('total_captions', 0),
                    'captions_with_numbers': caption_analysis.get('captions_with_numbers', 0),
                    'captions_with_issues': caption_analysis.get('captions_with_issues', 0),
                    'caption_confidence_score': caption_analysis.get('confidence_score', 0.0)
                })
            
            query = """
            INSERT INTO qa_quality_metrics (session_id, metric_name, metric_value, metric_unit)
            VALUES (%s, %s, %s, %s)
            """
            
            # Store individual metrics
            for metric_name, metric_value in metrics_data.items():
                if isinstance(metric_value, (int, float)):
                    self.db_manager.execute_command(
                        query, 
                        (self.session_id, metric_name, metric_value, 'score')
                    )
            
            logger.info(f"Stored quality metrics for session {self.session_id}")
            
        except Exception as e:
            logger.error(f"Error storing quality metrics: {e}")
