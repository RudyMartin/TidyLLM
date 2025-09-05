#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced QA Coordinator - Proper MCP Hierarchy Implementation

This coordinator manages enhanced QA tasks with database integration, document inspection,
and caption analysis through the proper MCP chain of command:
MCPOrchestrator → Planner → EnhancedQACoordinator → Specialized Workers

Migrated from the old EnhancedQAOrchestrator to eliminate hierarchy violations:
- No direct coordinator instantiation
- No bypass of planning layer  
- Proper worker delegation instead of coordinator ownership
"""

import logging
import uuid
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from ..coordinator import Coordinator
from ..base import MCPContext

logger = logging.getLogger(__name__)


class DatabaseQualityAnalyzer:
    """Analyzes quality patterns using database data (converted to worker task)"""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
    
    def predict_quality(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Predict quality score based on historical patterns"""
        
        try:
            if not self.db_manager:
                # Fallback when no database available
                return {
                    'predicted_score': 0.5,
                    'likely_issues': [],
                    'confidence': 0.0,
                    'pattern_count': 0,
                    'doc_features': self.extract_document_features(document)
                }
            
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
            r'\|.*\|',  # Pipe-separated
            r'^\s*\+[-+]+\+\s*$',  # ASCII table borders
            r'\t.*\t',  # Tab-separated
        ]
        return any(re.search(pattern, content, re.MULTILINE) for pattern in table_patterns)
    
    def detect_numerical_content(self, content: str) -> bool:
        """Detect numerical content"""
        import re
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', content)
        return len(numbers) > 5
    
    def calculate_complexity(self, content: str) -> float:
        """Calculate content complexity score"""
        if not content:
            return 0.0
        
        # Simple complexity metrics
        words = content.split()
        sentences = [s for s in content.split('.') if s.strip()]
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        avg_sentence_length = sum(len(sentence.split()) for sentence in sentences) / len(sentences) if sentences else 0
        
        # Normalize complexity score
        complexity = (avg_word_length * 0.3) + (avg_sentence_length * 0.7)
        return min(complexity / 20, 1.0)  # Cap at 1.0
    
    def calculate_prediction_confidence(self, doc_features: Dict[str, Any], pattern_count: int) -> float:
        """Calculate confidence in prediction"""
        if pattern_count == 0:
            return 0.0
        
        # Base confidence on pattern count and document features
        base_confidence = min(pattern_count / 100, 0.8)  # More patterns = more confidence
        
        # Adjust based on document complexity
        complexity = doc_features.get('complexity_score', 0.5)
        confidence_adjustment = 0.2 if complexity > 0.5 else -0.1
        
        return max(min(base_confidence + confidence_adjustment, 1.0), 0.0)


class EnhancedQACoordinator(Coordinator):
    """Enhanced QA Coordinator with database integration and specialized analysis"""
    
    def __init__(self, model_config: Dict[str, Any] = None, db_manager=None):
        # Initialize as proper MCP Coordinator with Enhanced QA domain
        super().__init__(
            name="enhanced_qa_coordinator",
            domain="enhanced_qa_processing", 
            model_config=model_config or self._get_default_model_config()
        )
        
        self.session_id = str(uuid.uuid4())
        self.quality_metrics = {}
        
        # Database integration (but not ownership!)
        self.db_manager = db_manager
        self.db_analyzer = DatabaseQualityAnalyzer(db_manager)
        
        # NOTE: NO DIRECT COORDINATOR INSTANTIATION!
        # Old code violated hierarchy:
        # self.document_inspector = DocumentInspectorCoordinator()  ❌ 
        # self.caption_inspector = CaptionInspectorCoordinator()    ❌
        # 
        # New approach: Request these services through workers ✅
        
        logger.info(f"Enhanced QA Coordinator initialized with session ID: {self.session_id}")
    
    def _get_default_model_config(self) -> Dict[str, Any]:
        """Default model configuration for Enhanced QA tasks"""
        return {
            "model": "gpt-4o",
            "max_tokens": 2000,
            "temperature": 0.1,
            "provider": "openai"
        }
    
    def process_enhanced_qa_document(self, document: Dict[str, Any], context: MCPContext = None) -> Dict[str, Any]:
        """Process document through proper MCP Enhanced QA workflow"""
        
        # Create Enhanced QA-specific task for the MCP workflow
        enhanced_qa_task = {
            "task": "perform_enhanced_qa_analysis_with_database_integration",
            "priority": "high",
            "input_data": {"document": document},
            "constraints": {
                "qa_type": "enhanced",
                "quality_checks": ["structure", "content", "completeness", "database_patterns"],
                "require_database_analysis": True,
                "require_document_inspection": True,
                "require_caption_analysis": True,
                "require_metrics": True
            }
        }
        
        # Use parent coordinator process method (proper MCP flow)
        return self.process(context or MCPContext(user_request="Enhanced QA Document Analysis"), enhanced_qa_task)
    
    def _decompose_task(self, task: Dict[str, Any], context: MCPContext) -> Dict[str, Any]:
        """Decompose Enhanced QA task into worker tasks (override parent method)"""
        
        task_description = task.get("task", "")
        document = task.get("input_data", {}).get("document", {})
        constraints = task.get("constraints", {})
        
        if not document:
            raise ValueError("No document provided for Enhanced QA analysis")
        
        # Decompose into Enhanced QA worker tasks
        worker_tasks = {}
        
        # Database Quality Analysis Worker
        if "database_qa_analyzer" in self.workers:
            worker_tasks["database_qa_analyzer"] = {
                "task": f"analyze_document_quality_using_database_patterns",
                "priority": "high",
                "input_data": {
                    "document": document,
                    "analysis_type": "database_enhanced_qa",
                    "db_manager": self.db_manager
                },
                "constraints": constraints
            }
        
        # Document Inspector Worker (replaces direct coordinator call)
        if "document_inspector" in self.workers:
            worker_tasks["document_inspector"] = {
                "task": f"inspect_document_structure_and_content_quality",
                "priority": "high", 
                "input_data": {
                    "document": document,
                    "inspection_type": "enhanced_qa"
                },
                "constraints": constraints
            }
        
        # Caption Analysis Worker (replaces direct coordinator call)
        if "caption_analyzer" in self.workers:
            worker_tasks["caption_analyzer"] = {
                "task": f"analyze_document_captions_and_visual_elements",
                "priority": "medium",
                "input_data": {
                    "document": document,
                    "analysis_type": "caption_qa"
                },
                "constraints": constraints
            }
        
        # Enhanced Quality Metrics Worker  
        if "enhanced_qa_metrics" in self.workers:
            worker_tasks["enhanced_qa_metrics"] = {
                "task": "calculate_enhanced_quality_metrics_with_database_context",
                "priority": "medium",
                "input_data": {
                    "document": document,
                    "metrics_type": "enhanced"
                },
                "constraints": constraints
            }
        
        # Enhanced Report Generator Worker
        if "enhanced_qa_reporter" in self.workers:
            worker_tasks["enhanced_qa_reporter"] = {
                "task": "generate_enhanced_qa_report_with_database_insights",
                "priority": "medium", 
                "input_data": {
                    "document": document,
                    "report_type": "enhanced_qa"
                },
                "constraints": constraints
            }
        
        # Fallback: if no specialized workers, use enhanced analysis directly
        if not worker_tasks:
            return self._direct_enhanced_qa_processing(document, task, context)
        
        return worker_tasks
    
    def _direct_enhanced_qa_processing(self, document: Dict[str, Any], task: Dict[str, Any], context: MCPContext) -> Dict[str, Any]:
        """Direct Enhanced QA processing when no workers available (maintains functionality)"""
        
        logger.info("No Enhanced QA workers available, performing direct enhanced processing...")
        
        start_time = datetime.now()
        
        try:
            # Step 1: Basic QA analysis (inherited from SimpleQA patterns)
            basic_info = self._extract_document_info(document)
            basic_checks = self._perform_basic_quality_checks(document) 
            
            # Step 2: Database-enhanced analysis
            db_prediction = self.db_analyzer.predict_quality(document)
            
            # Step 3: Enhanced quality checks
            enhanced_checks = self._perform_enhanced_quality_checks(document, db_prediction)
            
            # Step 4: Calculate enhanced quality score
            enhanced_score = self._calculate_enhanced_quality_score(
                basic_checks, enhanced_checks, db_prediction
            )
            
            # Step 5: Generate enhanced report
            enhanced_report = self._generate_enhanced_report(
                basic_info, basic_checks, enhanced_checks, enhanced_score, db_prediction
            )
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Return in MCP Coordinator format
            return {
                "coordinator": self.name,
                "domain": self.domain,
                "success": True,
                "synthesized_response": f"Enhanced QA analysis completed with quality score: {enhanced_score}",
                "worker_results": {
                    "direct_enhanced_processing": {
                        "success": True,
                        "result": {
                            'document_info': basic_info,
                            'basic_quality_checks': basic_checks,
                            'enhanced_quality_checks': enhanced_checks,
                            'database_prediction': db_prediction,
                            'quality_score': enhanced_score,
                            'report': enhanced_report,
                            'processing_time_ms': processing_time,
                            'session_id': self.session_id,
                            'coordinator_type': 'enhanced_qa',
                            'status': 'success'
                        }
                    }
                },
                "synthesis_metadata": {
                    "total_workers": 0,
                    "successful_workers": 1,
                    "failed_workers": 0,
                    "synthesis_method": "direct_enhanced_qa_processing",
                    "database_integration": bool(self.db_manager),
                    "prediction_confidence": db_prediction.get('confidence', 0.0)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in direct Enhanced QA processing: {e}")
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                "coordinator": self.name,
                "domain": self.domain,
                "success": False,
                "error": f"Enhanced QA processing failed: {e}",
                "worker_results": {
                    "direct_enhanced_processing": {
                        "success": False,
                        "error": str(e),
                        "processing_time_ms": processing_time
                    }
                }
            }
    
    # === Enhanced QA Methods (Improved from original) ===
    
    def _perform_enhanced_quality_checks(self, document: Dict[str, Any], db_prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Perform enhanced quality checks beyond basic analysis"""
        
        content = document.get('content', '')
        metadata = document.get('metadata', {})
        doc_features = db_prediction.get('doc_features', {})
        
        enhanced_checks = {}
        
        # Database-informed checks
        enhanced_checks['predicted_quality_high'] = db_prediction.get('predicted_score', 0) > 0.7
        enhanced_checks['has_historical_patterns'] = db_prediction.get('pattern_count', 0) > 0
        enhanced_checks['prediction_confident'] = db_prediction.get('confidence', 0) > 0.5
        
        # Content complexity checks
        enhanced_checks['appropriate_complexity'] = 0.3 <= doc_features.get('complexity_score', 0.5) <= 0.8
        enhanced_checks['has_numerical_content'] = doc_features.get('has_numbers', False)
        enhanced_checks['has_structured_data'] = doc_features.get('has_tables', False)
        
        # Content quality checks
        if content:
            # Check for common quality issues predicted by database
            likely_issues = db_prediction.get('likely_issues', [])
            enhanced_checks['no_predicted_issues'] = len(likely_issues) == 0
            
            # Advanced content analysis
            enhanced_checks['sufficient_word_count'] = doc_features.get('word_count', 0) >= 50
            enhanced_checks['reasonable_sentence_structure'] = doc_features.get('sentence_count', 0) > 0
            
            # Metadata quality
            enhanced_checks['comprehensive_metadata'] = len(metadata) >= 3
            enhanced_checks['has_document_type'] = bool(metadata.get('type'))
        
        return enhanced_checks
    
    def _calculate_enhanced_quality_score(self, basic_checks: Dict[str, Any], 
                                        enhanced_checks: Dict[str, Any], 
                                        db_prediction: Dict[str, Any]) -> float:
        """Calculate enhanced quality score incorporating database predictions"""
        
        # Base score from basic checks (40% weight)
        basic_score = self._calculate_simple_quality_score(basic_checks)
        
        # Enhanced checks score (40% weight)
        enhanced_weights = {
            'predicted_quality_high': 0.25,
            'has_historical_patterns': 0.15,
            'prediction_confident': 0.15,
            'appropriate_complexity': 0.10,
            'has_numerical_content': 0.10,
            'has_structured_data': 0.10,
            'no_predicted_issues': 0.15
        }
        
        enhanced_score = 0.0
        enhanced_total_weight = 0.0
        
        for check, passed in enhanced_checks.items():
            if check in enhanced_weights:
                weight = enhanced_weights[check]
                enhanced_score += weight if passed else 0
                enhanced_total_weight += weight
        
        enhanced_score = enhanced_score / enhanced_total_weight if enhanced_total_weight > 0 else 0.0
        
        # Database prediction score (20% weight)
        db_score = db_prediction.get('predicted_score', 0.5)
        db_confidence = db_prediction.get('confidence', 0.0)
        
        # Weighted final score
        final_score = (basic_score * 0.4) + (enhanced_score * 0.4) + (db_score * db_confidence * 0.2)
        
        return min(max(final_score, 0.0), 1.0)
    
    def _generate_enhanced_report(self, document_info: Dict[str, Any], 
                                basic_checks: Dict[str, Any],
                                enhanced_checks: Dict[str, Any],
                                quality_score: float,
                                db_prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Generate enhanced quality report with database insights"""
        
        # Categorize quality score
        if quality_score >= 0.85:
            quality_category = "Excellent"
        elif quality_score >= 0.7:
            quality_category = "Good"
        elif quality_score >= 0.5:
            quality_category = "Fair"
        else:
            quality_category = "Poor"
        
        # Collect all issues
        issues = []
        recommendations = []
        
        # Basic issues
        for check, passed in basic_checks.items():
            if not passed:
                if check == 'has_content':
                    issues.append("Document has no content")
                    recommendations.append("Add content to the document")
                elif check == 'content_length_adequate':
                    issues.append("Content is too short for quality analysis")
                    recommendations.append("Expand content to provide more comprehensive information")
                # Add other basic issue mappings...
        
        # Enhanced issues
        for check, passed in enhanced_checks.items():
            if not passed:
                if check == 'predicted_quality_high':
                    issues.append("Database analysis predicts below-average quality")
                    recommendations.append("Review similar high-quality documents for improvement patterns")
                elif check == 'prediction_confident':
                    issues.append("Limited historical data for reliable quality prediction")
                    recommendations.append("This document type may need manual review")
                elif check == 'appropriate_complexity':
                    issues.append("Content complexity may not be appropriate for target audience")
                    recommendations.append("Adjust content complexity to be more suitable")
                # Add other enhanced issue mappings...
        
        # Database-specific insights
        likely_issues = db_prediction.get('likely_issues', [])
        if likely_issues:
            issues.extend([f"Predicted issue: {issue}" for issue in likely_issues])
            recommendations.append("Address predicted issues based on historical patterns")
        
        # Generate enhanced summary
        summary = f"Enhanced QA assessment completed. Overall quality: {quality_category} ({quality_score:.2f})"
        summary += f". Database prediction: {db_prediction.get('predicted_score', 0.5):.2f} "
        summary += f"(confidence: {db_prediction.get('confidence', 0.0):.2f})"
        
        if issues:
            summary += f". {len(issues)} issues identified."
        else:
            summary += ". No major issues found."
        
        return {
            'summary': summary,
            'quality_score': quality_score,
            'quality_category': quality_category,
            'document_info': document_info,
            'basic_quality_checks': basic_checks,
            'enhanced_quality_checks': enhanced_checks,
            'database_prediction': db_prediction,
            'issues': issues,
            'recommendations': recommendations,
            'enhancement_features': {
                'database_integration': bool(self.db_manager),
                'pattern_analysis': db_prediction.get('pattern_count', 0) > 0,
                'complexity_analysis': True,
                'predictive_quality': True
            },
            'timestamp': datetime.now().isoformat()
        }
    
    # === Utility Methods ===
    
    def _extract_document_info(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced document information extraction"""
        # Use database analyzer for more comprehensive feature extraction
        return self.db_analyzer.extract_document_features(document)
    
    def _perform_basic_quality_checks(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Basic quality checks (inherited pattern from SimpleQA)"""
        content = document.get('content', '')
        metadata = document.get('metadata', {})
        
        checks = {}
        checks['has_content'] = bool(content and content.strip())
        checks['content_length_adequate'] = len(content) >= 100 if content else False
        checks['has_metadata'] = bool(metadata)
        checks['has_title'] = bool(metadata.get('title'))
        
        return checks
    
    def _calculate_simple_quality_score(self, quality_checks: Dict[str, Any]) -> float:
        """Simple quality score calculation (for base score)"""
        if not quality_checks:
            return 0.0
        
        passed_checks = sum(1 for passed in quality_checks.values() if passed)
        total_checks = len(quality_checks)
        
        return passed_checks / total_checks if total_checks > 0 else 0.0
    
    def get_enhanced_qa_metrics(self) -> Dict[str, Any]:
        """Get Enhanced QA coordinator performance metrics"""
        base_metrics = {}
        
        enhanced_metrics = {
            'session_id': self.session_id,
            'coordinator_type': 'enhanced_qa',
            'domain': self.domain,
            'database_integration': bool(self.db_manager),
            'quality_metrics': self.quality_metrics
        }
        
        return {**base_metrics, **enhanced_metrics}

    # Legacy compatibility method
    def process_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy method for backward compatibility with old EnhancedQAOrchestrator interface"""
        logger.warning("Using legacy process_document method - consider migrating to process_enhanced_qa_document")
        
        result = self.process_enhanced_qa_document(document)
        
        # Convert MCP coordinator result back to legacy format for compatibility
        if result.get("success"):
            worker_result = result.get("worker_results", {}).get("direct_enhanced_processing", {}).get("result", {})
            return worker_result
        else:
            return {
                'document': document,
                'document_info': {},
                'basic_quality_checks': {},
                'enhanced_quality_checks': {},
                'database_prediction': {},
                'quality_score': 0.0,
                'report': {'error': result.get("error", "Unknown error")},
                'processing_time_ms': 0,
                'session_id': self.session_id,
                'coordinator_type': 'enhanced_qa',
                'status': 'error',
                'error': result.get("error", "Unknown error")
            }