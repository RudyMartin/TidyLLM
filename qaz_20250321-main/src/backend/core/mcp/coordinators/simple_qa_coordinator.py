#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple QA Coordinator - Proper MCP Hierarchy Implementation

This coordinator manages basic QA tasks through the proper MCP chain of command:
MCPOrchestrator → Planner → SimpleQACoordinator → QA Workers

Migrated from the old SimpleQAOrchestrator to follow proper organizational structure.
"""

import logging
import uuid
import json
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from ..coordinator import Coordinator
from ..base import MCPContext

logger = logging.getLogger(__name__)


class SimpleQACoordinator(Coordinator):
    """QA Coordinator for basic document quality analysis tasks"""
    
    def __init__(self, model_config: Dict[str, Any] = None):
        # Initialize as proper MCP Coordinator with QA domain
        super().__init__(
            name="simple_qa_coordinator",
            domain="basic_qa_processing", 
            model_config=model_config or self._get_default_model_config()
        )
        
        self.session_id = str(uuid.uuid4())
        self.quality_metrics = {}
        
        logger.info(f"Simple QA Coordinator initialized with session ID: {self.session_id}")
    
    def _get_default_model_config(self) -> Dict[str, Any]:
        """Default model configuration for QA tasks"""
        return {
            "model": "gpt-4o-mini",
            "max_tokens": 1500,
            "temperature": 0.2,
            "provider": "openai"
        }
    
    def process_qa_document(self, document: Dict[str, Any], context: MCPContext = None) -> Dict[str, Any]:
        """Process document through proper MCP coordinator workflow"""
        
        # Create QA-specific task for the MCP workflow
        qa_task = {
            "task": "perform_basic_qa_analysis",
            "priority": "medium",
            "input_data": {"document": document},
            "constraints": {
                "qa_type": "basic",
                "quality_checks": ["structure", "content", "completeness"],
                "require_metrics": True
            }
        }
        
        # Use parent coordinator process method (proper MCP flow)
        return self.process(context or MCPContext(user_request="QA Document Analysis"), qa_task)
    
    def _decompose_task(self, task: Dict[str, Any], context: MCPContext) -> Dict[str, Any]:
        """Decompose QA task into worker tasks (override parent method)"""
        
        task_description = task.get("task", "")
        document = task.get("input_data", {}).get("document", {})
        
        if not document:
            raise ValueError("No document provided for QA analysis")
        
        # Decompose into QA worker tasks
        worker_tasks = {}
        
        # Document Analysis Worker Task
        if "qa_analyzer" in self.workers:
            worker_tasks["qa_analyzer"] = {
                "task": f"analyze_document_quality for basic QA processing",
                "priority": "high",
                "input_data": {
                    "document": document,
                    "analysis_type": "basic_qa",
                    "quality_checks": task.get("constraints", {}).get("quality_checks", [])
                },
                "constraints": task.get("constraints", {})
            }
        
        # Quality Metrics Worker Task  
        if "qa_metrics" in self.workers:
            worker_tasks["qa_metrics"] = {
                "task": "calculate_quality_metrics for basic document assessment",
                "priority": "medium",
                "input_data": {
                    "document": document,
                    "metrics_type": "basic"
                },
                "constraints": task.get("constraints", {})
            }
        
        # Report Generator Worker Task
        if "qa_reporter" in self.workers:
            worker_tasks["qa_reporter"] = {
                "task": "generate_qa_report for basic quality analysis",
                "priority": "medium", 
                "input_data": {
                    "document": document,
                    "report_type": "basic_qa"
                },
                "constraints": task.get("constraints", {})
            }
        
        # Fallback: if no specialized workers, use general analysis
        if not worker_tasks and self.workers:
            first_worker = list(self.workers.keys())[0]
            worker_tasks[first_worker] = {
                "task": f"perform_basic_qa_analysis on provided document",
                "priority": "high",
                "input_data": {"document": document},
                "constraints": task.get("constraints", {})
            }
        
        if not worker_tasks:
            # No workers available, process directly
            return self._direct_qa_processing(document, task, context)
        
        return worker_tasks
    
    def _direct_qa_processing(self, document: Dict[str, Any], task: Dict[str, Any], context: MCPContext) -> Dict[str, Any]:
        """Direct QA processing when no workers available (maintains backward compatibility)"""
        
        logger.info("No QA workers available, performing direct QA processing...")
        
        start_time = datetime.now()
        
        try:
            # Extract basic document information
            document_info = self._extract_document_info(document)
            
            # Perform basic quality checks
            quality_checks = self._perform_basic_quality_checks(document)
            
            # Calculate simple quality score
            quality_score = self._calculate_simple_quality_score(quality_checks)
            
            # Generate basic report
            report = self._generate_basic_report(document_info, quality_checks, quality_score)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Return in MCP Coordinator format
            return {
                "coordinator": self.name,
                "domain": self.domain,
                "success": True,
                "synthesized_response": f"Basic QA analysis completed with quality score: {quality_score}",
                "worker_results": {
                    "direct_processing": {
                        "success": True,
                        "result": {
                            'document_info': document_info,
                            'quality_checks': quality_checks,
                            'quality_score': quality_score,
                            'report': report,
                            'processing_time_ms': processing_time,
                            'session_id': self.session_id,
                            'coordinator_type': 'simple_qa',
                            'status': 'success'
                        }
                    }
                },
                "synthesis_metadata": {
                    "total_workers": 0,
                    "successful_workers": 1,
                    "failed_workers": 0,
                    "synthesis_method": "direct_qa_processing"
                }
            }
            
        except Exception as e:
            logger.error(f"Error in direct QA processing: {e}")
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                "coordinator": self.name,
                "domain": self.domain,
                "success": False,
                "error": f"QA processing failed: {e}",
                "worker_results": {
                    "direct_processing": {
                        "success": False,
                        "error": str(e),
                        "processing_time_ms": processing_time
                    }
                }
            }
    
    # === Original SimpleQAOrchestrator Methods (Preserved) ===
    
    def _extract_document_info(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Extract basic document information"""
        
        content = document.get('content', '')
        metadata = document.get('metadata', {})
        
        # Basic text analysis
        word_count = len(content.split()) if content else 0
        character_count = len(content) if content else 0
        line_count = len(content.split('\n')) if content else 0
        
        # Basic structure detection
        has_title = bool(re.search(r'^#+\s+', content, re.MULTILINE)) if content else False
        has_headers = bool(re.search(r'^#{2,6}\s+', content, re.MULTILINE)) if content else False
        has_lists = bool(re.search(r'^[\s]*[-\*\+]\s+', content, re.MULTILINE)) if content else False
        has_code_blocks = bool(re.search(r'```', content)) if content else False
        
        return {
            'word_count': word_count,
            'character_count': character_count,
            'line_count': line_count,
            'has_title': has_title,
            'has_headers': has_headers,
            'has_lists': has_lists,
            'has_code_blocks': has_code_blocks,
            'metadata': metadata,
            'content_preview': content[:200] + '...' if len(content) > 200 else content
        }
    
    def _perform_basic_quality_checks(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Perform basic quality checks on document"""
        
        content = document.get('content', '')
        metadata = document.get('metadata', {})
        
        checks = {}
        
        # Content completeness check
        checks['has_content'] = bool(content and content.strip())
        checks['content_length_adequate'] = len(content) >= 100 if content else False
        
        # Structure checks
        checks['has_structure'] = any([
            bool(re.search(r'^#+\s+', content, re.MULTILINE)),  # Headers
            bool(re.search(r'^[\s]*[-\*\+]\s+', content, re.MULTILINE)),  # Lists
            bool(re.search(r'^\d+\.\s+', content, re.MULTILINE))  # Numbered lists
        ]) if content else False
        
        # Metadata checks
        checks['has_metadata'] = bool(metadata)
        checks['has_title'] = bool(metadata.get('title') or re.search(r'^#\s+(.+)', content, re.MULTILINE))
        
        # Basic formatting checks
        checks['proper_spacing'] = not bool(re.search(r'\n{3,}', content)) if content else True
        checks['no_excessive_caps'] = len(re.findall(r'[A-Z]{5,}', content)) < 10 if content else True
        
        # Language quality checks (basic)
        if content:
            sentences = re.split(r'[.!?]+', content)
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
            checks['reasonable_sentence_length'] = 5 <= avg_sentence_length <= 30
        else:
            checks['reasonable_sentence_length'] = False
        
        return checks
    
    def _calculate_simple_quality_score(self, quality_checks: Dict[str, Any]) -> float:
        """Calculate a simple quality score from checks"""
        
        if not quality_checks:
            return 0.0
        
        # Weight different checks
        weights = {
            'has_content': 0.25,
            'content_length_adequate': 0.15,
            'has_structure': 0.15,
            'has_metadata': 0.10,
            'has_title': 0.10,
            'proper_spacing': 0.08,
            'no_excessive_caps': 0.07,
            'reasonable_sentence_length': 0.10
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for check, passed in quality_checks.items():
            if check in weights:
                weight = weights[check]
                total_score += weight if passed else 0
                total_weight += weight
        
        # Normalize to 0-1 range
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _generate_basic_report(self, document_info: Dict[str, Any], quality_checks: Dict[str, Any], quality_score: float) -> Dict[str, Any]:
        """Generate a basic quality report"""
        
        # Categorize quality score
        if quality_score >= 0.8:
            quality_category = "High"
        elif quality_score >= 0.6:
            quality_category = "Medium"
        elif quality_score >= 0.4:
            quality_category = "Low"
        else:
            quality_category = "Poor"
        
        # Identify issues
        issues = []
        recommendations = []
        
        for check, passed in quality_checks.items():
            if not passed:
                if check == 'has_content':
                    issues.append("Document has no content")
                    recommendations.append("Add content to the document")
                elif check == 'content_length_adequate':
                    issues.append("Content is too short")
                    recommendations.append("Expand content to at least 100 characters")
                elif check == 'has_structure':
                    issues.append("Document lacks clear structure")
                    recommendations.append("Add headers, lists, or other structural elements")
                elif check == 'has_metadata':
                    issues.append("No metadata provided")
                    recommendations.append("Add document metadata (title, author, etc.)")
                elif check == 'has_title':
                    issues.append("Document has no title")
                    recommendations.append("Add a clear title to the document")
                elif check == 'proper_spacing':
                    issues.append("Excessive line breaks found")
                    recommendations.append("Clean up formatting and spacing")
                elif check == 'no_excessive_caps':
                    issues.append("Too many capital letters")
                    recommendations.append("Reduce use of ALL CAPS text")
                elif check == 'reasonable_sentence_length':
                    issues.append("Sentence length issues")
                    recommendations.append("Vary sentence length for better readability")
        
        # Generate summary
        summary = f"Document quality assessment completed. Overall quality: {quality_category} ({quality_score:.2f})"
        if issues:
            summary += f". {len(issues)} issues identified."
        else:
            summary += ". No major issues found."
        
        return {
            'summary': summary,
            'quality_score': quality_score,
            'quality_category': quality_category,
            'document_info': document_info,
            'quality_checks': quality_checks,
            'issues': issues,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        }

    def get_qa_metrics(self) -> Dict[str, Any]:
        """Get QA coordinator performance metrics"""
        base_metrics = self.get_performance_metrics() if hasattr(self, 'get_performance_metrics') else {}
        
        qa_specific_metrics = {
            'session_id': self.session_id,
            'coordinator_type': 'simple_qa',
            'domain': self.domain,
            'quality_metrics': self.quality_metrics
        }
        
        return {**base_metrics, **qa_specific_metrics}

    # Legacy compatibility method
    def process_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy method for backward compatibility with old SimpleQAOrchestrator interface"""
        logger.warning("Using legacy process_document method - consider migrating to process_qa_document")
        
        result = self.process_qa_document(document)
        
        # Convert MCP coordinator result back to legacy format for compatibility
        if result.get("success"):
            worker_result = result.get("worker_results", {}).get("direct_processing", {}).get("result", {})
            return worker_result
        else:
            return {
                'document': document,
                'document_info': {},
                'quality_checks': {},
                'quality_score': 0.0,
                'report': {'error': result.get("error", "Unknown error")},
                'processing_time_ms': 0,
                'session_id': self.session_id,
                'orchestrator_type': 'simple',
                'status': 'error',
                'error': result.get("error", "Unknown error")
            }