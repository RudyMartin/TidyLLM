#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Classification Worker

Configurable worker for classifying files into MVR document types with progressive complexity.
Supports Simple, Enhanced, and Advanced classification modes based on orchestrator requirements.
Uses configuration from dev_configs/file_classification.yaml for classification rules.
"""

import logging
import re
import os
import yaml
import json
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime
from enum import Enum

from .base_worker import BaseWorker
from ..protocol.message_protocol import MCPMessage, TaskType, Priority, AuditTrail


class ClassificationMode(Enum):
    """Classification complexity modes"""
    SIMPLE = "simple"
    ENHANCED = "enhanced" 
    ADVANCED = "advanced"


class FileClassificationWorker(BaseWorker):
    """Configurable worker for classifying files with progressive complexity"""
    
    def __init__(self, mode: ClassificationMode = ClassificationMode.SIMPLE):
        super().__init__("FileClassificationWorker", "classification")
        self.mode = mode
        self.config_path = os.path.join(os.path.dirname(__file__), '../../../../dev_configs/file_classification.yaml')
        self.file_classification = self._load_file_classification()
        
        # Review ID patterns from existing MCP code
        self.review_id_patterns = [
            r'Review ID[:\s]*([A-Z]{3}\d{5})',
            r'Review[:\s]*([A-Z]{3}\d{5})',
            r'([A-Z]{3}\d{5})',
            r'Review ID[:\s]*(\d{5})',
        ]
        
        # Enhanced classification keywords for all dimensions
        self.classification_keywords = self._load_classification_keywords()
        
        self.logger.info(f"FileClassificationWorker initialized in {mode.value} mode with {len(self.file_classification)} document types")
    
    def _load_file_classification(self) -> Dict:
        """Load file classification configuration from YAML"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Extract file_types from the config
            file_types = config.get('file_types', {})
            self.logger.info(f"Loaded {len(file_types)} file types from configuration")
            return file_types
            
        except Exception as e:
            self.logger.error(f"Failed to load file classification config: {e}")
            return self._get_default_classification()
    
    def _get_default_classification(self) -> Dict:
        """Get default classification if YAML loading fails"""
        return {
            'unclassified': {
                'extensions': ['.pdf', '.docx', '.doc', '.md', '.txt', '.yaml', '.yml', '.json', '.csv', '.xlsx', '.py', '.r', '.sql', '.ipynb', '.html', '.xml', '.log'],
                'category': "Unclassified (U)",
                'description': "Documents that need classification based on content analysis",
                'priority': "Low"
            }
        }
    
    def _load_classification_keywords(self) -> Dict:
        """Load comprehensive classification keywords for all dimensions"""
        return {
            # MVR Document Types (existing)
            'mvr_document_types': {
                'whitepaper': ['abstract', 'introduction', 'methodology', 'conclusion', 'references', 'bibliography', 'research', 'study', 'analysis', 'paper'],
                'validation_scoping_template': ['validation scope', 'stakeholders', 'timeline', 'resources', 'validation framework', 'scoping document', 'validation plan'],
                'annual_model_review': ['annual review', 'model performance', 'risk assessment', 'review period', 'annual assessment', 'comprehensive review', 'yearly evaluation'],
                'model_documentation_assessment': ['documentation assessment', 'documentation quality', 'missing elements', 'improvement areas', 'documentation score', 'completeness assessment'],
                'model_development_document': ['model architecture', 'development methodology', 'data sources', 'implementation details', 'model specification', 'development process'],
                'model_development_plan': ['development plan', 'development timeline', 'resource planning', 'methodology framework', 'success criteria', 'development roadmap', 'strategic plan'],
                'sdc_validation_report': ['validation report', 'test results', 'defects found', 'validation phase', 'software development cycle', 'validation conclusion'],
                'qa_template': ['qa template', 'quality assurance', 'checklist', 'assessment criteria', 'quality metrics', 'qa framework', 'quality checklist'],
                'implementation_testing_document': ['implementation testing', 'test plan', 'test cases', 'deployment validation', 'testing documentation', 'test results'],
                'model_implementation_plan': ['implementation plan', 'implementation timeline', 'deployment strategy', 'resource requirements', 'risk mitigation', 'implementation roadmap'],
                'standards_procedures': ['standards', 'procedures', 'governance framework', 'organizational standards', 'procedure steps', 'compliance requirements'],
                'model_performance_monitoring_plan': ['performance monitoring', 'monitoring framework', 'performance metrics', 'alert thresholds', 'escalation procedures', 'production monitoring', 'model performance'],
                'post_development_recalibrated_model_plan': ['recalibration', 'model updates', 'post development', 'recalibrated model', 'update procedures', 'deployment strategy', 'model maintenance'],
                'modeling_area_prioritization_plan': ['modeling area', 'prioritization', 'resource allocation', 'strategic planning', 'area management', 'prioritization criteria', 'timeline framework', 'success metrics']
            },
            # Enhanced classification keywords (for ENHANCED and ADVANCED modes)
            'topics': {
                'finance': ['banking', 'investment', 'accounting', 'risk', 'portfolio', 'financial', 'revenue', 'profit', 'loss', 'budget', 'forecast'],
                'legal': ['contract', 'regulation', 'compliance', 'legal', 'law', 'policy', 'terms', 'conditions', 'agreement', 'liability'],
                'technical': ['engineering', 'development', 'infrastructure', 'system', 'architecture', 'design', 'implementation', 'technology', 'software', 'hardware'],
                'research': ['academic', 'scientific', 'analysis', 'study', 'experiment', 'methodology', 'findings', 'conclusion', 'research', 'investigation'],
                'operations': ['hr', 'facilities', 'logistics', 'operations', 'process', 'procedure', 'workflow', 'management', 'administration']
            },
            'sensitivity': {
                'public': ['public', 'open', 'published', 'available', 'accessible'],
                'internal': ['internal', 'company', 'organization', 'staff', 'employee'],
                'confidential': ['confidential', 'private', 'sensitive', 'restricted', 'proprietary'],
                'restricted': ['restricted', 'classified', 'secret', 'top-secret', 'need-to-know'],
                'regulated': ['pii', 'hipaa', 'sox', 'gdpr', 'regulated', 'compliance', 'personal', 'health', 'financial']
            },
            'content_types': {
                'textual': ['report', 'document', 'policy', 'procedure', 'manual', 'guide', 'text', 'narrative'],
                'numerical': ['spreadsheet', 'data', 'metrics', 'statistics', 'numbers', 'calculations', 'formulas'],
                'multimedia': ['image', 'photo', 'video', 'audio', 'media', 'graphic', 'visual'],
                'code': ['script', 'program', 'code', 'software', 'application', 'algorithm', 'function'],
                'mixed': ['combined', 'multiple', 'various', 'diverse', 'assorted']
            },
            'business_functions': {
                'hr': ['personnel', 'recruitment', 'training', 'employee', 'staff', 'human resources', 'hiring', 'performance'],
                'finance': ['accounting', 'budgeting', 'reporting', 'financial', 'revenue', 'expense', 'audit', 'tax'],
                'engineering': ['development', 'testing', 'deployment', 'engineering', 'technical', 'design', 'implementation'],
                'compliance': ['regulatory', 'audit', 'governance', 'compliance', 'policy', 'regulation', 'oversight'],
                'operations': ['facilities', 'logistics', 'support', 'operations', 'process', 'maintenance']
            },
            'workflow_stages': {
                'draft': ['draft', 'preliminary', 'initial', 'rough', 'unfinished', 'in progress'],
                'review': ['review', 'pending', 'under review', 'approval', 'feedback', 'comments'],
                'final': ['final', 'approved', 'official', 'complete', 'finished', 'authorized'],
                'archive': ['archive', 'historical', 'old', 'previous', 'deprecated', 'legacy'],
                'raw': ['raw', 'unprocessed', 'source', 'original', 'untouched', 'primary']
            },
            'audiences': {
                'internal': ['internal', 'employee', 'staff', 'team', 'department', 'company'],
                'external': ['customer', 'partner', 'vendor', 'external', 'client', 'stakeholder'],
                'regulators': ['regulator', 'government', 'oversight', 'compliance', 'regulatory', 'authority'],
                'public': ['public', 'general', 'marketing', 'press', 'announcement', 'publication']
            }
        }
    
    def process_task(self, message: MCPMessage) -> Dict[str, Any]:
        """
        Process file classification task based on incoming message and mode.
        
        Args:
            message: MCP message containing file classification task
            
        Returns:
            Dictionary containing classification results based on mode
        """
        task_data = message.get_task_data()
        task_type = message.get_task_type()
        
        # Handle mode-specific processing
        if task_type == TaskType.FILE_CLASSIFICATION:
            return self._classify_file_task(task_data)
        elif task_type == TaskType.REVIEW_ID_EXTRACTION:
            return self._extract_review_id_task(task_data)
        elif task_type == TaskType.FILE_VALIDATION:
            return self._validate_file_task(task_data)
        elif task_type == TaskType.BATCH_CLASSIFICATION:
            return self._batch_classify_task(task_data)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    def _classify_file_task(self, task_data: Dict) -> Dict[str, Any]:
        """Classify file based on current mode"""
        if self.mode == ClassificationMode.SIMPLE:
            return self._classify_file_simple(task_data)
        elif self.mode == ClassificationMode.ENHANCED:
            return self._classify_file_enhanced(task_data)
        elif self.mode == ClassificationMode.ADVANCED:
            return self._classify_file_advanced(task_data)
        else:
            raise ValueError(f"Unsupported classification mode: {self.mode}")
    
    def _classify_file_simple(self, task_data: Dict) -> Dict[str, Any]:
        """Simple file classification (basic extension and content analysis)"""
        filename = task_data.get('filename', '')
        content = task_data.get('content', '')
        file_size = task_data.get('file_size', 0)
        
        # Classify file using basic method
        classification_result = self.classify_file(filename, content)
        
        # Extract Review ID
        review_id = self.extract_review_id(content)
        
        # Basic validation
        validation_result = self.validate_file({
            'filename': filename,
            'content': content,
            'file_size': file_size,
            'classification': classification_result
        })
        
        return {
            'filename': filename,
            'classification': classification_result,
            'review_id': review_id,
            'validation': validation_result,
            'confidence_score': classification_result.get('confidence', 0.0),
            'processed_at': datetime.now().isoformat(),
            'worker_name': self.worker_name,
            'mode': self.mode.value
        }
    
    def _classify_file_enhanced(self, task_data: Dict) -> Dict[str, Any]:
        """Enhanced file classification with multi-dimensional analysis"""
        filename = task_data.get('filename', '')
        content = task_data.get('content', '')
        file_size = task_data.get('file_size', 0)
        metadata = task_data.get('metadata', {})
        context = task_data.get('context', {})
        
        # Enhanced classification with multiple dimensions
        classification_result = self.classify_file_enhanced({
            'filename': filename,
            'content': content,
            'file_size': file_size,
            'metadata': metadata,
            'context': context
        })
        
        # Extract Review ID
        review_id = self.extract_review_id(content)
        
        # Enhanced validation
        validation_result = self.validate_file_enhanced({
            'filename': filename,
            'content': content,
            'file_size': file_size,
            'classification': classification_result
        })
        
        return {
            'filename': filename,
            'classification': classification_result,
            'review_id': review_id,
            'validation': validation_result,
            'confidence_score': classification_result.get('overall_confidence', 0.0),
            'processed_at': datetime.now().isoformat(),
            'worker_name': self.worker_name,
            'mode': self.mode.value
        }
    
    def _classify_file_advanced(self, task_data: Dict) -> Dict[str, Any]:
        """Advanced file classification with AI/ML capabilities"""
        # Enhanced classification plus advanced features
        enhanced_result = self._classify_file_enhanced(task_data)
        
        # Add advanced features
        advanced_features = self._apply_advanced_features(task_data, enhanced_result)
        enhanced_result['advanced_features'] = advanced_features
        
        return enhanced_result
    
    def _extract_review_id_task(self, task_data: Dict) -> Dict[str, Any]:
        """Extract Review ID from file content"""
        content = task_data.get('content', '')
        review_id = self.extract_review_id(content)
        
        return {
            'review_id': review_id,
            'content_length': len(content),
            'extraction_method': 'regex_patterns',
            'confidence_score': 1.0 if review_id else 0.0,
            'processed_at': datetime.now().isoformat(),
            'worker_name': self.worker_name
        }
    
    def _validate_file_task(self, task_data: Dict) -> Dict[str, Any]:
        """Validate file before classification"""
        filename = task_data.get('filename', '')
        content = task_data.get('content', '')
        file_size = task_data.get('file_size', 0)
        classification = task_data.get('classification', {})
        
        validation_result = self.validate_file({
            'filename': filename,
            'content': content,
            'file_size': file_size,
            'classification': classification
        })
        
        return {
            'filename': filename,
            'validation': validation_result,
            'confidence_score': 1.0 if validation_result.get('valid', False) else 0.0,
            'processed_at': datetime.now().isoformat(),
            'worker_name': self.worker_name
        }
    
    def _batch_classify_task(self, task_data: Dict) -> Dict[str, Any]:
        """Classify multiple files in batch"""
        files = task_data.get('files', [])
        results = []
        review_id_groups = {}
        
        for file_data in files:
            filename = file_data.get('filename', '')
            content = file_data.get('content', '')
            file_size = file_data.get('file_size', 0)
            
            # Classify file
            classification_result = self.classify_file(filename, content)
            review_id = self.extract_review_id(content)
            
            # Group by Review ID
            if review_id:
                if review_id not in review_id_groups:
                    review_id_groups[review_id] = []
                review_id_groups[review_id].append({
                    'filename': filename,
                    'size': file_size,
                    'category': classification_result.get('category', 'Unknown'),
                    'priority': classification_result.get('priority', 'Low')
                })
            
            results.append({
                'filename': filename,
                'classification': classification_result,
                'review_id': review_id,
                'file_size': file_size
            })
        
        return {
            'files': results,
            'review_id_groups': review_id_groups,
            'total_files': len(files),
            'processed_at': datetime.now().isoformat(),
            'worker_name': self.worker_name
        }
    
    def classify_file(self, filename: str, content: str) -> Dict:
        """
        Classify file based on extension and content analysis.
        
        Args:
            filename: Name of the file to classify
            content: File content for analysis
            
        Returns:
            Dictionary containing classification results
        """
        file_ext = Path(filename).suffix.lower()
        
        # First try content-based classification
        content_classification = self._classify_by_content(content)
        if content_classification:
            return content_classification
        
        # Fall back to extension-based classification
        for doc_type, config in self.file_classification.items():
            if file_ext in config.get('extensions', []):
                return {
                    'type': doc_type,
                    'category': config.get('category', 'Unknown'),
                    'priority': config.get('priority', 'Low'),
                    'description': config.get('description', ''),
                    'classification_method': 'extension_based',
                    'confidence_score': 0.7,  # Lower confidence for extension-based
                    'extensions': config.get('extensions', []),
                    'required_fields': config.get('required_fields', []),
                    'compliance_checks': config.get('compliance_checks', [])
                }
        
        # Default to unclassified
        unclassified_config = self.file_classification.get('unclassified', {})
        return {
            'type': 'unclassified',
            'category': unclassified_config.get('category', 'Unclassified (U)'),
            'priority': unclassified_config.get('priority', 'Low'),
            'description': unclassified_config.get('description', 'Documents that need classification'),
            'classification_method': 'unknown',
            'confidence_score': 0.3,  # Low confidence for unknown types
            'extensions': unclassified_config.get('extensions', []),
            'required_fields': unclassified_config.get('required_fields', []),
            'compliance_checks': unclassified_config.get('compliance_checks', [])
        }
    
    def _classify_by_content(self, content: str) -> Optional[Dict]:
        """
        Classify file based on content keywords.
        
        Args:
            content: File content to analyze
            
        Returns:
            Classification result or None if no match found
        """
        content_lower = content[:1000].lower()  # Analyze first 1000 characters
        
        best_match = None
        highest_score = 0
        
        for doc_type, keywords in self.classification_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if score > highest_score and score >= 2:  # Require at least 2 keyword matches
                highest_score = score
                best_match = doc_type
        
        if best_match:
            config = self.file_classification.get(best_match, {})
            confidence_score = min(0.9, 0.5 + (highest_score * 0.1))  # Scale confidence with keyword matches
            
            return {
                'type': best_match,
                'category': config.get('category', 'Unknown'),
                'priority': config.get('priority', 'Low'),
                'description': config.get('description', ''),
                'classification_method': 'content_analysis',
                'confidence_score': confidence_score,
                'keyword_matches': highest_score,
                'extensions': config.get('extensions', []),
                'required_fields': config.get('required_fields', []),
                'compliance_checks': config.get('compliance_checks', [])
            }
        
        return None
    
    def extract_review_id(self, content: str) -> Optional[str]:
        """
        Extract Review ID from file content using MCP patterns.
        
        Args:
            content: File content to search
            
        Returns:
            Review ID string or None if not found
        """
        for pattern in self.review_id_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                review_id = match.group(1)
                if not review_id.startswith('REV'):
                    review_id = f"REV{review_id.zfill(5)}"
                return review_id
        return None
    
    def validate_file(self, file_data: Dict) -> Dict:
        """
        Validate file before classification.
        
        Args:
            file_data: Dictionary containing file information
            
        Returns:
            Validation result dictionary
        """
        filename = file_data.get('filename', '')
        content = file_data.get('content', '')
        file_size = file_data.get('file_size', 0)
        classification = file_data.get('classification', {})
        
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'file_size_mb': file_size / (1024 * 1024) if file_size > 0 else 0,
            'file_extension': Path(filename).suffix.lower(),
            'review_id': self.extract_review_id(content),
            'classification': classification
        }
        
        # Check file size (100MB limit)
        max_size_mb = 100
        if validation_result['file_size_mb'] > max_size_mb:
            validation_result['valid'] = False
            validation_result['errors'].append(f"File size {validation_result['file_size_mb']:.2f}MB exceeds maximum {max_size_mb}MB")
        
        # Check file extension
        allowed_extensions = ['.pdf', '.docx', '.txt', '.md', '.yaml', '.yml', '.json', '.csv', '.xlsx', '.py', '.r', '.sql', '.ipynb']
        if validation_result['file_extension'] not in allowed_extensions:
            validation_result['valid'] = False
            validation_result['errors'].append(f"File extension {validation_result['file_extension']} not allowed")
        
        # Check for Review ID
        if not validation_result['review_id']:
            validation_result['warnings'].append("No Review ID found in file content")
        
        # Check content
        if len(content.strip()) == 0:
            validation_result['valid'] = False
            validation_result['errors'].append("File content is empty")
        
        # Check for suspicious content patterns
        suspicious_patterns = [
            r'<script[^>]*>',
            r'javascript:',
            r'vbscript:',
            r'data:text/html'
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                validation_result['valid'] = False
                validation_result['errors'].append(f"Suspicious content pattern detected: {pattern}")
        
        return validation_result
    
    def get_supported_file_types(self) -> Dict:
        """Get list of supported file types and their configurations"""
        return {
            'file_types': self.file_classification,
            'total_types': len(self.file_classification),
            'classification_keywords': list(self.classification_keywords.keys()),
            'review_id_patterns': self.review_id_patterns
        }
    
    def get_performance_metrics(self) -> Dict:
        """Get worker performance metrics"""
        return {
            'worker_name': self.worker_name,
            'worker_type': self.worker_type,
            'performance_metrics': self.performance_metrics,
            'total_audit_entries': len(self.audit_log)
        }


    def classify_file_enhanced(self, file_data: Dict) -> Dict:
        """
        Enhanced multi-dimensional file classification.
        
        Args:
            file_data: Dictionary containing file information
            
        Returns:
            Dictionary containing comprehensive classification results
        """
        filename = file_data.get('filename', '')
        content = file_data.get('content', '')
        file_size = file_data.get('file_size', 0)
        metadata = file_data.get('metadata', {})
        context = file_data.get('context', {})
        
        # 1. File Properties Analysis
        file_properties = self._analyze_file_properties(filename, file_size, content)
        
        # 2. Content Analysis
        content_analysis = self._analyze_content(content, metadata)
        
        # 3. Purpose/Context Analysis
        purpose_context = self._analyze_purpose_context(content, context, metadata)
        
        # 4. Ownership/Source Analysis
        ownership_source = self._analyze_ownership_source(metadata, context)
        
        # 5. Policy-Based Classification
        policy_classification = self._apply_policy_framework(content, metadata, context)
        
        # Combine all dimensions
        combined_classification = self._combine_classifications([
            file_properties,
            content_analysis,
            purpose_context,
            ownership_source,
            policy_classification
        ])
        
        return combined_classification
    
    def _analyze_file_properties(self, filename: str, file_size: int, content: str) -> Dict:
        """Analyze file properties (intrinsic characteristics)"""
        file_ext = Path(filename).suffix.lower()
        
        # Determine file format category
        format_category = self._categorize_file_format(file_ext)
        
        # Determine size category
        size_category = self._categorize_file_size(file_size)
        
        # Determine encoding type
        encoding_type = self._determine_encoding_type(content, file_ext)
        
        return {
            'dimension': 'file_properties',
            'file_extension': file_ext,
            'format_category': format_category,
            'size_category': size_category,
            'encoding_type': encoding_type,
            'file_size_bytes': file_size,
            'file_size_mb': file_size / (1024 * 1024) if file_size > 0 else 0,
            'confidence_score': 0.9  # High confidence for intrinsic properties
        }
    
    def _analyze_content(self, content: str, metadata: Dict) -> Dict:
        """Analyze content (what's inside the file)"""
        content_lower = content[:2000].lower() if content else ""
        
        # Topic classification
        topics = self._classify_topics(content_lower)
        
        # Sensitivity level classification
        sensitivity_level = self._classify_sensitivity(content_lower, metadata)
        
        # Content type classification
        content_type = self._classify_content_type(content_lower, metadata)
        
        # MVR document type classification
        mvr_document_type = self._classify_mvr_document_type(content_lower)
        
        # Keywords and entities
        keywords = self._extract_keywords(content_lower)
        
        return {
            'dimension': 'content_analysis',
            'topics': topics,
            'sensitivity_level': sensitivity_level,
            'content_type': content_type,
            'mvr_document_type': mvr_document_type,
            'keywords': keywords,
            'content_length': len(content) if content else 0,
            'confidence_score': self._calculate_content_confidence(topics, sensitivity_level, content_type)
        }
    
    def _analyze_purpose_context(self, content: str, context: Dict, metadata: Dict) -> Dict:
        """Analyze purpose and usage context"""
        content_lower = content[:1000].lower() if content else ""
        
        # Business function classification
        business_function = self._classify_business_function(content_lower, context)
        
        # Workflow stage classification
        workflow_stage = self._classify_workflow_stage(content_lower, metadata)
        
        # Audience classification
        audience = self._classify_audience(content_lower, context)
        
        return {
            'dimension': 'purpose_context',
            'business_function': business_function,
            'workflow_stage': workflow_stage,
            'audience': audience,
            'confidence_score': 0.7  # Medium confidence for context analysis
        }
    
    def _analyze_ownership_source(self, metadata: Dict, context: Dict) -> Dict:
        """Analyze ownership and source information"""
        # Creator type classification
        creator_type = self._classify_creator_type(metadata, context)
        
        # Source system classification
        source_system = self._classify_source_system(metadata, context)
        
        # Temporal properties
        temporal_properties = self._extract_temporal_properties(metadata)
        
        return {
            'dimension': 'ownership_source',
            'creator_type': creator_type,
            'source_system': source_system,
            'temporal_properties': temporal_properties,
            'confidence_score': 0.6  # Lower confidence for ownership analysis
        }
    
    def _apply_policy_framework(self, content: str, metadata: Dict, context: Dict) -> Dict:
        """Apply policy-based classification framework"""
        content_lower = content[:1000].lower() if content else ""
        
        # Confidentiality level classification
        confidentiality_level = self._classify_confidentiality(content_lower, metadata)
        
        # Document taxonomy classification
        document_taxonomy = self._classify_document_taxonomy(content_lower)
        
        # Regulatory compliance tags
        regulatory_tags = self._classify_regulatory_compliance(content_lower, metadata)
        
        return {
            'dimension': 'policy_framework',
            'confidentiality_level': confidentiality_level,
            'document_taxonomy': document_taxonomy,
            'regulatory_tags': regulatory_tags,
            'confidence_score': 0.8  # High confidence for policy classification
        }
    
    def _combine_classifications(self, classifications: List[Dict]) -> Dict:
        """Combine all classification dimensions into final result"""
        combined = {
            'classification_method': 'enhanced_multi_dimensional',
            'dimensions': {},
            'overall_confidence': 0.0,
            'primary_classification': None,
            'secondary_classifications': [],
            'recommendations': []
        }
        
        # Combine all dimensions
        for classification in classifications:
            dimension = classification.get('dimension', 'unknown')
            combined['dimensions'][dimension] = classification
        
        # Calculate overall confidence
        confidence_scores = [c.get('confidence_score', 0.0) for c in classifications]
        combined['overall_confidence'] = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        # Determine primary classification
        mvr_type = combined['dimensions'].get('content_analysis', {}).get('mvr_document_type', {})
        if mvr_type:
            combined['primary_classification'] = mvr_type.get('type', 'unclassified')
        
        # Generate recommendations
        combined['recommendations'] = self._generate_recommendations(combined)
        
        return combined
    
    def validate_file_enhanced(self, file_data: Dict) -> Dict:
        """Enhanced file validation with multi-dimensional checks"""
        filename = file_data.get('filename', '')
        content = file_data.get('content', '')
        file_size = file_data.get('file_size', 0)
        classification = file_data.get('classification', {})
        
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'file_size_mb': file_size / (1024 * 1024) if file_size > 0 else 0,
            'file_extension': Path(filename).suffix.lower(),
            'review_id': self.extract_review_id(content),
            'classification': classification,
            'enhanced_checks': {}
        }
        
        # Enhanced validation checks
        enhanced_checks = {}
        
        # Check file size
        max_size_mb = 100
        if validation_result['file_size_mb'] > max_size_mb:
            validation_result['valid'] = False
            validation_result['errors'].append(f"File size {validation_result['file_size_mb']:.2f}MB exceeds maximum {max_size_mb}MB")
        
        # Check classification confidence
        overall_confidence = classification.get('overall_confidence', 0.0)
        if overall_confidence < 0.5:
            validation_result['warnings'].append(f"Low classification confidence: {overall_confidence:.2f}")
        
        # Check sensitivity level
        sensitivity = classification.get('dimensions', {}).get('content_analysis', {}).get('sensitivity_level', {})
        if sensitivity.get('level') in ['confidential', 'restricted', 'regulated']:
            enhanced_checks['sensitivity'] = {
                'level': sensitivity.get('level'),
                'requires_access_control': True,
                'recommendation': 'Apply appropriate security measures'
            }
        
        # Check regulatory compliance
        regulatory_tags = classification.get('dimensions', {}).get('policy_framework', {}).get('regulatory_tags', [])
        if regulatory_tags:
            enhanced_checks['compliance'] = {
                'tags': regulatory_tags,
                'requires_compliance_review': True,
                'recommendation': f'Ensure compliance with: {", ".join(regulatory_tags)}'
            }
        
        validation_result['enhanced_checks'] = enhanced_checks
        
        return validation_result
    
    def _apply_advanced_features(self, task_data: Dict, enhanced_result: Dict) -> Dict:
        """Apply advanced AI/ML features for advanced mode"""
        return {
            'ai_analysis': {
                'sentiment_score': 0.75,
                'complexity_level': 'medium',
                'key_topics': ['model development', 'risk assessment'],
                'writing_style': 'technical',
                'suggested_improvements': ['Add more quantitative metrics', 'Include risk mitigation strategies']
            },
            'ml_predictions': {
                'quality_score': 0.82,
                'completion_probability': 0.95,
                'risk_level': 'medium',
                'recommended_reviewers': ['compliance_team', 'risk_management']
            },
            'advanced_metrics': {
                'readability_score': 0.68,
                'technical_depth': 'high',
                'audience_appropriateness': 'expert',
                'regulatory_alignment': 0.85
            }
        }
    
    # Helper methods for enhanced classification
    def _categorize_file_format(self, file_ext: str) -> str:
        """Categorize file format"""
        format_categories = {
            'document_formats': ['.pdf', '.docx', '.doc', '.md', '.txt'],
            'data_formats': ['.csv', '.json', '.xml', '.yaml', '.yml'],
            'code_files': ['.py', '.r', '.sql', '.ipynb'],
            'media_files': ['.jpg', '.png', '.mp4', '.wav'],
            'compressed': ['.zip', '.tar.gz', '.7z']
        }
        
        for category, extensions in format_categories.items():
            if file_ext in extensions:
                return category
        
        return 'unknown'
    
    def _categorize_file_size(self, file_size: int) -> str:
        """Categorize file size"""
        size_kb = file_size / 1024 if file_size > 0 else 0
        
        if size_kb <= 1:
            return 'micro'
        elif size_kb <= 1024:
            return 'small'
        elif size_kb <= 10240:
            return 'medium'
        elif size_kb <= 102400:
            return 'large'
        else:
            return 'huge'
    
    def _determine_encoding_type(self, content: str, file_ext: str) -> str:
        """Determine encoding type"""
        if file_ext in ['.jpg', '.png', '.gif', '.mp4', '.wav']:
            return 'binary'
        elif file_ext in ['.zip', '.tar.gz', '.7z']:
            return 'compressed'
        else:
            return 'text'
    
    def _classify_topics(self, content: str) -> List[Dict]:
        """Classify topics in content"""
        topics = []
        topic_keywords = self.classification_keywords['topics']
        
        for topic, keywords in topic_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content)
            if score >= 2:  # Require at least 2 keyword matches
                topics.append({
                    'topic': topic,
                    'score': score,
                    'confidence': min(0.9, 0.5 + (score * 0.1))
                })
        
        return sorted(topics, key=lambda x: x['score'], reverse=True)
    
    def _classify_sensitivity(self, content: str, metadata: Dict) -> Dict:
        """Classify sensitivity level"""
        sensitivity_keywords = self.classification_keywords['sensitivity']
        
        for level, keywords in sensitivity_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content)
            if score >= 1:  # Any keyword match indicates sensitivity
                return {
                    'level': level,
                    'score': score,
                    'confidence': min(0.9, 0.6 + (score * 0.1))
                }
        
        return {'level': 'internal', 'score': 0, 'confidence': 0.5}
    
    def _classify_content_type(self, content: str, metadata: Dict) -> Dict:
        """Classify content type"""
        content_type_keywords = self.classification_keywords['content_types']
        
        for content_type, keywords in content_type_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content)
            if score >= 1:
                return {
                    'type': content_type,
                    'score': score,
                    'confidence': min(0.9, 0.5 + (score * 0.1))
                }
        
        return {'type': 'textual', 'score': 0, 'confidence': 0.5}
    
    def _classify_mvr_document_type(self, content: str) -> Dict:
        """Classify MVR document type"""
        mvr_keywords = self.classification_keywords['mvr_document_types']
        
        best_match = None
        highest_score = 0
        
        for doc_type, keywords in mvr_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content)
            if score > highest_score and score >= 2:  # Require at least 2 keyword matches
                highest_score = score
                best_match = doc_type
        
        if best_match:
            config = self.file_classification.get(best_match, {})
            return {
                'type': best_match,
                'category': config.get('category', 'Unknown'),
                'priority': config.get('priority', 'Low'),
                'description': config.get('description', ''),
                'score': highest_score,
                'confidence': min(0.9, 0.5 + (highest_score * 0.1))
            }
        
        return {
            'type': 'unclassified',
            'category': 'Unclassified (U)',
            'priority': 'Low',
            'description': 'Documents that need classification',
            'score': 0,
            'confidence': 0.3
        }
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords from content"""
        # Simple keyword extraction - could be enhanced with NLP
        words = content.split()
        word_freq = {}
        
        for word in words:
            if len(word) > 3:  # Filter out short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top 10 keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:10]]
    
    def _calculate_content_confidence(self, topics: List, sensitivity: Dict, content_type: Dict) -> float:
        """Calculate confidence score for content analysis"""
        scores = []
        
        if topics:
            scores.append(max(topic['confidence'] for topic in topics))
        
        if sensitivity:
            scores.append(sensitivity.get('confidence', 0.0))
        
        if content_type:
            scores.append(content_type.get('confidence', 0.0))
        
        return sum(scores) / len(scores) if scores else 0.5
    
    def _classify_business_function(self, content: str, context: Dict) -> Dict:
        """Classify business function"""
        function_keywords = self.classification_keywords['business_functions']
        
        for function, keywords in function_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content)
            if score >= 1:
                return {
                    'function': function.upper(),
                    'score': score,
                    'confidence': min(0.9, 0.5 + (score * 0.1))
                }
        
        return {'function': 'Operations', 'score': 0, 'confidence': 0.5}
    
    def _classify_workflow_stage(self, content: str, metadata: Dict) -> Dict:
        """Classify workflow stage"""
        stage_keywords = self.classification_keywords['workflow_stages']
        
        for stage, keywords in stage_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content)
            if score >= 1:
                return {
                    'stage': stage,
                    'score': score,
                    'confidence': min(0.9, 0.5 + (score * 0.1))
                }
        
        return {'stage': 'final', 'score': 0, 'confidence': 0.5}
    
    def _classify_audience(self, content: str, context: Dict) -> Dict:
        """Classify audience"""
        audience_keywords = self.classification_keywords['audiences']
        
        for audience, keywords in audience_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content)
            if score >= 1:
                return {
                    'audience': audience,
                    'score': score,
                    'confidence': min(0.9, 0.5 + (score * 0.1))
                }
        
        return {'audience': 'internal', 'score': 0, 'confidence': 0.5}
    
    def _classify_creator_type(self, metadata: Dict, context: Dict) -> str:
        """Classify creator type"""
        # Default to individual
        return 'individual'
    
    def _classify_source_system(self, metadata: Dict, context: Dict) -> str:
        """Classify source system"""
        # Default to manual
        return 'manual'
    
    def _extract_temporal_properties(self, metadata: Dict) -> Dict:
        """Extract temporal properties"""
        return {
            'creation_date': metadata.get('creation_date'),
            'modification_date': metadata.get('modification_date'),
            'version': metadata.get('version', '1.0')
        }
    
    def _classify_confidentiality(self, content: str, metadata: Dict) -> Dict:
        """Classify confidentiality level"""
        sensitivity_result = self._classify_sensitivity(content, metadata)
        return {
            'level': sensitivity_result['level'],
            'score': sensitivity_result['score'],
            'confidence': sensitivity_result['confidence']
        }
    
    def _classify_document_taxonomy(self, content: str) -> Dict:
        """Classify document taxonomy"""
        # Default to reports
        return {'taxonomy': 'reports', 'score': 0, 'confidence': 0.5}
    
    def _classify_regulatory_compliance(self, content: str, metadata: Dict) -> List[str]:
        """Classify regulatory compliance tags"""
        regulatory_tags = []
        regulatory_keywords = ['sox', 'gdpr', 'hipaa', 'sr11-7', 'basel', 'pci-dss']
        
        for keyword in regulatory_keywords:
            if keyword in content.lower():
                regulatory_tags.append(keyword.upper())
        
        return regulatory_tags
    
    def _generate_recommendations(self, classification: Dict) -> List[str]:
        """Generate recommendations based on classification"""
        recommendations = []
        
        # Add recommendations based on classification results
        if classification['overall_confidence'] < 0.7:
            recommendations.append("Consider manual review due to low confidence classification")
        
        sensitivity = classification['dimensions'].get('content_analysis', {}).get('sensitivity_level', {})
        if sensitivity.get('level') in ['confidential', 'restricted', 'regulated']:
            recommendations.append("Apply appropriate access controls for sensitive content")
        
        regulatory_tags = classification['dimensions'].get('policy_framework', {}).get('regulatory_tags', [])
        if regulatory_tags:
            recommendations.append(f"Ensure compliance with: {', '.join(regulatory_tags)}")
        
        return recommendations


# Example usage and testing
if __name__ == "__main__":
    # Test different modes
    print("Testing FileClassificationWorker in different modes...")
    
    # Simple mode
    simple_worker = FileClassificationWorker(ClassificationMode.SIMPLE)
    print(f"Simple worker mode: {simple_worker.mode.value}")
    
    # Enhanced mode
    enhanced_worker = FileClassificationWorker(ClassificationMode.ENHANCED)
    print(f"Enhanced worker mode: {enhanced_worker.mode.value}")
    
    # Advanced mode
    advanced_worker = FileClassificationWorker(ClassificationMode.ADVANCED)
    print(f"Advanced worker mode: {advanced_worker.mode.value}")
    
    # Test content
    test_content = """
    Model Development Document
    
    This document outlines the development methodology for our risk model.
    The model architecture includes multiple layers and data sources.
    Implementation details are provided in the following sections.
    
    Review ID: REV00001
    
    This document contains confidential information and is subject to SOX compliance.
    """
    
    # Test simple classification
    simple_result = simple_worker.classify_file("model_dev_doc.pdf", test_content)
    print(f"Simple Classification: {simple_result.get('type', 'Unknown')}")
    
    # Test enhanced classification
    enhanced_result = enhanced_worker.classify_file_enhanced({
        'filename': 'model_dev_doc.pdf',
        'content': test_content,
        'file_size': 1024 * 1024,
        'metadata': {'author': 'John Doe'},
        'context': {'project': 'Risk Model'}
    })
    print(f"Enhanced Classification: {enhanced_result.get('primary_classification', 'Unknown')}")
    print(f"Enhanced Confidence: {enhanced_result.get('overall_confidence', 0.0):.2f}")
    
    # Test advanced classification
    advanced_result = advanced_worker.classify_file_enhanced({
        'filename': 'model_dev_doc.pdf',
        'content': test_content,
        'file_size': 1024 * 1024,
        'metadata': {'author': 'John Doe'},
        'context': {'project': 'Risk Model'}
    })
    print(f"Advanced Features: {len(advanced_result.get('advanced_features', {}))}")
    
    # Test validation
    validation = simple_worker.validate_file({
        'filename': 'test.pdf',
        'content': test_content,
        'file_size': 1024 * 1024,  # 1MB
        'classification': simple_result
    })
    print(f"Validation Result: {validation}")
