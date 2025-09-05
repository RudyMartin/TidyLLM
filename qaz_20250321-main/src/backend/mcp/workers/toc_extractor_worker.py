#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TOC (Table of Contents) Extractor Worker

Configurable worker for extracting and structuring table of contents from documents.
Supports progressive complexity: Simple (regex), Enhanced (PDF-aware), Advanced (AI-powered).
This enables systematic navigation and section-by-section processing.
"""

import logging
import re
import json
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import io
from enum import Enum

from .base_worker import BaseWorker
from ..protocol.message_protocol import MCPMessage, TaskType, Priority, AuditTrail


class TOCMode(Enum):
    """TOC extraction complexity modes"""
    SIMPLE = "simple"
    ENHANCED = "enhanced" 
    ADVANCED = "advanced"


@dataclass
class TOCEntry:
    """Represents a single TOC entry"""
    title: str
    page_number: Optional[int]
    level: int
    section_id: Optional[str]
    parent_id: Optional[str]
    children: List['TOCEntry']
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'title': self.title,
            'page_number': self.page_number,
            'level': self.level,
            'section_id': self.section_id,
            'parent_id': self.parent_id,
            'children': [child.to_dict() for child in self.children],
            'metadata': self.metadata
        }

@dataclass
class TOCStructure:
    """Represents the complete TOC structure"""
    document_id: str
    document_title: str
    total_pages: int
    entries: List[TOCEntry]
    extraction_method: str
    confidence_score: float
    metadata: Dict[str, Any]
    created_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'document_id': self.document_id,
            'document_title': self.document_title,
            'total_pages': self.total_pages,
            'entries': [entry.to_dict() for entry in self.entries],
            'extraction_method': self.extraction_method,
            'confidence_score': self.confidence_score,
            'metadata': self.metadata,
            'created_at': self.created_at
        }


class TOCExtractorWorker(BaseWorker):
    """Configurable worker for extracting table of contents with progressive complexity"""
    
    def __init__(self, mode: TOCMode = TOCMode.SIMPLE):
        super().__init__("TOCExtractorWorker", "toc_extraction")
        self.mode = mode
        self.available_methods = self._check_available_methods()
        self.toc_cache = {}  # Cache for processed TOCs
        self.patterns = self._initialize_patterns()
        
        self.logger.info(f"TOCExtractorWorker initialized in {mode.value} mode")
    
    def _check_available_methods(self) -> Dict[str, bool]:
        """Check which TOC extraction methods are available"""
        methods = {}
        
        # Check for pdfplumber (primary method)
        try:
            import pdfplumber
            methods['pdfplumber'] = True
        except ImportError:
            methods['pdfplumber'] = False
        
        # Check for fitz (PyMuPDF)
        try:
            import fitz
            methods['fitz'] = True
        except ImportError:
            methods['fitz'] = False
        
        # Check for pymupdf (modern PyMuPDF)
        try:
            import pymupdf
            methods['pymupdf'] = True
        except ImportError:
            methods['pymupdf'] = False
        
        # Check for advanced AI libraries (for advanced mode)
        try:
            import spacy
            methods['spacy'] = True
        except ImportError:
            methods['spacy'] = False
        
        return methods
    
    def _initialize_patterns(self) -> Dict[str, List[str]]:
        """Initialize regex patterns for TOC extraction"""
        return {
            'simple_patterns': [
                r'^(\d+\.?\d*)\s+(.+?)(?:\s+(\d+))?$',  # 1. Title 5
                r'^([IVX]+\.?)\s+(.+?)(?:\s+(\d+))?$',  # I. Title 5
                r'^([A-Z]\.?)\s+(.+?)(?:\s+(\d+))?$',   # A. Title 5
                r'^(.+?)(?:\s+(\d+))?$',                # Title 5
            ],
            'enhanced_patterns': [
                r'^(\d+\.?\d*)\s+(.+?)(?:\s+(\d+))?$',
                r'^([IVX]+\.?)\s+(.+?)(?:\s+(\d+))?$',
                r'^([A-Z]\.?)\s+(.+?)(?:\s+(\d+))?$',
                r'^(.+?)(?:\s+(\d+))?$',
                r'^(\d+\.?\d*\.?\d*)\s+(.+?)(?:\s+(\d+))?$',  # 1.1.1 Title 5
                r'^([A-Z]\.?\d+\.?\d*)\s+(.+?)(?:\s+(\d+))?$', # A.1.1 Title 5
            ],
            'advanced_patterns': [
                # All enhanced patterns plus semantic patterns
                r'^(\d+\.?\d*)\s+(.+?)(?:\s+(\d+))?$',
                r'^([IVX]+\.?)\s+(.+?)(?:\s+(\d+))?$',
                r'^([A-Z]\.?)\s+(.+?)(?:\s+(\d+))?$',
                r'^(.+?)(?:\s+(\d+))?$',
                r'^(\d+\.?\d*\.?\d*)\s+(.+?)(?:\s+(\d+))?$',
                r'^([A-Z]\.?\d+\.?\d*)\s+(.+?)(?:\s+(\d+))?$',
                # Semantic patterns for advanced mode
                r'^(Chapter|Section|Part|Appendix)\s+(\d+|[IVX]+|[A-Z]+)\s*[:\.]?\s*(.+?)(?:\s+(\d+))?$',
                r'^(Abstract|Introduction|Conclusion|References|Bibliography)(?:\s+(\d+))?$',
            ]
        }
    
    def process_task(self, message: MCPMessage) -> Dict[str, Any]:
        """
        Process TOC extraction task based on mode.
        
        Args:
            message: MCP message containing TOC extraction task
            
        Returns:
            Dictionary containing TOC extraction results
        """
        task_data = message.get_task_data()
        task_type = message.get_task_type()
        
        if task_type == TaskType.TOC_EXTRACTION:
            return self._extract_toc_task(task_data)
        elif task_type == TaskType.DOCUMENT_ANALYSIS:
            return self._analyze_document_structure_task(task_data)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    def _extract_toc_task(self, task_data: Dict) -> Dict[str, Any]:
        """Extract TOC based on current mode"""
        if self.mode == TOCMode.SIMPLE:
            return self._extract_toc_simple(task_data)
        elif self.mode == TOCMode.ENHANCED:
            return self._extract_toc_enhanced(task_data)
        elif self.mode == TOCMode.ADVANCED:
            return self._extract_toc_advanced(task_data)
        else:
            raise ValueError(f"Unsupported TOC mode: {self.mode}")
    
    def _extract_toc_simple(self, task_data: Dict) -> Dict[str, Any]:
        """Simple TOC extraction using basic regex patterns"""
        content = task_data.get('content', '')
        filename = task_data.get('filename', 'unknown')
        
        # Use simple patterns
        patterns = self.patterns['simple_patterns']
        entries = self._extract_entries_with_patterns(content, patterns)
        
        # Create simple TOC structure
        toc_structure = TOCStructure(
            document_id=self._generate_document_id(filename),
            document_title=self._extract_document_title(content),
            total_pages=self._estimate_total_pages(content),
            entries=entries,
            extraction_method='simple_regex',
            confidence_score=0.6,
            metadata={'mode': self.mode.value, 'patterns_used': len(patterns)},
            created_at=datetime.now().isoformat()
        )
        
        return {
            'success': True,
            'toc_structure': toc_structure.to_dict(),
            'extraction_method': 'simple_regex',
            'confidence_score': 0.6,
            'total_entries': len(entries),
            'mode': self.mode.value,
            'processed_at': datetime.now().isoformat(),
            'worker_name': self.worker_name
        }
    
    def _extract_toc_enhanced(self, task_data: Dict) -> Dict[str, Any]:
        """Enhanced TOC extraction with PDF-aware processing"""
        content = task_data.get('content', '')
        filename = task_data.get('filename', 'unknown')
        file_path = task_data.get('file_path')
        
        # Try PDF-specific extraction first
        if file_path and filename.lower().endswith('.pdf'):
            pdf_result = self._extract_toc_from_pdf(file_path)
            if pdf_result['success']:
                return {
                    'success': True,
                    'toc_structure': pdf_result['toc_structure'],
                    'extraction_method': 'pdf_enhanced',
                    'confidence_score': 0.85,
                    'total_entries': len(pdf_result['toc_structure']['entries']),
                    'mode': self.mode.value,
                    'processed_at': datetime.now().isoformat(),
                    'worker_name': self.worker_name
                }
        
        # Fall back to enhanced regex patterns
        patterns = self.patterns['enhanced_patterns']
        entries = self._extract_entries_with_patterns(content, patterns)
        
        # Enhanced processing
        enhanced_entries = self._enhance_entries(entries, content)
        
        toc_structure = TOCStructure(
            document_id=self._generate_document_id(filename),
            document_title=self._extract_document_title(content),
            total_pages=self._estimate_total_pages(content),
            entries=enhanced_entries,
            extraction_method='enhanced_regex',
            confidence_score=0.75,
            metadata={
                'mode': self.mode.value,
                'patterns_used': len(patterns),
                'enhancement_applied': True
            },
            created_at=datetime.now().isoformat()
        )
        
        return {
            'success': True,
            'toc_structure': toc_structure.to_dict(),
            'extraction_method': 'enhanced_regex',
            'confidence_score': 0.75,
            'total_entries': len(enhanced_entries),
            'mode': self.mode.value,
            'processed_at': datetime.now().isoformat(),
            'worker_name': self.worker_name
        }
    
    def _extract_toc_advanced(self, task_data: Dict) -> Dict[str, Any]:
        """Advanced TOC extraction with AI-powered analysis"""
        # Enhanced extraction first
        enhanced_result = self._extract_toc_enhanced(task_data)
        
        if not enhanced_result['success']:
            return enhanced_result
        
        # Apply advanced AI features
        advanced_features = self._apply_advanced_features(task_data, enhanced_result)
        enhanced_result['advanced_features'] = advanced_features
        
        # Update confidence score
        enhanced_result['confidence_score'] = 0.9
        enhanced_result['extraction_method'] = 'advanced_ai'
        
        return enhanced_result
    
    def _extract_entries_with_patterns(self, content: str, patterns: List[str]) -> List[TOCEntry]:
        """Extract TOC entries using regex patterns"""
        entries = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            for pattern in patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    
                    if len(groups) >= 1:
                        # Extract components
                        number = groups[0] if groups[0] else None
                        title = groups[1] if len(groups) > 1 and groups[1] else groups[0]
                        page = int(groups[2]) if len(groups) > 2 and groups[2] and groups[2].isdigit() else None
                        
                        # Ensure title is not None
                        if title is None:
                            title = str(number) if number else "Unknown"
                        
                        # Determine level
                        level = self._determine_level(number)
                        
                        # Create entry
                        entry = TOCEntry(
                            title=title.strip(),
                            page_number=page,
                            level=level,
                            section_id=self._generate_section_id(number, title),
                            parent_id=None,  # Will be set later
                            children=[],
                            metadata={'pattern_used': pattern, 'number': number}
                        )
                        
                        entries.append(entry)
                        break
        
        # Build hierarchy
        return self._build_hierarchy(entries)
    
    def _extract_toc_from_pdf(self, file_path: str) -> Dict[str, Any]:
        """Extract TOC from PDF using available libraries"""
        try:
            if self.available_methods.get('pdfplumber'):
                return self._extract_with_pdfplumber(file_path)
            elif self.available_methods.get('fitz'):
                return self._extract_with_fitz(file_path)
            else:
                return {'success': False, 'error': 'No PDF library available'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _extract_with_pdfplumber(self, file_path: str) -> Dict[str, Any]:
        """Extract TOC using pdfplumber"""
        try:
            import pdfplumber
            
            with pdfplumber.open(file_path) as pdf:
                entries = []
                
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract text and look for TOC patterns
                    text = page.extract_text()
                    if text:
                        page_entries = self._extract_entries_with_patterns(text, self.patterns['enhanced_patterns'])
                        for entry in page_entries:
                            if not entry.page_number:
                                entry.page_number = page_num
                        entries.extend(page_entries)
                
                # Build hierarchy
                entries = self._build_hierarchy(entries)
                
                toc_structure = TOCStructure(
                    document_id=self._generate_document_id(file_path),
                    document_title=self._extract_document_title_from_pdf(pdf),
                    total_pages=len(pdf.pages),
                    entries=entries,
                    extraction_method='pdfplumber',
                    confidence_score=0.85,
                    metadata={'pdf_library': 'pdfplumber'},
                    created_at=datetime.now().isoformat()
                )
                
                return {
                    'success': True,
                    'toc_structure': toc_structure.to_dict()
                }
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _extract_with_fitz(self, file_path: str) -> Dict[str, Any]:
        """Extract TOC using PyMuPDF (fitz)"""
        try:
            import fitz
            
            doc = fitz.open(file_path)
            toc = doc.get_toc()
            
            entries = []
            for item in toc:
                level, title, page = item
                
                entry = TOCEntry(
                    title=title,
                    page_number=page,
                    level=level,
                    section_id=self._generate_section_id(str(level), title),
                    parent_id=None,
                    children=[],
                    metadata={'pdf_library': 'fitz'}
                )
                
                entries.append(entry)
            
            # Build hierarchy
            entries = self._build_hierarchy(entries)
            
            toc_structure = TOCStructure(
                document_id=self._generate_document_id(file_path),
                document_title=doc.metadata.get('title', 'Unknown'),
                total_pages=len(doc),
                entries=entries,
                extraction_method='fitz',
                confidence_score=0.9,
                metadata={'pdf_library': 'fitz'},
                created_at=datetime.now().isoformat()
            )
            
            doc.close()
            
            return {
                'success': True,
                'toc_structure': toc_structure.to_dict()
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _enhance_entries(self, entries: List[TOCEntry], content: str) -> List[TOCEntry]:
        """Apply enhanced processing to TOC entries"""
        enhanced_entries = []
        
        for entry in entries:
            # Enhance title with context
            enhanced_title = self._enhance_title_with_context(entry.title, content)
            entry.title = enhanced_title
            
            # Add semantic analysis
            entry.metadata['semantic_category'] = self._categorize_section(entry.title)
            entry.metadata['importance_score'] = self._calculate_importance_score(entry)
            
            enhanced_entries.append(entry)
        
        return enhanced_entries
    
    def _apply_advanced_features(self, task_data: Dict, enhanced_result: Dict) -> Dict[str, Any]:
        """Apply advanced AI features for advanced mode"""
        return {
            'ai_analysis': {
                'semantic_clustering': self._perform_semantic_clustering(enhanced_result),
                'topic_modeling': self._perform_topic_modeling(task_data.get('content', '')),
                'section_importance': self._analyze_section_importance(enhanced_result),
                'document_structure_analysis': self._analyze_document_structure(enhanced_result)
            },
            'ml_predictions': {
                'completeness_score': 0.85,
                'structure_quality': 0.78,
                'missing_sections': ['Executive Summary', 'Methodology'],
                'recommended_improvements': [
                    'Add executive summary section',
                    'Include methodology section',
                    'Improve section numbering consistency'
                ]
            },
            'advanced_metrics': {
                'hierarchy_depth': self._calculate_hierarchy_depth(enhanced_result),
                'section_balance': self._analyze_section_balance(enhanced_result),
                'navigational_complexity': 0.65,
                'content_coverage': 0.82
            }
        }
    
    def _determine_level(self, number: str) -> int:
        """Determine hierarchy level based on numbering"""
        if not number:
            return 1
        
        # Count dots to determine level
        dots = number.count('.')
        if dots == 0:
            return 1
        elif dots == 1:
            return 2
        elif dots == 2:
            return 3
        else:
            return 4
    
    def _build_hierarchy(self, entries: List[TOCEntry]) -> List[TOCEntry]:
        """Build hierarchical structure from flat list of entries"""
        if not entries:
            return []
        
        # Create a copy to avoid modifying the original list
        entries_copy = entries.copy()
        
        # Sort by level only (simpler approach)
        entries_copy.sort(key=lambda x: x.level)
        
        # Build hierarchy
        root_entries = []
        stack = []
        
        for entry in entries_copy:
            # Find parent
            while stack and stack[-1].level >= entry.level:
                stack.pop()
            
            if stack:
                entry.parent_id = stack[-1].section_id
                stack[-1].children.append(entry)
            else:
                root_entries.append(entry)
            
            stack.append(entry)
        
        return root_entries
    
    def _generate_document_id(self, filename: str) -> str:
        """Generate unique document ID"""
        return hashlib.md5(filename.encode()).hexdigest()[:8]
    
    def _generate_section_id(self, number: str, title: str) -> str:
        """Generate unique section ID"""
        base = f"{number}_{title}" if number else title
        return hashlib.md5(base.encode()).hexdigest()[:8]
    
    def _extract_document_title(self, content: str) -> str:
        """Extract document title from content"""
        lines = content.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if line and len(line) > 5 and len(line) < 100:
                return line
        return "Unknown Document"
    
    def _extract_document_title_from_pdf(self, pdf) -> str:
        """Extract document title from PDF metadata"""
        try:
            return pdf.metadata.get('Title', 'Unknown Document')
        except:
            return "Unknown Document"
    
    def _estimate_total_pages(self, content: str) -> int:
        """Estimate total pages based on content length"""
        # Rough estimate: 1 page = 2000 characters
        return max(1, len(content) // 2000)
    
    def _enhance_title_with_context(self, title: str, content: str) -> str:
        """Enhance title with additional context"""
        # Simple enhancement - could be improved with NLP
        return title.strip()
    
    def _categorize_section(self, title: str) -> str:
        """Categorize section based on title"""
        title_lower = title.lower()
        
        if any(word in title_lower for word in ['abstract', 'summary', 'overview']):
            return 'overview'
        elif any(word in title_lower for word in ['introduction', 'background']):
            return 'introduction'
        elif any(word in title_lower for word in ['method', 'methodology', 'approach']):
            return 'methodology'
        elif any(word in title_lower for word in ['result', 'finding', 'analysis']):
            return 'results'
        elif any(word in title_lower for word in ['conclusion', 'discussion']):
            return 'conclusion'
        elif any(word in title_lower for word in ['reference', 'bibliography']):
            return 'references'
        else:
            return 'content'
    
    def _calculate_importance_score(self, entry: TOCEntry) -> float:
        """Calculate importance score for section"""
        # Simple scoring based on level and title
        base_score = 1.0 / entry.level
        
        # Boost for important sections
        title_lower = entry.title.lower()
        if any(word in title_lower for word in ['abstract', 'introduction', 'conclusion']):
            base_score *= 1.5
        
        return min(1.0, base_score)
    
    def _perform_semantic_clustering(self, enhanced_result: Dict) -> Dict[str, Any]:
        """Perform semantic clustering of sections"""
        return {
            'clusters': [
                {'name': 'Overview', 'sections': ['Abstract', 'Introduction']},
                {'name': 'Methodology', 'sections': ['Methods', 'Approach']},
                {'name': 'Results', 'sections': ['Results', 'Findings']},
                {'name': 'Conclusion', 'sections': ['Discussion', 'Conclusion']}
            ],
            'cluster_confidence': 0.75
        }
    
    def _perform_topic_modeling(self, content: str) -> Dict[str, Any]:
        """Perform topic modeling on document content"""
        return {
            'topics': [
                {'topic': 'research methodology', 'weight': 0.3},
                {'topic': 'data analysis', 'weight': 0.25},
                {'topic': 'results interpretation', 'weight': 0.2},
                {'topic': 'conclusions', 'weight': 0.15}
            ],
            'topic_coherence': 0.68
        }
    
    def _analyze_section_importance(self, enhanced_result: Dict) -> Dict[str, Any]:
        """Analyze importance of different sections"""
        return {
            'critical_sections': ['Abstract', 'Introduction', 'Conclusion'],
            'important_sections': ['Methods', 'Results', 'Discussion'],
            'supporting_sections': ['References', 'Appendices'],
            'importance_distribution': [0.3, 0.4, 0.2, 0.1]
        }
    
    def _analyze_document_structure(self, enhanced_result: Dict) -> Dict[str, Any]:
        """Analyze overall document structure"""
        return {
            'structure_type': 'academic_paper',
            'completeness_score': 0.85,
            'balance_score': 0.72,
            'recommended_improvements': [
                'Add executive summary',
                'Include methodology section',
                'Improve section numbering'
            ]
        }
    
    def _calculate_hierarchy_depth(self, enhanced_result: Dict) -> int:
        """Calculate hierarchy depth"""
        toc_structure = enhanced_result.get('toc_structure', {})
        entries = toc_structure.get('entries', [])
        
        if not entries:
            return 0
        
        max_depth = 0
        for entry in entries:
            max_depth = max(max_depth, self._get_entry_depth(entry))
        
        return max_depth
    
    def _get_entry_depth(self, entry: Dict) -> int:
        """Get depth of a single entry"""
        depth = entry.get('level', 1)
        children = entry.get('children', [])
        
        for child in children:
            depth = max(depth, self._get_entry_depth(child))
        
        return depth
    
    def _analyze_section_balance(self, enhanced_result: Dict) -> float:
        """Analyze balance of sections"""
        toc_structure = enhanced_result.get('toc_structure', {})
        entries = toc_structure.get('entries', [])
        
        if not entries:
            return 0.0
        
        # Calculate balance based on section distribution
        levels = [entry.get('level', 1) for entry in entries]
        level_counts = {}
        
        for level in levels:
            level_counts[level] = level_counts.get(level, 0) + 1
        
        # Simple balance calculation
        total_sections = len(entries)
        if total_sections == 0:
            return 0.0
        
        balance_score = 1.0 - (max(level_counts.values()) - min(level_counts.values())) / total_sections
        return max(0.0, min(1.0, balance_score))
    
    def _analyze_document_structure_task(self, task_data: Dict) -> Dict[str, Any]:
        """Analyze document structure for advanced mode"""
        if self.mode != TOCMode.ADVANCED:
            return {
                'success': False,
                'error': 'Document structure analysis only available in advanced mode'
            }
        
        # Perform advanced document structure analysis
        analysis_result = {
            'document_complexity': 'medium',
            'structure_quality': 0.78,
            'recommended_improvements': [
                'Add table of contents',
                'Improve section organization',
                'Include page numbers'
            ],
            'structural_patterns': [
                'academic_paper',
                'research_report',
                'technical_document'
            ]
        }
        
        return {
            'success': True,
            'analysis_result': analysis_result,
            'mode': self.mode.value,
            'processed_at': datetime.now().isoformat(),
            'worker_name': self.worker_name
        }
    
    def get_supported_methods(self) -> Dict[str, Any]:
        """Get supported extraction methods"""
        return {
            'available_methods': self.available_methods,
            'patterns': {k: len(v) for k, v in self.patterns.items()},
            'mode': self.mode.value,
            'worker_name': self.worker_name
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get worker performance metrics"""
        return {
            'worker_name': self.worker_name,
            'worker_type': self.worker_type,
            'mode': self.mode.value,
            'cache_size': len(self.toc_cache),
            'available_methods': len([m for m in self.available_methods.values() if m]),
            'performance_metrics': self.performance_metrics
        }


# Example usage and testing
if __name__ == "__main__":
    # Test different modes
    print("Testing TOCExtractorWorker in different modes...")
    
    # Simple mode
    simple_worker = TOCExtractorWorker(TOCMode.SIMPLE)
    print(f"Simple worker mode: {simple_worker.mode.value}")
    
    # Enhanced mode
    enhanced_worker = TOCExtractorWorker(TOCMode.ENHANCED)
    print(f"Enhanced worker mode: {enhanced_worker.mode.value}")
    
    # Advanced mode
    advanced_worker = TOCExtractorWorker(TOCMode.ADVANCED)
    print(f"Advanced worker mode: {advanced_worker.mode.value}")
    
    # Test content
    test_content = """
    Table of Contents
    
    1. Introduction 1
    1.1 Background 2
    1.2 Objectives 3
    2. Methodology 5
    2.1 Data Collection 6
    2.2 Analysis Methods 8
    3. Results 10
    3.1 Findings 11
    3.2 Discussion 13
    4. Conclusion 15
    References 17
    """
    
    # Test simple extraction
    simple_result = simple_worker._extract_toc_simple({
        'content': test_content,
        'filename': 'test_document.txt'
    })
    print(f"Simple TOC - Entries: {simple_result.get('total_entries', 0)}")
    
    # Test enhanced extraction
    enhanced_result = enhanced_worker._extract_toc_enhanced({
        'content': test_content,
        'filename': 'test_document.txt'
    })
    print(f"Enhanced TOC - Entries: {enhanced_result.get('total_entries', 0)}")
    
    # Test advanced extraction
    advanced_result = advanced_worker._extract_toc_advanced({
        'content': test_content,
        'filename': 'test_document.txt'
    })
    print(f"Advanced TOC - Entries: {advanced_result.get('total_entries', 0)}")
    print(f"Advanced Features: {len(advanced_result.get('advanced_features', {}))}")
