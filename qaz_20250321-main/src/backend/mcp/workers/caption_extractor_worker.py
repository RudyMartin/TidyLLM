#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Caption Extractor Worker

Specialized worker for extracting and analyzing captions from documents.
This enables systematic caption analysis, validation, and quality assessment
for images, figures, tables, and other visual elements.
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

logger = logging.getLogger(__name__)

@dataclass
class Caption:
    """Represents a single caption found in a document"""
    caption_id: str
    raw_text: str
    caption_type: str  # 'figure', 'table', 'image', 'chart', 'graph', 'photo', 'screenshot', 'diagram'
    caption_number: Optional[str]  # "1.1", "2", "A.3", etc.
    caption_text: str  # The descriptive text
    page_number: Optional[int]
    section: Optional[str]
    context: str  # Surrounding text for context
    confidence_score: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'caption_id': self.caption_id,
            'raw_text': self.raw_text,
            'caption_type': self.caption_type,
            'caption_number': self.caption_number,
            'caption_text': self.caption_text,
            'page_number': self.page_number,
            'section': self.section,
            'context': self.context,
            'confidence_score': self.confidence_score,
            'metadata': self.metadata
        }

@dataclass
class CaptionQualityAssessment:
    """Represents quality assessment for a caption"""
    caption_id: str
    has_number: bool
    has_descriptive_text: bool
    text_length: int
    word_count: int
    quality_score: float
    issues: List[str]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'caption_id': self.caption_id,
            'has_number': self.has_number,
            'has_descriptive_text': self.has_descriptive_text,
            'text_length': self.text_length,
            'word_count': self.word_count,
            'quality_score': self.quality_score,
            'issues': self.issues,
            'recommendations': self.recommendations
        }

@dataclass
class CaptionAnalysisStructure:
    """Represents the complete caption analysis structure"""
    document_id: str
    document_title: str
    total_captions: int
    captions_by_type: Dict[str, int]  # type -> count
    captions_with_numbers: int
    captions_without_numbers: int
    captions: List[Caption]
    quality_assessments: List[CaptionQualityAssessment]
    extraction_method: str
    confidence_score: float
    caption_sections: Dict[str, List[str]]  # section -> list of caption_ids
    metadata: Dict[str, Any]
    created_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'document_id': self.document_id,
            'document_title': self.document_title,
            'total_captions': self.total_captions,
            'captions_by_type': self.captions_by_type,
            'captions_with_numbers': self.captions_with_numbers,
            'captions_without_numbers': self.captions_without_numbers,
            'captions': [caption.to_dict() for caption in self.captions],
            'quality_assessments': [assessment.to_dict() for assessment in self.quality_assessments],
            'extraction_method': self.extraction_method,
            'confidence_score': self.confidence_score,
            'caption_sections': self.caption_sections,
            'metadata': self.metadata,
            'created_at': self.created_at
        }

class CaptionExtractorWorker:
    """Specialized worker for extracting and analyzing captions from documents"""
    
    def __init__(self):
        self.available_methods = self._check_available_methods()
        self.caption_cache = {}
        self.patterns = self._initialize_patterns()
    
    def _check_available_methods(self) -> Dict[str, bool]:
        """Check which caption extraction methods are available"""
        methods = {}
        
        # Check for pdfplumber (for PDF caption extraction)
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
        
        # Check for PIL/Pillow (for image analysis)
        try:
            import PIL
            methods['pil'] = True
        except ImportError:
            methods['pil'] = False
        
        logger.info(f"Available caption extraction methods: {methods}")
        return methods
    
    def _initialize_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize regex patterns for caption detection"""
        patterns = {
            # Figure captions
            'figure_caption': re.compile(
                r'(?:^|\n)\s*(?:Figure|Fig\.?)\s*([A-Za-z0-9\.\-]+)[:\.]?\s*(.+?)(?=\n|$)',
                re.IGNORECASE | re.MULTILINE
            ),
            
            # Table captions
            'table_caption': re.compile(
                r'(?:^|\n)\s*(?:Table|Tab\.?)\s*([A-Za-z0-9\.\-]+)[:\.]?\s*(.+?)(?=\n|$)',
                re.IGNORECASE | re.MULTILINE
            ),
            
            # Image captions
            'image_caption': re.compile(
                r'(?:^|\n)\s*(?:Image|Photo|Picture|Screenshot)\s*([A-Za-z0-9\.\-]*)[:\.]?\s*(.+?)(?=\n|$)',
                re.IGNORECASE | re.MULTILINE
            ),
            
            # Chart/Graph captions
            'chart_caption': re.compile(
                r'(?:^|\n)\s*(?:Chart|Graph|Diagram)\s*([A-Za-z0-9\.\-]*)[:\.]?\s*(.+?)(?=\n|$)',
                re.IGNORECASE | re.MULTILINE
            ),
            
            # Generic numbered captions
            'numbered_caption': re.compile(
                r'(?:^|\n)\s*([A-Za-z]+)\s*([A-Za-z0-9\.\-]+)[:\.]?\s*(.+?)(?=\n|$)',
                re.IGNORECASE | re.MULTILINE
            ),
            
            # Unnumbered captions (just descriptive text)
            'unnumbered_caption': re.compile(
                r'(?:^|\n)\s*(.+?)(?=\n\s*\n|\n\s*[A-Z]|\n\s*Figure|\n\s*Table|\n\s*Image|\n\s*Chart|\n\s*Graph|\n\s*Diagram|\n\s*Photo|\n\s*Screenshot|\n\s*Picture|\n\s*$|\n\s*\d+\.|\n\s*[A-Z]\.)',
                re.IGNORECASE | re.MULTILINE
            ),
            
            # Caption with parentheses
            'parentheses_caption': re.compile(
                r'\(([^)]+)\)\s*[:\.]?\s*(.+?)(?=\n|$)',
                re.IGNORECASE | re.MULTILINE
            )
        }
        
        return patterns
    
    def extract_captions_from_text(self, text: str, document_id: str, page_number: Optional[int] = None, section: Optional[str] = None) -> List[Caption]:
        """Extract captions from plain text"""
        captions = []
        
        # Extract figure captions
        for match in self.patterns['figure_caption'].finditer(text):
            caption_number = match.group(1).strip()
            caption_text = match.group(2).strip()
            
            caption = Caption(
                caption_id=self._generate_caption_id(document_id, 'figure', caption_number, match.start()),
                raw_text=match.group(0).strip(),
                caption_type='figure',
                caption_number=caption_number if caption_number else None,
                caption_text=caption_text,
                page_number=page_number,
                section=section,
                context=self._get_context(text, match.start(), match.end()),
                confidence_score=0.9,
                metadata={'extraction_method': 'regex', 'position': match.start()}
            )
            captions.append(caption)
        
        # Extract table captions
        for match in self.patterns['table_caption'].finditer(text):
            caption_number = match.group(1).strip()
            caption_text = match.group(2).strip()
            
            caption = Caption(
                caption_id=self._generate_caption_id(document_id, 'table', caption_number, match.start()),
                raw_text=match.group(0).strip(),
                caption_type='table',
                caption_number=caption_number if caption_number else None,
                caption_text=caption_text,
                page_number=page_number,
                section=section,
                context=self._get_context(text, match.start(), match.end()),
                confidence_score=0.9,
                metadata={'extraction_method': 'regex', 'position': match.start()}
            )
            captions.append(caption)
        
        # Extract image captions
        for match in self.patterns['image_caption'].finditer(text):
            caption_number = match.group(1).strip()
            caption_text = match.group(2).strip()
            
            # Determine specific image type
            image_type = self._determine_image_type(match.group(0))
            
            caption = Caption(
                caption_id=self._generate_caption_id(document_id, image_type, caption_number, match.start()),
                raw_text=match.group(0).strip(),
                caption_type=image_type,
                caption_number=caption_number if caption_number else None,
                caption_text=caption_text,
                page_number=page_number,
                section=section,
                context=self._get_context(text, match.start(), match.end()),
                confidence_score=0.85,
                metadata={'extraction_method': 'regex', 'position': match.start()}
            )
            captions.append(caption)
        
        # Extract chart/graph captions
        for match in self.patterns['chart_caption'].finditer(text):
            caption_number = match.group(1).strip()
            caption_text = match.group(2).strip()
            
            caption = Caption(
                caption_id=self._generate_caption_id(document_id, 'chart', caption_number, match.start()),
                raw_text=match.group(0).strip(),
                caption_type='chart',
                caption_number=caption_number if caption_number else None,
                caption_text=caption_text,
                page_number=page_number,
                section=section,
                context=self._get_context(text, match.start(), match.end()),
                confidence_score=0.85,
                metadata={'extraction_method': 'regex', 'position': match.start()}
            )
            captions.append(caption)
        
        # Extract generic numbered captions
        for match in self.patterns['numbered_caption'].finditer(text):
            caption_type = match.group(1).strip().lower()
            caption_number = match.group(2).strip()
            caption_text = match.group(3).strip()
            
            # Skip if already captured by specific patterns
            if any(c.raw_text == match.group(0).strip() for c in captions):
                continue
            
            caption = Caption(
                caption_id=self._generate_caption_id(document_id, caption_type, caption_number, match.start()),
                raw_text=match.group(0).strip(),
                caption_type=caption_type,
                caption_number=caption_number if caption_number else None,
                caption_text=caption_text,
                page_number=page_number,
                section=section,
                context=self._get_context(text, match.start(), match.end()),
                confidence_score=0.7,
                metadata={'extraction_method': 'regex', 'position': match.start()}
            )
            captions.append(caption)
        
        # Extract unnumbered captions (potential standalone captions)
        for match in self.patterns['unnumbered_caption'].finditer(text):
            caption_text = match.group(1).strip()
            
            # Filter out very short or very long text that's unlikely to be a caption
            if len(caption_text) < 10 or len(caption_text) > 200:
                continue
            
            # Check if this looks like a caption (descriptive text about visual content)
            if self._is_likely_caption(caption_text):
                caption = Caption(
                    caption_id=self._generate_caption_id(document_id, 'unnumbered', None, match.start()),
                    raw_text=match.group(0).strip(),
                    caption_type='unnumbered',
                    caption_number=None,
                    caption_text=caption_text,
                    page_number=page_number,
                    section=section,
                    context=self._get_context(text, match.start(), match.end()),
                    confidence_score=0.5,  # Lower confidence for unnumbered captions
                    metadata={'extraction_method': 'regex', 'position': match.start()}
                )
                captions.append(caption)
        
        return captions
    
    def extract_captions_from_pdf(self, pdf_path: str, document_id: str) -> List[Caption]:
        """Extract captions from PDF document"""
        captions = []
        
        if self.available_methods.get('pdfplumber'):
            captions.extend(self._extract_captions_with_pdfplumber(pdf_path, document_id))
        elif self.available_methods.get('fitz'):
            captions.extend(self._extract_captions_with_fitz(pdf_path, document_id))
        
        return captions
    
    def _extract_captions_with_pdfplumber(self, pdf_path: str, document_id: str) -> List[Caption]:
        """Extract captions using pdfplumber"""
        captions = []
        
        try:
            import pdfplumber
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract text
                    text = page.extract_text() or ""
                    
                    # Extract captions from text
                    page_captions = self.extract_captions_from_text(
                        text, document_id, page_num, f"page_{page_num}"
                    )
                    captions.extend(page_captions)
                    
                    # Extract captions from annotations (if available)
                    if hasattr(page, 'annots') and page.annots:
                        for annot in page.annots:
                            if annot.get('subtype') == 'FreeText':
                                # This might be a caption annotation
                                caption_text = annot.get('contents', '')
                                if caption_text and self._is_likely_caption(caption_text):
                                    caption = Caption(
                                        caption_id=self._generate_caption_id(document_id, 'annotation', None, 0),
                                        raw_text=caption_text,
                                        caption_type='annotation',
                                        caption_number=None,
                                        caption_text=caption_text,
                                        page_number=page_num,
                                        section=f"page_{page_num}",
                                        context=f"PDF annotation on page {page_num}",
                                        confidence_score=0.8,
                                        metadata={'extraction_method': 'pdf_annotation', 'page': page_num}
                                    )
                                    captions.append(caption)
        
        except Exception as e:
            logger.error(f"Error extracting captions with pdfplumber: {e}")
        
        return captions
    
    def _extract_captions_with_fitz(self, pdf_path: str, document_id: str) -> List[Caption]:
        """Extract captions using PyMuPDF (fitz)"""
        captions = []
        
        try:
            import fitz
            
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text
                text = page.get_text()
                
                # Extract captions from text
                page_captions = self.extract_captions_from_text(
                    text, document_id, page_num + 1, f"page_{page_num + 1}"
                )
                captions.extend(page_captions)
                
                # Extract captions from annotations
                annots = page.get_annotations()
                for annot in annots:
                    if annot.type[0] == 1:  # FreeText annotation
                        caption_text = annot.get_text()
                        if caption_text and self._is_likely_caption(caption_text):
                            caption = Caption(
                                caption_id=self._generate_caption_id(document_id, 'annotation', None, 0),
                                raw_text=caption_text,
                                caption_type='annotation',
                                caption_number=None,
                                caption_text=caption_text,
                                page_number=page_num + 1,
                                section=f"page_{page_num + 1}",
                                context=f"PDF annotation on page {page_num + 1}",
                                confidence_score=0.8,
                                metadata={'extraction_method': 'pdf_annotation', 'page': page_num + 1}
                            )
                            captions.append(caption)
            
            doc.close()
        
        except Exception as e:
            logger.error(f"Error extracting captions with fitz: {e}")
        
        return captions
    
    def analyze_document_captions(self, document_path: str, document_id: str, document_title: str = "") -> CaptionAnalysisStructure:
        """Analyze captions in a document"""
        
        # Check cache
        cache_key = f"{document_id}_{hash(document_path)}"
        if cache_key in self.caption_cache:
            logger.info(f"Using cached caption analysis for {document_id}")
            return self.caption_cache[cache_key]
        
        logger.info(f"Starting caption analysis for document: {document_path}")
        
        # Extract captions based on file type
        if document_path.lower().endswith('.pdf'):
            captions = self.extract_captions_from_pdf(document_path, document_id)
        else:
            # For text files, read and extract captions
            try:
                with open(document_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                captions = self.extract_captions_from_text(text, document_id)
            except Exception as e:
                logger.error(f"Error reading document {document_path}: {e}")
                captions = []
        
        # Analyze captions
        quality_assessments = [self._assess_caption_quality(caption) for caption in captions]
        
        # Count captions by type
        captions_by_type = {}
        for caption in captions:
            caption_type = caption.caption_type
            captions_by_type[caption_type] = captions_by_type.get(caption_type, 0) + 1
        
        # Count captions with/without numbers
        captions_with_numbers = sum(1 for caption in captions if caption.caption_number)
        captions_without_numbers = len(captions) - captions_with_numbers
        
        # Group captions by section
        caption_sections = {}
        for caption in captions:
            section = caption.section or "unknown"
            if section not in caption_sections:
                caption_sections[section] = []
            caption_sections[section].append(caption.caption_id)
        
        # Create analysis structure
        analysis = CaptionAnalysisStructure(
            document_id=document_id,
            document_title=document_title,
            total_captions=len(captions),
            captions_by_type=captions_by_type,
            captions_with_numbers=captions_with_numbers,
            captions_without_numbers=captions_without_numbers,
            captions=captions,
            quality_assessments=quality_assessments,
            extraction_method="caption_extractor_worker",
            confidence_score=self._calculate_confidence_score(captions, quality_assessments),
            caption_sections=caption_sections,
            metadata={
                'file_path': document_path,
                'extraction_timestamp': datetime.now().isoformat()
            },
            created_at=datetime.now().isoformat()
        )
        
        # Cache the result
        self.caption_cache[cache_key] = analysis
        
        logger.info(f"Caption analysis completed for {document_id}: {len(captions)} captions found")
        
        return analysis
    
    def _generate_caption_id(self, document_id: str, caption_type: str, caption_number: Optional[str], position: int) -> str:
        """Generate a unique caption ID"""
        content = f"{document_id}_{caption_type}_{caption_number}_{position}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _get_context(self, text: str, start: int, end: int) -> str:
        """Get context around a caption"""
        context_start = max(0, start - 100)
        context_end = min(len(text), end + 100)
        return text[context_start:context_end].strip()
    
    def _determine_image_type(self, caption_text: str) -> str:
        """Determine the specific type of image caption"""
        caption_lower = caption_text.lower()
        
        if 'photo' in caption_lower:
            return 'photo'
        elif 'screenshot' in caption_lower:
            return 'screenshot'
        elif 'picture' in caption_lower:
            return 'picture'
        elif 'image' in caption_lower:
            return 'image'
        else:
            return 'image'  # Default
    
    def _is_likely_caption(self, text: str) -> bool:
        """Determine if text is likely to be a caption"""
        text_lower = text.lower()
        
        # Caption indicators
        caption_indicators = [
            'shows', 'displays', 'illustrates', 'depicts', 'represents',
            'demonstrates', 'presents', 'contains', 'includes', 'features',
            'view of', 'image of', 'photo of', 'screenshot of', 'diagram of',
            'chart showing', 'graph of', 'table showing', 'figure showing'
        ]
        
        # Check for caption indicators
        for indicator in caption_indicators:
            if indicator in text_lower:
                return True
        
        # Check for visual content references
        visual_indicators = [
            'data', 'results', 'analysis', 'comparison', 'trends',
            'performance', 'statistics', 'metrics', 'values', 'numbers'
        ]
        
        visual_count = sum(1 for indicator in visual_indicators if indicator in text_lower)
        if visual_count >= 2:
            return True
        
        # Check for descriptive language (not just a title)
        if len(text.split()) > 3 and len(text.split()) < 20:
            return True
        
        return False
    
    def _assess_caption_quality(self, caption: Caption) -> CaptionQualityAssessment:
        """Assess the quality of a caption"""
        issues = []
        recommendations = []
        
        # Check for caption number
        has_number = caption.caption_number is not None
        if not has_number:
            issues.append("Missing caption number")
            recommendations.append("Add a number to the caption for better organization")
        
        # Check for descriptive text
        has_descriptive_text = len(caption.caption_text.strip()) > 0
        if not has_descriptive_text:
            issues.append("Missing descriptive text")
            recommendations.append("Add descriptive text to explain the visual content")
        
        # Check text length
        text_length = len(caption.caption_text)
        word_count = len(caption.caption_text.split())
        
        if text_length < 10:
            issues.append("Caption text too short")
            recommendations.append("Provide more detailed description")
        elif text_length > 200:
            issues.append("Caption text too long")
            recommendations.append("Keep caption concise and focused")
        
        if word_count < 3:
            issues.append("Insufficient words in caption")
            recommendations.append("Use more descriptive language")
        
        # Calculate quality score
        quality_score = 0.0
        
        if has_number:
            quality_score += 0.3
        if has_descriptive_text:
            quality_score += 0.3
        if 10 <= text_length <= 200:
            quality_score += 0.2
        if word_count >= 3:
            quality_score += 0.2
        
        return CaptionQualityAssessment(
            caption_id=caption.caption_id,
            has_number=has_number,
            has_descriptive_text=has_descriptive_text,
            text_length=text_length,
            word_count=word_count,
            quality_score=quality_score,
            issues=issues,
            recommendations=recommendations
        )
    
    def _calculate_confidence_score(self, captions: List[Caption], quality_assessments: List[CaptionQualityAssessment]) -> float:
        """Calculate overall confidence score for the analysis"""
        if not captions:
            return 0.0
        
        # Base confidence on extraction quality
        avg_extraction_confidence = sum(caption.confidence_score for caption in captions) / len(captions)
        
        # Adjust based on quality assessments
        if quality_assessments:
            avg_quality_score = sum(assessment.quality_score for assessment in quality_assessments) / len(quality_assessments)
            return (avg_extraction_confidence + avg_quality_score) / 2
        else:
            return avg_extraction_confidence
    
    def get_caption_quality_report(self, analysis: CaptionAnalysisStructure) -> Dict[str, Any]:
        """Generate a report of caption quality issues"""
        quality_issues = [
            assessment for assessment in analysis.quality_assessments 
            if assessment.issues
        ]
        
        return {
            'document_id': analysis.document_id,
            'document_title': analysis.document_title,
            'total_captions': analysis.total_captions,
            'captions_with_issues': len(quality_issues),
            'captions_with_issues_percentage': (len(quality_issues) / analysis.total_captions * 100) if analysis.total_captions > 0 else 0,
            'captions_without_numbers': analysis.captions_without_numbers,
            'captions_without_numbers_percentage': (analysis.captions_without_numbers / analysis.total_captions * 100) if analysis.total_captions > 0 else 0,
            'quality_issues': [
                {
                    'caption_id': assessment.caption_id,
                    'issues': assessment.issues,
                    'recommendations': assessment.recommendations,
                    'quality_score': assessment.quality_score
                }
                for assessment in quality_issues
            ],
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def save_analysis_to_json(self, analysis: CaptionAnalysisStructure, output_path: str) -> bool:
        """Save caption analysis to JSON file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Caption analysis saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving caption analysis to {output_path}: {e}")
            return False
    
    def load_analysis_from_json(self, json_path: str) -> Optional[CaptionAnalysisStructure]:
        """Load caption analysis from JSON file"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Reconstruct the analysis structure
            captions = [Caption(**caption_data) for caption_data in data.get('captions', [])]
            quality_assessments = [CaptionQualityAssessment(**assessment_data) for assessment_data in data.get('quality_assessments', [])]
            
            analysis = CaptionAnalysisStructure(
                document_id=data['document_id'],
                document_title=data['document_title'],
                total_captions=data['total_captions'],
                captions_by_type=data['captions_by_type'],
                captions_with_numbers=data['captions_with_numbers'],
                captions_without_numbers=data['captions_without_numbers'],
                captions=captions,
                quality_assessments=quality_assessments,
                extraction_method=data['extraction_method'],
                confidence_score=data['confidence_score'],
                caption_sections=data['caption_sections'],
                metadata=data['metadata'],
                created_at=data['created_at']
            )
            
            return analysis
        except Exception as e:
            logger.error(f"Error loading caption analysis from {json_path}: {e}")
            return None
