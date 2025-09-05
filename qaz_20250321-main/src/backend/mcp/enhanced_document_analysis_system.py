#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Document Analysis System - MCP Architecture

This system implements the Model Context Protocol (MCP) for comprehensive document analysis:
- Document Analysis Planner: Orchestrates the overall analysis workflow
- Specialized Coordinators: Manage different analysis domains
- Workers: Execute specific analysis tasks
- Context Management: Maintains analysis state and cross-document relationships
"""

import os
import sys
import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from dotenv import load_dotenv
import fitz  # pymupdf
import pandas as pd

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

class AnalysisType(Enum):
    """Types of document analysis"""
    IMAGE_EXTRACTION = "image_extraction"
    TABLE_ANALYSIS = "table_analysis"
    REFERENCE_VALIDATION = "reference_validation"
    CONTRADICTION_DETECTION = "contradiction_detection"
    ARGUMENT_REASONING = "argument_reasoning"
    CROSS_DOCUMENT_CONSISTENCY = "cross_document_consistency"
    CONTENT_CHUNKING = "content_chunking"
    METADATA_EXTRACTION = "metadata_extraction"

@dataclass
class AnalysisContext:
    """MCP Context for document analysis"""
    document_id: str
    document_name: str
    file_path: str
    file_size: int
    processing_timestamp: datetime
    analysis_state: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

@dataclass
class AnalysisRequest:
    """MCP Request for document analysis"""
    analysis_type: AnalysisType
    context: AnalysisContext
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    dependencies: List[str] = field(default_factory=list)

@dataclass
class AnalysisResponse:
    """MCP Response from document analysis"""
    success: bool
    data: Dict[str, Any]
    context_updates: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time: float = 0.0

class DocumentAnalysisPlanner:
    """MCP Planner: Orchestrates the overall document analysis workflow"""
    
    def __init__(self):
        self.coordinators = {}
        self.analysis_queue = []
        self.completed_analyses = {}
        self.global_context = {}
        
    def register_coordinator(self, analysis_type: AnalysisType, coordinator):
        """Register a coordinator for a specific analysis type"""
        self.coordinators[analysis_type] = coordinator
        print(f"📋 Registered coordinator for {analysis_type.value}")
    
    def plan_analysis_workflow(self, document_path: str) -> List[AnalysisRequest]:
        """Plan the analysis workflow for a document"""
        document_name = os.path.splitext(os.path.basename(document_path))[0]
        document_id = hashlib.md5(document_path.encode()).hexdigest()[:8]
        
        # Create analysis context
        context = AnalysisContext(
            document_id=document_id,
            document_name=document_name,
            file_path=document_path,
            file_size=os.path.getsize(document_path),
            processing_timestamp=datetime.now()
        )
        
        # Define analysis workflow with dependencies
        workflow = [
            # Phase 1: Basic extraction (no dependencies)
            AnalysisRequest(
                analysis_type=AnalysisType.METADATA_EXTRACTION,
                context=context,
                priority=1
            ),
            AnalysisRequest(
                analysis_type=AnalysisType.CONTENT_CHUNKING,
                context=context,
                priority=1
            ),
            
            # Phase 2: Content analysis (depends on chunking)
            AnalysisRequest(
                analysis_type=AnalysisType.IMAGE_EXTRACTION,
                context=context,
                dependencies=[AnalysisType.CONTENT_CHUNKING.value],
                priority=2
            ),
            AnalysisRequest(
                analysis_type=AnalysisType.TABLE_ANALYSIS,
                context=context,
                dependencies=[AnalysisType.CONTENT_CHUNKING.value],
                priority=2
            ),
            
            # Phase 3: Advanced analysis (depends on content analysis)
            AnalysisRequest(
                analysis_type=AnalysisType.REFERENCE_VALIDATION,
                context=context,
                dependencies=[AnalysisType.CONTENT_CHUNKING.value],
                priority=3
            ),
            AnalysisRequest(
                analysis_type=AnalysisType.CONTRADICTION_DETECTION,
                context=context,
                dependencies=[AnalysisType.CONTENT_CHUNKING.value],
                priority=3
            ),
            AnalysisRequest(
                analysis_type=AnalysisType.ARGUMENT_REASONING,
                context=context,
                dependencies=[AnalysisType.CONTENT_CHUNKING.value],
                priority=3
            ),
        ]
        
        return workflow
    
    def execute_workflow(self, workflow: List[AnalysisRequest]) -> Dict[str, Any]:
        """Execute the analysis workflow"""
        print(f"🚀 Executing analysis workflow for {workflow[0].context.document_name}")
        
        # Initialize execution state
        completed_analyses = set()
        pending_requests = workflow.copy()
        results = {}
        
        while pending_requests:
            # Find requests that can be executed (dependencies satisfied)
            executable_requests = [
                req for req in pending_requests
                if all(dep in completed_analyses for dep in req.dependencies)
            ]
            
            if not executable_requests:
                print("⚠️ Circular dependency detected or missing dependencies")
                break
            
            # Execute requests in priority order
            executable_requests.sort(key=lambda x: x.priority)
            
            for request in executable_requests:
                try:
                    print(f"  🔄 Executing {request.analysis_type.value}...")
                    
                    # Get coordinator
                    coordinator = self.coordinators.get(request.analysis_type)
                    if not coordinator:
                        print(f"  ❌ No coordinator registered for {request.analysis_type.value}")
                        continue
                    
                    # Execute analysis
                    response = coordinator.execute_analysis(request)
                    
                    if response.success:
                        # Update context
                        request.context.results[request.analysis_type.value] = response.data
                        request.context.analysis_state.update(response.context_updates)
                        
                        # Store results
                        results[request.analysis_type.value] = response.data
                        completed_analyses.add(request.analysis_type.value)
                        
                        print(f"  ✅ Completed {request.analysis_type.value}")
                    else:
                        print(f"  ❌ Failed {request.analysis_type.value}: {response.errors}")
                        request.context.errors.extend(response.errors)
                        request.context.warnings.extend(response.warnings)
                
                except Exception as e:
                    print(f"  ❌ Error in {request.analysis_type.value}: {e}")
                    request.context.errors.append(str(e))
                
                # Remove from pending
                pending_requests.remove(request)
        
        return results

class MetadataExtractionCoordinator:
    """MCP Coordinator: Manages metadata extraction"""
    
    def execute_analysis(self, request: AnalysisRequest) -> AnalysisResponse:
        """Execute metadata extraction analysis"""
        start_time = datetime.now()
        
        try:
            doc = fitz.open(request.context.file_path)
            
            metadata = {
                "document_name": request.context.document_name,
                "file_size": request.context.file_size,
                "page_count": len(doc),
                "creation_date": datetime.now().isoformat(),
                "file_type": "PDF",
                "extraction_timestamp": datetime.now().isoformat()
            }
            
            # Extract basic document properties
            if doc.metadata:
                metadata.update({
                    "title": doc.metadata.get("title", ""),
                    "author": doc.metadata.get("author", ""),
                    "subject": doc.metadata.get("subject", ""),
                    "creator": doc.metadata.get("creator", ""),
                    "producer": doc.metadata.get("producer", ""),
                    "creation_date_pdf": doc.metadata.get("creationDate", ""),
                    "modification_date": doc.metadata.get("modDate", "")
                })
            
            doc.close()
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AnalysisResponse(
                success=True,
                data=metadata,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return AnalysisResponse(
                success=False,
                data={},
                errors=[str(e)],
                execution_time=execution_time
            )

class ContentChunkingCoordinator:
    """MCP Coordinator: Manages content chunking and text extraction"""
    
    def execute_analysis(self, request: AnalysisRequest) -> AnalysisResponse:
        """Execute content chunking analysis"""
        start_time = datetime.now()
        
        try:
            doc = fitz.open(request.context.file_path)
            all_text = ""
            page_texts = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                page_texts.append({
                    "page_number": page_num + 1,
                    "text": text,
                    "word_count": len(text.split()),
                    "char_count": len(text)
                })
                all_text += f"\n--- Page {page_num + 1} ---\n{text}"
            
            doc.close()
            
            # Create chunks
            chunks = self._create_smart_chunks(all_text, request.context.document_name)
            
            chunking_data = {
                "total_pages": len(page_texts),
                "total_text_length": len(all_text),
                "total_words": len(all_text.split()),
                "chunks": chunks,
                "page_breakdown": page_texts
            }
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AnalysisResponse(
                success=True,
                data=chunking_data,
                context_updates={"text_content": all_text, "chunks": chunks},
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return AnalysisResponse(
                success=False,
                data={},
                errors=[str(e)],
                execution_time=execution_time
            )
    
    def _create_smart_chunks(self, text: str, document_name: str) -> List[Dict]:
        """Create smart chunks with enhanced metadata"""
        chunks = []
        sentences = re.split(r'[.!?]+', text)
        
        current_chunk = []
        current_length = 0
        chunk_index = 1
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > 200 and current_chunk:
                # Create chunk
                chunk_text = " ".join(current_chunk)
                chunk_data = {
                    "chunk_id": f"{document_name}_chunk_{chunk_index:03d}",
                    "text": chunk_text,
                    "word_count": len(chunk_text.split()),
                    "char_count": len(chunk_text),
                    "chunk_number": chunk_index,
                    "metadata": {
                        "has_numbers": bool(re.search(r'\d+', chunk_text)),
                        "has_percentages": bool(re.search(r'\d+%', chunk_text)),
                        "has_references": bool(re.search(r'[A-Z][a-z]+,\s*[A-Z]\.\s*\d{4}', chunk_text)),
                        "has_urls": bool(re.search(r'https?://', chunk_text))
                    }
                }
                chunks.append(chunk_data)
                
                current_chunk = []
                current_length = 0
                chunk_index += 1
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk_data = {
                "chunk_id": f"{document_name}_chunk_{chunk_index:03d}",
                "text": chunk_text,
                "word_count": len(chunk_text.split()),
                "char_count": len(chunk_text),
                "chunk_number": chunk_index,
                "metadata": {
                    "has_numbers": bool(re.search(r'\d+', chunk_text)),
                    "has_percentages": bool(re.search(r'\d+%', chunk_text)),
                    "has_references": bool(re.search(r'[A-Z][a-z]+,\s*[A-Z]\.\s*\d{4}', chunk_text)),
                    "has_urls": bool(re.search(r'https?://', chunk_text))
                }
            }
            chunks.append(chunk_data)
        
        return chunks

class ImageExtractionCoordinator:
    """MCP Coordinator: Manages image extraction and analysis"""
    
    def execute_analysis(self, request: AnalysisRequest) -> AnalysisResponse:
        """Execute image extraction analysis"""
        start_time = datetime.now()
        
        try:
            doc = fitz.open(request.context.file_path)
            images = []
            charts = []
            diagrams = []
            icons = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract images
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = {
                                "page": page_num + 1,
                                "index": img_index,
                                "width": pix.width,
                                "height": pix.height,
                                "colorspace": pix.colorspace.name,
                                "size_bytes": len(pix.tobytes()),
                                "aspect_ratio": pix.width / pix.height if pix.height > 0 else 0
                            }
                            
                            # Classify image type
                            img_type = self._classify_image(pix)
                            img_data["type"] = img_type
                            
                            if img_type == "chart":
                                charts.append(img_data)
                            elif img_type == "diagram":
                                diagrams.append(img_data)
                            elif img_type == "icon":
                                icons.append(img_data)
                            else:
                                images.append(img_data)
                        
                        pix = None
                        
                    except Exception as e:
                        print(f"⚠️ Error processing image on page {page_num + 1}: {e}")
            
            doc.close()
            
            image_analysis = {
                "total_images": len(images) + len(charts) + len(diagrams) + len(icons),
                "images": images,
                "charts": charts,
                "diagrams": diagrams,
                "icons": icons,
                "image_density": (len(images) + len(charts) + len(diagrams) + len(icons)) / len(doc) if len(doc) > 0 else 0
            }
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AnalysisResponse(
                success=True,
                data=image_analysis,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return AnalysisResponse(
                success=False,
                data={},
                errors=[str(e)],
                execution_time=execution_time
            )
    
    def _classify_image(self, pix) -> str:
        """Classify image type based on characteristics"""
        aspect_ratio = pix.width / pix.height if pix.height > 0 else 0
        
        if aspect_ratio > 1.5:
            return "chart"  # Wide images are likely charts
        elif pix.width < 200 and pix.height < 200:
            return "icon"  # Small images are likely icons
        elif pix.colorspace.name == "DeviceGray":
            return "diagram"  # Grayscale images are likely diagrams
        else:
            return "image"  # Default classification

class TableAnalysisCoordinator:
    """MCP Coordinator: Manages table extraction and analysis"""
    
    def execute_analysis(self, request: AnalysisRequest) -> AnalysisResponse:
        """Execute table analysis"""
        start_time = datetime.now()
        
        try:
            doc = fitz.open(request.context.file_path)
            tables = []
            data_tables = []
            reference_tables = []
            summary_tables = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                try:
                    # Extract tables using PyMuPDF's table finder
                    page_tables = page.find_tables()
                    
                    for table_index, table in enumerate(page_tables):
                        table_content = table.extract()
                        
                        table_data = {
                            "page": page_num + 1,
                            "index": table_index,
                            "rows": len(table.rows),
                            "columns": len(table.header),
                            "content": table_content,
                            "header": table.header,
                            "total_cells": len(table.rows) * len(table.header)
                        }
                        
                        # Classify table type
                        table_type = self._classify_table(table_content)
                        table_data["type"] = table_type
                        
                        if table_type == "data_table":
                            data_tables.append(table_data)
                        elif table_type == "reference_table":
                            reference_tables.append(table_data)
                        else:
                            summary_tables.append(table_data)
                        
                        tables.append(table_data)
                        
                except Exception as e:
                    print(f"⚠️ Error extracting tables from page {page_num + 1}: {e}")
            
            doc.close()
            
            table_analysis = {
                "total_tables": len(tables),
                "data_tables": data_tables,
                "reference_tables": reference_tables,
                "summary_tables": summary_tables,
                "tables": tables,
                "table_density": len(tables) / len(doc) if len(doc) > 0 else 0
            }
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AnalysisResponse(
                success=True,
                data=table_analysis,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return AnalysisResponse(
                success=False,
                data={},
                errors=[str(e)],
                execution_time=execution_time
            )
    
    def _classify_table(self, table_content: List[List[str]]) -> str:
        """Classify table type based on content"""
        if not table_content:
            return "unknown"
        
        # Count numerical data
        numeric_count = 0
        total_cells = 0
        
        for row in table_content:
            for cell in row:
                total_cells += 1
                if re.search(r'\d+\.?\d*', cell):
                    numeric_count += 1
        
        numeric_ratio = numeric_count / total_cells if total_cells > 0 else 0
        
        if numeric_ratio > 0.5:
            return "data_table"
        elif len(table_content) > 10:
            return "reference_table"
        else:
            return "summary_table"

class ReferenceValidationCoordinator:
    """MCP Coordinator: Manages reference extraction and validation"""
    
    def execute_analysis(self, request: AnalysisRequest) -> AnalysisResponse:
        """Execute reference validation analysis"""
        start_time = datetime.now()
        
        try:
            text_content = request.context.analysis_state.get("text_content", "")
            
            references = self._extract_references(text_content)
            validated_references = self._validate_references(references)
            
            reference_analysis = {
                "total_references": len(references),
                "validated_references": len([r for r in references if r["validated"]]),
                "academic_references": [r for r in references if r["type"] == "academic_reference"],
                "url_references": [r for r in references if r["type"] == "url"],
                "references": references,
                "validation_summary": {
                    "valid": len([r for r in references if r["validated"]]),
                    "invalid": len([r for r in references if not r["validated"]]),
                    "validation_rate": len([r for r in references if r["validated"]]) / len(references) if references else 0
                }
            }
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AnalysisResponse(
                success=True,
                data=reference_analysis,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return AnalysisResponse(
                success=False,
                data={},
                errors=[str(e)],
                execution_time=execution_time
            )
    
    def _extract_references(self, text: str) -> List[Dict[str, Any]]:
        """Extract references from text"""
        references = []
        
        # Academic reference patterns
        academic_patterns = [
            r'([A-Z][a-z]+,\s*[A-Z]\.\s*\d{4})',  # Author, A. YYYY
            r'([A-Z][a-z]+\s+et\s+al\.\s*\d{4})',  # Author et al. YYYY
            r'([A-Z][a-z]+,\s*[A-Z][a-z]+,\s*\d{4})',  # Author, Author, YYYY
            r'([A-Z][a-z]+\s+&\s+[A-Z][a-z]+,\s*\d{4})',  # Author & Author, YYYY
        ]
        
        for pattern in academic_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                ref = {
                    "text": match.group(1),
                    "position": match.start(),
                    "validated": False,
                    "type": "academic_reference",
                    "validation_notes": []
                }
                references.append(ref)
        
        # URL patterns
        url_pattern = r'https?://[^\s]+'
        url_matches = re.finditer(url_pattern, text)
        for match in url_matches:
            ref = {
                "text": match.group(0),
                "position": match.start(),
                "validated": False,
                "type": "url",
                "validation_notes": []
            }
            references.append(ref)
        
        return references
    
    def _validate_references(self, references: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate references (simplified validation)"""
        for ref in references:
            if ref["type"] == "academic_reference":
                # Basic academic reference validation
                if re.match(r'^[A-Z][a-z]+,\s*[A-Z]\.\s*\d{4}$', ref["text"]):
                    ref["validated"] = True
                    ref["validation_notes"].append("Valid academic reference format")
                else:
                    ref["validation_notes"].append("Invalid academic reference format")
            
            elif ref["type"] == "url":
                # Basic URL validation
                if re.match(r'^https?://[^\s]+$', ref["text"]):
                    ref["validated"] = True
                    ref["validation_notes"].append("Valid URL format")
                else:
                    ref["validation_notes"].append("Invalid URL format")
        
        return references

class ContradictionDetectionCoordinator:
    """MCP Coordinator: Manages contradiction detection"""
    
    def execute_analysis(self, request: AnalysisRequest) -> AnalysisResponse:
        """Execute contradiction detection analysis"""
        start_time = datetime.now()
        
        try:
            text_content = request.context.analysis_state.get("text_content", "")
            
            contradictions = self._detect_contradictions(text_content, request.context.document_name)
            
            contradiction_analysis = {
                "total_contradictions": len(contradictions),
                "high_severity": len([c for c in contradictions if c["severity"] == "high"]),
                "medium_severity": len([c for c in contradictions if c["severity"] == "medium"]),
                "low_severity": len([c for c in contradictions if c["severity"] == "low"]),
                "contradictions": contradictions,
                "contradiction_density": len(contradictions) / len(text_content.split()) if text_content else 0
            }
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AnalysisResponse(
                success=True,
                data=contradiction_analysis,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return AnalysisResponse(
                success=False,
                data={},
                errors=[str(e)],
                execution_time=execution_time
            )
    
    def _detect_contradictions(self, text: str, document_name: str) -> List[Dict[str, Any]]:
        """Detect contradictions in text"""
        contradictions = []
        
        # Contradiction indicators
        contradiction_phrases = [
            r'however.*but',
            r'although.*nevertheless',
            r'on the other hand',
            r'in contrast',
            r'contradicts',
            r'inconsistent',
            r'disagrees with',
            r'conflicts with',
            r'nevertheless.*however',
            r'despite.*however'
        ]
        
        for phrase in contradiction_phrases:
            matches = re.finditer(phrase, text, re.IGNORECASE)
            for match in matches:
                # Get surrounding context
                start = max(0, match.start() - 200)
                end = min(len(text), match.end() + 200)
                context = text[start:end]
                
                contradiction = {
                    "document": document_name,
                    "phrase": match.group(0),
                    "position": match.start(),
                    "context": context,
                    "severity": self._assess_contradiction_severity(context)
                }
                contradictions.append(contradiction)
        
        return contradictions
    
    def _assess_contradiction_severity(self, context: str) -> str:
        """Assess contradiction severity"""
        strong_indicators = ['contradicts', 'conflicts', 'inconsistent', 'disagrees', 'false']
        moderate_indicators = ['however', 'although', 'nevertheless', 'despite']
        
        context_lower = context.lower()
        
        if any(indicator in context_lower for indicator in strong_indicators):
            return "high"
        elif any(indicator in context_lower for indicator in moderate_indicators):
            return "medium"
        else:
            return "low"

class ArgumentReasoningCoordinator:
    """MCP Coordinator: Manages argument reasoning analysis"""
    
    def execute_analysis(self, request: AnalysisRequest) -> AnalysisResponse:
        """Execute argument reasoning analysis"""
        start_time = datetime.now()
        
        try:
            text_content = request.context.analysis_state.get("text_content", "")
            
            reasoning_analysis = self._analyze_argument_reasoning(text_content)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AnalysisResponse(
                success=True,
                data=reasoning_analysis,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return AnalysisResponse(
                success=False,
                data={},
                errors=[str(e)],
                execution_time=execution_time
            )
    
    def _analyze_argument_reasoning(self, text: str) -> Dict[str, Any]:
        """Analyze argument reasoning structure"""
        analysis = {
            "claims": [],
            "evidence": [],
            "assumptions": [],
            "logical_fallacies": [],
            "reasoning_quality": "unknown",
            "argument_strength": "unknown"
        }
        
        # Extract claims
        claim_patterns = [
            r'[A-Z][^.!?]*\s+(is|are|was|were|will be|should be|must be)[^.!?]*[.!?]',
            r'[A-Z][^.!?]*\s+(proves|demonstrates|shows|indicates|suggests)[^.!?]*[.!?]',
            r'[A-Z][^.!?]*\s+(therefore|thus|consequently|as a result)[^.!?]*[.!?]'
        ]
        
        for pattern in claim_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                analysis["claims"].append({
                    "text": match.group(0),
                    "type": "claim",
                    "position": match.start()
                })
        
        # Extract evidence
        evidence_patterns = [
            r'\d+%',
            r'\d+\.\d+',
            r'study shows',
            r'research indicates',
            r'data suggests',
            r'example',
            r'case study',
            r'according to',
            r'statistics show'
        ]
        
        for pattern in evidence_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                analysis["evidence"].append({
                    "text": match.group(0),
                    "type": "evidence",
                    "position": match.start()
                })
        
        # Assess reasoning quality
        claim_count = len(analysis["claims"])
        evidence_count = len(analysis["evidence"])
        
        if evidence_count > claim_count * 2:
            analysis["reasoning_quality"] = "strong"
            analysis["argument_strength"] = "high"
        elif evidence_count > claim_count:
            analysis["reasoning_quality"] = "moderate"
            analysis["argument_strength"] = "medium"
        else:
            analysis["reasoning_quality"] = "weak"
            analysis["argument_strength"] = "low"
        
        return analysis

class EnhancedDocumentAnalysisSystem:
    """Main MCP-based document analysis system"""
    
    def __init__(self):
        self.planner = DocumentAnalysisPlanner()
        self._register_coordinators()
    
    def _register_coordinators(self):
        """Register all coordinators with the planner"""
        self.planner.register_coordinator(AnalysisType.METADATA_EXTRACTION, MetadataExtractionCoordinator())
        self.planner.register_coordinator(AnalysisType.CONTENT_CHUNKING, ContentChunkingCoordinator())
        self.planner.register_coordinator(AnalysisType.IMAGE_EXTRACTION, ImageExtractionCoordinator())
        self.planner.register_coordinator(AnalysisType.TABLE_ANALYSIS, TableAnalysisCoordinator())
        self.planner.register_coordinator(AnalysisType.REFERENCE_VALIDATION, ReferenceValidationCoordinator())
        self.planner.register_coordinator(AnalysisType.CONTRADICTION_DETECTION, ContradictionDetectionCoordinator())
        self.planner.register_coordinator(AnalysisType.ARGUMENT_REASONING, ArgumentReasoningCoordinator())
    
    def analyze_document(self, document_path: str) -> Dict[str, Any]:
        """Analyze a single document using MCP architecture"""
        print(f"🔍 MCP Document Analysis: {os.path.basename(document_path)}")
        
        # Plan workflow
        workflow = self.planner.plan_analysis_workflow(document_path)
        
        # Execute workflow
        results = self.planner.execute_workflow(workflow)
        
        return results
    
    def analyze_documents_batch(self, document_paths: List[str]) -> Dict[str, Any]:
        """Analyze multiple documents using MCP architecture"""
        print(f"🚀 MCP Batch Analysis: {len(document_paths)} documents")
        
        all_results = {}
        
        for doc_path in document_paths:
            try:
                results = self.analyze_document(doc_path)
                doc_name = os.path.splitext(os.path.basename(doc_path))[0]
                all_results[doc_name] = results
                print(f"✅ Completed: {doc_name}")
            except Exception as e:
                print(f"❌ Failed to analyze {doc_path}: {e}")
        
        return all_results

def main():
    """Main function to demonstrate MCP-based document analysis"""
    # Initialize the MCP system
    analysis_system = EnhancedDocumentAnalysisSystem()
    
    # Get all PDF files
    reviews_dir = Path("data/input/reviews")
    pdf_files = list(reviews_dir.glob("*.pdf"))
    
    # Skip already processed files
    pdf_files = [f for f in pdf_files if f.name not in [
        "Whitepaper-Model-Validation-Best-Practices-1.pdf", 
        "investment-model-validation.pdf"
    ]]
    
    print(f"🚀 MCP Enhanced Document Analysis - {len(pdf_files)} documents")
    print("=" * 70)
    
    # Analyze documents
    results = analysis_system.analyze_documents_batch([str(f) for f in pdf_files])
    
    # Generate comprehensive report
    report = {
        "mcp_analysis_summary": {
            "total_documents": len(results),
            "analysis_timestamp": datetime.now().isoformat(),
            "system_architecture": "MCP (Model Context Protocol)",
            "coordinators_used": [
                "MetadataExtractionCoordinator",
                "ContentChunkingCoordinator", 
                "ImageExtractionCoordinator",
                "TableAnalysisCoordinator",
                "ReferenceValidationCoordinator",
                "ContradictionDetectionCoordinator",
                "ArgumentReasoningCoordinator"
            ]
        },
        "document_analyses": results,
        "cross_document_insights": {
            "total_images_across_docs": sum(
                r.get("image_extraction", {}).get("total_images", 0) 
                for r in results.values()
            ),
            "total_tables_across_docs": sum(
                r.get("table_analysis", {}).get("total_tables", 0) 
                for r in results.values()
            ),
            "total_references_across_docs": sum(
                r.get("reference_validation", {}).get("total_references", 0) 
                for r in results.values()
            ),
            "total_contradictions_across_docs": sum(
                r.get("contradiction_detection", {}).get("total_contradictions", 0) 
                for r in results.values()
            )
        }
    }
    
    # Save report
    with open("mcp_enhanced_analysis_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("\n🎉 MCP Enhanced Document Analysis completed!")
    print(f"📊 Processed {len(results)} documents")
    print(f"📄 Report saved to: mcp_enhanced_analysis_report.json")
    
    # Print summary statistics
    total_images = report["cross_document_insights"]["total_images_across_docs"]
    total_tables = report["cross_document_insights"]["total_tables_across_docs"]
    total_references = report["cross_document_insights"]["total_references_across_docs"]
    total_contradictions = report["cross_document_insights"]["total_contradictions_across_docs"]
    
    print(f"📊 Total images found: {total_images}")
    print(f"📋 Total tables found: {total_tables}")
    print(f"📚 Total references found: {total_references}")
    print(f"⚠️ Total contradictions found: {total_contradictions}")

if __name__ == "__main__":
    main()
