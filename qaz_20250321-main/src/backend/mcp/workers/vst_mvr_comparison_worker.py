#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VST vs MVR Comparison Worker

Advanced worker for comparing Validation Scope Templates (VST) against 
Model Validation Reports (MVR) using embedding-based semantic analysis.
"""

import logging
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import uuid

# Import MCP components
from .base_worker import BaseWorker
from ..protocol.message_protocol import MCPMessage, TaskType, MessageType, Priority
from ...core.embedding_helper import EmbeddingHelper, EmbeddingMetadata

logger = logging.getLogger(__name__)


class VSTMVRComparisonWorker(BaseWorker):
    """Worker for VST vs MVR comparison analysis using embeddings"""
    
    def __init__(self, target_embedding_dimensions: int = 1024):
        super().__init__(worker_name="VSTMVRComparisonWorker", worker_type="vst_mvr_comparison")
        self.embedding_helper = EmbeddingHelper(target_dimensions=target_embedding_dimensions)
        self.comparison_cache = {}
        
        logger.info(f"VST vs MVR Comparison Worker initialized with {target_embedding_dimensions}D embeddings")
    
    def process_task(self, message: MCPMessage) -> Dict[str, Any]:
        """Process task based on MCP message (required abstract method implementation)"""
        return self.process_message(message)
    
    def process_message(self, message: MCPMessage) -> Dict[str, Any]:
        """Process comparison message"""
        try:
            task_type = message.task_type
            payload = message.payload
            
            if task_type == TaskType.ANALYSIS:
                return self.perform_comparison_analysis(payload)
            elif task_type == TaskType.EMBEDDING_GENERATION:
                return self.generate_document_embeddings(payload)
            elif task_type == TaskType.VALIDATION:
                return self.validate_comparison_results(payload)
            else:
                raise ValueError(f"Unsupported task type: {task_type}")
                
        except Exception as e:
            logger.error(f"Error processing VST vs MVR comparison: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'worker_type': self.worker_type,
                'processing_time_ms': 0
            }
    
    def perform_comparison_analysis(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive VST vs MVR comparison analysis"""
        start_time = datetime.now()
        
        try:
            vst_content = payload.get('vst_content', '')
            mvr_content = payload.get('mvr_content', '')
            comparison_type = payload.get('comparison_type', 'semantic_similarity')
            
            if not vst_content or not mvr_content:
                raise ValueError("Both VST and MVR content required for comparison")
            
            # Extract sections from both documents
            vst_sections = self.extract_document_sections(vst_content, document_type='VST')
            mvr_sections = self.extract_document_sections(mvr_content, document_type='MVR')
            
            # Generate embeddings for all sections
            vst_embeddings = self.generate_section_embeddings(vst_sections, 'VST')
            mvr_embeddings = self.generate_section_embeddings(mvr_sections, 'MVR')
            
            # Perform similarity analysis
            similarity_matrix = self.calculate_similarity_matrix(vst_embeddings, mvr_embeddings)
            
            # Identify gaps and coverage
            gap_analysis = self.perform_gap_analysis(vst_sections, mvr_sections, similarity_matrix)
            
            # Generate comparison report
            comparison_report = self.generate_comparison_report(
                vst_sections, mvr_sections, similarity_matrix, gap_analysis
            )
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                'status': 'success',
                'worker_type': self.worker_type,
                'comparison_type': comparison_type,
                'vst_sections_count': len(vst_sections),
                'mvr_sections_count': len(mvr_sections),
                'similarity_matrix': similarity_matrix.tolist() if hasattr(similarity_matrix, 'tolist') else similarity_matrix,
                'gap_analysis': gap_analysis,
                'comparison_report': comparison_report,
                'processing_time_ms': processing_time,
                'embeddings_metadata': {
                    'vst_embeddings_count': len(vst_embeddings),
                    'mvr_embeddings_count': len(mvr_embeddings),
                    'embedding_model': self.embedding_helper.get_model_info()
                }
            }
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Error in comparison analysis: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'worker_type': self.worker_type,
                'processing_time_ms': processing_time
            }
    
    def extract_document_sections(self, content: str, document_type: str = 'Unknown') -> List[Dict[str, Any]]:
        """Extract sections from document content"""
        sections = []
        
        # Split content by headers (markdown or numbered sections)
        lines = content.split('\n')
        current_section = None
        section_content = []
        
        for i, line in enumerate(lines):
            # Match various header patterns
            header_match = re.match(r'^(#{1,6})\s*(\d*\.?\d*\.?\d*\.?\d*)\s*(.+)$', line.strip())
            if not header_match:
                # Try numbered sections without markdown
                header_match = re.match(r'^(\d+(?:\.\d+)*)\s+(.+)$', line.strip())
            
            if header_match:
                # Save previous section
                if current_section is not None:
                    current_section['content'] = '\n'.join(section_content).strip()
                    current_section['word_count'] = len(current_section['content'].split())
                    sections.append(current_section)
                
                # Start new section
                if len(header_match.groups()) == 3:
                    level_indicator, section_number, title = header_match.groups()
                    level = len(level_indicator) if level_indicator.startswith('#') else 1
                else:
                    section_number, title = header_match.groups()
                    level = len(section_number.split('.'))
                
                current_section = {
                    'section_id': section_number or f"section_{len(sections)+1}",
                    'title': title.strip(),
                    'level': level,
                    'line_number': i + 1,
                    'document_type': document_type
                }
                section_content = []
            else:
                if current_section is not None:
                    section_content.append(line)
        
        # Save final section
        if current_section is not None:
            current_section['content'] = '\n'.join(section_content).strip()
            current_section['word_count'] = len(current_section['content'].split())
            sections.append(current_section)
        
        logger.info(f"Extracted {len(sections)} sections from {document_type} document")
        return sections
    
    def generate_section_embeddings(self, sections: List[Dict[str, Any]], doc_type: str) -> List[Dict[str, Any]]:
        """Generate embeddings for document sections"""
        embeddings = []
        
        for section in sections:
            section_text = f"{section['title']}\n{section['content']}"
            content_id = f"{doc_type}_{section['section_id']}"
            
            try:
                embedding_vector, metadata = self.embedding_helper.generate_embedding(
                    text=section_text,
                    content_id=content_id
                )
                
                embeddings.append({
                    'section_id': section['section_id'],
                    'title': section['title'],
                    'document_type': doc_type,
                    'embedding': embedding_vector,
                    'metadata': {
                        'content_hash': metadata.content_hash,
                        'model_name': metadata.model_name,
                        'dimensions': metadata.target_dimensions,
                        'padding_method': metadata.padding_method
                    },
                    'text_length': len(section_text),
                    'word_count': section['word_count']
                })
                
            except Exception as e:
                logger.error(f"Error generating embedding for section {section['section_id']}: {e}")
                continue
        
        logger.info(f"Generated {len(embeddings)} embeddings for {doc_type} sections")
        return embeddings
    
    def calculate_similarity_matrix(self, vst_embeddings: List[Dict[str, Any]], 
                                  mvr_embeddings: List[Dict[str, Any]]) -> List[List[float]]:
        """Calculate cosine similarity matrix between VST and MVR sections"""
        try:
            # Import numpy substitute
            from ...core.datamart_numpy_substitution import np
            
            similarity_matrix = []
            
            for vst_embed in vst_embeddings:
                vst_vector = vst_embed['embedding']
                row_similarities = []
                
                for mvr_embed in mvr_embeddings:
                    mvr_vector = mvr_embed['embedding']
                    
                    # Calculate cosine similarity
                    similarity = self.cosine_similarity(vst_vector, mvr_vector)
                    row_similarities.append(float(similarity))
                
                similarity_matrix.append(row_similarities)
            
            logger.info(f"Calculated similarity matrix: {len(similarity_matrix)}x{len(similarity_matrix[0]) if similarity_matrix else 0}")
            return similarity_matrix
            
        except Exception as e:
            logger.error(f"Error calculating similarity matrix: {e}")
            return []
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            # Import numpy substitute
            from ...core.datamart_numpy_substitution import np
            
            # Convert to numpy arrays
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            
            # Calculate cosine similarity
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            if norm_v1 == 0 or norm_v2 == 0:
                return 0.0
            
            return dot_product / (norm_v1 * norm_v2)
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def perform_gap_analysis(self, vst_sections: List[Dict[str, Any]], 
                           mvr_sections: List[Dict[str, Any]], 
                           similarity_matrix: List[List[float]]) -> Dict[str, Any]:
        """Perform gap analysis between VST requirements and MVR coverage"""
        
        gap_threshold = 0.7  # Sections below this similarity are considered gaps
        
        gaps = []
        covered_requirements = []
        partial_coverage = []
        
        for i, vst_section in enumerate(vst_sections):
            if i >= len(similarity_matrix):
                continue
                
            # Find best matching MVR section
            max_similarity = max(similarity_matrix[i]) if similarity_matrix[i] else 0.0
            best_match_idx = similarity_matrix[i].index(max_similarity) if similarity_matrix[i] else -1
            
            if max_similarity >= gap_threshold:
                covered_requirements.append({
                    'vst_section_id': vst_section['section_id'],
                    'vst_title': vst_section['title'],
                    'mvr_section_id': mvr_sections[best_match_idx]['section_id'] if best_match_idx >= 0 else 'unknown',
                    'mvr_title': mvr_sections[best_match_idx]['title'] if best_match_idx >= 0 and best_match_idx < len(mvr_sections) else 'unknown',
                    'similarity_score': max_similarity,
                    'coverage_level': 'full'
                })
            elif max_similarity >= 0.5:
                partial_coverage.append({
                    'vst_section_id': vst_section['section_id'],
                    'vst_title': vst_section['title'],
                    'mvr_section_id': mvr_sections[best_match_idx]['section_id'] if best_match_idx >= 0 and best_match_idx < len(mvr_sections) else 'unknown',
                    'mvr_title': mvr_sections[best_match_idx]['title'] if best_match_idx >= 0 and best_match_idx < len(mvr_sections) else 'unknown',
                    'similarity_score': max_similarity,
                    'coverage_level': 'partial'
                })
            else:
                gaps.append({
                    'vst_section_id': vst_section['section_id'],
                    'vst_title': vst_section['title'],
                    'similarity_score': max_similarity,
                    'coverage_level': 'missing',
                    'priority': self.assess_gap_priority(vst_section)
                })
        
        return {
            'total_vst_sections': len(vst_sections),
            'covered_requirements': covered_requirements,
            'partial_coverage': partial_coverage,
            'gaps': gaps,
            'coverage_statistics': {
                'full_coverage_count': len(covered_requirements),
                'partial_coverage_count': len(partial_coverage),
                'gap_count': len(gaps),
                'coverage_percentage': (len(covered_requirements) + 0.5 * len(partial_coverage)) / len(vst_sections) * 100 if vst_sections else 0
            }
        }
    
    def assess_gap_priority(self, vst_section: Dict[str, Any]) -> str:
        """Assess priority of a gap based on VST section characteristics"""
        title = vst_section['title'].lower()
        content = vst_section.get('content', '').lower()
        
        # High priority keywords
        high_priority_keywords = [
            'compliance', 'regulatory', 'governance', 'risk', 'control',
            'validation', 'testing', 'monitoring', 'audit', 'oversight'
        ]
        
        # Medium priority keywords  
        medium_priority_keywords = [
            'documentation', 'reporting', 'methodology', 'framework',
            'process', 'procedure', 'standard', 'guideline'
        ]
        
        # Check for high priority
        for keyword in high_priority_keywords:
            if keyword in title or keyword in content:
                return 'high'
        
        # Check for medium priority
        for keyword in medium_priority_keywords:
            if keyword in title or keyword in content:
                return 'medium'
        
        return 'low'
    
    def generate_comparison_report(self, vst_sections: List[Dict[str, Any]], 
                                 mvr_sections: List[Dict[str, Any]], 
                                 similarity_matrix: List[List[float]], 
                                 gap_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive comparison report"""
        
        return {
            'report_id': str(uuid.uuid4()),
            'generated_at': datetime.now().isoformat(),
            'comparison_summary': {
                'vst_sections_analyzed': len(vst_sections),
                'mvr_sections_analyzed': len(mvr_sections),
                'average_similarity': self.calculate_average_similarity(similarity_matrix),
                'coverage_percentage': gap_analysis['coverage_statistics']['coverage_percentage']
            },
            'similarity_analysis': {
                'highest_similarity_pairs': self.find_highest_similarity_pairs(
                    vst_sections, mvr_sections, similarity_matrix, top_n=5
                ),
                'lowest_similarity_pairs': self.find_lowest_similarity_pairs(
                    vst_sections, mvr_sections, similarity_matrix, top_n=5
                )
            },
            'gap_analysis_summary': {
                'critical_gaps': [gap for gap in gap_analysis['gaps'] if gap['priority'] == 'high'],
                'total_gaps': len(gap_analysis['gaps']),
                'coverage_score': gap_analysis['coverage_statistics']['coverage_percentage']
            },
            'recommendations': self.generate_recommendations(gap_analysis),
            'next_steps': [
                "Review identified gaps and prioritize remediation",
                "Enhance MVR documentation for partially covered areas",
                "Consider additional validation testing for high-priority gaps",
                "Schedule follow-up comparison analysis after improvements"
            ]
        }
    
    def calculate_average_similarity(self, similarity_matrix: List[List[float]]) -> float:
        """Calculate average similarity across all section pairs"""
        if not similarity_matrix:
            return 0.0
        
        total_similarity = 0.0
        total_pairs = 0
        
        for row in similarity_matrix:
            for similarity in row:
                total_similarity += similarity
                total_pairs += 1
        
        return total_similarity / total_pairs if total_pairs > 0 else 0.0
    
    def find_highest_similarity_pairs(self, vst_sections: List[Dict[str, Any]], 
                                    mvr_sections: List[Dict[str, Any]], 
                                    similarity_matrix: List[List[float]], 
                                    top_n: int = 5) -> List[Dict[str, Any]]:
        """Find pairs with highest similarity scores"""
        pairs = []
        
        for i, vst_section in enumerate(vst_sections):
            if i >= len(similarity_matrix):
                continue
            for j, mvr_section in enumerate(mvr_sections):
                if j >= len(similarity_matrix[i]):
                    continue
                    
                pairs.append({
                    'vst_section': {
                        'id': vst_section['section_id'],
                        'title': vst_section['title']
                    },
                    'mvr_section': {
                        'id': mvr_section['section_id'],
                        'title': mvr_section['title']
                    },
                    'similarity_score': similarity_matrix[i][j]
                })
        
        # Sort by similarity and return top N
        pairs.sort(key=lambda x: x['similarity_score'], reverse=True)
        return pairs[:top_n]
    
    def find_lowest_similarity_pairs(self, vst_sections: List[Dict[str, Any]], 
                                   mvr_sections: List[Dict[str, Any]], 
                                   similarity_matrix: List[List[float]], 
                                   top_n: int = 5) -> List[Dict[str, Any]]:
        """Find pairs with lowest similarity scores"""
        pairs = []
        
        for i, vst_section in enumerate(vst_sections):
            if i >= len(similarity_matrix):
                continue
            for j, mvr_section in enumerate(mvr_sections):
                if j >= len(similarity_matrix[i]):
                    continue
                    
                pairs.append({
                    'vst_section': {
                        'id': vst_section['section_id'],
                        'title': vst_section['title']
                    },
                    'mvr_section': {
                        'id': mvr_section['section_id'],
                        'title': mvr_section['title']
                    },
                    'similarity_score': similarity_matrix[i][j]
                })
        
        # Sort by similarity and return bottom N
        pairs.sort(key=lambda x: x['similarity_score'])
        return pairs[:top_n]
    
    def generate_recommendations(self, gap_analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on gap analysis"""
        recommendations = []
        
        gaps = gap_analysis['gaps']
        coverage_percentage = gap_analysis['coverage_statistics']['coverage_percentage']
        
        # Coverage-based recommendations
        if coverage_percentage < 50:
            recommendations.append("Major gaps identified. Consider comprehensive MVR revision.")
        elif coverage_percentage < 75:
            recommendations.append("Moderate gaps present. Focus on addressing high-priority missing sections.")
        else:
            recommendations.append("Good coverage overall. Address remaining gaps to achieve full compliance.")
        
        # Priority-based recommendations
        high_priority_gaps = [gap for gap in gaps if gap['priority'] == 'high']
        if high_priority_gaps:
            recommendations.append(f"Immediately address {len(high_priority_gaps)} high-priority gaps.")
        
        # Specific section recommendations
        if len(gaps) > 0:
            gap_titles = [gap['vst_title'] for gap in gaps[:3]]  # Top 3 gaps
            recommendations.append(f"Focus on missing sections: {', '.join(gap_titles)}")
        
        return recommendations
    
    def generate_document_embeddings(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate embeddings for a single document"""
        try:
            content = payload.get('content', '')
            document_type = payload.get('document_type', 'Unknown')
            
            if not content:
                raise ValueError("Document content is required")
            
            sections = self.extract_document_sections(content, document_type)
            embeddings = self.generate_section_embeddings(sections, document_type)
            
            return {
                'status': 'success',
                'worker_type': self.worker_type,
                'document_type': document_type,
                'sections_count': len(sections),
                'embeddings_count': len(embeddings),
                'sections': sections,
                'embeddings': embeddings
            }
            
        except Exception as e:
            logger.error(f"Error generating document embeddings: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'worker_type': self.worker_type
            }
    
    def validate_comparison_results(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Validate comparison results for quality and consistency"""
        try:
            results = payload.get('comparison_results', {})
            
            validation_checks = {
                'similarity_matrix_valid': self.validate_similarity_matrix(results.get('similarity_matrix', [])),
                'gap_analysis_consistent': self.validate_gap_analysis(results.get('gap_analysis', {})),
                'coverage_calculation_correct': self.validate_coverage_calculation(results.get('gap_analysis', {}))
            }
            
            all_valid = all(validation_checks.values())
            
            return {
                'status': 'success',
                'worker_type': self.worker_type,
                'validation_results': {
                    'overall_valid': all_valid,
                    'checks': validation_checks,
                    'quality_score': sum(validation_checks.values()) / len(validation_checks)
                }
            }
            
        except Exception as e:
            logger.error(f"Error validating comparison results: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'worker_type': self.worker_type
            }
    
    def validate_similarity_matrix(self, similarity_matrix: List[List[float]]) -> bool:
        """Validate similarity matrix structure and values"""
        try:
            if not similarity_matrix:
                return False
            
            # Check if all values are between 0 and 1
            for row in similarity_matrix:
                for value in row:
                    if not (0 <= value <= 1):
                        return False
            
            # Check if matrix is rectangular
            first_row_length = len(similarity_matrix[0]) if similarity_matrix else 0
            for row in similarity_matrix:
                if len(row) != first_row_length:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def validate_gap_analysis(self, gap_analysis: Dict[str, Any]) -> bool:
        """Validate gap analysis consistency"""
        try:
            stats = gap_analysis.get('coverage_statistics', {})
            
            total_sections = gap_analysis.get('total_vst_sections', 0)
            full_coverage = stats.get('full_coverage_count', 0)
            partial_coverage = stats.get('partial_coverage_count', 0) 
            gaps = stats.get('gap_count', 0)
            
            # Check if counts add up
            if full_coverage + partial_coverage + gaps != total_sections:
                return False
            
            # Check coverage percentage calculation
            expected_percentage = (full_coverage + 0.5 * partial_coverage) / total_sections * 100 if total_sections > 0 else 0
            actual_percentage = stats.get('coverage_percentage', 0)
            
            # Allow small floating point differences
            if abs(expected_percentage - actual_percentage) > 0.1:
                return False
            
            return True
            
        except Exception:
            return False
    
    def validate_coverage_calculation(self, gap_analysis: Dict[str, Any]) -> bool:
        """Validate coverage percentage calculation"""
        try:
            stats = gap_analysis.get('coverage_statistics', {})
            coverage_percentage = stats.get('coverage_percentage', 0)
            
            # Coverage should be between 0 and 100
            return 0 <= coverage_percentage <= 100
            
        except Exception:
            return False
    
    def get_worker_status(self) -> Dict[str, Any]:
        """Get worker status and capabilities"""
        return {
            'worker_type': self.worker_type,
            'status': 'ready',
            'capabilities': [
                'vst_mvr_comparison',
                'embedding_generation', 
                'similarity_calculation',
                'gap_analysis',
                'comparison_reporting',
                'knowledge_base_matching'  # New capability
            ],
            'embedding_model_info': self.embedding_helper.get_model_info(),
            'cache_size': len(self.comparison_cache)
        }
    
    def match_against_knowledge_base(self, uploaded_content: str, filename: str) -> Dict[str, Any]:
        """Match uploaded document against knowledge base papers using existing infrastructure"""
        try:
            start_time = datetime.now()
            
            # Use existing embedding infrastructure
            uploaded_embedding, metadata = self.embedding_helper.generate_embedding(uploaded_content)
            
            # Load knowledge base papers (reuse existing search logic)
            kb_papers = self._load_knowledge_base_papers()
            
            # Use existing similarity calculation methods
            matches = []
            for kb_paper in kb_papers:
                # Generate embedding for knowledge base paper if not cached
                if 'embedding' not in kb_paper:
                    kb_content = self._extract_paper_content(kb_paper['path'])
                    kb_embedding, _ = self.embedding_helper.generate_embedding(kb_content)
                    kb_paper['embedding'] = kb_embedding
                
                # Use existing similarity calculation
                similarity = self._calculate_cosine_similarity(uploaded_embedding, kb_paper['embedding'])
                
                if similarity > 0.2:  # Threshold for relevance
                    matches.append({
                        'paper': kb_paper,
                        'similarity_score': similarity,
                        'match_type': self._determine_match_type(similarity)
                    })
            
            # Sort by similarity score
            matches.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                'status': 'success',
                'worker_type': self.worker_type,
                'uploaded_document': {
                    'filename': filename,
                    'content_length': len(uploaded_content),
                    'embedding_metadata': metadata
                },
                'matches': matches,
                'recommendations': self._generate_kb_recommendations(matches),
                'summary': self._generate_kb_summary(matches),
                'processing_time_ms': processing_time
            }
            
        except Exception as e:
            logger.error(f"Error in knowledge base matching: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'worker_type': self.worker_type,
                'processing_time_ms': 0
            }
    
    def _load_knowledge_base_papers(self) -> List[Dict[str, Any]]:
        """Load knowledge base papers using existing search logic"""
        papers = []
        knowledge_base_dir = Path("knowledge_base")
        
        if not knowledge_base_dir.exists():
            logger.warning("Knowledge base directory not found")
            return papers
        
        # Reuse existing search_knowledge_base.py logic
        for pdf_path in knowledge_base_dir.rglob("*.pdf"):
            try:
                papers.append({
                    'id': pdf_path.stem,
                    'title': pdf_path.stem.replace('_', ' ').title(),
                    'path': str(pdf_path),
                    'category': self._get_paper_category(pdf_path),
                    'size_mb': pdf_path.stat().st_size / (1024 * 1024),
                    'filename': pdf_path.name
                })
            except Exception as e:
                logger.warning(f"Error loading paper {pdf_path}: {e}")
        
        return papers
    
    def _get_paper_category(self, pdf_path: Path) -> str:
        """Get paper category from path"""
        parts = pdf_path.parts
        if len(parts) >= 3:
            return f"{parts[-3]}/{parts[-2]}"
        elif len(parts) >= 2:
            return parts[-2]
        return "unknown"
    
    def _extract_paper_content(self, pdf_path: str) -> str:
        """Extract content from PDF paper"""
        try:
            # Simple text extraction - in production, use proper PDF extraction
            return f"Content from {Path(pdf_path).name}"
        except Exception as e:
            logger.warning(f"Error extracting content from {pdf_path}: {e}")
            return ""
    
    def _calculate_cosine_similarity(self, embedding1, embedding2) -> float:
        """Calculate cosine similarity between embeddings"""
        try:
            # Use numpy for calculation (already imported via embedding_helper)
            from .base_worker import np
            
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        except Exception as e:
            logger.warning(f"Error calculating similarity: {e}")
            return 0.0
    
    def _determine_match_type(self, similarity: float) -> str:
        """Determine the type of match based on similarity score"""
        if similarity > 0.9:
            return 'exact_match'
        elif similarity > 0.7:
            return 'high_similarity'
        elif similarity > 0.5:
            return 'semantic_match'
        else:
            return 'related_work'
    
    def _generate_kb_recommendations(self, matches: List[Dict[str, Any]]) -> List[str]:
        """Generate intelligent recommendations based on matches"""
        recommendations = []
        
        if not matches:
            recommendations.append("🔍 No similar papers found. Consider expanding your search to related topics.")
            return recommendations
        
        # High similarity matches
        high_matches = [m for m in matches if m['similarity_score'] > 0.8]
        if high_matches:
            recommendations.append(f"🎯 Found {len(high_matches)} highly similar papers. These are likely very relevant to your research.")
        
        # Research gaps
        if len(matches) < 3:
            recommendations.append("📚 Limited similar papers found. This might indicate a research gap or novel approach.")
        
        # Related work suggestions
        if matches:
            top_match = matches[0]
            recommendations.append(f"📖 Consider reading '{top_match['paper']['title']}' for foundational context.")
        
        return recommendations
    
    def _generate_kb_summary(self, matches: List[Dict[str, Any]]) -> str:
        """Generate a summary of the matching results"""
        if not matches:
            return "No similar papers found in the knowledge base."
        
        top_match = matches[0]
        match_count = len(matches)
        
        summary = f"Found {match_count} similar papers. "
        summary += f"Top match: '{top_match['paper']['title']}' ({top_match['similarity_score']:.1%} similarity). "
        
        if match_count > 1:
            summary += f"Additional {match_count - 1} related papers available for review."
        
        return summary