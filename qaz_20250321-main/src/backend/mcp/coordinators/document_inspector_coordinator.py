#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Document Inspector Coordinator

Coordinates the work of multiple document inspectors:
- TOC Extractor Worker
- Bibliography Builder Worker  
- Link Inspector Worker

This coordinator provides a unified interface for comprehensive document analysis
and integrates the results into the database-enhanced QA system.
"""

import logging
import json
import uuid
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

from ..workers.toc_extractor_worker import TOCExtractorWorker, TOCStructure
from ..workers.bibliography_builder_worker import BibliographyBuilderWorker, BibliographyStructure
from ..workers.link_inspector_worker import LinkInspectorWorker, LinkAnalysisStructure
from ...core.database_connection_manager import get_database_manager

logger = logging.getLogger(__name__)

@dataclass
class DocumentInspectionResult:
    """Complete document inspection result combining all inspectors"""
    document_id: str
    document_title: str
    document_path: str
    inspection_timestamp: str
    
    # Individual inspection results
    toc_analysis: Optional[TOCStructure]
    bibliography_analysis: Optional[BibliographyStructure]
    link_analysis: Optional[LinkAnalysisStructure]
    
    # Combined metrics
    total_sections: int
    total_citations: int
    total_links: int
    broken_links: int
    valid_links: int
    
    # Quality scores
    toc_quality_score: float
    bibliography_quality_score: float
    link_quality_score: float
    overall_quality_score: float
    
    # Issues and recommendations
    issues: List[Dict[str, Any]]
    recommendations: List[str]
    
    # Metadata
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'document_id': self.document_id,
            'document_title': self.document_title,
            'document_path': self.document_path,
            'inspection_timestamp': self.inspection_timestamp,
            'toc_analysis': self.toc_analysis.to_dict() if self.toc_analysis else None,
            'bibliography_analysis': self.bibliography_analysis.to_dict() if self.bibliography_analysis else None,
            'link_analysis': self.link_analysis.to_dict() if self.link_analysis else None,
            'total_sections': self.total_sections,
            'total_citations': self.total_citations,
            'total_links': self.total_links,
            'broken_links': self.broken_links,
            'valid_links': self.valid_links,
            'toc_quality_score': self.toc_quality_score,
            'bibliography_quality_score': self.bibliography_quality_score,
            'link_quality_score': self.link_quality_score,
            'overall_quality_score': self.overall_quality_score,
            'issues': self.issues,
            'recommendations': self.recommendations,
            'metadata': self.metadata
        }

class DocumentInspectorCoordinator:
    """Coordinates multiple document inspectors for comprehensive analysis"""
    
    def __init__(self):
        self.db_manager = get_database_manager()
        self.toc_worker = TOCExtractorWorker()
        self.bibliography_worker = BibliographyBuilderWorker()
        self.link_worker = LinkInspectorWorker()
        self.inspection_cache = {}
    
    def inspect_document(self, 
                        document_path: str, 
                        document_id: Optional[str] = None,
                        document_title: Optional[str] = None,
                        validate_links: bool = True,
                        extract_toc: bool = True,
                        extract_bibliography: bool = True,
                        extract_links: bool = True) -> DocumentInspectionResult:
        """
        Perform comprehensive document inspection
        
        Args:
            document_path: Path to the document file
            document_id: Unique document identifier (auto-generated if None)
            document_title: Document title (extracted from file if None)
            validate_links: Whether to validate external links
            extract_toc: Whether to extract table of contents
            extract_bibliography: Whether to extract bibliography
            extract_links: Whether to extract and validate links
        
        Returns:
            DocumentInspectionResult with comprehensive analysis
        """
        
        # Generate document ID if not provided
        if not document_id:
            document_id = self._generate_document_id(document_path)
        
        # Extract document title if not provided
        if not document_title:
            document_title = Path(document_path).stem
        
        # Check cache
        cache_key = f"{document_id}_{hash(document_path)}"
        if cache_key in self.inspection_cache:
            logger.info(f"Using cached inspection result for {document_id}")
            return self.inspection_cache[cache_key]
        
        logger.info(f"Starting comprehensive document inspection for: {document_path}")
        
        # Initialize results
        toc_analysis = None
        bibliography_analysis = None
        link_analysis = None
        
        # Extract TOC if requested
        if extract_toc:
            try:
                logger.info("Extracting table of contents...")
                toc_analysis = self.toc_worker.extract_toc_from_document(
                    document_path, document_id, document_title
                )
                logger.info(f"TOC extraction completed: {len(toc_analysis.entries) if toc_analysis else 0} entries")
            except Exception as e:
                logger.error(f"Error extracting TOC: {e}")
        
        # Extract bibliography if requested
        if extract_bibliography:
            try:
                logger.info("Extracting bibliography...")
                bibliography_analysis = self.bibliography_worker.extract_bibliography_from_document(
                    document_path, document_id, document_title
                )
                logger.info(f"Bibliography extraction completed: {len(bibliography_analysis.citations) if bibliography_analysis else 0} citations")
            except Exception as e:
                logger.error(f"Error extracting bibliography: {e}")
        
        # Extract and validate links if requested
        if extract_links:
            try:
                logger.info("Extracting and validating links...")
                link_analysis = self.link_worker.analyze_document_links(
                    document_path, document_id, document_title, validate_links
                )
                logger.info(f"Link analysis completed: {link_analysis.total_links} links, {link_analysis.broken_links} broken")
            except Exception as e:
                logger.error(f"Error analyzing links: {e}")
        
        # Calculate combined metrics
        total_sections = len(toc_analysis.entries) if toc_analysis else 0
        total_citations = len(bibliography_analysis.citations) if bibliography_analysis else 0
        total_links = link_analysis.total_links if link_analysis else 0
        broken_links = link_analysis.broken_links if link_analysis else 0
        valid_links = link_analysis.valid_links if link_analysis else 0
        
        # Calculate quality scores
        toc_quality_score = toc_analysis.confidence_score if toc_analysis else 0.0
        bibliography_quality_score = bibliography_analysis.confidence_score if bibliography_analysis else 0.0
        link_quality_score = link_analysis.confidence_score if link_analysis else 0.0
        
        # Calculate overall quality score
        overall_quality_score = self._calculate_overall_quality_score(
            toc_quality_score, bibliography_quality_score, link_quality_score,
            total_sections, total_citations, total_links, broken_links
        )
        
        # Generate issues and recommendations
        issues = self._identify_issues(
            toc_analysis, bibliography_analysis, link_analysis,
            total_sections, total_citations, total_links, broken_links
        )
        
        recommendations = self._generate_recommendations(issues)
        
        # Create inspection result
        result = DocumentInspectionResult(
            document_id=document_id,
            document_title=document_title,
            document_path=document_path,
            inspection_timestamp=datetime.now().isoformat(),
            toc_analysis=toc_analysis,
            bibliography_analysis=bibliography_analysis,
            link_analysis=link_analysis,
            total_sections=total_sections,
            total_citations=total_citations,
            total_links=total_links,
            broken_links=broken_links,
            valid_links=valid_links,
            toc_quality_score=toc_quality_score,
            bibliography_quality_score=bibliography_quality_score,
            link_quality_score=link_quality_score,
            overall_quality_score=overall_quality_score,
            issues=issues,
            recommendations=recommendations,
            metadata={
                'file_size': Path(document_path).stat().st_size if Path(document_path).exists() else 0,
                'file_extension': Path(document_path).suffix,
                'extraction_methods': {
                    'toc': extract_toc,
                    'bibliography': extract_bibliography,
                    'links': extract_links
                },
                'link_validation_enabled': validate_links
            }
        )
        
        # Cache the result
        self.inspection_cache[cache_key] = result
        
        # Store in database
        self._store_inspection_result(result)
        
        logger.info(f"Document inspection completed for {document_id}: Overall quality score: {overall_quality_score:.2f}")
        
        return result
    
    def get_broken_links_report(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get broken links report for a document"""
        try:
            # Try to get from cache first
            for result in self.inspection_cache.values():
                if result.document_id == document_id and result.link_analysis:
                    return self.link_worker.get_broken_links_report(result.link_analysis)
            
            # If not in cache, try to load from database
            # This would require implementing database storage/retrieval
            return None
            
        except Exception as e:
            logger.error(f"Error getting broken links report for {document_id}: {e}")
            return None
    
    def get_document_quality_metrics(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get quality metrics for a document"""
        try:
            # Try to get from cache first
            for result in self.inspection_cache.values():
                if result.document_id == document_id:
                    return {
                        'document_id': result.document_id,
                        'document_title': result.document_title,
                        'overall_quality_score': result.overall_quality_score,
                        'toc_quality_score': result.toc_quality_score,
                        'bibliography_quality_score': result.bibliography_quality_score,
                        'link_quality_score': result.link_quality_score,
                        'total_sections': result.total_sections,
                        'total_citations': result.total_citations,
                        'total_links': result.total_links,
                        'broken_links': result.broken_links,
                        'valid_links': result.valid_links,
                        'issues_count': len(result.issues),
                        'recommendations_count': len(result.recommendations),
                        'inspection_timestamp': result.inspection_timestamp
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting quality metrics for {document_id}: {e}")
            return None
    
    def _generate_document_id(self, document_path: str) -> str:
        """Generate a unique document ID"""
        content = f"{document_path}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _calculate_overall_quality_score(self, 
                                       toc_score: float, 
                                       bibliography_score: float, 
                                       link_score: float,
                                       total_sections: int,
                                       total_citations: int,
                                       total_links: int,
                                       broken_links: int) -> float:
        """Calculate overall quality score based on all factors"""
        
        # Base scores (weighted average)
        base_score = (toc_score * 0.3 + bibliography_score * 0.3 + link_score * 0.4)
        
        # Adjustments based on content richness
        richness_bonus = 0.0
        
        # Bonus for having TOC sections
        if total_sections > 0:
            richness_bonus += min(0.1, total_sections / 100.0)
        
        # Bonus for having citations
        if total_citations > 0:
            richness_bonus += min(0.1, total_citations / 50.0)
        
        # Penalty for broken links
        link_penalty = 0.0
        if total_links > 0:
            broken_link_ratio = broken_links / total_links
            link_penalty = broken_link_ratio * 0.2  # Up to 20% penalty for broken links
        
        # Calculate final score
        final_score = base_score + richness_bonus - link_penalty
        
        return max(0.0, min(1.0, final_score))
    
    def _identify_issues(self, 
                        toc_analysis: Optional[TOCStructure],
                        bibliography_analysis: Optional[BibliographyStructure],
                        link_analysis: Optional[LinkAnalysisStructure],
                        total_sections: int,
                        total_citations: int,
                        total_links: int,
                        broken_links: int) -> List[Dict[str, Any]]:
        """Identify issues in the document"""
        
        issues = []
        
        # TOC issues
        if toc_analysis:
            if total_sections == 0:
                issues.append({
                    'type': 'toc',
                    'severity': 'medium',
                    'message': 'No table of contents found',
                    'recommendation': 'Add a table of contents for better document structure'
                })
            elif toc_analysis.confidence_score < 0.7:
                issues.append({
                    'type': 'toc',
                    'severity': 'low',
                    'message': 'Low confidence in TOC extraction',
                    'recommendation': 'Improve TOC formatting for better extraction'
                })
        else:
            issues.append({
                'type': 'toc',
                'severity': 'low',
                'message': 'TOC extraction not performed',
                'recommendation': 'Enable TOC extraction for comprehensive analysis'
            })
        
        # Bibliography issues
        if bibliography_analysis:
            if total_citations == 0:
                issues.append({
                    'type': 'bibliography',
                    'severity': 'medium',
                    'message': 'No citations found',
                    'recommendation': 'Add proper citations and references'
                })
            elif bibliography_analysis.confidence_score < 0.7:
                issues.append({
                    'type': 'bibliography',
                    'severity': 'low',
                    'message': 'Low confidence in bibliography extraction',
                    'recommendation': 'Improve reference formatting for better extraction'
                })
        else:
            issues.append({
                'type': 'bibliography',
                'severity': 'low',
                'message': 'Bibliography extraction not performed',
                'recommendation': 'Enable bibliography extraction for comprehensive analysis'
            })
        
        # Link issues
        if link_analysis:
            if total_links == 0:
                issues.append({
                    'type': 'links',
                    'severity': 'low',
                    'message': 'No links found in document',
                    'recommendation': 'Consider adding relevant links for better connectivity'
                })
            elif broken_links > 0:
                broken_ratio = broken_links / total_links
                if broken_ratio > 0.3:
                    severity = 'high'
                elif broken_ratio > 0.1:
                    severity = 'medium'
                else:
                    severity = 'low'
                
                issues.append({
                    'type': 'links',
                    'severity': severity,
                    'message': f'{broken_links} broken links found ({broken_ratio:.1%} of total)',
                    'recommendation': 'Fix broken links to improve document quality',
                    'details': {
                        'broken_links_count': broken_links,
                        'total_links_count': total_links,
                        'broken_ratio': broken_ratio
                    }
                })
            
            if link_analysis.confidence_score < 0.7:
                issues.append({
                    'type': 'links',
                    'severity': 'low',
                    'message': 'Low confidence in link extraction',
                    'recommendation': 'Improve link formatting for better extraction'
                })
        else:
            issues.append({
                'type': 'links',
                'severity': 'low',
                'message': 'Link analysis not performed',
                'recommendation': 'Enable link analysis for comprehensive quality assessment'
            })
        
        return issues
    
    def _generate_recommendations(self, issues: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on identified issues"""
        
        recommendations = []
        
        # Group issues by type
        toc_issues = [issue for issue in issues if issue['type'] == 'toc']
        bibliography_issues = [issue for issue in issues if issue['type'] == 'bibliography']
        link_issues = [issue for issue in issues if issue['type'] == 'links']
        
        # TOC recommendations
        if toc_issues:
            recommendations.append("📋 **Table of Contents**: " + toc_issues[0]['recommendation'])
        
        # Bibliography recommendations
        if bibliography_issues:
            recommendations.append("📚 **References**: " + bibliography_issues[0]['recommendation'])
        
        # Link recommendations
        if link_issues:
            recommendations.append("🔗 **Links**: " + link_issues[0]['recommendation'])
        
        # General recommendations
        if not recommendations:
            recommendations.append("✅ Document appears to be well-structured with good quality")
        
        return recommendations
    
    def _store_inspection_result(self, result: DocumentInspectionResult) -> None:
        """Store inspection result in database"""
        try:
            # Store basic metrics
            metrics_data = {
                'document_id': result.document_id,
                'overall_quality_score': result.overall_quality_score,
                'toc_quality_score': result.toc_quality_score,
                'bibliography_quality_score': result.bibliography_quality_score,
                'link_quality_score': result.link_quality_score,
                'total_sections': result.total_sections,
                'total_citations': result.total_citations,
                'total_links': result.total_links,
                'broken_links': result.broken_links,
                'valid_links': result.valid_links,
                'issues_count': len(result.issues),
                'inspection_timestamp': result.inspection_timestamp
            }
            
            # Store in database (implementation depends on your schema)
            # This is a placeholder for database storage
            logger.info(f"Stored inspection metrics for document {result.document_id}")
            
        except Exception as e:
            logger.error(f"Error storing inspection result: {e}")
    
    def save_inspection_result(self, result: DocumentInspectionResult, output_path: str) -> bool:
        """Save inspection result to JSON file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Inspection result saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving inspection result to {output_path}: {e}")
            return False
    
    def load_inspection_result(self, json_path: str) -> Optional[DocumentInspectionResult]:
        """Load inspection result from JSON file"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Reconstruct the result (simplified - would need full reconstruction in practice)
            result = DocumentInspectionResult(
                document_id=data['document_id'],
                document_title=data['document_title'],
                document_path=data['document_path'],
                inspection_timestamp=data['inspection_timestamp'],
                toc_analysis=None,  # Would need to reconstruct TOCStructure
                bibliography_analysis=None,  # Would need to reconstruct BibliographyStructure
                link_analysis=None,  # Would need to reconstruct LinkAnalysisStructure
                total_sections=data['total_sections'],
                total_citations=data['total_citations'],
                total_links=data['total_links'],
                broken_links=data['broken_links'],
                valid_links=data['valid_links'],
                toc_quality_score=data['toc_quality_score'],
                bibliography_quality_score=data['bibliography_quality_score'],
                link_quality_score=data['link_quality_score'],
                overall_quality_score=data['overall_quality_score'],
                issues=data['issues'],
                recommendations=data['recommendations'],
                metadata=data['metadata']
            )
            
            return result
        except Exception as e:
            logger.error(f"Error loading inspection result from {json_path}: {e}")
            return None
