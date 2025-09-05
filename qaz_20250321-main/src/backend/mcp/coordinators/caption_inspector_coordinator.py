#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Caption Inspector Coordinator

Coordinates caption extraction and analysis across multiple documents.
Integrates with the Caption Extractor Worker to provide comprehensive
caption quality assessment and reporting.
"""

import logging
import json
import os
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import hashlib

from ..workers.caption_extractor_worker import CaptionExtractorWorker, CaptionAnalysisStructure

logger = logging.getLogger(__name__)

class CaptionInspectorCoordinator:
    """Coordinates caption inspection across multiple documents"""
    
    def __init__(self, output_dir: str = "caption_analysis"):
        self.worker = CaptionExtractorWorker()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.analysis_cache = {}
        
        # Create subdirectories
        (self.output_dir / "json").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "quality").mkdir(exist_ok=True)
        
        logger.info(f"Caption Inspector Coordinator initialized with output directory: {self.output_dir}")
    
    def inspect_document_captions(self, document_path: str, document_id: Optional[str] = None, document_title: str = "") -> CaptionAnalysisStructure:
        """Inspect captions in a single document"""
        
        if document_id is None:
            document_id = self._generate_document_id(document_path)
        
        logger.info(f"Starting caption inspection for document: {document_path}")
        
        # Check if we have cached analysis
        cache_key = f"{document_id}_{hash(document_path)}"
        if cache_key in self.analysis_cache:
            logger.info(f"Using cached caption analysis for {document_id}")
            return self.analysis_cache[cache_key]
        
        # Perform caption analysis
        analysis = self.worker.analyze_document_captions(document_path, document_id, document_title)
        
        # Cache the result
        self.analysis_cache[cache_key] = analysis
        
        # Save analysis to JSON
        json_path = self.output_dir / "json" / f"{document_id}_captions.json"
        self.worker.save_analysis_to_json(analysis, str(json_path))
        
        # Generate quality report
        quality_report = self.worker.get_caption_quality_report(analysis)
        quality_path = self.output_dir / "quality" / f"{document_id}_quality.json"
        with open(quality_path, 'w', encoding='utf-8') as f:
            json.dump(quality_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Caption inspection completed for {document_id}: {analysis.total_captions} captions found")
        
        return analysis
    
    def inspect_multiple_documents(self, document_paths: List[str], document_titles: Optional[List[str]] = None) -> Dict[str, CaptionAnalysisStructure]:
        """Inspect captions in multiple documents"""
        
        if document_titles is None:
            document_titles = [""] * len(document_paths)
        
        results = {}
        
        for i, (doc_path, doc_title) in enumerate(zip(document_paths, document_titles)):
            try:
                document_id = self._generate_document_id(doc_path)
                analysis = self.inspect_document_captions(doc_path, document_id, doc_title)
                results[document_id] = analysis
                
                logger.info(f"Processed document {i+1}/{len(document_paths)}: {document_id}")
                
            except Exception as e:
                logger.error(f"Error processing document {doc_path}: {e}")
                continue
        
        # Generate comprehensive report
        self._generate_comprehensive_report(results)
        
        return results
    
    def _generate_document_id(self, document_path: str) -> str:
        """Generate a unique document ID"""
        path_hash = hashlib.md5(document_path.encode()).hexdigest()[:8]
        filename = Path(document_path).stem
        return f"{filename}_{path_hash}"
    
    def _generate_comprehensive_report(self, analyses: Dict[str, CaptionAnalysisStructure]) -> None:
        """Generate a comprehensive report for all analyzed documents"""
        
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_documents': len(analyses),
                'total_captions': sum(analysis.total_captions for analysis in analyses.values()),
                'analysis_method': 'caption_inspector_coordinator'
            },
            'document_summaries': {},
            'overall_statistics': {
                'captions_by_type': {},
                'quality_metrics': {
                    'captions_with_numbers': 0,
                    'captions_without_numbers': 0,
                    'total_quality_issues': 0
                }
            },
            'quality_issues': []
        }
        
        # Aggregate statistics
        for doc_id, analysis in analyses.items():
            # Document summary
            report['document_summaries'][doc_id] = {
                'document_title': analysis.document_title,
                'total_captions': analysis.total_captions,
                'captions_by_type': analysis.captions_by_type,
                'captions_with_numbers': analysis.captions_with_numbers,
                'captions_without_numbers': analysis.captions_without_numbers,
                'confidence_score': analysis.confidence_score
            }
            
            # Aggregate overall statistics
            for caption_type, count in analysis.captions_by_type.items():
                report['overall_statistics']['captions_by_type'][caption_type] = \
                    report['overall_statistics']['captions_by_type'].get(caption_type, 0) + count
            
            report['overall_statistics']['quality_metrics']['captions_with_numbers'] += analysis.captions_with_numbers
            report['overall_statistics']['quality_metrics']['captions_without_numbers'] += analysis.captions_without_numbers
            
            # Collect quality issues
            for assessment in analysis.quality_assessments:
                if assessment.issues:
                    report['quality_issues'].append({
                        'document_id': doc_id,
                        'caption_id': assessment.caption_id,
                        'issues': assessment.issues,
                        'recommendations': assessment.recommendations,
                        'quality_score': assessment.quality_score
                    })
        
        report['overall_statistics']['quality_metrics']['total_quality_issues'] = len(report['quality_issues'])
        
        # Save comprehensive report
        report_path = self.output_dir / "reports" / f"comprehensive_caption_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Comprehensive caption report generated: {report_path}")
    
    def load_analysis_from_json(self, json_path: str) -> Optional[CaptionAnalysisStructure]:
        """Load caption analysis from JSON file"""
        return self.worker.load_analysis_from_json(json_path)
    
    def get_caption_statistics(self, analyses: Dict[str, CaptionAnalysisStructure]) -> Dict[str, Any]:
        """Get statistical summary of caption analyses"""
        
        if not analyses:
            return {}
        
        total_captions = sum(analysis.total_captions for analysis in analyses.values())
        total_documents = len(analyses)
        
        # Aggregate caption types
        caption_types = {}
        for analysis in analyses.values():
            for caption_type, count in analysis.captions_by_type.items():
                caption_types[caption_type] = caption_types.get(caption_type, 0) + count
        
        # Quality metrics
        total_with_numbers = sum(analysis.captions_with_numbers for analysis in analyses.values())
        total_without_numbers = sum(analysis.captions_without_numbers for analysis in analyses.values())
        
        # Average confidence scores
        avg_confidence = sum(analysis.confidence_score for analysis in analyses.values()) / total_documents
        
        return {
            'total_documents': total_documents,
            'total_captions': total_captions,
            'average_captions_per_document': total_captions / total_documents if total_documents > 0 else 0,
            'captions_by_type': caption_types,
            'captions_with_numbers': total_with_numbers,
            'captions_without_numbers': total_without_numbers,
            'percentage_with_numbers': (total_with_numbers / total_captions * 100) if total_captions > 0 else 0,
            'average_confidence_score': avg_confidence,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def generate_markdown_report(self, analyses: Dict[str, CaptionAnalysisStructure], output_path: Optional[str] = None) -> str:
        """Generate a markdown report of caption analysis"""
        
        if output_path is None:
            output_path = self.output_dir / "reports" / f"caption_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        stats = self.get_caption_statistics(analyses)
        
        markdown_content = f"""# Caption Analysis Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

- **Total Documents Analyzed**: {stats['total_documents']}
- **Total Captions Found**: {stats['total_captions']}
- **Average Captions per Document**: {stats['average_captions_per_document']:.1f}
- **Captions with Numbers**: {stats['captions_with_numbers']} ({stats['percentage_with_numbers']:.1f}%)
- **Captions without Numbers**: {stats['captions_without_numbers']}
- **Average Confidence Score**: {stats['average_confidence_score']:.2f}

## Caption Types Distribution

"""
        
        for caption_type, count in stats['captions_by_type'].items():
            percentage = (count / stats['total_captions'] * 100) if stats['total_captions'] > 0 else 0
            markdown_content += f"- **{caption_type.title()}**: {count} ({percentage:.1f}%)\n"
        
        markdown_content += "\n## Document Details\n\n"
        
        for doc_id, analysis in analyses.items():
            markdown_content += f"### {analysis.document_title or doc_id}\n\n"
            markdown_content += f"- **Total Captions**: {analysis.total_captions}\n"
            markdown_content += f"- **Confidence Score**: {analysis.confidence_score:.2f}\n"
            markdown_content += f"- **Captions with Numbers**: {analysis.captions_with_numbers}\n"
            markdown_content += f"- **Captions without Numbers**: {analysis.captions_without_numbers}\n\n"
            
            if analysis.captions_by_type:
                markdown_content += "**Caption Types**:\n"
                for caption_type, count in analysis.captions_by_type.items():
                    markdown_content += f"- {caption_type.title()}: {count}\n"
                markdown_content += "\n"
            
            # Show sample captions
            if analysis.captions:
                markdown_content += "**Sample Captions**:\n"
                for i, caption in enumerate(analysis.captions[:5]):  # Show first 5 captions
                    markdown_content += f"{i+1}. **{caption.caption_type.title()} {caption.caption_number or 'N/A'}**: {caption.caption_text[:100]}{'...' if len(caption.caption_text) > 100 else ''}\n"
                markdown_content += "\n"
        
        # Quality issues summary
        quality_issues = []
        for analysis in analyses.values():
            for assessment in analysis.quality_assessments:
                if assessment.issues:
                    quality_issues.append(assessment)
        
        if quality_issues:
            markdown_content += "## Quality Issues Summary\n\n"
            markdown_content += f"**Total Captions with Issues**: {len(quality_issues)}\n\n"
            
            # Group issues by type
            issue_types = {}
            for assessment in quality_issues:
                for issue in assessment.issues:
                    issue_types[issue] = issue_types.get(issue, 0) + 1
            
            markdown_content += "**Issue Types**:\n"
            for issue, count in issue_types.items():
                markdown_content += f"- {issue}: {count} occurrences\n"
            markdown_content += "\n"
        
        # Save markdown report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        logger.info(f"Markdown report generated: {output_path}")
        
        return str(output_path)
    
    def cleanup_cache(self) -> None:
        """Clean up the analysis cache"""
        self.analysis_cache.clear()
        logger.info("Caption analysis cache cleared")
    
    def get_available_methods(self) -> Dict[str, bool]:
        """Get available caption extraction methods"""
        return self.worker.available_methods.copy()
