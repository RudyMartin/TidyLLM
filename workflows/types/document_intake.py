"""
Intelligent Document Intake System
===================================

Reads PDFs and automatically determines which workflow process they qualify for
based on JSON criteria files. Routes documents to appropriate workflows.

Key Features:
- PDF content extraction and analysis
- Criteria matching against JSON definitions
- Automatic workflow routing
- Confidence scoring for routing decisions
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import fitz  # PyMuPDF

from workflow_registry_system import WorkflowRegistrySystem, RegisteredWorkflow
from documents.simple import SimpleDocumentProcessor

@dataclass
class DocumentProfile:
    """Profile of an analyzed document."""
    file_path: str
    file_name: str
    extracted_text: str
    
    # Document characteristics
    document_type: Optional[str] = None
    page_count: int = 0
    word_count: int = 0
    
    # Content indicators
    keywords_found: List[str] = field(default_factory=list)
    patterns_matched: List[str] = field(default_factory=list)
    sections_identified: List[str] = field(default_factory=list)
    
    # Metadata
    extraction_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    extraction_method: str = "PyMuPDF"
    extraction_success: bool = True
    
    # Analysis results
    compliance_indicators: Dict[str, bool] = field(default_factory=dict)
    risk_indicators: Dict[str, float] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class WorkflowMatch:
    """Result of matching a document to a workflow."""
    workflow_id: str
    workflow_name: str
    confidence_score: float
    
    # Matching details
    criteria_met: List[str] = field(default_factory=list)
    criteria_failed: List[str] = field(default_factory=list)
    required_fields_present: List[str] = field(default_factory=list)
    required_fields_missing: List[str] = field(default_factory=list)
    
    # Scoring breakdown
    scoring_details: Dict[str, float] = field(default_factory=dict)
    weighted_score: float = 0.0
    
    # Recommendation
    recommended: bool = False
    recommendation_reason: str = ""

class IntelligentDocumentIntake:
    """
    Intelligent intake system that reads PDFs and determines workflow routing
    based on JSON criteria.
    """
    
    def __init__(self, registry_system: Optional[WorkflowRegistrySystem] = None):
        """Initialize the intake system."""
        self.registry = registry_system or WorkflowRegistrySystem()
        self.pdf_extractor = SimpleDocumentProcessor()
        
        # Pattern libraries for document type detection
        self.document_patterns = {
            'mvr': {
                'keywords': ['motor vehicle', 'mvr', 'driving record', 'license', 'violation'],
                'patterns': [r'MVR\s*#?\s*\d+', r'License\s*#?\s*\w+', r'Violation\s*Code']
            },
            'financial': {
                'keywords': ['financial', 'revenue', 'balance sheet', 'income', 'expense'],
                'patterns': [r'\$[\d,]+\.?\d*', r'Total\s*:\s*\$?[\d,]+', r'Net\s*(Income|Loss)']
            },
            'compliance': {
                'keywords': ['compliance', 'regulatory', 'audit', 'requirement', 'standard'],
                'patterns': [r'Section\s*\d+\.\d+', r'Requirement\s*#?\d+', r'Compliance\s*Status']
            },
            'contract': {
                'keywords': ['agreement', 'contract', 'party', 'terms', 'conditions'],
                'patterns': [r'WHEREAS', r'NOW\s*THEREFORE', r'Party\s*(A|B|1|2)']
            },
            'quality': {
                'keywords': ['quality', 'qa', 'test', 'validation', 'verification'],
                'patterns': [r'Test\s*Case\s*#?\d+', r'Pass/Fail', r'Quality\s*Score']
            },
            'model_risk': {
                'keywords': ['model', 'risk', 'validation', 'basel', 'sr 11-7', 'occ'],
                'patterns': [r'SR\s*11-7', r'Basel\s*(II|III|IV)', r'Model\s*Risk\s*Management']
            }
        }
    
    def analyze_document(self, pdf_path: str) -> DocumentProfile:
        """Analyze a PDF document and create its profile."""
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Extract text from PDF
        try:
            result = self.pdf_extractor.process_document(str(pdf_path))
            text = result.content
            extraction_success = True
        except Exception as e:
            print(f"Error extracting text: {e}")
            text = ""
            extraction_success = False
        
        # Create document profile
        profile = DocumentProfile(
            file_path=str(pdf_path),
            file_name=pdf_path.name,
            extracted_text=text,
            extraction_success=extraction_success
        )
        
        if text:
            # Basic metrics
            profile.word_count = len(text.split())
            
            # Try to get page count
            try:
                with fitz.open(str(pdf_path)) as doc:
                    profile.page_count = len(doc)
            except:
                profile.page_count = 1
            
            # Detect document type
            profile.document_type = self._detect_document_type(text)
            
            # Find keywords and patterns
            profile.keywords_found = self._find_keywords(text)
            profile.patterns_matched = self._match_patterns(text)
            
            # Identify sections
            profile.sections_identified = self._identify_sections(text)
            
            # Analyze compliance and risk indicators
            profile.compliance_indicators = self._analyze_compliance_indicators(text)
            profile.risk_indicators = self._analyze_risk_indicators(text)
            profile.quality_metrics = self._calculate_quality_metrics(profile)
        
        return profile
    
    def _detect_document_type(self, text: str) -> str:
        """Detect the document type based on content."""
        text_lower = text.lower()
        scores = {}
        
        for doc_type, patterns in self.document_patterns.items():
            score = 0
            
            # Check keywords
            for keyword in patterns['keywords']:
                if keyword in text_lower:
                    score += 1
            
            # Check regex patterns
            for pattern in patterns['patterns']:
                if re.search(pattern, text, re.IGNORECASE):
                    score += 2  # Patterns are weighted higher
            
            scores[doc_type] = score
        
        # Return the type with highest score
        if scores:
            best_type = max(scores, key=scores.get)
            if scores[best_type] > 0:
                return best_type
        
        return 'general'
    
    def _find_keywords(self, text: str) -> List[str]:
        """Find important keywords in the document."""
        keywords_found = []
        text_lower = text.lower()
        
        # Check all keyword lists
        for doc_type, patterns in self.document_patterns.items():
            for keyword in patterns['keywords']:
                if keyword in text_lower and keyword not in keywords_found:
                    keywords_found.append(keyword)
        
        return keywords_found
    
    def _match_patterns(self, text: str) -> List[str]:
        """Match regex patterns in the document."""
        patterns_matched = []
        
        # Check all pattern lists
        for doc_type, patterns in self.document_patterns.items():
            for pattern in patterns['patterns']:
                if re.search(pattern, text, re.IGNORECASE):
                    patterns_matched.append(pattern)
        
        return patterns_matched
    
    def _identify_sections(self, text: str) -> List[str]:
        """Identify document sections."""
        sections = []
        
        # Common section patterns
        section_patterns = [
            r'^#+\s*(.+)$',  # Markdown headers
            r'^([A-Z][A-Z\s]+):?\s*$',  # All caps headers
            r'^\d+\.\s+(.+)$',  # Numbered sections
            r'^Section\s+\d+[:\.]?\s*(.+)$',  # Section headers
        ]
        
        lines = text.split('\n')
        for line in lines[:100]:  # Check first 100 lines
            for pattern in section_patterns:
                match = re.match(pattern, line.strip())
                if match:
                    section = match.group(1) if match.lastindex else match.group(0)
                    sections.append(section.strip())
                    break
        
        return sections[:20]  # Return up to 20 sections
    
    def _analyze_compliance_indicators(self, text: str) -> Dict[str, bool]:
        """Analyze compliance-related indicators in the document."""
        indicators = {}
        text_lower = text.lower()
        
        # Check for common compliance indicators
        indicators['has_regulatory_references'] = bool(
            re.search(r'(regulation|regulatory|compliance|standard)', text_lower)
        )
        indicators['has_section_numbers'] = bool(
            re.search(r'section\s+\d+', text_lower)
        )
        indicators['has_requirements'] = bool(
            re.search(r'(requirement|required|must|shall)', text_lower)
        )
        indicators['has_audit_trail'] = bool(
            re.search(r'(audit|review|validation|verification)', text_lower)
        )
        indicators['has_risk_assessment'] = bool(
            re.search(r'(risk\s+(assessment|analysis|score|level))', text_lower)
        )
        
        return indicators
    
    def _analyze_risk_indicators(self, text: str) -> Dict[str, float]:
        """Analyze risk-related indicators in the document."""
        indicators = {}
        text_lower = text.lower()
        
        # Count risk-related terms
        high_risk_terms = ['critical', 'severe', 'high risk', 'material', 'significant']
        medium_risk_terms = ['moderate', 'medium risk', 'notable', 'important']
        low_risk_terms = ['low risk', 'minimal', 'acceptable', 'minor']
        
        high_count = sum(1 for term in high_risk_terms if term in text_lower)
        medium_count = sum(1 for term in medium_risk_terms if term in text_lower)
        low_count = sum(1 for term in low_risk_terms if term in text_lower)
        
        total = high_count + medium_count + low_count
        if total > 0:
            indicators['high_risk_ratio'] = high_count / total
            indicators['medium_risk_ratio'] = medium_count / total
            indicators['low_risk_ratio'] = low_count / total
        else:
            indicators['high_risk_ratio'] = 0.0
            indicators['medium_risk_ratio'] = 0.0
            indicators['low_risk_ratio'] = 0.0
        
        # Overall risk score (0-1, higher is riskier)
        indicators['overall_risk_score'] = (
            indicators['high_risk_ratio'] * 1.0 +
            indicators['medium_risk_ratio'] * 0.5 +
            indicators['low_risk_ratio'] * 0.1
        )
        
        return indicators
    
    def _calculate_quality_metrics(self, profile: DocumentProfile) -> Dict[str, float]:
        """Calculate document quality metrics."""
        metrics = {}
        
        # Completeness (based on extraction success and content)
        metrics['completeness'] = 1.0 if profile.extraction_success else 0.0
        if profile.word_count < 100:
            metrics['completeness'] *= 0.5
        
        # Structure (based on sections identified)
        metrics['structure'] = min(len(profile.sections_identified) / 5, 1.0)
        
        # Clarity (based on patterns matched)
        metrics['clarity'] = min(len(profile.patterns_matched) / 3, 1.0)
        
        # Relevance (based on keywords found)
        metrics['relevance'] = min(len(profile.keywords_found) / 5, 1.0)
        
        # Overall quality score
        metrics['overall_quality'] = (
            metrics['completeness'] * 0.3 +
            metrics['structure'] * 0.2 +
            metrics['clarity'] * 0.2 +
            metrics['relevance'] * 0.3
        )
        
        return metrics
    
    def match_to_workflows(self, profile: DocumentProfile, 
                          min_confidence: float = 0.5) -> List[WorkflowMatch]:
        """Match a document profile to available workflows based on criteria."""
        matches = []
        
        for workflow in self.registry.list_workflows(status='active'):
            match = self._evaluate_workflow_match(profile, workflow)
            
            if match.confidence_score >= min_confidence:
                matches.append(match)
        
        # Sort by confidence score
        matches.sort(key=lambda x: x.confidence_score, reverse=True)
        
        # Mark top match as recommended if confidence is high enough
        if matches and matches[0].confidence_score >= 0.7:
            matches[0].recommended = True
            matches[0].recommendation_reason = "Highest confidence match with strong criteria alignment"
        
        return matches
    
    def _evaluate_workflow_match(self, profile: DocumentProfile, 
                                 workflow: RegisteredWorkflow) -> WorkflowMatch:
        """Evaluate how well a document matches a workflow's criteria."""
        match = WorkflowMatch(
            workflow_id=workflow.workflow_id,
            workflow_name=workflow.workflow_name,
            confidence_score=0.0
        )
        
        criteria = workflow.criteria
        scores = {}
        
        # Check document type alignment
        if workflow.workflow_type == profile.document_type:
            scores['document_type'] = 1.0
            match.criteria_met.append('document_type_match')
        elif workflow.workflow_type == 'general':
            scores['document_type'] = 0.5
        else:
            scores['document_type'] = 0.2
            match.criteria_failed.append('document_type_mismatch')
        
        # Check required fields
        for field in criteria.required_fields:
            if field.lower() in profile.extracted_text.lower():
                match.required_fields_present.append(field)
            else:
                match.required_fields_missing.append(field)
        
        if criteria.required_fields:
            field_score = len(match.required_fields_present) / len(criteria.required_fields)
            scores['required_fields'] = field_score
            if field_score < 1.0:
                match.criteria_failed.append('missing_required_fields')
        
        # Check validation rules
        validation_score = 0.0
        for rule in criteria.validation_rules:
            # Simple keyword-based validation for now
            if rule.lower() in profile.extracted_text.lower():
                validation_score += 1
                match.criteria_met.append(f'validation_rule_{rule}')
        
        if criteria.validation_rules:
            scores['validation_rules'] = validation_score / len(criteria.validation_rules)
        
        # Check quality threshold
        if profile.quality_metrics.get('overall_quality', 0) >= criteria.minimum_score:
            scores['quality_threshold'] = 1.0
            match.criteria_met.append('quality_threshold_met')
        else:
            scores['quality_threshold'] = profile.quality_metrics.get('overall_quality', 0) / criteria.minimum_score
            match.criteria_failed.append('quality_below_threshold')
        
        # Check compliance indicators if relevant
        if 'compliance' in workflow.workflow_type or 'regulatory' in criteria.regulatory_standards:
            compliance_score = sum(profile.compliance_indicators.values()) / max(len(profile.compliance_indicators), 1)
            scores['compliance_alignment'] = compliance_score
            if compliance_score > 0.5:
                match.criteria_met.append('compliance_indicators_present')
        
        # Calculate weighted score based on workflow's weight scheme
        if criteria.weight_scheme:
            weighted_sum = 0.0
            weight_total = 0.0
            
            for score_type, score_value in scores.items():
                weight = criteria.weight_scheme.get(score_type, 0.25)
                weighted_sum += score_value * weight
                weight_total += weight
            
            match.weighted_score = weighted_sum / weight_total if weight_total > 0 else 0
        else:
            # Simple average if no weights defined
            match.weighted_score = sum(scores.values()) / len(scores) if scores else 0
        
        match.confidence_score = match.weighted_score
        match.scoring_details = scores
        
        return match
    
    def process_document(self, pdf_path: str, auto_route: bool = True) -> Dict[str, Any]:
        """
        Process a document through the intake system.
        
        Args:
            pdf_path: Path to the PDF document
            auto_route: If True, automatically route to best matching workflow
        
        Returns:
            Dictionary with processing results
        """
        # Analyze document
        profile = self.analyze_document(pdf_path)
        
        # Find matching workflows
        matches = self.match_to_workflows(profile)
        
        result = {
            'document_profile': profile,
            'workflow_matches': matches,
            'recommended_workflow': None,
            'routing_decision': None
        }
        
        if matches and matches[0].recommended:
            result['recommended_workflow'] = matches[0]
            
            if auto_route:
                # Route to recommended workflow
                result['routing_decision'] = {
                    'workflow_id': matches[0].workflow_id,
                    'workflow_name': matches[0].workflow_name,
                    'confidence': matches[0].confidence_score,
                    'timestamp': datetime.now().isoformat(),
                    'auto_routed': True
                }
        
        return result
    
    def generate_intake_report(self, result: Dict[str, Any]) -> str:
        """Generate a human-readable intake report."""
        profile = result['document_profile']
        matches = result['workflow_matches']
        
        report = []
        report.append("=" * 60)
        report.append("DOCUMENT INTAKE ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Document Information
        report.append("DOCUMENT INFORMATION:")
        report.append(f"  File: {profile.file_name}")
        report.append(f"  Type Detected: {profile.document_type}")
        report.append(f"  Pages: {profile.page_count}")
        report.append(f"  Words: {profile.word_count}")
        report.append(f"  Extraction Success: {profile.extraction_success}")
        report.append("")
        
        # Quality Metrics
        report.append("QUALITY METRICS:")
        for metric, value in profile.quality_metrics.items():
            report.append(f"  {metric}: {value:.2%}")
        report.append("")
        
        # Content Analysis
        report.append("CONTENT ANALYSIS:")
        report.append(f"  Keywords Found: {', '.join(profile.keywords_found[:10])}")
        report.append(f"  Patterns Matched: {len(profile.patterns_matched)}")
        report.append(f"  Sections Identified: {len(profile.sections_identified)}")
        
        if profile.compliance_indicators:
            report.append("  Compliance Indicators:")
            for indicator, present in profile.compliance_indicators.items():
                if present:
                    report.append(f"    ✓ {indicator}")
        report.append("")
        
        # Workflow Matches
        report.append("WORKFLOW MATCHING RESULTS:")
        if matches:
            for i, match in enumerate(matches[:5], 1):
                status = "★ RECOMMENDED" if match.recommended else ""
                report.append(f"  {i}. {match.workflow_name} ({match.confidence_score:.2%}) {status}")
                report.append(f"     Criteria Met: {', '.join(match.criteria_met[:3])}")
                if match.criteria_failed:
                    report.append(f"     Criteria Failed: {', '.join(match.criteria_failed[:3])}")
        else:
            report.append("  No suitable workflows found")
        report.append("")
        
        # Routing Decision
        if result.get('routing_decision'):
            decision = result['routing_decision']
            report.append("ROUTING DECISION:")
            report.append(f"  Workflow: {decision['workflow_name']}")
            report.append(f"  Confidence: {decision['confidence']:.2%}")
            report.append(f"  Auto-Routed: {decision['auto_routed']}")
        
        return "\n".join(report)


def main():
    """Example usage of the intelligent intake system."""
    print("Initializing Intelligent Document Intake System...")
    
    # Initialize systems
    registry = WorkflowRegistrySystem()
    intake = IntelligentDocumentIntake(registry)
    
    # Test with a sample PDF
    test_pdf = Path("knowledge_base/pdfs/2019-02-26-Model-Validation.pdf")
    
    if test_pdf.exists():
        print(f"\nProcessing document: {test_pdf.name}")
        
        # Process the document
        result = intake.process_document(str(test_pdf))
        
        # Generate and print report
        report = intake.generate_intake_report(result)
        print(report)
        
        # Save report
        report_file = Path("workflow_registry") / f"intake_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\nReport saved to: {report_file}")
    else:
        print(f"Test PDF not found: {test_pdf}")
    
    return intake


if __name__ == "__main__":
    intake_system = main()