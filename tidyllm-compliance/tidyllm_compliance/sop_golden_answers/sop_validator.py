"""
SOP Validator Implementation
===========================

Validates MVR analysis against SOP Golden Answers with precedence logic.
Integrates with existing tidyllm-compliance validators but gives SOP
standards the highest priority.

Part of tidyllm-compliance: Automated compliance with complete transparency
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import json
from dataclasses import dataclass

# Import existing compliance components
from ..model_risk import ModelRiskMonitor
from ..evidence import EvidenceValidator

# Import SOP conflict analysis components
from ..sop_conflict_analysis import SOPConflictReporter, YRSNNoiseAnalyzer, TemporalResolver, ComplianceSOPFallback

@dataclass
class SOPAnswer:
    """Structure for SOP golden answer responses"""
    question: str
    answer: str
    confidence: float
    sop_source: str
    precedence_level: str  # 'golden', 'standard', 'guidance'
    workflow_stage: Optional[str] = None
    checklist_items: List[str] = None

@dataclass  
class SOPValidationResult:
    """Results from SOP validation with precedence"""
    overall_compliance: str  # 'compliant', 'non_compliant', 'needs_review'
    sop_score: float
    sop_answers: List[SOPAnswer]
    fallback_used: bool
    recommendations: List[str]
    stage_guidance: Dict[str, Any]

class SOPValidator:
    """
    SOP Golden Answers validator with precedence over general compliance rules.
    
    This is where the actual SOP compliance logic gets implemented - 
    the core functionality that supports the MVR analysis workflow.
    """
    
    def __init__(self, sop_knowledge_base_path: Optional[Path] = None):
        """Initialize SOP validator with golden answers knowledge base."""
        self.sop_knowledge_base = sop_knowledge_base_path
        self.sop_answers = self._load_sop_golden_answers()
        
        # Leverage existing compliance components as fallback
        self.model_risk_monitor = ModelRiskMonitor()
        self.evidence_validator = EvidenceValidator()
        
        # Initialize SOP conflict analysis components
        self.conflict_reporter = SOPConflictReporter()
        self.yrsn_analyzer = YRSNNoiseAnalyzer()
        self.temporal_resolver = TemporalResolver()
        self.fallback_strategy = ComplianceSOPFallback()
        
        # SOP precedence hierarchy
        self.precedence_levels = {
            'golden': 1.0,      # Golden answers - highest precedence
            'standard': 0.8,    # Standard procedures - high precedence  
            'guidance': 0.6,    # General guidance - medium precedence
            'fallback': 0.4     # General compliance rules - lowest precedence
        }
    
    def _load_sop_golden_answers(self) -> Dict[str, SOPAnswer]:
        """Load SOP golden answers from knowledge base."""
        sop_answers = {}
        
        # This is where we'd load actual SOP content
        # For now, implement core MVR workflow SOP answers
        
        # MVR Analysis Stage 1: Classification & Tagging
        sop_answers['mvr_tag_classification'] = SOPAnswer(
            question="How should MVR documents be classified and tagged?",
            answer="MVR documents must be classified using REV00000 format matching. Extract metadata including document type (MVR vs VST), revision number, and business purpose. Perform YNSR noise analysis to determine document quality before proceeding to QA stage.",
            confidence=1.0,
            sop_source="MVR SOP Section 3.1",
            precedence_level="golden",
            workflow_stage="mvr_tag",
            checklist_items=[
                "REV00000 format ID extracted",
                "Document type classified (MVR/VST)", 
                "YNSR noise analysis completed",
                "Business purpose identified",
                "Metadata stored in TidyMart"
            ]
        )
        
        # MVR Analysis Stage 2: QA Comparison
        sop_answers['mvr_qa_comparison'] = SOPAnswer(
            question="How should MVR vs VST comparison be performed?",
            answer="MVR and VST documents must be matched by REV00000 metadata ID. Perform section-by-section comparison documenting discrepancies. Generate both digest and detailed markdown reports. All findings must be validated against domain RAG knowledge before proceeding to peer review.",
            confidence=1.0,
            sop_source="MVR SOP Section 4.2",
            precedence_level="golden", 
            workflow_stage="mvr_qa",
            checklist_items=[
                "REV00000 metadata matching confirmed",
                "Section-by-section comparison completed",
                "Digest markdown report generated",
                "Detailed markdown report generated", 
                "Domain RAG validation performed",
                "Results stored in TidyMart"
            ]
        )
        
        # MVR Analysis Stage 3: Peer Review
        sop_answers['mvr_peer_review'] = SOPAnswer(
            question="What is required for MVR peer review stage?",
            answer="Peer review must triangulate analysis from three sources: original MVR text, digest review, and stepwise review. Use domain RAG for contextual validation. All three analyses must achieve consensus before final report generation. Document any disagreements and resolution approach.",
            confidence=1.0,
            sop_source="MVR SOP Section 5.1", 
            precedence_level="golden",
            workflow_stage="mvr_peer",
            checklist_items=[
                "Domain RAG knowledge loaded",
                "MVR text analysis completed",
                "Digest review analysis completed", 
                "Stepwise review analysis completed",
                "Triangulation consensus achieved",
                "Disagreements documented and resolved",
                "Results saved to database"
            ]
        )
        
        # MVR Analysis Stage 4: Report Generation  
        sop_answers['mvr_report_generation'] = SOPAnswer(
            question="What are the requirements for MVR final reports?",
            answer="Final reports must include comprehensive markdown analysis, formatted PDF report, and structured JSON summary. All outputs must maintain audit trail linking back to original documents and analysis steps. Archive all final outputs with cleanup of temporary files.",
            confidence=1.0,
            sop_source="MVR SOP Section 6.3",
            precedence_level="golden",
            workflow_stage="mvr_report", 
            checklist_items=[
                "Comprehensive markdown report created",
                "Formatted PDF report generated",
                "Structured JSON summary produced",
                "Audit trail maintained throughout",
                "All outputs archived properly",
                "Temporary files cleaned up"
            ]
        )
        
        return sop_answers
    
    def validate_with_sop_precedence(self, question: str, context: Dict[str, Any]) -> SOPValidationResult:
        """
        Validate using SOP golden answers with highest precedence.
        
        This is the core method that implements SOP-guided validation.
        """
        # Step 1: Query SOP golden answers first (highest precedence)
        sop_matches = self._query_sop_answers(question, context)
        
        if sop_matches and max(match.confidence for match in sop_matches) >= 0.8:
            # High-confidence SOP answer found - use it
            return SOPValidationResult(
                overall_compliance="compliant" if self._check_sop_compliance(sop_matches, context) else "needs_review",
                sop_score=max(match.confidence for match in sop_matches),
                sop_answers=sop_matches,
                fallback_used=False,
                recommendations=self._generate_sop_recommendations(sop_matches, context),
                stage_guidance=self._get_stage_specific_guidance(sop_matches, context)
            )
        
        # Step 2: Fall back to general compliance rules (lower precedence)
        fallback_result = self._fallback_validation(question, context)
        
        return SOPValidationResult(
            overall_compliance="needs_review",
            sop_score=0.4,  # Fallback precedence level
            sop_answers=sop_matches or [],
            fallback_used=True,
            recommendations=fallback_result.get("recommendations", []),
            stage_guidance=fallback_result.get("stage_guidance", {})
        )
    
    def _query_sop_answers(self, question: str, context: Dict[str, Any]) -> List[SOPAnswer]:
        """Query SOP golden answers for relevant responses."""
        matches = []
        
        # Get current workflow stage from context
        current_stage = context.get('workflow_stage', 'unknown')
        
        # Find SOP answers relevant to current stage and question
        for answer_id, sop_answer in self.sop_answers.items():
            if (sop_answer.workflow_stage == current_stage or 
                any(keyword in question.lower() for keyword in answer_id.split('_'))):
                matches.append(sop_answer)
        
        # Sort by precedence level and confidence
        matches.sort(key=lambda x: (self.precedence_levels.get(x.precedence_level, 0), x.confidence), reverse=True)
        
        return matches
    
    def _check_sop_compliance(self, sop_answers: List[SOPAnswer], context: Dict[str, Any]) -> bool:
        """Check if context meets SOP compliance requirements."""
        for sop_answer in sop_answers:
            if sop_answer.checklist_items:
                # Check if all required checklist items are satisfied
                completed_items = context.get('completed_checklist_items', [])
                required_items = sop_answer.checklist_items
                
                missing_items = [item for item in required_items if item not in completed_items]
                if missing_items:
                    return False
        
        return True
    
    def _generate_sop_recommendations(self, sop_answers: List[SOPAnswer], context: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on SOP answers and current context."""
        recommendations = []
        
        for sop_answer in sop_answers:
            if sop_answer.checklist_items:
                completed_items = context.get('completed_checklist_items', [])
                missing_items = [item for item in sop_answer.checklist_items if item not in completed_items]
                
                for missing_item in missing_items:
                    recommendations.append(f"Complete required SOP item: {missing_item}")
        
        return recommendations
    
    def _get_stage_specific_guidance(self, sop_answers: List[SOPAnswer], context: Dict[str, Any]) -> Dict[str, Any]:
        """Get stage-specific SOP guidance."""
        current_stage = context.get('workflow_stage', 'unknown')
        
        stage_guidance = {
            'stage': current_stage,
            'sop_requirements': [],
            'checklist_items': [],
            'next_steps': []
        }
        
        for sop_answer in sop_answers:
            if sop_answer.workflow_stage == current_stage:
                stage_guidance['sop_requirements'].append(sop_answer.answer)
                if sop_answer.checklist_items:
                    stage_guidance['checklist_items'].extend(sop_answer.checklist_items)
        
        return stage_guidance
    
    def _fallback_validation(self, question: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fall back to general compliance validation when no SOP answers available."""
        # Use existing model risk and evidence validators
        document_text = context.get('document_text', '')
        
        if document_text:
            model_risk_result = self.model_risk_monitor.assess_document_compliance(document_text)
            evidence_result = self.evidence_validator.validate_document(document_text)
            
            return {
                'recommendations': [
                    "No specific SOP guidance found - using general compliance rules",
                    f"Model risk compliance: {model_risk_result.get('overall_score', 0):.1%}",
                    f"Evidence validation: {evidence_result.get('overall_validity', 'unknown')}"
                ],
                'stage_guidance': {
                    'stage': 'fallback',
                    'general_compliance': True,
                    'model_risk_score': model_risk_result.get('overall_score', 0),
                    'evidence_score': evidence_result.get('quality_score', 0)
                }
            }
        
        return {
            'recommendations': ["Insufficient context for validation"],
            'stage_guidance': {'stage': 'insufficient_context'}
        }
    
    def get_workflow_stage_requirements(self, stage: str) -> Dict[str, Any]:
        """Get all SOP requirements for a specific workflow stage."""
        stage_requirements = {
            'stage': stage,
            'sop_answers': [],
            'checklist_items': [],
            'compliance_rules': []
        }
        
        for answer_id, sop_answer in self.sop_answers.items():
            if sop_answer.workflow_stage == stage:
                stage_requirements['sop_answers'].append(sop_answer)
                if sop_answer.checklist_items:
                    stage_requirements['checklist_items'].extend(sop_answer.checklist_items)
        
        return stage_requirements
    
    def chat_with_sop(self, question: str, workflow_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Chat interface for SOP guidance during MVR analysis.
        
        This is the method that would be called from the Streamlit chat interface.
        """
        validation_result = self.validate_with_sop_precedence(question, workflow_context)
        
        # Format for chat interface
        chat_response = {
            'sop_guidance': validation_result.sop_answers[0].answer if validation_result.sop_answers else "No specific SOP guidance available",
            'compliance_status': validation_result.overall_compliance,
            'recommendations': validation_result.recommendations,
            'checklist_items': [],
            'confidence': validation_result.sop_score
        }
        
        # Add checklist items if available
        for sop_answer in validation_result.sop_answers:
            if sop_answer.checklist_items:
                chat_response['checklist_items'].extend(sop_answer.checklist_items)
        
        return chat_response
    
    def detect_and_resolve_sop_conflicts(self, queries: List[str] = None, authoritative_date: str = None) -> Dict[str, Any]:
        """
        Detect and resolve SOP conflicts using integrated conflict analysis system.
        
        This method integrates the SOP validator with the conflict detection system,
        providing compliance-owned conflict resolution with YRSN analysis.
        
        Args:
            queries: List of SOP queries to check for conflicts
            authoritative_date: Preferred authoritative date for guidance
            
        Returns:
            Comprehensive conflict analysis with SOP validation results
        """
        
        # Use default SOP validation queries if none provided
        if not queries:
            queries = [
                'What is the official session management pattern for TidyLLM?',
                'Which embedding system should be used: tidyllm-sentence or tidyllm-vectorqa?',
                'How should MVR documents be classified and tagged?',
                'What is the process for YRSN noise analysis validation?'
            ]
        
        # Generate conflict report using compliance-owned system
        conflict_reports = self.conflict_reporter.generate_compliance_report(queries, authoritative_date)
        
        # Validate conflict resolutions against SOP golden answers
        sop_validated_results = []
        
        for query in queries:
            # Get SOP golden answer if available
            sop_matches = self._query_sop_answers(query, {})
            
            # Get fallback guidance using compliance system
            fallback_result = self.fallback_strategy.retrieve_compliant_guidance(query, authoritative_date)
            
            # Perform YRSN analysis on both golden answers and fallback
            if sop_matches:
                golden_content = [match.answer for match in sop_matches]
                yrsn_golden = self.yrsn_analyzer.analyze_guidance_quality(golden_content, query)
                
                sop_validated_results.append({
                    'query': query,
                    'sop_golden_answer': sop_matches[0].answer,
                    'sop_source': sop_matches[0].sop_source,
                    'sop_confidence': sop_matches[0].confidence,
                    'sop_yrsn_score': yrsn_golden.noise_percentage,
                    'fallback_guidance': fallback_result.guidance_content[:3] if fallback_result.guidance_content else [],
                    'fallback_yrsn_score': self.yrsn_analyzer.analyze_guidance_quality(
                        fallback_result.guidance_content, query
                    ).noise_percentage if fallback_result.guidance_content else 100,
                    'recommendation': self._get_integrated_recommendation(sop_matches[0], fallback_result, yrsn_golden),
                    'compliance_status': 'GOLDEN_SOP_AVAILABLE'
                })
            else:
                # No golden answer - rely on fallback with YRSN validation
                yrsn_fallback = self.yrsn_analyzer.analyze_guidance_quality(
                    fallback_result.guidance_content, query
                ) if fallback_result.guidance_content else None
                
                sop_validated_results.append({
                    'query': query,
                    'sop_golden_answer': None,
                    'sop_source': None,
                    'sop_confidence': 0.0,
                    'sop_yrsn_score': 100.0,
                    'fallback_guidance': fallback_result.guidance_content[:3] if fallback_result.guidance_content else [],
                    'fallback_yrsn_score': yrsn_fallback.noise_percentage if yrsn_fallback else 100,
                    'recommendation': self._get_fallback_recommendation(fallback_result, yrsn_fallback),
                    'compliance_status': 'FALLBACK_GUIDANCE_ONLY'
                })
        
        # Combine conflict reports with SOP validation
        integrated_result = {
            'report_type': 'Integrated SOP Conflict Analysis',
            'timestamp': conflict_reports['compliance_summary']['generated_at'],
            'conflict_reports': conflict_reports,
            'sop_validation_results': sop_validated_results,
            'overall_compliance_status': self._determine_overall_compliance(sop_validated_results),
            'integrated_recommendations': self._generate_integrated_recommendations(sop_validated_results)
        }
        
        return integrated_result
    
    def _get_integrated_recommendation(self, sop_match: SOPAnswer, fallback_result, yrsn_score) -> str:
        """Generate recommendation when both SOP golden answer and fallback are available"""
        if yrsn_score.noise_percentage < 30:
            return f"EXCELLENT: Use SOP golden answer from {sop_match.sop_source} - high quality guidance available"
        elif yrsn_score.noise_percentage < 50:
            return f"GOOD: SOP guidance available from {sop_match.sop_source} - acceptable quality"
        else:
            return f"REVIEW: SOP guidance from {sop_match.sop_source} has {yrsn_score.noise_percentage:.1f}% noise - consider updating"
    
    def _get_fallback_recommendation(self, fallback_result, yrsn_score) -> str:
        """Generate recommendation when only fallback guidance is available"""
        if not fallback_result.guidance_content:
            return "CRITICAL: No guidance available - create SOP golden answer immediately"
        elif yrsn_score and yrsn_score.noise_percentage < 50:
            return f"ACCEPTABLE: Fallback guidance available with {yrsn_score.noise_percentage:.1f}% noise - consider creating SOP golden answer"
        else:
            return f"HIGH RISK: Fallback guidance has {yrsn_score.noise_percentage if yrsn_score else 100:.1f}% noise - create SOP golden answer urgently"
    
    def _determine_overall_compliance(self, results: List[Dict]) -> str:
        """Determine overall compliance status from integrated results"""
        golden_available = len([r for r in results if r['compliance_status'] == 'GOLDEN_SOP_AVAILABLE'])
        total_queries = len(results)
        
        if golden_available == total_queries:
            return 'FULL_SOP_COMPLIANCE'
        elif golden_available > total_queries * 0.5:
            return 'PARTIAL_SOP_COMPLIANCE'
        else:
            return 'FALLBACK_GUIDANCE_ONLY'
    
    def _generate_integrated_recommendations(self, results: List[Dict]) -> List[str]:
        """Generate integrated recommendations combining SOP and conflict analysis"""
        recommendations = []
        
        no_golden = [r for r in results if not r['sop_golden_answer']]
        high_noise = [r for r in results if r['sop_yrsn_score'] > 70 or r['fallback_yrsn_score'] > 70]
        
        if no_golden:
            recommendations.append(f"HIGH PRIORITY: Create SOP golden answers for {len(no_golden)} queries without specific guidance")
            
        if high_noise:
            recommendations.append(f"MEDIUM PRIORITY: Review and improve {len(high_noise)} high-noise guidance responses")
            
        recommendations.append("ONGOING: Integrate YRSN analysis into CI/CD pipeline for continuous compliance monitoring")
        
        return recommendations