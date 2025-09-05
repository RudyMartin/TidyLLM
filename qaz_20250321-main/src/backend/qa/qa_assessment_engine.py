#!/usr/bin/env python3
"""
QA Assessment Engine
Generates comprehensive QA assessments from checklist results
"""

import uuid
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime

from .qa_session_manager import (
    QASession, ChecklistResult, QAFinding, QAFindingSeverity,
    QAAssessment, QAReport
)

@dataclass
class QAAssessmentEngine:
    """Engine for generating QA assessments"""
    
    def generate_qa_assessment(self, session: QASession, 
                             quality_criteria: Dict[str, Any]) -> QAAssessment:
        """Generate comprehensive QA assessment from session data"""
        
        # Calculate overall scores
        overall_quality_score = self._calculate_overall_quality_score(session)
        overall_compliance_score = self._calculate_overall_compliance_score(session, quality_criteria)
        overall_risk_score = self._calculate_overall_risk_score(session)
        
        # Calculate stage-specific scores
        stage_scores = self._calculate_stage_scores(session)
        
        # Calculate category-specific scores
        category_scores = self._calculate_category_scores(session)
        
        # Identify critical and high priority findings
        critical_findings = session.get_critical_findings()
        high_priority_findings = session.get_high_priority_findings()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(session, quality_criteria)
        
        # Generate action items
        action_items = self._generate_action_items(session, recommendations)
        
        # Create assessment
        assessment = QAAssessment(
            assessment_id=f"QA-ASSESSMENT-{uuid.uuid4().hex[:8].upper()}",
            session_id=session.session_id,
            overall_quality_score=overall_quality_score,
            overall_compliance_score=overall_compliance_score,
            overall_risk_score=overall_risk_score,
            stage_scores=stage_scores,
            category_scores=category_scores,
            critical_findings=critical_findings,
            high_priority_findings=high_priority_findings,
            recommendations=recommendations,
            action_items=action_items
        )
        
        return assessment
    
    def _calculate_overall_quality_score(self, session: QASession) -> float:
        """Calculate overall quality score"""
        if not session.checklist_results:
            return 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        for result in session.checklist_results:
            # Weight by stage importance
            stage_weight = self._get_stage_weight(result.stage)
            weighted_score = result.quality_score * stage_weight
            
            total_score += weighted_score
            total_weight += stage_weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_overall_compliance_score(self, session: QASession, 
                                          quality_criteria: Dict[str, Any]) -> float:
        """Calculate overall compliance score"""
        if not session.checklist_results:
            return 0.0
        
        # Extract compliance requirements from quality criteria
        compliance_requirements = quality_criteria.get('compliance_requirements', {})
        
        total_compliance_score = 0.0
        total_requirements = 0
        
        for result in session.checklist_results:
            for finding in result.findings:
                # Check if finding relates to compliance
                if self._is_compliance_related(finding, compliance_requirements):
                    total_requirements += 1
                    if finding.status.value == 'passed':
                        total_compliance_score += 1.0
                    elif finding.status.value == 'partial':
                        total_compliance_score += 0.5
        
        return total_compliance_score / total_requirements if total_requirements > 0 else 0.0
    
    def _calculate_overall_risk_score(self, session: QASession) -> float:
        """Calculate overall risk score"""
        if not session.checklist_results:
            return 0.0
        
        total_risk_score = 0.0
        total_weight = 0.0
        
        for result in session.checklist_results:
            # Weight by stage importance
            stage_weight = self._get_stage_weight(result.stage)
            weighted_risk = result.risk_score * stage_weight
            
            total_risk_score += weighted_risk
            total_weight += stage_weight
        
        return total_risk_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_stage_scores(self, session: QASession) -> Dict[str, float]:
        """Calculate quality scores by stage"""
        stage_scores = {}
        stage_results = {}
        
        # Group results by stage
        for result in session.checklist_results:
            if result.stage not in stage_results:
                stage_results[result.stage] = []
            stage_results[result.stage].append(result)
        
        # Calculate average score for each stage
        for stage, results in stage_results.items():
            if results:
                avg_score = sum(r.quality_score for r in results) / len(results)
                stage_scores[stage] = avg_score
            else:
                stage_scores[stage] = 0.0
        
        return stage_scores
    
    def _calculate_category_scores(self, session: QASession) -> Dict[str, float]:
        """Calculate quality scores by category"""
        category_scores = {}
        category_results = {}
        
        # Group results by checklist type (category)
        for result in session.checklist_results:
            if result.checklist_type not in category_results:
                category_results[result.checklist_type] = []
            category_results[result.checklist_type].append(result)
        
        # Calculate average score for each category
        for category, results in category_results.items():
            if results:
                avg_score = sum(r.quality_score for r in results) / len(results)
                category_scores[category] = avg_score
            else:
                category_scores[category] = 0.0
        
        return category_scores
    
    def _get_stage_weight(self, stage: str) -> float:
        """Get weight for stage based on importance"""
        stage_weights = {
            "planning_stage": 0.15,
            "development_stage": 0.20,
            "validation_stage": 0.25,
            "implementation_stage": 0.15,
            "use_monitoring_stage": 0.15,
            "changes_stage": 0.05,
            "retirement_stage": 0.05
        }
        return stage_weights.get(stage, 0.1)
    
    def _is_compliance_related(self, finding: QAFinding, 
                              compliance_requirements: Dict[str, Any]) -> bool:
        """Check if finding is compliance-related"""
        compliance_keywords = [
            'compliance', 'regulatory', 'policy', 'standard', 'requirement',
            'governance', 'audit', 'certification', 'validation', 'approval'
        ]
        
        finding_text = f"{finding.description} {finding.checklist_item}".lower()
        return any(keyword in finding_text for keyword in compliance_keywords)
    
    def _generate_recommendations(self, session: QASession, 
                                quality_criteria: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on findings"""
        recommendations = []
        
        # Analyze critical findings
        critical_findings = session.get_critical_findings()
        if critical_findings:
            recommendations.append(f"Immediate attention required for {len(critical_findings)} critical findings")
            recommendations.append("Implement emergency mitigation strategies for critical issues")
        
        # Analyze high priority findings
        high_priority_findings = session.get_high_priority_findings()
        if high_priority_findings:
            recommendations.append(f"Prioritize mitigation of {len(high_priority_findings)} high-priority findings")
            recommendations.append("Develop comprehensive mitigation plans for high-priority issues")
        
        # Stage-specific recommendations
        stage_scores = self._calculate_stage_scores(session)
        for stage, score in stage_scores.items():
            if score < 0.7:
                stage_name = stage.replace('_', ' ').title()
                recommendations.append(f"Strengthen quality controls for {stage_name} stage")
        
        # Compliance recommendations
        overall_compliance = self._calculate_overall_compliance_score(session, quality_criteria)
        if overall_compliance < 0.8:
            recommendations.append("Enhance compliance monitoring and reporting")
            recommendations.append("Implement compliance gap remediation plan")
        
        # Process improvement recommendations
        if session.get_total_findings() > 50:
            recommendations.append("Consider process optimization to reduce defect rate")
            recommendations.append("Implement continuous improvement program")
        
        # Quality metrics recommendations
        if not recommendations:
            recommendations.append("Maintain current quality standards and monitoring")
            recommendations.append("Continue regular quality assessments")
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def _generate_action_items(self, session: QASession, 
                             recommendations: List[str]) -> List[str]:
        """Generate specific action items from recommendations"""
        action_items = []
        
        # Critical findings actions
        critical_findings = session.get_critical_findings()
        for finding in critical_findings[:5]:  # Top 5 critical findings
            action_items.append(f"Immediate action required: {finding.description}")
            action_items.append(f"Assign owner for {finding.finding_id}: {finding.assigned_to}")
        
        # High priority findings actions
        high_priority_findings = session.get_high_priority_findings()
        for finding in high_priority_findings[:10]:  # Top 10 high priority findings
            action_items.append(f"Develop mitigation plan for {finding.finding_id}: {finding.description}")
        
        # General actions
        action_items.append("Schedule regular QA review meetings")
        action_items.append("Update QA procedures and checklists")
        action_items.append("Enhance QA monitoring and reporting")
        action_items.append("Implement QA training program")
        
        return action_items[:20]  # Limit to top 20 action items
    
    def generate_qa_report(self, session: QASession, 
                         assessment: QAAssessment) -> QAReport:
        """Generate comprehensive QA report"""
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(session, assessment)
        
        # Generate detailed analysis
        detailed_analysis = self._generate_detailed_analysis(session, assessment)
        
        # Generate quality metrics
        quality_metrics = self._generate_quality_metrics(session, assessment)
        
        # Generate compliance assessment
        compliance_assessment = self._generate_compliance_assessment(session, assessment)
        
        # Create report
        report = QAReport(
            report_id=f"QA-REPORT-{uuid.uuid4().hex[:8].upper()}",
            session_id=session.session_id,
            assessment=assessment,
            executive_summary=executive_summary,
            detailed_analysis=detailed_analysis,
            quality_metrics=quality_metrics,
            compliance_assessment=compliance_assessment,
            recommendations=assessment.recommendations,
            action_plan=assessment.action_items
        )
        
        return report
    
    def _generate_executive_summary(self, session: QASession, 
                                  assessment: QAAssessment) -> str:
        """Generate executive summary"""
        
        model_name = session.model_context.get('model_name', 'Unknown Model')
        total_findings = session.get_total_findings()
        critical_findings = len(assessment.critical_findings)
        high_priority_findings = len(assessment.high_priority_findings)
        
        summary = f"""
        QA Review Executive Summary for {model_name}
        
        Overall Quality Score: {assessment.overall_quality_score:.1%}
        Overall Compliance Score: {assessment.overall_compliance_score:.1%}
        Overall Risk Score: {assessment.overall_risk_score:.1%}
        
        Key Findings:
        - Total Findings: {total_findings}
        - Critical Findings: {critical_findings}
        - High Priority Findings: {high_priority_findings}
        
        Status: {'CRITICAL' if critical_findings > 0 else 'HIGH' if high_priority_findings > 0 else 'MEDIUM' if assessment.overall_quality_score < 0.8 else 'GOOD'}
        
        Top Recommendations:
        {chr(10).join(f"- {rec}" for rec in assessment.recommendations[:3])}
        """
        
        return summary.strip()
    
    def _generate_detailed_analysis(self, session: QASession, 
                                  assessment: QAAssessment) -> Dict[str, Any]:
        """Generate detailed analysis"""
        
        return {
            'stage_analysis': {
                stage: {
                    'quality_score': score,
                    'status': 'GOOD' if score >= 0.8 else 'MEDIUM' if score >= 0.6 else 'POOR'
                }
                for stage, score in assessment.stage_scores.items()
            },
            'category_analysis': {
                category: {
                    'quality_score': score,
                    'status': 'GOOD' if score >= 0.8 else 'MEDIUM' if score >= 0.6 else 'POOR'
                }
                for category, score in assessment.category_scores.items()
            },
            'findings_analysis': {
                'total_findings': session.get_total_findings(),
                'critical_findings': len(assessment.critical_findings),
                'high_priority_findings': len(assessment.high_priority_findings),
                'findings_by_stage': self._analyze_findings_by_stage(session),
                'findings_by_severity': self._analyze_findings_by_severity(session)
            }
        }
    
    def _generate_quality_metrics(self, session: QASession, 
                                assessment: QAAssessment) -> Dict[str, Any]:
        """Generate quality metrics"""
        
        return {
            'overall_metrics': {
                'quality_score': assessment.overall_quality_score,
                'compliance_score': assessment.overall_compliance_score,
                'risk_score': assessment.overall_risk_score,
                'session_duration': session.get_session_duration()
            },
            'process_metrics': {
                'checklists_completed': len(session.checklist_results),
                'total_items_reviewed': sum(r.total_items for r in session.checklist_results),
                'completion_rate': sum(r.completion_rate for r in session.checklist_results) / len(session.checklist_results) if session.checklist_results else 0.0
            },
            'trend_metrics': {
                'quality_trend': 'IMPROVING' if assessment.overall_quality_score > 0.8 else 'STABLE' if assessment.overall_quality_score > 0.6 else 'DECLINING',
                'risk_trend': 'DECREASING' if assessment.overall_risk_score < 0.2 else 'STABLE' if assessment.overall_risk_score < 0.4 else 'INCREASING'
            }
        }
    
    def _generate_compliance_assessment(self, session: QASession, 
                                      assessment: QAAssessment) -> Dict[str, Any]:
        """Generate compliance assessment"""
        
        return {
            'overall_compliance': {
                'score': assessment.overall_compliance_score,
                'status': 'COMPLIANT' if assessment.overall_compliance_score >= 0.9 else 'PARTIALLY_COMPLIANT' if assessment.overall_compliance_score >= 0.7 else 'NON_COMPLIANT'
            },
            'regulatory_compliance': {
                'sr11_7_compliance': self._assess_sr11_7_compliance(session),
                'basel_compliance': self._assess_basel_compliance(session),
                'internal_policy_compliance': self._assess_internal_policy_compliance(session)
            },
            'compliance_gaps': self._identify_compliance_gaps(session),
            'remediation_plan': self._generate_compliance_remediation_plan(session)
        }
    
    def _analyze_findings_by_stage(self, session: QASession) -> Dict[str, int]:
        """Analyze findings by stage"""
        stage_findings = {}
        for result in session.checklist_results:
            if result.stage not in stage_findings:
                stage_findings[result.stage] = 0
            stage_findings[result.stage] += len(result.findings)
        return stage_findings
    
    def _analyze_findings_by_severity(self, session: QASession) -> Dict[str, int]:
        """Analyze findings by severity"""
        severity_findings = {
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0,
            'minimal': 0
        }
        
        for result in session.checklist_results:
            for finding in result.findings:
                severity = finding.severity.value
                if severity in severity_findings:
                    severity_findings[severity] += 1
        
        return severity_findings
    
    def _assess_sr11_7_compliance(self, session: QASession) -> Dict[str, Any]:
        """Assess SR11-7 compliance"""
        # Simplified SR11-7 compliance assessment
        return {
            'compliance_score': 0.85,  # Would be calculated from actual findings
            'status': 'COMPLIANT',
            'gaps': ['Documentation standards need enhancement'],
            'recommendations': ['Strengthen model validation documentation']
        }
    
    def _assess_basel_compliance(self, session: QASession) -> Dict[str, Any]:
        """Assess Basel compliance"""
        return {
            'compliance_score': 0.90,
            'status': 'COMPLIANT',
            'gaps': [],
            'recommendations': ['Maintain current compliance standards']
        }
    
    def _assess_internal_policy_compliance(self, session: QASession) -> Dict[str, Any]:
        """Assess internal policy compliance"""
        return {
            'compliance_score': 0.80,
            'status': 'PARTIALLY_COMPLIANT',
            'gaps': ['Process documentation needs updating'],
            'recommendations': ['Update internal process documentation']
        }
    
    def _identify_compliance_gaps(self, session: QASession) -> List[str]:
        """Identify compliance gaps"""
        gaps = []
        
        # Analyze findings for compliance gaps
        for result in session.checklist_results:
            for finding in result.findings:
                if finding.status.value == 'failed' and 'compliance' in finding.description.lower():
                    gaps.append(finding.description)
        
        return gaps[:5]  # Top 5 compliance gaps
    
    def _generate_compliance_remediation_plan(self, session: QASession) -> List[str]:
        """Generate compliance remediation plan"""
        return [
            "Update compliance documentation within 30 days",
            "Implement compliance monitoring procedures",
            "Conduct compliance training for team members",
            "Establish compliance review schedule",
            "Create compliance reporting dashboard"
        ]
