#!/usr/bin/env python3
"""
QA Session Manager
Manages QA review sessions and tracking
"""

import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

class QASessionStatus(Enum):
    """QA session status enumeration"""
    INITIALIZED = "initialized"
    IN_PROGRESS = "in_progress"
    ASSESSMENT_COMPLETE = "assessment_complete"
    REPORT_COMPLETE = "report_complete"
    COMPLETED = "completed"
    FAILED = "failed"

class QAFindingStatus(Enum):
    """QA finding status enumeration"""
    PASSED = "passed"
    FAILED = "failed"
    PARTIAL = "partial"
    NOT_APPLICABLE = "not_applicable"
    PENDING = "pending"

class QAFindingSeverity(Enum):
    """QA finding severity enumeration"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"

@dataclass
class QAFinding:
    """Individual QA finding"""
    finding_id: str
    checklist_item: str
    status: QAFindingStatus
    severity: QAFindingSeverity
    description: str
    evidence: str
    recommendation: str
    action_required: bool
    assigned_to: str
    due_date: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def update_status(self, new_status: QAFindingStatus, evidence: str = None):
        """Update finding status"""
        self.status = new_status
        if evidence:
            self.evidence = evidence
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'finding_id': self.finding_id,
            'checklist_item': self.checklist_item,
            'status': self.status.value,
            'severity': self.severity.value,
            'description': self.description,
            'evidence': self.evidence,
            'recommendation': self.recommendation,
            'action_required': self.action_required,
            'assigned_to': self.assigned_to,
            'due_date': self.due_date.isoformat() if self.due_date else None,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

@dataclass
class ChecklistResult:
    """Result of applying a QA checklist"""
    checklist_id: str
    checklist_name: str
    stage: str
    checklist_type: str
    total_items: int
    completed_items: int
    passed_items: int
    failed_items: int
    partial_items: int
    not_applicable_items: int
    completion_rate: float
    quality_score: float
    risk_score: float
    findings: List[QAFinding]
    recommendations: List[str]
    created_at: datetime = field(default_factory=datetime.now)
    
    def calculate_scores(self):
        """Calculate quality and risk scores"""
        self.completed_items = self.passed_items + self.failed_items + self.partial_items
        
        if self.total_items > 0:
            self.completion_rate = self.completed_items / self.total_items
        else:
            self.completion_rate = 0.0
        
        if self.completed_items > 0:
            self.quality_score = self.passed_items / self.completed_items
        else:
            self.quality_score = 0.0
        
        self.risk_score = self.failed_items / self.total_items if self.total_items > 0 else 0.0
    
    def add_finding(self, finding: QAFinding):
        """Add a finding to the checklist result"""
        self.findings.append(finding)
        
        # Update counts based on finding status
        if finding.status == QAFindingStatus.PASSED:
            self.passed_items += 1
        elif finding.status == QAFindingStatus.FAILED:
            self.failed_items += 1
        elif finding.status == QAFindingStatus.PARTIAL:
            self.partial_items += 1
        elif finding.status == QAFindingStatus.NOT_APPLICABLE:
            self.not_applicable_items += 1
        
        # Recalculate scores
        self.calculate_scores()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'checklist_id': self.checklist_id,
            'checklist_name': self.checklist_name,
            'stage': self.stage,
            'checklist_type': self.checklist_type,
            'total_items': self.total_items,
            'completed_items': self.completed_items,
            'passed_items': self.passed_items,
            'failed_items': self.failed_items,
            'partial_items': self.partial_items,
            'not_applicable_items': self.not_applicable_items,
            'completion_rate': self.completion_rate,
            'quality_score': self.quality_score,
            'risk_score': self.risk_score,
            'findings': [finding.to_dict() for finding in self.findings],
            'recommendations': self.recommendations,
            'created_at': self.created_at.isoformat()
        }

@dataclass
class QAAssessment:
    """QA assessment results"""
    assessment_id: str
    session_id: str
    overall_quality_score: float
    overall_compliance_score: float
    overall_risk_score: float
    stage_scores: Dict[str, float]
    category_scores: Dict[str, float]
    critical_findings: List[QAFinding]
    high_priority_findings: List[QAFinding]
    recommendations: List[str]
    action_items: List[str]
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'assessment_id': self.assessment_id,
            'session_id': self.session_id,
            'overall_quality_score': self.overall_quality_score,
            'overall_compliance_score': self.overall_compliance_score,
            'overall_risk_score': self.overall_risk_score,
            'stage_scores': self.stage_scores,
            'category_scores': self.category_scores,
            'critical_findings': [finding.to_dict() for finding in self.critical_findings],
            'high_priority_findings': [finding.to_dict() for finding in self.high_priority_findings],
            'recommendations': self.recommendations,
            'action_items': self.action_items,
            'created_at': self.created_at.isoformat()
        }

@dataclass
class QAReport:
    """QA report structure"""
    report_id: str
    session_id: str
    assessment: QAAssessment
    executive_summary: str
    detailed_analysis: Dict[str, Any]
    quality_metrics: Dict[str, Any]
    compliance_assessment: Dict[str, Any]
    recommendations: List[str]
    action_plan: List[str]
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'report_id': self.report_id,
            'session_id': self.session_id,
            'assessment': self.assessment.to_dict(),
            'executive_summary': self.executive_summary,
            'detailed_analysis': self.detailed_analysis,
            'quality_metrics': self.quality_metrics,
            'compliance_assessment': self.compliance_assessment,
            'recommendations': self.recommendations,
            'action_plan': self.action_plan,
            'created_at': self.created_at.isoformat()
        }

@dataclass
class QASession:
    """QA review session for tracking and management"""
    session_id: str
    model_context: Dict[str, Any]
    review_scope: List[str]
    quality_criteria: Dict[str, Any]
    start_time: datetime
    checklist_results: List[ChecklistResult] = field(default_factory=list)
    qa_assessment: Optional[QAAssessment] = None
    qa_report: Optional[QAReport] = None
    status: QASessionStatus = QASessionStatus.INITIALIZED
    end_time: Optional[datetime] = None
    
    def add_checklist_result(self, result: ChecklistResult):
        """Add checklist result to session"""
        self.checklist_results.append(result)
        self.status = QASessionStatus.IN_PROGRESS
    
    def complete_assessment(self, assessment: QAAssessment):
        """Complete QA assessment"""
        self.qa_assessment = assessment
        self.status = QASessionStatus.ASSESSMENT_COMPLETE
    
    def complete_report(self, report: QAReport):
        """Complete QA report"""
        self.qa_report = report
        self.status = QASessionStatus.REPORT_COMPLETE
    
    def finalize_session(self):
        """Finalize the QA session"""
        self.status = QASessionStatus.COMPLETED
        self.end_time = datetime.now()
    
    def get_session_duration(self) -> float:
        """Get session duration in seconds"""
        end_time = self.end_time or datetime.now()
        return (end_time - self.start_time).total_seconds()
    
    def get_total_findings(self) -> int:
        """Get total number of findings across all checklists"""
        total = 0
        for result in self.checklist_results:
            total += len(result.findings)
        return total
    
    def get_critical_findings(self) -> List[QAFinding]:
        """Get all critical findings from the session"""
        critical_findings = []
        for result in self.checklist_results:
            for finding in result.findings:
                if finding.severity == QAFindingSeverity.CRITICAL:
                    critical_findings.append(finding)
        return critical_findings
    
    def get_high_priority_findings(self) -> List[QAFinding]:
        """Get all high priority findings from the session"""
        high_findings = []
        for result in self.checklist_results:
            for finding in result.findings:
                if finding.severity in [QAFindingSeverity.CRITICAL, QAFindingSeverity.HIGH]:
                    high_findings.append(finding)
        return high_findings
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'session_id': self.session_id,
            'model_context': self.model_context,
            'review_scope': self.review_scope,
            'quality_criteria': self.quality_criteria,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'status': self.status.value,
            'checklist_results': [result.to_dict() for result in self.checklist_results],
            'qa_assessment': self.qa_assessment.to_dict() if self.qa_assessment else None,
            'qa_report': self.qa_report.to_dict() if self.qa_report else None,
            'session_duration': self.get_session_duration(),
            'total_findings': self.get_total_findings(),
            'critical_findings_count': len(self.get_critical_findings()),
            'high_priority_findings_count': len(self.get_high_priority_findings())
        }

class QASessionManager:
    """Manager for QA sessions"""
    
    def __init__(self):
        self.sessions: Dict[str, QASession] = {}
    
    def create_session(self, model_context: Dict[str, Any], 
                      review_scope: List[str], 
                      quality_criteria: Dict[str, Any]) -> QASession:
        """Create a new QA session"""
        session_id = f"QA-SESSION-{uuid.uuid4().hex[:8].upper()}"
        
        session = QASession(
            session_id=session_id,
            model_context=model_context,
            review_scope=review_scope,
            quality_criteria=quality_criteria,
            start_time=datetime.now()
        )
        
        self.sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[QASession]:
        """Get session by ID"""
        return self.sessions.get(session_id)
    
    def update_session(self, session: QASession):
        """Update session in manager"""
        self.sessions[session.session_id] = session
    
    def list_sessions(self) -> List[QASession]:
        """List all sessions"""
        return list(self.sessions.values())
    
    def get_active_sessions(self) -> List[QASession]:
        """Get all active sessions"""
        return [session for session in self.sessions.values() 
                if session.status != QASessionStatus.COMPLETED]
    
    def get_completed_sessions(self) -> List[QASession]:
        """Get all completed sessions"""
        return [session for session in self.sessions.values() 
                if session.status == QASessionStatus.COMPLETED]
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get statistics for all sessions"""
        total_sessions = len(self.sessions)
        active_sessions = len(self.get_active_sessions())
        completed_sessions = len(self.get_completed_sessions())
        
        total_findings = sum(session.get_total_findings() for session in self.sessions.values())
        critical_findings = sum(len(session.get_critical_findings()) for session in self.sessions.values())
        high_priority_findings = sum(len(session.get_high_priority_findings()) for session in self.sessions.values())
        
        avg_session_duration = 0
        if total_sessions > 0:
            avg_session_duration = sum(session.get_session_duration() for session in self.sessions.values()) / total_sessions
        
        return {
            'total_sessions': total_sessions,
            'active_sessions': active_sessions,
            'completed_sessions': completed_sessions,
            'total_findings': total_findings,
            'critical_findings': critical_findings,
            'high_priority_findings': high_priority_findings,
            'average_session_duration_seconds': avg_session_duration
        }
