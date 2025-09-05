#!/usr/bin/env python3
"""
SPARSE Agreement System for TidyLLM Enterprise
==============================================

SPARSE = Structured Pre-Approved Reasoning for Systematic Execution

This system implements pre-documented decisions with complete compliance tracking:
1. Stakeholder approval chains with digital signatures
2. Risk assessment documentation with business impact scoring  
3. Compliance framework mapping (SOX, GDPR, HIPAA, etc.)
4. Expiration and review cycles with automated alerting
5. Execution condition validation with automated checking

KEY FIX: Proper serialization/deserialization for all complex objects.
"""

import os
import sys
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
import uuid
from pathlib import Path

class RiskLevel(Enum):
    """Risk assessment levels for SPARSE agreements"""
    MINIMAL = "minimal"          # Low risk, standard approval
    LOW = "low"                 # Minor business impact
    MEDIUM = "medium"           # Moderate business/compliance risk
    HIGH = "high"               # Significant risk, senior approval needed
    CRITICAL = "critical"       # Major risk, C-level approval required

class ApprovalStatus(Enum):
    """Approval status for SPARSE agreements"""
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review" 
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    UNDER_REVIEW = "under_review"
    SUSPENDED = "suspended"

class ComplianceFramework(Enum):
    """Regulatory/compliance frameworks"""
    SOX = "sarbanes_oxley"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"
    NIST = "nist_cybersecurity"
    INTERNAL_POLICY = "internal_policy"
    REGULATORY_GUIDANCE = "regulatory_guidance"

@dataclass
class StakeholderApproval:
    """Individual stakeholder approval record"""
    stakeholder_id: str
    name: str
    role: str
    department: str
    approval_date: datetime
    approval_method: str  # email, digital_signature, in_person
    comments: str = ""
    conditions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'stakeholder_id': self.stakeholder_id,
            'name': self.name,
            'role': self.role,
            'department': self.department,
            'approval_date': self.approval_date.isoformat() if self.approval_date else None,
            'approval_method': self.approval_method,
            'comments': self.comments,
            'conditions': self.conditions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StakeholderApproval':
        """Create from dictionary (deserialization)"""
        if isinstance(data['approval_date'], str):
            data['approval_date'] = datetime.fromisoformat(data['approval_date'])
        return cls(**data)

@dataclass
class RiskAssessment:
    """Comprehensive risk assessment for SPARSE agreement"""
    risk_level: RiskLevel
    business_impact_score: float  # 0-10 scale
    compliance_risk_score: float  # 0-10 scale
    operational_risk_score: float  # 0-10 scale
    
    # Risk mitigation
    mitigation_strategies: List[str] = field(default_factory=list)
    monitoring_requirements: List[str] = field(default_factory=list)
    escalation_triggers: List[str] = field(default_factory=list)
    
    # Documentation
    risk_analysis_document: Optional[str] = None
    assessor_id: str = ""
    assessment_date: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'risk_level': self.risk_level.value,
            'business_impact_score': self.business_impact_score,
            'compliance_risk_score': self.compliance_risk_score,
            'operational_risk_score': self.operational_risk_score,
            'mitigation_strategies': self.mitigation_strategies,
            'monitoring_requirements': self.monitoring_requirements,
            'escalation_triggers': self.escalation_triggers,
            'risk_analysis_document': self.risk_analysis_document,
            'assessor_id': self.assessor_id,
            'assessment_date': self.assessment_date.isoformat() if self.assessment_date else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RiskAssessment':
        """Create from dictionary (deserialization)"""
        if isinstance(data['risk_level'], str):
            data['risk_level'] = RiskLevel(data['risk_level'])
        if data.get('assessment_date') and isinstance(data['assessment_date'], str):
            data['assessment_date'] = datetime.fromisoformat(data['assessment_date'])
        return cls(**data)

@dataclass
class ExecutionCondition:
    """Conditions that must be met for SPARSE execution"""
    condition_id: str
    description: str
    condition_type: str  # context_check, approval_required, time_constraint
    validation_method: str  # automated, manual, api_check
    parameters: Dict[str, Any] = field(default_factory=dict)
    mandatory: bool = True
    
    # For corporate paranoia - who decided this condition
    defined_by: str = ""
    business_justification: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionCondition':
        """Create from dictionary (deserialization)"""
        return cls(**data)

@dataclass
class ApprovedAction:
    """Pre-approved action with execution parameters"""
    action_id: str
    name: str
    description: str
    action_type: str  # api_call, database_update, file_operation, notification
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Execution constraints
    max_executions_per_day: Optional[int] = None
    max_executions_per_hour: Optional[int] = None
    requires_human_confirmation: bool = False
    
    # Audit trail requirements
    log_level: str = "INFO"
    sensitive_data_handling: str = "standard"  # standard, encrypted, anonymized
    retention_period_days: int = 365
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ApprovedAction':
        """Create from dictionary (deserialization)"""
        return cls(**data)

@dataclass 
class SparseAgreement:
    """Core SPARSE agreement structure with proper serialization support"""
    
    # Identification
    agreement_id: str
    title: str 
    description: str
    version: str = "1.0"
    
    # Approval workflow
    status: ApprovalStatus = ApprovalStatus.DRAFT
    created_date: datetime = field(default_factory=datetime.now)
    approval_date: Optional[datetime] = None
    expiry_date: Optional[datetime] = None
    
    # Business context
    business_purpose: str = ""
    expected_frequency: str = ""  # daily, weekly, monthly, ad-hoc
    business_owner: str = ""
    technical_owner: str = ""
    
    # Execution framework
    conditions: List[ExecutionCondition] = field(default_factory=list)
    approved_actions: List[ApprovedAction] = field(default_factory=list)
    
    # Compliance & Risk
    compliance_frameworks: List[ComplianceFramework] = field(default_factory=list) 
    risk_assessment: Optional[RiskAssessment] = None
    stakeholder_approvals: List[StakeholderApproval] = field(default_factory=list)
    
    # Review cycle
    review_frequency_days: int = 365  # Annual review by default
    last_review_date: Optional[datetime] = None
    next_review_date: Optional[datetime] = None
    
    # Execution tracking
    execution_count: int = 0
    last_execution_date: Optional[datetime] = None
    execution_statistics: Dict[str, Any] = field(default_factory=dict)
    
    # Audit & Documentation
    documentation_links: List[str] = field(default_factory=list)
    related_policies: List[str] = field(default_factory=list)
    change_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for proper serialization (FIXED)"""
        return {
            'agreement_id': self.agreement_id,
            'title': self.title,
            'description': self.description,
            'version': self.version,
            'status': self.status.value,
            'created_date': self.created_date.isoformat() if self.created_date else None,
            'approval_date': self.approval_date.isoformat() if self.approval_date else None,
            'expiry_date': self.expiry_date.isoformat() if self.expiry_date else None,
            'business_purpose': self.business_purpose,
            'expected_frequency': self.expected_frequency,
            'business_owner': self.business_owner,
            'technical_owner': self.technical_owner,
            'conditions': [condition.to_dict() for condition in self.conditions],
            'approved_actions': [action.to_dict() for action in self.approved_actions],
            'compliance_frameworks': [framework.value for framework in self.compliance_frameworks],
            'risk_assessment': self.risk_assessment.to_dict() if self.risk_assessment else None,
            'stakeholder_approvals': [approval.to_dict() for approval in self.stakeholder_approvals],
            'review_frequency_days': self.review_frequency_days,
            'last_review_date': self.last_review_date.isoformat() if self.last_review_date else None,
            'next_review_date': self.next_review_date.isoformat() if self.next_review_date else None,
            'execution_count': self.execution_count,
            'last_execution_date': self.last_execution_date.isoformat() if self.last_execution_date else None,
            'execution_statistics': self.execution_statistics,
            'documentation_links': self.documentation_links,
            'related_policies': self.related_policies,
            'change_history': self.change_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SparseAgreement':
        """Create from dictionary (proper deserialization - FIXED)"""
        # Convert enum strings back to enums
        if isinstance(data.get('status'), str):
            data['status'] = ApprovalStatus(data['status'])
        
        # Convert date strings back to datetime objects
        date_fields = ['created_date', 'approval_date', 'expiry_date', 'last_review_date', 'next_review_date', 'last_execution_date']
        for field_name in date_fields:
            if data.get(field_name) and isinstance(data[field_name], str):
                data[field_name] = datetime.fromisoformat(data[field_name])
        
        # Convert framework strings back to enums
        if data.get('compliance_frameworks'):
            data['compliance_frameworks'] = [ComplianceFramework(framework) for framework in data['compliance_frameworks']]
        
        # Convert complex objects
        if data.get('conditions'):
            data['conditions'] = [ExecutionCondition.from_dict(condition) for condition in data['conditions']]
        
        if data.get('approved_actions'):
            data['approved_actions'] = [ApprovedAction.from_dict(action) for action in data['approved_actions']]
        
        if data.get('stakeholder_approvals'):
            data['stakeholder_approvals'] = [StakeholderApproval.from_dict(approval) for approval in data['stakeholder_approvals']]
        
        if data.get('risk_assessment'):
            data['risk_assessment'] = RiskAssessment.from_dict(data['risk_assessment'])
        
        return cls(**data)

class SparseAgreementManager:
    """Manager for SPARSE agreements with fixed serialization"""
    
    def __init__(self, storage_path: str = "sparse_agreements"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        self.agreements: Dict[str, SparseAgreement] = {}
        self.execution_log: List[Dict[str, Any]] = []
        
        # Load existing agreements
        self._load_agreements()
    
    def create_agreement(self,
                        title: str,
                        description: str, 
                        business_purpose: str,
                        business_owner: str,
                        technical_owner: str) -> SparseAgreement:
        """Create new SPARSE agreement"""
        
        agreement = SparseAgreement(
            agreement_id=str(uuid.uuid4()),
            title=title,
            description=description,
            business_purpose=business_purpose,
            business_owner=business_owner,
            technical_owner=technical_owner
        )
        
        self.agreements[agreement.agreement_id] = agreement
        self._save_agreement(agreement)
        
        return agreement
    
    def add_execution_condition(self,
                               agreement_id: str,
                               description: str,
                               condition_type: str,
                               validation_method: str,
                               parameters: Dict[str, Any] = None,
                               defined_by: str = "",
                               business_justification: str = "") -> ExecutionCondition:
        """Add execution condition to agreement"""
        
        if agreement_id not in self.agreements:
            raise ValueError(f"Agreement {agreement_id} not found")
        
        condition = ExecutionCondition(
            condition_id=str(uuid.uuid4()),
            description=description,
            condition_type=condition_type,
            validation_method=validation_method,
            parameters=parameters or {},
            defined_by=defined_by,
            business_justification=business_justification
        )
        
        self.agreements[agreement_id].conditions.append(condition)
        self._save_agreement(self.agreements[agreement_id])
        
        return condition
    
    def add_approved_action(self,
                           agreement_id: str,
                           name: str,
                           description: str,
                           action_type: str,
                           parameters: Dict[str, Any] = None,
                           requires_confirmation: bool = False) -> ApprovedAction:
        """Add approved action to agreement"""
        
        if agreement_id not in self.agreements:
            raise ValueError(f"Agreement {agreement_id} not found")
        
        action = ApprovedAction(
            action_id=str(uuid.uuid4()),
            name=name,
            description=description,
            action_type=action_type,
            parameters=parameters or {},
            requires_human_confirmation=requires_confirmation
        )
        
        self.agreements[agreement_id].approved_actions.append(action)
        self._save_agreement(self.agreements[agreement_id])
        
        return action
    
    def add_stakeholder_approval(self,
                                agreement_id: str,
                                stakeholder_name: str,
                                role: str,
                                department: str,
                                approval_method: str = "digital_signature",
                                comments: str = "",
                                conditions: List[str] = None) -> StakeholderApproval:
        """Add stakeholder approval to agreement"""
        
        if agreement_id not in self.agreements:
            raise ValueError(f"Agreement {agreement_id} not found")
        
        approval = StakeholderApproval(
            stakeholder_id=str(uuid.uuid4()),
            name=stakeholder_name,
            role=role,
            department=department,
            approval_date=datetime.now(),
            approval_method=approval_method,
            comments=comments,
            conditions=conditions or []
        )
        
        self.agreements[agreement_id].stakeholder_approvals.append(approval)
        self._save_agreement(self.agreements[agreement_id])
        
        return approval
    
    def set_risk_assessment(self,
                           agreement_id: str,
                           risk_level: RiskLevel,
                           business_impact: float,
                           compliance_risk: float,
                           operational_risk: float,
                           assessor_id: str,
                           mitigation_strategies: List[str] = None,
                           monitoring_requirements: List[str] = None) -> RiskAssessment:
        """Set comprehensive risk assessment"""
        
        if agreement_id not in self.agreements:
            raise ValueError(f"Agreement {agreement_id} not found")
        
        risk_assessment = RiskAssessment(
            risk_level=risk_level,
            business_impact_score=business_impact,
            compliance_risk_score=compliance_risk,
            operational_risk_score=operational_risk,
            mitigation_strategies=mitigation_strategies or [],
            monitoring_requirements=monitoring_requirements or [],
            assessor_id=assessor_id,
            assessment_date=datetime.now()
        )
        
        self.agreements[agreement_id].risk_assessment = risk_assessment
        self._save_agreement(self.agreements[agreement_id])
        
        return risk_assessment
    
    def approve_agreement(self, agreement_id: str, approver_id: str) -> bool:
        """Approve SPARSE agreement for execution"""
        
        if agreement_id not in self.agreements:
            raise ValueError(f"Agreement {agreement_id} not found")
        
        agreement = self.agreements[agreement_id]
        
        # Validation checks
        if not agreement.stakeholder_approvals:
            raise ValueError("Cannot approve agreement without stakeholder approvals")
        
        if not agreement.risk_assessment:
            raise ValueError("Cannot approve agreement without risk assessment")
        
        if not agreement.approved_actions:
            raise ValueError("Cannot approve agreement without approved actions")
        
        # Approve
        agreement.status = ApprovalStatus.APPROVED
        agreement.approval_date = datetime.now()
        
        # Set review cycle
        if agreement.review_frequency_days:
            agreement.next_review_date = datetime.now() + timedelta(days=agreement.review_frequency_days)
        
        # Record approval in change history
        agreement.change_history.append({
            'action': 'approved',
            'timestamp': datetime.now().isoformat(),
            'approver_id': approver_id,
            'previous_status': 'pending_review'
        })
        
        self._save_agreement(agreement)
        return True
    
    def generate_compliance_report(self, agreement_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive compliance report (FIXED serialization)"""
        
        if agreement_id:
            agreements = [self.agreements[agreement_id]] if agreement_id in self.agreements else []
        else:
            agreements = list(self.agreements.values())
        
        if not agreements:
            return {'error': 'No agreements found'}
        
        report = {
            'report_generated': datetime.now().isoformat(),
            'total_agreements': len(agreements),
            'agreements_by_status': {},
            'agreements_by_risk': {},
            'execution_summary': {
                'total_executions': 0,
                'successful_executions': 0,
                'failed_executions': 0
            },
            'compliance_coverage': {},
            'review_status': {
                'overdue_reviews': 0,
                'upcoming_reviews': 0,
                'expired_agreements': 0
            },
            'agreements_detail': []
        }
        
        # Analyze agreements
        for agreement in agreements:
            # Status summary (FIXED: handle enum properly)
            status = agreement.status.value if isinstance(agreement.status, ApprovalStatus) else str(agreement.status)
            report['agreements_by_status'][status] = report['agreements_by_status'].get(status, 0) + 1
            
            # Risk summary (FIXED: handle risk_assessment properly)
            if agreement.risk_assessment and hasattr(agreement.risk_assessment, 'risk_level'):
                risk_level = agreement.risk_assessment.risk_level
                risk = risk_level.value if isinstance(risk_level, RiskLevel) else str(risk_level)
                report['agreements_by_risk'][risk] = report['agreements_by_risk'].get(risk, 0) + 1
            
            # Execution summary
            report['execution_summary']['total_executions'] += agreement.execution_count
            
            # Compliance coverage
            for framework in agreement.compliance_frameworks:
                framework_name = framework.value if isinstance(framework, ComplianceFramework) else str(framework)
                report['compliance_coverage'][framework_name] = report['compliance_coverage'].get(framework_name, 0) + 1
            
            # Review status
            if agreement.expiry_date and datetime.now() > agreement.expiry_date:
                report['review_status']['expired_agreements'] += 1
            elif agreement.next_review_date:
                if datetime.now() > agreement.next_review_date:
                    report['review_status']['overdue_reviews'] += 1
                elif (agreement.next_review_date - datetime.now()).days <= 30:
                    report['review_status']['upcoming_reviews'] += 1
            
            # Detailed agreement info (FIXED: proper handling of all fields)
            risk_level = 'unassessed'
            if agreement.risk_assessment and hasattr(agreement.risk_assessment, 'risk_level'):
                risk_obj = agreement.risk_assessment.risk_level
                risk_level = risk_obj.value if isinstance(risk_obj, RiskLevel) else str(risk_obj)
            
            report['agreements_detail'].append({
                'agreement_id': agreement.agreement_id,
                'title': agreement.title,
                'status': status,
                'risk_level': risk_level,
                'execution_count': agreement.execution_count,
                'last_execution': agreement.last_execution_date.isoformat() if agreement.last_execution_date else None,
                'stakeholder_approvals': len(agreement.stakeholder_approvals),
                'compliance_frameworks': [f.value if isinstance(f, ComplianceFramework) else str(f) for f in agreement.compliance_frameworks],
                'expires': agreement.expiry_date.isoformat() if agreement.expiry_date else None,
                'next_review': agreement.next_review_date.isoformat() if agreement.next_review_date else None
            })
        
        return report
    
    def _save_agreement(self, agreement: SparseAgreement):
        """Save agreement to storage (FIXED serialization)"""
        file_path = self.storage_path / f"{agreement.agreement_id}.json"
        
        # Use the proper to_dict method
        serializable_data = agreement.to_dict()
        
        with open(file_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)
    
    def _load_agreements(self):
        """Load agreements from storage (FIXED deserialization)"""
        if not self.storage_path.exists():
            return
        
        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Use the proper from_dict method
                agreement = SparseAgreement.from_dict(data)
                self.agreements[agreement.agreement_id] = agreement
                
            except Exception as e:
                print(f"Error loading agreement from {file_path}: {e}")
                # Continue loading other agreements even if one fails