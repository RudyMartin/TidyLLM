#!/usr/bin/env python3
"""
SPARSE Agreement System for TidyLLM-HeirOS
==========================================

SPARSE = Structured Pre-Approved Reasoning for Systematic Execution

This system implements the [Learn Sparse] concept for pre-documented decisions
that corporate users have already approved, providing:
1. Compliance audit trails
2. Risk assessment documentation  
3. Stakeholder approval tracking
4. Expiration and review cycles
5. Conditional execution frameworks

Key for paranoid corporate users who need complete transparency and control.
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

@dataclass 
class SparseAgreement:
    """Core SPARSE agreement structure"""
    
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

class SparseAgreementManager:
    """Manager for SPARSE agreements with corporate compliance features"""
    
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
    
    def validate_execution_conditions(self, 
                                    agreement_id: str,
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate all execution conditions for SPARSE agreement"""
        
        if agreement_id not in self.agreements:
            return {'valid': False, 'error': f"Agreement {agreement_id} not found"}
        
        agreement = self.agreements[agreement_id]
        
        # Check agreement status
        if agreement.status != ApprovalStatus.APPROVED:
            return {'valid': False, 'error': f"Agreement not approved (status: {agreement.status.value})"}
        
        # Check expiry
        if agreement.expiry_date and datetime.now() > agreement.expiry_date:
            return {'valid': False, 'error': "Agreement has expired"}
        
        # Check review date
        if agreement.next_review_date and datetime.now() > agreement.next_review_date:
            return {'valid': False, 'error': "Agreement past due for review"}
        
        # Validate conditions
        failed_conditions = []
        for condition in agreement.conditions:
            if not self._validate_condition(condition, context):
                failed_conditions.append({
                    'condition_id': condition.condition_id,
                    'description': condition.description,
                    'reason': 'Validation failed'
                })
        
        if failed_conditions:
            return {
                'valid': False, 
                'error': 'Conditions not met',
                'failed_conditions': failed_conditions
            }
        
        return {'valid': True, 'agreement': agreement}
    
    def execute_sparse_agreement(self,
                                agreement_id: str,
                                context: Dict[str, Any],
                                executor_id: str) -> Dict[str, Any]:
        """Execute SPARSE agreement with full audit trail"""
        
        execution_id = str(uuid.uuid4())
        execution_start = datetime.now()
        
        # Validate conditions
        validation_result = self.validate_execution_conditions(agreement_id, context)
        if not validation_result['valid']:
            execution_record = {
                'execution_id': execution_id,
                'agreement_id': agreement_id,
                'status': 'validation_failed',
                'error': validation_result['error'],
                'timestamp': execution_start,
                'executor_id': executor_id
            }
            self.execution_log.append(execution_record)
            return execution_record
        
        agreement = validation_result['agreement']
        
        # Execute approved actions
        action_results = []
        overall_success = True
        
        for action in agreement.approved_actions:
            try:
                # Human confirmation if required
                if action.requires_human_confirmation:
                    # In practice, would integrate with approval system
                    confirmation_result = self._request_human_confirmation(action, context)
                    if not confirmation_result['confirmed']:
                        action_results.append({
                            'action_id': action.action_id,
                            'status': 'skipped',
                            'reason': 'Human confirmation denied'
                        })
                        continue
                
                # Execute action
                action_result = self._execute_action(action, context)
                action_results.append(action_result)
                
                if action_result['status'] != 'success':
                    overall_success = False
                
            except Exception as e:
                action_results.append({
                    'action_id': action.action_id,
                    'status': 'error',
                    'error': str(e)
                })
                overall_success = False
        
        # Update agreement statistics
        agreement.execution_count += 1
        agreement.last_execution_date = datetime.now()
        
        # Update execution statistics
        if 'total_executions' not in agreement.execution_statistics:
            agreement.execution_statistics['total_executions'] = 0
        agreement.execution_statistics['total_executions'] += 1
        agreement.execution_statistics['last_execution_status'] = 'success' if overall_success else 'partial_failure'
        
        self._save_agreement(agreement)
        
        # Create execution record
        execution_record = {
            'execution_id': execution_id,
            'agreement_id': agreement_id,
            'status': 'success' if overall_success else 'partial_failure',
            'timestamp': execution_start,
            'execution_duration': (datetime.now() - execution_start).total_seconds(),
            'executor_id': executor_id,
            'actions_executed': len(action_results),
            'action_results': action_results,
            'compliance_frameworks': [f.value for f in agreement.compliance_frameworks],
            'risk_level': agreement.risk_assessment.risk_level.value if agreement.risk_assessment else 'unknown'
        }
        
        self.execution_log.append(execution_record)
        self._save_execution_log()
        
        return execution_record
    
    def generate_compliance_report(self, agreement_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        
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
            # Status summary
            status = agreement.status.value if hasattr(agreement.status, 'value') else str(agreement.status)
            report['agreements_by_status'][status] = report['agreements_by_status'].get(status, 0) + 1
            
            # Risk summary
            if agreement.risk_assessment:
                risk = agreement.risk_assessment.risk_level.value if hasattr(agreement.risk_assessment.risk_level, 'value') else str(agreement.risk_assessment.risk_level)
                report['agreements_by_risk'][risk] = report['agreements_by_risk'].get(risk, 0) + 1
            
            # Execution summary
            report['execution_summary']['total_executions'] += agreement.execution_count
            
            # Compliance coverage
            for framework in agreement.compliance_frameworks:
                framework_name = framework.value
                report['compliance_coverage'][framework_name] = report['compliance_coverage'].get(framework_name, 0) + 1
            
            # Review status
            if agreement.expiry_date and datetime.now() > agreement.expiry_date:
                report['review_status']['expired_agreements'] += 1
            elif agreement.next_review_date:
                if datetime.now() > agreement.next_review_date:
                    report['review_status']['overdue_reviews'] += 1
                elif (agreement.next_review_date - datetime.now()).days <= 30:
                    report['review_status']['upcoming_reviews'] += 1
            
            # Detailed agreement info
            report['agreements_detail'].append({
                'agreement_id': agreement.agreement_id,
                'title': agreement.title,
                'status': agreement.status.value,
                'risk_level': agreement.risk_assessment.risk_level.value if agreement.risk_assessment else 'unassessed',
                'execution_count': agreement.execution_count,
                'last_execution': agreement.last_execution_date.isoformat() if agreement.last_execution_date else None,
                'stakeholder_approvals': len(agreement.stakeholder_approvals),
                'compliance_frameworks': [f.value for f in agreement.compliance_frameworks],
                'expires': agreement.expiry_date.isoformat() if agreement.expiry_date else None,
                'next_review': agreement.next_review_date.isoformat() if agreement.next_review_date else None
            })
        
        return report
    
    def _validate_condition(self, condition: ExecutionCondition, context: Dict[str, Any]) -> bool:
        """Validate individual execution condition"""
        
        if condition.condition_type == "context_check":
            # Check if required context values exist
            required_keys = condition.parameters.get('required_keys', [])
            return all(key in context for key in required_keys)
        
        elif condition.condition_type == "approval_required":
            # Check if required approval exists in context
            required_approval = condition.parameters.get('approval_key')
            return context.get(required_approval, False)
        
        elif condition.condition_type == "time_constraint":
            # Check time-based constraints
            current_hour = datetime.now().hour
            allowed_hours = condition.parameters.get('allowed_hours', list(range(24)))
            return current_hour in allowed_hours
        
        else:
            # Unknown condition type - fail safe
            return False
    
    def _execute_action(self, action: ApprovedAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute individual approved action"""
        
        # Mock implementation - in practice would execute real actions
        return {
            'action_id': action.action_id,
            'status': 'success',
            'result': f"Mock execution of {action.name}",
            'timestamp': datetime.now().isoformat()
        }
    
    def _request_human_confirmation(self, action: ApprovedAction, context: Dict[str, Any]) -> Dict[str, Any]:
        """Request human confirmation for action"""
        
        # Mock implementation - in practice would integrate with approval system
        return {
            'confirmed': True,  # Mock approval
            'confirmer_id': 'mock_user',
            'timestamp': datetime.now().isoformat()
        }
    
    def _save_agreement(self, agreement: SparseAgreement):
        """Save agreement to storage"""
        file_path = self.storage_path / f"{agreement.agreement_id}.json"
        
        # Convert to serializable format with enum handling
        def serialize_obj(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, Enum):
                return obj.value
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            return str(obj)
        
        # Convert agreement to dict with proper serialization
        data = {
            'agreement_id': agreement.agreement_id,
            'title': agreement.title,
            'description': agreement.description,
            'version': agreement.version,
            'status': agreement.status.value if agreement.status else 'draft',
            'created_date': agreement.created_date.isoformat() if agreement.created_date else None,
            'approval_date': agreement.approval_date.isoformat() if agreement.approval_date else None,
            'expiry_date': agreement.expiry_date.isoformat() if agreement.expiry_date else None,
            'business_purpose': agreement.business_purpose,
            'expected_frequency': agreement.expected_frequency,
            'business_owner': agreement.business_owner,
            'technical_owner': agreement.technical_owner,
            'conditions': [asdict(c) for c in agreement.conditions],
            'approved_actions': [asdict(a) for a in agreement.approved_actions],
            'compliance_frameworks': [f.value for f in agreement.compliance_frameworks],
            'risk_assessment': asdict(agreement.risk_assessment) if agreement.risk_assessment else None,
            'stakeholder_approvals': [asdict(s) for s in agreement.stakeholder_approvals],
            'review_frequency_days': agreement.review_frequency_days,
            'last_review_date': agreement.last_review_date.isoformat() if agreement.last_review_date else None,
            'next_review_date': agreement.next_review_date.isoformat() if agreement.next_review_date else None,
            'execution_count': agreement.execution_count,
            'last_execution_date': agreement.last_execution_date.isoformat() if agreement.last_execution_date else None,
            'execution_statistics': agreement.execution_statistics,
            'documentation_links': agreement.documentation_links,
            'related_policies': agreement.related_policies,
            'change_history': agreement.change_history
        }
        
        serializable_data = json.loads(json.dumps(data, default=serialize_obj))
        
        with open(file_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)
    
    def _load_agreements(self):
        """Load agreements from storage"""
        if not self.storage_path.exists():
            return
        
        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Convert datetime strings back to datetime objects
                # (Simplified implementation - in practice would be more robust)
                agreement = SparseAgreement(**data)
                self.agreements[agreement.agreement_id] = agreement
                
            except Exception as e:
                print(f"Error loading agreement from {file_path}: {e}")
    
    def _save_execution_log(self):
        """Save execution log to storage"""
        log_path = self.storage_path / "execution_log.json"
        
        def serialize_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj
        
        serializable_log = json.loads(json.dumps(self.execution_log, default=serialize_datetime))
        
        with open(log_path, 'w') as f:
            json.dump(serializable_log, f, indent=2)