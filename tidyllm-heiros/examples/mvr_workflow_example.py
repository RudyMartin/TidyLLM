#!/usr/bin/env python3
"""
TidyLLM-HeirOS: MVR Workflow Example
===================================

Demonstrates hierarchical DAG workflow for MVR peer review process with:
1. SPARSE agreements for pre-approved decisions
2. Dynamic AI flows for uncertain processes  
3. Corporate compliance tracking
4. Transparency for paranoid users

This example shows how to handle "all the twists and turns" in a clear,
auditable, hierarchical structure.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'dag-manager'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'sparse-agreement'))

from hierarchical_dag_manager import *
from sparse_system import *
from datetime import datetime, timedelta

def create_mvr_sparse_agreements(sparse_manager: SparseAgreementManager) -> Dict[str, SparseAgreement]:
    """Create SPARSE agreements for common MVR decisions"""
    
    agreements = {}
    
    # 1. Document Classification Agreement
    doc_classify_agreement = sparse_manager.create_agreement(
        title="Automated Document Type Classification",
        description="Pre-approved classification of uploaded documents as MVR, peer review, or supporting materials",
        business_purpose="Streamline document intake process while maintaining audit trail",
        business_owner="Risk Management Team",
        technical_owner="AI Systems Team"
    )
    
    # Add conditions
    sparse_manager.add_execution_condition(
        doc_classify_agreement.agreement_id,
        "Document size must be under 50MB",
        "context_check",
        "automated",
        {"required_keys": ["file_size"], "max_size": 52428800},
        "Security Team",
        "Large files require manual security screening"
    )
    
    sparse_manager.add_execution_condition(
        doc_classify_agreement.agreement_id,
        "Document must be PDF or TXT format",
        "context_check", 
        "automated",
        {"required_keys": ["file_extension"], "allowed_extensions": [".pdf", ".txt"]},
        "Compliance Team",
        "Only approved file formats for regulatory compliance"
    )
    
    # Add approved actions
    sparse_manager.add_approved_action(
        doc_classify_agreement.agreement_id,
        "Classify Document Type",
        "Use ML model to classify document type with confidence scoring",
        "ml_classification",
        {"model": "document_classifier_v2", "confidence_threshold": 0.85},
        requires_confirmation=False
    )
    
    sparse_manager.add_approved_action(
        doc_classify_agreement.agreement_id,
        "Extract Document Metadata",
        "Extract title, author, date, and section structure",
        "metadata_extraction",
        {"extraction_method": "hybrid_ocr_nlp"},
        requires_confirmation=False
    )
    
    # Add stakeholder approvals
    sparse_manager.add_stakeholder_approval(
        doc_classify_agreement.agreement_id,
        "Jane Smith",
        "VP Risk Management", 
        "Risk Management",
        "digital_signature",
        "Approved for standard document classification workflow"
    )
    
    sparse_manager.add_stakeholder_approval(
        doc_classify_agreement.agreement_id,
        "Bob Johnson",
        "Chief Compliance Officer",
        "Compliance",
        "digital_signature", 
        "Complies with document handling policies"
    )
    
    # Set risk assessment
    sparse_manager.set_risk_assessment(
        doc_classify_agreement.agreement_id,
        RiskLevel.LOW,
        business_impact=2.0,
        compliance_risk=1.5,
        operational_risk=2.5,
        assessor_id="risk_team",
        mitigation_strategies=[
            "Confidence threshold prevents misclassification",
            "Manual review queue for low-confidence results", 
            "Audit trail for all classifications"
        ],
        monitoring_requirements=[
            "Weekly accuracy metrics review",
            "Monthly false positive analysis"
        ]
    )
    
    # Set compliance frameworks
    doc_classify_agreement.compliance_frameworks = [
        ComplianceFramework.SOX,
        ComplianceFramework.INTERNAL_POLICY
    ]
    
    # Approve agreement
    sparse_manager.approve_agreement(doc_classify_agreement.agreement_id, "system_admin")
    agreements['document_classification'] = doc_classify_agreement
    
    # 2. Standard Compliance Check Agreement  
    compliance_agreement = sparse_manager.create_agreement(
        title="Standard MVR Compliance Validation",
        description="Automated validation against MVS and VST requirements for standard risk tier models",
        business_purpose="Ensure consistent compliance checking while reducing manual review time",
        business_owner="Model Validation Team",
        technical_owner="AI Systems Team"
    )
    
    # Add conditions for compliance checking
    sparse_manager.add_execution_condition(
        compliance_agreement.agreement_id,
        "Model must be standard risk tier",
        "context_check",
        "automated", 
        {"required_keys": ["model_risk_tier"], "allowed_tiers": ["standard", "low"]},
        "Model Validation Head",
        "High risk models require enhanced manual review"
    )
    
    # Add compliance actions
    sparse_manager.add_approved_action(
        compliance_agreement.agreement_id,
        "MVS Requirements Check",
        "Validate against Model Validation Standards checklist",
        "compliance_check",
        {"framework": "MVS", "version": "2024.1", "checklist_items": 47}
    )
    
    sparse_manager.add_approved_action(
        compliance_agreement.agreement_id,
        "VST Section Validation", 
        "Check Validation Scoping Template coverage",
        "compliance_check",
        {"framework": "VST", "required_sections": ["conceptual_soundness", "data_quality"]}
    )
    
    # Add stakeholder approval
    sparse_manager.add_stakeholder_approval(
        compliance_agreement.agreement_id,
        "Dr. Sarah Wilson",
        "Head of Model Validation",
        "Model Risk Management",
        "digital_signature",
        "Approved for standard tier models only"
    )
    
    # Risk assessment
    sparse_manager.set_risk_assessment(
        compliance_agreement.agreement_id,
        RiskLevel.MEDIUM,
        business_impact=4.0,
        compliance_risk=6.0,
        operational_risk=3.0, 
        assessor_id="model_risk_team",
        mitigation_strategies=[
            "Limited to standard risk tier models",
            "Human review for any compliance gaps",
            "Regular calibration against manual reviews"
        ]
    )
    
    compliance_agreement.compliance_frameworks = [
        ComplianceFramework.SOX,
        ComplianceFramework.REGULATORY_GUIDANCE,
        ComplianceFramework.INTERNAL_POLICY
    ]
    
    sparse_manager.approve_agreement(compliance_agreement.agreement_id, "cro_office")
    agreements['standard_compliance'] = compliance_agreement
    
    return agreements

def create_mvr_dag_workflow(dag_manager: HierarchicalDAGManager, 
                           sparse_agreements: Dict[str, SparseAgreement]) -> HierarchicalDAGManager:
    """Create hierarchical DAG workflow for MVR processing"""
    
    # Root sequence: Main MVR workflow
    root_workflow = SequenceNode(
        "mvr_root_workflow", 
        "MVR Peer Review Workflow",
        description="Complete MVR peer review process with compliance tracking"
    )
    
    # Phase 1: Document Intake (SPARSE decisions)
    intake_phase = SequenceNode("intake_phase", "Document Intake Phase")
    
    # Document classification - SPARSE agreement
    doc_classification = SparseDecisionNode(
        "doc_classify_sparse",
        "Document Type Classification", 
        sparse_agreements['document_classification'],
        description="Classify uploaded documents using pre-approved ML model"
    )
    
    # Document validation
    doc_validation = ActionNode(
        "doc_validation",
        "Document Validation",
        action=lambda ctx: {
            "validation_result": "passed",
            "file_integrity": "verified", 
            "security_scan": "clean"
        }
    )
    
    intake_phase.add_child(doc_classification).add_child(doc_validation)
    
    # Phase 2: Analysis Selection (Selector for different analysis paths)
    analysis_selector = SelectorNode("analysis_selector", "Analysis Path Selection")
    
    # Path A: Standard Compliance (SPARSE)
    standard_path = SequenceNode("standard_compliance_path", "Standard Compliance Path")
    
    standard_compliance = SparseDecisionNode(
        "standard_compliance_sparse",
        "Standard MVR Compliance Check",
        sparse_agreements['standard_compliance'], 
        description="Automated compliance validation for standard models"
    )
    
    standard_path.add_child(standard_compliance)
    
    # Path B: Complex Analysis (Dynamic AI flow for uncertain cases)
    complex_path = DynamicFlowNode(
        "complex_analysis_dynamic",
        "Complex Analysis Dynamic Flow",
        ai_orchestrator=create_complex_analysis_orchestrator,
        description="AI-generated workflow for complex or high-risk models"
    )
    
    # Path C: Manual Review Required  
    manual_path = ActionNode(
        "manual_review_required",
        "Manual Review Required",
        action=lambda ctx: {
            "status": "escalated_to_human",
            "reason": "Complex case requiring human expertise",
            "assigned_reviewer": "senior_model_validator"
        }
    )
    
    analysis_selector.add_child(standard_path).add_child(complex_path).add_child(manual_path)
    
    # Phase 3: Report Generation
    report_generation = SequenceNode("report_generation", "Report Generation Phase")
    
    # Generate findings summary
    findings_summary = ActionNode(
        "findings_summary",
        "Generate Findings Summary", 
        action=lambda ctx: {
            "findings_generated": True,
            "summary_sections": ["executive_summary", "key_findings", "recommendations"],
            "confidence_scores": {"overall": 0.92, "critical_issues": 0.88}
        }
    )
    
    # Compliance report
    compliance_report = ActionNode(
        "compliance_report_generation",
        "Generate Compliance Report",
        action=lambda ctx: {
            "compliance_report": "generated",
            "frameworks_covered": ["MVS", "VST", "Internal Policy"],
            "compliance_score": 0.94
        }
    )
    
    report_generation.add_child(findings_summary).add_child(compliance_report)
    
    # Phase 4: Final Review & Approval
    final_review = SelectorNode("final_review", "Final Review Process")
    
    # Auto-approval for low risk
    auto_approve = ActionNode(
        "auto_approve",
        "Automatic Approval",
        action=lambda ctx: {
            "approved": True,
            "approval_type": "automated",
            "confidence": 0.95
        }
    )
    
    # Human review required
    human_review = ActionNode(
        "human_review_required", 
        "Human Review Required",
        action=lambda ctx: {
            "status": "pending_human_review",
            "reviewer_assigned": "senior_validator",
            "priority": "normal"
        }
    )
    
    final_review.add_child(auto_approve).add_child(human_review)
    
    # Assemble complete workflow
    root_workflow.add_child(intake_phase)
    root_workflow.add_child(analysis_selector) 
    root_workflow.add_child(report_generation)
    root_workflow.add_child(final_review)
    
    dag_manager.add_root_node(root_workflow)
    
    return dag_manager

def create_complex_analysis_orchestrator(context: Dict[str, Any], 
                                       global_context: Dict[str, Any]) -> List[HierarchicalNode]:
    """AI orchestrator for complex analysis cases (inspired by Elysia)"""
    
    # Simulate AI decision-making for uncertain processes
    # In practice, this would use LLM reasoning to generate appropriate workflow
    
    model_complexity = context.get('model_complexity', 'unknown')
    risk_tier = context.get('model_risk_tier', 'unknown') 
    
    generated_nodes = []
    
    if model_complexity == 'high' or risk_tier == 'high':
        # High complexity requires enhanced analysis
        
        # Deep validation node
        deep_validation = ActionNode(
            f"deep_validation_{uuid.uuid4().hex[:8]}",
            "AI-Generated Deep Validation",
            action=lambda ctx: {
                "deep_validation_completed": True,
                "additional_checks": ["stress_testing", "scenario_analysis", "back_testing"],
                "confidence": 0.78
            }
        )
        generated_nodes.append(deep_validation)
        
        # Risk assessment node  
        risk_assessment = ActionNode(
            f"risk_assessment_{uuid.uuid4().hex[:8]}",
            "AI-Generated Risk Assessment",
            action=lambda ctx: {
                "risk_assessment_completed": True,
                "risk_factors_identified": 8,
                "mitigation_strategies": 5
            }
        )
        generated_nodes.append(risk_assessment)
        
        # Senior review requirement
        senior_review = ActionNode(
            f"senior_review_{uuid.uuid4().hex[:8]}", 
            "AI-Generated Senior Review Requirement",
            action=lambda ctx: {
                "senior_review_required": True,
                "escalation_level": "c_suite",
                "urgency": "high"
            }
        )
        generated_nodes.append(senior_review)
    
    else:
        # Standard enhanced analysis
        enhanced_validation = ActionNode(
            f"enhanced_validation_{uuid.uuid4().hex[:8]}",
            "AI-Generated Enhanced Validation", 
            action=lambda ctx: {
                "enhanced_validation_completed": True,
                "additional_validations": 3,
                "confidence": 0.85
            }
        )
        generated_nodes.append(enhanced_validation)
    
    return generated_nodes

def demonstrate_mvr_workflow():
    """Demonstrate complete MVR workflow with SPARSE agreements and dynamic flows"""
    
    print("=" * 80)
    print("TidyLLM-HeirOS: MVR Workflow Demonstration")
    print("Hierarchical DAG with SPARSE Agreements & Dynamic AI Flows")
    print("=" * 80)
    
    # Initialize managers
    sparse_manager = SparseAgreementManager("demo_sparse_agreements")
    dag_manager = HierarchicalDAGManager("MVR Peer Review System", ComplianceLevel.FULL_TRANSPARENCY)
    
    # Create SPARSE agreements
    print("\n1. Creating SPARSE Agreements for Pre-Approved Decisions...")
    sparse_agreements = create_mvr_sparse_agreements(sparse_manager)
    
    for name, agreement in sparse_agreements.items():
        print(f"   + {agreement.title} (ID: {agreement.agreement_id[:8]}...)")
        print(f"     Status: {agreement.status.value}")
        print(f"     Risk Level: {agreement.risk_assessment.risk_level.value if agreement.risk_assessment else 'N/A'}")
        print(f"     Stakeholder Approvals: {len(agreement.stakeholder_approvals)}")
        print()
    
    # Create DAG workflow
    print("2. Building Hierarchical DAG Workflow...")
    dag_manager = create_mvr_dag_workflow(dag_manager, sparse_agreements)
    
    # Visualize hierarchy
    print("\n3. Workflow Hierarchy Visualization:")
    print(dag_manager.visualize_hierarchy())
    
    # Execute workflow with different scenarios
    print("\n4. Executing Workflow Scenarios...")
    
    # Scenario 1: Standard document with automated processing
    print("\n   SCENARIO 1: Standard MVR Document")
    print("   " + "-" * 45)
    
    standard_context = {
        'file_size': 2048000,  # 2MB
        'file_extension': '.pdf',
        'document_type': 'mvr',
        'model_risk_tier': 'standard',
        'model_complexity': 'medium',
        'compliance_required': True
    }
    
    result1 = dag_manager.execute_dag(standard_context)
    print(f"   Execution Status: {result1['overall_status']}")
    print(f"   Duration: {result1['duration_seconds']:.2f} seconds")
    print(f"   Nodes Executed: {result1['nodes_executed']}")
    
    # Scenario 2: High-risk model requiring dynamic AI flow
    print("\n   SCENARIO 2: High-Risk Model (Dynamic AI Flow)")
    print("   " + "-" * 50)
    
    complex_context = {
        'file_size': 5120000,  # 5MB
        'file_extension': '.pdf', 
        'document_type': 'mvr',
        'model_risk_tier': 'high',
        'model_complexity': 'high',
        'compliance_required': True
    }
    
    result2 = dag_manager.execute_dag(complex_context)
    print(f"   Execution Status: {result2['overall_status']}")
    print(f"   Duration: {result2['duration_seconds']:.2f} seconds") 
    print(f"   Nodes Executed: {result2['nodes_executed']}")
    
    # Generate compliance reports
    print("\n5. Compliance & Audit Reports...")
    
    # SPARSE agreements compliance
    sparse_compliance = sparse_manager.generate_compliance_report()
    print(f"\n   SPARSE Agreements Summary:")
    print(f"   Total Agreements: {sparse_compliance['total_agreements']}")
    print(f"   Total Executions: {sparse_compliance['execution_summary']['total_executions']}")
    print(f"   Compliance Coverage: {list(sparse_compliance['compliance_coverage'].keys())}")
    
    # DAG execution compliance
    dag_compliance = dag_manager.generate_compliance_report()
    print(f"\n   DAG Workflow Summary:")
    print(f"   Total Nodes: {dag_compliance['total_nodes']}")
    print(f"   Total Decisions: {dag_compliance['total_decisions']}")
    print(f"   Audit Completeness: {dag_compliance['audit_completeness']:.1%}")
    print(f"   Risk Factors: {dag_compliance['risk_factors']}")
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("+ Hierarchical workflow with clear structure")
    print("+ SPARSE agreements for pre-approved decisions") 
    print("+ Dynamic AI flows for uncertain processes")
    print("+ Complete compliance audit trails")
    print("+ Corporate transparency and control")
    print("=" * 80)
    
    return {
        'dag_manager': dag_manager,
        'sparse_manager': sparse_manager,
        'execution_results': [result1, result2],
        'compliance_reports': {
            'sparse': sparse_compliance,
            'dag': dag_compliance
        }
    }

if __name__ == "__main__":
    results = demonstrate_mvr_workflow()