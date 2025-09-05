"""
TidyLLM Enterprise: Complete Compliance & Workflow Platform

The only enterprise platform that combines compliance analysis with workflow 
orchestration in complete algorithmic transparency.

Core Capabilities:
1. Document compliance analysis (model risk, evidence validation, consistency)
2. Workflow orchestration with SPARSE pre-approved decisions  
3. Complete audit trails and regulatory compliance
4. Integration with TidyLLM ecosystem

Perfect for financial services, healthcare, legal, and heavily regulated industries.
"""

# Analysis Layer - Document Intelligence
from .analysis import (
    ModelRiskAnalyzer, 
    ModelRiskMonitor, 
    EvidenceValidator, 
    ConsistencyAnalyzer
)

# Workflow Layer - Process Orchestration  
from .workflows.dag_manager import (
    HierarchicalDAGManager, 
    SequenceNode, 
    SelectorNode, 
    ActionNode,
    SparseDecisionNode,
    DynamicFlowNode
)
from .workflows.sparse_system import (
    SparseAgreementManager, 
    SparseAgreement,
    RiskLevel,
    ComplianceFramework
)

# Unified Platform Interface
from .platform import EnterpriseCompliancePlatform

# Framework Integration
from .frameworks import UnifiedComplianceFramework

__version__ = "1.0.0"
__author__ = "Rudy Martin"

__all__ = [
    # Analysis Layer
    "ModelRiskAnalyzer",
    "ModelRiskMonitor", 
    "EvidenceValidator",
    "ConsistencyAnalyzer",
    
    # Workflow Layer
    "HierarchicalDAGManager",
    "SequenceNode",
    "SelectorNode", 
    "ActionNode",
    "SparseDecisionNode",
    "DynamicFlowNode",
    "SparseAgreementManager",
    "SparseAgreement",
    "RiskLevel",
    "ComplianceFramework",
    
    # Unified Platform
    "EnterpriseCompliancePlatform",
    "UnifiedComplianceFramework",
]

# Package metadata
DESCRIPTION = "Enterprise compliance and workflow platform with complete algorithmic transparency"
LICENSE = "CC-BY-4.0"  
HOMEPAGE = "https://github.com/tidyllm-verse/tidyllm-enterprise"

# Quick start examples
QUICK_START = """
# Document Compliance Analysis
import tidyllm_enterprise as tidy_ent

# Analyze for model risk compliance  
analyzer = tidy_ent.ModelRiskAnalyzer()
result = analyzer.assess_document("model_validation_report.pdf")
print(f"SR 11-7 Compliance: {result.compliance_score:.1%}")

# Create compliance workflow
platform = tidy_ent.EnterpriseCompliancePlatform()
workflow = platform.create_compliance_workflow("Document Review")
workflow.add_analysis_step(analyzer)
workflow.add_audit_trail()

result = workflow.execute({"document": "investment_memo.pdf"})
audit_report = workflow.generate_compliance_report()
"""