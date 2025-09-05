"""
TidyLLM Enterprise Platform - Unified Interface

Main platform class that brings together:
1. Analysis Layer (document compliance analysis)  
2. Workflow Layer (process orchestration)
3. Framework Layer (unified compliance mapping)
4. Integration Layer (TidyLLM ecosystem connectivity)

Provides a single, easy-to-use interface for complete enterprise compliance.
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import uuid

# Analysis Layer
from .analysis import ModelRiskAnalyzer, EvidenceValidator, ConsistencyAnalyzer

# Workflow Layer
from .workflows import (
    HierarchicalDAGManager,
    SparseAgreementManager,
    SequenceNode,
    SelectorNode,
    ActionNode,
    AnalysisNode,
    SparseAgreement,
    ComplianceLevel,
    RiskLevel,
    ComplianceFramework
)

# Framework Layer
from .frameworks import UnifiedComplianceFramework

class EnterpriseCompliancePlatform:
    """
    Main platform interface for enterprise compliance and workflow management.
    
    Combines document analysis, workflow orchestration, and compliance management
    in a single, easy-to-use platform.
    """
    
    def __init__(self, 
                 storage_path: str = "enterprise_compliance",
                 compliance_level: ComplianceLevel = ComplianceLevel.FULL_TRANSPARENCY):
        """
        Initialize the enterprise compliance platform.
        
        Args:
            storage_path: Directory for storing agreements and execution history
            compliance_level: Level of audit trail detail
        """
        self.storage_path = storage_path
        self.compliance_level = compliance_level
        
        # Initialize core components
        self.workflow_manager = HierarchicalDAGManager(
            "Enterprise Compliance Platform", 
            compliance_level
        )
        self.sparse_manager = SparseAgreementManager(storage_path)
        self.compliance_framework = UnifiedComplianceFramework()
        
        # Initialize analyzers
        self.model_risk_analyzer = ModelRiskAnalyzer()
        self.evidence_validator = EvidenceValidator() if EvidenceValidator else None
        self.consistency_analyzer = ConsistencyAnalyzer() if ConsistencyAnalyzer else None
        
        # Track created workflows and agreements
        self.workflows: Dict[str, HierarchicalDAGManager] = {}
        self.agreements: Dict[str, SparseAgreement] = {}
        
    def create_compliance_workflow(self, 
                                  name: str, 
                                  description: str = "") -> 'ComplianceWorkflowBuilder':
        """
        Create a new compliance workflow with builder pattern.
        
        Args:
            name: Workflow name
            description: Workflow description
            
        Returns:
            ComplianceWorkflowBuilder for chaining workflow construction
        """
        workflow_id = str(uuid.uuid4())
        workflow = HierarchicalDAGManager(name, self.compliance_level)
        workflow.description = description
        
        self.workflows[workflow_id] = workflow
        
        return ComplianceWorkflowBuilder(
            workflow_id=workflow_id,
            workflow=workflow,
            platform=self
        )
    
    def create_sparse_agreement(self,
                               title: str,
                               description: str,
                               business_purpose: str,
                               business_owner: str,
                               technical_owner: str,
                               compliance_frameworks: List[str] = None) -> SparseAgreement:
        """
        Create a new SPARSE agreement for pre-approved decisions.
        
        Args:
            title: Agreement title
            description: Detailed description
            business_purpose: Business justification
            business_owner: Business owner name
            technical_owner: Technical owner name
            compliance_frameworks: List of applicable frameworks
            
        Returns:
            Created SparseAgreement
        """
        agreement = self.sparse_manager.create_agreement(
            title=title,
            description=description,
            business_purpose=business_purpose,
            business_owner=business_owner,
            technical_owner=technical_owner
        )
        
        # Add compliance frameworks if specified
        if compliance_frameworks:
            framework_enums = []
            for framework in compliance_frameworks:
                try:
                    framework_enums.append(ComplianceFramework(framework.lower().replace(" ", "_")))
                except ValueError:
                    # Skip invalid framework names
                    continue
            agreement.compliance_frameworks = framework_enums
        
        self.agreements[agreement.agreement_id] = agreement
        return agreement
    
    def analyze_document(self, 
                        document_content: str = None,
                        document_path: str = None,
                        analysis_type: str = "model_risk") -> Dict[str, Any]:
        """
        Analyze document for compliance using specified analyzer.
        
        Args:
            document_content: Document text content
            document_path: Path to document file
            analysis_type: Type of analysis (model_risk, evidence, consistency)
            
        Returns:
            Analysis results dictionary
        """
        if not document_content and not document_path:
            raise ValueError("Either document_content or document_path must be provided")
        
        if document_path and not document_content:
            with open(document_path, 'r', encoding='utf-8') as f:
                document_content = f.read()
        
        # Select analyzer
        if analysis_type == "model_risk":
            analyzer = self.model_risk_analyzer
            result = analyzer.assess_document_compliance(document_content)
        elif analysis_type == "evidence" and self.evidence_validator:
            result = self.evidence_validator.validate_document(document_content)
        elif analysis_type == "consistency" and self.consistency_analyzer:
            result = self.consistency_analyzer.analyze_document(document_content)
        else:
            raise ValueError(f"Unknown or unavailable analysis type: {analysis_type}")
        
        # Enhance with compliance framework mapping
        framework_assessment = self._assess_framework_compliance(result)
        result['framework_compliance'] = framework_assessment
        
        return result
    
    def execute_workflow(self, 
                        workflow_id: str, 
                        context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a created workflow with given context.
        
        Args:
            workflow_id: ID of workflow to execute
            context: Execution context data
            
        Returns:
            Execution results
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        return workflow.execute_dag(context or {})
    
    def generate_enterprise_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive enterprise compliance report.
        
        Returns:
            Complete enterprise compliance status report
        """
        report = {
            'report_generated': datetime.now().isoformat(),
            'platform_summary': {
                'total_workflows': len(self.workflows),
                'total_agreements': len(self.agreements),
                'compliance_level': self.compliance_level.value
            },
            'workflow_summary': self._generate_workflow_summary(),
            'sparse_summary': self.sparse_manager.generate_compliance_report(),
            'framework_compliance': self._generate_framework_summary(),
            'integration_status': self._check_integration_status()
        }
        
        return report
    
    def _assess_framework_compliance(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Assess compliance against unified framework"""
        
        # Map analysis results to framework requirements
        assessed_requirements = set()
        
        if 'rule_assessments' in analysis_result:
            for rule_id, assessment in analysis_result['rule_assessments'].items():
                if assessment.get('compliance_score', 0) >= 0.7:
                    # Map to framework requirements based on rule
                    if 'MRD-001' in rule_id:
                        assessed_requirements.add('FIN-SR11-7-001')
                    elif 'MRD-002' in rule_id:
                        assessed_requirements.add('FIN-SR11-7-002')
        
        return self.compliance_framework.assess_compliance_coverage(assessed_requirements)
    
    def _generate_workflow_summary(self) -> Dict[str, Any]:
        """Generate summary of all workflows"""
        summary = {
            'total_workflows': len(self.workflows),
            'workflows_by_status': {},
            'execution_statistics': {
                'total_executions': 0,
                'successful_executions': 0,
                'failed_executions': 0
            }
        }
        
        for workflow in self.workflows.values():
            # Aggregate execution history
            summary['execution_statistics']['total_executions'] += len(workflow.execution_history)
            
            for execution in workflow.execution_history:
                if execution['overall_status'] == 'success':
                    summary['execution_statistics']['successful_executions'] += 1
                else:
                    summary['execution_statistics']['failed_executions'] += 1
        
        return summary
    
    def _generate_framework_summary(self) -> Dict[str, Any]:
        """Generate compliance framework summary"""
        return self.compliance_framework.generate_compliance_matrix()
    
    def _check_integration_status(self) -> Dict[str, Any]:
        """Check integration status with TidyLLM ecosystem"""
        integrations = {
            'tidyllm_core': False,
            'tidymart': False,
            'gateway': False,
            'sentence_embeddings': False
        }
        
        # Check for TidyLLM core
        try:
            import tidyllm
            integrations['tidyllm_core'] = True
        except ImportError:
            pass
        
        # Check for TidyMart (would be in tidyllm.tidymart)
        try:
            import tidyllm.tidymart
            integrations['tidymart'] = True
        except (ImportError, AttributeError):
            pass
        
        # Check for Gateway
        try:
            import tidyllm.gateway
            integrations['gateway'] = True
        except (ImportError, AttributeError):
            pass
        
        # Check for sentence embeddings
        try:
            import tidyllm.sentence
            integrations['sentence_embeddings'] = True
        except (ImportError, AttributeError):
            pass
        
        return {
            'integrations_available': integrations,
            'integration_score': sum(integrations.values()) / len(integrations),
            'recommendations': self._get_integration_recommendations(integrations)
        }
    
    def _get_integration_recommendations(self, integrations: Dict[str, bool]) -> List[str]:
        """Get recommendations for missing integrations"""
        recommendations = []
        
        if not integrations['tidyllm_core']:
            recommendations.append("Install tidyllm core package for enhanced ML capabilities")
        
        if not integrations['tidymart']:
            recommendations.append("Enable TidyMart for performance tracking and optimization")
        
        if not integrations['gateway']:
            recommendations.append("Enable Gateway for LLM governance and enterprise controls")
            
        if not integrations['sentence_embeddings']:
            recommendations.append("Enable sentence embeddings for enhanced document analysis")
        
        return recommendations

class ComplianceWorkflowBuilder:
    """
    Builder class for constructing compliance workflows with method chaining.
    """
    
    def __init__(self, workflow_id: str, workflow: HierarchicalDAGManager, platform: EnterpriseCompliancePlatform):
        self.workflow_id = workflow_id
        self.workflow = workflow
        self.platform = platform
        self.current_node = None
        self.root_sequence = SequenceNode("root_sequence", "Main Workflow Sequence")
        self.workflow.add_root_node(self.root_sequence)
    
    def add_document_analysis(self, 
                             node_id: str,
                             name: str, 
                             analysis_type: str = "model_risk") -> 'ComplianceWorkflowBuilder':
        """Add document analysis step to workflow"""
        
        # Select appropriate analyzer
        if analysis_type == "model_risk":
            analyzer = self.platform.model_risk_analyzer
            method = "assess_document_compliance"
        elif analysis_type == "evidence" and self.platform.evidence_validator:
            analyzer = self.platform.evidence_validator
            method = "validate_document"
        elif analysis_type == "consistency" and self.platform.consistency_analyzer:
            analyzer = self.platform.consistency_analyzer
            method = "analyze_document"
        else:
            raise ValueError(f"Unknown or unavailable analysis type: {analysis_type}")
        
        analysis_node = AnalysisNode(
            node_id=node_id,
            name=name,
            analyzer=analyzer,
            analyzer_method=method,
            description=f"Document analysis using {analysis_type} analyzer"
        )
        
        self.root_sequence.add_child(analysis_node)
        self.current_node = analysis_node
        return self
    
    def add_sparse_decision(self, 
                           node_id: str,
                           name: str,
                           agreement: SparseAgreement) -> 'ComplianceWorkflowBuilder':
        """Add SPARSE decision step to workflow"""
        
        # Import here to avoid circular imports
        from .workflows.dag_manager import SparseDecisionNode
        
        sparse_node = SparseDecisionNode(
            node_id=node_id,
            name=name, 
            sparse_agreement=agreement,
            description=f"Pre-approved decision: {agreement.title}"
        )
        
        self.root_sequence.add_child(sparse_node)
        self.current_node = sparse_node
        return self
    
    def add_action(self,
                   node_id: str,
                   name: str,
                   action_function) -> 'ComplianceWorkflowBuilder':
        """Add custom action step to workflow"""
        
        action_node = ActionNode(
            node_id=node_id,
            name=name,
            action=action_function,
            description=f"Custom action: {name}"
        )
        
        self.root_sequence.add_child(action_node)
        self.current_node = action_node
        return self
    
    def add_audit_trail(self) -> 'ComplianceWorkflowBuilder':
        """Add audit trail generation step"""
        
        def generate_audit_trail(context):
            return {
                'audit_generated': True,
                'timestamp': datetime.now().isoformat(),
                'compliance_level': self.platform.compliance_level.value,
                'workflow_id': self.workflow_id
            }
        
        return self.add_action("audit_trail", "Generate Audit Trail", generate_audit_trail)
    
    def build(self) -> str:
        """Finalize workflow construction and return workflow ID"""
        return self.workflow_id