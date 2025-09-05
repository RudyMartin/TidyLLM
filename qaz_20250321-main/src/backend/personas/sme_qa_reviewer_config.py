#!/usr/bin/env python3
"""
SME_QAReviewer Configuration
Defines the requirements and standards for the QA Reviewer persona
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum
from datetime import datetime

class QAMethodology(Enum):
    """QA methodologies supported by the QA Reviewer"""
    SIX_SIGMA = "six_sigma"
    ISO_9001 = "iso_9001"
    AGILE_QA = "agile_qa"
    DEVOPS_QA = "devops_qa"
    MODEL_RISK_QA = "model_risk_qa"
    STATISTICAL_QC = "statistical_qc"
    PROCESS_IMPROVEMENT = "process_improvement"

class QATool(Enum):
    """QA tools and techniques"""
    CHECKLISTS = "checklists"
    PROCESS_FLOW_ANALYSIS = "process_flow_analysis"
    ROOT_CAUSE_ANALYSIS = "root_cause_analysis"
    STATISTICAL_PROCESS_CONTROL = "statistical_process_control"
    QUALITY_METRICS_DASHBOARD = "quality_metrics_dashboard"
    DEFECT_TRACKING = "defect_tracking"
    CONTINUOUS_IMPROVEMENT = "continuous_improvement"

class QAStandard(Enum):
    """QA standards and frameworks"""
    ISO_9001 = "iso_9001"
    SIX_SIGMA = "six_sigma"
    MODEL_RISK_QA = "model_risk_qa"
    AGILE_QA = "agile_qa"
    DEVOPS_QA = "devops_qa"

@dataclass
class SME_QAReviewer_Requirements:
    """Comprehensive requirements for the SME_QAReviewer persona"""
    
    # Core Identity
    persona_name: str = "SME_QAReviewer"
    authority_level: str = "Senior_QA_Review_Expert"
    expertise_domain: str = "Model_Lifecycle_Quality_Assurance"
    
    # QA Specialization Areas
    qa_methodologies: List[QAMethodology] = field(default_factory=lambda: [
        QAMethodology.SIX_SIGMA,
        QAMethodology.ISO_9001,
        QAMethodology.AGILE_QA,
        QAMethodology.DEVOPS_QA,
        QAMethodology.MODEL_RISK_QA,
        QAMethodology.STATISTICAL_QC,
        QAMethodology.PROCESS_IMPROVEMENT
    ])
    
    # Model Lifecycle Expertise
    lifecycle_stages: List[str] = field(default_factory=lambda: [
        "Planning and Requirements",
        "Development and Design", 
        "Validation and Testing",
        "Implementation and Deployment",
        "Use and Monitoring",
        "Changes and Updates",
        "Retirement and Decommissioning"
    ])
    
    # QA Tool Proficiency
    qa_tools: List[QATool] = field(default_factory=lambda: [
        QATool.CHECKLISTS,
        QATool.PROCESS_FLOW_ANALYSIS,
        QATool.ROOT_CAUSE_ANALYSIS,
        QATool.STATISTICAL_PROCESS_CONTROL,
        QATool.QUALITY_METRICS_DASHBOARD,
        QATool.DEFECT_TRACKING,
        QATool.CONTINUOUS_IMPROVEMENT
    ])
    
    # QA Standards and Frameworks
    qa_standards: Dict[str, Any] = None
    qa_best_practices: Dict[str, Any] = None
    qa_checklists: Dict[str, Any] = None
    qa_scoring_weights: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize comprehensive QA standards and practices"""
        
        # QA Standards Library
        if self.qa_standards is None:
            self.qa_standards = {
                "iso_9001": {
                    "quality_management": [
                        "Leadership", "Planning", "Support", "Operation", 
                        "Performance", "Improvement"
                    ],
                    "documentation_requirements": [
                        "Quality Manual", "Procedures", "Work Instructions", "Records"
                    ],
                    "process_approach": [
                        "Process Identification", "Process Management", "Process Improvement"
                    ]
                },
                "six_sigma": {
                    "dmaic_methodology": [
                        "Define", "Measure", "Analyze", "Improve", "Control"
                    ],
                    "statistical_tools": [
                        "Control Charts", "Process Capability", "Design of Experiments"
                    ],
                    "quality_metrics": [
                        "Defects per Million", "Process Sigma Level", "Cost of Poor Quality"
                    ]
                },
                "model_risk_qa": {
                    "sr11_7_compliance": [
                        "Model Validation", "Governance", "Documentation", "Monitoring"
                    ],
                    "model_lifecycle_qa": [
                        "Development QA", "Validation QA", "Implementation QA", "Monitoring QA"
                    ],
                    "risk_assessment": [
                        "Model Risk Identification", "Risk Quantification", "Risk Mitigation"
                    ]
                }
            }
        
        # QA Best Practices Repository
        if self.qa_best_practices is None:
            self.qa_best_practices = {
                "checklist_design": [
                    "Clear and unambiguous criteria",
                    "Measurable and testable requirements", 
                    "Risk-based prioritization",
                    "Continuous improvement feedback"
                ],
                "process_validation": [
                    "Independent review requirements",
                    "Documentation standards",
                    "Approval workflows",
                    "Quality gates and checkpoints"
                ],
                "continuous_monitoring": [
                    "Real-time quality metrics",
                    "Automated quality checks",
                    "Trend analysis and reporting",
                    "Proactive issue identification"
                ]
            }
        
        # QA Checklists by Stage
        if self.qa_checklists is None:
            self.qa_checklists = {
                "planning_stage": {
                    "requirements_validation": [
                        "Business requirements clearly defined",
                        "Stakeholder requirements captured",
                        "Success criteria established",
                        "Resource requirements identified",
                        "Timeline and milestones defined"
                    ],
                    "risk_assessment": [
                        "Initial risk assessment completed",
                        "Risk mitigation strategies defined",
                        "Risk acceptance criteria established",
                        "Risk monitoring plan developed"
                    ],
                    "governance_setup": [
                        "Governance framework established",
                        "Roles and responsibilities defined",
                        "Approval processes documented",
                        "Communication plan developed"
                    ]
                },
                "development_stage": {
                    "design_quality": [
                        "Architecture design reviewed",
                        "Technical specifications complete",
                        "Design standards followed",
                        "Peer review completed",
                        "Design approval obtained"
                    ],
                    "implementation_quality": [
                        "Coding standards followed",
                        "Code review completed",
                        "Unit testing performed",
                        "Integration testing planned",
                        "Documentation updated"
                    ],
                    "data_quality": [
                        "Data requirements defined",
                        "Data quality assessment completed",
                        "Data processing documented",
                        "Data validation procedures established"
                    ]
                },
                "validation_stage": {
                    "validation_planning": [
                        "Validation scope defined",
                        "Validation methodology selected",
                        "Validation timeline established",
                        "Validation resources allocated",
                        "Validation success criteria defined"
                    ],
                    "statistical_validation": [
                        "Data quality assessment completed",
                        "Model performance analysis completed",
                        "Backtesting performed",
                        "Stress testing completed",
                        "Sensitivity analysis performed"
                    ],
                    "qualitative_validation": [
                        "Model design review completed",
                        "Business logic validation completed",
                        "Assumption analysis completed",
                        "Implementation review completed",
                        "Expert judgment integrated"
                    ]
                },
                "implementation_stage": {
                    "deployment_planning": [
                        "Deployment strategy developed",
                        "Environment prepared",
                        "Integration planning completed",
                        "Testing procedures defined",
                        "Rollback procedures established"
                    ],
                    "system_integration": [
                        "System compatibility assessed",
                        "Integration testing completed",
                        "Performance testing completed",
                        "Security testing completed",
                        "User acceptance testing completed"
                    ],
                    "go_live": [
                        "Go-live readiness assessed",
                        "Go-live executed successfully",
                        "Post-go-live monitoring established",
                        "Issue resolution procedures active",
                        "Performance validated"
                    ]
                },
                "use_monitoring_stage": {
                    "performance_monitoring": [
                        "Performance metrics defined",
                        "Monitoring frequency established",
                        "Alerting procedures active",
                        "Performance analysis conducted",
                        "Performance reporting established"
                    ],
                    "model_drift_detection": [
                        "Drift detection methods implemented",
                        "Thresholds defined",
                        "Drift analysis procedures active",
                        "Drift response procedures established",
                        "Drift documentation maintained"
                    ],
                    "business_environment_monitoring": [
                        "Business environment assessment completed",
                        "Market condition monitoring active",
                        "Regulatory change monitoring active",
                        "Impact assessment procedures established",
                        "Response planning completed"
                    ]
                },
                "changes_stage": {
                    "change_request_process": [
                        "Change request submitted",
                        "Change impact assessed",
                        "Change approved",
                        "Change implementation planned",
                        "Change testing procedures defined"
                    ],
                    "change_implementation": [
                        "Change implementation prepared",
                        "Change testing completed",
                        "Change deployed",
                        "Change validated",
                        "Change documented"
                    ],
                    "change_monitoring": [
                        "Change performance monitored",
                        "Change risk monitored",
                        "Change compliance monitored",
                        "Change issues identified",
                        "Change escalation procedures active"
                    ]
                },
                "retirement_stage": {
                    "retirement_planning": [
                        "Retirement criteria defined",
                        "Retirement impact assessed",
                        "Retirement timeline developed",
                        "Replacement model planned",
                        "Stakeholder communication completed"
                    ],
                    "retirement_execution": [
                        "Retirement execution planned",
                        "Data migration completed",
                        "System decommissioned",
                        "Documentation archived",
                        "Knowledge transferred"
                    ],
                    "post_retirement": [
                        "Post-retirement monitoring established",
                        "Lessons learned documented",
                        "Process improvements identified",
                        "Compliance verified",
                        "Final documentation completed"
                    ]
                }
            }
        
        # QA Scoring Weights
        if self.qa_scoring_weights is None:
            self.qa_scoring_weights = {
                "checklist_weights": {
                    "planning_stage": 0.15,
                    "development_stage": 0.20,
                    "validation_stage": 0.25,
                    "implementation_stage": 0.15,
                    "use_monitoring_stage": 0.15,
                    "changes_stage": 0.05,
                    "retirement_stage": 0.05
                },
                "severity_weights": {
                    "critical": 1.0,
                    "high": 0.7,
                    "medium": 0.4,
                    "low": 0.2,
                    "minimal": 0.1
                },
                "compliance_weights": {
                    "regulatory_compliance": 0.4,
                    "internal_policies": 0.3,
                    "best_practices": 0.2,
                    "industry_standards": 0.1
                }
            }
    
    def get_qa_methodology_description(self, methodology: QAMethodology) -> str:
        """Get description for QA methodology"""
        descriptions = {
            QAMethodology.SIX_SIGMA: "DMAIC methodology for process improvement and defect reduction",
            QAMethodology.ISO_9001: "Quality management system standards and certification",
            QAMethodology.AGILE_QA: "Iterative quality assurance practices for agile development",
            QAMethodology.DEVOPS_QA: "Continuous integration/continuous deployment quality gates",
            QAMethodology.MODEL_RISK_QA: "Specialized QA framework for model risk management",
            QAMethodology.STATISTICAL_QC: "Statistical process control and quality metrics",
            QAMethodology.PROCESS_IMPROVEMENT: "Kaizen and continuous process enhancement"
        }
        return descriptions.get(methodology, "QA methodology")
    
    def get_qa_tool_description(self, tool: QATool) -> str:
        """Get description for QA tool"""
        descriptions = {
            QATool.CHECKLISTS: "Structured QA checklists and scorecards",
            QATool.PROCESS_FLOW_ANALYSIS: "Workflow optimization and efficiency analysis",
            QATool.ROOT_CAUSE_ANALYSIS: "Problem identification and resolution techniques",
            QATool.STATISTICAL_PROCESS_CONTROL: "Statistical monitoring and control charts",
            QATool.QUALITY_METRICS_DASHBOARD: "KPIs, SLAs, and performance indicators",
            QATool.DEFECT_TRACKING: "Issue tracking and resolution workflows",
            QATool.CONTINUOUS_IMPROVEMENT: "Kaizen and process enhancement tools"
        }
        return descriptions.get(tool, "QA tool")
    
    def get_checklist_for_stage(self, stage: str, checklist_type: str = None) -> Dict[str, Any]:
        """Get QA checklist for specific stage and type"""
        stage_checklists = self.qa_checklists.get(stage, {})
        
        if checklist_type:
            return {checklist_type: stage_checklists.get(checklist_type, [])}
        
        return stage_checklists
    
    def get_scoring_weights(self, category: str) -> Dict[str, float]:
        """Get scoring weights for specific category"""
        return self.qa_scoring_weights.get(category, {})
    
    def get_qa_standards(self, standard: str) -> Dict[str, Any]:
        """Get QA standards for specific framework"""
        return self.qa_standards.get(standard, {})
    
    def get_best_practices(self, practice_area: str) -> List[str]:
        """Get best practices for specific area"""
        return self.qa_best_practices.get(practice_area, [])
