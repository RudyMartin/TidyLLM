"""
Unified Compliance Framework for TidyLLM Enterprise

Central mapping and management system that unifies:
1. Analysis layer compliance rules (SR 11-7, OCC, etc.)
2. Workflow layer compliance frameworks (SOX, GDPR, HIPAA, etc.)
3. Cross-framework dependencies and relationships
4. Unified reporting and audit trail integration

This creates a single source of truth for all compliance requirements.
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import json

class ComplianceCategory(Enum):
    """High-level compliance categories"""
    FINANCIAL_REGULATION = "financial_regulation"
    DATA_PRIVACY = "data_privacy" 
    INFORMATION_SECURITY = "information_security"
    CORPORATE_GOVERNANCE = "corporate_governance"
    INDUSTRY_SPECIFIC = "industry_specific"
    INTERNAL_POLICY = "internal_policy"

class RegulationType(Enum):
    """Types of regulatory requirements"""
    MANDATORY = "mandatory"          # Legal/regulatory requirement
    RECOMMENDED = "recommended"      # Industry best practice
    INTERNAL = "internal"           # Company policy
    CONTRACTUAL = "contractual"     # Client/partner requirement

@dataclass
class ComplianceRequirement:
    """Individual compliance requirement"""
    requirement_id: str
    title: str
    description: str
    framework_source: str
    category: ComplianceCategory
    regulation_type: RegulationType
    severity: str  # critical, high, medium, low
    
    # Cross-references
    related_requirements: List[str] = field(default_factory=list)
    conflicts_with: List[str] = field(default_factory=list)
    
    # Implementation details
    validation_patterns: Dict[str, str] = field(default_factory=dict)
    required_evidence: List[str] = field(default_factory=list)
    automation_level: str = "manual"  # automated, semi_automated, manual

@dataclass
class ComplianceMapping:
    """Mapping between different compliance frameworks"""
    source_framework: str
    target_framework: str
    source_requirement_id: str
    target_requirement_id: str
    mapping_type: str  # equivalent, related, conflicts, extends
    confidence: float  # 0.0 to 1.0
    notes: str = ""

class UnifiedComplianceFramework:
    """Central compliance framework management system"""
    
    def __init__(self):
        self.requirements: Dict[str, ComplianceRequirement] = {}
        self.mappings: List[ComplianceMapping] = []
        self.frameworks: Dict[str, Dict[str, Any]] = {}
        
        # Initialize with standard frameworks
        self._initialize_standard_frameworks()
    
    def _initialize_standard_frameworks(self):
        """Initialize standard compliance frameworks"""
        
        # Financial Regulation (from analysis layer)
        self._add_financial_requirements()
        
        # Corporate Governance (from workflow layer) 
        self._add_corporate_governance_requirements()
        
        # Data Privacy
        self._add_privacy_requirements()
        
        # Information Security
        self._add_security_requirements()
        
        # Create cross-framework mappings
        self._create_framework_mappings()
    
    def _add_financial_requirements(self):
        """Add financial regulation requirements (SR 11-7, OCC)"""
        
        # Model Development Documentation (SR 11-7)
        self.add_requirement(ComplianceRequirement(
            requirement_id="FIN-SR11-7-001",
            title="Model Development Documentation",
            description="Comprehensive documentation of model development process",
            framework_source="Federal Reserve SR 11-7",
            category=ComplianceCategory.FINANCIAL_REGULATION,
            regulation_type=RegulationType.MANDATORY,
            severity="critical",
            validation_patterns={
                'business_purpose': r'(?:business purpose|objective|intended use)[\s\w]*:?\s*([^\n\.]+)',
                'data_sources': r'(?:data source|dataset|training data)[\s\w]*:?\s*([^\n\.]+)',
                'methodology': r'(?:methodology|approach|algorithm|model type)[\s\w]*:?\s*([^\n\.]+)',
            },
            required_evidence=[
                "Business purpose statement",
                "Data source documentation", 
                "Model methodology description",
                "Validation test results"
            ],
            automation_level="semi_automated"
        ))
        
        # Model Validation (SR 11-7)
        self.add_requirement(ComplianceRequirement(
            requirement_id="FIN-SR11-7-002", 
            title="Independent Model Validation",
            description="Independent validation of model performance and accuracy",
            framework_source="Federal Reserve SR 11-7",
            category=ComplianceCategory.FINANCIAL_REGULATION,
            regulation_type=RegulationType.MANDATORY,
            severity="high",
            validation_patterns={
                'independent_validation': r'(?:independent|separate|third.?party)[\s\w]*(?:validation|review|testing)',
                'out_of_sample': r'(?:out.?of.?sample|holdout|test set)[\s\w]*(?:testing|validation)',
            },
            required_evidence=[
                "Independent validation report",
                "Out-of-sample testing results",
                "Sensitivity analysis"
            ],
            automation_level="semi_automated"
        ))
    
    def _add_corporate_governance_requirements(self):
        """Add corporate governance requirements (SOX, etc.)"""
        
        # SOX Section 404 - Internal Controls
        self.add_requirement(ComplianceRequirement(
            requirement_id="CORP-SOX-404-001",
            title="Internal Control Documentation",
            description="Document and test internal controls over financial reporting",
            framework_source="Sarbanes-Oxley Act Section 404",
            category=ComplianceCategory.CORPORATE_GOVERNANCE,
            regulation_type=RegulationType.MANDATORY,
            severity="critical",
            required_evidence=[
                "Internal control documentation",
                "Control testing results",
                "Management attestation"
            ],
            automation_level="manual"
        ))
        
        # Board Oversight
        self.add_requirement(ComplianceRequirement(
            requirement_id="CORP-GOV-001",
            title="Board Risk Committee Approval",
            description="Board-level approval for significant risk decisions",
            framework_source="Corporate Governance Best Practices",
            category=ComplianceCategory.CORPORATE_GOVERNANCE,
            regulation_type=RegulationType.RECOMMENDED,
            severity="high",
            required_evidence=[
                "Board meeting minutes",
                "Risk committee approval",
                "Stakeholder sign-offs"
            ],
            automation_level="manual"
        ))
    
    def _add_privacy_requirements(self):
        """Add data privacy requirements (GDPR, HIPAA)"""
        
        # GDPR Article 25 - Data Protection by Design
        self.add_requirement(ComplianceRequirement(
            requirement_id="PRIV-GDPR-025-001",
            title="Data Protection by Design",
            description="Implement privacy protections from system design phase",
            framework_source="GDPR Article 25",
            category=ComplianceCategory.DATA_PRIVACY,
            regulation_type=RegulationType.MANDATORY,
            severity="high",
            required_evidence=[
                "Privacy impact assessment",
                "Data processing documentation",
                "Technical safeguards documentation"
            ],
            automation_level="semi_automated"
        ))
        
        # HIPAA Security Rule
        self.add_requirement(ComplianceRequirement(
            requirement_id="PRIV-HIPAA-SEC-001",
            title="Administrative Safeguards", 
            description="Implement administrative safeguards for PHI",
            framework_source="HIPAA Security Rule",
            category=ComplianceCategory.DATA_PRIVACY,
            regulation_type=RegulationType.MANDATORY,
            severity="critical",
            required_evidence=[
                "Security policies and procedures",
                "Access control documentation",
                "Training records"
            ],
            automation_level="semi_automated"
        ))
    
    def _add_security_requirements(self):
        """Add information security requirements (ISO 27001, NIST)"""
        
        # ISO 27001 - Information Security Management
        self.add_requirement(ComplianceRequirement(
            requirement_id="SEC-ISO27001-001",
            title="Information Security Policy",
            description="Establish and maintain information security management system",
            framework_source="ISO 27001",
            category=ComplianceCategory.INFORMATION_SECURITY,
            regulation_type=RegulationType.RECOMMENDED,
            severity="high",
            required_evidence=[
                "Security policy document",
                "Risk assessment results",
                "Security control implementation"
            ],
            automation_level="semi_automated"
        ))
        
        # NIST Cybersecurity Framework
        self.add_requirement(ComplianceRequirement(
            requirement_id="SEC-NIST-CSF-001",
            title="Asset Management",
            description="Identify and manage information system assets",
            framework_source="NIST Cybersecurity Framework",
            category=ComplianceCategory.INFORMATION_SECURITY,
            regulation_type=RegulationType.RECOMMENDED,
            severity="medium",
            required_evidence=[
                "Asset inventory",
                "Asset classification scheme",
                "Asset management procedures"
            ],
            automation_level="automated"
        ))
    
    def _create_framework_mappings(self):
        """Create mappings between different frameworks"""
        
        # SR 11-7 Model Governance <-> SOX Internal Controls
        self.add_mapping(ComplianceMapping(
            source_framework="Federal Reserve SR 11-7",
            target_framework="Sarbanes-Oxley Act Section 404",
            source_requirement_id="FIN-SR11-7-001",
            target_requirement_id="CORP-SOX-404-001",
            mapping_type="related",
            confidence=0.8,
            notes="Model documentation supports internal control requirements"
        ))
        
        # GDPR Privacy <-> ISO 27001 Security
        self.add_mapping(ComplianceMapping(
            source_framework="GDPR Article 25",
            target_framework="ISO 27001",
            source_requirement_id="PRIV-GDPR-025-001", 
            target_requirement_id="SEC-ISO27001-001",
            mapping_type="extends",
            confidence=0.9,
            notes="Privacy by design extends information security controls"
        ))
    
    def add_requirement(self, requirement: ComplianceRequirement):
        """Add compliance requirement to framework"""
        self.requirements[requirement.requirement_id] = requirement
        
        # Track by framework
        framework = requirement.framework_source
        if framework not in self.frameworks:
            self.frameworks[framework] = {
                'requirements': [],
                'category': requirement.category.value,
                'regulation_type': requirement.regulation_type.value
            }
        self.frameworks[framework]['requirements'].append(requirement.requirement_id)
    
    def add_mapping(self, mapping: ComplianceMapping):
        """Add mapping between frameworks"""
        self.mappings.append(mapping)
    
    def get_requirements_by_category(self, category: ComplianceCategory) -> List[ComplianceRequirement]:
        """Get all requirements for a specific compliance category"""
        return [req for req in self.requirements.values() if req.category == category]
    
    def get_requirements_by_framework(self, framework: str) -> List[ComplianceRequirement]:
        """Get all requirements for a specific framework"""
        return [req for req in self.requirements.values() if req.framework_source == framework]
    
    def find_related_requirements(self, requirement_id: str) -> List[ComplianceRequirement]:
        """Find requirements related to the given requirement"""
        related_ids = set()
        
        # Direct relationships
        if requirement_id in self.requirements:
            req = self.requirements[requirement_id]
            related_ids.update(req.related_requirements)
        
        # Framework mappings
        for mapping in self.mappings:
            if mapping.source_requirement_id == requirement_id:
                related_ids.add(mapping.target_requirement_id)
            elif mapping.target_requirement_id == requirement_id:
                related_ids.add(mapping.source_requirement_id)
        
        return [self.requirements[req_id] for req_id in related_ids if req_id in self.requirements]
    
    def assess_compliance_coverage(self, assessed_requirements: Set[str]) -> Dict[str, Any]:
        """Assess compliance coverage across all frameworks"""
        
        coverage_by_category = {}
        coverage_by_framework = {}
        
        for category in ComplianceCategory:
            category_reqs = self.get_requirements_by_category(category)
            if category_reqs:
                covered = sum(1 for req in category_reqs if req.requirement_id in assessed_requirements)
                coverage_by_category[category.value] = {
                    'covered': covered,
                    'total': len(category_reqs),
                    'percentage': (covered / len(category_reqs)) * 100
                }
        
        for framework in self.frameworks:
            framework_reqs = self.get_requirements_by_framework(framework)
            if framework_reqs:
                covered = sum(1 for req in framework_reqs if req.requirement_id in assessed_requirements)
                coverage_by_framework[framework] = {
                    'covered': covered,
                    'total': len(framework_reqs),
                    'percentage': (covered / len(framework_reqs)) * 100
                }
        
        return {
            'overall_coverage': {
                'covered': len(assessed_requirements),
                'total': len(self.requirements),
                'percentage': (len(assessed_requirements) / len(self.requirements)) * 100 if self.requirements else 0
            },
            'by_category': coverage_by_category,
            'by_framework': coverage_by_framework,
            'gaps': self._identify_compliance_gaps(assessed_requirements)
        }
    
    def _identify_compliance_gaps(self, assessed_requirements: Set[str]) -> List[Dict[str, Any]]:
        """Identify compliance gaps and priorities"""
        gaps = []
        
        for req_id, requirement in self.requirements.items():
            if req_id not in assessed_requirements:
                gaps.append({
                    'requirement_id': req_id,
                    'title': requirement.title,
                    'framework': requirement.framework_source,
                    'category': requirement.category.value,
                    'severity': requirement.severity,
                    'regulation_type': requirement.regulation_type.value,
                    'automation_possible': requirement.automation_level != "manual"
                })
        
        # Sort by severity and regulation type
        severity_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        regulation_order = {'mandatory': 4, 'contractual': 3, 'recommended': 2, 'internal': 1}
        
        gaps.sort(key=lambda x: (
            severity_order.get(x['severity'], 0),
            regulation_order.get(x['regulation_type'], 0)
        ), reverse=True)
        
        return gaps
    
    def generate_compliance_matrix(self) -> Dict[str, Any]:
        """Generate comprehensive compliance matrix for enterprise reporting"""
        
        matrix = {
            'frameworks': {},
            'cross_references': [],
            'risk_assessment': {},
            'automation_opportunities': []
        }
        
        # Framework details
        for framework, details in self.frameworks.items():
            reqs = [self.requirements[req_id] for req_id in details['requirements']]
            
            matrix['frameworks'][framework] = {
                'total_requirements': len(reqs),
                'by_severity': self._count_by_attribute(reqs, 'severity'),
                'by_automation': self._count_by_attribute(reqs, 'automation_level'),
                'category': details['category'],
                'regulation_type': details['regulation_type']
            }
        
        # Cross-references
        for mapping in self.mappings:
            matrix['cross_references'].append({
                'source': f"{mapping.source_framework}:{mapping.source_requirement_id}",
                'target': f"{mapping.target_framework}:{mapping.target_requirement_id}",
                'relationship': mapping.mapping_type,
                'confidence': mapping.confidence
            })
        
        # Risk assessment
        critical_reqs = [req for req in self.requirements.values() if req.severity == 'critical']
        mandatory_reqs = [req for req in self.requirements.values() if req.regulation_type == RegulationType.MANDATORY]
        
        matrix['risk_assessment'] = {
            'critical_requirements': len(critical_reqs),
            'mandatory_requirements': len(mandatory_reqs),
            'high_risk_frameworks': [
                framework for framework, details in self.frameworks.items()
                if any(self.requirements[req_id].severity == 'critical' for req_id in details['requirements'])
            ]
        }
        
        # Automation opportunities
        for requirement in self.requirements.values():
            if requirement.automation_level == "semi_automated":
                matrix['automation_opportunities'].append({
                    'requirement_id': requirement.requirement_id,
                    'title': requirement.title,
                    'framework': requirement.framework_source,
                    'current_patterns': len(requirement.validation_patterns),
                    'potential_automation': "high" if requirement.validation_patterns else "medium"
                })
        
        return matrix
    
    def _count_by_attribute(self, requirements: List[ComplianceRequirement], attribute: str) -> Dict[str, int]:
        """Count requirements by a specific attribute"""
        counts = {}
        for req in requirements:
            value = getattr(req, attribute)
            if isinstance(value, Enum):
                value = value.value
            counts[str(value)] = counts.get(str(value), 0) + 1
        return counts

def create_framework_mapping() -> UnifiedComplianceFramework:
    """Factory function to create unified compliance framework"""
    return UnifiedComplianceFramework()