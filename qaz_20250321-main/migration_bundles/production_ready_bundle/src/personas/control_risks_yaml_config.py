#!/usr/bin/env python3
"""
Control Risks YAML Configuration System
"""

import yaml
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

class RiskCategory(Enum):
    """Categories of Control Risks"""
    DEVELOPMENT_RISKS = "development_risks"
    VALIDATION_RISKS = "validation_risks"
    IMPLEMENTATION_RISKS = "implementation_risks"
    MONITORING_RISKS = "monitoring_risks"
    GOVERNANCE_RISKS = "governance_risks"
    DATA_RISKS = "data_risks"
    MODEL_RISKS = "model_risks"
    BUSINESS_RISKS = "business_risks"
    REGULATORY_RISKS = "regulatory_risks"
    TECHNICAL_RISKS = "technical_risks"

class RiskSeverity(Enum):
    """Risk severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"

class ModelStage(Enum):
    """Model lifecycle stages"""
    PLANNING = "planning"
    DEVELOPMENT = "development"
    VALIDATION = "validation"
    IMPLEMENTATION = "implementation"
    USE_AND_MONITORING = "use_and_monitoring"
    CHANGES = "changes"
    RETIREMENT = "retirement"

@dataclass
class StageDistribution:
    """Stage distribution for a control risk"""
    planning: int = 0
    development: int = 0
    validation: int = 0
    implementation: int = 0
    use_and_monitoring: int = 0
    changes: int = 0
    retirement: int = 0
    
    def get_total_stages(self) -> int:
        """Calculate total stages with controls"""
        stages = [
            self.planning, self.development, self.validation, 
            self.implementation, self.use_and_monitoring, 
            self.changes, self.retirement
        ]
        return sum(1 for stage in stages if stage > 0)
    
    def get_stages_with_controls(self) -> List[str]:
        """Get list of stages that have controls"""
        stage_mapping = {
            'planning': self.planning,
            'development': self.development,
            'validation': self.validation,
            'implementation': self.implementation,
            'use_and_monitoring': self.use_and_monitoring,
            'changes': self.changes,
            'retirement': self.retirement
        }
        return [stage for stage, count in stage_mapping.items() if count > 0]

@dataclass
class ControlRisk:
    """Control Risk definition from YAML"""
    risk_id: str
    template_risk_name: str
    template_risk_description: str
    total_controls: int
    stage_distribution: StageDistribution
    total_stages: int
    risk_category: RiskCategory = RiskCategory.GOVERNANCE_RISKS
    risk_severity: RiskSeverity = RiskSeverity.MEDIUM
    risk_owner: str = "Model Owner"
    risk_assessor: str = "Model Risk Manager"
    risk_indicators: List[str] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)
    monitoring_metrics: List[str] = field(default_factory=list)
    escalation_triggers: List[str] = field(default_factory=list)
    documentation_requirements: List[str] = field(default_factory=list)
    compliance_requirements: List[str] = field(default_factory=list)

class ControlRisksYAMLConfig:
    """YAML-based configuration for Control Risks"""
    
    def __init__(self, yaml_file_path: str = None):
        self.yaml_file_path = yaml_file_path or "config/control_risks.yaml"
        self.control_risks: List[ControlRisk] = []
        self.risk_categories: Dict[RiskCategory, List[str]] = {}
        self.stage_mappings: Dict[ModelStage, List[str]] = {}
        self.severity_distributions: Dict[RiskSeverity, int] = {}
        
        self._load_control_risks()
        self._build_mappings()
    
    def _load_control_risks(self):
        """Load control risks from YAML file"""
        
        if not os.path.exists(self.yaml_file_path):
            self._create_default_yaml()
        
        with open(self.yaml_file_path, 'r') as file:
            data = yaml.safe_load(file)
        
        for risk_data in data.get('control_risks', []):
            control_risk = self._parse_risk_data(risk_data)
            self.control_risks.append(control_risk)
    
    def _parse_risk_data(self, risk_data: Dict[str, Any]) -> ControlRisk:
        """Parse risk data from YAML"""
        
        # Parse stage distribution
        stage_dist = StageDistribution(
            planning=risk_data.get('stage_distribution', {}).get('planning', 0),
            development=risk_data.get('stage_distribution', {}).get('development', 0),
            validation=risk_data.get('stage_distribution', {}).get('validation', 0),
            implementation=risk_data.get('stage_distribution', {}).get('implementation', 0),
            use_and_monitoring=risk_data.get('stage_distribution', {}).get('use_and_monitoring', 0),
            changes=risk_data.get('stage_distribution', {}).get('changes', 0),
            retirement=risk_data.get('stage_distribution', {}).get('retirement', 0)
        )
        
        # Parse risk category
        category_str = risk_data.get('risk_category', 'governance_risks')
        risk_category = self._parse_risk_category(category_str)
        
        # Parse risk severity
        severity_str = risk_data.get('risk_severity', 'medium')
        risk_severity = self._parse_risk_severity(severity_str)
        
        return ControlRisk(
            risk_id=risk_data['risk_id'],
            template_risk_name=risk_data['template_risk_name'],
            template_risk_description=risk_data['template_risk_description'],
            total_controls=risk_data['total_controls'],
            stage_distribution=stage_dist,
            total_stages=risk_data.get('total_stages', stage_dist.get_total_stages()),
            risk_category=risk_category,
            risk_severity=risk_severity,
            risk_owner=risk_data.get('risk_owner', 'Model Owner'),
            risk_assessor=risk_data.get('risk_assessor', 'Model Risk Manager'),
            risk_indicators=risk_data.get('risk_indicators', []),
            mitigation_strategies=risk_data.get('mitigation_strategies', []),
            monitoring_metrics=risk_data.get('monitoring_metrics', []),
            escalation_triggers=risk_data.get('escalation_triggers', []),
            documentation_requirements=risk_data.get('documentation_requirements', []),
            compliance_requirements=risk_data.get('compliance_requirements', [])
        )
    
    def _parse_risk_category(self, category_str: str) -> RiskCategory:
        """Parse risk category from string"""
        
        category_mapping = {
            'development_risks': RiskCategory.DEVELOPMENT_RISKS,
            'validation_risks': RiskCategory.VALIDATION_RISKS,
            'implementation_risks': RiskCategory.IMPLEMENTATION_RISKS,
            'monitoring_risks': RiskCategory.MONITORING_RISKS,
            'governance_risks': RiskCategory.GOVERNANCE_RISKS,
            'data_risks': RiskCategory.DATA_RISKS,
            'model_risks': RiskCategory.MODEL_RISKS,
            'business_risks': RiskCategory.BUSINESS_RISKS,
            'regulatory_risks': RiskCategory.REGULATORY_RISKS,
            'technical_risks': RiskCategory.TECHNICAL_RISKS
        }
        
        return category_mapping.get(category_str, RiskCategory.GOVERNANCE_RISKS)
    
    def _parse_risk_severity(self, severity_str: str) -> RiskSeverity:
        """Parse risk severity from string"""
        
        severity_mapping = {
            'critical': RiskSeverity.CRITICAL,
            'high': RiskSeverity.HIGH,
            'medium': RiskSeverity.MEDIUM,
            'low': RiskSeverity.LOW,
            'minimal': RiskSeverity.MINIMAL
        }
        
        return severity_mapping.get(severity_str, RiskSeverity.MEDIUM)
    
    def _create_default_yaml(self):
        """Create default YAML file with sample data"""
        
        default_data = {
            'control_risks': [
                {
                    'risk_id': '29',
                    'template_risk_name': 'Ineffective model communication',
                    'template_risk_description': 'The risk of ineffective communication between stakeholders',
                    'total_controls': 18,
                    'stage_distribution': {
                        'planning': 3,
                        'development': 1,
                        'validation': 8,
                        'implementation': 0,
                        'use_and_monitoring': 6,
                        'changes': 0,
                        'retirement': 0
                    },
                    'total_stages': 4,
                    'risk_category': 'governance_risks',
                    'risk_severity': 'medium',
                    'risk_owner': 'Model Owner',
                    'risk_assessor': 'Model Risk Manager',
                    'risk_indicators': [
                        'Stakeholder communication gaps',
                        'Misaligned expectations',
                        'Inadequate documentation sharing'
                    ],
                    'mitigation_strategies': [
                        'Establish communication protocols',
                        'Regular stakeholder meetings',
                        'Comprehensive documentation'
                    ],
                    'monitoring_metrics': [
                        'Communication effectiveness',
                        'Stakeholder satisfaction',
                        'Documentation completeness'
                    ],
                    'escalation_triggers': [
                        'Communication breakdowns',
                        'Stakeholder conflicts',
                        'Documentation gaps'
                    ],
                    'documentation_requirements': [
                        'Communication plan',
                        'Stakeholder register',
                        'Meeting minutes'
                    ],
                    'compliance_requirements': [
                        'SR11-7 governance standards',
                        'Internal communication policies'
                    ]
                }
            ]
        }
        
        # Ensure config directory exists
        os.makedirs(os.path.dirname(self.yaml_file_path), exist_ok=True)
        
        with open(self.yaml_file_path, 'w') as file:
            yaml.dump(default_data, file, default_flow_style=False, indent=2)
    
    def _build_mappings(self):
        """Build category and stage mappings"""
        
        # Build category mappings
        for risk in self.control_risks:
            category = risk.risk_category
            if category not in self.risk_categories:
                self.risk_categories[category] = []
            self.risk_categories[category].append(risk.risk_id)
        
        # Build stage mappings
        for risk in self.control_risks:
            stages_with_controls = risk.stage_distribution.get_stages_with_controls()
            for stage_name in stages_with_controls:
                stage = self._get_model_stage(stage_name)
                if stage not in self.stage_mappings:
                    self.stage_mappings[stage] = []
                self.stage_mappings[stage].append(risk.risk_id)
        
        # Build severity distributions
        for risk in self.control_risks:
            severity = risk.risk_severity
            if severity not in self.severity_distributions:
                self.severity_distributions[severity] = 0
            self.severity_distributions[severity] += 1
    
    def _get_model_stage(self, stage_name: str) -> ModelStage:
        """Get ModelStage enum from stage name"""
        
        stage_mapping = {
            'planning': ModelStage.PLANNING,
            'development': ModelStage.DEVELOPMENT,
            'validation': ModelStage.VALIDATION,
            'implementation': ModelStage.IMPLEMENTATION,
            'use_and_monitoring': ModelStage.USE_AND_MONITORING,
            'changes': ModelStage.CHANGES,
            'retirement': ModelStage.RETIREMENT
        }
        
        return stage_mapping.get(stage_name, ModelStage.PLANNING)
    
    def get_risk_by_id(self, risk_id: str) -> Optional[ControlRisk]:
        """Get control risk by ID"""
        
        for risk in self.control_risks:
            if risk.risk_id == risk_id:
                return risk
        
        return None
    
    def get_risks_by_category(self, category: RiskCategory) -> List[ControlRisk]:
        """Get risks by category"""
        
        return [risk for risk in self.control_risks if risk.risk_category == category]
    
    def get_risks_by_stage(self, stage: ModelStage) -> List[ControlRisk]:
        """Get risks by model stage"""
        
        stage_name = stage.value
        return [
            risk for risk in self.control_risks 
            if getattr(risk.stage_distribution, stage_name, 0) > 0
        ]
    
    def get_risks_by_severity(self, severity: RiskSeverity) -> List[ControlRisk]:
        """Get risks by severity"""
        
        return [risk for risk in self.control_risks if risk.risk_severity == severity]
    
    def get_total_risk_count(self) -> int:
        """Get total number of risks"""
        
        return len(self.control_risks)
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        
        return {
            'total_risks': self.get_total_risk_count(),
            'severity_distribution': {sev.value: count for sev, count in self.severity_distributions.items()},
            'category_distribution': {cat.value: len(risks) for cat, risks in self.risk_categories.items()},
            'stage_distribution': {stage.value: len(risks) for stage, risks in self.stage_mappings.items()},
            'total_controls': sum(risk.total_controls for risk in self.control_risks),
            'average_controls_per_risk': sum(risk.total_controls for risk in self.control_risks) / len(self.control_risks) if self.control_risks else 0
        }
