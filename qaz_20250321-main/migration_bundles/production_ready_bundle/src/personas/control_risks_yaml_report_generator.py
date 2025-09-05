#!/usr/bin/env python3
"""
Control Risks YAML Report Generator
"""

import json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime

from .control_risks_yaml_config import (
    ControlRisksYAMLConfig, 
    ControlRisk, 
    RiskCategory, 
    RiskSeverity, 
    ModelStage
)

@dataclass
class ControlRisksYAMLReport:
    """Comprehensive Control Risks Report from YAML data"""
    report_id: str
    report_date: datetime
    model_context: Dict[str, Any]
    risk_summary: Dict[str, Any]
    risk_details: List[ControlRisk]
    stage_analysis: Dict[str, Any]
    category_analysis: Dict[str, Any]
    severity_analysis: Dict[str, Any]
    controls_analysis: Dict[str, Any]
    recommendations: List[str]
    action_items: List[str]
    compliance_assessment: Dict[str, Any]
    risk_score: float
    overall_status: str

class ControlRisksYAMLReportGenerator:
    """Generator for comprehensive Control Risks Reports from YAML data"""
    
    def __init__(self, yaml_config: ControlRisksYAMLConfig):
        self.yaml_config = yaml_config
    
    def generate_control_risks_report(self, model_context: Dict[str, Any], 
                                    model_stage: ModelStage = None,
                                    risk_categories: List[RiskCategory] = None,
                                    risk_severities: List[RiskSeverity] = None) -> ControlRisksYAMLReport:
        """Generate comprehensive Control Risks Report from YAML data"""
        
        # Filter risks based on criteria
        applicable_risks = self._filter_applicable_risks(
            model_stage, risk_categories, risk_severities
        )
        
        # Generate risk summary
        risk_summary = self._generate_risk_summary(applicable_risks)
        
        # Generate stage analysis
        stage_analysis = self._generate_stage_analysis(applicable_risks)
        
        # Generate category analysis
        category_analysis = self._generate_category_analysis(applicable_risks)
        
        # Generate severity analysis
        severity_analysis = self._generate_severity_analysis(applicable_risks)
        
        # Generate controls analysis
        controls_analysis = self._generate_controls_analysis(applicable_risks)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(applicable_risks, risk_summary)
        
        # Generate action items
        action_items = self._generate_action_items(applicable_risks, recommendations)
        
        # Generate compliance assessment
        compliance_assessment = self._generate_compliance_assessment(applicable_risks)
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(applicable_risks, risk_summary)
        
        # Determine overall status
        overall_status = self._determine_overall_status(risk_score, risk_summary)
        
        return ControlRisksYAMLReport(
            report_id=f"CRR-YAML-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            report_date=datetime.now(),
            model_context=model_context,
            risk_summary=risk_summary,
            risk_details=applicable_risks,
            stage_analysis=stage_analysis,
            category_analysis=category_analysis,
            severity_analysis=severity_analysis,
            controls_analysis=controls_analysis,
            recommendations=recommendations,
            action_items=action_items,
            compliance_assessment=compliance_assessment,
            risk_score=risk_score,
            overall_status=overall_status
        )
    
    def _filter_applicable_risks(self, model_stage: ModelStage = None,
                               risk_categories: List[RiskCategory] = None,
                               risk_severities: List[RiskSeverity] = None) -> List[ControlRisk]:
        """Filter risks based on criteria"""
        
        applicable_risks = self.yaml_config.control_risks.copy()
        
        # Filter by model stage
        if model_stage:
            stage_name = model_stage.value
            applicable_risks = [
                risk for risk in applicable_risks 
                if getattr(risk.stage_distribution, stage_name, 0) > 0
            ]
        
        # Filter by risk categories
        if risk_categories:
            applicable_risks = [
                risk for risk in applicable_risks 
                if risk.risk_category in risk_categories
            ]
        
        # Filter by risk severities
        if risk_severities:
            applicable_risks = [
                risk for risk in applicable_risks 
                if risk.risk_severity in risk_severities
            ]
        
        return applicable_risks
    
    def _generate_risk_summary(self, risks: List[ControlRisk]) -> Dict[str, Any]:
        """Generate comprehensive risk summary"""
        
        total_risks = len(risks)
        total_controls = sum(risk.total_controls for risk in risks)
        
        # Severity distribution
        severity_dist = {}
        for risk in risks:
            severity = risk.risk_severity.value
            if severity not in severity_dist:
                severity_dist[severity] = 0
            severity_dist[severity] += 1
        
        # Category distribution
        category_dist = {}
        for risk in risks:
            category = risk.risk_category.value
            if category not in category_dist:
                category_dist[category] = 0
            category_dist[category] += 1
        
        # Stage distribution
        stage_dist = {}
        for risk in risks:
            stages_with_controls = risk.stage_distribution.get_stages_with_controls()
            for stage in stages_with_controls:
                if stage not in stage_dist:
                    stage_dist[stage] = 0
                stage_dist[stage] += 1
        
        # Controls distribution
        controls_dist = {
            'total_controls': total_controls,
            'average_controls_per_risk': total_controls / total_risks if total_risks > 0 else 0,
            'controls_by_stage': self._calculate_controls_by_stage(risks)
        }
        
        return {
            'total_risks': total_risks,
            'total_controls': total_controls,
            'severity_distribution': severity_dist,
            'category_distribution': category_dist,
            'stage_distribution': stage_dist,
            'controls_distribution': controls_dist,
            'critical_risks': severity_dist.get('critical', 0),
            'high_risks': severity_dist.get('high', 0),
            'medium_risks': severity_dist.get('medium', 0),
            'low_risks': severity_dist.get('low', 0),
            'minimal_risks': severity_dist.get('minimal', 0)
        }
    
    def _calculate_controls_by_stage(self, risks: List[ControlRisk]) -> Dict[str, int]:
        """Calculate total controls by stage"""
        
        controls_by_stage = {
            'planning': 0,
            'development': 0,
            'validation': 0,
            'implementation': 0,
            'use_and_monitoring': 0,
            'changes': 0,
            'retirement': 0
        }
        
        for risk in risks:
            controls_by_stage['planning'] += risk.stage_distribution.planning
            controls_by_stage['development'] += risk.stage_distribution.development
            controls_by_stage['validation'] += risk.stage_distribution.validation
            controls_by_stage['implementation'] += risk.stage_distribution.implementation
            controls_by_stage['use_and_monitoring'] += risk.stage_distribution.use_and_monitoring
            controls_by_stage['changes'] += risk.stage_distribution.changes
            controls_by_stage['retirement'] += risk.stage_distribution.retirement
        
        return controls_by_stage
    
    def _generate_stage_analysis(self, risks: List[ControlRisk]) -> Dict[str, Any]:
        """Generate stage-based analysis"""
        
        stage_analysis = {}
        
        for stage in ModelStage:
            stage_name = stage.value
            stage_risks = [
                risk for risk in risks 
                if getattr(risk.stage_distribution, stage_name, 0) > 0
            ]
            
            if stage_risks:
                total_controls = sum(
                    getattr(risk.stage_distribution, stage_name, 0) 
                    for risk in stage_risks
                )
                
                stage_analysis[stage_name] = {
                    'risk_count': len(stage_risks),
                    'total_controls': total_controls,
                    'average_controls_per_risk': total_controls / len(stage_risks) if stage_risks else 0,
                    'severity_distribution': self._get_severity_distribution(stage_risks),
                    'category_distribution': self._get_category_distribution(stage_risks),
                    'critical_risks': [r for r in stage_risks if r.risk_severity == RiskSeverity.CRITICAL],
                    'high_risks': [r for r in stage_risks if r.risk_severity == RiskSeverity.HIGH]
                }
        
        return stage_analysis
    
    def _generate_category_analysis(self, risks: List[ControlRisk]) -> Dict[str, Any]:
        """Generate category-based analysis"""
        
        category_analysis = {}
        
        for category in RiskCategory:
            category_risks = [risk for risk in risks if risk.risk_category == category]
            
            if category_risks:
                total_controls = sum(risk.total_controls for risk in category_risks)
                
                category_analysis[category.value] = {
                    'risk_count': len(category_risks),
                    'total_controls': total_controls,
                    'average_controls_per_risk': total_controls / len(category_risks) if category_risks else 0,
                    'severity_distribution': self._get_severity_distribution(category_risks),
                    'stage_distribution': self._get_stage_distribution(category_risks),
                    'critical_risks': [r for r in category_risks if r.risk_severity == RiskSeverity.CRITICAL],
                    'high_risks': [r for r in category_risks if r.risk_severity == RiskSeverity.HIGH]
                }
        
        return category_analysis
    
    def _generate_severity_analysis(self, risks: List[ControlRisk]) -> Dict[str, Any]:
        """Generate severity-based analysis"""
        
        severity_analysis = {}
        
        for severity in RiskSeverity:
            severity_risks = [risk for risk in risks if risk.risk_severity == severity]
            
            if severity_risks:
                total_controls = sum(risk.total_controls for risk in severity_risks)
                
                severity_analysis[severity.value] = {
                    'risk_count': len(severity_risks),
                    'total_controls': total_controls,
                    'average_controls_per_risk': total_controls / len(severity_risks) if severity_risks else 0,
                    'category_distribution': self._get_category_distribution(severity_risks),
                    'stage_distribution': self._get_stage_distribution(severity_risks)
                }
        
        return severity_analysis
    
    def _generate_controls_analysis(self, risks: List[ControlRisk]) -> Dict[str, Any]:
        """Generate controls-based analysis"""
        
        controls_analysis = {
            'total_controls': sum(risk.total_controls for risk in risks),
            'average_controls_per_risk': sum(risk.total_controls for risk in risks) / len(risks) if risks else 0,
            'controls_by_stage': self._calculate_controls_by_stage(risks),
            'controls_by_category': self._calculate_controls_by_category(risks),
            'controls_by_severity': self._calculate_controls_by_severity(risks),
            'high_control_risks': [r for r in risks if r.total_controls > 20],
            'low_control_risks': [r for r in risks if r.total_controls < 5]
        }
        
        return controls_analysis
    
    def _calculate_controls_by_category(self, risks: List[ControlRisk]) -> Dict[str, int]:
        """Calculate total controls by category"""
        
        controls_by_category = {}
        
        for risk in risks:
            category = risk.risk_category.value
            if category not in controls_by_category:
                controls_by_category[category] = 0
            controls_by_category[category] += risk.total_controls
        
        return controls_by_category
    
    def _calculate_controls_by_severity(self, risks: List[ControlRisk]) -> Dict[str, int]:
        """Calculate total controls by severity"""
        
        controls_by_severity = {}
        
        for risk in risks:
            severity = risk.risk_severity.value
            if severity not in controls_by_severity:
                controls_by_severity[severity] = 0
            controls_by_severity[severity] += risk.total_controls
        
        return controls_by_severity
    
    def _generate_recommendations(self, risks: List[ControlRisk], 
                                risk_summary: Dict[str, Any]) -> List[str]:
        """Generate risk-based recommendations"""
        
        recommendations = []
        
        # Critical risks recommendations
        if risk_summary['critical_risks'] > 0:
            recommendations.append(f"Immediate attention required for {risk_summary['critical_risks']} critical risks")
            recommendations.append("Implement emergency mitigation strategies for critical risks")
        
        # High risks recommendations
        if risk_summary['high_risks'] > 0:
            recommendations.append(f"Prioritize mitigation of {risk_summary['high_risks']} high risks")
            recommendations.append("Develop comprehensive mitigation plans for high risks")
        
        # Controls recommendations
        total_controls = risk_summary['total_controls']
        if total_controls > 1000:
            recommendations.append("Consider consolidating controls to reduce complexity")
        elif total_controls < 100:
            recommendations.append("Consider adding additional controls for comprehensive coverage")
        
        # Stage-specific recommendations
        controls_by_stage = risk_summary['controls_distribution']['controls_by_stage']
        for stage, controls in controls_by_stage.items():
            if controls == 0:
                recommendations.append(f"Add controls for {stage} stage")
            elif controls < 10:
                recommendations.append(f"Strengthen controls for {stage} stage")
        
        return recommendations
    
    def _generate_action_items(self, risks: List[ControlRisk], 
                             recommendations: List[str]) -> List[str]:
        """Generate specific action items"""
        
        action_items = []
        
        # Critical risks actions
        critical_risks = [r for r in risks if r.risk_severity == RiskSeverity.CRITICAL]
        for risk in critical_risks:
            action_items.append(f"Immediate action required: {risk.template_risk_name}")
            action_items.append(f"Assign owner for {risk.risk_id}: {risk.risk_owner}")
        
        # High risks actions
        high_risks = [r for r in risks if r.risk_severity == RiskSeverity.HIGH]
        for risk in high_risks:
            action_items.append(f"Develop mitigation plan for {risk.risk_id}: {risk.template_risk_name}")
        
        # Low control risks actions
        low_control_risks = [r for r in risks if r.total_controls < 5]
        for risk in low_control_risks:
            action_items.append(f"Review controls for {risk.risk_id}: {risk.template_risk_name}")
        
        # General actions
        action_items.append("Schedule regular risk review meetings")
        action_items.append("Update risk mitigation strategies")
        action_items.append("Enhance risk monitoring procedures")
        
        return action_items
    
    def _generate_compliance_assessment(self, risks: List[ControlRisk]) -> Dict[str, Any]:
        """Generate compliance assessment"""
        
        compliance_assessment = {
            'regulatory_compliance': {},
            'internal_compliance': {},
            'overall_compliance_score': 0.0
        }
        
        # Regulatory compliance
        regulatory_risks = [r for r in risks if r.risk_category == RiskCategory.REGULATORY_RISKS]
        if regulatory_risks:
            compliance_assessment['regulatory_compliance'] = {
                'total_risks': len(regulatory_risks),
                'total_controls': sum(r.total_controls for r in regulatory_risks),
                'critical_risks': len([r for r in regulatory_risks if r.risk_severity == RiskSeverity.CRITICAL]),
                'high_risks': len([r for r in regulatory_risks if r.risk_severity == RiskSeverity.HIGH]),
                'compliance_score': self._calculate_compliance_score(regulatory_risks)
            }
        
        # Internal compliance
        internal_risks = [r for r in risks if r.risk_category in [
            RiskCategory.GOVERNANCE_RISKS
        ]]
        if internal_risks:
            compliance_assessment['internal_compliance'] = {
                'total_risks': len(internal_risks),
                'total_controls': sum(r.total_controls for r in internal_risks),
                'critical_risks': len([r for r in internal_risks if r.risk_severity == RiskSeverity.CRITICAL]),
                'high_risks': len([r for r in internal_risks if r.risk_severity == RiskSeverity.HIGH]),
                'compliance_score': self._calculate_compliance_score(internal_risks)
            }
        
        # Overall compliance score
        compliance_assessment['overall_compliance_score'] = self._calculate_compliance_score(risks)
        
        return compliance_assessment
    
    def _calculate_risk_score(self, risks: List[ControlRisk], 
                            risk_summary: Dict[str, Any]) -> float:
        """Calculate overall risk score"""
        
        if not risks:
            return 0.0
        
        # Weighted risk score based on severity and controls
        severity_weights = {
            RiskSeverity.CRITICAL: 1.0,
            RiskSeverity.HIGH: 0.7,
            RiskSeverity.MEDIUM: 0.4,
            RiskSeverity.LOW: 0.2,
            RiskSeverity.MINIMAL: 0.1
        }
        
        total_weighted_score = 0.0
        total_risks = len(risks)
        
        for risk in risks:
            weight = severity_weights.get(risk.risk_severity, 0.0)
            # Adjust weight based on controls (more controls = lower risk)
            control_factor = max(0.1, 1.0 - (risk.total_controls / 100.0))
            weight *= control_factor
            total_weighted_score += weight
        
        # Normalize to 0-1 scale
        risk_score = total_weighted_score / total_risks
        
        return min(risk_score, 1.0)
    
    def _determine_overall_status(self, risk_score: float, 
                                risk_summary: Dict[str, Any]) -> str:
        """Determine overall risk status"""
        
        if risk_score >= 0.8 or risk_summary['critical_risks'] > 5:
            return "CRITICAL"
        elif risk_score >= 0.6 or risk_summary['critical_risks'] > 0:
            return "HIGH"
        elif risk_score >= 0.4 or risk_summary['high_risks'] > 10:
            return "MEDIUM"
        elif risk_score >= 0.2:
            return "LOW"
        else:
            return "MINIMAL"
    
    def _get_severity_distribution(self, risks: List[ControlRisk]) -> Dict[str, int]:
        """Get severity distribution for risks"""
        
        dist = {}
        for risk in risks:
            severity = risk.risk_severity.value
            if severity not in dist:
                dist[severity] = 0
            dist[severity] += 1
        
        return dist
    
    def _get_category_distribution(self, risks: List[ControlRisk]) -> Dict[str, int]:
        """Get category distribution for risks"""
        
        dist = {}
        for risk in risks:
            category = risk.risk_category.value
            if category not in dist:
                dist[category] = 0
            dist[category] += 1
        
        return dist
    
    def _get_stage_distribution(self, risks: List[ControlRisk]) -> Dict[str, int]:
        """Get stage distribution for risks"""
        
        dist = {}
        for risk in risks:
            stages_with_controls = risk.stage_distribution.get_stages_with_controls()
            for stage in stages_with_controls:
                if stage not in dist:
                    dist[stage] = 0
                dist[stage] += 1
        
        return dist
    
    def _calculate_compliance_score(self, risks: List[ControlRisk]) -> float:
        """Calculate compliance score"""
        
        if not risks:
            return 1.0
        
        # Compliance score based on controls and severity
        total_score = 0.0
        for risk in risks:
            # Higher controls = higher compliance
            control_score = min(1.0, risk.total_controls / 20.0)
            # Lower severity = higher compliance
            severity_scores = {
                RiskSeverity.CRITICAL: 0.2,
                RiskSeverity.HIGH: 0.4,
                RiskSeverity.MEDIUM: 0.6,
                RiskSeverity.LOW: 0.8,
                RiskSeverity.MINIMAL: 1.0
            }
            severity_score = severity_scores.get(risk.risk_severity, 0.5)
            
            # Combined score
            combined_score = (control_score + severity_score) / 2.0
            total_score += combined_score
        
        return total_score / len(risks)
    
    def generate_json_report(self, report: ControlRisksYAMLReport, output_path: str = None) -> str:
        """Generate JSON format report"""
        
        # Convert ControlRisk objects to dictionaries
        risk_details = []
        for risk in report.risk_details:
            risk_dict = {
                'risk_id': risk.risk_id,
                'template_risk_name': risk.template_risk_name,
                'template_risk_description': risk.template_risk_description,
                'total_controls': risk.total_controls,
                'stage_distribution': {
                    'planning': risk.stage_distribution.planning,
                    'development': risk.stage_distribution.development,
                    'validation': risk.stage_distribution.validation,
                    'implementation': risk.stage_distribution.implementation,
                    'use_and_monitoring': risk.stage_distribution.use_and_monitoring,
                    'changes': risk.stage_distribution.changes,
                    'retirement': risk.stage_distribution.retirement
                },
                'total_stages': risk.total_stages,
                'risk_category': risk.risk_category.value,
                'risk_severity': risk.risk_severity.value,
                'risk_owner': risk.risk_owner,
                'risk_assessor': risk.risk_assessor,
                'risk_indicators': risk.risk_indicators,
                'mitigation_strategies': risk.mitigation_strategies,
                'monitoring_metrics': risk.monitoring_metrics,
                'escalation_triggers': risk.escalation_triggers,
                'documentation_requirements': risk.documentation_requirements,
                'compliance_requirements': risk.compliance_requirements
            }
            risk_details.append(risk_dict)
        
        # Convert stage analysis to remove ControlRisk objects
        stage_analysis_clean = {}
        for stage_name, analysis in report.stage_analysis.items():
            stage_analysis_clean[stage_name] = {
                'risk_count': analysis['risk_count'],
                'total_controls': analysis['total_controls'],
                'average_controls_per_risk': analysis['average_controls_per_risk'],
                'severity_distribution': analysis['severity_distribution'],
                'category_distribution': analysis['category_distribution'],
                'critical_risks_count': len(analysis['critical_risks']),
                'high_risks_count': len(analysis['high_risks'])
            }
        
        # Convert category analysis to remove ControlRisk objects
        category_analysis_clean = {}
        for category_name, analysis in report.category_analysis.items():
            category_analysis_clean[category_name] = {
                'risk_count': analysis['risk_count'],
                'total_controls': analysis['total_controls'],
                'average_controls_per_risk': analysis['average_controls_per_risk'],
                'severity_distribution': analysis['severity_distribution'],
                'stage_distribution': analysis['stage_distribution'],
                'critical_risks_count': len(analysis['critical_risks']),
                'high_risks_count': len(analysis['high_risks'])
            }
        
        # Convert controls analysis to remove ControlRisk objects
        controls_analysis_clean = {
            'total_controls': report.controls_analysis['total_controls'],
            'average_controls_per_risk': report.controls_analysis['average_controls_per_risk'],
            'controls_by_stage': report.controls_analysis['controls_by_stage'],
            'controls_by_category': report.controls_analysis['controls_by_category'],
            'controls_by_severity': report.controls_analysis['controls_by_severity'],
            'high_control_risks_count': len(report.controls_analysis['high_control_risks']),
            'low_control_risks_count': len(report.controls_analysis['low_control_risks'])
        }
        
        report_data = {
            'report_id': report.report_id,
            'report_date': report.report_date.isoformat(),
            'model_context': report.model_context,
            'risk_summary': report.risk_summary,
            'risk_details': risk_details,
            'stage_analysis': stage_analysis_clean,
            'category_analysis': category_analysis_clean,
            'severity_analysis': report.severity_analysis,
            'controls_analysis': controls_analysis_clean,
            'recommendations': report.recommendations,
            'action_items': report.action_items,
            'compliance_assessment': report.compliance_assessment,
            'risk_score': report.risk_score,
            'overall_status': report.overall_status
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report_data, f, indent=2)
        
        return json.dumps(report_data, indent=2)
