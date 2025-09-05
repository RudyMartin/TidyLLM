#!/usr/bin/env python3
"""
Convert Control Risks Text File to YAML Configuration
"""

import yaml
import os
from typing import Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class ControlRiskData:
    """Control Risk data structure"""
    risk_id: str
    risk_name: str
    risk_description: str
    total_controls: int
    planning: int
    development: int
    validation: int
    implementation: int
    use_and_monitoring: int
    changes: int
    retirement: int
    total_stages: int

def parse_control_risks_file(file_path: str) -> List[ControlRiskData]:
    """Parse the control risks text file"""
    
    control_risks = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Skip header line
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        
        # Split by pipe delimiter
        parts = line.split('|')
        if len(parts) >= 12:
            try:
                control_risk = ControlRiskData(
                    risk_id=parts[0].strip(),
                    risk_name=parts[1].strip(),
                    risk_description=parts[2].strip(),
                    total_controls=int(parts[3].strip()),
                    planning=int(parts[4].strip()),
                    development=int(parts[5].strip()),
                    validation=int(parts[6].strip()),
                    implementation=int(parts[7].strip()),
                    use_and_monitoring=int(parts[8].strip()),
                    changes=int(parts[9].strip()),
                    retirement=int(parts[10].strip()),
                    total_stages=int(parts[11].strip())
                )
                control_risks.append(control_risk)
            except (ValueError, IndexError) as e:
                print(f"Error parsing line: {line}")
                print(f"Error: {e}")
                continue
    
    return control_risks

def determine_risk_category(risk_name: str) -> str:
    """Determine risk category based on risk name"""
    
    risk_name_lower = risk_name.lower()
    
    # Development risks
    if any(keyword in risk_name_lower for keyword in ['development', 'design', 'implementation', 'coding']):
        return 'development_risks'
    
    # Validation risks
    if any(keyword in risk_name_lower for keyword in ['validation', 'testing', 'backtesting', 'stress testing']):
        return 'validation_risks'
    
    # Implementation risks
    if any(keyword in risk_name_lower for keyword in ['implementation', 'deployment', 'go-live', 'production']):
        return 'implementation_risks'
    
    # Monitoring risks
    if any(keyword in risk_name_lower for keyword in ['monitoring', 'performance', 'drift', 'threshold']):
        return 'monitoring_risks'
    
    # Governance risks
    if any(keyword in risk_name_lower for keyword in ['governance', 'approval', 'compliance', 'policy']):
        return 'governance_risks'
    
    # Data risks
    if any(keyword in risk_name_lower for keyword in ['data', 'privacy', 'security', 'information']):
        return 'data_risks'
    
    # Model risks
    if any(keyword in risk_name_lower for keyword in ['model', 'algorithm', 'bias', 'accuracy']):
        return 'model_risks'
    
    # Business risks
    if any(keyword in risk_name_lower for keyword in ['business', 'stakeholder', 'communication', 'objective']):
        return 'business_risks'
    
    # Regulatory risks
    if any(keyword in risk_name_lower for keyword in ['regulatory', 'compliance', 'audit', 'regulatory']):
        return 'regulatory_risks'
    
    # Technical risks
    if any(keyword in risk_name_lower for keyword in ['technical', 'infrastructure', 'system', 'technology']):
        return 'technical_risks'
    
    # Default to governance risks
    return 'governance_risks'

def determine_risk_severity(total_controls: int, total_stages: int) -> str:
    """Determine risk severity based on controls and stages"""
    
    # Higher controls and stages = higher severity
    severity_score = (total_controls / 25.0) + (total_stages / 7.0)
    
    if severity_score >= 1.5:
        return 'critical'
    elif severity_score >= 1.2:
        return 'high'
    elif severity_score >= 0.9:
        return 'medium'
    elif severity_score >= 0.6:
        return 'low'
    else:
        return 'minimal'

def generate_risk_indicators(risk_name: str, risk_description: str) -> List[str]:
    """Generate risk indicators based on risk name and description"""
    
    indicators = []
    
    # Common indicators based on risk patterns
    if 'timely' in risk_name.lower() or 'delayed' in risk_name.lower():
        indicators.append("Delays in process execution")
        indicators.append("Missed deadlines")
    
    if 'incomplete' in risk_name.lower() or 'missing' in risk_name.lower():
        indicators.append("Incomplete documentation")
        indicators.append("Missing deliverables")
    
    if 'inaccurate' in risk_name.lower() or 'error' in risk_name.lower():
        indicators.append("Data quality issues")
        indicators.append("Calculation errors")
    
    if 'governance' in risk_name.lower() or 'compliance' in risk_name.lower():
        indicators.append("Policy violations")
        indicators.append("Compliance gaps")
    
    if 'monitoring' in risk_name.lower() or 'performance' in risk_name.lower():
        indicators.append("Performance degradation")
        indicators.append("Monitoring failures")
    
    # Default indicators
    if not indicators:
        indicators.append("Process inefficiencies")
        indicators.append("Quality control issues")
    
    return indicators

def generate_mitigation_strategies(risk_category: str, risk_name: str) -> List[str]:
    """Generate mitigation strategies based on risk category and name"""
    
    strategies = []
    
    if risk_category == 'development_risks':
        strategies.extend([
            "Implement development standards",
            "Establish code review processes",
            "Enhance testing procedures"
        ])
    elif risk_category == 'validation_risks':
        strategies.extend([
            "Strengthen validation framework",
            "Implement comprehensive testing",
            "Establish validation oversight"
        ])
    elif risk_category == 'implementation_risks':
        strategies.extend([
            "Improve deployment procedures",
            "Enhance system integration",
            "Establish rollback procedures"
        ])
    elif risk_category == 'monitoring_risks':
        strategies.extend([
            "Implement monitoring framework",
            "Establish alerting procedures",
            "Enhance performance tracking"
        ])
    elif risk_category == 'governance_risks':
        strategies.extend([
            "Strengthen governance framework",
            "Establish approval processes",
            "Enhance compliance monitoring"
        ])
    elif risk_category == 'data_risks':
        strategies.extend([
            "Implement data governance",
            "Enhance data quality controls",
            "Establish data security measures"
        ])
    else:
        strategies.extend([
            "Establish risk mitigation procedures",
            "Implement quality controls",
            "Enhance oversight processes"
        ])
    
    return strategies

def generate_monitoring_metrics(risk_category: str) -> List[str]:
    """Generate monitoring metrics based on risk category"""
    
    metrics = []
    
    if risk_category == 'development_risks':
        metrics.extend([
            "Development timeline adherence",
            "Code quality metrics",
            "Testing coverage"
        ])
    elif risk_category == 'validation_risks':
        metrics.extend([
            "Validation completion rate",
            "Testing accuracy",
            "Validation findings"
        ])
    elif risk_category == 'implementation_risks':
        metrics.extend([
            "Implementation success rate",
            "System performance",
            "Integration effectiveness"
        ])
    elif risk_category == 'monitoring_risks':
        metrics.extend([
            "Monitoring coverage",
            "Alert response time",
            "Performance metrics"
        ])
    elif risk_category == 'governance_risks':
        metrics.extend([
            "Policy compliance rate",
            "Approval cycle time",
            "Governance effectiveness"
        ])
    elif risk_category == 'data_risks':
        metrics.extend([
            "Data quality scores",
            "Security incident rate",
            "Privacy compliance"
        ])
    else:
        metrics.extend([
            "Process efficiency",
            "Quality metrics",
            "Risk indicators"
        ])
    
    return metrics

def generate_escalation_triggers(risk_severity: str) -> List[str]:
    """Generate escalation triggers based on risk severity"""
    
    if risk_severity == 'critical':
        return [
            "Immediate escalation required",
            "Critical threshold breaches",
            "System failures"
        ]
    elif risk_severity == 'high':
        return [
            "High-risk threshold breaches",
            "Significant process failures",
            "Compliance violations"
        ]
    elif risk_severity == 'medium':
        return [
            "Moderate threshold breaches",
            "Process inefficiencies",
            "Quality issues"
        ]
    else:
        return [
            "Minor threshold breaches",
            "Process delays",
            "Documentation gaps"
        ]

def generate_documentation_requirements(risk_category: str) -> List[str]:
    """Generate documentation requirements based on risk category"""
    
    requirements = []
    
    if risk_category == 'development_risks':
        requirements.extend([
            "Development documentation",
            "Technical specifications",
            "Code documentation"
        ])
    elif risk_category == 'validation_risks':
        requirements.extend([
            "Validation documentation",
            "Testing procedures",
            "Validation reports"
        ])
    elif risk_category == 'implementation_risks':
        requirements.extend([
            "Implementation documentation",
            "Deployment procedures",
            "System documentation"
        ])
    elif risk_category == 'monitoring_risks':
        requirements.extend([
            "Monitoring documentation",
            "Performance reports",
            "Alert procedures"
        ])
    elif risk_category == 'governance_risks':
        requirements.extend([
            "Governance documentation",
            "Policy documents",
            "Approval records"
        ])
    elif risk_category == 'data_risks':
        requirements.extend([
            "Data documentation",
            "Privacy policies",
            "Security procedures"
        ])
    else:
        requirements.extend([
            "Process documentation",
            "Quality procedures",
            "Risk documentation"
        ])
    
    return requirements

def generate_compliance_requirements(risk_category: str) -> List[str]:
    """Generate compliance requirements based on risk category"""
    
    requirements = []
    
    if risk_category == 'governance_risks':
        requirements.extend([
            "SR11-7 governance standards",
            "Internal governance policies",
            "Regulatory compliance requirements"
        ])
    elif risk_category == 'data_risks':
        requirements.extend([
            "Data protection regulations",
            "Privacy compliance standards",
            "Information security standards"
        ])
    elif risk_category == 'validation_risks':
        requirements.extend([
            "Model validation standards",
            "Testing compliance requirements",
            "Quality assurance standards"
        ])
    elif risk_category == 'monitoring_risks':
        requirements.extend([
            "Monitoring compliance standards",
            "Performance monitoring requirements",
            "Risk monitoring standards"
        ])
    else:
        requirements.extend([
            "SR11-7 model risk management",
            "Industry best practices",
            "Internal compliance standards"
        ])
    
    return requirements

def convert_to_yaml_format(control_risks: List[ControlRiskData]) -> Dict[str, Any]:
    """Convert control risks to YAML format"""
    
    yaml_data = {
        'control_risks': []
    }
    
    for risk in control_risks:
        # Determine risk category and severity
        risk_category = determine_risk_category(risk.risk_name)
        risk_severity = determine_risk_severity(risk.total_controls, risk.total_stages)
        
        # Generate additional fields
        risk_indicators = generate_risk_indicators(risk.risk_name, risk.risk_description)
        mitigation_strategies = generate_mitigation_strategies(risk_category, risk.risk_name)
        monitoring_metrics = generate_monitoring_metrics(risk_category)
        escalation_triggers = generate_escalation_triggers(risk_severity)
        documentation_requirements = generate_documentation_requirements(risk_category)
        compliance_requirements = generate_compliance_requirements(risk_category)
        
        risk_data = {
            'risk_id': risk.risk_id,
            'template_risk_name': risk.risk_name,
            'template_risk_description': risk.risk_description,
            'total_controls': risk.total_controls,
            'stage_distribution': {
                'planning': risk.planning,
                'development': risk.development,
                'validation': risk.validation,
                'implementation': risk.implementation,
                'use_and_monitoring': risk.use_and_monitoring,
                'changes': risk.changes,
                'retirement': risk.retirement
            },
            'total_stages': risk.total_stages,
            'risk_category': risk_category,
            'risk_severity': risk_severity,
            'risk_owner': 'Model Owner',
            'risk_assessor': 'Model Risk Manager',
            'risk_indicators': risk_indicators,
            'mitigation_strategies': mitigation_strategies,
            'monitoring_metrics': monitoring_metrics,
            'escalation_triggers': escalation_triggers,
            'documentation_requirements': documentation_requirements,
            'compliance_requirements': compliance_requirements
        }
        
        yaml_data['control_risks'].append(risk_data)
    
    return yaml_data

def main():
    """Main conversion function"""
    
    # Input and output file paths
    input_file = "input/control_risks_with_biased_metrics.txt"
    output_file = "config/control_risks.yaml"
    
    # Ensure config directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Converting {input_file} to {output_file}...")
    
    # Parse the input file
    control_risks = parse_control_risks_file(input_file)
    print(f"Parsed {len(control_risks)} control risks")
    
    # Convert to YAML format
    yaml_data = convert_to_yaml_format(control_risks)
    
    # Write to YAML file
    with open(output_file, 'w', encoding='utf-8') as file:
        yaml.dump(yaml_data, file, default_flow_style=False, indent=2, allow_unicode=True)
    
    print(f"Successfully created {output_file}")
    
    # Print summary
    print("\nConversion Summary:")
    print(f"- Total risks converted: {len(control_risks)}")
    
    # Count by category
    categories = {}
    severities = {}
    for risk in yaml_data['control_risks']:
        category = risk['risk_category']
        severity = risk['risk_severity']
        
        categories[category] = categories.get(category, 0) + 1
        severities[severity] = severities.get(severity, 0) + 1
    
    print("\nRisk Categories:")
    for category, count in sorted(categories.items()):
        print(f"  - {category}: {count}")
    
    print("\nRisk Severities:")
    for severity, count in sorted(severities.items()):
        print(f"  - {severity}: {count}")
    
    # Calculate total controls
    total_controls = sum(risk['total_controls'] for risk in yaml_data['control_risks'])
    print(f"\nTotal Controls: {total_controls}")
    print(f"Average Controls per Risk: {total_controls / len(control_risks):.1f}")

if __name__ == "__main__":
    main()
