#!/usr/bin/env python3
"""
Model Risk Development Standards Compliance Demo

Demonstrates automated compliance checking against Federal Reserve SR 11-7 
and OCC model risk management guidance.

Part of the tidyllm-verse: Educational ML with complete transparency
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tidyllm_compliance import ModelRiskMonitor

def main():
    """Demonstrate model risk compliance monitoring."""
    print("MODEL RISK DEVELOPMENT STANDARDS COMPLIANCE")
    print("=" * 60)
    print("Automated checking against Federal Reserve SR 11-7 and OCC guidance\n")
    
    # Sample model development document
    model_document = """
    MODEL DEVELOPMENT DOCUMENTATION
    Credit Risk Scorecard Model v2.1
    
    Business Purpose: This model is designed to assess credit risk for consumer lending
    applications and support decision-making in the loan approval process.
    
    Data Sources: The model uses customer application data from our CRM system,
    credit bureau data from Experian and TransUnion, and historical loan performance
    data covering 5 years of loan originations.
    
    Methodology: The model employs logistic regression with feature engineering
    including debt-to-income ratios, payment history indicators, and credit utilization.
    
    Limitations: The model has limited performance for customers with thin credit files
    and may exhibit bias for certain demographic segments.
    
    Validation: Independent validation was performed by the Model Risk Management team
    using out-of-sample testing on 2023 data with AUC of 0.73.
    
    Monitoring: Monthly performance monitoring includes population stability index
    and characteristic analysis reports.
    
    Approved by: Model Risk Committee on January 15, 2024
    """
    
    # Initialize monitor
    monitor = ModelRiskMonitor()
    
    # Assess compliance
    results = monitor.assess_document_compliance(model_document)
    
    # Display results
    print(f"Overall Compliance Score: {results['overall_score']:.1%}")
    print(f"Rules Assessed: {len(results['rule_assessments'])}")
    
    print("\nDetailed Assessment:")
    for rule_id, assessment in results['rule_assessments'].items():
        print(f"\n[RULE] {rule_id}: {assessment['description']}")
        print(f"   Score: {assessment['compliance_score']:.1%}")
        print(f"   Severity: {assessment['severity']}")
        print(f"   Found Elements: {len(assessment['found_elements'])}")
        if assessment['missing_elements']:
            print(f"   Missing: {assessment['missing_elements'][:2]}...")
    
    if results['recommendations']:
        print("\n[RECOMMENDATIONS]:")
        for rec in results['recommendations']:
            print(f"   - {rec}")
    
    # Show standards summary
    summary = monitor.get_standards_summary()
    print(f"\n[STANDARDS SUMMARY]")
    print(f"Total Rules: {summary['total_rules']}")
    print(f"Rules by Severity: {summary['rules_by_severity']}")
    
    print("\n" + "=" * 60)
    print("Model risk compliance assessment complete!")

if __name__ == "__main__":
    main()