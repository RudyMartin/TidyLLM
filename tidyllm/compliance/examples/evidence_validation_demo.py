#!/usr/bin/env python3
"""
Evidence Validation Demo

Demonstrates automated assessment of document authenticity and evidential value
for audit and investigation purposes.

Part of the tidyllm-verse: Educational ML with complete transparency
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tidyllm_compliance import EvidenceValidator

def main():
    """Demonstrate evidence validation monitoring."""
    print("EVIDENCE VALIDATION MONITORING")
    print("=" * 45)
    print("Authenticity and completeness assessment for audit evidence\n")
    
    # Sample audit evidence document
    evidence_document = """
    AUDIT EVIDENCE REPORT
    Internal Control Assessment Q4 2023
    
    Prepared by: Sarah Johnson, CPA
    Date: January 10, 2024
    Reviewed by: Michael Chen, CIA
    
    Executive Summary:
    This report presents findings from the quarterly assessment of internal controls
    over financial reporting for the fourth quarter of 2023.
    
    Methodology:
    We performed walk-through testing of 25 key controls and substantive testing
    of 150 transactions across all material business processes.
    
    Data Analysis:
    Our testing revealed 3 control deficiencies affecting accounts payable processing,
    with error rates of 2.3% compared to 1.1% in the previous quarter.
    
    Statistical analysis shows p-value of 0.02, indicating statistically significant
    deterioration in control effectiveness.
    
    Conclusions:
    Management should implement corrective actions for the identified deficiencies
    within 60 days to maintain SOX compliance.
    
    Cross-references: See Appendix A for detailed test results and Exhibit B for
    management responses.
    
    Data validation was performed by the Quality Assurance team and confirmed
    accuracy of all testing procedures.
    """
    
    # Initialize validator
    validator = EvidenceValidator()
    
    # Validate document
    results = validator.validate_document(evidence_document)
    
    # Display results
    print(f"Overall Validity: {results['overall_validity'].replace('_', ' ').title()}")
    print(f"Authenticity Score: {results['authenticity_score']:.1%}")
    print(f"Completeness Score: {results['completeness_score']:.1%}")
    print(f"Quality Score: {results['quality_score']:.1%}")
    
    print("\nKey Findings:")
    for finding, value in results['findings'].items():
        if value and value is not False:
            print(f"   [OK] {finding.replace('_', ' ').title()}: {value}")
    
    if results['recommendations']:
        print("\n[RECOMMENDATIONS]:")
        for rec in results['recommendations']:
            print(f"   - {rec}")
    
    # Show validation criteria summary
    criteria = validator.get_validation_criteria_summary()
    print(f"\n[VALIDATION CRITERIA]")
    print(f"Total Criteria: {criteria['total_criteria']}")
    print(f"Authenticity Indicators: {len(criteria['authenticity_indicators'])}")
    print(f"Completeness Requirements: {len(criteria['completeness_requirements'])}")
    print(f"Quality Indicators: {len(criteria['quality_indicators'])}")
    
    print("\n" + "=" * 45)
    print("Evidence validation assessment complete!")

if __name__ == "__main__":
    main()