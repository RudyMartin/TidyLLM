#!/usr/bin/env python3
"""
Argument Consistency Analysis Demo

Demonstrates analysis of document logical consistency and determination of 
appropriate review scope based on detected inconsistencies and risk factors.

Part of the tidyllm-verse: Educational ML with complete transparency
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tidyllm_compliance import ConsistencyAnalyzer

def main():
    """Demonstrate argument consistency analysis."""
    print("ARGUMENT CONSISTENCY ANALYSIS")
    print("=" * 45)
    print("Logic analysis and review scope determination\n")
    
    # Sample document with potential inconsistencies
    analysis_document = """
    INVESTMENT RECOMMENDATION REPORT
    
    Executive Summary:
    Based on our analysis, we recommend increasing allocation to technology stocks
    because the sector shows strong growth potential and regulatory tailwinds.
    
    Market Analysis:
    Technology stocks have consistently outperformed the market over the past 5 years.
    However, recent earnings reports show declining profit margins across major tech companies.
    
    Risk Assessment:
    We consider this investment strategy low risk given the sector's historical performance.
    Although regulatory changes could create significant headwinds, we believe
    the material impact on returns will be minimal.
    
    Financial Impact:
    The proposed allocation change represents $2.5 million in additional exposure,
    which could increase portfolio volatility by 15%.
    
    Therefore, we conclude this strategy offers excellent risk-adjusted returns
    despite the high risk nature of technology investments.
    
    Recommendation: Immediate implementation is required due to urgent market conditions.
    """
    
    # Initialize analyzer
    analyzer = ConsistencyAnalyzer()
    
    # Analyze document
    results = analyzer.analyze_document(analysis_document)
    
    # Display results
    print(f"Consistency Score: {results['consistency_score']:.1%}")
    print(f"Logical Structure Score: {results['logical_structure_score']:.1%}")
    print(f"Review Scope: {results['review_scope_recommendation'].replace('_', ' ').title()}")
    print(f"Priority Level: {results['priority_level'].title()}")
    
    if results['identified_issues']:
        print("\n[ISSUES IDENTIFIED]:")
        for issue in results['identified_issues']:
            print(f"   - {issue}")
    
    print("\nScope Determination Factors:")
    for factor, details in results['scope_factors'].items():
        if details['found']:
            print(f"   [OK] {factor.replace('_', ' ').title()}: {details['count']} instances")
            if details['examples']:
                print(f"      Examples: {details['examples'][:2]}")
    
    # Show analysis framework summary
    framework = analyzer.get_analysis_framework_summary()
    print(f"\n[ANALYSIS FRAMEWORK]")
    print(f"Logical Structure Patterns: {len(framework['logical_structure_patterns'])}")
    print(f"Scope Criteria: {len(framework['scope_criteria'])}")
    print(f"Contradiction Patterns: {framework['contradiction_pattern_count']}")
    print(f"Review Scope Options: {framework['review_scope_options']}")
    print(f"Priority Levels: {framework['priority_levels']}")
    
    print("\n" + "=" * 45)
    print("Argument consistency analysis complete!")

if __name__ == "__main__":
    main()