#!/usr/bin/env python3
"""
Compliance-Owned SOP Conflict Detection Demo
============================================

Demonstrates the integrated compliance system with:
- SOP golden answer validation
- YRSN noise analysis 
- Temporal conflict resolution
- Fallback guidance strategy

Uses tidyllm-compliance module as the single source of truth.
"""

import sys
from pathlib import Path

# Add tidyllm-compliance to path
sys.path.insert(0, str(Path(__file__).parent / 'tidyllm-compliance'))

from tidyllm_compliance import SOPValidator

def main():
    """Demonstrate compliance-owned SOP conflict detection"""
    
    print("=" * 60)
    print("COMPLIANCE-OWNED SOP CONFLICT DETECTION DEMO")
    print("=" * 60)
    print("Using tidyllm-compliance as single source of truth")
    print("=" * 60)
    
    # Initialize compliance validator
    validator = SOPValidator()
    
    # Test queries that demonstrate the system
    test_queries = [
        'What is the official session management pattern for TidyLLM?',
        'Which embedding system should be used: tidyllm-sentence or tidyllm-vectorqa?',
        'How should MVR documents be classified and tagged?',
        'What is the process for YRSN noise analysis validation?'
    ]
    
    print(f"\nTesting {len(test_queries)} SOP compliance queries...")
    
    # Run integrated conflict detection and resolution
    result = validator.detect_and_resolve_sop_conflicts(test_queries, "2025-09-05")
    
    print(f"\n{'='*60}")
    print("COMPLIANCE RESULTS")
    print("="*60)
    print(f"Report Type: {result['report_type']}")
    print(f"Overall Compliance Status: {result['overall_compliance_status']}")
    print(f"Number of Queries Analyzed: {len(result['sop_validation_results'])}")
    
    # Show detailed results
    print(f"\n{'='*60}")
    print("DETAILED SOP VALIDATION RESULTS")
    print("="*60)
    
    for i, sop_result in enumerate(result['sop_validation_results'], 1):
        print(f"\n[{i}] Query: {sop_result['query']}")
        print(f"    Compliance Status: {sop_result['compliance_status']}")
        
        if sop_result['sop_golden_answer']:
            print(f"    SOP Golden Answer: Available from {sop_result['sop_source']}")
            print(f"    SOP YRSN Noise Score: {sop_result['sop_yrsn_score']:.1f}%")
        else:
            print(f"    SOP Golden Answer: Not Available")
            
        print(f"    Fallback YRSN Noise Score: {sop_result['fallback_yrsn_score']:.1f}%")
        print(f"    Recommendation: {sop_result['recommendation']}")
    
    # Show integrated recommendations
    print(f"\n{'='*60}")
    print("INTEGRATED COMPLIANCE RECOMMENDATIONS")
    print("="*60)
    
    for i, rec in enumerate(result['integrated_recommendations'], 1):
        print(f"{i}. {rec}")
    
    # Show YRSN validation summary
    yrsn_report = result['conflict_reports']['yrsn_validation']
    print(f"\n{'='*60}")
    print("YRSN NOISE ANALYSIS SUMMARY")
    print("="*60)
    print(f"Average Noise Score: {yrsn_report['yrsn_metrics']['average_noise_score']:.1f}%")
    print(f"Highest Noise Score: {yrsn_report['yrsn_metrics']['highest_noise_score']:.1f}%")
    print(f"Queries Above 50% Noise: {yrsn_report['yrsn_metrics']['queries_above_50_percent_noise']}")
    print(f"Queries Above 70% Noise: {yrsn_report['yrsn_metrics']['queries_above_70_percent_noise']}")
    
    print(f"\n{'='*60}")
    print("DEMO COMPLETE")
    print("="*60)
    print("Compliance-owned conflict detection system operational!")
    print(f"Reports saved to: tidyllm-compliance/sop_conflict_reports/")
    print("The compliance module now owns all SOP conflict analysis.")

if __name__ == "__main__":
    main()