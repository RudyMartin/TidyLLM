#!/usr/bin/env python3
"""
Code Review DSPy Signature
==========================

DSPy signature definition for code review workflow integration with TidyLLM's
existing DSPy infrastructure. Follows established patterns from core/ai_wrapper.py
and other TidyLLM DSPy implementations.
"""

import dspy
from typing import Dict, Any, Optional
from dataclasses import dataclass

class CodeReviewSignature(dspy.Signature):
    """Comprehensive code review analysis using DSPy Chain of Thought"""

    # Input fields
    code_changes = dspy.InputField(desc="Code changes to review (diff format or full code)")
    architectural_specs = dspy.InputField(desc="Architectural ground truth from specs.json")
    test_coverage_report = dspy.InputField(desc="Test coverage metrics and analysis")
    security_scan_results = dspy.InputField(desc="Security vulnerability scan results")
    performance_metrics = dspy.InputField(desc="Performance benchmark data")

    # Output fields - structured for markdown generation
    architectural_compliance_analysis = dspy.OutputField(desc="Detailed architectural compliance assessment")
    security_compliance_analysis = dspy.OutputField(desc="Security validation and S3-first compliance check")
    dependency_validation_analysis = dspy.OutputField(desc="Forbidden dependency analysis with alternatives")
    performance_impact_analysis = dspy.OutputField(desc="Performance impact assessment and recommendations")
    regulatory_compliance_analysis = dspy.OutputField(desc="SR-11-7, Basel-III, SOX-404 compliance validation")

    # Decision outputs
    overall_score = dspy.OutputField(desc="Overall review score 0.0-1.0")
    approval_decision = dspy.OutputField(desc="APPROVED|APPROVED_WITH_CONDITIONS|CHANGES_REQUESTED|REJECTED")
    critical_issues = dspy.OutputField(desc="List of blocking/critical issues requiring immediate attention")
    recommendations = dspy.OutputField(desc="Prioritized list of improvements and fixes")
    escalation_required = dspy.OutputField(desc="true/false - whether architect/security review needed")

    # Structured markdown report
    markdown_report = dspy.OutputField(desc="Complete code review report in DSPy-optimized markdown format")

class ArchitecturalReviewSignature(dspy.Signature):
    """Specialized architectural compliance review signature"""

    # Input fields
    code_changes = dspy.InputField(desc="Code changes focusing on architectural patterns")
    architectural_ground_truth = dspy.InputField(desc="specs.json architectural specifications")
    current_architecture = dspy.InputField(desc="Current system architecture context")

    # Output fields
    hexagonal_compliance = dspy.OutputField(desc="Hexagonal architecture pattern compliance assessment")
    adapter_pattern_usage = dspy.OutputField(desc="Proper adapter (not gateway) pattern validation")
    layer_separation_analysis = dspy.OutputField(desc="Clean architecture layer separation validation")
    interface_pattern_compliance = dspy.OutputField(desc="TidyLLM interface pattern adherence check")
    architectural_violations = dspy.OutputField(desc="List of architectural violations with severity")
    remediation_guidance = dspy.OutputField(desc="Step-by-step remediation instructions")

class SecurityReviewSignature(dspy.Signature):
    """Specialized security compliance review signature"""

    # Input fields
    code_changes = dspy.InputField(desc="Code changes with security focus")
    s3_requirements = dspy.InputField(desc="S3-first architecture requirements")
    security_policies = dspy.InputField(desc="Corporate security policies and constraints")

    # Output fields
    s3_first_compliance = dspy.OutputField(desc="S3-first data handling compliance validation")
    app_folder_cleanup_check = dspy.OutputField(desc="App folder cleanup implementation verification")
    credential_security_analysis = dspy.OutputField(desc="Credential management and secrets handling review")
    data_security_patterns = dspy.OutputField(desc="Data encryption and access control validation")
    security_vulnerabilities = dspy.OutputField(desc="Identified security vulnerabilities with severity")
    security_recommendations = dspy.OutputField(desc="Security improvement recommendations")

class DependencyReviewSignature(dspy.Signature):
    """Specialized dependency constraint validation signature"""

    # Input fields
    code_changes = dspy.InputField(desc="Code changes with focus on imports and dependencies")
    forbidden_dependencies = dspy.InputField(desc="List of forbidden dependencies from specs.json")
    approved_alternatives = dspy.InputField(desc="Approved alternatives mapping")

    # Output fields
    forbidden_dependency_violations = dspy.OutputField(desc="List of forbidden dependencies detected")
    dependency_alternatives = dspy.OutputField(desc="Recommended alternatives for each violation")
    migration_guidance = dspy.OutputField(desc="Step-by-step migration instructions")
    infrastructure_sovereignty_impact = dspy.OutputField(desc="Impact on infrastructure sovereignty goals")

@dataclass
class CodeReviewDSPyProcessor:
    """
    DSPy processor for code review workflow
    Integrates with TidyLLM's existing DSPy infrastructure
    """

    def __init__(self, ai_wrapper=None):
        """Initialize with TidyLLM AI wrapper if available"""
        self.ai_wrapper = ai_wrapper

    def process_comprehensive_review(self,
                                   code_changes: str,
                                   metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process comprehensive code review using DSPy Chain of Thought

        Args:
            code_changes: Code to review
            metadata: Additional context (test coverage, security scans, etc.)

        Returns:
            Structured review results with markdown report
        """
        try:
            # Configure DSPy if AI wrapper available
            if self.ai_wrapper and hasattr(self.ai_wrapper, 'ai_gateway'):
                dspy.configure(lm=self.ai_wrapper.ai_gateway.get_dspy_model())

            # Execute comprehensive review
            cot_processor = dspy.ChainOfThought(CodeReviewSignature)

            result = cot_processor(
                code_changes=code_changes,
                architectural_specs=metadata.get('architectural_specs', ''),
                test_coverage_report=metadata.get('test_coverage_report', ''),
                security_scan_results=metadata.get('security_scan_results', ''),
                performance_metrics=metadata.get('performance_metrics', '')
            )

            return {
                'success': True,
                'overall_score': float(result.overall_score),
                'approval_decision': result.approval_decision,
                'critical_issues': result.critical_issues,
                'recommendations': result.recommendations,
                'escalation_required': result.escalation_required.lower() == 'true',
                'markdown_report': result.markdown_report,
                'detailed_analysis': {
                    'architectural_compliance': result.architectural_compliance_analysis,
                    'security_compliance': result.security_compliance_analysis,
                    'dependency_validation': result.dependency_validation_analysis,
                    'performance_impact': result.performance_impact_analysis,
                    'regulatory_compliance': result.regulatory_compliance_analysis
                }
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'fallback_analysis': 'DSPy processing failed, using fallback analysis'
            }

    def process_specialized_review(self,
                                 code_changes: str,
                                 review_type: str,
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process specialized review (architectural, security, or dependency)

        Args:
            code_changes: Code to review
            review_type: 'architectural', 'security', or 'dependency'
            context: Relevant context for the review type

        Returns:
            Specialized review results
        """
        try:
            if review_type == 'architectural':
                processor = dspy.ChainOfThought(ArchitecturalReviewSignature)
                result = processor(
                    code_changes=code_changes,
                    architectural_ground_truth=context.get('architectural_specs', ''),
                    current_architecture=context.get('current_architecture', '')
                )

                return {
                    'review_type': 'architectural',
                    'hexagonal_compliance': result.hexagonal_compliance,
                    'adapter_pattern_usage': result.adapter_pattern_usage,
                    'violations': result.architectural_violations,
                    'guidance': result.remediation_guidance
                }

            elif review_type == 'security':
                processor = dspy.ChainOfThought(SecurityReviewSignature)
                result = processor(
                    code_changes=code_changes,
                    s3_requirements=context.get('s3_requirements', ''),
                    security_policies=context.get('security_policies', '')
                )

                return {
                    'review_type': 'security',
                    's3_compliance': result.s3_first_compliance,
                    'app_cleanup_check': result.app_folder_cleanup_check,
                    'vulnerabilities': result.security_vulnerabilities,
                    'recommendations': result.security_recommendations
                }

            elif review_type == 'dependency':
                processor = dspy.ChainOfThought(DependencyReviewSignature)
                result = processor(
                    code_changes=code_changes,
                    forbidden_dependencies=context.get('forbidden_deps', ''),
                    approved_alternatives=context.get('alternatives', '')
                )

                return {
                    'review_type': 'dependency',
                    'violations': result.forbidden_dependency_violations,
                    'alternatives': result.dependency_alternatives,
                    'migration_guidance': result.migration_guidance
                }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'review_type': review_type
            }

# Example usage and integration patterns
def create_dspy_code_review_report(code_changes: str, metadata: Dict[str, Any]) -> str:
    """
    Create a DSPy-optimized markdown report for code review
    Following TidyLLM's established DSPy patterns
    """

    processor = CodeReviewDSPyProcessor()
    result = processor.process_comprehensive_review(code_changes, metadata)

    if result['success']:
        return result['markdown_report']
    else:
        return f"""
# Code Review Report - Processing Error

**Status**: Error during DSPy processing
**Error**: {result.get('error', 'Unknown error')}
**Fallback**: {result.get('fallback_analysis', 'No fallback available')}

## Recommendations
1. Verify DSPy infrastructure configuration
2. Check TidyLLM AI wrapper initialization
3. Validate input parameters and context
"""

if __name__ == "__main__":
    # Example usage
    sample_code = """
import pandas as pd
import numpy as np

class DataGateway:
    def process_data(self):
        df = pd.read_csv('local_file.csv')
        return df.to_dict()
"""

    sample_metadata = {
        'test_coverage_report': 'Coverage: 65%',
        'security_scan_results': 'No critical vulnerabilities found',
        'architectural_specs': 'Hexagonal architecture required, no pandas/numpy',
        'performance_metrics': 'Baseline performance: 2.3s response time'
    }

    report = create_dspy_code_review_report(sample_code, sample_metadata)
    print("DSPy Code Review Report Generated:")
    print(report)