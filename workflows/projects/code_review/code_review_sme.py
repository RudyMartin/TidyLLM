#!/usr/bin/env python3
"""
Code Review SME (Subject Matter Expert) System
============================================

An expert system for complex code review decisions, architectural compliance validation,
and technical guidance for code review workflows.

This SME system provides specialized expertise in:
- Architectural pattern validation (hexagonal architecture, clean architecture)
- Security compliance (S3-first, app cleanup, credential management)
- Performance optimization and benchmarking
- Regulatory compliance (SR-11-7, Basel-III, SOX-404)
- TidyLLM integration patterns and best practices
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

class ReviewSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    BLOCKING = "blocking"

class ReviewCategory(Enum):
    ARCHITECTURE = "architecture"
    SECURITY = "security"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"
    DEPENDENCIES = "dependencies"
    PATTERNS = "patterns"

@dataclass
class CodeReviewIssue:
    """Represents a code review issue identified by the SME"""
    category: ReviewCategory
    severity: ReviewSeverity
    title: str
    description: str
    violation: str
    recommendation: str
    specs_reference: Optional[str] = None
    auto_fixable: bool = False
    requires_architect_review: bool = False

@dataclass
class CodeReviewAnalysis:
    """Complete code review analysis from the SME"""
    overall_score: float
    approval_status: str
    issues: List[CodeReviewIssue]
    architectural_compliance: bool
    security_compliance: bool
    performance_compliance: bool
    regulatory_compliance: bool
    summary: str
    recommendations: List[str]
    escalation_required: bool
    sme_consultation_areas: List[str]

class CodeReviewSME:
    """
    Subject Matter Expert system for comprehensive code review analysis

    This SME system provides expert-level analysis for:
    - Architectural ground truth compliance
    - Security pattern validation
    - Performance impact assessment
    - Regulatory compliance verification
    - Dependency constraint validation
    """

    def __init__(self, criteria_path: str = None, specs_path: str = "specs.json"):
        """Initialize the Code Review SME with both focused criteria and comprehensive specs"""
        if criteria_path is None:
            criteria_path = os.path.join(os.path.dirname(__file__), "code_review_criteria.json")
        self.criteria_path = criteria_path
        self.specs_path = specs_path
        self.review_criteria = self._load_review_criteria()
        self.system_specs = self._load_system_specs()  # For comprehensive reference
        self.forbidden_dependencies = self._get_forbidden_dependencies()
        self.security_requirements = self._get_security_requirements()
        self.performance_benchmarks = self._get_performance_benchmarks()

    def _load_review_criteria(self) -> Dict[str, Any]:
        """Load focused code review criteria from dedicated file"""
        try:
            with open(self.criteria_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load {self.criteria_path}: {e}. Using minimal constraints.")
            return self._get_default_criteria()

    def _load_system_specs(self) -> Dict[str, Any]:
        """Load comprehensive system specifications for reference"""
        try:
            # Try to import the safe JSON loader
            from tidyllm.utils.json_scrubber import safe_load_json_with_scrubbing

            result = safe_load_json_with_scrubbing(self.specs_path)
            if result['success']:
                if result.get('scrubbing_required'):
                    print(f"Note: {self.specs_path} required Unicode scrubbing to load properly")
                return result['data']
            else:
                print(f"Warning: Could not load {self.specs_path}: {result['error']}")
                return {}
        except ImportError:
            # Fallback to manual loading if scrubber not available
            try:
                with open(self.specs_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as e:
                print(f"Warning: Could not load {self.specs_path}: {e}. Comprehensive specs not available.")
                return {}

    def _get_forbidden_dependencies(self) -> List[str]:
        """Get list of forbidden dependencies from review criteria"""
        return self.review_criteria.get('forbidden_dependencies', [
            'numpy', 'pandas', 'sklearn', 'sentence_transformers',
            'tensorflow', 'pytorch', 'transformers'
        ])

    def _get_security_requirements(self) -> Dict[str, Any]:
        """Get security requirements from review criteria"""
        return self.review_criteria.get('security_requirements', {})

    def _get_performance_benchmarks(self) -> Dict[str, Any]:
        """Get performance benchmarks from review criteria"""
        return self.review_criteria.get('quality_thresholds', {
            'performance_degradation_maximum': 0.1,
            'test_coverage_minimum': 0.9
        })

    def _get_default_criteria(self) -> Dict[str, Any]:
        """Provide default criteria if file can't be loaded"""
        return {
            'forbidden_dependencies': ['numpy', 'pandas', 'sklearn', 'sentence_transformers'],
            'security_requirements': {'s3_first_mandatory': True, 'app_folder_cleanup_required': True},
            'quality_thresholds': {'test_coverage_minimum': 0.9, 'performance_degradation_maximum': 0.1},
            'architectural_patterns': {'required_patterns': ['hexagonal_architecture', 'adapter_pattern']},
            'compliance_standards': ['SR-11-7', 'Basel-III', 'SOX-404']
        }

    def analyze_code_changes(self, code_changes: str, metadata: Dict[str, Any] = None) -> CodeReviewAnalysis:
        """
        Perform comprehensive SME analysis of code changes

        Args:
            code_changes: The code changes to analyze
            metadata: Additional metadata about the changes

        Returns:
            CodeReviewAnalysis: Comprehensive analysis with issues and recommendations
        """
        if metadata is None:
            metadata = {}

        issues = []

        # Architectural compliance analysis
        arch_issues = self._analyze_architectural_compliance(code_changes, metadata)
        issues.extend(arch_issues)

        # Security compliance analysis
        security_issues = self._analyze_security_compliance(code_changes, metadata)
        issues.extend(security_issues)

        # Performance impact analysis
        performance_issues = self._analyze_performance_impact(code_changes, metadata)
        issues.extend(performance_issues)

        # Dependency constraint validation
        dependency_issues = self._analyze_dependency_constraints(code_changes)
        issues.extend(dependency_issues)

        # Regulatory compliance check
        compliance_issues = self._analyze_regulatory_compliance(code_changes, metadata)
        issues.extend(compliance_issues)

        # Calculate overall assessment
        analysis = self._generate_analysis_summary(issues, metadata)
        return analysis

    def _analyze_architectural_compliance(self, code_changes: str, metadata: Dict[str, Any]) -> List[CodeReviewIssue]:
        """Analyze code changes for architectural pattern compliance"""
        issues = []

        # Check for gateway vs adapter pattern violations
        if 'gateway' in code_changes.lower() and 'class' in code_changes.lower():
            issues.append(CodeReviewIssue(
                category=ReviewCategory.ARCHITECTURE,
                severity=ReviewSeverity.CRITICAL,
                title="Gateway Pattern Usage Detected",
                description="Code appears to implement gateway pattern instead of adapter pattern",
                violation="Using 'gateway' terminology violates hexagonal architecture principles",
                recommendation="Rename to use 'adapter' pattern and implement proper port interfaces",
                specs_reference="architectural_ground_truth.gateway_to_adapter_transformation",
                requires_architect_review=True
            ))

        # Check for proper layer separation
        if self._violates_layer_separation(code_changes):
            issues.append(CodeReviewIssue(
                category=ReviewCategory.ARCHITECTURE,
                severity=ReviewSeverity.CRITICAL,
                title="Clean Architecture Layer Violation",
                description="Code violates clean architecture layer separation principles",
                violation="Direct dependencies between layers that should be separated",
                recommendation="Implement proper dependency inversion through port interfaces",
                specs_reference="v2_official_clean_architecture.architecture_layers",
                requires_architect_review=True
            ))

        return issues

    def _analyze_security_compliance(self, code_changes: str, metadata: Dict[str, Any]) -> List[CodeReviewIssue]:
        """Analyze code changes for security compliance"""
        issues = []

        # Check for S3-first compliance violations
        local_storage_patterns = ['open(', 'with open', 'tempfile', '/tmp/', 'local_file']
        for pattern in local_storage_patterns:
            if pattern in code_changes and 's3' not in code_changes.lower():
                issues.append(CodeReviewIssue(
                    category=ReviewCategory.SECURITY,
                    severity=ReviewSeverity.BLOCKING,
                    title="Local Storage Usage Detected",
                    description=f"Code uses local storage pattern '{pattern}' without S3 integration",
                    violation="Violates S3-first security architecture",
                    recommendation="Replace with S3 streaming operations and ensure app folder cleanup",
                    specs_reference="s3_first_bidirectional_architecture.mandatory_patterns_app_cleanup_driven"
                ))

        # Check for credential hardcoding
        credential_patterns = ['password=', 'secret=', 'key=', 'token=', 'aws_access_key']
        for pattern in credential_patterns:
            if pattern in code_changes:
                issues.append(CodeReviewIssue(
                    category=ReviewCategory.SECURITY,
                    severity=ReviewSeverity.BLOCKING,
                    title="Hardcoded Credentials Detected",
                    description=f"Potential hardcoded credentials found with pattern '{pattern}'",
                    violation="Hardcoded credentials violate security best practices",
                    recommendation="Use AWS Secrets Manager or environment-based credential loading",
                    specs_reference="security_compliance.secret_scanning"
                ))

        return issues

    def _analyze_performance_impact(self, code_changes: str, metadata: Dict[str, Any]) -> List[CodeReviewIssue]:
        """Analyze code changes for performance impact"""
        issues = []

        # Check for performance anti-patterns
        performance_warnings = []

        if 'for' in code_changes and 'append' in code_changes:
            performance_warnings.append("List comprehension or vectorized operations preferred over loop+append")

        if 'import pandas' in code_changes:
            issues.append(CodeReviewIssue(
                category=ReviewCategory.PERFORMANCE,
                severity=ReviewSeverity.CRITICAL,
                title="Pandas Usage Violates Architecture",
                description="Code imports pandas, which is forbidden in favor of polars",
                violation="Violates dependency constraints for performance and architectural sovereignty",
                recommendation="Replace pandas usage with polars for better performance and compliance",
                specs_reference="architectural_ground_truth.forbidden_dependencies"
            ))

        # Check for test coverage metadata
        test_coverage = metadata.get('test_coverage', 0)
        if test_coverage < self.performance_benchmarks['test_coverage_minimum']:
            issues.append(CodeReviewIssue(
                category=ReviewCategory.PERFORMANCE,
                severity=ReviewSeverity.WARNING,
                title="Insufficient Test Coverage",
                description=f"Test coverage {test_coverage:.1%} below required {self.performance_benchmarks['test_coverage_minimum']:.1%}",
                violation="Does not meet minimum test coverage requirements",
                recommendation="Add unit tests and integration tests to achieve minimum coverage",
                specs_reference="code_review_workflow.quality_gates.code_quality"
            ))

        return issues

    def _analyze_dependency_constraints(self, code_changes: str) -> List[CodeReviewIssue]:
        """Analyze code changes for forbidden dependency usage"""
        issues = []

        for forbidden_dep in self.forbidden_dependencies:
            if f'import {forbidden_dep}' in code_changes or f'from {forbidden_dep}' in code_changes:
                # Get approved alternative
                alternatives = {
                    'numpy': 'tidyllm.tlm',
                    'pandas': 'polars',
                    'sklearn': 'tidyllm.tlm algorithms',
                    'sentence_transformers': 'tidyllm_sentence',
                    'tensorflow': 'tidyllm.tlm',
                    'pytorch': 'tidyllm.tlm',
                    'transformers': 'tidyllm_sentence'
                }

                alternative = alternatives.get(forbidden_dep, 'approved TidyLLM components')

                issues.append(CodeReviewIssue(
                    category=ReviewCategory.DEPENDENCIES,
                    severity=ReviewSeverity.BLOCKING,
                    title=f"Forbidden Dependency: {forbidden_dep}",
                    description=f"Code imports forbidden dependency '{forbidden_dep}'",
                    violation="Violates infrastructure sovereignty and dependency constraints",
                    recommendation=f"Replace {forbidden_dep} with {alternative}",
                    specs_reference="architectural_ground_truth.forbidden_dependencies",
                    auto_fixable=False
                ))

        return issues

    def _analyze_regulatory_compliance(self, code_changes: str, metadata: Dict[str, Any]) -> List[CodeReviewIssue]:
        """Analyze code changes for regulatory compliance impact"""
        issues = []

        # Check for audit trail requirements
        if 'mlflow' not in code_changes.lower() and self._affects_ai_operations(code_changes):
            issues.append(CodeReviewIssue(
                category=ReviewCategory.COMPLIANCE,
                severity=ReviewSeverity.WARNING,
                title="Missing MLflow Audit Trail",
                description="AI operations detected without MLflow integration",
                violation="May not maintain required audit trail for regulatory compliance",
                recommendation="Ensure all AI operations are logged through MLflow for compliance",
                specs_reference="code_review_workflow.compliance_review"
            ))

        return issues

    def _violates_layer_separation(self, code_changes: str) -> bool:
        """Check if code violates clean architecture layer separation"""
        # Simple heuristic - look for direct infrastructure imports in domain code
        infrastructure_patterns = ['boto3', 'psycopg2', 'requests', 'flask', 'fastapi']
        domain_patterns = ['class.*Entity', 'class.*Service', 'def.*use_case']

        has_infrastructure = any(pattern in code_changes for pattern in infrastructure_patterns)
        has_domain = any(pattern in code_changes for pattern in domain_patterns)

        return has_infrastructure and has_domain

    def _affects_ai_operations(self, code_changes: str) -> bool:
        """Check if code changes affect AI operations requiring audit trail"""
        ai_patterns = ['bedrock', 'openai', 'llm', 'model', 'embedding', 'predict']
        return any(pattern in code_changes.lower() for pattern in ai_patterns)

    def _generate_analysis_summary(self, issues: List[CodeReviewIssue], metadata: Dict[str, Any]) -> CodeReviewAnalysis:
        """Generate comprehensive analysis summary"""

        # Count issues by severity
        blocking_count = len([i for i in issues if i.severity == ReviewSeverity.BLOCKING])
        critical_count = len([i for i in issues if i.severity == ReviewSeverity.CRITICAL])
        warning_count = len([i for i in issues if i.severity == ReviewSeverity.WARNING])

        # Calculate compliance scores
        architectural_compliance = not any(i.category == ReviewCategory.ARCHITECTURE and
                                         i.severity in [ReviewSeverity.BLOCKING, ReviewSeverity.CRITICAL]
                                         for i in issues)
        security_compliance = not any(i.category == ReviewCategory.SECURITY and
                                    i.severity in [ReviewSeverity.BLOCKING, ReviewSeverity.CRITICAL]
                                    for i in issues)
        performance_compliance = not any(i.category == ReviewCategory.PERFORMANCE and
                                       i.severity == ReviewSeverity.BLOCKING
                                       for i in issues)
        regulatory_compliance = not any(i.category == ReviewCategory.COMPLIANCE and
                                      i.severity in [ReviewSeverity.BLOCKING, ReviewSeverity.CRITICAL]
                                      for i in issues)

        # Calculate overall score
        if blocking_count > 0:
            overall_score = 0.0
            approval_status = "REJECTED"
        elif critical_count > 0:
            overall_score = 0.3
            approval_status = "CHANGES_REQUESTED"
        elif warning_count > 0:
            overall_score = 0.7
            approval_status = "APPROVED_WITH_CONDITIONS"
        else:
            overall_score = 1.0
            approval_status = "APPROVED"

        # Generate summary
        if blocking_count > 0:
            summary = f"Code review REJECTED due to {blocking_count} blocking issues that must be resolved before merge."
        elif critical_count > 0:
            summary = f"Code review requires changes due to {critical_count} critical issues."
        elif warning_count > 0:
            summary = f"Code review approved with {warning_count} warnings to address."
        else:
            summary = "Code review APPROVED - all quality standards met."

        # Generate recommendations
        recommendations = []
        if not architectural_compliance:
            recommendations.append("Review architectural patterns and ensure hexagonal architecture compliance")
        if not security_compliance:
            recommendations.append("Address security violations, particularly S3-first compliance")
        if not performance_compliance:
            recommendations.append("Resolve performance issues and dependency constraints")
        if not regulatory_compliance:
            recommendations.append("Ensure regulatory compliance requirements are met")

        # Check if escalation is required
        escalation_required = any(i.requires_architect_review for i in issues) or blocking_count > 0

        # Identify SME consultation areas
        sme_areas = []
        categories = set(i.category.value for i in issues if i.severity in [ReviewSeverity.BLOCKING, ReviewSeverity.CRITICAL])
        if ReviewCategory.ARCHITECTURE.value in categories:
            sme_areas.append("architectural_patterns")
        if ReviewCategory.SECURITY.value in categories:
            sme_areas.append("security_compliance")
        if ReviewCategory.PERFORMANCE.value in categories:
            sme_areas.append("performance_optimization")
        if ReviewCategory.COMPLIANCE.value in categories:
            sme_areas.append("regulatory_requirements")

        return CodeReviewAnalysis(
            overall_score=overall_score,
            approval_status=approval_status,
            issues=issues,
            architectural_compliance=architectural_compliance,
            security_compliance=security_compliance,
            performance_compliance=performance_compliance,
            regulatory_compliance=regulatory_compliance,
            summary=summary,
            recommendations=recommendations,
            escalation_required=escalation_required,
            sme_consultation_areas=sme_areas
        )

    def get_architectural_guidance(self, question: str) -> Dict[str, Any]:
        """Get architectural guidance from both criteria and comprehensive specs"""
        guidance = {
            'question': question,
            'timestamp': datetime.now().isoformat(),
            'guidance_type': 'architectural',
            'response': '',
            'references': [],
            'comprehensive_specs_available': bool(self.system_specs)
        }

        question_lower = question.lower()

        # Use comprehensive specs if available, otherwise fall back to criteria
        if self.system_specs and 'architectural_ground_truth' in self.system_specs:
            arch_truth = self.system_specs['architectural_ground_truth']

            if 'gateway' in question_lower and 'adapter' in question_lower:
                guidance['response'] = """
                Use ADAPTERS not GATEWAYS. The V1 'gateways' are actually adapters in hexagonal architecture.
                Adapters should only translate between ports and external services, implementing port interfaces
                defined by the application layer. Avoid business logic in adapters.
                """
                guidance['references'] = ['specs.json:architectural_ground_truth.gateway_to_adapter_transformation']

            elif 'layer' in question_lower or 'separation' in question_lower:
                guidance['response'] = """
                Follow clean architecture layer separation: Domain entities never depend on infrastructure.
                Use dependency inversion through port interfaces. Infrastructure adapts to business needs,
                not the other way around.
                """
                guidance['references'] = ['specs.json:v2_official_clean_architecture.architecture_layers']

            elif 's3' in question_lower or 'storage' in question_lower:
                guidance['response'] = """
                Follow S3-first architecture: ALL data must reside in S3, not local storage.
                App folders must be cleaned after every operation. Use S3 streaming patterns
                and ensure zero local data persistence.
                """
                guidance['references'] = ['specs.json:s3_first_bidirectional_architecture']

        # Fall back to criteria-based guidance
        if not guidance['response']:
            patterns = self.review_criteria.get('architectural_patterns', {})
            security = self.review_criteria.get('security_requirements', {})

            guidance['response'] = f"""
            Based on focused review criteria:
            - Required patterns: {', '.join(patterns.get('required_patterns', []))}
            - Forbidden patterns: {', '.join(patterns.get('forbidden_patterns', []))}
            - S3-first mandatory: {security.get('s3_first_mandatory', False)}
            - App cleanup required: {security.get('app_folder_cleanup_required', False)}
            """
            guidance['references'] = ['code_review_criteria.json']

        return guidance

    def export_analysis_report(self, analysis: CodeReviewAnalysis, output_path: str) -> None:
        """Export detailed analysis report to JSON file"""
        report = {
            'analysis': asdict(analysis),
            'timestamp': datetime.now().isoformat(),
            'sme_version': '1.0.0',
            'architectural_specs_version': self.architectural_ground_truth.get('version', 'unknown')
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

def main():
    """Example usage of the Code Review SME system"""
    sme = CodeReviewSME()

    # Example code changes to analyze
    sample_code = """
import pandas as pd
import numpy as np
from flask import Flask

class DataGateway:
    def __init__(self):
        self.local_data = open('data.txt', 'r')

    def process_data(self):
        df = pd.read_csv('input.csv')
        return df.to_dict()
"""

    # Analyze the code changes
    analysis = sme.analyze_code_changes(sample_code, {'test_coverage': 0.6})

    print("Code Review SME Analysis:")
    print(f"Overall Score: {analysis.overall_score:.2f}")
    print(f"Status: {analysis.approval_status}")
    print(f"Summary: {analysis.summary}")
    print(f"\nIssues Found: {len(analysis.issues)}")

    for issue in analysis.issues:
        print(f"  - {issue.title} ({issue.severity.value})")
        print(f"    {issue.description}")
        print(f"    Recommendation: {issue.recommendation}")
        if issue.specs_reference:
            print(f"    Reference: {issue.specs_reference}")
        print()

    # Example architectural guidance
    guidance = sme.get_architectural_guidance("Should I use a gateway or adapter pattern?")
    print("Architectural Guidance:")
    print(guidance['response'])

if __name__ == "__main__":
    main()