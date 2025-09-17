# Code Review SME Creation Guide

## Overview
This document describes the creation process, source files, and configuration used to generate the Code Review Subject Matter Expert (SME) system. This documentation is automatically generated as part of the Boss Portal SME creation workflow.

## SME Creation Metadata
- **SME Name**: Code Review SME
- **Creation Date**: 2025-09-14
- **Version**: 1.0.0
- **Boss Portal Integration**: ✅ Enabled
- **Auto-Generated**: ✅ Yes (via Boss Portal workflow)

## Source Files Used in SME Creation

### Primary Configuration Files

#### 1. specs.json (Architectural Ground Truth)
**File Path**: `C:\Users\marti\AI-Scoring\specs.json`
**Role**: Primary source of architectural ground truth and constraints
**Key Sections Used**:
```json
{
  "architectural_ground_truth": {
    "forbidden_dependencies": { ... },
    "s3_first_bidirectional_architecture": { ... },
    "v2_official_clean_architecture": { ... },
    "gateway_to_adapter_transformation": { ... }
  },
  "code_review_workflow": {
    "code_review_stages": { ... },
    "quality_gates": { ... }
  }
}
```

#### 2. workflow_registry/criteria/code_review_criteria.json
**File Path**: `C:\Users\marti\AI-Scoring\workflow_registry\criteria\code_review_criteria.json`
**Role**: Defines validation rules, scoring rubric, and compliance checks
**Key Elements**:
- Scoring rubric weights
- Required and optional fields
- Validation rules list
- Compliance check specifications
- SME integration requirements

#### 3. workflow_registry/templates/code_review.md
**File Path**: `C:\Users\marti\AI-Scoring\workflow_registry\templates\code_review.md`
**Role**: Provides structured template for code review analysis
**Template Sections**:
- Code Quality Analysis
- Architectural Compliance Review
- Security Validation
- Performance Impact Assessment
- Regulatory Compliance Check
- SME Consultation Requirements

#### 4. workflow_registry/registry.json
**File Path**: `C:\Users\marti\AI-Scoring\workflow_registry\registry.json`
**Role**: Registry entry for code_review workflow integration
**Entry**: `"code_review": { ... }`

### SME Implementation Files

#### 1. v2/code_review_sme.py
**File Path**: `C:\Users\marti\AI-Scoring\v2\code_review_sme.py`
**Role**: Main SME implementation with expert analysis capabilities
**Key Classes**:
- `CodeReviewSME`: Primary SME system
- `CodeReviewIssue`: Issue representation
- `CodeReviewAnalysis`: Analysis results
- `ReviewSeverity` & `ReviewCategory`: Enumeration types

## SME Knowledge Base Construction

### 1. Architectural Knowledge
**Source**: `specs.json` → `architectural_ground_truth` section
**Knowledge Areas**:
- Hexagonal architecture principles
- Gateway vs Adapter pattern corrections
- Clean architecture layer separation
- V2 architecture specifications

**Code Implementation**:
```python
def _load_architectural_specs(self) -> Dict[str, Any]:
    # Loads architectural ground truth from specs.json
    # Used for validation and guidance decisions
```

### 2. Security Knowledge
**Source**: `specs.json` → `s3_first_bidirectional_architecture` section
**Knowledge Areas**:
- S3-first compliance requirements
- App folder cleanup mandates
- Forbidden local storage patterns
- Credential management best practices

**Code Implementation**:
```python
def _analyze_security_compliance(self, code_changes: str, metadata: Dict[str, Any]):
    # Validates S3-first compliance and security patterns
    # Identifies credential hardcoding and local storage violations
```

### 3. Dependency Constraint Knowledge
**Source**: `specs.json` → `forbidden_dependencies` section
**Knowledge Areas**:
- Forbidden library list (numpy, pandas, sklearn, etc.)
- Approved alternatives (tidyllm.tlm, polars, tidyllm_sentence)
- Infrastructure sovereignty principles

**Code Implementation**:
```python
def _analyze_dependency_constraints(self, code_changes: str):
    # Validates against forbidden dependency list
    # Provides approved alternatives
```

### 4. Performance Benchmark Knowledge
**Source**: `specs.json` → `code_review_workflow` → `quality_gates`
**Knowledge Areas**:
- Test coverage requirements (>90%)
- Performance degradation thresholds
- Memory and CPU utilization limits

### 5. Regulatory Compliance Knowledge
**Source**: `specs.json` → `code_review_workflow` → `compliance_review`
**Knowledge Areas**:
- SR-11-7 compliance requirements
- Basel-III regulatory standards
- SOX-404 audit trail requirements
- MLflow audit integration

## SME Decision Matrix

### Issue Severity Classification
```
BLOCKING: Code cannot be merged
├── Forbidden dependencies detected
├── Security violations (local storage, hardcoded credentials)
└── Critical architectural violations

CRITICAL: Major issues requiring architect review
├── Gateway pattern usage (should be adapter)
├── Clean architecture layer violations
└── Major security pattern violations

WARNING: Issues requiring attention
├── Insufficient test coverage
├── Performance concerns
└── Minor compliance issues

INFO: Recommendations for improvement
├── Code quality suggestions
├── Performance optimizations
└── Documentation improvements
```

### Escalation Triggers
```
AUTOMATIC ESCALATION:
├── Any BLOCKING severity issues
├── requires_architect_review flag set
├── Multiple CRITICAL issues in same category
└── Security compliance failures

SME CONSULTATION AREAS:
├── architectural_patterns
├── security_compliance
├── performance_optimization
├── regulatory_requirements
└── tidyllm_integration_patterns
```

## Integration with Boss Portal

### 1. Automatic SME Creation
The Boss Portal can automatically create and configure SMEs using this guide:

```python
# Boss Portal SME Creation Workflow
def create_code_review_sme():
    """
    Automatically creates Code Review SME using:
    1. specs.json for architectural ground truth
    2. workflow_registry files for criteria and templates
    3. This creation guide for configuration
    """
    sme = CodeReviewSME(specs_path="specs.json")
    return sme
```

### 2. Dynamic Knowledge Updates
The SME system dynamically loads knowledge from source files:

```python
# Knowledge is loaded at runtime from specs.json
architectural_ground_truth = self._load_architectural_specs()
forbidden_dependencies = self._get_forbidden_dependencies()
security_requirements = self._get_security_requirements()
```

### 3. Workflow Integration
The SME integrates with the workflow registry system:

```python
# Called from workflow execution
analysis = sme.analyze_code_changes(code_changes, metadata)
# Returns structured CodeReviewAnalysis with issues and recommendations
```

## File Dependencies Map

```
code_review_sme.py
├── specs.json (architectural ground truth)
│   ├── architectural_ground_truth.*
│   ├── code_review_workflow.*
│   └── forbidden_dependencies.*
├── workflow_registry/criteria/code_review_criteria.json
│   ├── scoring_rubric
│   ├── validation_rules
│   └── compliance_checks
├── workflow_registry/templates/code_review.md
│   ├── analysis_structure
│   └── reporting_format
└── workflow_registry/registry.json
    └── code_review workflow entry
```

## Quality Assurance

### Validation Checks Performed
✅ **Architectural Compliance**: Validates against specs.json ground truth
✅ **Security Standards**: Enforces S3-first and app cleanup requirements
✅ **Dependency Constraints**: Blocks forbidden libraries, suggests alternatives
✅ **Performance Standards**: Validates test coverage and benchmark compliance
✅ **Regulatory Requirements**: Checks SR-11-7, Basel-III, SOX-404 compliance

### SME Self-Validation
The SME system includes self-validation capabilities:
- Loads and validates specs.json structure
- Verifies workflow registry integration
- Confirms template accessibility
- Validates knowledge base completeness

## Maintenance and Updates

### Automatic Updates
The SME system automatically updates when source files change:
1. **specs.json changes**: Architectural knowledge updates automatically
2. **Criteria updates**: New validation rules loaded dynamically
3. **Template changes**: Analysis structure updated on next execution

### Manual Maintenance
**Configuration Review**: Quarterly review of SME decisions and accuracy
**Knowledge Base Updates**: Updates when architectural ground truth evolves
**Performance Tuning**: Analysis performance optimization as needed

## Usage Examples

### 1. Basic Code Review Analysis
```python
sme = CodeReviewSME()
analysis = sme.analyze_code_changes(code_changes, metadata)
print(f"Status: {analysis.approval_status}")
print(f"Issues: {len(analysis.issues)}")
```

### 2. Architectural Guidance Request
```python
guidance = sme.get_architectural_guidance("Should I use gateway or adapter?")
print(guidance['response'])
# Output: "Use ADAPTERS not GATEWAYS..."
```

### 3. Export Analysis Report
```python
sme.export_analysis_report(analysis, "review_report.json")
# Creates detailed JSON report for audit purposes
```

## Boss Portal Integration Points

### 1. SME Creation Workflow
Boss Portal can create SMEs using this guide as a template, automatically:
- Reading source file specifications
- Configuring knowledge bases
- Setting up validation rules
- Enabling workflow integration

### 2. Dynamic SME Updates
Boss Portal can update SMEs when source files change:
- Monitor specs.json for architectural updates
- Reload workflow registry changes
- Update SME knowledge bases automatically

### 3. SME Analytics Dashboard
Boss Portal can provide analytics on SME usage:
- Code review decisions statistics
- Issue category breakdowns
- Escalation frequency analysis
- SME consultation patterns

---

**Note**: This document is automatically generated as part of the Boss Portal SME creation workflow. It serves as both documentation and a blueprint for creating similar SME systems for other specialized domains.

**Last Updated**: 2025-09-14
**Generated By**: Boss Portal SME Creation System
**Validation Status**: ✅ All source files accessible and valid