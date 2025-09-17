# Code Review Analysis Template

## Overview
This template provides comprehensive code review analysis covering quality, architecture, security, and compliance validation according to the architectural ground truth documented in specs.json.

## Input Parameters
- **Code Changes**: {code_changes}
- **Impact Assessment**: {impact_assessment}
- **Architectural Compliance**: {architectural_compliance}
- **Security Validation**: {security_validation}
- **Performance Analysis**: {performance_analysis}

---

## 1. Code Quality Analysis

### Syntax and Style Validation
- **Style Guide Compliance**: Verify 100% adherence to coding standards
- **Code Formatting**: Consistent indentation, naming conventions
- **Documentation Coverage**: All public interfaces properly documented
- **Code Complexity**: Cyclomatic complexity < 10 per function

**Assessment Criteria:**
```
PASS: All quality standards met
PARTIAL: Minor issues requiring fixes
FAIL: Critical quality issues requiring rework
```

### Test Coverage Analysis
- **Coverage Threshold**: Minimum 90% test coverage for new code
- **Test Quality**: Unit tests, integration tests, edge cases covered
- **Test Maintainability**: Clear, readable, maintainable test code

---

## 2. Architectural Compliance Review

### Hexagonal Architecture Compliance
- **Adapter Pattern**: Verify proper use of adapters (NOT gateways)
- **Port Interfaces**: Clean port definitions and implementations
- **Dependency Direction**: Business logic never depends on infrastructure
- **Layer Separation**: Clean architecture layers properly separated

### Architectural Ground Truth Validation
```
CRITICAL CHECKS:
☐ Uses adapters pattern (NOT gateway pattern)
☐ Follows hexagonal architecture principles
☐ Proper separation of concerns
☐ Interface-based abstraction
☐ Dependency injection compatible
```

### TidyLLM Integration Patterns
- **Interface Usage**: Proper use of tidyllm.interfaces.web patterns
- **Session Management**: Unified session management approach
- **Configuration**: Centralized configuration system usage

---

## 3. Security Validation

### S3-First Compliance
- **Data Storage**: ALL data must reside in S3, not local storage
- **App Folder Cleanup**: MANDATORY cleanup of app folders after processing
- **No Local Persistence**: Zero local data persistence patterns
- **Credential Management**: Proper AWS credential handling

**Security Checklist:**
```
MANDATORY SECURITY REQUIREMENTS:
☐ No hardcoded secrets or credentials
☐ S3-only data handling (no local storage)
☐ App folder cleanup implemented
☐ Input validation and sanitization
☐ Authentication/authorization patterns
☐ No security vulnerabilities identified
```

### Data Security Patterns
- **Encryption**: Data encrypted at rest in S3 and in transit
- **Access Control**: Proper S3 bucket permissions
- **Audit Logging**: Complete data access tracking

---

## 4. Dependency and Constraint Validation

### Forbidden Dependencies Check
```
STRICTLY FORBIDDEN (Infrastructure Sovereignty):
❌ numpy - Use tidyllm.tlm instead
❌ pandas - Use polars instead
❌ sklearn - Use tidyllm.tlm algorithms instead
❌ sentence_transformers - Use tidyllm_sentence instead
❌ tensorflow - Use tidyllm.tlm instead
❌ pytorch - Use tidyllm.tlm instead
❌ transformers - Use tidyllm_sentence instead
```

### Approved Alternatives
```
REQUIRED ALTERNATIVES:
✅ tidyllm.tlm for mathematical operations
✅ polars for data manipulation
✅ tidyllm_sentence for embedding operations
✅ Direct AWS SDK usage for cloud operations
```

---

## 5. Performance Impact Assessment

### Performance Benchmarks
- **Response Time Impact**: < 10% degradation from baseline
- **Memory Usage**: < 5% increase from baseline
- **CPU Overhead**: < 3% additional utilization
- **I/O Operations**: Optimized for S3 streaming patterns

### Scalability Considerations
- **Concurrent Users**: Impact on multi-user scenarios
- **Resource Scaling**: Horizontal scaling compatibility
- **Caching Strategy**: Appropriate caching patterns

---

## 6. Regulatory Compliance Check

### Compliance Standards
- **SR-11-7**: Model risk management compliance
- **Basel-III**: Financial regulatory requirements
- **SOX-404**: Audit trail and controls

### Audit Trail Requirements
- **MLflow Integration**: All operations logged to MLflow
- **PostgreSQL Backend**: Audit data stored in PostgreSQL
- **Data Lineage**: Complete data processing lineage
- **Compliance Reporting**: Automated compliance report generation

---

## 7. Integration Validation

### System Integration Points
- **Boss Portal Integration**: Compatibility with unified boss portal
- **Workflow Registry**: Proper workflow system integration
- **Service Coordination**: Multi-service interaction patterns

### External Dependencies
- **AWS Services**: Proper S3, Lambda, PostgreSQL integration
- **MLflow**: Experiment tracking and audit integration
- **Corporate Systems**: Enterprise authentication and authorization

---

## 8. Code Review Decision Matrix

### Approval Criteria
```
AUTOMATIC APPROVAL:
☐ All automated checks pass
☐ Architectural compliance verified
☐ Security requirements satisfied
☐ Performance benchmarks met
☐ Regulatory compliance maintained
☐ Test coverage > 90%
☐ Documentation complete
```

### Review Required Scenarios
```
MANUAL REVIEW REQUIRED:
☐ Architectural pattern changes
☐ Security-sensitive modifications
☐ Performance-critical changes
☐ Regulatory compliance impact
☐ Complex integration changes
☐ New dependency introductions
```

### Escalation Triggers
```
SENIOR ARCHITECT REVIEW:
☐ Architectural ground truth changes
☐ Major system design modifications
☐ Cross-component integration changes

SECURITY TEAM REVIEW:
☐ Security pattern modifications
☐ Data handling changes
☐ Authentication/authorization changes

COMPLIANCE TEAM REVIEW:
☐ Regulatory impact changes
☐ Audit trail modifications
☐ Compliance reporting changes
```

---

## 9. SME Consultation Requirements

### Code Review SME Integration
When complex architectural, security, or compliance issues are identified, this workflow should consult with specialized SMEs:

**SME Specialization Areas:**
- **Architectural Patterns SME**: Hexagonal architecture, clean architecture, adapter patterns
- **Security Compliance SME**: S3-first security, app cleanup, credential management
- **Performance Optimization SME**: Benchmarking, scalability, resource optimization
- **Regulatory Requirements SME**: SR-11-7, Basel-III, SOX-404 compliance
- **TidyLLM Integration SME**: Interface patterns, session management, configuration

**SME Consultation Triggers:**
- Architectural changes requiring ground truth updates
- Security pattern modifications affecting S3-first compliance
- Performance changes impacting system benchmarks
- Regulatory compliance modifications
- Complex TidyLLM integration patterns

---

## 10. Final Recommendations

### Action Items
Based on the code review analysis, provide prioritized action items:

1. **Critical Issues**: Must be resolved before merge
2. **High Priority**: Should be resolved before merge
3. **Medium Priority**: Can be addressed in follow-up
4. **Low Priority**: Consider for future improvement

### Approval Decision
```
FINAL DECISION:
☐ APPROVED - All criteria met, ready for merge
☐ APPROVED WITH CONDITIONS - Minor issues to address
☐ CHANGES REQUESTED - Significant issues requiring rework
☐ REJECTED - Critical failures requiring major changes
```

### Documentation Updates
If architectural changes are made, update the following:
- **specs.json**: Architectural ground truth updates
- **Workflow Registry**: Any workflow modifications
- **System Documentation**: API or integration documentation

---

## Validation Rules Applied

This code review has validated against the following rules from the criteria:
- `no_forbidden_dependencies`: ✅ Validated
- `s3_first_compliance`: ✅ Validated
- `architectural_pattern_adherence`: ✅ Validated
- `security_best_practices`: ✅ Validated
- `hexagonal_architecture_compliance`: ✅ Validated
- `clean_architecture_layer_separation`: ✅ Validated
- `app_folder_cleanup_validation`: ✅ Validated

## Review Completion

**Reviewer Information:**
- **Primary Reviewer**: {reviewer_name}
- **Review Date**: {review_date}
- **Review Duration**: {review_duration}
- **SME Consultations**: {sme_consultations_if_any}

**Quality Assurance:**
- All validation rules applied and documented
- Compliance with architectural ground truth verified
- Security and performance standards maintained
- Regulatory requirements satisfied

---

*This code review template ensures comprehensive validation against the architectural ground truth documented in specs.json and maintains the highest standards for production code quality.*