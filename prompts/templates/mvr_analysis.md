# MVR Analysis Workflow Template
*Motor Vehicle Record comprehensive analysis workflow*

## Document Classification
```
MVR Document Classification:

Analyze the following MVR (Motor Vehicle Record) document and provide structured classification:

Document: {document_content}

Classification Requirements:
1. **Document Type**: Confirm this is an MVR and identify sub-type (standard, employment, insurance, etc.)
2. **Completeness Score**: Rate 1-10 based on required data presence
3. **Risk Level**: High, Medium, Low based on violations and patterns
4. **Business Purpose**: Employment Verification, Credit Assessment, Insurance Review, Other
5. **Data Quality**: Assess accuracy, consistency, and formatting
6. **Regulatory Compliance**: Check against relevant standards (FCRA, state requirements)

Output Format: Structured JSON with scores and explanations.
```

## MVR Violation Analysis
```
MVR Violation and Pattern Analysis:

Examine the MVR document for violations, patterns, and risk indicators:

MVR Content: {document_content}

Analysis Requirements:
1. **Violation Summary**: List all violations with dates, types, and severity
2. **Pattern Recognition**: Identify concerning patterns (frequency, escalation, types)
3. **License Status**: Current status, suspensions, restrictions, expirations
4. **Risk Assessment**: Calculate risk score based on violation history
5. **Time-based Analysis**: Recent vs. historical violations (last 3, 5, 7 years)
6. **Impact Assessment**: How violations affect employability/insurability

Provide detailed analysis with risk scoring and recommendations.
```

## MVR vs VST Compliance Check
```
MVR vs VST Compliance Validation:

Compare the MVR document against Validation Scoping Template requirements:

MVR Document: {document_content}
VST Requirements: [If VST provided separately, reference here]

Compliance Check:
1. **Requirement Mapping**: Map MVR data to VST requirements
2. **Gap Analysis**: Identify missing required information
3. **Data Validation**: Verify data meets VST quality standards
4. **Compliance Score**: Rate compliance percentage
5. **Exception Handling**: List items requiring manual review
6. **Approval Recommendation**: Approve, Reject, or Request Additional Info

Format as compliance report with clear pass/fail indicators.
```

## Executive Summary Generation
```
MVR Executive Summary Report:

Generate executive summary suitable for business decision makers:

MVR Analysis Data: {document_content}

Executive Summary Components:
1. **Executive Overview** (2-3 sentences of key findings)
2. **Risk Assessment** (Clear risk level with justification)
3. **Key Violations** (Most significant issues identified)
4. **Business Impact** (How this affects business decision)
5. **Compliance Status** (Pass/Fail with regulatory considerations)
6. **Recommendations** (Specific actions: Approve, Reject, Conditional, etc.)
7. **Supporting Metrics** (Risk scores, violation counts, compliance percentages)

Target audience: HR managers, insurance underwriters, compliance officers.
```