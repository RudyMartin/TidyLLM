# Analyst Report Workflow Prompts

## MVR Tagging Stage
```
Document Classification Prompt:

Analyze the following document and classify it according to these categories:
- Document Type: MVR, VST, or Research
- Risk Level: High, Medium, Low
- Business Purpose: Employment Verification, Credit Assessment, Insurance Review
- Data Quality Score: 1-10 based on completeness and accuracy

Document Content:
{document_content}

Please provide structured analysis in JSON format.
```

## MVR vs VST Comparison Stage
```
Comparison Analysis Prompt:

You are comparing an MVR (Motor Vehicle Record) document against its corresponding VST (Validation Scoping Template).

MVR Document: {mvr_content}
VST Template: {vst_content}

Perform the following analysis:

1. **Compliance Check**: Does the MVR meet all requirements specified in the VST?
2. **Data Completeness**: What required data points are missing or incomplete?
3. **Risk Assessment**: Based on the comparison, what is the risk level?
4. **Discrepancy Analysis**: List any significant discrepancies between expected and actual data
5. **Recommendation**: Approve, Reject, or Request Additional Information

Format your response as a structured report with clear recommendations.
```

## Final Report Generation Stage
```
Executive Summary Prompt:

Generate a comprehensive analyst report based on the following analysis:

Document Analysis: {document_analysis}
Comparison Results: {comparison_results}
Risk Assessment: {risk_assessment}

Create an executive summary that includes:

1. **Executive Summary** (2-3 sentences)
2. **Key Findings** (bullet points)
3. **Risk Indicators** (specific concerns identified)
4. **Compliance Status** (Pass/Fail with reasons)
5. **Recommendations** (specific actions to take)
6. **Supporting Data** (metrics and scores)

The report should be suitable for business stakeholders and regulatory compliance.
```