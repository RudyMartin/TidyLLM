# Step 4: Report Synthesis

## Purpose
Synthesize analysis results into comprehensive executive summary and actionable recommendations.

## Prompt Template

```
You are a senior analyst specializing in report synthesis and executive communication. Using the analysis results from Step 3, create:

1. **Executive Summary:**
   - High-level overview of key findings
   - Critical insights in business terms
   - Executive-friendly language and metrics

2. **Detailed Findings Integration:**
   - Synthesize all analytical results
   - Connect patterns across different data sources
   - Prioritize findings by business impact

3. **Actionable Recommendations:**
   - Specific, measurable action items
   - Priority levels and timelines
   - Resource requirements and dependencies

## Input Variables
- `{analysis_results}` - Output from Step 3 analysis
- `{report_template}` - Template format (executive_summary, technical_report, etc.)
- `{output_format}` - Desired level of detail (brief, standard, comprehensive)

## RAG Integration Points
- Query best practices for report writing
- Access industry-specific terminology and standards
- Retrieve comparable case studies and benchmarks

## Expected Output Format
```json
{
  "synthesis_status": "completed|failed",
  "executive_summary": "Clear, concise summary for executive audience",
  "detailed_findings": {
    "critical_issues": [
      {
        "issue": "Description of critical finding",
        "impact": "high|medium|low",
        "urgency": "immediate|short_term|long_term",
        "evidence": "Supporting evidence"
      }
    ],
    "opportunities": [
      {
        "opportunity": "Description of opportunity",
        "potential_value": "Estimated value or benefit",
        "implementation_complexity": "low|medium|high"
      }
    ],
    "risk_factors": [
      {
        "risk": "Description of risk",
        "probability": "high|medium|low",
        "impact": "high|medium|low",
        "mitigation": "Suggested mitigation strategy"
      }
    ]
  },
  "action_items": [
    {
      "action": "Specific action to be taken",
      "priority": "high|medium|low",
      "timeline": "immediate|1-week|1-month|3-months",
      "owner": "Suggested responsible party",
      "resources_required": "Description of resources needed",
      "success_metrics": "How to measure success"
    }
  ],
  "confidence_score": 0.92,
  "synthesis_methodology": "Description of synthesis approach used",
  "limitations": [
    "Any limitations or caveats in the analysis"
  ],
  "next_steps": [
    "Recommended follow-up actions or deeper analysis"
  ]
}
```

## Success Criteria
- Clear executive summary under 200 words
- All critical findings properly prioritized
- Actionable recommendations with specific timelines
- High confidence score (>0.85)
- Business impact clearly articulated
```