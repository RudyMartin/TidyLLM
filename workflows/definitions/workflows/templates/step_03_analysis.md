# Step 3: Analysis

## Purpose
Analyze extracted data to identify patterns, insights, and relationships.

## Prompt Template

```
You are a data analysis expert. Using the structured data from Step 2, perform:

1. **Pattern Recognition:**
   - Identify trends and patterns in the data
   - Detect anomalies or outliers
   - Find correlations between data points

2. **Comparative Analysis:**
   - Compare data across different sources/time periods
   - Benchmark against industry standards or baselines
   - Identify significant variations or changes

3. **Insight Generation:**
   - Derive meaningful insights from the patterns
   - Identify implications and potential impacts
   - Generate hypotheses for further investigation

## Input Variables
- `{extracted_data}` - Output from Step 2
- `{analysis_framework}` - Analytical approach to use
- `{comparison_baselines}` - Reference data for comparison

## RAG Integration Points
- Query historical data for trend analysis
- Access industry benchmarks and standards
- Retrieve analytical frameworks and methodologies

## Expected Output Format
```json
{
  "analysis_status": "completed|in_progress|failed",
  "findings": {
    "key_patterns": [
      {
        "pattern_type": "trend|anomaly|correlation",
        "description": "Pattern description",
        "significance": "high|medium|low",
        "evidence": []
      }
    ],
    "comparative_results": [
      {
        "comparison_type": "benchmark|historical|peer",
        "result": "above|below|equal",
        "variance": 15.2,
        "significance": "high"
      }
    ],
    "insights": [
      {
        "insight": "Key insight description",
        "confidence": 0.87,
        "impact": "high|medium|low",
        "supporting_evidence": []
      }
    ]
  },
  "risk_indicators": [],
  "recommendations": []
}
```

## Success Criteria
- Comprehensive pattern identification
- Statistically significant findings
- Clear insight articulation
- Actionable recommendations generated
```