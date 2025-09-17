# Sample Input Document

This is a sample input document for testing the sequential analysis workflow.

## Document Information
- **Type**: Analysis Report
- **Domain**: Financial Analysis
- **Date**: January 15, 2025
- **Version**: 1.0

## Content Summary

This document contains sample financial data and metrics that can be used to test the workflow's ability to:

1. **Validate Input Documents** - The validation step should verify this document is properly formatted and accessible
2. **Extract Data** - Key entities like amounts, dates, and financial metrics should be identified
3. **Analyze Patterns** - Trends and relationships in the financial data should be discovered
4. **Synthesize Results** - Executive summary and recommendations should be generated
5. **Generate Output** - Final JSON output with audit trail should be produced

## Sample Financial Data

### Quarterly Performance
- Q4 2024 Revenue: $2.5M (15% increase from Q3)
- Operating Expenses: $1.8M
- Net Profit: $700K (28% profit margin)
- Cash Flow: $850K positive

### Key Metrics
- Customer Acquisition Cost: $125
- Customer Lifetime Value: $2,400
- Monthly Recurring Revenue: $650K
- Churn Rate: 2.3%

### Risk Factors
1. **Market Volatility**: Increased competition in Q1 2025
2. **Supply Chain**: Dependencies on two primary vendors
3. **Regulatory**: New compliance requirements effective March 2025

## Expected Workflow Outputs

This document should trigger the following processing steps:

1. **Validation**: Document should pass with high quality score (>0.90)
2. **Extraction**: Should identify 4 financial entities, 8 numeric values, 3 risk factors
3. **Analysis**: Should detect positive revenue trend, moderate risk profile
4. **Synthesis**: Should recommend continued growth strategy with risk mitigation
5. **Output**: Should generate comprehensive JSON report with confidence >0.85

## Test Criteria

The workflow should demonstrate:
- Proper template field resolution (`{input_files}`, `{quality_threshold}`, etc.)
- RAG system integration for domain knowledge enhancement
- Sequential step execution with dependency management
- JSON output compliance with specified schema
- Audit trail generation with processing metadata