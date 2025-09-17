# Step 5: Output Generation

## Purpose
Generate final structured output with comprehensive metadata and audit trail.

## Prompt Template

```
You are a data formatting specialist responsible for creating the final workflow output. Using the synthesis results from Step 4, generate:

1. **Structured Final Report:**
   - Compile all workflow results into final format
   - Ensure JSON schema compliance
   - Include comprehensive metadata

2. **Quality Validation:**
   - Validate all required fields are present
   - Verify data consistency across steps
   - Calculate overall workflow confidence

3. **Audit Trail Generation:**
   - Document complete processing history
   - Include error handling and recovery actions
   - Generate processing summary statistics

## Input Variables
- `{synthesis_results}` - Output from Step 4 synthesis
- `{output_format}` - Final output format (json, xml, csv)
- `{include_metadata}` - Whether to include detailed metadata

## RAG Integration Points
- Query output format standards and schemas
- Access data validation rules and best practices
- Retrieve audit trail requirements for compliance

## Expected Output Format
```json
{
  "final_report": {
    "workflow_execution": {
      "workflow_id": "sequential_analysis_001",
      "execution_id": "exec_20250115_120000_001",
      "execution_timestamp": "2025-01-15T12:00:00Z",
      "total_processing_time_ms": 2147483,
      "status": "completed|failed|partial"
    },
    "step_results": [
      {
        "step_number": 1,
        "step_name": "Input Validation",
        "status": "completed",
        "processing_time_ms": 245123,
        "output_summary": "5 documents validated successfully",
        "confidence": 0.95
      }
    ],
    "final_analysis": {
      "executive_summary": "Executive summary from synthesis step",
      "key_findings": [
        {
          "finding": "Primary finding description",
          "significance": "high|medium|low",
          "supporting_evidence": "Evidence details"
        }
      ],
      "risk_assessment": {
        "overall_risk_level": "high|medium|low",
        "critical_risks": ["List of critical risks"],
        "risk_mitigation_priority": "immediate|short_term|long_term"
      },
      "recommendations": [
        {
          "recommendation": "Specific recommendation",
          "priority": "high|medium|low",
          "implementation_timeline": "Timeline for implementation",
          "expected_impact": "Expected business impact"
        }
      ],
      "confidence_metrics": {
        "overall_confidence": 0.89,
        "data_quality_score": 0.92,
        "analysis_completeness": 0.87,
        "recommendation_certainty": 0.85
      }
    }
  },
  "metadata": {
    "input_files_processed": [
      {
        "filename": "document1.pdf",
        "size_bytes": 1048576,
        "processing_time_ms": 45123,
        "quality_score": 0.91
      }
    ],
    "rag_systems_used": ["ai_powered", "intelligent", "dspy"],
    "template_fields_resolved": {
      "quality_threshold": 0.85,
      "domain_context": "financial_analysis",
      "analysis_depth": "comprehensive"
    },
    "processing_environment": {
      "workflow_version": "1.0",
      "execution_mode": "sequential",
      "rag2dag_optimization": true,
      "performance_optimizations": ["parallel_rag", "caching_enabled"]
    }
  },
  "audit_trail": {
    "steps_executed": [
      {
        "step": 1,
        "started_at": "2025-01-15T12:00:00Z",
        "completed_at": "2025-01-15T12:00:05Z",
        "rag_queries": 3,
        "retry_attempts": 0
      }
    ],
    "errors_encountered": [
      {
        "step": 2,
        "error_type": "validation_warning",
        "message": "Low confidence score for entity extraction",
        "resolution": "Proceeded with manual review flag"
      }
    ],
    "recovery_actions": [
      "Applied fallback extraction method in step 2",
      "Enhanced confidence threshold validation"
    ],
    "performance_metrics": {
      "total_rag_queries": 15,
      "cache_hit_rate": 0.73,
      "average_step_time_ms": 429696,
      "peak_memory_usage_mb": 256
    },
    "final_status": "completed_successfully"
  },
  "processing_summary": {
    "workflow_success": true,
    "steps_completed": 5,
    "steps_failed": 0,
    "overall_quality_score": 0.89,
    "processing_efficiency": 0.91,
    "output_completeness": 1.0,
    "ready_for_delivery": true
  }
}
```

## Success Criteria
- All required fields populated
- JSON schema validation passes
- Audit trail complete and accurate
- Processing summary calculated correctly
- Final confidence scores above threshold
- Output ready for consumption by downstream systems
```