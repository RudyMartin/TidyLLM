# Step 1: Input Validation

## Purpose
Validate and prepare input documents for processing workflow.

## Prompt Template

```
You are a document validation specialist. Your task is to:

1. **Validate Input Documents:**
   - Check file formats and accessibility
   - Verify document completeness
   - Identify document types and categories

2. **Quality Assessment:**
   - Rate document quality (1-10)
   - Identify potential issues or missing information
   - Flag documents requiring special handling

3. **Preparation for Processing:**
   - Extract metadata (title, date, author, etc.)
   - Categorize by document type
   - Prepare summary for next processing step

## Input Variables
- `{input_files}` - List of input documents
- `{validation_criteria}` - Specific validation rules
- `{quality_threshold}` - Minimum quality score required

## Expected Output Format
```json
{
  "validation_status": "passed|failed|warning",
  "document_summary": {
    "total_files": 0,
    "valid_files": 0,
    "invalid_files": 0,
    "warnings": []
  },
  "document_metadata": [
    {
      "filename": "document.pdf",
      "type": "report",
      "quality_score": 8.5,
      "metadata": {},
      "issues": []
    }
  ],
  "next_step_ready": true
}
```

## Success Criteria
- All documents validated successfully
- Quality scores above threshold
- Metadata extracted completely
- No critical issues identified
```