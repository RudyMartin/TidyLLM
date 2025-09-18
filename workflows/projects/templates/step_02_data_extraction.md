# Step 2: Data Extraction

## Purpose
Extract structured data and key information from validated documents.

## Prompt Template

```
You are a data extraction specialist. Using the validated documents from Step 1, perform:

1. **Content Extraction:**
   - Extract key facts, figures, and data points
   - Identify main themes and topics
   - Extract entities (people, organizations, dates, amounts)

2. **Structure Analysis:**
   - Identify document sections and hierarchy
   - Extract tables, lists, and structured data
   - Map relationships between data elements

3. **Context Preservation:**
   - Maintain source references for all extracted data
   - Preserve context and meaning
   - Note uncertainties or ambiguities

## Input Variables
- `{validated_documents}` - Output from Step 1
- `{extraction_schema}` - Desired data structure
- `{entity_types}` - Types of entities to extract

## RAG Integration Points
- Query domain-specific knowledge bases
- Validate extracted entities against known data
- Enhance extraction with contextual information

## Expected Output Format
```json
{
  "extraction_status": "completed|partial|failed",
  "extracted_data": {
    "entities": {
      "people": [],
      "organizations": [],
      "dates": [],
      "amounts": []
    },
    "key_facts": [],
    "structured_data": [],
    "themes": []
  },
  "source_mapping": {
    "data_point_id": {
      "source_file": "document.pdf",
      "page": 5,
      "confidence": 0.95
    }
  },
  "processing_notes": []
}
```

## Success Criteria
- High-confidence data extraction (>90%)
- Complete entity identification
- Preserved source traceability
- Ready for analysis step
```