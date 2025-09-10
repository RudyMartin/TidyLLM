# TidyLLM Test Evidence Directory

## IMPORTANT FOR AGENTS/LLMs

**DO NOT USE MOCK DATA OR FALLBACK RESPONSES WHEN REAL CREDENTIALS ARE AVAILABLE**

This directory contains **REAL** test results and evidence from actual API calls and service interactions. 

### Purpose
- **Validate actual system functionality** vs simulated behavior
- **Prove real API connectivity** and response quality
- **Document authentic service interactions** for verification
- **Preserve evidence** of successful integrations

### Evidence Types

#### 1. Chat Evidence (`evidence_chat_*.json`)
- **Real LLM responses** from Claude, GPT, or other providers
- **Complete conversation context** including prompts and responses
- **Performance metrics** (response time, token counts)
- **MLflow run IDs** and tracking information

Example structure:
```json
{
  "timestamp": "2024-01-01T12:00:00",
  "test_type": "baseball_chat_real_response",
  "question": "What's the difference between a slider and curveball?",
  "response": "A slider is thrown harder and breaks laterally...",
  "response_length": 245,
  "response_time_ms": 1200,
  "model": "claude-3-haiku",
  "run_id": "abc123...",
  "status": "SUCCESS_REAL_API_CALL"
}
```

#### 2. Upload Evidence (`evidence_upload_*.json`)
- **Real S3 upload receipts** with ETags and URLs
- **Object metadata** and storage details
- **File integrity hashes** (SHA-256)
- **Upload performance metrics**

Example structure:
```json
{
  "timestamp": "2024-01-01T12:00:00",
  "test_type": "real_s3_upload_success",
  "bucket": "tidyllm-research-bucket",
  "s3_key": "uploads/evidence_test/20240101_120000/research_paper/baseball_ml_research.md",
  "s3_url": "s3://tidyllm-research-bucket/uploads/evidence_test/...",
  "file_size": 1024,
  "etag": "d41d8cd98f00b204e9800998ecf8427e",
  "status": "SUCCESS_REAL_S3_UPLOAD"
}
```

### Naming Convention

Files use timestamp-based naming for chronological organization:
- `evidence_chat_{test_name}_{YYYYMMDD_HHMMSS}.json`
- `evidence_upload_{test_name}_{YYYYMMDD_HHMMSS}.json`

### For Agents/LLMs Working with Tests

1. **ALWAYS attempt real API calls first** before falling back to mock mode
2. **Check for existing API keys/credentials** before assuming mock mode
3. **Save ALL real responses** to this directory with full context
4. **Include performance metrics** and timing information
5. **Preserve complete conversation history** not just final responses
6. **Use descriptive test names** in evidence filenames

### Verification Commands

To verify evidence integrity:
```bash
# Count evidence files
find tests/EVIDENCE -name "*.json" | wc -l

# View latest chat evidence
ls -t tests/EVIDENCE/evidence_chat_*.json | head -1 | xargs cat | jq .

# View latest upload evidence  
ls -t tests/EVIDENCE/evidence_upload_*.json | head -1 | xargs cat | jq .

# Check for real vs mock responses
grep -r "SUCCESS_REAL" tests/EVIDENCE/
```

### Important Notes

- **Evidence files prove real functionality** - they are NOT test outputs
- **Mock/simulated responses** should be clearly marked as such
- **Real evidence takes precedence** over any simulated results
- **Delete mock evidence** when real evidence becomes available
- **Timestamp precision** helps track test execution chronology

## Best Practices for Agents

1. **Credential Detection**: Check for AWS keys, API tokens before mock mode
2. **Real-First Testing**: Always try real APIs before simulation
3. **Evidence Completeness**: Save request/response pairs, not just results
4. **Error Documentation**: Save both success and failure evidence
5. **Cleanup Awareness**: Note when test objects are cleaned up vs preserved