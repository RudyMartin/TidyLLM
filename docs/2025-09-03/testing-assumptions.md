# Testing Assumptions Design Decisions

## Decision ID: DD-201 - Mock vs Real API Testing Strategy

**Date**: 2025-09-03  
**Context**: Tests were failing due to inconsistent assumptions about API availability and mock implementations.

### Decision

**Testing Environment Strategy**:
- **Development**: Use mock responses for speed and reliability
- **Integration**: Use real APIs for critical path validation  
- **Evidence Generation**: Always generate evidence regardless of mock/real status

**Mock Response Pattern**:
```python
# Mock responses should be descriptive and identifiable
response_content = f"Mock response to: {processed_content[:100]}..."

# Real responses would be actual API content
response_content = actual_api_response.text
```

### Rationale

1. **Test Reliability**: Mocks prevent external API failures from breaking tests
2. **Development Speed**: Fast iteration without API rate limits
3. **Evidence Integrity**: Evidence files show what actually happened
4. **Cost Control**: Reduces API usage costs during development

### Consequences

**For Evidence Files**:
- Mock responses clearly identifiable: `"Mock response to: Hello..."`
- Real responses contain actual model outputs
- Error messages distinguish between mock and API failures
- Performance metrics reflect actual vs simulated timing

**For Test Design**:
```python
# Test should work with both mock and real responses
def test_chat_functionality():
    response = chat(claude("claude-3-haiku"))(LLMMessage("Test prompt"))
    
    # Check response exists and has content
    assert isinstance(response, str)
    assert len(response) > 0
    
    # Don't assert specific content since it could be mock or real
    # Instead check for error patterns
    assert not response.startswith("Chat error:")
```

## Decision ID: DD-202 - Evidence File Completeness

**Date**: 2025-09-03  
**Context**: Evidence review found truncated files missing critical data needed for debugging.

### Decision

**Complete Evidence Preservation**: Always save `result.__dict__` instead of manual summaries

```python
# Correct pattern
evidence_path = self.save_evidence(result.__dict__, "test_name")

# Incorrect pattern (causes data loss)
evidence = {"summary": "partial data"}  # WRONG
evidence_path = self.save_evidence(evidence, "test_name")
```

### Rationale

1. **Debugging**: Complete data needed to understand test failures
2. **Analysis**: Allows post-test analysis of performance patterns
3. **Consistency**: Same evidence format across all tests
4. **Future-Proofing**: Preserves data that might be needed later

### Consequences

**Evidence File Structure**:
```json
{
  "test_id": "load_test_concurrent_chat_1756902525",
  "config": "LoadTestConfig(...)",
  "start_time": "2025-09-03T07:28:45.940893",
  "total_requests": 15,
  "successful_requests": 15,
  "requests": ["RequestResult(...)", "..."],
  "system_metrics": ["SystemMetrics(...)", "..."],
  "performance_summary": {
    "response_time_analysis": {...},
    "throughput_analysis": {...}
  }
}
```

**Testing Requirements**:
- All tests must use the complete evidence pattern
- Evidence files should be >500 characters (not <200)
- Critical fields: `config`, `requests`, `system_metrics`, `performance_summary`

## Decision ID: DD-203 - System Dependencies and Availability

**Date**: 2025-09-03  
**Context**: Tests assume various system components are available but may not be in all environments.

### Decision

**Graceful Degradation Pattern**: Check for dependency availability and adapt behavior

```python
# Check for MLflow availability
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("WARNING: MLFlow not available - experiment tracking disabled")

# Check for attachments system
try:
    from .attachments_enhanced import attach, load, present
    ATTACHMENTS_AVAILABLE = True
except ImportError:
    ATTACHMENTS_AVAILABLE = False
    # Provide placeholder functions
```

### Rationale

1. **Environment Flexibility**: Tests work in minimal and full environments
2. **Clear Feedback**: Users know which features are available
3. **Partial Success**: Core functionality works even if extras are missing
4. **Development Ease**: New developers don't need full setup immediately

### Consequences

**Dependency Patterns**:
```python
# Gateway availability
if gateway_available:
    response = gateway.query(endpoint, data)
else:
    response = {"mock": "response for testing"}

# Database availability  
if database_available:
    result = db.execute_query(sql)
else:
    result = mock_database_result()
```

**Test Adaptation**:
- Tests should pass with or without optional dependencies
- Evidence files should indicate which systems were available
- Mock implementations should be functionally equivalent

## Decision ID: DD-204 - Test Isolation and State Management

**Date**: 2025-09-03  
**Context**: Tests were interfering with each other due to shared state.

### Decision

**Stateless Test Pattern**: Each test should be independent and not rely on external state

```python
def test_functionality(self, settings_loader):
    # Create fresh instances for each test
    config = LoadTestConfig(test_name="unique_name_with_timestamp")
    tester = get_load_tester()  # Fresh instance
    
    # Run test with isolated data
    result = await tester.run_load_test(config)
    
    # Save evidence with unique timestamp
    evidence_path = self.save_evidence(result.__dict__, f"{test_name}_{timestamp}")
```

### Rationale

1. **Reliability**: Tests don't fail due to previous test state
2. **Parallelization**: Tests can run concurrently if needed
3. **Debugging**: Failures are isolated to specific functionality
4. **Evidence Clarity**: Each test produces its own evidence

### Consequences

**Test Structure Requirements**:
- Unique test identifiers with timestamps
- Fresh object instances for each test
- Independent evidence files
- No shared global state between tests

**Evidence File Naming**:
```
evidence_load_concurrent_chat_20250903_072856.json
evidence_models_claude_comparison_20250903_072929.json
evidence_mlflow_custom_metrics_20250903_072335.json
```