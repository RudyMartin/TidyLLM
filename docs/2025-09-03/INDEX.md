# Design Decisions Index

## Core Architecture Decisions (DD-001 to DD-099)

- **DD-001**: Multiple Competing Sentence Embedding Implementations - CRITICAL consolidation needed
- **DD-002**: DataTable Integration Completely Broken - Undefined variables and crashes  
- **DD-003**: Ecosystem Repository Fragmentation - 14+ repos creating maintenance nightmare
- **DD-004**: Import Strategy Chaos - Inconsistent patterns causing failures

## API and Interface Decisions (DD-101 to DD-199)

- **DD-101**: Chat Function Call Pattern - `chat(provider)(message)` standard pattern
- **DD-102**: Provider Factory Pattern - Consistent factory function signatures
- **DD-103**: Error Handling Pattern - Graceful error returns instead of exceptions

## Testing Strategy Decisions (DD-201 to DD-299)

- **DD-201**: Mock vs Real API Testing Strategy - Mock for development, real for validation
- **DD-202**: Evidence File Completeness - Always save `result.__dict__` for complete data
- **DD-203**: System Dependencies and Availability - Graceful degradation when dependencies missing
- **DD-204**: Test Isolation and State Management - Stateless tests with unique identifiers

## Data Format and Structure Decisions (DD-301 to DD-399)

- **DD-301**: Test Scripts Generally Follow Design Decisions - High compliance rate
- **DD-302**: Need Architectural Validation Tests - Test critical issue resolution  
- **DD-303**: Need Error Scenario Tests - Test failure modes and recovery

## Provider and Integration Decisions (DD-401 to DD-499)

- **DD-401**: Model Naming Convention - Provider-specific model identifier patterns
- **DD-402**: Provider Assumption Strategy - Primary focus on Claude and Bedrock

## Critical Issues Resolved

**Function Call Pattern Bug** (DD-101): Fixed `chat(message, provider)` → `chat(provider)(message)`
- Impact: Prevented function objects being returned instead of responses
- Evidence: All model comparison and load testing now works correctly

**Evidence Data Truncation Bug** (DD-202): Fixed manual summaries → complete `result.__dict__`
- Impact: Evidence files now contain full diagnostic data instead of minimal summaries
- Evidence: Memory monitoring and cost analysis files now >500 characters vs <200

**Model Provider Assumptions** (DD-401, DD-402): Standardized naming and availability
- Impact: Consistent test patterns across Claude and Bedrock integrations
- Evidence: Tests no longer fail due to model name mismatches

## Usage Examples

**Correct API Pattern**:
```python
response = chat(claude("claude-3-haiku"))(LLMMessage("Hello"))
response = chat(bedrock("anthropic.claude-3-haiku-20240307-v1:0"))(LLMMessage("Hello"))
```

**Correct Evidence Pattern**:
```python
evidence_path = self.save_evidence(result.__dict__, "test_name")
```

**Correct Model Testing Pattern**:
```python
models = [("claude", "claude-3-haiku"), ("bedrock", "anthropic.claude-3-haiku-20240307-v1:0")]
```