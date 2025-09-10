# Model Provider Design Decisions

## Decision ID: DD-401 - Model Naming Convention

**Date**: 2025-09-03  
**Context**: During evidence review, we discovered inconsistent model naming and provider assumptions causing test failures.

### Decision

**Model Name Pattern**: `provider:model-identifier`
- Claude: `"claude-3-haiku"`, `"claude-3-sonnet"`, `"claude-3-5-sonnet"`  
- Bedrock: `"anthropic.claude-3-haiku-20240307-v1:0"`, `"anthropic.claude-3-sonnet-20240229-v1:0"`
- OpenAI: `"gpt-4"`, `"gpt-3.5-turbo"`

**Provider Factory Pattern**:
```python
# Correct usage
claude_provider = claude("claude-3-haiku")
bedrock_provider = bedrock("anthropic.claude-3-haiku-20240307-v1:0")

# Usage in chat
response = chat(claude("claude-3-haiku"))(LLMMessage("Hello"))
```

### Rationale

1. **Consistency**: Each provider has its own naming scheme that we preserve
2. **Clarity**: Model names clearly indicate their capabilities and versions  
3. **Testing**: Allows tests to assume specific model availability patterns
4. **API Compatibility**: Matches actual provider API requirements

### Consequences

**For Tests**:
- Test files can assume Claude models use simple names like `"claude-3-haiku"`
- Bedrock tests must use full ARN-style names
- Mock responses should match the naming convention

**For Implementation**:
- Provider factory functions validate model name formats
- Error messages should indicate expected naming patterns
- Documentation must show examples for each provider

### Examples

**Working Test Pattern**:
```python
# Claude comparison
models = [("claude", "claude-3-haiku"), ("claude", "claude-3-sonnet")]

# Bedrock comparison  
models = [
    ("bedrock", "anthropic.claude-3-haiku-20240307-v1:0"),
    ("bedrock", "anthropic.claude-3-sonnet-20240229-v1:0")
]
```

**Provider Configuration**:
```python
config = {
    "claude": {"cost_per_token": 0.0001},
    "bedrock": {"cost_per_token": 0.0001, "region": "us-east-1"},
    "openai": {"cost_per_token": 0.00002}
}
```

## Decision ID: DD-402 - Provider Assumption Strategy

**Date**: 2025-09-03  
**Context**: Tests were failing because they assumed all providers are available, but in reality we primarily support Bedrock and Claude.

### Decision

**Primary Providers**: Focus on Claude and Bedrock as the primary supported providers
- Tests should primarily use these two providers
- Other providers (OpenAI, Gemini) are secondary and may not be available in all environments

**Testing Strategy**:
- Use Claude for most basic functionality tests
- Use Bedrock for AWS-specific integration tests
- Mock or skip other providers if not available

### Rationale

1. **Reliability**: Claude and Bedrock are the most stable integrations
2. **Business Focus**: Primary use cases involve these providers
3. **Testing Stability**: Reduces test failures due to API availability
4. **Cost Management**: Limits testing costs to known providers

### Consequences

**For Tests**:
```python
# Preferred test pattern
models = [("claude", "claude-3-haiku"), ("bedrock", "anthropic.claude-3-haiku-20240307-v1:0")]

# Not recommended for critical tests
models = [("openai", "gpt-4"), ("gemini", "gemini-pro")]
```

**For Implementation**:
- Provider availability checks should gracefully handle missing providers
- Error messages should suggest using Claude or Bedrock as alternatives
- Configuration templates should include Claude and Bedrock examples