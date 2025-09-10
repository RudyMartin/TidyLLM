# API Pattern Design Decisions

## Decision ID: DD-101 - Chat Function Call Pattern

**Date**: 2025-09-03  
**Context**: Evidence review revealed chat function was returning function objects instead of responses due to incorrect call patterns.

### Decision

**Standard Chat Pattern**: `chat(provider)(message)`
```python
# Correct pattern
response = chat(claude("claude-3-haiku"))(LLMMessage("Hello"))
response = chat(bedrock("anthropic.claude-3-haiku-20240307-v1:0"))(LLMMessage("Hello"))

# Incorrect pattern (causes function object returns)
response = chat(LLMMessage("Hello"), claude("claude-3-haiku"))  # WRONG
```

### Rationale

1. **Functional Programming**: Follows curried function pattern
2. **Provider Separation**: Clear separation between provider config and message
3. **Composability**: Allows for easy function composition and reuse
4. **Type Safety**: Enables better type checking and IDE support

### Function Signature

```python
def chat(provider: Provider, stream: bool = False, **kwargs) -> Callable[[LLMMessage], str]:
    """
    Returns a function that takes LLMMessage and returns response string
    """
    def _chat(msg: LLMMessage) -> str:
        # Implementation here
        pass
    return _chat
```

### Consequences

**For Tests**:
- All test files must use the `chat(provider)(message)` pattern
- Direct function calls like `claude(prompt, model="...")` will fail
- Mock implementations must return callable functions

**For Implementation**:
- Chat function returns a callable, not a direct response
- Provider functions (claude, bedrock) return Provider objects
- Error handling must account for the two-step call pattern

### Examples

**Load Testing Pattern**:
```python
def make_chat_request(user_id, request_id):
    prompt = f"Hello from user {user_id}, request {request_id}"
    response = chat(claude("claude-3-haiku"))(LLMMessage(prompt))
    return response
```

**Model Comparison Pattern**:
```python
class ClaudeProvider:
    async def generate_response(self, prompt: str, model: str) -> ModelResponse:
        response = chat(claude(model))(LLMMessage(prompt))
        return ModelResponse(response=response, ...)
```

## Decision ID: DD-102 - Provider Factory Pattern

**Date**: 2025-09-03  
**Context**: Consistency needed across provider creation functions.

### Decision

**Factory Function Signature**: All provider factories follow the same pattern
```python
def provider_name(model: str = default_model, **kwargs) -> Provider:
    return Provider(provider_name, model, **kwargs)
```

**Standard Factories**:
```python
claude(model="claude-3-5-sonnet", **kwargs)
bedrock(model="anthropic.claude-3-sonnet-20240229-v1:0", **kwargs) 
ollama(model="llama3.1", **kwargs)
gemini(model="gemini-pro", **kwargs)
```

### Rationale

1. **Consistency**: Same interface across all providers
2. **Flexibility**: Kwargs allow provider-specific configuration
3. **Defaults**: Sensible defaults for each provider
4. **Extensibility**: Easy to add new providers

### Consequences

**Provider Configuration**:
```python
# With defaults
provider = claude()  # Uses claude-3-5-sonnet

# With specific model
provider = claude("claude-3-haiku")

# With additional config
provider = claude("claude-3-haiku", temperature=0.1, max_tokens=1000)

# Bedrock with region
provider = bedrock("anthropic.claude-3-haiku-20240307-v1:0", region="us-west-2")
```

## Decision ID: DD-103 - Error Handling Pattern

**Date**: 2025-09-03  
**Context**: Tests need consistent error handling to avoid cascading failures.

### Decision

**Graceful Error Handling**: Functions should return error messages rather than raise exceptions in test environments

```python
def _chat(msg: LLMMessage) -> str:
    try:
        # API call here
        return response_content
    except Exception as e:
        # Return error message instead of raising
        error_msg = f"Chat error: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg
```

### Rationale

1. **Test Stability**: Prevents single API failures from breaking entire test suites
2. **Evidence Collection**: Error responses still generate evidence data
3. **Debugging**: Error messages are preserved in evidence files
4. **Graceful Degradation**: System continues functioning with partial failures

### Consequences

**For Tests**:
- Tests should check for error messages in responses
- Evidence files will contain error information instead of missing data
- Mock implementations should return error strings when appropriate

**Error Message Patterns**:
```python
"Chat error: claude() got multiple values for argument 'model'"
"Mock response to: Hello world..."
"API timeout: Request exceeded 30 second limit"
```