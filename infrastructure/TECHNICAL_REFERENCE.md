# Infrastructure Delegate - Technical Reference

## Quick Start

```python
from tidyllm.infrastructure.infra_delegate import get_infra_delegate

# Get infrastructure (decided once at startup)
infra = get_infra_delegate()

# Use it anywhere
conn = infra.get_db_connection()
response = infra.invoke_bedrock("prompt")
embedding = infra.generate_embedding("text")
```

## What Gets Detected

The delegate automatically detects and uses:

| Service | Parent Infrastructure | Fallback |
|---------|----------------------|----------|
| **Database** | ResilientPoolManager (3-pool failover) | SimpleConnectionPool |
| **AWS** | aws_service (unified client) | Direct boto3 |
| **LLM** | CorporateLLMGateway | Bedrock direct |
| **Embeddings** | SentenceTransformers | TF-IDF |

## API Reference

### Database Operations

```python
# Get connection
conn = infra.get_db_connection()

# Return to pool
infra.return_db_connection(conn)
```

### LLM Operations

```python
# Generate response
response = infra.generate_llm_response(
    prompt="Your prompt",
    config={
        'model': 'claude-3-sonnet',
        'temperature': 0.7,
        'max_tokens': 1500
    }
)

# Response format
{
    'success': True,
    'text': 'Generated response...',
    'model': 'claude-3-sonnet'
}
```

### Bedrock Operations

```python
# Invoke Bedrock directly
response = infra.invoke_bedrock(
    prompt="Your prompt",
    model_id="anthropic.claude-3-haiku-20240307-v1:0"
)
```

### Embedding Operations

```python
# Generate embedding
embedding = infra.generate_embedding("Text to embed")
# Returns: List[float] of dimension 384 or 768
```

## Environment Detection

The delegate checks for parent infrastructure at startup:

```python
# Initialization flow (happens once)
1. Try import infrastructure.services.resilient_pool_manager
   ✓ Found → Use ResilientPoolManager
   ✗ Not found → Use SimpleConnectionPool

2. Try import infrastructure.services.aws_service
   ✓ Found → Use parent aws_service
   ✗ Not found → Use boto3 directly

3. Try import tidyllm.gateways.corporate_llm_gateway
   ✓ Found → Use CorporateLLMGateway
   ✗ Not found → Use Bedrock directly

4. Try import sentence_transformers
   ✓ Found → Use SentenceTransformers
   ✗ Not found → Use TF-IDF fallback
```

## Configuration

### Database Configuration

When using fallback mode, reads from `tidyllm/admin/settings.yaml`:

```yaml
credentials:
  postgresql:
    host: localhost
    port: 5432
    database: rag_db
    username: rag_user
    password: rag_password
    ssl_mode: prefer
```

### AWS Configuration

Uses standard AWS credential chain:
1. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
2. ~/.aws/credentials
3. IAM role (if on EC2/ECS)

## Logging

Enable debug logging to see what's being used:

```python
import logging
logging.basicConfig(level=logging.INFO)

# Will log:
# ✅ Database: Using parent ResilientPoolManager (3-pool failover)
# ✅ AWS: Using parent aws_service
# ✅ LLM: Using parent CorporateLLMGateway
# ✅ Embeddings: Using SentenceTransformers
```

## Testing

For unit tests, mock the delegate:

```python
from unittest.mock import Mock

def test_my_adapter():
    # Create mock infra
    mock_infra = Mock()
    mock_infra.get_db_connection.return_value = Mock()
    mock_infra.invoke_bedrock.return_value = {
        'success': True,
        'text': 'Test response'
    }

    # Use in adapter
    adapter = MyRAGAdapter()
    adapter.infra = mock_infra

    # Test normally
    result = adapter.query(test_query)
    assert result is not None
```

## Migration Guide

### Old Pattern (Don't Use)
```python
# ❌ OLD - Multiple delegates
from tidyllm.infrastructure.delegates.database_delegate import get_database_delegate
from tidyllm.infrastructure.delegates.aws_delegate import get_aws_delegate

db_delegate = get_database_delegate()
aws_delegate = get_aws_delegate()
```

### New Pattern (Use This)
```python
# ✅ NEW - Single delegate
from tidyllm.infrastructure.infra_delegate import get_infra_delegate

infra = get_infra_delegate()
# Use infra for everything
```

## Troubleshooting

### Issue: Not detecting parent infrastructure

**Check:** Are you in the compliance-qa directory structure?
```bash
pwd  # Should show: /path/to/compliance-qa
```

**Check:** Can you import parent modules?
```python
from infrastructure.services.resilient_pool_manager import ResilientPoolManager
```

### Issue: Database connection errors

**Check:** Database credentials in settings.yaml
**Check:** Database is running and accessible
**Check:** Network connectivity

### Issue: AWS operations failing

**Check:** AWS credentials configured
```bash
aws sts get-caller-identity
```

**Check:** Region is us-east-1
**Check:** IAM permissions for Bedrock/S3

## Performance Notes

- **Startup:** One-time detection takes ~1-2 seconds
- **Runtime:** No performance overhead vs direct access
- **Memory:** Single delegate instance (singleton pattern)
- **Connections:** Reuses connection pools efficiently

## Architecture Benefits

1. **Single Source of Truth** - One delegate for all infrastructure
2. **Progressive Enhancement** - Better features when deployed
3. **Zero Configuration** - Works out of the box
4. **Clean Boundaries** - Adapters don't know infrastructure details
5. **Easy Testing** - Just mock the delegate

## Code Location

```
packages/tidyllm/infrastructure/
├── infra_delegate.py          # The consolidated delegate
├── ARCHITECTURE_PATTERN.md    # Pattern documentation
└── TECHNICAL_REFERENCE.md     # This file
```

## Support

For issues or questions:
- Check logs for detection status
- Verify parent infrastructure availability
- Review settings.yaml configuration
- Test with simple script first

Remember: **The delegate decides once at startup and uses that consistently.**