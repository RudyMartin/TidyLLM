# TidyLLM Gateway - Enterprise LLM Governance

## Overview

TidyLLM Gateway is a corporate-safe LiteLLM alternative that provides enterprise governance to existing LLMData workflows while maintaining full backward compatibility. It implements the **Unified Gateway** approach (Option 3) that seamlessly bridges enterprise controls with tidyverse-style LLM workflows.

## Architecture

```
Existing Macros → LLMData Verbs → TidyLLM Gateway → MLFlow → LLM Providers
                               (Enterprise Layer Added Here)
```

### Key Components

- **LiteLLM Clone**: Drop-in replacement with 100+ provider support
- **Enterprise Governance**: Audit trails, cost tracking, budget enforcement
- **MLFlow Backend**: Corporate-controlled LLM provider access
- **LLMData Integration**: Enhances existing verbs with zero breaking changes
- **Spend Tracking**: Multi-level budgets with real-time monitoring

## Corporate Security Model

✅ **Corporate-Safe Design**
- No direct LLM provider API dependencies
- All access routed through MLFlow Gateway
- IT-controlled provider registry
- Centralized authentication and rate limiting

❌ **What We Don't Do**
- No `anthropic`, `openai`, `google-cloud-aiplatform` dependencies
- No direct API key management in application code
- No bypassing corporate firewalls or proxy settings

## Installation

```bash
# Core package (corporate-safe)
pip install -e packages/tidyllm-gateway/

# Optional: Enhanced features
pip install mlflow datatable tidymart
```

## Quick Start

### Existing LLMData Code (Works Unchanged)

```python
from llmdata import llm_message, chat, claude

# Your existing code continues to work
response = llm_message("Analyze quarterly data") | chat(claude())
```

### Add Enterprise Governance (Optional)

```python
from llmdata import set_enterprise_context

# Set context once for the session
set_enterprise_context(
    user_id="analyst@company.com",
    department="risk-management",
    audit_reason="Q4 model validation"
)

# Same code, now automatically governed
response = llm_message("Analyze quarterly data") | chat(claude())
```

### Enterprise Features Automatically Added

When TidyLLM Gateway is available, all LLM operations gain:
- ✅ Audit trail recording
- ✅ Cost tracking and budgets  
- ✅ MLFlow backend routing
- ✅ Corporate compliance controls

## Enterprise Features

### 1. Provider Registry (100+ Models)

```python
from tidyllm_gateway import get_provider_registry

registry = get_provider_registry()
models = registry.get_approved_models(
    department="risk-management",
    security_tier="confidential"
)
```

### 2. Spend Tracking & Budgets

```python
from tidyllm_gateway.enterprise import EnterpriseSpendTracker

tracker = EnterpriseSpendTracker()

# Create department budget
tracker.create_budget(
    name="Risk Analysis Q4",
    limit_usd=5000.0,
    department="risk-management",
    hard_limit=True
)

# Real-time monitoring
status = tracker.get_budget_status("risk-management")
```

### 3. Audit & Compliance

```python
# All operations automatically logged with:
{
    "user_id": "analyst@company.com",
    "department": "risk-management", 
    "audit_reason": "Model validation review",
    "timestamp": "2025-01-15T10:30:00Z",
    "model": "claude-3-5-sonnet",
    "cost_usd": 0.0045,
    "tokens": 150
}
```

### 4. Cost Optimization & Fallbacks

```python
# Intelligent fallback routing
response = chat(claude(), fallbacks=[
    "claude-3-haiku",  # Cost-effective fallback
    "gpt-4o-mini",     # Cross-provider fallback
    "local-llama3"     # On-premises fallback
])
```

## Integration with Existing Systems

### MVR Macros Enhanced

```python
from src.backend.macros.built_ins.mvr_peer_review import create_mvr_peer_review_macro

# Macro now supports enterprise context
macro = create_mvr_peer_review_macro(
    user_id="mvr_analyst@company.com",
    department="model_validation"
)

# All LLM operations in macro automatically governed
```

### Advanced QA Orchestrator

```python
from src.backend.mcp.orchestrators.advanced_qa_orchestrator import AdvancedQAOrchestrator

# Enterprise configuration
config = {
    'user_id': 'qa_analyst@company.com',
    'department': 'model_validation',
    'max_cost_usd': 100.0
}

orchestrator = AdvancedQAOrchestrator(config)
```

## Configuration

### Enterprise Settings

```yaml
# config/enterprise.yaml
provider_registry:
  approved_models:
    - model: "claude-3-5-sonnet"
      providers: ["anthropic"]
      security_tier: "confidential"
      max_cost_per_request: 1.0
      
spend_tracking:
  default_budgets:
    department: 1000.0  # USD per month
    user: 100.0        # USD per month
    
audit:
  retention_days: 90
  log_level: "info"
```

### MLFlow Gateway Setup

```yaml
# mlflow_gateway_config.yaml
routes:
  - name: "claude"
    route_type: "llm/v1/chat"
    model:
      provider: "anthropic"
      name: "claude-3-5-sonnet"
      config:
        anthropic_api_key: "$ANTHROPIC_API_KEY"
```

## API Reference

### Core Functions

```python
# Enhanced LLMData verbs (backward compatible)
from llmdata import chat, embed, analyze_data, send_batch

# Enterprise context management  
from llmdata import set_enterprise_context, get_enterprise_context

# Provider management
from tidyllm_gateway import completion, embedding
```

### LiteLLM Compatibility

```python
# Drop-in LiteLLM replacement
from tidyllm_gateway.litellm_clone import completion, embedding

response = completion(
    model="claude-3-5-sonnet",
    messages=[{"role": "user", "content": "Hello"}],
    user_id="analyst@company.com",
    audit_reason="Customer analysis"
)
```

## Deployment

### Development Environment

```bash
# Install with development dependencies
pip install -e "packages/tidyllm-gateway/[dev]"

# Run tests
python test_enterprise_llmdata_integration.py
```

### Production Environment

1. **Deploy MLFlow Gateway**
```bash
mlflow gateway start --config-path config/mlflow_gateway_config.yaml
```

2. **Configure Environment**
```bash
export MLFLOW_GATEWAY_URI="https://mlflow-gateway.company.com"
export ENTERPRISE_CONFIG_PATH="/etc/tidyllm/enterprise.yaml"
```

3. **Install Package**
```bash
pip install tidyllm-gateway[production]
```

## Migration Guide

### From Direct LLM APIs

```python
# Before: Direct API calls
import anthropic
client = anthropic.Anthropic(api_key="...")
response = client.messages.create(...)

# After: Enterprise-governed
from llmdata import llm_message, chat, claude
response = llm_message("...") | chat(claude())
```

### From LiteLLM

```python
# Before: LiteLLM
from litellm import completion
response = completion(model="claude-3-5-sonnet", ...)

# After: TidyLLM Gateway (drop-in replacement)
from tidyllm_gateway.litellm_clone import completion
response = completion(model="claude-3-5-sonnet", user_id="...", ...)
```

## Troubleshooting

### Common Issues

**Q: "Enterprise governance not working"**
```python
from llmdata import ENTERPRISE_GOVERNANCE_AVAILABLE
print(f"Enterprise available: {ENTERPRISE_GOVERNANCE_AVAILABLE}")
```

**Q: "MLFlow connection failed"**
```bash
# Check MLFlow Gateway status
curl -X GET "http://localhost:5000/gateway/routes"
```

**Q: "Budget exceeded errors"**
```python
from tidyllm_gateway.enterprise import EnterpriseSpendTracker
tracker = EnterpriseSpendTracker()
status = tracker.get_budget_status("your-department")
```

## Examples

See `/examples/` directory for complete working examples:
- `enterprise_integration_demo.py` - Full integration demonstration
- `04_multimodal_attachments_demo.py` - File processing with governance
- `test_enterprise_llmdata_integration.py` - Testing framework

## Support

- Documentation: [Internal Wiki Link]
- Issues: Contact IT Architecture Team
- Slack: #llm-governance

## License

Internal Use - Company Proprietary