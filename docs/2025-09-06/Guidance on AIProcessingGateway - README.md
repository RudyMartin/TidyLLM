# Guidance on AIProcessingGateway - README

**Document Version**: 1.0  
**Created**: 2025-09-06  
**Status**: Official Gateway Guidance  
**Priority**: MANDATORY READING FOR ALL AI AGENTS

---

## 🔴 CRITICAL: This is Gateway #2 - The Processing Layer

**AI AGENTS**: This gateway is the PROCESSING layer that builds on CorporateLLMGateway. It provides multi-model AI processing, caching, and enterprise features. You MUST understand CorporateLLMGateway before using this gateway.

---

## 📋 **Executive Summary**

The **AIProcessingGateway** is the second-tier gateway in TidyLLM's three-tier architecture. It provides multi-model AI processing capabilities, response caching, retry logic, and performance monitoring. This gateway transforms the basic LLM access from CorporateLLMGateway into a sophisticated AI processing engine.

### **Position in Hierarchy**
```
Level 1: CorporateLLMGateway (Foundation)
         └── Provides: Basic LLM access control
              ↓
Level 2: AIProcessingGateway (THIS GATEWAY)
         ├── Requires: CorporateLLMGateway
         ├── Adds: Caching, retries, monitoring
         └── Provides: Multi-model processing
              ↓
Level 3: WorkflowOptimizerGateway
         └── Requires: AIProcessingGateway + CorporateLLMGateway
```

---

## 🎯 **Core Purpose & Responsibilities**

### **Primary Responsibilities**
1. **Multi-Model Processing**: Support for multiple AI backends (Bedrock, OpenAI, Anthropic, etc.)
2. **Performance Optimization**: Response caching with configurable TTL
3. **Reliability**: Retry logic with exponential backoff
4. **Monitoring**: Metrics tracking and performance monitoring
5. **Abstraction**: Unified interface across different AI providers

### **What It Does**
- Processes AI requests through multiple backend providers
- Caches responses for improved performance
- Implements retry logic for failed requests
- Tracks metrics and performance data
- Provides unified API across different AI models
- Handles backend selection (AUTO, Bedrock, OpenAI, etc.)

### **What It Does NOT Do**
- Does NOT handle corporate access control (that's CorporateLLMGateway)
- Does NOT optimize workflows (that's WorkflowOptimizerGateway)
- Does NOT manage databases or files (that's other gateways)
- Does NOT make architectural decisions about workflows

---

## 🏗️ **Technical Architecture**

### **Location**
`tidyllm/gateways/ai_processing_gateway.py`

### **Dependencies**
```python
# Required Dependencies:
from .corporate_llm_gateway import CorporateLLMGateway  # MANDATORY

# Optional Backend Integrations:
import boto3  # For Bedrock backend
import openai  # For OpenAI backend  
import anthropic  # For Anthropic backend
```

### **Key Classes**
```python
class AIProcessingGateway(BaseGateway):
    """Multi-model AI processing with enterprise features."""
    
class AIRequest:
    """Request structure for AI processing operations."""
    
class AIBackend(Enum):
    """Available AI backends."""
    AUTO = "auto"           # Automatic selection
    BEDROCK = "bedrock"     # AWS Bedrock
    OPENAI = "openai"       # OpenAI API
    ANTHROPIC = "anthropic" # Anthropic API
    MLFLOW = "mlflow"       # MLFlow Gateway
    MOCK = "mock"           # Testing/Development
```

---

## 🔧 **Configuration & Setup**

### **Standard Configuration**
```python
from tidyllm.gateways import init_gateways

registry = init_gateways({
    # REQUIRED: Corporate LLM must be configured first
    "corporate_llm": {
        "available_providers": ["claude", "openai-corporate"],
        "budget_limit_daily_usd": 500.00
    },
    
    # AI Processing Gateway Configuration
    "ai_processing": {
        # Backend Selection
        "backend": "auto",  # auto, bedrock, openai, anthropic, mlflow
        "fallback_backends": ["bedrock", "openai"],
        
        # Default Model Settings
        "default_model": "claude-3-sonnet",
        "max_tokens": 2000,
        "temperature": 0.7,
        "timeout_seconds": 30,
        
        # Performance Features
        "enable_caching": True,
        "cache_ttl_seconds": 3600,  # 1 hour
        "enable_retries": True,
        "max_retries": 3,
        "retry_delay_base": 1.0,    # Exponential backoff base
        
        # Monitoring
        "enable_metrics": True,
        "metrics_backend": "cloudwatch",  # cloudwatch, prometheus, none
        
        # Backend-Specific Configuration
        "bedrock_config": {
            "region": "us-east-1",
            "model_ids": {
                "claude": "anthropic.claude-3-sonnet-20240229-v1:0"
            }
        },
        "openai_config": {
            "organization": "your-org-id",
            "api_version": "2023-12-01-preview"
        }
    }
})
```

### **Environment Variables**
```bash
# Backend Selection
export AI_PROCESSING_BACKEND="auto"
export AI_PROCESSING_FALLBACK="bedrock,openai"

# AWS Configuration (for Bedrock)
export AWS_REGION="us-east-1"
export AWS_PROFILE="ai-processing"

# Performance Settings
export AI_CACHE_ENABLED="true"
export AI_CACHE_TTL_SECONDS="3600"
export AI_MAX_RETRIES="3"

# Model Defaults
export AI_DEFAULT_MODEL="claude-3-sonnet"
export AI_MAX_TOKENS="2000"
export AI_TEMPERATURE="0.7"
```

---

## 💻 **Usage Examples**

### **Basic Processing**
```python
from tidyllm.gateways import get_gateway
from tidyllm.gateways import AIRequest

# Get the gateway (corporate_llm must be initialized first)
ai_gateway = get_gateway("ai_processing")

# Simple AI processing request
request = AIRequest(
    prompt="Explain the SOLID principles in software design",
    model="claude-3-sonnet",
    max_tokens=1500,
    temperature=0.7,
    reason="Educational content generation"
)

response = ai_gateway.process(request)

if response.status == "success":
    print(f"Response: {response.data['response']}")
    print(f"Backend used: {response.metadata['backend_used']}")
    print(f"Cache hit: {response.metadata['cache_hit']}")
    print(f"Processing time: {response.metadata['processing_time_ms']}ms")
```

### **Advanced Processing with Backend Selection**
```python
# Force specific backend
request = AIRequest(
    prompt="Generate Python code for data processing",
    backend=AIBackend.BEDROCK,  # Force Bedrock
    model="claude-3-opus",
    max_tokens=2000,
    temperature=0.3,  # Lower temperature for code generation
    reason="Code generation for feature XYZ"
)

response = ai_gateway.process(request)
```

### **Batch Processing**
```python
# Process multiple requests efficiently
requests = [
    AIRequest(prompt=f"Summarize: {doc}", model="claude-3-sonnet")
    for doc in document_list
]

# Process with automatic batching and caching
responses = ai_gateway.process_batch(requests)

for i, response in enumerate(responses):
    if response.status == "success":
        print(f"Doc {i}: {response.data['response'][:100]}...")
    else:
        print(f"Doc {i} failed: {response.error}")
```

### **Streaming Responses**
```python
# For long-running processing with real-time updates
request = AIRequest(
    prompt="Write a comprehensive guide on microservices",
    model="claude-3-opus",
    max_tokens=4000,
    stream=True
)

# Stream response tokens
for chunk in ai_gateway.process_stream(request):
    print(chunk['content'], end='', flush=True)
```

---

## ⚡ **Performance Features**

### **Response Caching**
```python
# Automatic caching based on request hash
request1 = AIRequest(prompt="What is Python?")
response1 = ai_gateway.process(request1)  # Cache miss - calls backend

request2 = AIRequest(prompt="What is Python?")  # Same request
response2 = ai_gateway.process(request2)  # Cache hit - instant response

print(f"Cache hit: {response2.metadata['cache_hit']}")  # True
```

### **Retry Logic**
```python
# Automatic retries with exponential backoff
request = AIRequest(
    prompt="Complex analysis task",
    max_retries=5,           # Override default
    retry_delay_base=2.0,    # Override default
    timeout_seconds=60       # Longer timeout
)

response = ai_gateway.process(request)
print(f"Attempts made: {response.metadata['retry_attempts']}")
```

### **Backend Failover**
```python
# Automatic failover between backends
request = AIRequest(
    prompt="Critical analysis needed",
    backend=AIBackend.AUTO,
    fallback_backends=[AIBackend.BEDROCK, AIBackend.OPENAI]
)

response = ai_gateway.process(request)
print(f"Final backend: {response.metadata['backend_used']}")
print(f"Failover occurred: {response.metadata['failover_used']}")
```

---

## 📊 **Monitoring & Metrics**

### **Key Metrics Tracked**
- Request count by backend
- Average response time by model
- Cache hit ratio
- Error rates and retry attempts
- Token usage and costs
- Backend availability

### **Performance Dashboard**
```python
# Get performance metrics
metrics = ai_gateway.get_metrics()

print(f"Cache hit ratio: {metrics['cache_hit_ratio']:.2%}")
print(f"Average response time: {metrics['avg_response_time_ms']}ms")
print(f"Backend distribution: {metrics['backend_usage']}")
print(f"Error rate: {metrics['error_rate']:.2%}")
```

---

## 🔒 **Integration with CorporateLLMGateway**

### **Dependency Relationship**
```python
# AIProcessingGateway automatically uses CorporateLLMGateway
class AIProcessingGateway(BaseGateway):
    def __init__(self, **config):
        # Automatically gets corporate_llm gateway
        self.corporate_llm = get_gateway("corporate_llm")
        if not self.corporate_llm:
            raise ValueError("corporate_llm gateway required")
```

### **Cost and Compliance Pass-Through**
```python
# AI Processing forwards compliance data
request = AIRequest(
    prompt="Generate report",
    reason="Monthly compliance report generation"  # Passed to CorporateLLMGateway
)

response = ai_gateway.process(request)

# Audit trail includes both layers
print(f"Corporate audit ID: {response.metadata['corporate_audit_id']}")
print(f"Processing metrics: {response.metadata['processing_metrics']}")
```

---

## ⚠️ **Common Pitfalls & Solutions**

### **Pitfall 1: Missing CorporateLLMGateway**
```python
# ❌ WRONG - AIProcessingGateway requires CorporateLLMGateway
registry = init_gateways({"ai_processing": {}})  # Will fail

# ✅ CORRECT - Initialize corporate_llm first
registry = init_gateways({
    "corporate_llm": {"budget_limit_daily_usd": 500},
    "ai_processing": {"backend": "auto"}
})
```

### **Pitfall 2: Backend Configuration Mismatch**
```python
# ❌ WRONG - Backend not available in corporate gateway
corporate_config = {"available_providers": ["claude"]}  # Only Claude
ai_config = {"backend": "openai"}  # Requests OpenAI

# ✅ CORRECT - Align backend with corporate providers
corporate_config = {"available_providers": ["claude", "openai-corporate"]}
ai_config = {"backend": "auto"}  # Let it choose appropriate
```

### **Pitfall 3: Ignoring Cache Implications**
```python
# ❌ WRONG - Not considering cache for sensitive data
request = AIRequest(prompt=f"Analyze PII data: {sensitive_data}")

# ✅ CORRECT - Disable cache for sensitive requests
request = AIRequest(
    prompt=f"Analyze sensitive data",
    disable_cache=True,  # Don't cache sensitive responses
    reason="PII analysis - no caching allowed"
)
```

---

## 🔧 **Advanced Configuration**

### **Custom Backend Configuration**
```python
# Advanced backend-specific settings
config = {
    "ai_processing": {
        "backend": "bedrock",
        "bedrock_config": {
            "region": "us-west-2",
            "model_ids": {
                "claude-3-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
                "claude-3-opus": "anthropic.claude-3-opus-20240229-v1:0"
            },
            "inference_config": {
                "max_tokens": 4096,
                "temperature": 0.7,
                "top_p": 0.9,
                "stop_sequences": ["</analysis>"]
            }
        }
    }
}
```

### **Model-Specific Optimization**
```python
# Different settings per model
model_configs = {
    "claude-3-opus": {
        "max_tokens": 4096,
        "temperature": 0.8,
        "cache_ttl": 7200  # 2 hours
    },
    "claude-3-sonnet": {
        "max_tokens": 2048,
        "temperature": 0.7,
        "cache_ttl": 3600  # 1 hour
    }
}
```

---

## 📚 **Related Documentation**

### **Must Read First**
- [Guidance on CorporateLLMGateway - README.md](./Guidance%20on%20CorporateLLMGateway%20-%20README.md) - REQUIRED DEPENDENCY
- [IMPORTANT-CONSTRAINTS-FOR-THIS-CODEBASE.md](../2025-09-05/IMPORTANT-CONSTRAINTS-FOR-THIS-CODEBASE.md)

### **Related Gateway Documentation**
- [Guidance on WorkflowOptimizerGateway - README.md](./Guidance%20on%20WorkflowOptimizerGateway%20-%20README.md)
- [GATEWAY_ARCHITECTURE_OVERVIEW.md](../2025-09-04/GATEWAY_ARCHITECTURE_OVERVIEW.md)

### **Implementation Files**
- Gateway: `tidyllm/gateways/ai_processing_gateway.py`
- Tests: `tests/test_ai_processing_gateway.py`
- Config: `tidyllm/config/ai_backends.yaml`

---

## 🎯 **Quick Reference Card**

### **Service Name**: `ai_processing`
### **Dependency Level**: 2 (Requires: corporate_llm)
### **Required By**: workflow_optimizer
### **Key Responsibility**: Multi-model AI processing with caching
### **Configuration Required**: Yes (backend, caching, models)
### **Cache Enabled**: Yes (configurable TTL)
### **Retries**: Yes (exponential backoff)

---

## 🚨 **Final Checklist for AI Agents**

Before using AIProcessingGateway:
- [ ] Confirmed CorporateLLMGateway is configured and working
- [ ] Selected appropriate backend (auto recommended)
- [ ] Configured caching settings (enabled by default)
- [ ] Set retry parameters (3 retries default)
- [ ] Verified model availability in corporate gateway
- [ ] Tested failover scenarios
- [ ] Reviewed cache policies for sensitive data

**Remember**: This gateway builds on CorporateLLMGateway. If corporate gateway fails, this gateway fails. Always ensure the dependency chain is healthy.

---

**Document Location**: `/docs/2025-09-06/Guidance on AIProcessingGateway - README.md`  
**Last Updated**: 2025-09-06  
**Status**: Official Gateway Documentation