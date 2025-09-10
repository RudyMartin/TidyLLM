# Guidance on CorporateLLMGateway - README

**Document Version**: 1.0  
**Created**: 2025-09-06  
**Status**: Official Gateway Guidance  
**Priority**: MANDATORY READING FOR ALL AI AGENTS

---

## 🔴 CRITICAL: This is Gateway #1 - The Foundation Layer

**AI AGENTS**: This gateway is the FOUNDATION of the entire system. It has NO dependencies and ALL other gateways depend on it. You MUST understand this gateway before working with ANY other gateway.

---

## 📋 **Executive Summary**

The **CorporateLLMGateway** is the foundational gateway in TidyLLM's three-tier gateway architecture. It provides enterprise-grade access control, cost tracking, and compliance for all LLM operations. This is the ONLY gateway that directly interfaces with external LLM providers, ensuring complete corporate control over AI operations.

### **Position in Hierarchy**
```
Level 1: CorporateLLMGateway (THIS GATEWAY)
         ├── No dependencies (foundation layer)
         ├── Direct MLFlow integration
         └── All other gateways depend on this
              ↓
Level 2: AIProcessingGateway 
         └── Requires: CorporateLLMGateway
              ↓
Level 3: WorkflowOptimizerGateway
         └── Requires: AIProcessingGateway + CorporateLLMGateway
```

---

## 🎯 **Core Purpose & Responsibilities**

### **Primary Responsibilities**
1. **Access Control**: ZERO direct external LLM access - everything goes through corporate infrastructure
2. **Cost Management**: Track and enforce budget limits per request/user/day
3. **Compliance**: Full audit trails with mandatory reason fields
4. **Provider Management**: IT-controlled whitelist of models and providers

### **What It Does**
- Routes ALL LLM requests through MLFlow Gateway
- Enforces corporate policies on model usage
- Tracks costs and prevents budget overruns
- Maintains complete audit trails for compliance
- Provides fallback mechanisms for high availability

### **What It Does NOT Do**
- Does NOT process complex AI workflows (that's AIProcessingGateway)
- Does NOT optimize workflows (that's WorkflowOptimizerGateway)
- Does NOT directly call external APIs (uses MLFlow Gateway)
- Does NOT make decisions about which model to use (follows configuration)

---

## 🏗️ **Technical Architecture**

### **Location**
`tidyllm/gateways/corporate_llm_gateway.py`

### **Dependencies**
```python
# NONE - This is the foundation layer
# Only depends on:
- mlflow.gateway (for LLM routing)
- Standard Python libraries
- NO other gateways
```

### **Key Classes**
```python
class CorporateLLMGateway(BaseGateway):
    """Corporate-controlled LLM access with compliance."""
    
class LLMRequest:
    """Request structure for LLM operations."""
    
class LLMProvider(Enum):
    """Available LLM providers (IT-controlled)."""
    CLAUDE = "claude"
    OPENAI = "openai-corporate"
    AZURE_GPT = "azure-gpt"
```

---

## 🔧 **Configuration & Setup**

### **Standard Configuration**
```python
from tidyllm.gateways import init_gateways

registry = init_gateways({
    "corporate_llm": {
        # Provider Configuration
        "available_providers": ["claude", "openai-corporate"],
        "default_provider": "claude",
        
        # Model Whitelist
        "allowed_models": {
            "claude": ["claude-3-5-sonnet", "claude-3-opus"],
            "openai-corporate": ["gpt-4o", "gpt-4-turbo"]
        },
        
        # Cost Controls
        "max_tokens_per_request": 4096,
        "max_cost_per_request_usd": 1.00,
        "budget_limit_daily_usd": 500.00,
        
        # Compliance
        "require_reason": True,
        "audit_level": "full",
        "track_costs": True,
        
        # MLFlow Configuration
        "mlflow_tracking_uri": "http://mlflow-server:5000",
        "mlflow_gateway_uri": "http://mlflow-gateway:8080"
    }
})
```

### **Environment Variables**
```bash
# MLFlow Configuration
export MLFLOW_TRACKING_URI="http://mlflow-server:5000"
export MLFLOW_GATEWAY_URI="http://mlflow-gateway:8080"

# Cost Limits
export LLM_MAX_COST_PER_REQUEST="1.00"
export LLM_DAILY_BUDGET_USD="500.00"

# Provider Settings
export LLM_DEFAULT_PROVIDER="claude"
export LLM_FALLBACK_PROVIDER="openai-corporate"
```

---

## 💻 **Usage Examples**

### **Basic Usage**
```python
from tidyllm.gateways import get_gateway
from tidyllm.gateways import LLMRequest

# Get the gateway
gateway = get_gateway("corporate_llm")

# Create request
request = LLMRequest(
    prompt="Explain the concept of dependency injection",
    provider="claude",
    model="claude-3-5-sonnet",
    max_tokens=500,
    temperature=0.7,
    reason="Documentation generation for design patterns guide"  # REQUIRED
)

# Process request
response = gateway.process(request)

if response.status == "success":
    print(f"Response: {response.data['response']}")
    print(f"Cost: ${response.metadata['cost_usd']:.4f}")
    print(f"Tokens: {response.metadata['token_count']}")
```

### **With Cost Controls**
```python
# Check remaining budget before expensive operation
budget_check = gateway.check_daily_budget()
if budget_check['remaining_usd'] < 10.00:
    print("Warning: Low daily budget remaining")

# Process with strict cost limit
request = LLMRequest(
    prompt="Generate comprehensive test suite",
    max_cost_usd=0.50,  # Strict limit for this request
    reason="Automated test generation for PR #123"
)

response = gateway.process(request)
```

### **With Fallback Handling**
```python
# Primary provider with automatic fallback
request = LLMRequest(
    prompt="Critical analysis needed",
    provider="claude",  # Primary
    fallback_providers=["openai-corporate", "azure-gpt"],  # Fallbacks
    reason="Production incident analysis"
)

response = gateway.process(request)
print(f"Used provider: {response.metadata['provider_used']}")
```

---

## 🔒 **Security & Compliance**

### **Access Control**
- NO direct external API access
- All requests routed through MLFlow Gateway
- IT-managed provider whitelist
- User-level access controls

### **Audit Requirements**
```python
# Every request MUST include:
{
    "reason": "Business justification",  # REQUIRED
    "user_id": "authenticated_user",    # Auto-captured
    "timestamp": "2025-09-06T10:30:00Z", # Auto-added
    "request_id": "uuid-v4",            # Auto-generated
    "cost_usd": 0.0234,                 # Auto-calculated
    "tokens_used": 567                   # Auto-tracked
}
```

### **Cost Protection**
- Per-request cost limits
- Daily budget caps
- User-level quotas
- Real-time cost tracking
- Automatic cutoff at limits

---

## ⚠️ **Common Pitfalls & Solutions**

### **Pitfall 1: Direct External API Calls**
```python
# ❌ WRONG - Never do this
import openai
response = openai.ChatCompletion.create(...)  # FORBIDDEN

# ✅ CORRECT - Always use gateway
gateway = get_gateway("corporate_llm")
response = gateway.process(LLMRequest(...))
```

### **Pitfall 2: Missing Reason Field**
```python
# ❌ WRONG - No audit trail
request = LLMRequest(prompt="Generate code")  # Missing reason

# ✅ CORRECT - Always provide reason
request = LLMRequest(
    prompt="Generate code",
    reason="Feature implementation for JIRA-456"
)
```

### **Pitfall 3: Ignoring Cost Limits**
```python
# ❌ WRONG - No cost consideration
request = LLMRequest(
    prompt="Analyze entire codebase",
    max_tokens=100000  # Too expensive!
)

# ✅ CORRECT - Set appropriate limits
request = LLMRequest(
    prompt="Analyze critical functions",
    max_tokens=4096,
    max_cost_usd=1.00
)
```

---

## 📊 **Monitoring & Metrics**

### **Key Metrics Tracked**
- Total requests per provider
- Average cost per request
- Daily/monthly spend
- Error rates by provider
- Fallback usage frequency
- Response times

### **Dashboard Queries**
```sql
-- Daily usage summary
SELECT 
    DATE(timestamp) as date,
    provider,
    COUNT(*) as requests,
    SUM(cost_usd) as total_cost,
    AVG(tokens_used) as avg_tokens
FROM llm_audit_log
WHERE timestamp >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY DATE(timestamp), provider;
```

---

## 🔗 **Integration Points**

### **Used By**
- **AIProcessingGateway**: For all LLM operations
- **WorkflowOptimizerGateway**: For workflow analysis
- **All other components**: That need LLM access

### **Integrates With**
- **MLFlow Gateway**: For LLM routing
- **PostgreSQL**: For audit logging
- **Cost Tracking System**: For budget management
- **Authentication System**: For user validation

---

## 📚 **Related Documentation**

### **Must Read**
- [IMPORTANT-CONSTRAINTS-FOR-THIS-CODEBASE.md](../2025-09-05/IMPORTANT-CONSTRAINTS-FOR-THIS-CODEBASE.md)
- [GATEWAY_ARCHITECTURE_OVERVIEW.md](../2025-09-04/GATEWAY_ARCHITECTURE_OVERVIEW.md)
- [Guidance on AIProcessingGateway - README.md](./Guidance%20on%20AIProcessingGateway%20-%20README.md)
- [Guidance on WorkflowOptimizerGateway - README.md](./Guidance%20on%20WorkflowOptimizerGateway%20-%20README.md)

### **Implementation Files**
- Gateway: `tidyllm/gateways/corporate_llm_gateway.py`
- Tests: `tests/test_corporate_llm_gateway.py`
- Config: `tidyllm/config/llm_providers.yaml`

---

## 🎯 **Quick Reference Card**

### **Service Name**: `corporate_llm`
### **Dependency Level**: 1 (Foundation - No dependencies)
### **Required By**: All other gateways
### **Key Responsibility**: Corporate LLM access control
### **Configuration Required**: Yes (providers, models, costs)
### **Audit Trail**: MANDATORY
### **Direct External Access**: FORBIDDEN

---

## 🚨 **Final Checklist for AI Agents**

Before using CorporateLLMGateway:
- [ ] Configured MLFlow Gateway connection
- [ ] Set cost limits and budgets
- [ ] Defined allowed providers and models
- [ ] Enabled audit logging
- [ ] Tested fallback mechanisms
- [ ] Verified reason field is included
- [ ] Confirmed NO direct API calls in code

**Remember**: This gateway is the FOUNDATION. If it fails, everything fails. Handle with care.

---

**Document Location**: `/docs/2025-09-06/Guidance on CorporateLLMGateway - README.md`  
**Last Updated**: 2025-09-06  
**Status**: Official Gateway Documentation