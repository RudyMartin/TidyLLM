# TidyLLM Architecture - GitHub Safe Diagrams
**Repository:** github.com/rudymartin/tidyllm  
**Purpose:** GitHub-compatible diagrams for team collaboration  
**Date:** 2025-09-03

---

## Diagram 1: System Overview

```mermaid
graph TD
    A[Your DSPy Application] --> B[UnifiedDSPyWrapper]
    
    B --> C[Retry Logic]
    B --> D[Caching]
    B --> E[Validation]
    B --> F[Metrics]
    
    B --> G[Gateway Backend]
    B --> H[Bedrock Backend]  
    B --> I[Direct Backend]
    B --> J[Mock Backend]
    
    G --> K[MLFlow Gateway]
    H --> L[AWS Bedrock]
    I --> M[Claude API]
    I --> N[GPT API]
    
    K --> M
    K --> N
    K --> L
```

**Benefits:**
- Single entry point replaces 5 competing patterns
- Pluggable backends for different environments
- Optional features with zero overhead
- Enterprise governance built-in

---

## Diagram 2: Backend Selection

```mermaid
graph TD
    A[UnifiedDSPyWrapper] --> B{Auto-Detect Backend}
    
    B -->|Enterprise| C[Gateway Backend]
    B -->|AWS Environment| D[Bedrock Backend]
    B -->|Simple Setup| E[Direct Backend]
    B -->|Testing| F[Mock Backend]
    
    C --> G[MLFlow Gateway]
    C --> H[Audit Logging]
    C --> I[Cost Tracking]
    
    D --> J[AWS Bedrock]
    D --> K[Region Selection]
    
    E --> L[LiteLLM]
    E --> M[Multiple Providers]
    
    F --> N[Mock Responses]
    F --> O[Test Scenarios]
```

**Backend Logic:**
- Gateway: Enterprise environments with governance needs
- Bedrock: AWS-native deployments with multi-region support
- Direct: Simple setups with minimal overhead
- Mock: Development and testing environments

---

## Diagram 3: Request Flow

```mermaid
sequenceDiagram
    participant App as Application
    participant Wrap as UnifiedDSPyWrapper
    participant Back as Selected Backend
    participant Ext as External Service
    
    App->>Wrap: create_module(signature)
    Wrap->>Back: detect_backend()
    Back-->>Wrap: backend_instance
    
    App->>Wrap: module.predict(input)
    Wrap->>Wrap: apply_features(retry, cache, etc)
    Wrap->>Back: generate(prompt)
    Back->>Ext: API call
    Ext-->>Back: response
    Back-->>Wrap: processed_response
    Wrap-->>App: final_result
```

**Flow Steps:**
1. Application creates DSPy module through wrapper
2. Wrapper auto-detects appropriate backend
3. Prediction request goes through feature pipeline
4. Backend handles external service communication
5. Response flows back with all enhancements applied

---

## Diagram 4: Migration Strategy  

```mermaid
graph LR
    A[5 Competing Patterns] --> B[Analysis Phase]
    B --> C[Extract Best Features]
    C --> D[Build Unified Wrapper]
    D --> E[Test & Validate]
    E --> F[Deploy to Production]
    
    subgraph "Old Patterns"
        G[Enhanced Wrapper]
        H[Gateway Backend] 
        I[Bedrock Enhanced]
        J[Simple Wrapper]
        K[Dynamic Module]
    end
    
    subgraph "New Architecture"
        L[UnifiedDSPyWrapper]
        M[Feature System]
        N[Backend System]
    end
    
    G --> L
    H --> N
    I --> N
    J --> N
    K --> L
```

**Migration Benefits:**
- 80% code reduction (5000+ lines to 1000 lines)
- Single development path
- Zero disruption during transition
- All existing features preserved

---

## Code Examples

### Basic Usage
```python
from tidyllm import UnifiedDSPyWrapper

# Auto-detect best backend
wrapper = UnifiedDSPyWrapper()
module = wrapper.create_module("question -> answer")
result = module.predict(question="What is AI?")
```

### Enterprise Configuration
```python
from tidyllm import UnifiedDSPyWrapper, UnifiedConfig, BackendType

wrapper = UnifiedDSPyWrapper(
    UnifiedConfig(
        backend=BackendType.GATEWAY,
        enable_retry=True,
        enable_cache=True,
        enable_validation=True
    )
)
```

### AWS Optimized
```python
wrapper = UnifiedDSPyWrapper(
    UnifiedConfig(backend=BackendType.BEDROCK)
)
```

### Development/Testing
```python
wrapper = UnifiedDSPyWrapper(
    UnifiedConfig(backend=BackendType.MOCK)
)
```

---

## Implementation Status

**Foundation Complete:**
- [x] Unified wrapper architecture designed
- [x] Backend system specification complete  
- [x] Feature composition system planned
- [x] Migration strategy documented

**In Development:**
- [ ] UnifiedDSPyWrapper implementation
- [ ] Backend implementations (Gateway, Bedrock, Direct, Mock)
- [ ] Feature decorators (Retry, Cache, Validation, Metrics)
- [ ] Integration testing and validation

**Next Steps:**
1. Implement core UnifiedDSPyWrapper class
2. Build pluggable backend system
3. Add feature composition layer
4. Create comprehensive tests
5. Document migration process
6. Deploy to production

---

This simplified architecture eliminates the 5 competing DSPy patterns while providing all necessary functionality through a clean, maintainable design.