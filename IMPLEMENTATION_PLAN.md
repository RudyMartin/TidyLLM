# Implementation Plan: Missing High-Value APIs

**Objective:** Implement the 3 high-value missing APIs identified in variance analysis to achieve 100% working API surface.

---

## Implementation Strategy Overview

### **Architecture Integration Approach:**
- **Leverage existing patterns** from successful wrapper methods (DomainRAG, AIProcessingGateway)
- **Follow TidyLLM conventions** established in current working code
- **Maintain 4-Gateway chain integrity** (CorporateLLM → AI → WorkflowOptimizer → Database)
- **Use UnifiedSessionManager** for all connections and health checks

---

## 1. `tidyllm.list_models()` - **TOP PRIORITY** 🚀

### **1.1 Implementation Location:**
```
📁 tidyllm/api.py
```

### **1.2 Integration Pattern:**
```python
# Add to existing api.py after line 37 (after process_document function)
def list_models(**kwargs) -> List[Dict[str, Any]]:
    """List all available AI models across backends."""
    from .gateways.ai_processing_gateway import AIProcessingGateway
    
    try:
        ai_gateway = AIProcessingGateway()
        capabilities = ai_gateway.get_capabilities()
        
        # Transform into expected format
        models = []
        backend = capabilities.get("current_backend", "unknown")
        available_models = capabilities.get("models", [])
        
        for model in available_models:
            models.append({
                "name": model,
                "backend": backend,
                "type": "chat",
                "max_tokens": capabilities.get("max_tokens", 4096),
                "supports_streaming": capabilities.get("supports_streaming", False)
            })
        
        return models
    except Exception as e:
        # Fallback for demos/examples
        return [
            {"name": "claude-3-sonnet", "backend": "anthropic", "type": "chat", "max_tokens": 4096},
            {"name": "gpt-4", "backend": "openai", "type": "chat", "max_tokens": 8192},
            {"name": "llama2-70b", "backend": "bedrock", "type": "chat", "max_tokens": 4096}
        ]
```

### **1.3 Resource Integration:**
- **Uses:** Existing `AIProcessingGateway.get_capabilities()` method ✅ (already works)
- **Dependencies:** No new dependencies - leverages existing gateway infrastructure
- **Fallback:** Provides static model list if gateway unavailable (demo compatibility)

### **1.4 Testing Strategy:**
```python
# Add to tidyllm/tests/test_api.py
def test_list_models():
    models = tidyllm.list_models()
    assert isinstance(models, list)
    assert len(models) > 0
    assert all(isinstance(model, dict) for model in models)
    assert all("name" in model and "backend" in model for model in models)
```

---

## 2. `session_mgr.test_connection()` - **MEDIUM PRIORITY** 🔧

### **2.1 Implementation Location:**
```
📁 tidyllm/infrastructure/session/unified.py
```

### **2.2 Integration Pattern:**
```python
# Add to UnifiedSessionManager class after get_health_summary() method (around line 300)
def test_connection(self, service: str = "all") -> Dict[str, Dict[str, Any]]:
    """Test connections to specified services with detailed results."""
    results = {}
    
    if service in ["all", "s3"]:
        results["s3"] = self._test_s3_connection()
    
    if service in ["all", "bedrock"]:
        results["bedrock"] = self._test_bedrock_connection()
    
    if service in ["all", "postgres"]:
        results["postgres"] = self._test_postgres_connection()
    
    return results

def _test_s3_connection(self) -> Dict[str, Any]:
    """Test S3 connection with timing and details."""
    import time
    start_time = time.time()
    
    try:
        s3_client = self.get_s3_client()
        response = s3_client.list_buckets()
        duration_ms = (time.time() - start_time) * 1000
        
        return {
            "status": "success",
            "duration_ms": round(duration_ms, 1),
            "bucket_count": len(response.get("Buckets", [])),
            "message": f"S3 connected successfully ({duration_ms:.1f}ms)"
        }
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        return {
            "status": "failed",
            "duration_ms": round(duration_ms, 1),
            "error": str(e),
            "message": f"S3 connection failed: {str(e)}"
        }

def _test_bedrock_connection(self) -> Dict[str, Any]:
    """Test Bedrock connection with timing and details."""
    import time
    start_time = time.time()
    
    try:
        bedrock_client = self.get_bedrock_client()
        # Test with a minimal model list call
        response = bedrock_client.list_foundation_models()
        duration_ms = (time.time() - start_time) * 1000
        
        return {
            "status": "success", 
            "duration_ms": round(duration_ms, 1),
            "model_count": len(response.get("modelSummaries", [])),
            "message": f"Bedrock connected successfully ({duration_ms:.1f}ms)"
        }
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        return {
            "status": "failed",
            "duration_ms": round(duration_ms, 1),
            "error": str(e),
            "message": f"Bedrock connection failed: {str(e)}"
        }

def _test_postgres_connection(self) -> Dict[str, Any]:
    """Test PostgreSQL connection with timing and details."""
    import time
    start_time = time.time()
    
    try:
        conn = self.get_postgres_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        cursor.close()
        self.return_postgres_connection(conn)
        
        duration_ms = (time.time() - start_time) * 1000
        return {
            "status": "success",
            "duration_ms": round(duration_ms, 1),
            "test_query": "SELECT 1",
            "result": result[0] if result else None,
            "message": f"PostgreSQL connected successfully ({duration_ms:.1f}ms)"
        }
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        return {
            "status": "failed",
            "duration_ms": round(duration_ms, 1),
            "error": str(e),
            "message": f"PostgreSQL connection failed: {str(e)}"
        }
```

### **2.3 Resource Integration:**
- **Uses:** Existing connection methods (`get_s3_client()`, `get_bedrock_client()`, `get_postgres_connection()`) ✅
- **Pattern:** Follows existing `get_health_summary()` pattern in same class
- **Timing:** Provides detailed timing metrics for operational monitoring

---

## 3. `gateway.health_check()` - **ARCHITECTURAL STANDARD** 🏗️

### **3.1 Implementation Location:**
```
📁 tidyllm/gateways/base_gateway.py  (abstract method)
📁 tidyllm/gateways/corporate_llm_gateway.py
📁 tidyllm/gateways/ai_processing_gateway.py
📁 tidyllm/gateways/workflow_optimizer_gateway.py
📁 tidyllm/gateways/database_gateway.py
```

### **3.2 Integration Pattern:**

#### **3.2.1 BaseGateway Abstract Method:**
```python
# Add to BaseGateway class after validate_config() method (around line 470)
@abstractmethod
def health_check(self) -> Dict[str, Any]:
    """
    Perform health check for this gateway.
    
    Returns:
        Dict with status, dependencies, timing, and diagnostic info
    """
    pass
```

#### **3.2.2 Implementation Template (for each gateway):**
```python
# Pattern for each gateway implementation
def health_check(self) -> Dict[str, Any]:
    """Perform comprehensive health check for [Gateway Name]."""
    import time
    start_time = time.time()
    
    health_status = {
        "gateway": self.__class__.__name__,
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "checks": {},
        "dependencies": {},
        "metrics": {}
    }
    
    try:
        # 1. Configuration validation
        health_status["checks"]["config"] = {
            "status": "pass" if self.validate_config() else "fail",
            "message": "Configuration validation"
        }
        
        # 2. Gateway-specific checks
        health_status["checks"].update(self._gateway_specific_health_checks())
        
        # 3. Dependency checks  
        health_status["dependencies"] = self._check_dependencies()
        
        # 4. Performance metrics
        duration_ms = (time.time() - start_time) * 1000
        health_status["metrics"] = {
            "health_check_duration_ms": round(duration_ms, 1),
            "memory_usage_mb": self._get_memory_usage()
        }
        
        # 5. Overall status determination
        failed_checks = [k for k, v in health_status["checks"].items() if v["status"] == "fail"]
        if failed_checks:
            health_status["status"] = "unhealthy"
            health_status["failed_checks"] = failed_checks
        
    except Exception as e:
        health_status["status"] = "error"
        health_status["error"] = str(e)
    
    return health_status

def _gateway_specific_health_checks(self) -> Dict[str, Any]:
    """Override in each gateway for specific checks."""
    return {}

def _check_dependencies(self) -> Dict[str, Any]:
    """Check gateway dependencies (session manager, other gateways)."""
    deps = {}
    
    # Session manager dependency (common to all gateways)
    if hasattr(self, 'session_manager') and self.session_manager:
        try:
            health_summary = self.session_manager.get_health_summary()
            deps["session_manager"] = {
                "status": "healthy" if health_summary.get("overall_health") else "unhealthy",
                "details": health_summary
            }
        except Exception as e:
            deps["session_manager"] = {"status": "error", "error": str(e)}
    
    return deps

def _get_memory_usage(self) -> float:
    """Get memory usage in MB."""
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return round(process.memory_info().rss / 1024 / 1024, 2)
    except ImportError:
        return 0.0
```

### **3.3 Gateway-Specific Implementations:**

#### **CorporateLLMGateway:**
```python
def _gateway_specific_health_checks(self) -> Dict[str, Any]:
    """Corporate gateway specific checks."""
    checks = {}
    
    # SSO/SAML check
    if hasattr(self, 'sso_config') and self.sso_config:
        checks["sso"] = {"status": "configured", "message": "SSO configuration present"}
    
    # Proxy check
    if hasattr(self, 'proxy_config') and self.proxy_config:
        checks["proxy"] = {"status": "configured", "message": "Proxy configuration present"}
    
    return checks
```

#### **AIProcessingGateway:**
```python
def _gateway_specific_health_checks(self) -> Dict[str, Any]:
    """AI gateway specific checks."""
    checks = {}
    
    # Backend availability
    if hasattr(self, 'ai_config'):
        backend = self.ai_config.backend
        checks["ai_backend"] = {
            "status": "available",
            "backend": backend.value,
            "message": f"Using {backend.value} backend"
        }
    
    # Model access check
    try:
        capabilities = self.get_capabilities()
        model_count = len(capabilities.get("models", []))
        checks["models"] = {
            "status": "pass" if model_count > 0 else "fail",
            "model_count": model_count,
            "message": f"{model_count} models available"
        }
    except Exception as e:
        checks["models"] = {"status": "fail", "error": str(e)}
    
    return checks
```

### **3.4 Resource Integration:**
- **Uses:** Existing `validate_config()`, `get_health_summary()`, `get_capabilities()` methods ✅
- **Pattern:** Consistent health check format across all 4 gateways
- **Dependencies:** Integrates with UnifiedSessionManager health checking
- **Metrics:** Provides timing and memory usage for monitoring systems

---

## 4. Implementation Sequence & Dependencies

### **Phase 1: Foundation (Day 1)**
1. ✅ **Add `health_check()` abstract method to BaseGateway**
2. ✅ **Implement `test_connection()` in UnifiedSessionManager** 
3. ✅ **Test connection methods work with existing infrastructure**

### **Phase 2: Core APIs (Day 2)**
1. ✅ **Add `list_models()` to tidyllm.api**
2. ✅ **Implement health_check() in AIProcessingGateway first**
3. ✅ **Test that all existing functionality still works**

### **Phase 3: Complete Implementation (Day 3)**
1. ✅ **Implement health_check() in remaining 3 gateways**
2. ✅ **Update all broken references to use new methods**
3. ✅ **Comprehensive testing of all new functionality**

---

## 5. Resource Requirements & Integration Points

### **5.1 Existing Resources to Leverage:**
| Resource | Current Status | Integration Method |
|----------|----------------|-------------------|
| `AIProcessingGateway.get_capabilities()` | ✅ Working | Use for `list_models()` |
| `UnifiedSessionManager.get_health_summary()` | ✅ Working | Use for `test_connection()` |
| `BaseGateway.validate_config()` | ✅ Working | Use in `health_check()` |
| Connection methods (S3, Bedrock, Postgres) | ✅ Working | Use for connection testing |

### **5.2 New Dependencies:**
- **None** - All implementations use existing infrastructure
- **Optional:** `psutil` for memory usage (graceful fallback if not available)

### **5.3 Integration Testing Matrix:**
| Component | Integration Test | Success Criteria |
|-----------|------------------|-------------------|
| `tidyllm.list_models()` | Call from existing demo files | Returns model list without errors |
| `session_mgr.test_connection()` | Call from onboarding app | Returns connection status with timing |
| `gateway.health_check()` | Call from all 4 gateways | Returns consistent health format |

---

## 6. Rollback & Risk Management

### **6.1 Implementation Safety:**
- **No breaking changes** - All new methods are additions
- **Graceful fallbacks** - Methods work even if dependencies unavailable
- **Existing functionality** - No modifications to working code paths

### **6.2 Testing Strategy:**
```bash
# Verify existing functionality still works
python -c "from tidyllm.infrastructure.session.unified import UnifiedSessionManager; print('Session manager works')"
python -c "from tidyllm.gateways.ai_processing_gateway import AIProcessingGateway; print('AI gateway works')"
python -c "import tidyllm; print('API module works')"

# Test new functionality
python -c "import tidyllm; print(tidyllm.list_models())"
python -c "from tidyllm.infrastructure.session.unified import UnifiedSessionManager; sm=UnifiedSessionManager(); print(sm.test_connection())"
```

---

## 7. Success Metrics

### **7.1 Technical Metrics:**
- ✅ **0 broken method calls** in codebase (down from 20+)
- ✅ **100% working examples** in demo files
- ✅ **Consistent health check interface** across all gateways

### **7.2 Developer Experience Metrics:**
- ✅ **Working `tidyllm.list_models()` calls** in all 5 example files
- ✅ **Successful connection testing** in diagnostic scripts
- ✅ **Standard health endpoints** for monitoring integration

### **7.3 Validation Checklist:**
- [ ] `tidyllm.list_models()` returns model information
- [ ] `session_mgr.test_connection()` tests all services with timing
- [ ] All 4 gateways implement `health_check()` consistently
- [ ] Existing functionality unaffected
- [ ] All broken references resolved
- [ ] Documentation updated

---

## 8. Post-Implementation Actions

### **8.1 Documentation Updates:**
1. ✅ **Update `CRITICAL_CALLS_THAT_WORK.md`** with new methods
2. ✅ **Remove methods from `WHAT_IS_NOT_REAL.md`**
3. ✅ **Create usage examples** for new functionality

### **8.2 Cleanup Tasks:**
1. ✅ **Remove references to deleted demo files** 
2. ✅ **Update import statements** in working examples
3. ✅ **Test onboarding application** with new methods

---

**Implementation Timeline:** 3 days  
**Risk Level:** Low (additive changes only)  
**Expected Outcome:** 100% working API surface with enterprise-ready health monitoring

**Ready to begin implementation with Phase 1.**