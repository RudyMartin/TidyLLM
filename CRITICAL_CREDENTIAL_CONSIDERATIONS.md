# 🚨 CRITICAL CREDENTIAL CONSIDERATIONS FOR AI AGENTS

**Warning for AI Agents working on TidyLLM**  
**Date**: 2025-09-09  
**Status**: ✅ **WORKING** - Maintain this state!

---

## ⚠️ **ATTENTION: AI AGENTS & FUTURE DEVELOPERS**

This document contains **CRITICAL INFORMATION** about the credential management system that **MUST BE UNDERSTOOD** before making any changes to AWS-related code in TidyLLM.

### **🚨 READ THIS FIRST**

The credential management system was **JUST FIXED** after causing system-wide "partial results" issues. **DO NOT BREAK IT AGAIN.**

---

## 🤖 **FOR AI AGENTS: WHAT YOU NEED TO KNOW**

### **✅ SAFE PATTERNS TO FOLLOW**

#### **1. ALWAYS Use UnifiedSessionManager**
```python
# ✅ CORRECT - Use this pattern
if self.session_manager:
    bedrock_client = self.session_manager.get_bedrock_client()
    s3_client = self.session_manager.get_s3_client()
    logger.info("Component: Using UnifiedSessionManager for AWS access")
else:
    # Fallback only if session manager unavailable
    import boto3
    bedrock_client = boto3.client('bedrock-runtime')
    s3_client = boto3.client('s3')
    logger.warning("Component: Using direct boto3 (no session manager)")
```

#### **2. ALWAYS Import with Error Handling**
```python
# ✅ CORRECT - Safe import pattern
try:
    from tidyllm.infrastructure.session.unified import UnifiedSessionManager
    UNIFIED_SESSION_AVAILABLE = True
except ImportError:
    UNIFIED_SESSION_AVAILABLE = False

# ✅ CORRECT - Safe initialization
if UNIFIED_SESSION_AVAILABLE:
    try:
        self.session_manager = UnifiedSessionManager()
        logger.info("Component: UnifiedSessionManager integrated")
    except Exception as e:
        logger.warning(f"Component: Failed to initialize UnifiedSessionManager: {e}")
        self.session_manager = None
else:
    self.session_manager = None
```

#### **3. ALWAYS Inherit from BaseGateway for Gateways**
```python
# ✅ CORRECT - Gateway inherits session manager injection
class MyNewGateway(BaseGateway):
    def __init__(self):
        super().__init__()  # Gets session_manager automatically from GatewayRegistry
        # Use self.session_manager in AWS operations
```

---

## ❌ **DANGEROUS PATTERNS TO AVOID**

### **🚫 NEVER Do Direct boto3 Bypass**
```python
# ❌ WRONG - This breaks credential flow
import boto3
bedrock = boto3.client('bedrock-runtime')  # CREDENTIAL DROP!
s3 = boto3.client('s3')  # CREDENTIAL DROP!
```

### **🚫 NEVER Skip Session Manager Check**
```python
# ❌ WRONG - No session manager integration
def process_data(self):
    client = boto3.client('bedrock-runtime')  # BREAKS CREDENTIAL FLOW
    return client.invoke_model(...)
```

### **🚫 NEVER Import Without Error Handling**
```python
# ❌ WRONG - Will crash if UnifiedSessionManager unavailable
from tidyllm.infrastructure.session.unified import UnifiedSessionManager
self.session_manager = UnifiedSessionManager()  # WILL CRASH
```

---

## 🎯 **AI AGENT SPECIFIC GUIDANCE**

### **When Modifying Existing Code**:
1. **SEARCH for boto3.client()** - Replace with session manager pattern
2. **CHECK for session manager usage** - Ensure components use it
3. **VERIFY gateway inheritance** - New gateways must inherit from BaseGateway
4. **TEST credential flow** - Run `python test_credential_flow.py`

### **When Creating New Components**:
1. **START with session manager integration** - Use the safe patterns above
2. **FOLLOW existing patterns** - Look at fixed components as examples
3. **ADD proper logging** - Show whether using session manager or fallback
4. **INCLUDE error handling** - Don't crash if session manager unavailable

### **When Debugging Issues**:
1. **CHECK session manager injection** - Is `self.session_manager` available?
2. **VERIFY credential source** - Environment? Settings file? IAM role?
3. **LOOK for direct boto3 calls** - These cause credential drops
4. **TEST with fresh session** - Credentials might not persist across processes

---

## 📋 **VALIDATION CHECKLIST FOR AI AGENTS**

Before suggesting or implementing changes:

### **✅ Code Review Questions**:
- [ ] Does this component use `self.session_manager.get_*_client()` instead of `boto3.client()`?
- [ ] Is there proper fallback handling if UnifiedSessionManager is unavailable?
- [ ] Will this change affect credential flow in the 4-gateway chain?
- [ ] Are new AWS operations going through the credential management system?
- [ ] Is the import pattern safe with try/except ImportError handling?

### **✅ Testing Requirements**:
- [ ] Run `python test_credential_flow.py` - Must pass 4/4 tests
- [ ] Check `grep -r "boto3.client" tidyllm/` - Look for new direct calls
- [ ] Verify gateway registry injection works for new gateways
- [ ] Test in fresh terminal to ensure credential persistence

---

## 🚨 **CRITICAL FILES - HANDLE WITH EXTREME CARE**

### **🔥 DO NOT MODIFY WITHOUT DEEP UNDERSTANDING**:
- `tidyllm/infrastructure/session/unified.py` - **CORE SESSION MANAGER**
- `tidyllm/gateways/base_gateway.py` - **SESSION INJECTION MECHANISM**
- `tidyllm/gateways/gateway_registry.py` - **MASS SESSION INJECTION**
- `tidyllm/admin/settings.yaml` - **CONFIGURATION SOURCE**

### **📚 STUDY THESE EXAMPLES**:
- `tidyllm/gateways/corporate_llm_gateway.py:386-396` - **FIXED BYPASS**
- `tidyllm/knowledge_systems/core/dynamic_model_discovery.py` - **ADDED INTEGRATION**
- `test_credential_flow.py` - **TESTING PATTERNS**

---

## 🔄 **AUTOMATIC SESSION RECOVERY - ENHANCED 2025-09-09**

### **✅ CRITICAL: UnifiedSessionManager Auto-Recovery NOW WITH:**
1. **Path-Independent Settings Discovery** - Works from ANY directory
2. **Dynamic Credential Loading** - Auto-detects new AWS fields in settings.yaml
3. **Automatic Fallback Chain** - No environment variables needed!

#### **Auto-Recovery Process** (`tidyllm/infrastructure/session/unified.py`):
```python
# Line 292: _initialize_connections() - Runs on startup
def _initialize_connections(self):
    """Initialize all service connections"""
    logger.info("[LAUNCH] Initializing service connections...")
    self._init_s3()      # Fresh S3 client
    self._init_bedrock() # Fresh Bedrock client  
    self._init_postgresql() # Fresh PostgreSQL pool

# Lines 439-449: Thread-safe client access
def get_s3_client(self):
    """Get S3 client (thread-safe)"""
    return self._s3_client

def get_bedrock_client(self):
    """Get Bedrock client (thread-safe)"""
    return self._bedrock_client
```

#### **Credential Discovery Priority** (Lines 75-81):
1. **IAM Role** → **AWS Profile** → **Environment** → **Settings File**
2. Automatically detects best available credentials on restart
3. Health checks validate all connections (`check_health()` - Line 463)

#### **NEW Path-Independent Settings Discovery** (Lines 212-250):
```python
# Searches upward from ANY directory to find settings.yaml
# Works from C:\Users\marti\Documents or /home/user/projects/deep/nested
for _ in range(5):  # Search up to 5 levels up
    potential_paths = [
        current_dir / "tidyllm" / "admin" / "settings.yaml",
        current_dir / "admin" / "settings.yaml",
        current_dir / "settings.yaml",
    ]
# Fallback to explicit paths if not found
Path.home() / "github" / "tidyllm" / "admin" / "settings.yaml"
```

#### **NEW Dynamic Credential Loading** (Lines 256-288):
```python
# Auto-detects ANY AWS field variations in settings.yaml
for key, value in aws_config.items():
    if key in ['access_key_id', 'aws_access_key_id', 'access_key']:
        self.config.s3_access_key_id = value
    elif key in ['secret_access_key', 'aws_secret_access_key', 'secret_key']:
        self.config.s3_secret_access_key = value
    # ... auto-maps all variations

# Also checks api_keys section for legacy support
api_keys = settings.get('api_keys', {})
```

#### **Session Restart - NO ENVIRONMENT VARIABLES NEEDED**:
```bash
# Just run your app from ANYWHERE:
cd /any/random/directory
streamlit run app.py

# UnifiedSessionManager automatically:
# 1. Searches upward to find settings.yaml
# 2. Dynamically loads ALL AWS fields
# 3. Creates S3 and Bedrock clients with settings credentials
# 4. Falls back to environment variables if settings not found
```

### **🚨 DO NOT MODIFY THESE AUTO-RECOVERY FUNCTIONS**:
- `_initialize_connections()` - **Core connection setup**
- `get_s3_client()` / `get_bedrock_client()` - **Thread-safe access**
- `check_health()` - **Connection validation**
- `_health_check_service()` - **Service-specific health checks**

**These functions enable automatic session recovery. Modifying them could break the credential restoration process that users depend on.**

---

## 🔍 **ARCHITECTURAL UNDERSTANDING REQUIRED**

### **4-Gateway Chain Must Be Consistent**:
```
✅ CorporateLLMGateway (UnifiedSessionManager) → 
✅ AIProcessingGateway (UnifiedSessionManager) → 
✅ WorkflowOptimizer (UnifiedSessionManager) → 
✅ DatabaseGateway (UnifiedSessionManager)
          +
✅ DomainRAG/Knowledge Systems (UnifiedSessionManager)
```

**ALL components must use UnifiedSessionManager** - any deviation causes "partial results"

### **Credential Discovery Priority**:
1. Environment variables (`AWS_ACCESS_KEY_ID`, etc.)
2. Settings file (`tidyllm/admin/settings.yaml`)  
3. AWS profiles and IAM roles
4. Fallback to direct boto3 (with warnings)

---

## 🚀 **SUCCESS INDICATORS**

### **✅ System Working Correctly When**:
- `test_credential_flow.py` passes 4/4 tests
- All components log "Using UnifiedSessionManager"
- No "Using direct boto3" warnings in normal operation
- S3 and Bedrock access consistent across all components
- No "NoCredentialsError" exceptions

### **❌ System Broken When**:
- "Partial results" - some components work, others don't
- Mix of "UnifiedSessionManager" and "direct boto3" log messages
- Credential errors in some gateways but not others
- Tests pass individually but fail when run together

---

## 💡 **AI AGENT RESPONSE TEMPLATES**

### **When Asked to Add AWS Functionality**:
```
I'll add this AWS functionality using the UnifiedSessionManager pattern to maintain 
consistent credential flow. Let me:

1. Check if this component has session manager integration
2. Use self.session_manager.get_*_client() instead of direct boto3
3. Add proper fallback handling and logging
4. Test the credential flow to ensure no regressions
```

### **When Debugging AWS Issues**:
```
This appears to be a credential flow issue. Let me:

1. Check if the component uses UnifiedSessionManager properly  
2. Verify there are no direct boto3.client() bypasses
3. Test the credential discovery chain (env -> settings -> IAM)
4. Run the credential flow test to identify the exact problem
```

---

## 🎯 **FINAL REMINDER**

> **The credential system was JUST FIXED after causing system-wide issues. Every AWS-related change must preserve the UnifiedSessionManager pattern. When in doubt, follow the existing patterns and run the tests.**

**This document exists because the credential management system is CRITICAL to TidyLLM's operation. Please treat it with the appropriate level of care.**

---

*Required reading for all AI agents working on TidyLLM AWS integration.*