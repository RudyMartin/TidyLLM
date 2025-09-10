# Gateway System Build Guide - IKEA Style 🛠️

**Document Version**: 1.0  
**Created**: 2025-09-06  
**Status**: MASTER BUILD GUIDE  
**Priority**: 🚨 READ THIS FIRST - BUILD NOTHING WITHOUT THIS GUIDE

---

## 📦 **What's In The Box** (Complete Parts List)

Before you start building, ensure you have ALL these components. **Missing ANY item will cause build failure.**

### 🔧 **Core Infrastructure Dependencies**
```bash
# Database Layer
□ PostgreSQL Server (v13+)
□ MLflow Tracking Server (v2.0+)
□ MLflow Gateway Server (v2.0+)

# Storage Layer  
□ AWS S3 Bucket (or S3-compatible)
□ Redis/ElastiCache (OPTIONAL - current setup uses local file caching)

# AI Provider Access
□ AWS Bedrock access + credentials (MANDATORY - provides ALL AI models)
□ OpenAI API key (OPTIONAL - only if bypassing Bedrock for direct OpenAI)
□ Anthropic API key (OPTIONAL - only if bypassing Bedrock for direct Anthropic)

# Monitoring
□ CloudWatch OR Prometheus setup
□ Grafana (optional, for dashboards)
```

### 🐍 **Python Package Dependencies**
```bash
# 🚨 CRITICAL: Use ONLY These Approved Packages
# Core Gateway Framework
tidyllm>=2.0.0              # Main framework - provides gateway architecture
tidyllm-sentence>=1.0.0     # MANDATORY (replaces sentence-transformers) - corporate-approved embeddings
tidyllm-compliance>=1.0.0   # MANDATORY - audit trails, SOX/GDPR compliance, risk assessment

# 🚨 MANDATORY Replacements (DO NOT use the forbidden alternatives)
tidyllm.tlm                 # MANDATORY (replaces numpy) - corporate-vetted numerical operations
polars>=0.20.0              # MANDATORY (replaces pandas) - faster, memory-safe dataframes
# ❌ FORBIDDEN: numpy, pandas, sklearn, sentence-transformers
# WHY FORBIDDEN: Security vulnerabilities, licensing issues, performance problems

# Database & Session Management
psycopg2-binary>=2.9.0      # PostgreSQL driver - required for audit logging
sqlalchemy>=2.0.0           # ORM - database abstraction for compliance tables

# MLflow Integration
mlflow>=2.0.0               # Model lifecycle management - required for CorporateLLMGateway
mlflow[extras]>=2.0.0       # Additional MLflow components - gateway routing, model registry

# AI Provider SDKs (Corporate-Approved Only)
boto3>=1.34.0               # AWS SDK - MANDATORY for Bedrock AI models and S3 storage
# NOTE: openai and anthropic are OPTIONAL - Current setup uses ALL models through Bedrock
openai>=1.0.0               # OPTIONAL - Only if using direct OpenAI API (not Bedrock)
anthropic>=0.25.0           # OPTIONAL - Only if using direct Anthropic API (not Bedrock)

# Caching & Performance
redis>=4.5.0                # OPTIONAL - in-memory cache (current setup uses local file cache)
python-dotenv>=1.0.0        # Environment variable management - secure config loading

# Monitoring & Compliance
prometheus-client>=0.17.0   # Metrics collection - required for performance monitoring
boto3                       # AWS CloudWatch integration - compliance monitoring
```

### 📁 **File Structure Requirements**
```
your-project/
├── scripts/
│   └── start_unified_sessions.py    # MANDATORY
├── tidyllm/
│   ├── gateways/
│   │   ├── corporate_llm_gateway.py
│   │   ├── ai_processing_gateway.py
│   │   └── workflow_optimizer_gateway.py
│   ├── heiros/                     # For Gateway #3
│   └── config/
├── sql/                            # Database schemas
├── tests/                          # Gateway tests
└── .env                           # Environment variables
```

---

## 🎯 **Build Order - MUST Follow This Sequence**

**⚠️ CRITICAL**: Each step MUST pass its diagnostic test before proceeding to the next step.

### **STEP 1: Infrastructure Foundation** 
*Estimated Time: 30-45 minutes*

#### 1.1 Database Connection Test (Already Configured)
```bash
# YOUR SYSTEM ALREADY HAS AWS RDS POSTGRESQL CONFIGURED
# No local PostgreSQL installation needed!
# 
# Current configuration (from your settings.yaml):
# - Host: vectorqa-cluster.cluster-cu562e4m02nq.us-east-1.rds.amazonaws.com  
# - Database: vectorqa
# - User: vectorqa_user
# - SSL: Required (production security)

# Test your existing database connection
python -c "
import psycopg2
import yaml

print('Testing your existing AWS RDS PostgreSQL connection...')

with open('tidyllm/admin/settings.yaml', 'r') as f:
    settings = yaml.safe_load(f)

pg = settings['postgres']
conn = psycopg2.connect(
    host=pg['host'],
    port=pg['port'], 
    database=pg['db_name'],
    user=pg['db_user'],
    password=pg['db_password'],
    sslmode=pg['ssl_mode']
)

cursor = conn.cursor()
cursor.execute('SELECT version();')
version = cursor.fetchone()[0]
print(f'✅ AWS RDS PostgreSQL connected: {version}')

cursor.execute('SELECT current_database(), current_user;')
db_info = cursor.fetchone()
print(f'✅ Database: {db_info[0]}, User: {db_info[1]}')

conn.close()
print('✅ Database connection test passed!')
"
```

**✅ DIAGNOSTIC**: AWS RDS PostgreSQL connection successful

#### 1.2 MLflow Services
```bash
# Terminal 1: Start MLflow Tracking Server
mlflow server \
  --backend-store-uri postgresql://user:pass@localhost/tidyllm_system \
  --default-artifact-root s3://your-bucket/mlflow-artifacts \
  --host 0.0.0.0 \
  --port 5000

# Terminal 2: Start MLflow Gateway
mlflow gateway start \
  --config-path ./mlflow_gateway_config.yaml \
  --host 0.0.0.0 \
  --port 8080
```

**✅ DIAGNOSTIC**: Both services respond to health checks
```bash
curl http://localhost:5000/health
curl http://localhost:8080/health
```

#### 1.3 S3 Storage
```bash
# Test S3 access
aws s3 ls s3://your-tidyllm-bucket/

# Create required folders
aws s3api put-object --bucket your-tidyllm-bucket --key workflows/
aws s3api put-object --bucket your-tidyllm-bucket --key mlflow-artifacts/
```

**✅ DIAGNOSTIC**: S3 bucket accessible, folders created

#### 1.4 Local File Cache Setup
```bash
# Create cache directory (will be created automatically)
# Current setup uses local file caching (.bedrock_cache/)

# Verify cache directory permissions
mkdir -p .bedrock_cache
chmod 755 .bedrock_cache

# Optional: Start Redis (if upgrading to Redis caching later)
# sudo systemctl start redis
# redis-cli ping
```

**✅ DIAGNOSTIC**: Cache directory exists and is writable

---

## 🔍 **Current AWS Bedrock Model Availability**

**Your setup uses AWS Bedrock for ALL AI models. Here's what's available (as of 2025):**

### **Anthropic Models** (via Bedrock)
```
anthropic.claude-3-sonnet-20240229-v1:0    # Primary model in your settings.yaml
anthropic.claude-3-haiku-20240307-v1:0     # Fast, cost-effective
anthropic.claude-3-opus-20240229-v1:0      # Most capable
anthropic.claude-v2:1                      # Legacy model
anthropic.claude-instant-v1                # Legacy instant model
```

### **OpenAI Models** (NOW available via Bedrock in 2025)
```
gpt-oss-120b                              # NEW - High-reasoning, 116.8B parameters
gpt-oss-20b                               # NEW - Lower latency, specialized use cases
# Both models: 128K context, chain-of-thought, coding/math optimized
```

### **Amazon Titan Models** (via Bedrock)
```
amazon.titan-text-express-v1              # Text generation
amazon.titan-text-lite-v1                 # Lightweight text
amazon.titan-embed-text-v1                # Embeddings
```

### **Meta Llama Models** (via Bedrock)
```
meta.llama2-13b-chat-v1                   # 13B parameter chat model
meta.llama2-70b-chat-v1                   # 70B parameter chat model
meta.llama2-7b-chat-v1                    # 7B parameter chat model
```

---

### **STEP 2: Gateway #1 - CorporateLLMGateway** 
*Estimated Time: 20-30 minutes*

#### 2.1 Environment Configuration
```bash
# .env file - REQUIRED variables
cat > .env << EOF
# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_GATEWAY_URI=http://localhost:8080

# Cost Limits
LLM_MAX_COST_PER_REQUEST=1.00
LLM_DAILY_BUDGET_USD=500.00

# Provider Settings
LLM_DEFAULT_PROVIDER=claude
LLM_FALLBACK_PROVIDER=openai-corporate

# Database
DATABASE_URL=postgresql://user:pass@localhost/tidyllm_system
EOF
```

#### 2.2 Gateway Initialization (Using Your Actual GatewayRegistry)
```python
# Your actual gateway system uses GatewayRegistry
from tidyllm.gateways import init_gateways, get_global_registry

# Configuration for your actual gateways
gateway_config = {
    "corporate_llm": {
        # Configuration for CorporateLLMGateway
        "mlflow_tracking_uri": "http://localhost:5000",
        "mlflow_gateway_uri": "http://localhost:8080",
        "budget_limit_daily_usd": 500.00,
        "audit_level": "full"
    },
    "ai_processing": {
        # Configuration for AIProcessingGateway  
        "backend_selection": "bedrock",  # Uses your Bedrock setup
        "enable_caching": True,
        "cache_dir": ".bedrock_cache"
    },
    "workflow_optimizer": {
        # Configuration for WorkflowOptimizerGateway
        "enable_flow_agreements": True,
        "enable_hierarchical_dag": True
    }
}
```

#### 2.3 Initialize Gateway #1 (Using Your Actual System)
```python
# test_gateway1.py - Using your actual GatewayRegistry
from tidyllm.gateways import init_gateways, get_global_registry

# Initialize using your actual gateway system
registry = init_gateways({
    "corporate_llm": {
        "mlflow_tracking_uri": "http://localhost:5000",
        "mlflow_gateway_uri": "http://localhost:8080",
        "budget_limit_daily_usd": 500.00
    }
})

# Get gateway using your actual registry methods
corporate_gateway = registry.get("corporate_llm")
assert corporate_gateway is not None, "Gateway #1 failed to initialize"

# Check gateway status
health = registry.health_check()
assert health["services"]["corporate_llm"]["healthy"], "Gateway #1 health check failed"
```

**✅ DIAGNOSTIC**: Gateway #1 Basic Test
```python
# Test basic functionality
request = LLMRequest(
    prompt="Say 'Gateway 1 working' and nothing else",
    provider="claude",
    model="claude-3-5-sonnet",
    max_tokens=10,
    reason="Gateway #1 diagnostic test"
)

response = gateway.process(request)
assert response.status == "success", f"Gateway #1 test failed: {response.error}"
assert "Gateway 1 working" in response.data['response']
print("✅ Gateway #1 PASSED diagnostic test")
```

**✅ DIAGNOSTIC**: Database Audit Trail
```sql
-- Verify audit logging
SELECT * FROM llm_audit_log WHERE reason LIKE '%diagnostic test%' ORDER BY timestamp DESC LIMIT 1;
```

**🚨 STOP**: Do not proceed to Step 3 until Gateway #1 passes ALL diagnostics.

---

### **STEP 3: Gateway #2 - AIProcessingGateway** 
*Estimated Time: 25-35 minutes*

#### 3.1 Additional Dependencies Check
```bash
# Verify Gateway #1 is still working
python test_gateway1.py

# Install additional packages
pip install redis boto3 openai anthropic
```

#### 3.2 AI Backend Configuration
```python
# Add to gateway_config.py
AI_PROCESSING_CONFIG = {
    "backend": "auto",
    "fallback_backends": ["bedrock", "openai"],
    "default_model": "claude-3-sonnet",
    "max_tokens": 2000,
    "temperature": 0.7,
    "timeout_seconds": 30,
    "enable_caching": True,
    "cache_dir": ".bedrock_cache",      # Local file caching (current setup)
    "cache_ttl_seconds": 3600,
    "enable_retries": True,
    "max_retries": 3,
    "retry_delay_base": 1.0,
    "enable_metrics": True,
    "metrics_backend": "cloudwatch",
    "bedrock_config": {
        "region": "us-east-1",
        "model_ids": {
            "claude-3-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
            "claude-3-haiku": "anthropic.claude-3-haiku-20240307-v1:0", 
            "claude-3-opus": "anthropic.claude-3-opus-20240229-v1:0",
            "gpt-oss-120b": "gpt-oss-120b",  # NEW OpenAI via Bedrock
            "gpt-oss-20b": "gpt-oss-20b",   # NEW OpenAI via Bedrock
            "titan-express": "amazon.titan-text-express-v1",
            "llama2-70b": "meta.llama2-70b-chat-v1"
        }
    }
}
```

#### 3.3 Initialize Gateway #2
```python
# test_gateway2.py
# MUST initialize Gateway #1 first (dependency)
registry = init_gateways({
    "corporate_llm": CORPORATE_LLM_CONFIG,    # REQUIRED
    "ai_processing": AI_PROCESSING_CONFIG      # Depends on corporate_llm
})

ai_gateway = get_gateway("ai_processing")
assert ai_gateway is not None, "Gateway #2 failed to initialize"
```

**✅ DIAGNOSTIC**: Gateway #2 Basic Test
```python
from tidyllm.gateways import AIRequest

# Test basic AI processing
request = AIRequest(
    prompt="Say 'Gateway 2 working' and nothing else",
    model="claude-3-sonnet",
    max_tokens=10,
    reason="Gateway #2 diagnostic test"
)

response = ai_gateway.process(request)
assert response.status == "success", f"Gateway #2 test failed: {response.error}"
assert "Gateway 2 working" in response.data['response']
print("✅ Gateway #2 PASSED basic test")
```

**✅ DIAGNOSTIC**: Cache Test
```python
# Test caching functionality
request1 = AIRequest(prompt="What is 2+2?", reason="Cache test 1")
response1 = ai_gateway.process(request1)
assert not response1.metadata.get('cache_hit', False), "First request should be cache miss"

request2 = AIRequest(prompt="What is 2+2?", reason="Cache test 2") # Same prompt
response2 = ai_gateway.process(request2)
assert response2.metadata.get('cache_hit', False), "Second request should be cache hit"
print("✅ Gateway #2 PASSED cache test")
```

**✅ DIAGNOSTIC**: Backend Failover Test
```python
# Test backend selection
request = AIRequest(
    prompt="Backend test",
    backend="auto",
    reason="Backend diagnostic test"
)
response = ai_gateway.process(request)
backend_used = response.metadata.get('backend_used')
assert backend_used is not None, "Backend selection failed"
print(f"✅ Gateway #2 PASSED backend test - Used: {backend_used}")
```

**🚨 STOP**: Do not proceed to Step 4 until Gateway #2 passes ALL diagnostics.

---

### **STEP 4: UnifiedSessionManager Setup** 
*Estimated Time: 15-20 minutes*

**⚠️ CRITICAL**: Gateway #3 CANNOT function without UnifiedSessionManager.

#### 4.1 Session Manager Implementation
```python
# scripts/start_unified_sessions.py - MUST exist
import boto3
import psycopg2
from typing import Dict, Any, Optional

class UnifiedSessionManager:
    """S3-First session manager - MANDATORY for Gateway #3"""
    
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.db_connection = psycopg2.connect(
            host="localhost",
            database="tidyllm_system",
            user="your_user",
            password="your_pass"
        )
        self.bucket_name = "your-tidyllm-bucket"
    
    def store_workflow_s3(self, workflow_data: Dict[str, Any], workflow_identifier: str, folder_prefix: str = "workflows/") -> bool:
        """
        Store workflow in S3 with clear path construction
        
        Args:
            workflow_data: The workflow definition/data to store  
            workflow_identifier: Unique identifier for this workflow (becomes part of S3 path)
                                Examples: "user_123_approval_workflow", "mvr_analysis_v2", "compliance_check_001"
            folder_prefix: S3 folder prefix from settings.yaml (default: "workflows/")
                          
        S3 Path Construction:
            Final S3 Key = folder_prefix + workflow_identifier
            Example: "workflows/" + "user_123_approval.json" = "workflows/user_123_approval.json"
        """
        try:
            full_s3_path = f"{folder_prefix}{workflow_identifier}"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=full_s3_path,
                Body=json.dumps(workflow_data)
            )
            return True
        except Exception as e:
            print(f"S3 storage failed: {e}")
            return False
    
    def get_workflow_s3(self, workflow_identifier: str, folder_prefix: str = "workflows/") -> Optional[Dict[str, Any]]:
        """
        Retrieve workflow from S3
        
        Args:
            workflow_identifier: Unique identifier used when storing the workflow
            folder_prefix: S3 folder prefix (must match what was used for storage)
        """
        try:
            full_s3_path = f"{folder_prefix}{workflow_identifier}"
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=full_s3_path
            )
            return json.loads(response['Body'].read())
        except Exception as e:
            print(f"S3 retrieval failed: {e}")
            return None
    
    def log_audit_db(self, audit_data: Dict[str, Any]) -> bool:
        """Log audit trail to database"""
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO workflow_audit_log (operation_type, workflow_data, timestamp) VALUES (%s, %s, %s)",
                    (audit_data['operation_type'], json.dumps(audit_data), datetime.now())
                )
            self.db_connection.commit()
            return True
        except Exception as e:
            print(f"Database logging failed: {e}")
            return False
```

**✅ DIAGNOSTIC**: Session Manager Test
```python
# test_session_manager.py
from scripts.start_unified_sessions import UnifiedSessionManager
import json

session_mgr = UnifiedSessionManager()

# Test S3 operations
test_workflow = {"name": "test", "steps": []}
assert session_mgr.store_workflow_s3(test_workflow, "diagnostic_test.json"), "S3 store failed"

retrieved = session_mgr.get_workflow_s3("diagnostic_test.json")
assert retrieved == test_workflow, "S3 retrieve failed"

# Test database logging
audit_data = {"operation_type": "diagnostic", "test": True}
assert session_mgr.log_audit_db(audit_data), "Database logging failed"

print("✅ UnifiedSessionManager PASSED all tests")
```

---

### **STEP 5: Gateway #3 - WorkflowOptimizerGateway** 
*Estimated Time: 35-45 minutes*

#### 5.1 HeirOS Components Setup
```bash
# Create HeirOS directory structure
mkdir -p tidyllm/heiros

# Required HeirOS files
touch tidyllm/heiros/__init__.py
touch tidyllm/heiros/hierarchical_dag_manager.py
touch tidyllm/heiros/flow_agreement_manager.py
```

#### 5.2 Gateway #3 Configuration
```python
# Add to gateway_config.py
WORKFLOW_OPTIMIZER_CONFIG = {
    "optimization_level": "high",
    "enable_auto_fix": True,
    "enable_compliance_mode": True,
    "max_optimization_iterations": 5,
    "enable_hierarchical_dag": True,
    "dag_max_depth": 10,
    "enable_flow_agreements": True,
    "analysis_depth": "deep",
    "bottleneck_threshold": 0.8,
    "compliance_strictness": "high",
    "audit_level": "comprehensive",
    "track_all_optimizations": True,
    "enable_approval_workflows": True,
    "session_manager_required": True,  # MANDATORY
    "flow_config": {
        "require_stakeholder_approval": True,
        "risk_assessment_required": True,
        "compliance_frameworks": ["SOX", "GDPR", "Internal"],
        "approval_timeout_days": 7
    }
}
```

#### 5.3 Initialize All Three Gateways
```python
# test_all_gateways.py
# MUST initialize ALL dependencies in correct order
registry = init_gateways({
    "corporate_llm": CORPORATE_LLM_CONFIG,      # Level 1 - Foundation
    "ai_processing": AI_PROCESSING_CONFIG,       # Level 2 - Requires Level 1  
    "workflow_optimizer": WORKFLOW_OPTIMIZER_CONFIG  # Level 3 - Requires Levels 1&2
})

# Verify all gateways initialized
corporate_gateway = get_gateway("corporate_llm")
ai_gateway = get_gateway("ai_processing") 
optimizer_gateway = get_gateway("workflow_optimizer")

assert all([corporate_gateway, ai_gateway, optimizer_gateway]), "Gateway initialization failed"
```

**✅ DIAGNOSTIC**: Gateway #3 Basic Test
```python
from tidyllm.gateways import WorkflowRequest, WorkflowOperation

# Test workflow analysis
test_workflow = {
    "name": "Diagnostic Workflow",
    "steps": [
        {"type": "input", "config": {"format": "json"}},
        {"type": "process", "config": {"ai_model": "claude-3-sonnet"}},
        {"type": "output", "config": {"format": "json"}}
    ]
}

request = WorkflowRequest(
    operation=WorkflowOperation.ANALYZE,
    workflow_definition=test_workflow,
    reason="Gateway #3 diagnostic test"
)

response = optimizer_gateway.process(request)
assert response.status == "success", f"Gateway #3 test failed: {response.error}"
assert 'analysis' in response.data, "Workflow analysis missing"
print("✅ Gateway #3 PASSED basic test")
```

**✅ DIAGNOSTIC**: UnifiedSessionManager Integration Test
```python
# Test S3-First integration
from scripts.start_unified_sessions import UnifiedSessionManager

session_mgr = UnifiedSessionManager()
workflow_stored = session_mgr.store_workflow_s3(test_workflow, "gateway3_test.json")
assert workflow_stored, "Gateway #3 failed to integrate with UnifiedSessionManager"
print("✅ Gateway #3 PASSED S3-First integration test")
```

**✅ DIAGNOSTIC**: Full Dependency Chain Test
```python
# Test that Gateway #3 can use lower gateways
request = WorkflowRequest(
    operation=WorkflowOperation.OPTIMIZE,
    workflow_definition=test_workflow,
    optimization_level="medium",
    reason="Full dependency chain test"
)

response = optimizer_gateway.process(request)
assert response.status == "success", "Full dependency chain test failed"

# Verify all gateways were used in the chain
metadata = response.metadata
assert 'corporate_audit_id' in metadata, "CorporateLLMGateway not used"
assert 'processing_metrics' in metadata, "AIProcessingGateway not used"
assert 'optimization_applied' in metadata, "WorkflowOptimizer failed"
print("✅ Gateway #3 PASSED full dependency chain test")
```

---

## 🔍 **System Diagnostic Routine**

Use this for troubleshooting and health checks:

### **Level 1 Diagnostics** (Infrastructure)
```bash
#!/bin/bash
# diagnostic_level1.sh

echo "🔍 Level 1: Infrastructure Diagnostics"

# Database
echo "Testing PostgreSQL..."
psql -d tidyllm_system -c "SELECT 1;" || echo "❌ PostgreSQL FAILED"

# MLflow Services  
echo "Testing MLflow Tracking..."
curl -f http://localhost:5000/health || echo "❌ MLflow Tracking FAILED"

echo "Testing MLflow Gateway..."
curl -f http://localhost:8080/health || echo "❌ MLflow Gateway FAILED"

# S3 Storage
echo "Testing S3 access..."
aws s3 ls s3://your-tidyllm-bucket/ || echo "❌ S3 Access FAILED"

# Redis Cache
echo "Testing Redis..."
redis-cli ping | grep PONG || echo "❌ Redis FAILED"

echo "✅ Level 1 Infrastructure diagnostics complete"
```

### **Level 2 Diagnostics** (Gateway Chain)
```python
#!/usr/bin/env python3
# diagnostic_level2.py

def test_gateway_chain():
    print("🔍 Level 2: Gateway Chain Diagnostics")
    
    # Test Gateway #1 only
    print("Testing Gateway #1 (CorporateLLMGateway)...")
    registry1 = init_gateways({"corporate_llm": CORPORATE_LLM_CONFIG})
    gateway1 = get_gateway("corporate_llm")
    test_gateway1(gateway1)
    
    # Test Gateways #1+#2
    print("Testing Gateway #2 (AIProcessingGateway)...")
    registry2 = init_gateways({
        "corporate_llm": CORPORATE_LLM_CONFIG,
        "ai_processing": AI_PROCESSING_CONFIG
    })
    gateway2 = get_gateway("ai_processing")
    test_gateway2(gateway2)
    
    # Test all three gateways
    print("Testing Gateway #3 (WorkflowOptimizerGateway)...")
    registry3 = init_gateways({
        "corporate_llm": CORPORATE_LLM_CONFIG,
        "ai_processing": AI_PROCESSING_CONFIG,
        "workflow_optimizer": WORKFLOW_OPTIMIZER_CONFIG
    })
    gateway3 = get_gateway("workflow_optimizer")
    test_gateway3(gateway3)
    
    print("✅ Level 2 Gateway chain diagnostics complete")

if __name__ == "__main__":
    test_gateway_chain()
```

### **Level 3 Diagnostics** (Full System Integration)
```python
#!/usr/bin/env python3
# diagnostic_level3.py

def test_full_system():
    print("🔍 Level 3: Full System Integration Diagnostics")
    
    # Initialize complete system
    full_registry = init_gateways({
        "corporate_llm": CORPORATE_LLM_CONFIG,
        "ai_processing": AI_PROCESSING_CONFIG,
        "workflow_optimizer": WORKFLOW_OPTIMIZER_CONFIG
    })
    
    # Test end-to-end workflow
    optimizer = get_gateway("workflow_optimizer")
    session_mgr = UnifiedSessionManager()
    
    # Complex workflow test
    complex_workflow = {
        "name": "Full System Test",
        "type": "hierarchical_dag",
        "compliance_required": True,
        "steps": [
            {"type": "intake", "config": {"source": "s3"}},
            {"type": "ai_analysis", "config": {"model": "claude-3-sonnet"}},
            {"type": "optimization", "config": {"level": "high"}},
            {"type": "compliance_check", "config": {"frameworks": ["SOX", "GDPR"]}},
            {"type": "output", "config": {"destination": "s3"}}
        ]
    }
    
    # Test complete workflow lifecycle
    test_workflow_lifecycle(optimizer, session_mgr, complex_workflow)
    test_flow_agreement_system(optimizer)
    test_hierarchical_dag_system(optimizer)
    
    print("✅ Level 3 Full system integration diagnostics complete")

if __name__ == "__main__":
    test_full_system()
```

---

## 🚨 **Troubleshooting Guide**

### **Common Build Failures and Solutions**

#### **"Gateway #2 fails to initialize"**
```bash
# Check dependency order
python -c "
from tidyllm.gateways import get_gateway
gateway1 = get_gateway('corporate_llm')
print('Gateway #1 status:', 'OK' if gateway1 else 'MISSING')
"
```

#### **"UnifiedSessionManager not found"**
```bash
# Verify file exists and is importable
ls -la scripts/start_unified_sessions.py
python -c "from scripts.start_unified_sessions import UnifiedSessionManager; print('✅ Session manager OK')"
```

#### **"S3-First violations"**
```bash
# Check for forbidden local file operations
grep -r "open(\|with open\|\.write(\|\.read(" tidyllm/gateways/ || echo "✅ No local file operations found"
```

#### **"Forbidden dependency usage"**
```bash
# Check for forbidden imports
grep -r "import numpy\|import pandas\|import sklearn\|from numpy\|from pandas\|from sklearn" tidyllm/ && echo "❌ FORBIDDEN dependencies found" || echo "✅ No forbidden dependencies"

# Check for correct tidyllm.tlm usage
grep -r "import tidyllm.tlm\|from tidyllm.tlm" tidyllm/ || echo "⚠️  Check: Should be using tidyllm.tlm for numerical operations"

# Check for correct polars usage  
grep -r "import polars\|pd\." tidyllm/ || echo "⚠️  Check: Should be using polars for dataframes"

# Check for tidyllm-compliance usage
grep -r "import tidyllm.compliance\|from tidyllm.compliance" tidyllm/ || echo "⚠️  Check: Should be using tidyllm-compliance for audit trails"
```

#### **"MLflow services not responding"**
```bash
# Check service status
curl -v http://localhost:5000/health
curl -v http://localhost:8080/health

# Check logs
tail -f mlflow_tracking.log
tail -f mlflow_gateway.log
```

---

## 📋 **Final System Validation Checklist**

Run this final checklist to confirm your system is production-ready:

```bash
□ All infrastructure services running (PostgreSQL, MLflow, Redis, S3)
□ All three gateways initialize without errors
□ Gateway #1 passes audit trail verification
□ Gateway #2 passes cache and failover tests  
□ Gateway #3 passes workflow optimization tests
□ UnifiedSessionManager S3 operations work
□ Database connections established
□ All diagnostic scripts pass
□ No forbidden dependencies (pandas, numpy, sklearn, sentence-transformers)
□ Only approved dependencies used (polars, tidyllm.tlm, tidyllm-sentence, tidyllm-compliance)
□ Code uses tidyllm.tlm for all numerical operations (NOT numpy)
□ Code uses polars for all dataframes (NOT pandas)
□ S3-First architecture enforced (no local file operations)
□ Comprehensive logging enabled
□ Cost limits and budgets configured
□ Compliance frameworks activated
```

**🎉 SUCCESS**: If all checkboxes are checked, your TidyLLM Gateway system is ready for production use.

---

**Document Location**: `/docs/2025-09-06/Gateway System Build Guide - IKEA Style README.md`  
**Last Updated**: 2025-09-06  
**Status**: MASTER BUILD GUIDE - Use this for all gateway deployments