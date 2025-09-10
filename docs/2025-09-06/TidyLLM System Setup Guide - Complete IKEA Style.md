# TidyLLM System Setup Guide - Complete IKEA Style
*Zero to Production in Systematic Steps*

## 🎯 **OVERVIEW - WHAT YOU'RE BUILDING**

A complete TidyLLM AI gateway system with:
- **3-Tier Gateway Architecture**: Corporate LLM → AI Processing → Workflow Optimizer  
- **AWS Cloud Integration**: RDS PostgreSQL + S3 + Bedrock
- **FLOW Agreement Chaining**: Corporate-grade AI request management
- **Enterprise Monitoring**: Health checks, caching, audit trails

**⚠️ CRITICAL**: This is NOT a toy setup. This requires real AWS permissions, live databases, and production-grade configuration.

---

## 📋 **STEP 0: PREREQUISITES CHECKLIST**

Before starting, ensure you have:

### **Required Access:**
- [ ] **AWS Account** with admin/PowerUser permissions
- [ ] **AWS CLI** installed and configured
- [ ] **Database Admin Access** (to create RDS instance)
- [ ] **S3 Bucket Creation** permissions
- [ ] **Bedrock Service Access** in your AWS region

### **Required Software:**
- [ ] **Python 3.11+** installed
- [ ] **Git** installed and configured
- [ ] **PostgreSQL client tools** (psql command)
- [ ] **Text editor** or IDE

### **Required Information:**
- [ ] **AWS Region** where you'll deploy (e.g., `us-east-1`)
- [ ] **Database Name** for your TidyLLM instance
- [ ] **S3 Bucket Name** (must be globally unique)

**❌ STOP HERE** if you don't have these. Get AWS access first!

---

## 🏗️ **PART I: AWS FOUNDATION SETUP**

### **STEP 1: AWS Permissions Setup** 
*Time: 10 minutes*

#### 1.1 Verify AWS CLI Access
```bash
# Test AWS access
aws sts get-caller-identity

# Expected output: Your AWS account details
{
  "UserId": "...",
  "Account": "123456789012", 
  "Arn": "arn:aws:iam::123456789012:user/yourname"
}
```

**✅ SUCCESS CRITERIA**: Command returns your AWS account info

#### 1.2 Required AWS Permissions
Your AWS user/role needs these services:
- **RDS**: Create, modify, connect to PostgreSQL instances
- **S3**: Create buckets, upload/download objects, list buckets
- **Bedrock**: Access foundation models (Claude, etc.)
- **IAM**: Assume roles, get credentials (for service integration)

```bash
# Test Bedrock access
aws bedrock list-foundation-models --region us-east-1

# Test S3 access  
aws s3 ls

# Test RDS access
aws rds describe-db-instances
```

**✅ SUCCESS CRITERIA**: All commands work without permission errors

---

### **STEP 2: Create AWS RDS PostgreSQL Database**
*Time: 15-20 minutes*

#### 2.1 Create RDS Instance
```bash
# Set your variables
export DB_INSTANCE_ID="tidyllm-main"
export DB_NAME="tidyllm"
export DB_USERNAME="tidyllm_admin"
export DB_PASSWORD="your_secure_password_here"  # Use a strong password!
export AWS_REGION="us-east-1"  # Change to your region

# Create PostgreSQL RDS instance
aws rds create-db-instance \
    --db-instance-identifier $DB_INSTANCE_ID \
    --db-instance-class db.t3.micro \
    --engine postgres \
    --engine-version 15.4 \
    --master-username $DB_USERNAME \
    --master-user-password $DB_PASSWORD \
    --db-name $DB_NAME \
    --allocated-storage 20 \
    --storage-type gp2 \
    --vpc-security-group-ids default \
    --publicly-accessible \
    --backup-retention-period 7 \
    --region $AWS_REGION

echo "Database creation initiated. This takes 10-15 minutes..."
```

#### 2.2 Wait for Database to be Available
```bash
# Check status (repeat until 'available')
aws rds describe-db-instances \
    --db-instance-identifier $DB_INSTANCE_ID \
    --query 'DBInstances[0].DBInstanceStatus' \
    --region $AWS_REGION

# Get connection endpoint when ready
aws rds describe-db-instances \
    --db-instance-identifier $DB_INSTANCE_ID \
    --query 'DBInstances[0].Endpoint.Address' \
    --region $AWS_REGION --output text
```

**✅ SUCCESS CRITERIA**: Status shows "available" and you have an endpoint URL

#### 2.3 Test Database Connection
```bash
# Get the endpoint
DB_HOST=$(aws rds describe-db-instances \
    --db-instance-identifier $DB_INSTANCE_ID \
    --query 'DBInstances[0].Endpoint.Address' \
    --region $AWS_REGION --output text)

# Test connection
psql -h $DB_HOST -U $DB_USERNAME -d $DB_NAME -c "SELECT version();"
```

**✅ SUCCESS CRITERIA**: PostgreSQL version information displays

---

### **STEP 3: Create S3 Bucket**
*Time: 5 minutes*

#### 3.1 Create Unique S3 Bucket
```bash
# Choose a globally unique bucket name
export S3_BUCKET="tidyllm-$(date +%s)-$(whoami)"  # Example: tidyllm-1693123456-john
export AWS_REGION="us-east-1"

# Create bucket
aws s3 mb s3://$S3_BUCKET --region $AWS_REGION

# Verify bucket exists
aws s3 ls s3://$S3_BUCKET
```

#### 3.2 Configure Bucket Permissions
```bash
# Set bucket versioning (recommended)
aws s3api put-bucket-versioning \
    --bucket $S3_BUCKET \
    --versioning-configuration Status=Enabled

# Test upload/download
echo "TidyLLM Test File" > test-file.txt
aws s3 cp test-file.txt s3://$S3_BUCKET/
aws s3 cp s3://$S3_BUCKET/test-file.txt downloaded-test.txt
cat downloaded-test.txt
rm test-file.txt downloaded-test.txt
```

**✅ SUCCESS CRITERIA**: File upload/download works successfully

---

### **STEP 4: Verify Bedrock Access**
*Time: 5 minutes*

#### 4.1 Test Bedrock Foundation Models
```bash
# List available models
aws bedrock list-foundation-models \
    --region $AWS_REGION \
    --query 'modelSummaries[?contains(modelId, `claude`)].modelId'

# Test model access (optional - may have costs)
aws bedrock-runtime invoke-model \
    --region $AWS_REGION \
    --model-id anthropic.claude-3-haiku-20240307-v1:0 \
    --body '{"messages":[{"role":"user","content":"Test"}],"max_tokens":10,"anthropic_version":"bedrock-2023-05-31"}' \
    --cli-binary-format raw-in-base64-out \
    response.json

cat response.json
rm response.json
```

**✅ SUCCESS CRITERIA**: Claude models listed, test invocation works (optional)

---

## 🐍 **PART II: PYTHON ENVIRONMENT SETUP**

### **STEP 5: Python Environment Setup**
*Time: 10 minutes*

#### 5.1 Verify Python Installation
```bash
# Check Python version (must be 3.11+)
python --version
# or
python3 --version

# Check pip
pip --version
```

#### 5.2 Create Project Environment
```bash
# Navigate to your TidyLLM directory
cd /path/to/tidyllm  # Adjust path

# Create virtual environment (recommended)
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

#### 5.3 Install Required Dependencies
```bash
# Install core dependencies
pip install boto3 psycopg2-binary python-dotenv pyyaml

# Install TidyLLM specific requirements (if requirements.txt exists)
pip install -r requirements.txt

# Verify installations
python -c "import boto3, psycopg2, yaml; print('Dependencies installed successfully')"
```

**✅ SUCCESS CRITERIA**: All imports work without errors

---

## ⚙️ **PART III: TIDYLLM CONFIGURATION**

### **STEP 6: Create Configuration Files**
*Time: 10 minutes*

#### 6.1 Create settings.yaml
```bash
# Create settings.yaml with your AWS details
cat > settings.yaml << EOF
aws:
  region: $AWS_REGION
  default_bucket: $S3_BUCKET

database:
  host: $DB_HOST
  port: 5432
  database: $DB_NAME
  username: $DB_USERNAME
  password: $DB_PASSWORD

bedrock:
  region: $AWS_REGION
  
cache:
  enabled: true
  directory: .bedrock_cache
  ttl: 3600

logging:
  level: INFO
EOF

echo "Configuration created: settings.yaml"
```

#### 6.2 Create Environment Variables (Alternative)
```bash
# Create .env file for sensitive data
cat > .env << EOF
AWS_REGION=$AWS_REGION
S3_BUCKET=$S3_BUCKET
DB_HOST=$DB_HOST
DB_USERNAME=$DB_USERNAME
DB_PASSWORD=$DB_PASSWORD
PYTHONPATH=.
EOF

echo "Environment file created: .env"
```

#### 6.3 Create Cache Directory
```bash
# Create cache directory structure
mkdir -p .bedrock_cache/claude
mkdir -p .bedrock_cache/anthropic
mkdir -p logs

echo "Cache directories created"
```

**✅ SUCCESS CRITERIA**: Files exist and contain your AWS details

---

## 🗄️ **PART IV: DATABASE INITIALIZATION**

### **STEP 7: Initialize TidyLLM Database Schema**
*Time: 10 minutes*

#### 7.1 Create Database Tables
```bash
# Connect to database and create schema
psql -h $DB_HOST -U $DB_USERNAME -d $DB_NAME << EOF
-- Create FLOW agreements table
CREATE TABLE IF NOT EXISTS heiros_flow_agreements (
    id SERIAL PRIMARY KEY,
    trigger_pattern VARCHAR(255) NOT NULL,
    flow_encoding TEXT NOT NULL,
    expanded_meaning TEXT,
    action VARCHAR(100),
    real_implementation TEXT,
    fallback TEXT,
    parameters JSONB,
    expected_output TEXT,
    confidence_threshold FLOAT DEFAULT 0.8,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create execution history table
CREATE TABLE IF NOT EXISTS flow_execution_history (
    id SERIAL PRIMARY KEY,
    agreement_id INTEGER REFERENCES heiros_flow_agreements(id),
    execution_mode VARCHAR(50),
    result JSONB,
    execution_time_ms INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id VARCHAR(100),
    success BOOLEAN
);

-- Create audit trail table
CREATE TABLE IF NOT EXISTS llm_audit_trail (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(100),
    prompt TEXT,
    model VARCHAR(100),
    audit_reason TEXT,
    response_length INTEGER,
    cost_cents INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    gateway VARCHAR(50)
);

-- Verify tables created
\dt
EOF
```

#### 7.2 Insert Default FLOW Agreements
```bash
psql -h $DB_HOST -U $DB_USERNAME -d $DB_NAME << EOF
INSERT INTO heiros_flow_agreements 
    (trigger_pattern, flow_encoding, expanded_meaning, action, real_implementation, fallback) 
VALUES 
    ('Performance Test', '@performance#test!benchmark@dspy_operations', 
     'Run comprehensive performance benchmark of DSPy wrapper operations',
     'performance_benchmark', 'dspy_wrapper.benchmark_performance', 'simulate_performance_test'),
     
    ('Cost Analysis', '@cost#analysis!track@dspy_operations',
     'Analyze cost patterns and optimization opportunities for DSPy operations', 
     'cost_analysis', 'dspy_wrapper.analyze_costs', 'simulate_cost_analysis'),
     
    ('Error Analysis', '@error#analysis!identify@dspy_failures',
     'Analyze error patterns and failure modes in DSPy operations',
     'error_analysis', 'dspy_wrapper.analyze_errors', 'simulate_error_analysis');

-- Verify data inserted
SELECT trigger_pattern, action FROM heiros_flow_agreements;
EOF
```

**✅ SUCCESS CRITERIA**: Tables exist and sample data is inserted

---

## 🧪 **PART V: COMPONENT TESTING & RESTART PROCEDURES**

### **STEP 8: Create System Health Monitor**
*Time: 15 minutes*

#### 8.1 Create Component Health Checker
```bash
cat > check_components.py << 'EOF'
#!/usr/bin/env python3
"""
TidyLLM Component Health Checker and Restart Script
==================================================
Checks all components and restarts failed services automatically.
"""

import os
import sys
import subprocess
import time
import boto3
import psycopg2
from datetime import datetime

class ComponentChecker:
    def __init__(self):
        self.aws_region = os.getenv('AWS_REGION', 'us-east-1')
        self.s3_bucket = os.getenv('S3_BUCKET')
        self.db_host = os.getenv('DB_HOST')
        self.db_name = os.getenv('DB_NAME', 'tidyllm')
        self.db_user = os.getenv('DB_USERNAME')
        self.db_password = os.getenv('DB_PASSWORD')
        
    def check_aws_credentials(self):
        """Check AWS credentials and permissions"""
        try:
            sts = boto3.client('sts', region_name=self.aws_region)
            identity = sts.get_caller_identity()
            print(f"✅ AWS Credentials: {identity['Arn']}")
            return True
        except Exception as e:
            print(f"❌ AWS Credentials: {e}")
            return False
    
    def check_s3_access(self):
        """Check S3 bucket access"""
        try:
            s3 = boto3.client('s3', region_name=self.aws_region)
            s3.head_bucket(Bucket=self.s3_bucket)
            print(f"✅ S3 Bucket: {self.s3_bucket}")
            return True
        except Exception as e:
            print(f"❌ S3 Bucket: {e}")
            return False
    
    def check_database(self):
        """Check PostgreSQL database connection"""
        try:
            conn = psycopg2.connect(
                host=self.db_host,
                database=self.db_name,
                user=self.db_user,
                password=self.db_password,
                connect_timeout=10
            )
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            conn.close()
            print(f"✅ Database: {self.db_host}")
            return True
        except Exception as e:
            print(f"❌ Database: {e}")
            return False
    
    def check_bedrock_access(self):
        """Check Bedrock service access"""
        try:
            bedrock = boto3.client('bedrock', region_name=self.aws_region)
            models = bedrock.list_foundation_models()
            claude_models = [m for m in models['modelSummaries'] if 'claude' in m['modelId'].lower()]
            print(f"✅ Bedrock: {len(claude_models)} Claude models available")
            return True
        except Exception as e:
            print(f"❌ Bedrock: {e}")
            return False
    
    def restart_failed_services(self):
        """Restart or repair failed services"""
        print("\n🔄 Checking for repair options...")
        
        # Check if database needs restart (basic check)
        if not self.check_database():
            print("🔧 Database connection failed - check RDS instance status")
            try:
                rds = boto3.client('rds', region_name=self.aws_region)
                instances = rds.describe_db_instances()
                for instance in instances['DBInstances']:
                    if instance['Endpoint']['Address'] == self.db_host:
                        status = instance['DBInstanceStatus']
                        print(f"   RDS Status: {status}")
                        if status != 'available':
                            print(f"   ⚠️  Database is {status} - may need to wait or restart")
            except Exception as e:
                print(f"   Error checking RDS: {e}")
    
    def run_full_check(self):
        """Run complete component check"""
        print("🔍 TidyLLM Component Health Check")
        print("=" * 50)
        
        checks = [
            ("AWS Credentials", self.check_aws_credentials),
            ("S3 Access", self.check_s3_access),
            ("Database", self.check_database), 
            ("Bedrock", self.check_bedrock_access)
        ]
        
        results = {}
        for name, check_func in checks:
            results[name] = check_func()
        
        # Summary
        passed = sum(results.values())
        total = len(results)
        
        print(f"\n📊 Results: {passed}/{total} components healthy")
        
        if passed == total:
            print("🎉 All components operational!")
            return True
        else:
            print("⚠️  Some components need attention")
            self.restart_failed_services()
            return False

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    checker = ComponentChecker()
    success = checker.run_full_check()
    sys.exit(0 if success else 1)
EOF

chmod +x check_components.py
```

#### 8.2 Test Component Health Check
```bash
# Make sure environment is loaded
source .env  # or load your environment variables

# Run health check
python check_components.py
```

**✅ SUCCESS CRITERIA**: All 4 components show ✅ healthy status

---

### **STEP 9: Database Live Check and Restart**
*Time: 10 minutes*

#### 9.1 Create Database Monitor
```bash
cat > monitor_database.py << 'EOF'
#!/usr/bin/env python3
"""
Database Monitor and Auto-Restart Script
Monitors PostgreSQL RDS and restarts if needed.
"""

import boto3
import psycopg2
import time
import os
from datetime import datetime

def check_rds_status(db_instance_id, region):
    """Check RDS instance status"""
    try:
        rds = boto3.client('rds', region_name=region)
        response = rds.describe_db_instances(DBInstanceIdentifier=db_instance_id)
        instance = response['DBInstances'][0]
        return instance['DBInstanceStatus'], instance['Endpoint']['Address']
    except Exception as e:
        print(f"Error checking RDS: {e}")
        return None, None

def restart_rds_if_needed(db_instance_id, region):
    """Restart RDS if it's stopped"""
    try:
        rds = boto3.client('rds', region_name=region)
        status, endpoint = check_rds_status(db_instance_id, region)
        
        if status == 'stopped':
            print(f"🔄 Starting RDS instance {db_instance_id}...")
            rds.start_db_instance(DBInstanceIdentifier=db_instance_id)
            
            # Wait for it to become available
            print("⏳ Waiting for RDS to start (this may take 5-10 minutes)...")
            waiter = rds.get_waiter('db_instance_available')
            waiter.wait(DBInstanceIdentifier=db_instance_id, WaiterConfig={'MaxAttempts': 30})
            print("✅ RDS instance started successfully")
            
        elif status == 'available':
            print(f"✅ RDS instance {db_instance_id} is already running")
        else:
            print(f"⚠️  RDS instance status: {status}")
            
        return True
    except Exception as e:
        print(f"❌ Error managing RDS: {e}")
        return False

if __name__ == "__main__":
    # Configuration
    db_instance_id = os.getenv('DB_INSTANCE_ID', 'tidyllm-main')
    region = os.getenv('AWS_REGION', 'us-east-1')
    
    print("🗄️  Database Monitor Starting...")
    restart_rds_if_needed(db_instance_id, region)
EOF

chmod +x monitor_database.py
```

#### 9.2 Test Database Monitor
```bash
# Test database monitoring
export DB_INSTANCE_ID="tidyllm-main"  # Your RDS instance ID
python monitor_database.py
```

**✅ SUCCESS CRITERIA**: Database status shows "available" or successfully starts

---

## 🚀 **PART VI: FINAL SYSTEM VERIFICATION**

### **STEP 10: Complete System Test**
*Time: 10 minutes*

#### 10.1 Run Enhanced Diagnostic Script
```bash
# Update the diagnostic script with proper environment loading
cat > run_full_diagnostics.py << 'EOF'
#!/usr/bin/env python3
"""
Complete TidyLLM System Verification
Runs full end-to-end testing of configured system.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to Python path
sys.path.insert(0, '.')

# Import our diagnostic script functions
from run_diagnostics import main as run_diagnostics

def pre_flight_check():
    """Verify environment before running diagnostics"""
    required_vars = ['AWS_REGION', 'S3_BUCKET', 'DB_HOST', 'DB_USERNAME', 'DB_PASSWORD']
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        print(f"❌ Missing environment variables: {missing}")
        print("Make sure your .env file is properly configured")
        return False
    
    print("✅ Environment variables loaded")
    return True

if __name__ == "__main__":
    print("🚀 TidyLLM Complete System Verification")
    print("=" * 50)
    
    # Pre-flight check
    if not pre_flight_check():
        sys.exit(1)
    
    # Run diagnostics
    exit_code = run_diagnostics()
    
    if exit_code == 0:
        print("\n🎉 SYSTEM FULLY OPERATIONAL - READY FOR CLIENT DEMO!")
    else:
        print("\n⚠️  SYSTEM REQUIRES ATTENTION - Check failed components above")
    
    sys.exit(exit_code)
EOF

chmod +x run_full_diagnostics.py
```

#### 10.2 Final System Test
```bash
# Run complete verification
python run_full_diagnostics.py
```

**🎯 SUCCESS CRITERIA**: 
- All components show ✅ PASS
- Final message: "SYSTEM FULLY OPERATIONAL - READY FOR CLIENT DEMO!"

---

## 📚 **PART VII: OPERATIONAL PROCEDURES**

### **Daily Operations:**
```bash
# Morning health check
python check_components.py

# If database issues:
python monitor_database.py

# Full system verification:
python run_full_diagnostics.py
```

### **Before Client Demos:**
```bash
# Complete pre-demo check
python run_full_diagnostics.py

# Expected output:
# [SUCCESS] ALL SYSTEMS OPERATIONAL
# [PASS] X/X tests passed (100%)
# [STATUS] System Status: READY FOR DEMO
```

### **Troubleshooting Quick Reference:**

| Error | Solution |
|-------|----------|
| AWS Credentials | `aws configure` or check IAM permissions |
| Database Connection | Run `python monitor_database.py` |
| S3 Access Denied | Check bucket permissions and region |
| Bedrock Access | Enable Bedrock in AWS console for your region |
| Gateway Failures | Usually fixed once infrastructure is healthy |

---

## 🎯 **SUCCESS CHECKLIST**

Before declaring victory, verify:

- [ ] ✅ All AWS services accessible (RDS, S3, Bedrock)
- [ ] ✅ Database tables created and populated
- [ ] ✅ Python environment with all dependencies
- [ ] ✅ Configuration files contain real AWS details
- [ ] ✅ FLOW agreements system operational
- [ ] ✅ AI processing works end-to-end
- [ ] ✅ Cache system functional
- [ ] ✅ Diagnostic script shows 100% pass rate
- [ ] ✅ Timestamped reports generated

**🏁 FINAL TEST**: `python run_full_diagnostics.py` returns "READY FOR CLIENT DEMO!"

---

*This completes the systematic IKEA-style setup. Each step builds on the previous one, and you can't proceed until the current step passes. No shortcuts, no assumptions - everything is verified before moving forward.*