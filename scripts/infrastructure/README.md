# Infrastructure Scripts

AWS, database, and system infrastructure management tools for TidyLLM platform deployment.

## AWS & Session Management

### **restart_aws_session.py**
AWS session restart utility:
- Refreshes AWS credentials and sessions
- Validates S3 bucket accessibility
- Restarts database connections
- Ensures clean infrastructure state

### **start_production_with_aws.py**
Production environment launcher:
- Initializes production AWS configuration
- Sets up unified session management
- Validates all infrastructure components
- Launches production-ready TidyLLM services

### **start_unified_sessions.py**
Unified session manager initialization:
- Consolidates AWS, PostgreSQL, and Bedrock connections
- Implements single source of truth for credentials
- Manages connection pooling and lifecycle
- Provides centralized session health monitoring

### **verify_aws_access.py**
AWS access verification tool:
- Tests S3 bucket permissions and connectivity  
- Validates IAM role configurations
- Checks AWS credential validity
- Reports on available AWS resources

## Database & Storage

### **check_s3_papers.py**
S3 paper repository health check:
- Validates paper storage integrity
- Checks S3 object metadata
- Reports on paper processing status
- Identifies missing or corrupted documents

### **check_workflow_tables.py**
Database workflow table verification:
- Validates workflow table schemas
- Checks data integrity constraints
- Reports on workflow execution history
- Identifies table relationship issues

### **create_workflow_tables.py**
Database workflow table initialization:
- Creates necessary workflow tables
- Sets up proper indexes and constraints
- Initializes default workflow data
- Ensures database schema compliance

### **setup_mlflow_postgres.py**
MLFlow PostgreSQL integration setup:
- Configures MLFlow with PostgreSQL backend
- Creates MLFlow tracking database schemas
- Sets up experiment tracking infrastructure
- Validates MLFlow database connectivity

## Services & Diagnostics

### **run_diagnostics.py**
Comprehensive system diagnostics:
- Tests all infrastructure components
- Generates detailed health reports
- Identifies configuration issues
- Provides troubleshooting recommendations

### **tidyllm_services.py**
TidyLLM service management:
- Starts/stops TidyLLM services
- Manages service dependencies
- Provides service health monitoring
- Handles graceful service shutdowns

### **tidyllm_unified_services.py**
Unified TidyLLM service orchestration:
- Coordinates multiple TidyLLM components
- Manages inter-service communication
- Provides centralized service logging
- Handles service discovery and routing

## Configuration & Setup

### **enhanced_drop_zones_with_cleanup.py**
Document drop zone management:
- Sets up S3 document processing zones
- Manages automated cleanup policies
- Configures document lifecycle rules
- Monitors drop zone health and capacity

### **list_available_buckets.py**
AWS S3 bucket discovery:
- Lists accessible S3 buckets
- Reports bucket permissions and policies
- Identifies available storage resources
- Validates bucket configuration compliance

### **unified_credential_setup.py**
Centralized credential configuration:
- Sets up unified credential management
- Configures secure credential storage
- Validates credential access patterns
- Ensures credential security best practices

## Usage

### Production Deployment
```bash
python scripts/infrastructure/start_production_with_aws.py
```

### System Health Check
```bash
python scripts/infrastructure/run_diagnostics.py
```

### AWS Setup Validation
```bash
python scripts/infrastructure/verify_aws_access.py
python scripts/infrastructure/list_available_buckets.py
```

These scripts provide the foundation for reliable TidyLLM infrastructure deployment and management.