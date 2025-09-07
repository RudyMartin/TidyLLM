# Maintenance Scripts

System maintenance and health check utilities for the TidyLLM platform.

## Scripts

### **check_current_setup.py**
System health checker that validates:
- Database connectivity (PostgreSQL)
- AWS S3 access and permissions
- TidyLLM module imports
- Configuration file integrity
- Overall system readiness

### **embedding_verification.py**
Database embedding verification tool:
- Validates document_chunks table integrity
- Checks embedding dimensions and quality
- Reports on embedding coverage statistics
- Identifies missing or corrupted embeddings

### **final_credential_cleanup.py**
Security cleanup utility:
- Removes hardcoded credentials from code
- Validates credential management patterns
- Ensures secure credential storage practices
- Reports potential security issues

### **fix_hardcoded_credentials.py**
Automated credential security fixer:
- Scans codebase for hardcoded AWS keys
- Replaces with environment variable references
- Updates imports for centralized credential loading
- Maintains backup of original files

### **fix_hardcoded_s3_paths.py**
Infrastructure path standardization:
- Replaces hardcoded S3 bucket references
- Updates to use configuration-based paths
- Ensures consistent S3 bucket usage
- Validates S3 path accessibility

## Usage

Run maintenance scripts individually:
```bash
python scripts/maintenance/check_current_setup.py
python scripts/maintenance/embedding_verification.py
```

Or as part of system health checks before deployments.