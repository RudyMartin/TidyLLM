# VectorQA Sage Deployment Guide

## 🚀 Quick Start

This bundle contains everything needed to deploy VectorQA Sage to staging.

### Prerequisites

- Python 3.8+
- pip
- Internet connection (for package installation)

### Deployment Steps

1. **Extract the bundle:**
   ```bash
   unzip test_bundle_fixed.zip
   cd test_bundle_fixed
   ```

2. **Validate configuration:**
   ```bash
   python scripts/validate_config.py
   ```

3. **Setup environment:**
   ```bash
   python scripts/setup_environment.py
   ```

4. **Launch application:**
   
   **Option A: Use environment-specific launcher (Recommended)**
   ```bash
   # Auto-detect environment
   python launchers/launch.py
   
   # Or specify environment
   python launchers/launch_production.py
   python launchers/launch_staging.py
   python launchers/launch_development.py
   python launchers/launch_local.py
   ```
   
   **Option B: Use deployment script**
   ```bash
   python scripts/deploy.py
   ```

### Health Checks

Monitor the deployment:
```bash
python scripts/health_check.py
```

### Configuration

The bundle includes environment-specific settings files in the `settings/` directory:

- `local_settings.py` - Local development configuration
- `development_settings.py` - Development environment configuration  
- `staging_settings.py` - Staging environment configuration
- `production_settings.py` - Production environment configuration
- `settings_loader.py` - Dynamic settings loader

To use environment-specific settings:
```python
from settings.settings_loader import load_environment_settings
settings = load_environment_settings('production')
```

You can also edit `deployment_config.json` to customize:
- Environment variables
- Streamlit settings
- Backend configuration

### Account Migration (Individual → Service Accounts)

🚨 **CRITICAL**: Production deployments require service account migration!

#### Pre-Migration Checklist:
- [ ] Create service accounts for each environment
- [ ] Generate service account credentials
- [ ] Update access permissions
- [ ] Test service account access
- [ ] Plan rollback strategy

#### AWS Service Account Setup:
```bash
# Create IAM role for service account
aws iam create-role --role-name vectorqa-service-role --assume-role-policy-document file://trust-policy.json

# Attach required policies
aws iam attach-role-policy --role-name vectorqa-service-role --policy-arn arn:aws:iam::aws:policy/AmazonBedrockFullAccess

# Create service account user
aws iam create-user --user-name vectorqa-service-user

# Attach role to user
aws iam attach-user-policy --user-name vectorqa-service-user --policy-arn arn:aws:iam::aws:policy/AmazonBedrockFullAccess
```

#### GCP Service Account Setup:
```bash
# Create service account
gcloud iam service-accounts create vectorqa-service --display-name="VectorQA Service Account"

# Grant necessary roles
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID     --member="serviceAccount:vectorqa-service@YOUR_PROJECT_ID.iam.gserviceaccount.com"     --role="roles/aiplatform.user"

# Generate service account key
gcloud iam service-accounts keys create service-account-key.json     --iam-account=vectorqa-service@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

#### Migration Steps:
1. **Backup current credentials**
2. **Create service accounts** (see above)
3. **Update credential files** with service account keys
4. **Test with service accounts** in staging
5. **Deploy to production** with service accounts
6. **Monitor and validate** service account access
7. **Remove individual user access** (after validation)

#### Security Considerations:
- Service accounts should have minimal required permissions
- Rotate service account keys regularly
- Monitor service account usage
- Use cloud provider secret management (AWS Secrets Manager, GCP Secret Manager)

### Troubleshooting

1. **Port conflicts:** Change port in `deployment_config.json`
2. **Dependencies:** Run `pip install -r requirements_demo.txt`
3. **Permissions:** Ensure scripts are executable

### Support

For issues, check:
- Application logs
- Health check results
- Configuration validation

---
Generated: 2025-08-23 20:12:42
