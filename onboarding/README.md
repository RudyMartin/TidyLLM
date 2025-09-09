# TidyLLM Corporate Onboarding

Complete onboarding solution for deploying TidyLLM in corporate environments with enterprise-grade security and compliance.

## 📋 What's Included

This package provides everything you need to configure and deploy TidyLLM in your corporate environment:

### Core Files
- `template.settings.yaml` - Corporate configuration template with security defaults
- `config_generator.py` - Generates customized configurations from user input
- `session_validator.py` - Validates AWS connectivity and corporate network requirements
- `cli_onboarding.py` - Interactive CLI wizard
- `streamlit_app.py` - Web-based GUI for configuration

## 🚀 Quick Start

### Option 1: Interactive CLI Wizard
```bash
# Run the interactive wizard
python cli_onboarding.py

# Generate config only (no wizard)
python cli_onboarding.py --config-only

# Validate existing setup
python cli_onboarding.py --validate
```

### Option 2: Web Interface
```bash
# Install Streamlit first
pip install streamlit pandas

# Launch web interface
streamlit run streamlit_app.py
```

## 📦 Features

### 🏢 Corporate-Ready Defaults
- **Security**: Encryption at rest, audit logging, data masking
- **Authentication**: SSO integration (Okta, Azure AD, SAML)
- **Compliance**: GDPR, SOX, HIPAA considerations
- **Rate Limiting**: Corporate-appropriate API limits
- **Network**: Proxy support, corporate CA certificates

### 🔧 Configuration Management
- **Template-Based**: Start with secure corporate defaults
- **Validation**: Test AWS connectivity, database access, network requirements
- **Deployment**: Generate Docker, Kubernetes, and Docker Compose files
- **Environment**: Secure credential management via environment variables

### 🎯 Deployment Options
- **Docker Compose**: Development and staging environments
- **Kubernetes**: Production container orchestration
- **Standalone**: Single-server deployments

## 📋 Requirements

### Required Environment Variables
```bash
# Database (Required)
TIDYLLM_DB_PASSWORD=your_secure_password

# AWS Credentials (Required if not using IAM roles)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1

# Optional: Session token for temporary credentials
AWS_SESSION_TOKEN=your_session_token
```

### Network Requirements
Your corporate network must allow outbound HTTPS to:
- `bedrock-runtime.{region}.amazonaws.com:443` (AWS Bedrock)
- `s3.{region}.amazonaws.com:443` (S3 Storage)
- Your PostgreSQL database host on port 5432

### AWS Permissions
The AWS credentials need these IAM permissions:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:ListFoundationModels",
        "s3:GetObject",
        "s3:PutObject", 
        "s3:ListBucket",
        "kms:Encrypt",
        "kms:Decrypt"
      ],
      "Resource": "*"
    }
  ]
}
```

## 🛠️ Installation

### Python Dependencies
```bash
# Required
pip install pyyaml boto3 psycopg2-binary

# Optional (for CLI colors)
pip install rich

# Optional (for web interface)  
pip install streamlit pandas
```

### Quick Setup
```bash
# Clone or download this onboarding folder
cd onboarding

# Set your credentials
export AWS_ACCESS_KEY_ID="your_key"
export AWS_SECRET_ACCESS_KEY="your_secret"
export AWS_DEFAULT_REGION="us-east-1"
export TIDYLLM_DB_PASSWORD="your_db_password"

# Run the wizard
python cli_onboarding.py
```

## 📖 Usage Guide

### Step 1: Organization Setup
- Enter your organization name
- Choose deployment environment (production/staging/development)
- Select AWS region

### Step 2: Security Configuration
- Configure Single Sign-On (SSO)
- Set up audit logging
- Enable data encryption
- Configure rate limiting

### Step 3: AI Model Selection
- Choose default AI model (Claude 3 variants)
- Set model parameters (tokens, temperature)
- Configure corporate model aliases

### Step 4: Environment Validation
- Test environment variables
- Validate AWS connectivity
- Check network requirements
- Verify permissions

### Step 5: Generate Configuration
- Create complete settings.yaml
- Generate deployment files
- Create environment templates

### Step 6: Download & Deploy
- Download configuration package
- Extract to deployment server
- Edit .env with real credentials
- Deploy using provided scripts

## 🔧 Generated Files

After running the wizard, you'll get:

```
tidyllm-corporate-config/
├── settings.yaml              # Main TidyLLM configuration
├── .env.template             # Environment variables template
├── Dockerfile                # Container image definition
├── docker-compose.yml        # Local deployment
├── kubernetes/               # K8s manifests
│   ├── deployment.yaml
│   └── service.yaml
└── DEPLOYMENT_INSTRUCTIONS.md # Step-by-step deployment guide
```

## 🚀 Deployment Examples

### Docker Compose (Recommended for Development)
```bash
cd tidyllm-corporate-config
cp .env.template .env
# Edit .env with your actual credentials
docker-compose up -d
```

### Kubernetes (Production)
```bash
cd tidyllm-corporate-config
kubectl create namespace yourcompany
kubectl create secret generic tidyllm-secrets --from-env-file=.env --namespace=yourcompany
kubectl apply -f kubernetes/ --namespace=yourcompany
```

### Docker (Standalone)
```bash
cd tidyllm-corporate-config
docker build -t yourcompany/tidyllm:latest .
docker run -d --name tidyllm-corporate --env-file .env -p 8000:8000 yourcompany/tidyllm:latest
```

## 🔍 Validation & Testing

### Health Check
```bash
curl http://localhost:8000/health
# Should return: {"status": "healthy"}
```

### API Test
```bash
curl -X POST http://localhost:8000/api/v1/chat \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer your-token" \
     -d '{"message": "Hello, TidyLLM!"}'
```

### Environment Validation
```bash
python cli_onboarding.py --validate
```

## 🏢 Corporate Security Features

### Data Protection
- **Encryption at Rest**: All cached data and logs encrypted
- **Data Masking**: Automatic PII masking in logs
- **Audit Logging**: Complete request/response audit trail
- **Retention Policies**: Configurable data retention periods

### Access Control
- **SSO Integration**: Support for major enterprise identity providers
- **Rate Limiting**: Prevent API abuse and ensure fair usage
- **Authentication**: Multiple auth methods (OAuth2, LDAP, API keys)
- **Authorization**: Role-based access control

### Network Security
- **Proxy Support**: Works with corporate HTTP proxies
- **SSL/TLS**: Custom CA certificate support
- **Network Restrictions**: Configurable endpoint whitelisting
- **VPC Support**: Works within private networks

### Compliance
- **GDPR Ready**: Data privacy and right-to-delete support
- **SOX Compliant**: Financial data handling best practices
- **HIPAA Considerations**: Healthcare data protection features
- **Audit Trails**: Complete activity logging for compliance

## 🆘 Troubleshooting

### Common Issues

**1. AWS Permission Errors**
```
Error: User is not authorized to perform: bedrock:InvokeModel
```
**Solution**: Add required IAM permissions to your AWS user/role.

**2. Database Connection Issues**
```  
Error: could not connect to server: Connection refused
```
**Solution**: Check database host, port, and network connectivity.

**3. Corporate Proxy Issues**
```
Error: ProxyError or SSL verification failed
```
**Solution**: Configure proxy settings and corporate CA certificates.

**4. Model Not Available**
```
Error: Could not find model: anthropic.claude-3-sonnet
```
**Solution**: Verify model availability in your AWS region.

### Getting Help
- **Documentation**: Comprehensive deployment guide included
- **Validation**: Built-in connectivity and permission testing
- **Support**: Contact your IT administrator or TidyLLM support

## 📄 License

This onboarding package is part of TidyLLM and follows the same licensing terms.

## 🤝 Support

For corporate deployment assistance:
- **Documentation**: See generated DEPLOYMENT_INSTRUCTIONS.md
- **Issues**: Contact your IT administrator
- **Enterprise Support**: Available for corporate customers

---

**Generated by TidyLLM Corporate Onboarding v1.0**