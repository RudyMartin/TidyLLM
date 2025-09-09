# 🚀 TidyLLM Quick Start Guide

**Simple 5-minute setup for TidyLLM v1.0.4**

## Prerequisites
- Python 3.8+ installed
- AWS credentials (for S3 and Bedrock access)
- Git (to clone the repository)

## Easy Setup Options

### Option 1: Windows Batch Script (Recommended)
```cmd
setup_tidyllm.bat
```

### Option 2: Python Script
```bash
python tidyllm/quick_setup.py
```

### Option 3: Manual Steps
```bash
# Step 1: Set AWS credentials
tidyllm/admin/set_aws_env.bat

# Step 2: Test configuration  
python tidyllm/admin/test_config.py

# Step 3: Try TidyLLM
python -c "import tidyllm; print(tidyllm.chat('Hello!'))"
```

## What Gets Set Up

The setup process:
1. ✅ Configures AWS credentials using existing admin tools
2. ✅ Tests your configuration automatically
3. ✅ Verifies essential connectivity (S3, imports)
4. ✅ Shows you next steps and available tools

## What It Sets Up

### Architecture: 4-Gateway 2-Service Design
```
CorporateLLM → AIProcessing → WorkflowOptimizer → Database
     +              +
UnifiedSessionManager + DomainRAG
```

### Key Components Tested:
- **AWS S3**: Document storage and processing
- **AWS Bedrock**: AI model access  
- **TidyLLM Core**: Basic functionality
- **Admin Tools**: Configuration management

## After Setup

### Try TidyLLM:
```python
import tidyllm
response = tidyllm.chat("Hello, how are you?")
print(response)
```

### Configuration:
- Main config: `tidyllm/admin/settings.yaml`
- AWS setup: `tidyllm/admin/set_aws_env.bat`

### Admin Tools:
```bash
# Full configuration test
python tidyllm/admin/test_config.py

# Detailed diagnostics
python tidyllm/admin/run_diagnostics_real.py

# Reset AWS credentials
tidyllm/admin/set_aws_env.bat
```

## Troubleshooting

### AWS Issues:
```bash
# Re-run AWS setup
tidyllm/admin/set_aws_env.bat

# Check S3 access
python -c "import boto3; print(boto3.client('s3').list_buckets())"
```

### Import Issues:
```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Try from TidyLLM directory
cd tidyllm
python -c "import tidyllm; print('Success!')"
```

### Configuration Issues:
```bash
# Run full diagnostics
python tidyllm/admin/run_diagnostics_real.py

# Check settings file
cat tidyllm/admin/settings.yaml
```

## Architecture Constraints

⚠️ **IMPORTANT**: Before modifying any files, read:
`docs/2025-09-08/IMPORTANT-CONSTRAINTS-FOR-THIS-CODEBASE.md`

Key principles:
- **S3-First**: All processing happens S3 → S3 (no local files)
- **Gateway Chain**: All AI requests go through 4-gateway chain
- **Admin-First**: Use `tidyllm/admin/` tools, don't create alternatives
- **No Big Tech ML**: Use `tidyllm.tlm` and `tidyllm_sentence` instead

## Next Steps

1. **Try the Examples**: 
   - Chat: `tidyllm.chat("Tell me about AI")`
   - Document: `tidyllm.process_document("file.pdf")`
   
2. **Explore Admin Tools**:
   - `tidyllm/admin/` folder has all configuration utilities
   
3. **Read Architecture Docs**:
   - Start with `IMPORTANT-CONSTRAINTS-FOR-THIS-CODEBASE.md`
   
4. **Corporate Deployment**:
   - Use the Streamlit onboarding app in `onboarding/` folder

---

🎯 **Need Help?** Check the admin tools first - they have diagnostics and troubleshooting built-in.