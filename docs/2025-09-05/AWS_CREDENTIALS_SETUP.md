# TidyLLM AWS Credentials Setup

**Status**: ✅ **WORKING** (Successfully tested with S3 connectivity)

## Quick Setup

### For Windows (Command Prompt/PowerShell):
```cmd
call tidyllm\admin\set_aws_env.bat
```

### For Linux/Mac/Git Bash:
```bash
source tidyllm/admin/set_aws_env.sh
```

## Manual Setup (if scripts fail):

### Windows:
```cmd
set AWS_ACCESS_KEY_ID=***REMOVED***
set AWS_SECRET_ACCESS_KEY=***REMOVED***
set AWS_DEFAULT_REGION=us-east-1
```

### Linux/Mac/Git Bash:
```bash
export AWS_ACCESS_KEY_ID=***REMOVED***
export AWS_SECRET_ACCESS_KEY=***REMOVED***
export AWS_DEFAULT_REGION=us-east-1
```

## Verify Setup

Test AWS credentials are working:
```python
python -c "import boto3; print('S3 buckets:', boto3.client('s3').list_buckets())"
```

Expected output:
```
S3 buckets: {'Buckets': [...], 'Owner': {...}}
```

## Available S3 Buckets

- **nsc-mvp1** (configured in settings.yaml)
- **dsai-2025-asu**  
- **sagemaker-us-east-1-188494237500**

## Integration Status

✅ **Configuration Test Results (with credentials):**
- **Success Rate**: 92.3% (12 PASS, 0 FAIL, 1 WARN)
- **S3 Connection**: PASS - S3 access confirmed
- **AWS Credentials**: PASS - Found in environment  
- **PostgreSQL**: PASS - Database connection successful
- **MLflow**: PASS - MLflow configured and working

## Usage with Drop Zones

### Start with S3 Integration:
```bash
# Set credentials first
source tidyllm/admin/set_aws_env.sh

# Start drop zone monitoring
python scripts/production_tracking_drop_zones.py
```

The system will now show:
- **STEP 5: CLOUD STORAGE** → **SUCCESS** (instead of SKIPPED)
- S3 uploads to `s3://nsc-mvp1/dropzones/[timestamp]/[doc_type]/`
- Real AWS operations with ETags and metadata

## Troubleshooting

### "S3 upload not available"
- Ensure AWS credentials are set in the same terminal session
- Verify with: `echo $AWS_ACCESS_KEY_ID` (should show AKIASXYZBZ...)
- Test with: `aws s3 ls` (if AWS CLI installed)

### "Credentials not configured"
- AWS credentials only persist in the current session
- Re-run the setup script in each new terminal
- For permanent setup, add to your shell profile (.bashrc, .zshrc)

### "Access Denied" errors
- Credentials are correct but may have limited permissions
- Current permissions confirmed for: ListBucket, PutObject, GetObject

---

**AWS Infrastructure**: ✅ **OPERATIONAL**  
**Last Tested**: September 5, 2025  
**Total S3 Buckets**: 3 accessible  
**Primary Bucket**: nsc-mvp1