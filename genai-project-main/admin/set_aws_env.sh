#!/bin/bash
# TidyLLM AWS Environment Setup
# Sets AWS credentials for current session

echo "Setting TidyLLM AWS credentials..."

export AWS_ACCESS_KEY_ID=REMOVED_AWS_KEY
export AWS_SECRET_ACCESS_KEY=REMOVED_AWS_SECRET
export AWS_DEFAULT_REGION=us-east-1

echo "AWS credentials set for current session"
echo "Access Key: ${AWS_ACCESS_KEY_ID:0:10}..."
echo "Region: $AWS_DEFAULT_REGION"

# Test S3 connectivity
echo ""
echo "Testing S3 connectivity..."
python3 -c "import boto3; print('S3 buckets:', len(boto3.client('s3').list_buckets()['Buckets']))" 2>/dev/null && echo "S3 connection successful" || echo "S3 connection failed"

echo ""
echo "To use these credentials, source this script before starting TidyLLM systems:"
echo "  source tidyllm/admin/set_aws_env.sh"
echo "  python scripts/production_tracking_drop_zones.py"