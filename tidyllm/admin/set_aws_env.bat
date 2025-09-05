@echo off
REM TidyLLM AWS Environment Setup
REM Sets AWS credentials for current session

echo Setting TidyLLM AWS credentials...

set AWS_ACCESS_KEY_ID=REMOVED_AWS_KEY
set AWS_SECRET_ACCESS_KEY=REMOVED_AWS_SECRET
set AWS_DEFAULT_REGION=us-east-1

echo AWS credentials set for current session
echo Access Key: %AWS_ACCESS_KEY_ID:~0,10%...
echo Region: %AWS_DEFAULT_REGION%

REM Test S3 connectivity
echo.
echo Testing S3 connectivity...
python -c "import boto3; print('S3 buckets:', len(boto3.client('s3').list_buckets()['Buckets']))" 2>nul && echo S3 connection successful || echo S3 connection failed

echo.
echo To use these credentials, run this script before starting TidyLLM systems:
echo   call tidyllm\admin\set_aws_env.bat
echo   python scripts\production_tracking_drop_zones.py