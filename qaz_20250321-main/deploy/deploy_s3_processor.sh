#!/bin/bash

# MVR Review S3 File Processor Deployment Script
# This script packages and deploys the S3 file processor Lambda function

set -e

# Configuration
STACK_NAME="mvr-s3-processor"
REGION="us-east-1"
LAMBDA_FUNCTION_NAME="mvr-s3-processor"
PACKAGE_DIR="lambda_package"

echo "🚀 Starting MVR S3 File Processor Deployment"
echo "=============================================="

# Create package directory
echo "📦 Creating Lambda package..."
rm -rf $PACKAGE_DIR
mkdir -p $PACKAGE_DIR

# Copy Lambda function
echo "📋 Copying Lambda function..."
cp src/backend/s3_file_processor.py $PACKAGE_DIR/

# Copy configuration files
echo "⚙️ Copying configuration files..."
cp dev_configs/file_classification.yaml $PACKAGE_DIR/

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt -t $PACKAGE_DIR/ --no-deps

# Create deployment package
echo "📦 Creating deployment package..."
cd $PACKAGE_DIR
zip -r ../lambda_deployment.zip .
cd ..

# Deploy CloudFormation stack
echo "☁️ Deploying CloudFormation stack..."
aws cloudformation deploy \
    --template-file deploy/s3_processor_lambda.yaml \
    --stack-name $STACK_NAME \
    --capabilities CAPABILITY_IAM \
    --region $REGION \
    --parameter-overrides \
        LandingBucketName="mvr-landing-bucket-$(date +%Y%m%d)" \
        CatalogBucketName="mvr-catalog-bucket-$(date +%Y%m%d)" \
        QuarantineBucketName="mvr-quarantine-bucket-$(date +%Y%m%d)" \
        NotificationQueueName="mvr-file-notifications-$(date +%Y%m%d)"

# Update Lambda function code
echo "🔄 Updating Lambda function code..."
FUNCTION_NAME=$(aws cloudformation describe-stacks \
    --stack-name $STACK_NAME \
    --region $REGION \
    --query 'Stacks[0].Outputs[?OutputKey==`LambdaFunctionArn`].OutputValue' \
    --output text)

aws lambda update-function-code \
    --function-name $FUNCTION_NAME \
    --zip-file fileb://lambda_deployment.zip \
    --region $REGION

# Clean up
echo "🧹 Cleaning up..."
rm -rf $PACKAGE_DIR
rm lambda_deployment.zip

# Get stack outputs
echo "📊 Stack Outputs:"
aws cloudformation describe-stacks \
    --stack-name $STACK_NAME \
    --region $REGION \
    --query 'Stacks[0].Outputs' \
    --output table

echo ""
echo "✅ Deployment Complete!"
echo "======================"
echo "Landing Bucket: Upload files here to trigger processing"
echo "Catalog Bucket: Valid files are moved here with Review ID organization"
echo "Quarantine Bucket: Invalid files are moved here for review"
echo "SQS Queue: Processing notifications are sent here"
echo ""
echo "To test the deployment:"
echo "1. Upload a file to the landing bucket"
echo "2. Check CloudWatch logs for processing results"
echo "3. Monitor the SQS queue for notifications"


