#!/bin/bash

# 🔒 VectorQA Sage - AWS-Only Deployment Script
# This script deploys the application in AWS-only security mode

set -e  # Exit on any error

echo "🔒 Deploying VectorQA Sage in AWS-Only Security Mode"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running in AWS environment
check_aws_environment() {
    print_status "Checking AWS environment..."
    
    if [[ -n "$AWS_EXECUTION_ENV" || -n "$AWS_BATCH_JOB_ID" || -n "$ECS_CONTAINER_METADATA_URI" ]]; then
        print_success "Running in AWS environment"
        return 0
    else
        print_warning "Not running in AWS environment"
        print_warning "IAM role validation will be limited"
        return 1
    fi
}

# Set security environment variables
set_security_variables() {
    print_status "Setting security environment variables..."
    
    export AWS_ONLY_MODE=true
    export ALLOW_EXTERNAL_APIS=false
    export REQUIRE_IAM_ROLES=true
    export AUDIT_LOGGING=true
    
    print_success "Security variables set"
}

# Validate AWS credentials
validate_aws_credentials() {
    print_status "Validating AWS credentials..."
    
    if aws sts get-caller-identity > /dev/null 2>&1; then
        IDENTITY=$(aws sts get-caller-identity --query 'Arn' --output text)
        print_success "AWS credentials valid: $IDENTITY"
        return 0
    else
        print_error "AWS credentials validation failed"
        return 1
    fi
}

# Validate Bedrock access
validate_bedrock_access() {
    print_status "Validating Bedrock access..."
    
    if aws bedrock list-foundation-models --region us-east-1 > /dev/null 2>&1; then
        MODEL_COUNT=$(aws bedrock list-foundation-models --region us-east-1 --query 'length(modelSummaries)' --output text)
        print_success "Bedrock access valid: $MODEL_COUNT models available"
        return 0
    else
        print_error "Bedrock access validation failed"
        return 1
    fi
}

# Check for external API keys
check_external_keys() {
    print_status "Checking for external API keys..."
    
    EXTERNAL_KEYS=("OPENAI_API_KEY" "COHERE_API_KEY" "GOOGLE_API_KEY" "HUGGINGFACE_API_KEY" "ANTHROPIC_API_KEY")
    FOUND_KEYS=()
    
    for key in "${EXTERNAL_KEYS[@]}"; do
        if [[ -n "${!key}" ]]; then
            FOUND_KEYS+=("$key")
        fi
    done
    
    if [[ ${#FOUND_KEYS[@]} -eq 0 ]]; then
        print_success "No external API keys found"
        return 0
    else
        print_warning "External API keys found: ${FOUND_KEYS[*]}"
        print_warning "These should be removed for AWS-only mode"
        return 1
    fi
}

# Run security validation
run_security_validation() {
    print_status "Running security validation..."
    
    if python scripts/validate_security.py; then
        print_success "Security validation passed"
        return 0
    else
        print_error "Security validation failed"
        return 1
    fi
}

# Deploy application
deploy_application() {
    print_status "Deploying application..."
    
    # Check if we're in the right directory
    if [[ ! -f "src/run_app.py" ]]; then
        print_error "Application not found. Please run from project root directory."
        return 1
    fi
    
    # Start the application
    print_status "Starting VectorQA Sage application..."
    python src/run_app.py
}

# Main deployment function
main() {
    print_status "Starting AWS-Only deployment process..."
    
    # Step 1: Check AWS environment
    check_aws_environment
    
    # Step 2: Set security variables
    set_security_variables
    
    # Step 3: Validate AWS credentials
    if ! validate_aws_credentials; then
        print_error "AWS credentials validation failed. Exiting."
        exit 1
    fi
    
    # Step 4: Validate Bedrock access
    if ! validate_bedrock_access; then
        print_error "Bedrock access validation failed. Exiting."
        exit 1
    fi
    
    # Step 5: Check for external API keys
    check_external_keys
    
    # Step 6: Run security validation
    if ! run_security_validation; then
        print_error "Security validation failed. Exiting."
        exit 1
    fi
    
    # Step 7: Deploy application
    print_success "All validations passed. Deploying application..."
    deploy_application
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --validate     Run validation only (don't deploy)"
        echo "  --deploy       Deploy application (default)"
        echo ""
        echo "This script deploys VectorQA Sage in AWS-only security mode."
        exit 0
        ;;
    --validate)
        print_status "Running validation only..."
        set_security_variables
        validate_aws_credentials
        validate_bedrock_access
        check_external_keys
        run_security_validation
        print_success "Validation complete"
        exit 0
        ;;
    --deploy|"")
        main
        ;;
    *)
        print_error "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
esac
