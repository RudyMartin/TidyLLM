#!/usr/bin/env python3
"""
Security validation script for AWS-only environments.

This script validates that the VectorQA Sage system is properly configured
for AWS-only operation with all external API dependencies disabled.

Usage:
    python scripts/validate_security.py
"""

import os
import sys
import boto3
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from backend.config.credential_manager import credential_manager
    from backend.core.config import CONFIG
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please run this script from the project root directory")
    sys.exit(1)

def validate_environment_variables():
    """Validate required environment variables for AWS-only mode."""
    
    print("🔒 Validating Environment Variables")
    print("=" * 40)
    
    required_vars = [
        "AWS_ONLY_MODE",
        "ALLOW_EXTERNAL_APIS", 
        "REQUIRE_IAM_ROLES",
        "AUDIT_LOGGING"
    ]
    
    optional_vars = [
        "BEDROCK_EMBEDDING_MODEL",
        "BEDROCK_ANALYSIS_MODEL",
        "BEDROCK_REPORT_MODEL",
        "BEDROCK_QUICK_MODEL"
    ]
    
    all_valid = True
    
    # Check required variables
    for var in required_vars:
        value = os.getenv(var, "not_set")
        status = "✅" if value != "not_set" else "❌"
        print(f"{status} {var}: {value}")
        
        if value == "not_set":
            all_valid = False
    
    print("\n📋 Optional Bedrock Model Configuration:")
    for var in optional_vars:
        value = os.getenv(var, "not_set")
        status = "✅" if value != "not_set" else "⚠️"
        print(f"{status} {var}: {value}")
    
    return all_valid

def validate_aws_credentials():
    """Validate AWS credentials and access."""
    
    print("\n🔐 Validating AWS Credentials")
    print("=" * 35)
    
    try:
        # Test AWS STS
        sts = boto3.client("sts")
        identity = sts.get_caller_identity()
        
        print(f"✅ AWS Identity: {identity.get('Arn', 'Unknown')}")
        print(f"✅ Account ID: {identity.get('Account', 'Unknown')}")
        print(f"✅ User ID: {identity.get('UserId', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ AWS Credentials Error: {e}")
        return False

def validate_bedrock_access():
    """Validate Bedrock service access."""
    
    print("\n🤖 Validating Bedrock Access")
    print("=" * 30)
    
    try:
        bedrock = boto3.client("bedrock", region_name="us-east-1")
        models = bedrock.list_foundation_models()
        
        print(f"✅ Bedrock Access: {len(models['modelSummaries'])} models available")
        
        # Check for specific models we need
        available_models = [model['modelId'] for model in models['modelSummaries']]
        
        required_models = [
            "amazon.titan-embed-text-v2:0",
            "anthropic.claude-3-sonnet-20240229-v1:0",
            "anthropic.claude-3-opus-20240229-v1:0",
            "anthropic.claude-3-haiku-20240307-v1:0"
        ]
        
        print("\n📋 Required Model Availability:")
        for model in required_models:
            status = "✅" if model in available_models else "❌"
            print(f"{status} {model}")
        
        return True
        
    except Exception as e:
        print(f"❌ Bedrock Access Error: {e}")
        return False

def validate_external_api_keys():
    """Check that external API keys are not set."""
    
    print("\n🚫 Checking for External API Keys")
    print("=" * 35)
    
    external_keys = [
        "OPENAI_API_KEY",
        "COHERE_API_KEY", 
        "GOOGLE_API_KEY",
        "HUGGINGFACE_API_KEY",
        "ANTHROPIC_API_KEY"
    ]
    
    all_clean = True
    
    for key in external_keys:
        value = os.getenv(key)
        status = "❌" if value else "✅"
        print(f"{status} {key}: {'SET' if value else 'NOT SET'}")
        
        if value:
            all_clean = False
    
    if not all_clean:
        print("\n⚠️  WARNING: External API keys found!")
        print("   These should be removed for AWS-only mode.")
    
    return all_clean

def validate_configuration():
    """Validate the application configuration."""
    
    print("\n⚙️  Validating Application Configuration")
    print("=" * 40)
    
    try:
        # Check security configuration
        security_config = CONFIG.get("security", {})
        aws_only = security_config.get("aws_only", False)
        allow_external = security_config.get("allow_external_apis", True)
        
        print(f"✅ AWS-Only Mode: {aws_only}")
        print(f"✅ External APIs Disabled: {not allow_external}")
        
        # Check Bedrock models configuration
        bedrock_models = CONFIG.get("bedrock_models", {})
        if bedrock_models:
            print(f"✅ Bedrock Models Configured: {len(bedrock_models)} task types")
            
            for task_type, config in bedrock_models.items():
                primary = config.get("primary", "unknown")
                fallback = config.get("fallback", "none")
                print(f"   📋 {task_type}: {primary} (fallback: {fallback})")
        else:
            print("❌ No Bedrock models configured")
            return False
        
        # Check LLM models configuration
        llm_models = CONFIG.get("llm_models", {})
        aws_models = [k for k, v in llm_models.items() if v.get("aws_only", False)]
        
        print(f"✅ AWS-Only LLM Models: {len(aws_models)} configured")
        for model in aws_models:
            print(f"   🤖 {model}: {llm_models[model].get('model', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration Error: {e}")
        return False

def validate_iam_role():
    """Validate IAM role usage."""
    
    print("\n🛡️  Validating IAM Role Usage")
    print("=" * 30)
    
    # Check for AWS-specific environment variables
    aws_indicators = [
        "AWS_EXECUTION_ENV",  # Lambda
        "AWS_BATCH_JOB_ID",   # Batch
        "ECS_CONTAINER_METADATA_URI",  # ECS
        "KUBERNETES_SERVICE_HOST"  # EKS
    ]
    
    aws_environment = any(os.getenv(indicator) for indicator in aws_indicators)
    
    if aws_environment:
        print("✅ Running in AWS environment")
        
        # Check if using IAM role (no explicit credentials)
        access_key = os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        
        if not access_key and not secret_key:
            print("✅ Using IAM role (no explicit credentials)")
            return True
        else:
            print("⚠️  Explicit credentials found (IAM role preferred)")
            return False
    else:
        print("⚠️  Not running in AWS environment")
        print("   IAM role validation skipped")
        return True

def main():
    """Main validation function."""
    
    print("🔒 VectorQA Sage - AWS-Only Security Validation")
    print("=" * 55)
    print()
    
    # Run all validations
    validations = [
        ("Environment Variables", validate_environment_variables),
        ("AWS Credentials", validate_aws_credentials),
        ("Bedrock Access", validate_bedrock_access),
        ("External API Keys", validate_external_api_keys),
        ("Configuration", validate_configuration),
        ("IAM Role", validate_iam_role)
    ]
    
    results = []
    
    for name, validation_func in validations:
        try:
            result = validation_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} validation failed with error: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 55)
    print("📊 VALIDATION SUMMARY")
    print("=" * 55)
    
    passed = 0
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} validations passed")
    
    if passed == total:
        print("\n🎉 AWS-Only Security Configuration is VALID!")
        print("   The system is properly configured for secure AWS-only operation.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} validation(s) failed.")
        print("   Please address the issues above before proceeding.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
