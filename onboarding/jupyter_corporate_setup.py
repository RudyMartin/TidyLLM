"""
Corporate Jupyter Lab Setup for TidyLLM
======================================

This script provides the exact setup needed for corporate environments where:
1. AWS access is controlled through SSO (Single Sign-On)
2. Temporary IAM role credentials are used
3. Database connection strings are provided by IT
4. No permanent AWS keys are stored in the system

Usage in Jupyter Lab:
    exec(open('tidyllm/onboarding/jupyter_corporate_setup.py').read())
"""

import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional

def setup_corporate_environment(
    # AWS SSO Credentials (get these from your SSO portal)
    aws_access_key_id: str,
    aws_secret_access_key: str, 
    aws_session_token: str,
    aws_region: str = "us-east-1",
    
    # Database Configuration (provided by your IT team)
    postgres_host: str = "corporate-db.internal.company.com",
    postgres_port: int = 5432,
    postgres_db: str = "tidyllm_prod",
    postgres_user: str = "service_tidyllm", 
    postgres_password: str = "your_encoded_password",
    
    # Optional: Validation
    validate_setup: bool = True
) -> Dict[str, Any]:
    """
    Set up TidyLLM for corporate environment with SSO credentials.
    
    Args:
        aws_access_key_id: Temporary access key from SSO (starts with ASIA)
        aws_secret_access_key: Temporary secret key from SSO
        aws_session_token: Session token from SSO (very long string)
        aws_region: AWS region for services
        postgres_*: Database connection details from IT
        validate_setup: Whether to validate the setup after configuration
    
    Returns:
        Validation results if validate_setup=True, else empty dict
    """
    
    print("🏢 Setting up TidyLLM for Corporate Environment")
    print("=" * 55)
    print(f"⏰ Setup started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Step 1: Configure AWS environment variables
    print("1️⃣ Configuring AWS SSO credentials...")
    
    os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key_id
    os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_access_key
    os.environ['AWS_SESSION_TOKEN'] = aws_session_token
    os.environ['AWS_DEFAULT_REGION'] = aws_region
    
    # Validate SSO credential format
    if aws_access_key_id.startswith('ASIA'):
        print("   ✅ SSO credentials detected (access key starts with ASIA)")
    else:
        print("   ⚠️  Warning: Access key doesn't start with ASIA (may not be SSO)")
    
    if len(aws_session_token) > 400:
        print("   ✅ Session token length looks correct for SSO")
    else:
        print("   ⚠️  Warning: Session token seems short for SSO credentials")
    
    print(f"   ✅ AWS Region: {aws_region}")
    print()
    
    # Step 2: Configure database environment variables  
    print("2️⃣ Configuring database connection...")
    
    os.environ['POSTGRES_HOST'] = postgres_host
    os.environ['POSTGRES_PORT'] = str(postgres_port)
    os.environ['POSTGRES_DB'] = postgres_db
    os.environ['POSTGRES_USER'] = postgres_user
    os.environ['POSTGRES_PASSWORD'] = postgres_password
    
    print(f"   ✅ Database Host: {postgres_host}")
    print(f"   ✅ Database: {postgres_db}")
    print(f"   ✅ User: {postgres_user}")
    print(f"   ✅ Password: {'*' * len(postgres_password)} (masked)")
    print()
    
    # Step 3: Import and test TidyLLM
    print("3️⃣ Testing TidyLLM imports...")
    
    try:
        # Add current directory to path if needed
        if '.' not in sys.path:
            sys.path.insert(0, '.')
            
        # Test basic import
        import tidyllm
        print("   ✅ TidyLLM imported successfully")
        
        # Test infrastructure imports
        from tidyllm.infrastructure.standards import resolve_model_id
        print("   ✅ Standards module imported")
        
        # Test onboarding imports
        from tidyllm.onboarding.session_validator import CorporateSessionManager
        print("   ✅ Corporate session manager imported")
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        print("   💡 Make sure you're in the TidyLLM root directory")
        return {"setup_success": False, "error": str(e)}
    
    print()
    
    # Step 4: Validation (if requested)
    results = {"setup_success": True}
    
    if validate_setup:
        print("4️⃣ Validating corporate setup...")
        
        try:
            from tidyllm.onboarding.session_validator import test_full_aws_stack
            
            # Test the full stack
            validation_results = test_full_aws_stack(
                access_key_id=aws_access_key_id,
                secret_access_key=aws_secret_access_key,
                session_token=aws_session_token,
                region=aws_region,
                postgres_config={
                    'host': postgres_host,
                    'port': postgres_port,
                    'database': postgres_db,
                    'username': postgres_user,
                    'password': postgres_password
                }
            )
            
            # Display results
            if validation_results['overall_success']:
                print("   🎉 ALL VALIDATION TESTS PASSED!")
                print("   ✅ AWS session created successfully")
                print("   ✅ Bedrock service accessible")
                print("   ✅ S3 service accessible")
                if 'postgresql' in validation_results['service_validation']:
                    if validation_results['service_validation']['postgresql']['success']:
                        print("   ✅ PostgreSQL database accessible")
                    else:
                        print(f"   ❌ PostgreSQL: {validation_results['service_validation']['postgresql']['message']}")
            else:
                print("   ⚠️  Some validation tests failed:")
                for service, result in validation_results['service_validation'].items():
                    status = "✅" if result['success'] else "❌"
                    print(f"   {status} {service.title()}: {result['message']}")
            
            results.update(validation_results)
            
        except Exception as e:
            print(f"   ❌ Validation failed: {e}")
            results['validation_error'] = str(e)
    
    print()
    print("🎯 CORPORATE SETUP COMPLETE!")
    print()
    
    # Usage instructions
    print("📋 Next Steps:")
    print("   1. You can now use TidyLLM normally in this Jupyter session")
    print("   2. All AWS and database connections will use the configured credentials")
    print("   3. Credentials are temporary and will expire based on your SSO settings")
    print()
    
    print("💡 Example Usage:")
    print("""
   # Import TidyLLM components
   from tidyllm.gateways import CorporateLLMGateway
   from tidyllm.infrastructure.standards import TidyLLMStandardRequest
   
   # Create a request
   request = TidyLLMStandardRequest(
       model_id="claude-3-sonnet",  # Will resolve to actual Bedrock model ID
       user_id="your_user_id",
       session_id="jupyter_session_001",
       messages=[{"role": "user", "content": "Hello, this is a test"}]
   )
   
   # Process with corporate gateway
   gateway = CorporateLLMGateway()
   response = gateway.process_request(request)
   print(response.data)
   """)
    
    return results


def quick_sso_setup():
    """
    Interactive setup for users who want to input credentials manually.
    
    This function prompts for SSO credentials and database info.
    """
    print("🔐 Corporate SSO Quick Setup")
    print("=" * 35)
    print("Please enter your corporate credentials:")
    print()
    
    # Get AWS SSO credentials
    print("AWS SSO Credentials (from your SSO portal):")
    aws_access_key = input("AWS Access Key ID (starts with ASIA): ").strip()
    aws_secret_key = input("AWS Secret Access Key: ").strip()
    aws_session_token = input("AWS Session Token: ").strip()
    aws_region = input("AWS Region (default: us-east-1): ").strip() or "us-east-1"
    
    print()
    
    # Get database credentials
    print("Database Connection (from your IT team):")
    db_host = input("PostgreSQL Host (default: corporate-db.internal.company.com): ").strip() or "corporate-db.internal.company.com"
    db_port = input("PostgreSQL Port (default: 5432): ").strip() or "5432"
    db_name = input("Database Name (default: tidyllm_prod): ").strip() or "tidyllm_prod"
    db_user = input("Database User (default: service_tidyllm): ").strip() or "service_tidyllm"
    db_password = input("Database Password: ").strip()
    
    print()
    
    # Validate inputs
    if not aws_access_key or not aws_secret_key or not aws_session_token:
        print("❌ Error: AWS credentials are required")
        return False
        
    if not db_password:
        print("❌ Error: Database password is required")
        return False
    
    # Run setup
    return setup_corporate_environment(
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        aws_session_token=aws_session_token,
        aws_region=aws_region,
        postgres_host=db_host,
        postgres_port=int(db_port),
        postgres_db=db_name,
        postgres_user=db_user,
        postgres_password=db_password
    )


def check_credential_expiration():
    """
    Check when your SSO credentials will expire.
    
    SSO credentials typically expire within 1-12 hours.
    """
    print("⏰ Credential Expiration Check")
    print("=" * 35)
    
    access_key = os.environ.get('AWS_ACCESS_KEY_ID')
    
    if not access_key:
        print("❌ No AWS credentials found in environment")
        return
    
    if access_key.startswith('ASIA'):
        print("✅ SSO credentials detected")
        print("💡 SSO credentials typically expire in 1-12 hours")
        print("🔄 Refresh from your SSO portal when they expire")
    else:
        print("⚠️  Long-term credentials detected (not SSO)")
        print("💡 These may be permanent access keys")
    
    # Try to get actual expiration if possible
    try:
        import boto3
        session = boto3.Session()
        sts = session.client('sts')
        
        # Get caller identity (this will fail if credentials expired)
        identity = sts.get_caller_identity()
        print(f"✅ Credentials still valid - Account: {identity['Account']}")
        print(f"✅ Current identity: {identity['Arn']}")
        
    except Exception as e:
        print(f"❌ Credential validation failed: {e}")
        print("🔄 You may need to refresh your SSO credentials")


# Convenience functions for Jupyter
def corporate_setup():
    """Alias for quick_sso_setup() - easier to remember."""
    return quick_sso_setup()

def check_creds():
    """Alias for check_credential_expiration() - easier to remember."""
    return check_credential_expiration()


# Auto-run instructions when file is imported
if __name__ == "__main__":
    print(__doc__)
else:
    # When imported, show usage
    print("""
🏢 Corporate TidyLLM Setup Loaded!

Quick Start Options:
1. corporate_setup()     # Interactive credential input
2. check_creds()         # Check if credentials are still valid

Manual Setup:
setup_corporate_environment(
    aws_access_key_id="ASIA...",
    aws_secret_access_key="...",
    aws_session_token="...",
    postgres_password="your_db_password"
)

Need help? Run: help(setup_corporate_environment)
    """)