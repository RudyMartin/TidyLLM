#!/usr/bin/env python3
"""
Service Account Migration Script

Migrates from individual user accounts to service accounts for production deployment.
This script provides step-by-step guidance for secure account migration.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any

class ServiceAccountMigrator:
    """Handles migration from individual to service accounts"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.config_file = self.project_root / "deployment_config.json"
        self.credentials_file = self.project_root / "src" / "backend" / "config" / "credentials.env"
        
    def run_migration_checklist(self):
        """Run the complete migration checklist"""
        print("🔐 SERVICE ACCOUNT MIGRATION CHECKLIST")
        print("=" * 50)
        
        # Pre-migration checks
        self._check_current_credentials()
        self._check_environment()
        self._show_migration_steps()
        self._show_security_considerations()
        
    def _check_current_credentials(self):
        """Check current credential setup"""
        print("
📋 CURRENT CREDENTIALS CHECK:")
        
        if self.credentials_file.exists():
            print("✅ Credentials file found")
            self._analyze_credentials()
        else:
            print("⚠️  No credentials file found")
            
    def _analyze_credentials(self):
        """Analyze current credentials for migration needs"""
        try:
            with open(self.credentials_file) as f:
                content = f.read()
                
            print("
🔍 CREDENTIAL ANALYSIS:")
            
            # Check for individual user credentials
            if "your_aws_access_key_here" in content or "your_openai_api_key_here" in content:
                print("⚠️  Found placeholder credentials - needs real values")
            else:
                print("✅ Found configured credentials")
                
            # Check for service account indicators
            if "service" in content.lower() or "role" in content.lower():
                print("✅ Appears to be service account credentials")
            else:
                print("⚠️  Appears to be individual user credentials")
                
        except Exception as e:
            print(f"❌ Error analyzing credentials: {e}")
    
    def _check_environment(self):
        """Check current environment"""
        print("
🌍 ENVIRONMENT CHECK:")
        
        env = os.getenv('VECTORQA_ENV', 'unknown')
        print(f"Current environment: {env}")
        
        if env == 'production':
            print("🚨 PRODUCTION ENVIRONMENT - Service accounts REQUIRED!")
        elif env == 'staging':
            print("⚠️  STAGING ENVIRONMENT - Consider service accounts")
        else:
            print("ℹ️  DEVELOPMENT ENVIRONMENT - Individual accounts OK")
    
    def _show_migration_steps(self):
        """Show migration steps"""
        print("
📋 MIGRATION STEPS:")
        print("1. Backup current credentials")
        print("2. Create service accounts (see commands below)")
        print("3. Generate service account credentials")
        print("4. Update credential files")
        print("5. Test in staging environment")
        print("6. Deploy to production")
        print("7. Monitor and validate")
        print("8. Remove individual access")
        
    def _show_security_considerations(self):
        """Show security considerations"""
        print("
🔒 SECURITY CONSIDERATIONS:")
        print("• Use minimal required permissions")
        print("• Rotate service account keys regularly")
        print("• Monitor service account usage")
        print("• Use cloud provider secret management")
        print("• Implement proper access controls")
        
    def create_aws_service_account(self):
        """Create AWS service account"""
        print("
☁️  AWS SERVICE ACCOUNT CREATION:")
        print("Run these commands:")
        print("
# Create IAM role")
        print("aws iam create-role --role-name vectorqa-service-role \")
        print("  --assume-role-policy-document file://trust-policy.json")
        print("
# Attach policies")
        print("aws iam attach-role-policy --role-name vectorqa-service-role \")
        print("  --policy-arn arn:aws:iam::aws:policy/AmazonBedrockFullAccess")
        print("
# Create service user")
        print("aws iam create-user --user-name vectorqa-service-user")
        print("
# Generate access keys")
        print("aws iam create-access-key --user-name vectorqa-service-user")
        
    def create_gcp_service_account(self):
        """Create GCP service account"""
        print("
☁️  GCP SERVICE ACCOUNT CREATION:")
        print("Run these commands:")
        print("
# Create service account")
        print("gcloud iam service-accounts create vectorqa-service \")
        print("  --display-name='VectorQA Service Account'")
        print("
# Grant roles")
        print("gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \")
        print("  --member='serviceAccount:vectorqa-service@YOUR_PROJECT_ID.iam.gserviceaccount.com' \")
        print("  --role='roles/aiplatform.user'")
        print("
# Generate key")
        print("gcloud iam service-accounts keys create service-account-key.json \")
        print("  --iam-account=vectorqa-service@YOUR_PROJECT_ID.iam.gserviceaccount.com")

def main():
    """Main migration function"""
    migrator = ServiceAccountMigrator()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "checklist":
            migrator.run_migration_checklist()
        elif command == "aws":
            migrator.create_aws_service_account()
        elif command == "gcp":
            migrator.create_gcp_service_account()
        else:
            print("Usage: python migrate_to_service_accounts.py [checklist|aws|gcp]")
    else:
        migrator.run_migration_checklist()

if __name__ == "__main__":
    main()
