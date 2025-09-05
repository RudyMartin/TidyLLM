#!/usr/bin/env python3
"""
KMS Credential Encryption Utility
=================================

Encrypts AWS credentials using KMS for secure storage in settings.yaml
"""

import boto3
import base64
import yaml
from pathlib import Path


def encrypt_credentials():
    """Encrypt AWS credentials using KMS"""
    print("🔐 KMS Credential Encryption Utility")
    print("=" * 40)
    
    # Get KMS key ID
    kms_key_id = input("Enter your KMS Key ID or ARN: ").strip()
    if not kms_key_id:
        print("❌ KMS Key ID is required")
        return
    
    # Get credentials to encrypt
    access_key = input("Enter AWS Access Key ID: ").strip()
    secret_key = input("Enter AWS Secret Access Key: ").strip()
    
    if not access_key or not secret_key:
        print("❌ Both access key and secret key are required")
        return
    
    try:
        # Create KMS client (this requires existing AWS credentials to encrypt)
        kms = boto3.client('kms')
        
        print("\n🔐 Encrypting credentials...")
        
        # Encrypt access key
        access_key_response = kms.encrypt(
            KeyId=kms_key_id,
            Plaintext=access_key.encode('utf-8')
        )
        encrypted_access_key = base64.b64encode(access_key_response['CiphertextBlob']).decode('utf-8')
        
        # Encrypt secret key
        secret_key_response = kms.encrypt(
            KeyId=kms_key_id,
            Plaintext=secret_key.encode('utf-8')
        )
        encrypted_secret_key = base64.b64encode(secret_key_response['CiphertextBlob']).decode('utf-8')
        
        print("✅ Credentials encrypted successfully!")
        
        # Update settings.yaml
        settings_path = Path('settings.yaml')
        if settings_path.exists():
            with open(settings_path, 'r') as f:
                settings = yaml.safe_load(f)
        else:
            settings = {}
        
        # Update AWS config with encrypted credentials
        if 'aws' not in settings:
            settings['aws'] = {}
        
        settings['aws']['kms_key_id'] = kms_key_id
        settings['aws']['encrypted_access_key_id'] = encrypted_access_key
        settings['aws']['encrypted_secret_access_key'] = encrypted_secret_key
        
        # Remove any plain text credentials for security
        settings['aws']['access_key_id'] = ''
        settings['aws']['secret_access_key'] = ''
        
        # Save updated settings
        with open(settings_path, 'w') as f:
            yaml.dump(settings, f, default_flow_style=False, indent=2)
        
        print(f"✅ Updated {settings_path} with encrypted credentials")
        print("\n🔐 Your credentials are now securely encrypted with KMS!")
        print("\nNext steps:")
        print("1. Test with: python -c \"from s3_session_manager import S3SessionManager; print(S3SessionManager().get_credential_status())\"")
        print("2. The system will automatically decrypt credentials when needed")
        
    except Exception as e:
        print(f"❌ Encryption failed: {e}")
        print("\nCommon issues:")
        print("- Make sure you have AWS credentials configured to access KMS")
        print("- Verify the KMS key ID is correct and you have encrypt permissions")
        print("- Check that your current AWS credentials can access the KMS key")


def decrypt_test():
    """Test decryption of stored credentials"""
    print("\n🔍 Testing KMS Credential Decryption")
    print("=" * 40)
    
    try:
        from s3_session_manager import S3SessionManager
        
        manager = S3SessionManager()
        status = manager.get_credential_status()
        
        print(f"Credentials available: {status['available']}")
        print(f"Source: {status['source']}")
        
        if status['available']:
            print("✅ KMS decryption successful!")
            
            # Test connection
            test_result = manager.test_connection()
            print(f"Connection test: {'✅ SUCCESS' if test_result['success'] else '❌ FAILED'}")
            if test_result['success']:
                print(f"Found {test_result['bucket_count']} S3 buckets")
            else:
                print(f"Error: {test_result['message']}")
        else:
            print("❌ KMS decryption failed or no encrypted credentials found")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == '__main__':
    print("Choose an option:")
    print("1. Encrypt credentials")
    print("2. Test decryption")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == '1':
        encrypt_credentials()
    elif choice == '2':
        decrypt_test()
    else:
        print("Invalid choice")