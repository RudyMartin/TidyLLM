#!/usr/bin/env python3
"""
TidyLLM Settings Demo

Demonstrates how the SettingsLoader reads from admin/settings.yaml and shows
which fields are available, including S3 and PostgreSQL configurations.
"""

import sys
import os
from pathlib import Path

# Add tidyllm to path
sys.path.insert(0, str(Path(__file__).parent / 'tidyllm'))

from tidyllm.settings_loader import SettingsLoader

def main():
    print("=" * 60)
    print("  TidyLLM Settings Demo")
    print("  Reading from admin/settings.yaml")
    print("=" * 60)
    
    try:
        # Load settings from admin/settings.yaml specifically
        admin_settings_path = str(Path(__file__).parent / 'tidyllm' / 'tidyllm' / 'admin' / 'settings.yaml')
        print(f"\nAttempting to load settings from: {admin_settings_path}")
        
        if not Path(admin_settings_path).exists():
            print(f"ERROR: Admin settings file not found at: {admin_settings_path}")
            print("Falling back to automatic detection...")
            loader = SettingsLoader()
        else:
            loader = SettingsLoader(admin_settings_path)
        
        # Print full summary
        loader.print_summary()
        
        # Demonstrate accessing specific settings
        print("\n" + "=" * 60)
        print("  Specific Configuration Access")
        print("=" * 60)
        
        # S3 Configuration
        s3_config = loader.get_s3_config()
        print(f"\nS3 Configuration Fields:")
        print(f"   Fields available: {list(s3_config.keys())}")
        if s3_config:
            for key, value in s3_config.items():
                print(f"   {key}: {value}")
        
        # PostgreSQL Configuration
        postgres_config = loader.get_postgres_config()
        print(f"\nPostgreSQL Configuration Fields:")
        print(f"   Fields available: {list(postgres_config.keys())}")
        if postgres_config:
            for key, value in postgres_config.items():
                # Don't print password in full
                if 'password' in key.lower():
                    print(f"   {key}: {'*' * len(str(value))}")
                else:
                    print(f"   {key}: {value}")
        
        # AWS Configuration
        aws_config = loader.settings.aws
        print(f"\nAWS Configuration Fields:")
        print(f"   Fields available: {list(aws_config.keys())}")
        if aws_config:
            for key, value in aws_config.items():
                if isinstance(value, dict):
                    print(f"   {key}: {list(value.keys())}")
                else:
                    print(f"   {key}: {value}")
        
        # Integrations Configuration
        integrations_config = loader.get_integrations_config()
        print(f"\nIntegrations Configuration Fields:")
        print(f"   Fields available: {list(integrations_config.keys())}")
        if integrations_config:
            for key, value in integrations_config.items():
                if isinstance(value, dict):
                    print(f"   {key}: {list(value.keys())}")
                else:
                    print(f"   {key}: {value}")
        
        # Show all top-level sections
        print(f"\nAll Top-Level Sections in Settings:")
        all_sections = [
            'postgres', 's3', 'aws', 'retry', 'cache', 'validation', 
            'batch', 'metrics', 'logging', 'cost_optimization', 
            'security', 'performance', 'development', 'integrations', 
            'environments'
        ]
        
        for section in all_sections:
            section_data = getattr(loader.settings, section, {})
            if section_data:
                print(f"   SUCCESS: {section}: {len(section_data)} fields")
            else:
                print(f"   EMPTY: {section}: empty or missing")
        
        # Test validation
        print(f"\nSettings Validation:")
        is_valid = loader.validate_settings()
        if is_valid:
            print("   SUCCESS: All settings are valid")
        else:
            print("   ERROR: Some settings validation failed")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: Settings demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)