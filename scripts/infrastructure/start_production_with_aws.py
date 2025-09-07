#!/usr/bin/env python3
"""

# Centralized AWS credential management
import sys
from pathlib import Path

# Add admin directory to path for credential loading
sys.path.append(str(Path(__file__).parent.parent / 'tidyllm' / 'admin') if 'tidyllm' in str(Path(__file__)) else str(Path(__file__).parent / 'tidyllm' / 'admin'))
from credential_loader import set_aws_environment

# Load AWS credentials using centralized system
set_aws_environment()
Start Production Tracking with AWS Credentials
"""

import os
import sys
import subprocess

# Set AWS credentials




print("=" * 60)
print("STARTING PRODUCTION TRACKING WITH AWS S3")
print("=" * 60)
print(f"AWS Access Key: {os.environ['AWS_ACCESS_KEY_ID'][:10]}...")
print(f"AWS Region: {os.environ['AWS_DEFAULT_REGION']}")
print("=" * 60)

# Import and run the production tracking directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import after setting environment variables
from scripts.production_tracking_drop_zones import main

print("\nStarting production tracking drop zones...")
print("S3 uploads should now work with configured credentials")
print("Press Ctrl+C to stop\n")

try:
    main()
except KeyboardInterrupt:
    print("\n[STOPPED] Production tracking stopped by user")
except Exception as e:
    print(f"\n[ERROR] Production tracking failed: {e}")
    import traceback
    traceback.print_exc()