#!/usr/bin/env python3
"""
Health Check Script

Checks the health of the VectorQA Sage deployment.
"""

import requests
import sys
import time

def check_health():
    """Check application health"""
    try:
        # Check if Streamlit is running
        response = requests.get("http://localhost:8501/_stcore/health", timeout=5)
        if response.status_code == 200:
            print("✅ Application is healthy")
            return True
        else:
            print(f"❌ Application health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

if __name__ == "__main__":
    success = check_health()
    sys.exit(0 if success else 1)
