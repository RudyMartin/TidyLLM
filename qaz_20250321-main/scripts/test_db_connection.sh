#!/bin/bash

# Test database connection for error tracking tables
echo "🚀 Testing Error Tracking Database Connection"
echo "=============================================="

# Source the credentials file to set environment variables
if [ -f "src/backend/config/credentials.env" ]; then
    echo "📁 Loading credentials from src/backend/config/credentials.env"
    source src/backend/config/credentials.env
else
    echo "⚠️  Credentials file not found, using environment variables"
fi

# Run the Python test script (now uses credential manager)
python3 scripts/test_error_tracking_remote.py
