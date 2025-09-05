#!/bin/bash
# Quick upgrade script for VectorQA Sage
# Run this for immediate safe upgrades

echo "🚀 VectorQA Sage - Quick Safe Upgrades"
echo "======================================"

# Backup current state
echo "📋 Creating backup..."
python3 -m pip freeze > requirements_backup_$(date +%Y%m%d_%H%M%S).txt

# Test current state
echo "🧪 Testing current state..."
if ! python3 run_tests.py --fast; then
    echo "❌ Current tests failing. Fix first!"
    exit 1
fi

echo "✅ Tests passing. Proceeding with safe upgrades..."

# Phase 1: Safe upgrades only
echo "📦 Upgrading safe packages..."
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade boto3 botocore
python3 -m pip install --upgrade dill fsspec multiprocess

# Test after upgrades
echo "🧪 Testing after upgrades..."
if python3 run_tests.py --fast; then
    echo "✅ Safe upgrades completed successfully!"
    echo "📝 For major upgrades (dspy 3.0.1, litellm 1.75.9), run:"
    echo "   python3 safe_upgrade.py"
else
    echo "❌ Tests failed after upgrade. Check the issues."
    exit 1
fi
