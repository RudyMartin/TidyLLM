#!/bin/bash
# Quick cleanup script for VectorQA Sage
# Removes critical conflicting packages

echo "🧹 VectorQA Sage - Quick Package Cleanup"
echo "========================================"

# Backup current state
echo "📋 Creating backup..."
python3 -m pip freeze > requirements_backup_quick_cleanup_$(date +%Y%m%d_%H%M%S).txt

# Test current state
echo "🧪 Testing current state..."
if ! python3 run_tests.py --fast; then
    echo "❌ Current tests failing. Fix first!"
    exit 1
fi

echo "✅ Tests passing. Proceeding with safe cleanup..."

# Phase 1: Remove duplicates
echo "📦 Removing duplicates..."
python3 -m pip uninstall -y PyPDF2

# Check conflicts
echo "🔍 Checking for conflicts..."
if python3 -m pip check; then
    echo "✅ No conflicts found!"
else
    echo "⚠️  Some conflicts may remain (this is normal)"
fi

# Test after cleanup
echo "🧪 Testing after cleanup..."
if python3 run_tests.py --fast; then
    echo "✅ Quick cleanup completed successfully!"
    echo "📊 Package count after cleanup:"
    python3 -m pip list | wc -l
    echo "💡 For full cleanup, run: python3 safe_cleanup.py"
else
    echo "❌ Tests failed after cleanup. Restoring backup..."
    python3 -m pip install -r requirements_backup_quick_cleanup_*.txt
    exit 1
fi
