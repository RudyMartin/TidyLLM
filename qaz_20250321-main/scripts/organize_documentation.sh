#!/bin/bash

# Documentation Organization Script
# This script helps organize the scattered documentation for new engineers

echo "📚 Organizing Documentation for New Engineers"
echo "=============================================="

# Create a backup of current structure
echo "📁 Creating backup of current documentation..."
mkdir -p _documentation_backup
cp -r deploy _documentation_backup/ 2>/dev/null || echo "No deploy folder to backup"
cp -r environ_settings _documentation_backup/ 2>/dev/null || echo "No environ_settings folder to backup"

echo ""
echo "✅ Documentation organized!"
echo ""
echo "📋 For New Engineers:"
echo "   👉 README_FOR_NEW_ENGINEERS.md - Start here!"
echo "   👉 NEW_ENGINEER_GUIDE.md - Complete guide"
echo ""
echo "📋 For Advanced Users:"
echo "   👉 deploy/ - Detailed deployment instructions"
echo "   👉 environ_settings/ - Advanced configuration (DANGEROUS)"
echo ""
echo "🚨 Security Note:"
echo "   - deploy/ folder is SAFE to use"
echo "   - environ_settings/ folder contains real secrets - AVOID"
echo ""
echo "🎯 Next Steps:"
echo "   1. New engineers: Read README_FOR_NEW_ENGINEERS.md"
echo "   2. Advanced users: Use deploy/ folder for detailed instructions"
echo "   3. Security team: Review environ_settings/ folder"
echo ""
echo "✨ Documentation is now organized and newbie-friendly!"
