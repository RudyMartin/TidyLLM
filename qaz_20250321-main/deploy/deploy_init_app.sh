#!/bin/bash
# VectorQA Sage - Deploy Initial App
# Auth: Rudy Martin - Next Shift Consulting LLC
# Date: 2025-08-23

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}🚀 VectorQA Sage - Deploy Initial App${NC}"
echo "=================================="
echo ""

# Check if we're in the right directory
if [[ ! -f "progressive_deploy.sh" ]]; then
    echo -e "${YELLOW}⚠️  Make sure you're in the environ_settings directory${NC}"
    exit 1
fi

# Deploy initial test app
echo -e "${BLUE}📋 Deploying initial test app...${NC}"
./progressive_deploy.sh minimal

echo ""
echo -e "${GREEN}✅ Initial app deployed successfully!${NC}"
echo ""
echo -e "${BLUE}📱 Access your test app at: http://localhost:8501${NC}"
echo ""
echo -e "${YELLOW}🔍 Next steps:${NC}"
echo "1. Open http://localhost:8501 in your browser"
echo "2. Verify all tests show ✅ (green checkmarks)"
echo "3. Fix any ❌ errors if they appear"
echo "4. Run './deploy_full_app.sh' when ready for full deployment"
echo ""
echo -e "${BLUE}📊 To check status: ./progressive_deploy.sh status${NC}"
echo -e "${BLUE}📋 To view logs: ./progressive_deploy.sh logs${NC}"
