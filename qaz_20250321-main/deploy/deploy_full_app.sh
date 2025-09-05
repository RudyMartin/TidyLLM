#!/bin/bash
# VectorQA Sage - Deploy Full App
# Auth: Rudy Martin - Next Shift Consulting LLC
# Date: 2025-08-23

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🚀 VectorQA Sage - Deploy Full App${NC}"
echo "================================"
echo ""

# Check if we're in the right directory
if [[ ! -f "progressive_deploy.sh" ]]; then
    echo -e "${YELLOW}⚠️  Make sure you're in the environ_settings directory${NC}"
    exit 1
fi

# Check if initial app is running
echo -e "${BLUE}📊 Checking current deployment status...${NC}"
if curl -f http://localhost:8501/_stcore/health >/dev/null 2>&1; then
    if curl -s http://localhost:8501 | grep -q "Connection Test"; then
        echo -e "${GREEN}✅ Initial test app is running${NC}"
        echo -e "${YELLOW}💡 Proceeding with full deployment...${NC}"
    else
        echo -e "${GREEN}✅ Full application is already running${NC}"
        echo -e "${YELLOW}💡 No action needed${NC}"
        exit 0
    fi
else
    echo -e "${RED}❌ No application is currently running${NC}"
    echo -e "${YELLOW}💡 Run './deploy_init_app.sh' first to test connections${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}📋 Deploying full application...${NC}"
./progressive_deploy.sh full

echo ""
echo -e "${GREEN}✅ Full application deployed successfully!${NC}"
echo ""
echo -e "${BLUE}📱 Access your full application at: http://localhost:8501${NC}"
echo ""
echo -e "${YELLOW}🔍 Verification steps:${NC}"
echo "1. Open http://localhost:8501 in your browser"
echo "2. Verify all features work correctly"
echo "3. Test key functionality"
echo "4. Check for any error messages"
echo ""
echo -e "${BLUE}📊 To check status: ./progressive_deploy.sh status${NC}"
echo -e "${BLUE}📋 To view logs: ./progressive_deploy.sh logs${NC}"
echo -e "${BLUE}🛑 To stop app: ./progressive_deploy.sh stop${NC}"
