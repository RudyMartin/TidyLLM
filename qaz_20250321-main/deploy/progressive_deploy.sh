#!/bin/bash
# VectorQA Sage - Progressive Deployment Script
# Auth: Rudy Martin - Next Shift Consulting LLC
# Date: 2025-08-23

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to create minimal test app
create_minimal_app() {
    print_status "Creating minimal test application..."
    
    cat > ../app/src/minimal_test.py << 'EOF'
#!/usr/bin/env python3
"""
Minimal Test Application
Tests basic connectivity and environment setup
"""

import streamlit as st
import os
import sys
from pathlib import Path

def main():
    st.set_page_config(
        page_title="VectorQA Sage - Connection Test",
        page_icon="🔗",
        layout="wide"
    )
    
    st.title("🔗 VectorQA Sage - Connection Test")
    st.markdown("This is a minimal application to test connections and environment setup.")
    
    # Test 1: Basic Python environment
    st.header("✅ Test 1: Python Environment")
    st.write(f"Python version: {sys.version}")
    st.write(f"Working directory: {os.getcwd()}")
    st.write(f"Environment: {os.getenv('VECTORQA_ENV', 'Not set')}")
    
    # Test 2: Environment settings
    st.header("✅ Test 2: Environment Settings")
    environ_dir = Path("../environ_settings")
    if environ_dir.exists():
        st.success("✅ environ_settings directory found")
        env_files = list(environ_dir.glob("*.env*"))
        if env_files:
            st.write("Environment files found:")
            for file in env_files:
                st.write(f"  - {file.name}")
        else:
            st.warning("⚠️ No environment files found")
    else:
        st.error("❌ environ_settings directory not found")
    
    # Test 3: Basic imports
    st.header("✅ Test 3: Basic Imports")
    try:
        import pandas as pd
        st.success("✅ pandas imported successfully")
    except ImportError as e:
        st.error(f"❌ pandas import failed: {e}")
    
    try:
        import numpy as np
        st.success("✅ numpy imported successfully")
    except ImportError as e:
        st.error(f"❌ numpy import failed: {e}")
    
    try:
        import streamlit as st
        st.success("✅ streamlit imported successfully")
    except ImportError as e:
        st.error(f"❌ streamlit import failed: {e}")
    
    # Test 4: File system access
    st.header("✅ Test 4: File System Access")
    try:
        app_dir = Path(".")
        st.write(f"App directory: {app_dir.absolute()}")
        st.success("✅ File system access working")
    except Exception as e:
        st.error(f"❌ File system access failed: {e}")
    
    # Test 5: Network connectivity (basic)
    st.header("✅ Test 5: Network Connectivity")
    try:
        import urllib.request
        response = urllib.request.urlopen('http://httpbin.org/get', timeout=5)
        st.success("✅ Basic network connectivity working")
    except Exception as e:
        st.error(f"❌ Network connectivity failed: {e}")
    
    # Test 6: Environment variables
    st.header("✅ Test 6: Environment Variables")
    env_vars = ['VECTORQA_ENV', 'LOG_LEVEL', 'CACHE_DIR']
    for var in env_vars:
        value = os.getenv(var, 'Not set')
        if value != 'Not set':
            st.success(f"✅ {var}: {value}")
        else:
            st.warning(f"⚠️ {var}: Not set")
    
    # Summary
    st.header("📊 Test Summary")
    st.info("""
    **Next Steps:**
    1. If all tests pass, run: `./progressive_deploy.sh full`
    2. If tests fail, check environment setup
    3. Review logs for detailed error information
    """)
    
    st.success("🎉 Minimal test application is running!")
    st.write("This confirms basic connectivity and environment setup.")

if __name__ == "__main__":
    main()
EOF
    
    print_success "Minimal test application created"
}

# Function to deploy minimal app
deploy_minimal() {
    print_status "Deploying minimal test application..."
    
    # Create minimal app
    create_minimal_app
    
    # Deploy using Docker or virtual environment
    if command_exists docker && command_exists docker-compose; then
        print_status "Deploying minimal app with Docker..."
        
        # Create minimal docker-compose
        cat > docker-compose.minimal.yml << 'EOF'
version: '3.8'
services:
  vectorqa-minimal:
    build:
      context: ../app
      dockerfile: ../environ_settings/Dockerfile
    container_name: vectorqa-minimal
    ports:
      - "8501:8501"
    environment:
      - VECTORQA_ENV=production
    volumes:
      - ./environ_settings:/app/environ_settings:ro
    command: ["streamlit", "run", "src/minimal_test.py", "--server.port=8501", "--server.address=0.0.0.0"]
    restart: unless-stopped
EOF
        
        # Deploy minimal app
        docker-compose -f docker-compose.minimal.yml up -d --build
        
        # Wait and check
        sleep 10
        if curl -f http://localhost:8501/_stcore/health >/dev/null 2>&1; then
            print_success "Minimal app deployed successfully!"
            print_status "Access test app at: http://localhost:8501"
            print_status "Run './progressive_deploy.sh full' to deploy full application"
        else
            print_error "Minimal app failed to start"
            exit 1
        fi
        
    else
        print_status "Deploying minimal app with virtual environment..."
        
        cd ../app
        
        # Create virtual environment if not exists
        if [[ ! -d "venv" ]]; then
            python3 -m venv venv
        fi
        
        source venv/bin/activate
        pip install -r requirements_demo.txt
        
        # Start minimal app
        nohup streamlit run src/minimal_test.py --server.port=8501 --server.address=0.0.0.0 > minimal.log 2>&1 &
        
        sleep 5
        if curl -f http://localhost:8501/_stcore/health >/dev/null 2>&1; then
            print_success "Minimal app deployed successfully!"
            print_status "Access test app at: http://localhost:8501"
            print_status "Run './progressive_deploy.sh full' to deploy full application"
        else
            print_error "Minimal app failed to start"
            exit 1
        fi
    fi
}

# Function to deploy full application
deploy_full() {
    print_status "Deploying full application..."
    
    # Stop minimal app if running
    if command_exists docker-compose; then
        docker-compose -f docker-compose.minimal.yml down 2>/dev/null || true
    else
        pkill -f "minimal_test.py" 2>/dev/null || true
    fi
    
    # Deploy full application using existing deploy script
    ./deploy.sh docker 2>/dev/null || ./deploy.sh venv
    
    print_success "Full application deployed!"
    print_status "Access full application at: http://localhost:8501"
}

# Function to show usage
show_usage() {
    echo "VectorQA Sage - Progressive Deployment"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  minimal    Deploy minimal test app (connection test)"
    echo "  full       Deploy full application"
    echo "  stop       Stop all applications"
    echo "  status     Check application status"
    echo "  help       Show this help message"
    echo ""
    echo "Progressive Deployment Steps:"
    echo "  1. $0 minimal    # Test connections and environment"
    echo "  2. Verify all tests pass in browser"
    echo "  3. $0 full       # Deploy full application"
    echo ""
    echo "This approach helps identify issues early before full deployment."
}

# Function to stop applications
stop_applications() {
    print_status "Stopping all applications..."
    
    if command_exists docker-compose; then
        docker-compose down 2>/dev/null || true
        docker-compose -f docker-compose.minimal.yml down 2>/dev/null || true
    fi
    
    pkill -f "streamlit run" 2>/dev/null || true
    
    print_success "All applications stopped"
}

# Function to check status
check_status() {
    print_status "Checking application status..."
    
    if curl -f http://localhost:8501/_stcore/health >/dev/null 2>&1; then
        print_success "Application is running"
        print_status "Access at: http://localhost:8501"
        
        # Check if it's minimal or full app
        if curl -s http://localhost:8501 | grep -q "Connection Test"; then
            print_warning "Minimal test app is running"
            print_status "Run './progressive_deploy.sh full' to deploy full application"
        else
            print_success "Full application is running"
        fi
    else
        print_error "No application is running"
    fi
}

# Main script
main() {
    case "${1:-help}" in
        minimal)
            deploy_minimal
            ;;
        full)
            deploy_full
            ;;
        stop)
            stop_applications
            ;;
        status)
            check_status
            ;;
        help|*)
            show_usage
            ;;
    esac
}

# Run main function
main "$@"
