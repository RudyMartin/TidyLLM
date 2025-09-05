#!/bin/bash
# ===============================================================================
# TidyLLM-HeirOS Smart Launcher for Unix/Linux/Mac
# Cross-platform launcher with process management and auto-reload
# 
# LAUNCHER BEHAVIOR SPECIFICATION:
# ===============================
# a) OS Detection & Process Management:
#    - Detect Windows/Unix/Linux/Mac automatically
#    - Kill existing Streamlit processes using appropriate OS commands:
#      * Windows: taskkill /PID /T /F (process tree termination)
#      * Unix/Linux/Mac: SIGTERM -> SIGKILL progression
#    - Use psutil for cross-platform process discovery and management
#    
# b) Auto-Reload & No Browser Restrictions:
#    - Enable --server.runOnSave=true for automatic file change detection
#    - Use --server.headless=true to prevent forced browser opening
#    - Disable CORS and XSRF for local development freedom
#    - Allow external connections with --server.address=0.0.0.0
#    
# c) Dynamic Port Assignment:
#    - NO hardcoded ports - scan from 8501 upward for available port
#    - Handle port conflicts gracefully with automatic fallback
#    - Display multiple access URLs: localhost, network IP, mobile access
#    - Support up to 100 port attempts before failure
# 
# EXECUTION FLOW:
# ==============
# 1. Display system info (OS, Python version, working directory)
# 2. Kill any existing Streamlit processes (cross-platform)
# 3. Install/check dependencies if requested
# 4. Validate required packages are available
# 5. Find available port dynamically
# 6. Launch Streamlit with optimal settings for development
# 7. Monitor process and display logs in real-time
# 8. Handle graceful shutdown on Ctrl+C with proper cleanup
# ===============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Header
echo
echo "================================================================================"
echo -e "${CYAN}🌲 TidyLLM-HeirOS Smart Launcher (Unix/Linux/Mac)${NC}"
echo -e "${CYAN}Cross-platform Streamlit launcher with auto-reload and process management${NC}"
echo "================================================================================"
echo

# Function to show help
show_help() {
    echo
    echo -e "${BLUE}📖 TidyLLM-HeirOS Launcher Help${NC}"
    echo
    echo "Usage: ./start_heiros.sh [OPTIONS]"
    echo
    echo "Options:"
    echo "  --install-deps     Install/update Python requirements before launching"
    echo "  --open-browser     Automatically open browser after launch"
    echo "  --no-auto-reload   Disable auto-reload on file changes"
    echo "  --no-logs          Don't show Streamlit logs in console"
    echo "  --kill-only        Only kill existing Streamlit processes and exit"
    echo "  --help             Show this help message"
    echo
    echo "Examples:"
    echo "  ./start_heiros.sh                    # Standard launch"
    echo "  ./start_heiros.sh --install-deps     # Install deps and launch"
    echo "  ./start_heiros.sh --open-browser     # Launch and open browser"
    echo "  ./start_heiros.sh --kill-only        # Kill existing processes only"
    echo
    exit 0
}

# Parse arguments
ARGS=""
for arg in "$@"; do
    case $arg in
        --install-deps)
            ARGS="$ARGS --install-deps"
            ;;
        --open-browser)
            ARGS="$ARGS --open-browser"
            ;;
        --no-auto-reload)
            ARGS="$ARGS --no-auto-reload"
            ;;
        --no-logs)
            ARGS="$ARGS --no-logs"
            ;;
        --kill-only)
            ARGS="$ARGS --kill-only"
            ;;
        --help|-h)
            show_help
            ;;
        *)
            echo -e "${RED}❌ Unknown argument: $arg${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo -e "${RED}❌ Python not found! Please install Python 3.8+ and add to PATH${NC}"
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

# Check Python version
echo -e "${BLUE}🐍 Checking Python version...${NC}"
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
echo "   $PYTHON_VERSION"

# Check for launcher script
if [ ! -f "start_heiros.py" ]; then
    echo -e "${RED}❌ start_heiros.py not found!${NC}"
    echo "Please ensure you're in the correct directory"
    exit 1
fi

# Make launcher executable if needed
chmod +x start_heiros.py 2>/dev/null || true

echo -e "${GREEN}🚀 Launching HeirOS Dashboard with Python launcher...${NC}"
echo

# Install psutil if not available (needed for process management)
echo -e "${BLUE}📦 Ensuring psutil is available for process management...${NC}"
if ! $PYTHON_CMD -c "import psutil" &> /dev/null; then
    echo "Installing psutil..."
    $PYTHON_CMD -m pip install psutil
fi

# Set up signal handling for cleanup
cleanup() {
    echo
    echo -e "${YELLOW}🧹 Cleaning up...${NC}"
    # The Python script handles its own cleanup
    exit 0
}

trap cleanup SIGINT SIGTERM

# Run the Python launcher
$PYTHON_CMD start_heiros.py $ARGS
EXIT_CODE=$?

echo
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ HeirOS Dashboard session ended normally${NC}"
else
    echo -e "${RED}❌ Launch failed with exit code $EXIT_CODE${NC}"
fi

exit $EXIT_CODE