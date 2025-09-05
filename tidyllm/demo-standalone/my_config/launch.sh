#!/bin/bash
# Settings Configurator Launcher Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Starting Settings Configurator..."
echo "📁 Working directory: $SCRIPT_DIR"
echo "🌐 Opening browser at: http://localhost:8501"
echo "⏹️  Press Ctrl+C to stop"
echo "----------------------------------------"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if Streamlit is installed
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "⚠️  Streamlit not found. Installing dependencies..."
    pip3 install -r requirements.txt
fi

# Start Streamlit
python3 -m streamlit run settings_configurator.py \
    --server.port 8501 \
    --server.address localhost \
    --browser.gatherUsageStats False \
    --server.runOnSave True
