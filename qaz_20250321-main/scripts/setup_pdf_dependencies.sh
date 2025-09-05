#!/bin/bash
# PDF Dependencies Setup Script
# This script safely installs PDF processing dependencies and avoids naming conflicts

set -e  # Exit on any error

echo "🔧 Setting up PDF processing dependencies..."

# Check if we're in a conda environment
if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
    echo "✅ Detected conda environment: $CONDA_DEFAULT_ENV"
    PYTHON_CMD="python"
    PIP_CMD="pip"
else
    echo "⚠️  Not in a conda environment, using system Python"
    PYTHON_CMD="python3"
    PIP_CMD="pip3"
fi

# Function to check if a package is installed
check_package() {
    $PYTHON_CMD -c "import $1" 2>/dev/null && return 0 || return 1
}

# Function to install modern PDF stack (no fitz dependency)
install_modern_pdf_stack() {
    echo "📦 Installing modern PDF processing stack..."
    
    # Install pdfplumber (primary text and table extraction)
    if ! check_package "pdfplumber"; then
        echo "📥 Installing pdfplumber..."
        $PIP_CMD install pdfplumber>=0.11.0
    else
        echo "✅ pdfplumber already installed"
    fi
    
    # Install pypdfium2 (image extraction and advanced features)
    if ! check_package "pypdfium2"; then
        echo "📥 Installing pypdfium2..."
        $PIP_CMD install pypdfium2>=4.30.0
    else
        echo "✅ pypdfium2 already installed"
    fi
    
    # Install pypdf (modern replacement for PyPDF2)
    if ! check_package "pypdf"; then
        echo "📥 Installing pypdf..."
        $PIP_CMD install pypdf>=6.0.0
    else
        echo "✅ pypdf already installed"
    fi
    
    # Verify installation
    if $PYTHON_CMD -c "import pdfplumber, pypdfium2, pypdf; print('✅ Modern PDF stack installed successfully')"; then
        echo "✅ Modern PDF stack installation verified"
    else
        echo "❌ Modern PDF stack installation failed"
        exit 1
    fi
}

# Function to install modern PDF dependencies
install_modern_deps() {
    echo "📦 Installing modern PDF processing dependencies..."
    
    # pypdf (modern replacement for PyPDF2)
    if ! check_package "pypdf"; then
        echo "📥 Installing pypdf..."
        $PIP_CMD install pypdf>=6.0.0
    else
        echo "✅ pypdf already installed"
    fi
    
    # pdfplumber for advanced text and table extraction
    if ! check_package "pdfplumber"; then
        echo "📥 Installing pdfplumber..."
        $PIP_CMD install pdfplumber>=0.9.0
    else
        echo "✅ pdfplumber already installed"
    fi
}

# Main installation process
echo "🚀 Starting PDF dependencies setup..."

# Install modern PDF stack
install_modern_pdf_stack

# Install modern dependencies
install_modern_deps

echo ""
echo "✅ PDF dependencies setup complete!"
echo ""
echo "📋 Summary:"
echo "   - pdfplumber: $(python -c "import pdfplumber; print(pdfplumber.__version__)" 2>/dev/null || echo 'Not installed')"
echo "   - pypdfium2: $(python -c "import pypdfium2; print('Installed')" 2>/dev/null || echo 'Not installed')"
echo "   - pypdf: $(python -c "import pypdf; print(pypdf.__version__)" 2>/dev/null || echo 'Not installed')"
echo ""
echo "🔧 To test the installation, run:"
echo "   python -c \"import pdfplumber, pypdfium2, pypdf; print('Modern PDF stack works!')\""
echo ""
echo "✅ Modern PDF stack benefits:"
echo "   - No more fitz dependency conflicts"
echo "   - Better performance and reliability"
echo "   - Active maintenance and updates"
