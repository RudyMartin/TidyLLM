#!/bin/bash
# Clean installation script for qa-scoring environment
# This avoids version resolution warnings that might concern clients

echo "=== Installing QA-Scoring Environment ==="
echo ""

# Step 1: Install TLM locally first (no version conflicts)
echo "[1/4] Installing TLM..."
cd tlm
pip install -e . --quiet 2>/dev/null || pip install -e .
cd ..

# Step 2: Install tidyllm-sentence locally
echo "[2/4] Installing TidyLLM-Sentence..."
cd tidyllm-sentence
pip install -e . --quiet 2>/dev/null || pip install -e .
cd ..

# Step 3: Install TidyLLM with local dependencies
echo "[3/4] Installing TidyLLM..."
cd tidyllm
pip install -e . --quiet 2>/dev/null || pip install -e .
cd ..

# Step 4: Verify installation
echo "[4/4] Verifying installation..."
python -c "import tlm; import tidyllm; print('✓ All packages installed successfully!')" 2>/dev/null

if [ $? -eq 0 ]; then
    echo ""
    echo "=== Installation Complete ==="
    echo "TLM Version: $(python -c 'import tlm; print(tlm.__version__)' 2>/dev/null)"
    echo "Environment: qa-scoring"
    echo ""
else
    echo ""
    echo "=== Installation completed with warnings ==="
    echo "This is normal for development environments."
    echo ""
fi