@echo off
echo ================================================================================
echo TidyLLM-HeirOS Streamlit Dashboard
echo Hierarchical Workflow Management System
echo ================================================================================
echo.

echo [1/3] Installing requirements...
pip install -r heiros_requirements.txt

echo.
echo [2/3] Starting Streamlit application...
echo Open your browser to: http://localhost:8501
echo.

echo [3/3] Launching dashboard...
streamlit run heiros_streamlit_demo.py --server.port=8501 --server.address=localhost

pause