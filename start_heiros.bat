@echo off
REM ===============================================================================
REM TidyLLM-HeirOS Smart Launcher for Windows
REM Cross-platform launcher with process management and auto-reload
REM 
REM LAUNCHER BEHAVIOR SPECIFICATION:
REM ===============================
REM a) OS Detection & Process Management:
REM    - Detect Windows/Unix/Linux/Mac automatically
REM    - Kill existing Streamlit processes using appropriate OS commands:
REM      * Windows: taskkill /PID /T /F (process tree termination)
REM      * Unix/Linux/Mac: SIGTERM -> SIGKILL progression
REM    - Use psutil for cross-platform process discovery and management
REM    
REM b) Auto-Reload & No Browser Restrictions:
REM    - Enable --server.runOnSave=true for automatic file change detection
REM    - Use --server.headless=true to prevent forced browser opening
REM    - Disable CORS and XSRF for local development freedom
REM    - Allow external connections with --server.address=0.0.0.0
REM    
REM c) Dynamic Port Assignment:
REM    - NO hardcoded ports - scan from 8501 upward for available port
REM    - Handle port conflicts gracefully with automatic fallback
REM    - Display multiple access URLs: localhost, network IP, mobile access
REM    - Support up to 100 port attempts before failure
REM 
REM EXECUTION FLOW:
REM ==============
REM 1. Display system info (OS, Python version, working directory)
REM 2. Kill any existing Streamlit processes (cross-platform)
REM 3. Install/check dependencies if requested
REM 4. Validate required packages are available
REM 5. Find available port dynamically
REM 6. Launch Streamlit with optimal settings for development
REM 7. Monitor process and display logs in real-time
REM 8. Handle graceful shutdown on Ctrl+C with proper cleanup
REM ===============================================================================

echo.
echo ================================================================================
echo 🌲 TidyLLM-HeirOS Smart Launcher (Windows)
echo Cross-platform Streamlit launcher with auto-reload and process management
echo ================================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Python not found! Please install Python 3.8+ and add to PATH
    pause
    exit /b 1
)

REM Check for launcher script
if not exist "start_heiros.py" (
    echo ❌ start_heiros.py not found!
    echo Please ensure you're in the correct directory
    pause
    exit /b 1
)

REM Default arguments
set ARGS=

REM Parse command line arguments
:parse_args
if "%1"=="" goto run_launcher
if "%1"=="--install-deps" set ARGS=%ARGS% --install-deps
if "%1"=="--open-browser" set ARGS=%ARGS% --open-browser
if "%1"=="--no-auto-reload" set ARGS=%ARGS% --no-auto-reload
if "%1"=="--no-logs" set ARGS=%ARGS% --no-logs
if "%1"=="--kill-only" set ARGS=%ARGS% --kill-only
if "%1"=="--help" goto show_help
shift
goto parse_args

:run_launcher
echo 🚀 Launching HeirOS Dashboard with Python launcher...
echo.

REM Install psutil if not available (needed for process management)
echo 📦 Ensuring psutil is available for process management...
python -c "import psutil" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Installing psutil...
    python -m pip install psutil
)

REM Run the Python launcher
python start_heiros.py %ARGS%

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ Launch failed with error code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ✅ HeirOS Dashboard session ended
pause
exit /b 0

:show_help
echo.
echo 📖 TidyLLM-HeirOS Launcher Help
echo.
echo Usage: start_heiros.bat [OPTIONS]
echo.
echo Options:
echo   --install-deps     Install/update Python requirements before launching
echo   --open-browser     Automatically open browser after launch
echo   --no-auto-reload   Disable auto-reload on file changes
echo   --no-logs          Don't show Streamlit logs in console
echo   --kill-only        Only kill existing Streamlit processes and exit
echo   --help             Show this help message
echo.
echo Examples:
echo   start_heiros.bat                    # Standard launch
echo   start_heiros.bat --install-deps     # Install deps and launch
echo   start_heiros.bat --open-browser     # Launch and open browser
echo   start_heiros.bat --kill-only        # Kill existing processes only
echo.
pause
exit /b 0