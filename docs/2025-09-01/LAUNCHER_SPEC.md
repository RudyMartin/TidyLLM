# TidyLLM-HeirOS Launcher Specification

**Common behavior specification for all `start_` launcher files**

## 📋 LAUNCHER BEHAVIOR SPECIFICATION

### a) OS Detection & Process Management
- **Detect Windows/Unix/Linux/Mac automatically**
- **Kill existing Streamlit processes** using appropriate OS commands:
  - **Windows:** `taskkill /PID /T /F` (process tree termination)
  - **Unix/Linux/Mac:** `SIGTERM` → `SIGKILL` progression
- **Use psutil** for cross-platform process discovery and management

### b) Auto-Reload & No Browser Restrictions
- **Enable `--server.runOnSave=true`** for automatic file change detection
- **Use `--server.headless=true`** to prevent forced browser opening
- **Disable CORS and XSRF** for local development freedom
- **Allow external connections** with `--server.address=0.0.0.0`

### c) Dynamic Port Assignment
- **NO hardcoded ports** - scan from 8501 upward for available port
- **Handle port conflicts gracefully** with automatic fallback
- **Display multiple access URLs:** localhost, network IP, mobile access
- **Support up to 100 port attempts** before failure

## 🔄 EXECUTION FLOW

1. **Display system info** (OS, Python version, working directory)
2. **Kill any existing Streamlit processes** (cross-platform)
3. **Install/check dependencies** if requested
4. **Validate required packages** are available
5. **Find available port** dynamically
6. **Launch Streamlit** with optimal settings for development
7. **Monitor process** and display logs in real-time
8. **Handle graceful shutdown** on Ctrl+C with proper cleanup

## 📁 Implementation Files

- **`start_heiros.py`** - Python cross-platform launcher (primary implementation)
- **`start_heiros.bat`** - Windows batch wrapper
- **`start_heiros.sh`** - Unix/Linux/Mac shell wrapper

All files contain this specification as header comments for reference and consistency.