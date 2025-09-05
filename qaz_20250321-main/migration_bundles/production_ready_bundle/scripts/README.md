# Scripts

This directory contains utility scripts for the QA system.

## Contents

### Pre-Flight & Cleanup
- `pre_flight_cleanup.py` - Comprehensive pre-flight checks and cleanup utilities
- `manage_demo.py` - Demo management (start, stop, restart, status, test)

### Configuration & Setup
- `dev_config_manager.py` - Development configuration management
- `setup_rag_system.py` - RAG system setup utility

### Utilities
- `show_reports.py` - Report display utility
- `generate_pdf_report.py` - PDF generation utility
- `upgrade_latex.py` - LaTeX processing utility
- `upgrade_qa_criteria.py` - QA criteria upgrade utility
- `upgrade_qa_criteria_mcp.py` - MCP-integrated QA criteria upgrade

## Usage

Most scripts can be run directly from the project root:

```bash
# Pre-flight checks
python3 scripts/pre_flight_cleanup.py --pre-flight

# Demo management
python3 scripts/manage_demo.py start

# Setup RAG system
python3 scripts/setup_rag_system.py
```

## Note

These scripts are designed to be run from the project root directory, not from within the scripts directory.
