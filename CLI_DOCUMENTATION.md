# TidyLLM CLI Documentation

## Overview
TidyLLM now includes a comprehensive Command Line Interface (CLI) that provides unified access to all TidyLLM functionality. The CLI serves as the primary entry point for users, offering intuitive commands for QA processing, testing, system management, and more.

## 🚀 **Quick Start**

### Installation
```bash
# Install TidyLLM package with CLI
pip install -e .

# Verify installation
tidyllm --help
```

### First Steps
```bash
# Initialize TidyLLM in your project
tidyllm init

# Check system health
tidyllm status  

# Start processing files
tidyllm qa --help
```

## 📋 **Available Commands**

### **Core Commands**

#### `tidyllm help`
Shows comprehensive help with all available commands and examples.
```bash
tidyllm help
```

#### `tidyllm version`
Displays TidyLLM version and component availability status.
```bash
tidyllm version
# Output:
# TidyLLM version 1.0.0
# The Great Walled City of Enterprise AI
# 
# Components:
#   Gateways: Available
#   Knowledge Systems: Available
#   Knowledge Server: Available
```

#### `tidyllm status`
Comprehensive system health check showing all dependencies and integrations.
```bash
tidyllm status
# Output:
# [STATUS] TidyLLM System Health Check
# ========================================
# [OK] Gateways: Available
# [OK] Gateway Registry: Initialized
# [OK] MLflow: Available (3.3.2)
# [OK] AWS SDK: Available (1.40.21)
# [OK] pandas: Available (2.3.2)
# [OK] pyyaml: Available (6.0.2)
# [OK] dspy-ai: Available (3.0.3)
# [OK] openai: Available (1.102.0)
# [OK] anthropic: Available (0.66.0)
```

#### `tidyllm init`
Initializes TidyLLM project structure in the current directory.
```bash
tidyllm init
# Creates:
# - qa_files/           # Input files directory
# - qa_reports/         # Output reports directory  
# - qa_config/          # Configuration directory
# - workflows/          # Workflow definitions
# - knowledge/          # Knowledge base
# - tidyllm_config.yaml # Main configuration file
```

#### `tidyllm config`
Shows current TidyLLM configuration and settings.
```bash
tidyllm config
```

### **QA Processing Commands**

#### `tidyllm qa`
Main QA processing command with full argument support.
```bash
# Show QA processor help
tidyllm qa --help

# Debug system configuration
tidyllm qa --debug-config

# Test AWS connection
tidyllm qa --chat-test

# Process single file
tidyllm qa --file document.xlsx

# Batch process all files
tidyllm qa --batch

# Custom experiment name
tidyllm qa --experiment "my_custom_experiment"

# Add experiment tags
tidyllm qa --tag "version=1.0" --tag "env=prod"

# Disable MLflow tracking
tidyllm qa --no-mlflow

# Verbose output
tidyllm qa --verbose
```

#### `tidyllm qa-processor`
Alias for `tidyllm qa` - identical functionality.
```bash
tidyllm qa-processor --debug-config
```

#### `tidyllm chat-pdf`
Interactive PDF chat mode for document analysis.
```bash
# Chat with a PDF document
tidyllm chat-pdf document.pdf

# Chat with PDF and custom experiment
tidyllm chat-pdf report.pdf --experiment "pdf_analysis"

# Verbose PDF chat
tidyllm chat-pdf manual.pdf --verbose
```

### **Testing & Validation Commands**

#### `tidyllm test`
Test runner with support for all 5 test scenarios.
```bash
# Show test runner help
tidyllm test --help

# Create sample test files
tidyllm test --create-samples

# Run all 5 test scenarios
tidyllm test --all

# Run specific test scenario
tidyllm test --test 1
tidyllm test --test 2
tidyllm test --test 3
tidyllm test --test 4
tidyllm test --test 5

# Verbose test output
tidyllm test --all --verbose
```

#### `tidyllm test-runner`
Alias for `tidyllm test` - identical functionality.
```bash
tidyllm test-runner --create-samples
```

#### `tidyllm validate`
Runs system validation (alias for `tidyllm status`).
```bash
tidyllm validate
```

### **Development Commands**

#### `tidyllm debug`
Debug mode routing to QA processor with debug configuration.
```bash
tidyllm debug --help
# Routes to: qa_processor.py --debug-config
```

#### `tidyllm admin`
Administrative commands (coming soon).
```bash
tidyllm admin
# Output: [INFO] Admin commands coming soon!
# For now, use: python -m tidyllm.admin
```

### **Workflow Management**

#### `tidyllm workflow`
Workflow management commands (coming soon).
```bash
tidyllm workflow
# Output: [INFO] Workflow management commands coming soon!
# For now, use: python -m tidyllm.workflows
```

#### `tidyllm demo`
Launch TidyLLM demo interface.
```bash
tidyllm demo
# Note: Requires [web] feature installation
# pip install -e .[web]
```

## 🔧 **Command Examples**

### **Common Workflows**

#### Setup New Project
```bash
# Initialize project
tidyllm init

# Check system health  
tidyllm status

# Create test samples
tidyllm test --create-samples

# Run validation tests
tidyllm test --all
```

#### Process Documents
```bash
# Single Excel file
tidyllm qa --file analysis.xlsx

# Chat with PDF
tidyllm chat-pdf report.pdf

# Batch process directory
tidyllm qa --batch --verbose
```

#### Debug and Troubleshoot
```bash
# System configuration
tidyllm qa --debug-config

# Test AWS connectivity
tidyllm qa --chat-test

# Test MLflow integration
tidyllm qa --test-mlflow

# Full system health check
tidyllm status
```

#### Experiment Tracking
```bash
# Custom experiment name
tidyllm qa --experiment "quarterly_review_2024"

# Add metadata tags
tidyllm qa --tag "department=finance" --tag "quarter=Q4"

# Disable tracking for testing
tidyllm qa --no-mlflow --verbose
```

## ⚙️ **Technical Implementation**

### **CLI Architecture**
- **Entry Point**: Direct `tidyllm` command via setuptools entry point
- **Command Routing**: Centralized routing in `tidyllm/cli.py`
- **Script Integration**: Seamless integration with existing `qa_processor.py` and `qa_test_runner.py`
- **Argument Forwarding**: Complete argument pass-through to underlying scripts

### **Installation Integration**
```python
# setup.py entry points
entry_points={
    'console_scripts': [
        'tidyllm=tidyllm.cli:main',
        'qa-processor=qa_processor:main',
        'tidyllm-demo=tidyllm.demos.launch_demo:main',
        'tidyllm-workflow=tidyllm.workflows:main'
    ]
}
```

### **Command Mapping**
| CLI Command | Underlying Script | Arguments |
|-------------|------------------|-----------|
| `tidyllm qa` | `qa_processor.py` | All qa_processor arguments |
| `tidyllm test` | `qa_test_runner.py` | All test runner arguments |
| `tidyllm chat-pdf FILE` | `qa_processor.py --chat-pdf FILE` | PDF + additional args |
| `tidyllm debug` | `qa_processor.py --debug-config` | Debug + additional args |

### **Backward Compatibility**
All existing scripts continue to work independently:
```bash
# These are equivalent:
tidyllm qa --debug-config
python qa_processor.py --debug-config

# These are equivalent:
tidyllm test --all
python qa_test_runner.py --all
```

## 📊 **Status Outputs**

### **System Health Check**
```bash
$ tidyllm status
[STATUS] TidyLLM System Health Check
========================================
[OK] Gateways: Available
[OK] Gateway Registry: Initialized
[OK] MLflow: Available (3.3.2)
  Tracking URI: file:///C:/Users/marti/github/mlruns
[OK] AWS SDK: Available (1.40.21)
[OK] pandas: Available (2.3.2)
[OK] pyyaml: Available (6.0.2)
[OK] dspy-ai: Available (3.0.3)
[OK] openai: Available (1.102.0)
[OK] anthropic: Available (0.66.0)

[RECOMMENDATION]
If any components show as 'Not available':
  pip install -e .[all]
```

### **QA Processor Debug Output**
```bash
$ tidyllm qa --debug-config
==================================================
[DEBUG CONFIG] QA Processor Configuration
==================================================

[FOLDERS]
   Watch folder: ./qa_files
   Output folder: ./qa_reports
   Config folder: ./qa_config

[MODEL DETECTION]
   Detected model: sonnet
   Process name: qa_processor
   Generated experiment: qa_processor_sonnet

[TIDYLLM STATUS]
   Gateway registry: [OK] Available

[MLFLOW STATUS]
   MLflow enabled: [OK] Yes
   MLflow installed: [OK] Yes
   Tracking URI: file:///C:/Users/marti/github/mlruns

[FILE TYPES]
   Excel types: ['.xlsx', '.xls']
   PDF types: ['.pdf']
   Excel tabs: ['core_checklist', 'custom_checklist', 'custom_prompts']
==================================================
```

## 🐛 **Troubleshooting**

### **CLI Not Found**
```bash
# Reinstall package
pip install -e .

# Check installation
which tidyllm  # Unix/Mac
where tidyllm  # Windows
```

### **Missing Dependencies**
```bash
# Install all features
pip install -e .[all]

# Check status
tidyllm status
```

### **Unicode Encoding Issues (Windows)**
All Unicode symbols have been replaced with ASCII equivalents:
- `✓` → `[OK]`
- `✗` → `[ERROR]`
- `📁` → `[FOLDERS]`
- `🤖` → `[MODEL DETECTION]`

### **Script Import Errors**
```bash
# Ensure you're in correct directory
cd /path/to/tidyllm

# Reinstall in development mode
pip install -e .
```

## 🚀 **Advanced Usage**

### **Custom Configuration**
```bash
# Initialize with custom setup
tidyllm init

# Edit configuration
# tidyllm_config.yaml gets created with:
# - QA processing settings
# - Model configurations  
# - Integration settings
# - MLflow parameters
```

### **Integration with CI/CD**
```bash
# Automated testing
tidyllm test --create-samples
tidyllm test --all --verbose

# Health monitoring
tidyllm status | grep ERROR && exit 1
```

### **Development Workflow**
```bash
# Setup development environment
pip install -e .[dev,all]

# Run tests
tidyllm test --all

# Debug configuration
tidyllm qa --debug-config

# Process sample files
tidyllm qa --batch --verbose
```

## 📈 **Performance & Features**

### **CLI Performance**
- **Fast startup**: Direct Python module execution
- **Low overhead**: Minimal CLI wrapper around existing scripts  
- **Memory efficient**: Scripts only load when needed
- **Cross-platform**: Works on Windows, macOS, Linux

### **Feature Completeness**
- ✅ **100% Argument Coverage**: All underlying script arguments supported
- ✅ **Help Integration**: Complete help text from underlying scripts
- ✅ **Error Handling**: Proper error propagation and reporting
- ✅ **Output Formatting**: Consistent output formatting across commands
- ✅ **Configuration Management**: Unified configuration system

## 🎯 **Migration Guide**

### **From Script Execution to CLI**
```bash
# Old way:
python qa_processor.py --debug-config
python qa_test_runner.py --all

# New way:
tidyllm qa --debug-config  
tidyllm test --all

# Both ways work! Complete backward compatibility.
```

### **From Manual Setup to CLI Init**
```bash
# Old way: Manual directory creation
mkdir qa_files qa_reports qa_config

# New way: Automated setup
tidyllm init
```

## 🎊 **CLI Benefits**

### **For Users**
- **Unified Interface**: Single command for all TidyLLM functionality
- **Discoverability**: `tidyllm help` shows all available options
- **Consistency**: Standardized command patterns across all features
- **Documentation**: Built-in help and examples
- **Installation**: Works immediately after `pip install -e .`

### **For Developers**  
- **Maintainability**: Centralized command routing
- **Extensibility**: Easy to add new commands
- **Integration**: Seamless integration with existing scripts
- **Testing**: CLI commands can be easily tested
- **Distribution**: Professional CLI for package distribution

### **For DevOps**
- **Automation**: All commands scriptable and CI/CD friendly
- **Monitoring**: Built-in health checks and status reporting
- **Configuration**: Unified configuration management
- **Debugging**: Comprehensive debug and diagnostic tools

## 🔗 **Related Documentation**
- [Installation Guide](INSTALLATION.md)
- [Package Success Summary](PACKAGE_SUCCESS_SUMMARY.md)  
- [QA Processor Documentation](qa_processor.py)
- [QA Test Runner Documentation](qa_test_runner.py)

---

**The TidyLLM CLI provides a complete, professional interface to the entire TidyLLM ecosystem while maintaining full backward compatibility with existing workflows!** 🚀