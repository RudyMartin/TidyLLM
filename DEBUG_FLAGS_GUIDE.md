# Debug Flags Guide for QA Processor

## Overview
The QA Processor includes several debug flags to help troubleshoot connections, configurations, and functionality.

## 🔧 **Debug Flags Available**

### 1. **`--chat-test` - AWS Connection & Model Chat Test**
Tests AWS Bedrock connection and allows interactive chat with the default model.

```bash
python qa_processor.py --chat-test
```

**What it does:**
- ✅ Tests TidyLLM gateway connection
- ✅ Detects configured model (sonnet, haiku, opus, etc.)
- ✅ Tests AWS Bedrock connectivity with a simple query
- ✅ Opens interactive chat session with the model
- ✅ Type 'quit' to exit chat

**Example Output:**
```
[CHAT TEST] Testing AWS Connection & Model Chat
==================================================
🔍 [STEP 1/4] Testing TidyLLM Gateway Connection...
   ✓ TidyLLM gateways initialized
🔍 [STEP 2/4] Detecting Model Configuration...
   ✓ Model detected: sonnet
🔍 [STEP 3/4] Testing AWS Bedrock Connection...
   ✓ AWS Response: AWS connection successful
🔍 [STEP 4/4] Interactive Chat Test...
   Type 'quit' to exit chat test
   ----------------------------------------

💬 You: Hello, are you working?
🤖 sonnet: Hello! Yes, I'm working perfectly. The AWS connection is active and I'm ready to help with your QA processing tasks.

💬 You: quit
✓ [SUCCESS] Chat test completed successfully!
```

### 2. **`--debug-config` - Configuration & Model Detection**
Shows detailed configuration info and model detection results.

```bash
python qa_processor.py --debug-config
```

**What it displays:**
- 📁 Folder configurations (watch, output, config)
- 🤖 Model detection results and experiment naming
- 🔧 TidyLLM gateway status
- 📊 MLflow configuration and connection
- 📝 Supported file types and Excel tab requirements

**Example Output:**
```
[DEBUG CONFIG] QA Processor Configuration
==================================================

📁 FOLDERS:
   Watch folder: ./qa_files
   Output folder: ./qa_reports
   Config folder: ./qa_config

🤖 MODEL DETECTION:
   Detected model: sonnet
   Process name: qa_processor
   Generated experiment: qa_processor_sonnet

🔧 TIDYLLM STATUS:
   Gateway registry: ✓ Available

📊 MLFLOW STATUS:
   MLflow enabled: ✓ Yes
   MLflow installed: ✓ Yes
   Tracking URI: file:///C:/Users/user/mlruns

📝 FILE TYPES:
   Excel types: ['.xlsx', '.xls']
   PDF types: ['.pdf']
   Excel tabs: ['core_checklist', 'custom_checklist', 'custom_prompts']
```

### 3. **`--test-mlflow` - MLflow Connection Test**
Tests MLflow connection and logging functionality.

```bash
python qa_processor.py --test-mlflow
```

**What it does:**
- ✅ Checks MLflow module installation
- ✅ Tests experiment initialization 
- ✅ Tests metric/parameter logging
- ✅ Creates a debug experiment with test data

**Example Output:**
```
[MLFLOW TEST] Testing MLflow Connection & Logging
==================================================
✓ [STEP 1/3] MLflow module imported successfully
✓ [STEP 2/3] Test experiment initialized: debug_test_sonnet
✓ [STEP 3/3] Test experiment logged successfully

📊 MLflow UI: http://localhost:5000
   Experiment: debug_test_sonnet
✓ [SUCCESS] MLflow test completed!
```

### 4. **`--verbose` - Enable Verbose Output**
Enables detailed debug output throughout processing.

```bash
python qa_processor.py --verbose --file document.pdf
```

**Combined with other operations:**
```bash
# Verbose batch processing
python qa_processor.py --verbose --batch

# Verbose with custom experiment
python qa_processor.py --verbose --experiment "DebugTest" --file test.pdf
```

## 🚀 **Common Debug Scenarios**

### **Scenario 1: Connection Issues**
```bash
# Quick connection test
python qa_processor.py --chat-test

# If chat test fails, check config
python qa_processor.py --debug-config
```

### **Scenario 2: MLflow Not Working** 
```bash
# Test MLflow specifically
python qa_processor.py --test-mlflow

# If fails, install MLflow
pip install mlflow
```

### **Scenario 3: Model Detection Issues**
```bash
# Check what model is detected
python qa_processor.py --debug-config

# Look for "MODEL DETECTION" section
# Should show: sonnet, haiku, opus, claude2, or instant
```

### **Scenario 4: Troubleshooting Processing**
```bash
# Run with verbose output
python qa_processor.py --verbose --file problematic_document.pdf

# Combine with no MLflow to isolate issues  
python qa_processor.py --verbose --no-mlflow --file test.pdf
```

### **Scenario 5: Full System Check**
```bash
# Run all debug tests in sequence
python qa_processor.py --debug-config
python qa_processor.py --test-mlflow  
python qa_processor.py --chat-test
```

## 🔍 **Debug Flag Combinations**

### **Basic Setup Verification:**
```bash
python qa_processor.py --debug-config --verbose
```

### **Full Connectivity Test:**
```bash
python qa_processor.py --chat-test --verbose
```

### **MLflow + Model Testing:**
```bash
python qa_processor.py --test-mlflow --experiment "DebugSession"
```

### **Silent Testing (No MLflow):**
```bash
python qa_processor.py --no-mlflow --debug-config
```

## 📋 **Troubleshooting Checklist**

**If `--chat-test` fails:**
1. Check AWS credentials are configured
2. Verify TidyLLM installation: `pip install tidyllm`
3. Check `tidyllm/admin/settings.yaml` exists
4. Verify Bedrock model access in AWS account

**If `--debug-config` shows issues:**
1. Missing folders will be created automatically
2. Model detection fallback is 'sonnet'
3. Check file permissions on config directories

**If `--test-mlflow` fails:**
1. Install MLflow: `pip install mlflow`
2. Check database connectivity (if using remote MLflow)
3. Verify write permissions for MLflow tracking

## 💡 **Pro Tips**

1. **Always start with:** `--debug-config` to see system state
2. **For AWS issues:** Use `--chat-test` to verify end-to-end connectivity  
3. **For MLflow issues:** Use `--test-mlflow` before processing files
4. **Use `--verbose`** with any operation for detailed logging
5. **Combine flags** for comprehensive debugging

## 🎯 **Quick Debug Commands**

```bash
# System health check
python qa_processor.py --debug-config

# AWS connectivity check  
python qa_processor.py --chat-test

# MLflow functionality check
python qa_processor.py --test-mlflow

# Process with full debugging
python qa_processor.py --verbose --file test.pdf

# Silent mode (no MLflow tracking)
python qa_processor.py --no-mlflow --verbose --file test.pdf
```

These debug flags make it easy to troubleshoot issues and verify that all components (AWS, TidyLLM, MLflow) are working correctly before processing important QA files!