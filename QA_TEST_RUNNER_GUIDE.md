# QA Test Runner - Single Script Guide

## Overview
Single script implementation of all 5 QA test scenarios with automatic file creation, processing, and reporting.

**Formula**: `prompt_source + file_collection → response_type + expected_experiment_tag`

## 🚀 **Quick Start**

### 1. Create Sample Files
```bash
python qa_test_runner.py --create-samples
```
Creates all necessary test files in `./qa_test_files/`

### 2. Run All Tests  
```bash
python qa_test_runner.py --all
```
Executes all 5 test scenarios and generates comprehensive reports

### 3. Run Specific Test
```bash
python qa_test_runner.py --test 1
python qa_test_runner.py --test 2
# ... etc
```

## 📋 **Test Scenarios Implemented**

| Test # | Prompt Source | File Collection | Response Type | Expected Experiment Tag | Status |
|--------|---------------|----------------|---------------|-------------------------|--------|
| **1** | *Config only* | `qa_models.json` + `mvr.pdf` | stub echo of model + experiment_tag | `smoke_chat_haiku-3` | ✅ |
| **2** | *Default core only* | `mvr.pdf` | stub echo + PDF text injection | `pdf_mvr_review_sonnet-3-5` | ✅ |
| **3** | *Markdown (MD)* | `mvr.pdf`, `custom_prompts.md`, `custom_checklist.md` | stub echo + parsed MD prompts/questions | `pdf_mvr_review_sonnet-3-5` | ✅ |
| **4** | *Excel (3 tabs)* | `mvr.pdf`, `checklist.xlsx` | stub echo + Excel-driven questions/prompts | `pdf_mvr_review_sonnet-3-5` | ✅ |
| **5** | *Excel + Override* | `mvr.pdf`, `checklist.xlsx`, `qa_models.json` | stub echo with non-default model + experiment_tag | `pilot_excel_prompts_llama3-70b` | ✅ |

## 🔧 **Key Features Implemented**

### **Automatic Model Detection:**
- Reads from `qa_models.json` for overrides
- Falls back to `tidyllm/admin/settings.yaml`
- Maps full model IDs to short names:
  - `anthropic.claude-3-haiku-*` → `haiku-3`
  - `anthropic.claude-3-sonnet-*` → `sonnet-3-5` 
  - `meta.llama3-70b-*` → `llama3-70b`

### **Experiment Tag Generation:**
- Format: `{process_name}_{model_short}`
- Examples: `smoke_chat_haiku-3`, `pilot_excel_prompts_llama3-70b`

### **File Processing:**
- **PDF**: Text extraction (stub implementation)
- **Markdown**: Prompts and questions parsing
- **Excel**: 3-tab parsing (core_checklist, custom_checklist, custom_prompts)
- **JSON**: Model configuration overrides

### **MLflow Integration:**
- Automatic experiment creation with generated tags
- Comprehensive metrics and parameter logging
- Session tracking and results archival

### **Snapshot System:**
- All source files copied to `./qa_test_extracts/test_N/`
- Preserves exact state of inputs for each test
- Enables audit trail and reproducibility

## 📊 **Generated Files Structure**

```
./qa_test_files/           # Input files
├── qa_models_test1.json   # Test 1 config
├── qa_models_test5.json   # Test 5 override
├── custom_prompts.md      # Test 3 markdown
├── custom_checklist.md    # Test 3 markdown
├── checklist.xlsx         # Test 4 & 5 Excel
└── mvr.pdf               # All tests PDF

./qa_test_reports/         # Output reports
├── test_1_report_YYYYMMDD_HHMMSS.md
├── test_2_report_YYYYMMDD_HHMMSS.md
├── test_3_report_YYYYMMDD_HHMMSS.md
├── test_4_report_YYYYMMDD_HHMMSS.md
├── test_5_report_YYYYMMDD_HHMMSS.md
└── qa_test_summary_YYYYMMDD_HHMMSS.md

./qa_test_extracts/        # File snapshots
├── test_1/               # Snapshots for test 1
├── test_2/               # Snapshots for test 2
├── test_3/
│   ├── prompts.json      # Parsed MD prompts
│   └── questions.json    # Parsed MD questions
├── test_4/
└── test_5/
```

## 🎯 **Test Details**

### **Test 1: Config Only**
- **Proves**: Model resolution from `qa_models.json`
- **Config**: `{"process_name": "smoke_chat", "model_id": "anthropic.claude-3-haiku-20240307-v1:0"}`
- **Expected**: `smoke_chat_haiku-3`
- **Response**: Stub echo showing model and experiment tag

### **Test 2: Default Core Only**  
- **Proves**: PDF text injection with admin defaults
- **Uses**: Default admin Excel configuration
- **Expected**: `pdf_mvr_review_sonnet-3-5`
- **Response**: Stub echo + extracted PDF text

### **Test 3: Markdown**
- **Proves**: Markdown parsing into prompts.json & questions.json
- **Parses**: Custom prompts and checklist from MD files
- **Expected**: `pdf_mvr_review_sonnet-3-5`
- **Response**: Stub echo + parsed MD content

### **Test 4: Excel**
- **Proves**: 3-tab Excel processing
- **Tabs**: core_checklist, custom_checklist, custom_prompts
- **Expected**: `pdf_mvr_review_sonnet-3-5`
- **Response**: Stub echo + Excel-driven configuration

### **Test 5: Excel + Override**
- **Proves**: Per-REV model override + custom experiment tag
- **Config**: `{"process_name": "pilot_excel_prompts", "model_id": "meta.llama3-70b-instruct-v1:0"}`
- **Expected**: `pilot_excel_prompts_llama3-70b`
- **Response**: Same as Test 4 but with override model

## 🔬 **MLflow Experiment Tracking**

Each test logs comprehensive metrics:

### **Common Metrics:**
- `test_completion`: 1.0 (success indicator)
- Test-specific metrics (parsing success, item counts, etc.)

### **Common Parameters:**
- `test_type`: Scenario identifier
- `model_short`: Detected model short name
- File paths and processing metadata

### **Common Tags:**
- `test_number`: Test scenario number
- `experiment_tag`: Generated experiment tag
- `test_type`: "qa_scenario"

## 🎨 **Advanced Features**

### **Smart Model Detection:**
```python
def extract_model_short_name(self, model_id):
    if 'haiku' in model_id_lower:
        return 'haiku-3'
    elif 'sonnet' in model_id_lower:
        return 'sonnet-3-5'
    # ... handles all model types
```

### **Flexible File Processing:**
- **Markdown Parser**: Extracts prompts and questions from MD structure
- **Excel Parser**: Handles 3-tab configuration with pandas
- **JSON Config**: Simple override mechanism
- **PDF Extraction**: Stub implementation ready for real PDF libraries

### **Comprehensive Reporting:**
Each test generates detailed markdown reports with:
- Test metadata and experiment tags
- Complete stub responses
- Processing details and file snapshots
- Technical configuration information

## 💡 **Usage Examples**

### **Development Testing:**
```bash
# Create samples and run single test
python qa_test_runner.py --create-samples
python qa_test_runner.py --test 1 --verbose
```

### **Full Validation:**
```bash
# Run complete test suite
python qa_test_runner.py --all --verbose
```

### **Custom Files:**
```bash
# Create samples, modify files, then test
python qa_test_runner.py --create-samples
# ... edit files in ./qa_test_files/ ...
python qa_test_runner.py --test 5
```

## 🔧 **Dependencies**

```bash
pip install tidyllm pandas openpyxl pyyaml mlflow
```

## ✅ **Validation Results**

The script validates:
- ✅ Model resolution from config files
- ✅ Experiment tag generation with correct format
- ✅ PDF text extraction and injection
- ✅ Markdown parsing into structured data
- ✅ Excel 3-tab configuration processing
- ✅ Model override functionality
- ✅ MLflow experiment logging
- ✅ File snapshot creation
- ✅ Comprehensive report generation

**Ready for production testing with real files and live TidyLLM/MLflow integration!** 🚀