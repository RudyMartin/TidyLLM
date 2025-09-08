# MLflow Experimentation Guide for QA Processor

## Overview
The QA Processor now includes **MLflow experiment tracking** to help customers experiment with different configurations and analyze their QA processing results.

## MLflow Fields Available for Customer Experimentation

### 1. **Automatic Experiment Tracking**
Every PDF processing run automatically logs:

**Metrics:**
- `overall_score` - Overall QA score (0.0 to 1.0)
- `core_checks_passed` - Number of core checklist items passed
- `core_checks_total` - Total core checklist items
- `custom_checks_passed` - Number of custom checklist items passed  
- `custom_checks_total` - Total custom checklist items
- `core_pass_rate` - Core checklist pass percentage
- `custom_pass_rate` - Custom checklist pass percentage
- `analysis_time_seconds` - Processing time
- `pages_analyzed` - Number of pages processed

**Parameters:**
- `pdf_file` - Name of PDF file processed
- `config_source` - Source Excel file used for configuration
- `processing_mode` - Processing mode used (tidyllm vs basic)
- `excel_tabs_count` - Number of Excel tabs processed
- `core_checklist_items` - Number of core checklist items
- `custom_checklist_items` - Number of custom checklist items  
- `custom_prompts_count` - Number of custom prompts used

**Tags:**
- `experiment_type` - Always "qa_processing"
- `file_type` - Always "pdf" 
- `processing_date` - Date of processing
- `tidyllm_enabled` - Whether TidyLLM was available

### 2. **Customer Experimentation Commands**

#### Basic Usage (Default Experiment):
```bash
python qa_processor.py --file document.pdf
```
*Logs to experiment: "qa_processor_sonnet" (format: processname_shortmodelname)*

#### Custom Experiment Name:
```bash
python qa_processor.py --experiment "MyCustomExperiment" --file document.pdf
```
*Creates/uses experiment: "MyCustomExperiment"*

#### Add Custom Experiment Tags:
```bash
python qa_processor.py --tag client=ACME --tag version=v2.1 --file document.pdf
```
*Adds custom tags for tracking different client configurations*

#### Disable MLflow for Testing:
```bash
python qa_processor.py --no-mlflow --file document.pdf
```
*Runs without MLflow tracking*

#### Batch Experimentation:
```bash
python qa_processor.py --experiment "BatchTest_v1" --tag iteration=1 --batch
```
*Processes all files with custom experiment tracking*

### 3. **Advanced Experimentation Scenarios**

#### A/B Testing Different Excel Configurations:
```bash
# Test Configuration A
python qa_processor.py --experiment "ConfigTest" --tag config=A --file configA.xlsx
python qa_processor.py --experiment "ConfigTest" --tag config=A --file document1.pdf

# Test Configuration B  
python qa_processor.py --experiment "ConfigTest" --tag config=B --file configB.xlsx
python qa_processor.py --experiment "ConfigTest" --tag config=B --file document1.pdf
```

#### Client-Specific Experiments:
```bash
# Client 1 experiment
python qa_processor.py --experiment "Client_ACME" --tag client=ACME --tag phase=pilot --batch

# Client 2 experiment
python qa_processor.py --experiment "Client_TechCorp" --tag client=TechCorp --tag phase=production --batch
```

#### Performance Testing:
```bash
python qa_processor.py --experiment "PerformanceTest" --tag test_type=speed --tag file_size=large --file big_document.pdf
```

### 4. **Automatic Experiment Naming: processname_shortmodelname**

The QA Processor automatically generates experiment names in the format: `processname_shortmodelname`

#### **Model Detection:**
- **Automatic**: Reads from `tidyllm/admin/settings.yaml` → `bedrock.default_model`
- **Model Mapping**:
  - `anthropic.claude-3-sonnet-*` → `sonnet`
  - `anthropic.claude-3-haiku-*` → `haiku` 
  - `anthropic.claude-3-opus-*` → `opus`
  - `anthropic.claude-v2*` → `claude2`
  - `anthropic.claude-instant-*` → `instant`
- **Fallback**: Uses `sonnet` if detection fails

#### **Example Generated Names:**
- `qa_processor_sonnet` (Claude 3 Sonnet)
- `qa_processor_haiku` (Claude 3 Haiku)
- `qa_processor_opus` (Claude 3 Opus)

### 5. **MLflow Configuration Options**

#### In Code (qa_processor.py):
```python
QA_CONFIG = {
    'mlflow_enabled': True,  # Enable/disable MLflow
    'process_name': 'qa_processor',  # Base process name
    'default_model_short_name': 'sonnet',  # Default short model name
    'mlflow_tracking_uri': None,  # Uses environment/default
    'experiment_tags': {}  # Global tags for all runs
}
# Generates experiment names like: qa_processor_sonnet, qa_processor_haiku, etc.
```

#### Environment Variables:
```bash
# Set MLflow tracking server
export MLFLOW_TRACKING_URI="http://your-mlflow-server:5000"

# Use database backend
export MLFLOW_TRACKING_URI="postgresql://user:pass@host:port/db"
```

### 5. **Viewing Experiment Results**

#### Start MLflow UI:
```bash
mlflow ui
```
*Access at: http://localhost:5000*

#### View Specific Experiment:
```bash
mlflow experiments list
mlflow runs list --experiment-id 1
```

### 6. **Common Experiment Patterns**

#### Quality Threshold Testing:
- Process same document with different Excel configurations
- Compare `overall_score` metrics
- Track `core_pass_rate` and `custom_pass_rate` trends

#### Processing Speed Optimization:
- Test with different file sizes
- Monitor `analysis_time_seconds`
- Compare `tidyllm_enabled=True` vs `tidyllm_enabled=False`

#### Configuration Impact Analysis:
- Vary `custom_prompts_count` 
- Track impact on `overall_score`
- Analyze `custom_checks_passed` vs `custom_checks_total`

### 7. **Integration with TidyLLM System**

The QA Processor MLflow tracking integrates with the broader TidyLLM MLflow infrastructure:

- **Uses same database backend** as other TidyLLM experiments
- **Compatible with existing MLflow workflows** in the system
- **Follows same experiment naming conventions** as drop_zones workflows
- **Leverages UnifiedSessionManager** MLflow configuration when available

### 8. **Customer Benefits**

**Easy Experimentation:**
- No complex setup required
- Simple command-line flags
- Automatic metric collection

**Performance Insights:**
- Track processing speeds across different configurations
- Compare quality scores between Excel setups
- Monitor trends over time

**Client Management:**
- Separate experiments per client
- Tag-based filtering and analysis
- Historical performance tracking

**Quality Optimization:**
- A/B test different prompt configurations
- Optimize checklist effectiveness
- Track improvement over iterations

---

## Quick Start for Customers

1. **Install MLflow** (if not already installed):
   ```bash
   pip install mlflow
   ```

2. **Run your first experiment**:
   ```bash
   python qa_processor.py --experiment "MyFirstTest" --file sample.pdf
   ```

3. **View results**:
   ```bash
   mlflow ui
   ```

4. **Start experimenting** with different tags, configurations, and file sets!

The MLflow integration makes it easy for customers to systematically test and optimize their QA processing workflows while maintaining the simplicity of the single-script approach.