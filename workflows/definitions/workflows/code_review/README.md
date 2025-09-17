# Code Review Workflow

## 📋 **Overview**

**Workflow ID**: `code_review`
**Type**: `Unknown`
**Priority**: `critical`
**Status**: `active`

Workflow for code_review processing

## 🚀 **Quick Start**

### **CLI Usage**
```bash
# List available workflows
python workflow_cli.py list --name code_review

# Get workflow details
python workflow_cli.py info code_review

# Execute via Boss Portal
streamlit run unified_boss_portal.py
# Navigate to: 📝 Template Creator → Code Review
```

### **Python API Usage**
```python
from workflow_registry_system import WorkflowRegistrySystem
from intelligent_document_intake import IntelligentDocumentIntake

# Initialize systems
registry = WorkflowRegistrySystem()
intake = IntelligentDocumentIntake(registry)

# Route document to workflow
matches = intake.analyze_and_route_document("path/to/document.pdf")
for match in matches:
    if match.workflow_id == "code_review":
        print(f"Confidence: {match.confidence_score:.2f}")
        print(f"Recommended: {match.recommended}")
```

### **Bracket Commands** *(Optional - if enabled in settings)*
```bash
# CLI bracket command
tidyllm '[code_review]'

# API bracket command
curl -X POST http://localhost:8000/flow \\
  -H "Content-Type: application/json" \\
  -d '{"command": "[code_review]"}'

# S3 trigger file
aws s3 cp triggers/[code_review].trigger s3://workflow-bucket/triggers/
```

## ⚙️ **Configuration Options**

### **Template Variables**
{TEMPLATE_FIELDS_TABLE}

### **Criteria Settings**
{CRITERIA_SETTINGS}

### **Processing Options**
```yaml
processing_strategy: "multi_perspective"
priority_level: "critical"
estimated_duration_hours: 1.0

# Model Configuration
default_model: "anthropic.claude-3-sonnet-20240229-v1:0"
alternative_models:
  - "anthropic.claude-3-haiku-20240307-v1:0"
  - "anthropic.claude-3-opus-20240229-v1:0"
```

## 📊 **Input Requirements**

### **Document Types**
- PDF: Primary document format
- DOCX: Word documents
- XLSX: Spreadsheet data

### **Required Fields**
- Document identifier
- Content sections
- Processing date

### **Optional Fields**
- Metadata
- Tags
- Custom fields

### **Document Criteria**
```json
{
  "keywords": [
    "code_review"
  ],
  "patterns": [
    "Document:",
    "Date:"
  ],
  "min_pages": 1
}
```

## 📄 **Output Structure**

### **Generated Files**
- **Report**: `code_review_report_{timestamp}.pdf`
- **Analysis**: `code_review_analysis_{timestamp}.json`
- **Metadata**: `code_review_metadata_{timestamp}.json`

### **Output Fields**
| Field | Description |
|-------|-------------|
| result | Processing result |
| status | Execution status |

### **Quality Metrics**
- Processing Time: Variable
- Accuracy: Workflow dependent
- Coverage: Based on criteria

## 🔧 **Advanced Usage**

### **Custom Configuration**
```python
from workflow_registry_system import WorkflowRegistrySystem

registry = WorkflowRegistrySystem()
workflow = registry.get_workflow("code_review")

# Customize processing
workflow.processing_strategy = "multi_perspective"
workflow.priority_level = "high"

# Add custom fields
workflow.add_custom_field("custom_field", "Custom Value")

# Execute with overrides
result = workflow.execute(
    input_file="document.pdf",
    custom_experiment="custom_analysis_2024",
    model_override="claude-3-opus"
)
```

### **Batch Processing**
```bash
# Process multiple documents
python intelligent_document_intake.py \\
  --batch-process ./input_folder \\
  --workflow code_review \\
  --output ./output_folder \\
  --parallel 3
```

### **Integration with Boss Portal**
```python
# Access via Boss Portal workflow system
from unified_boss_portal import UnifiedSystemOrchestrator

orchestrator = UnifiedSystemOrchestrator()
result = orchestrator.execute_workflow(
    workflow_id="code_review",
    input_data=input_data,
    options={
        "model": "claude-3-sonnet",
        "experiment": "production_run",
        "tags": ["automated", "v1.0"]
    }
)
```

## 🛠️ **Development**

### **Template Editing**
```bash
# Edit via Boss Portal
streamlit run unified_boss_portal.py
# Navigate to: 📝 Template Creator → Code Review → Edit Template

# Direct file editing
nano workflow_registry/workflows/code_review/templates/code_review_template.md
```

### **Criteria Editing**
```bash
# Edit via Boss Portal
streamlit run unified_boss_portal.py
# Navigate to: 📝 Template Creator → Code Review → Edit Criteria

# Direct JSON editing (with JSON scrubbing)
nano workflow_registry/workflows/code_review/criteria/code_review_criteria.json
```

### **Testing**
```python
# Test workflow execution
from workflow_registry_system import WorkflowRegistrySystem

registry = WorkflowRegistrySystem()
workflow = registry.get_workflow("code_review")

# Validate workflow
validation_result = workflow.validate()
print(f"Valid: {validation_result.is_valid}")
print(f"Issues: {validation_result.issues}")

# Test with sample data
test_result = workflow.test_execute(sample_document="test.pdf")
print(f"Success: {test_result.success}")
print(f"Output: {test_result.output_path}")
```

## 📖 **API Reference**

### **REST API Endpoints**

#### Execute Workflow
```http
POST /api/v1/workflows/code_review/execute
Content-Type: application/json

{
  "input_file": "document.pdf",
  "options": {
    "model": "claude-3-sonnet",
    "experiment": "api_test",
    "tags": ["api", "automated"]
  }
}
```

#### Get Workflow Status
```http
GET /api/v1/workflows/code_review/status/{execution_id}
```

#### Update Workflow Configuration
```http
PUT /api/v1/workflows/code_review/config
Content-Type: application/json

{
  "processing_strategy": "multi_perspective",
  "priority_level": "high",
  "model_override": "claude-3-opus"
}
```

### **Python API Classes**

#### WorkflowExecutor
```python
from workflow_registry_system import WorkflowExecutor

executor = WorkflowExecutor()
result = executor.execute_workflow(
    workflow_id="code_review",
    input_file="document.pdf",
    **options
)
```

#### IntelligentDocumentIntake
```python
from intelligent_document_intake import IntelligentDocumentIntake

intake = IntelligentDocumentIntake()
matches = intake.analyze_document("document.pdf")
best_match = intake.get_best_workflow_match(matches)
```

## 🔍 **Troubleshooting**

### **Common Issues**

#### Workflow Not Found
```bash
# Check workflow is registered
python workflow_cli.py list --all

# Verify folder structure
python workflow_cli.py structure
```

#### Template Rendering Errors
```python
# Check template fields
workflow = registry.get_workflow("code_review")
missing_fields = workflow.validate_template()
print(f"Missing fields: {missing_fields}")
```

#### Model Access Issues
```bash
# Test model connectivity
python qa_processor.py --chat-test

# Check model configuration
tidyllm config
```

### **Debug Mode**
```bash
# Enable verbose logging
python intelligent_document_intake.py \\
  --workflow code_review \\
  --debug \\
  --verbose

# Check system status
tidyllm status
```

## 📋 **Validation & Testing**

### **Document Validation**
```python
# Check if document qualifies for workflow
from intelligent_document_intake import IntelligentDocumentIntake

intake = IntelligentDocumentIntake()
profile = intake.analyze_document_profile("test.pdf")

# Check match criteria
matches = intake.match_document_to_workflows(profile)
workflow_match = next(
    (m for m in matches if m.workflow_id == "code_review"),
    None
)

if workflow_match:
    print(f"Confidence: {workflow_match.confidence_score}")
    print(f"Criteria met: {workflow_match.criteria_met}")
    print(f"Missing: {workflow_match.criteria_failed}")
else:
    print("Document does not qualify for this workflow")
```

### **Performance Testing**
```bash
# Benchmark workflow execution
python workflow_benchmark.py \\
  --workflow code_review \\
  --sample-docs 10 \\
  --iterations 3 \\
  --report benchmark_report.json
```

## 🏢 **Enterprise Integration**

### **Corporate Compliance**
- **Audit Trail**: All executions logged to MLflow
- **Data Security**: S3 server-side encryption enabled
- **Access Control**: Role-based permissions via Boss Portal
- **Credential Management**: UnifiedSessionManager integration

### **Monitoring & Metrics**
```python
# View execution metrics
from workflow_registry_system import WorkflowRegistrySystem

registry = WorkflowRegistrySystem()
stats = registry.get_workflow_stats()
workflow_stats = stats["workflows"]["code_review"]

print(f"Success rate: {workflow_stats['success_rate']:.1%}")
print(f"Avg duration: {workflow_stats['avg_duration']:.2f}s")
print(f"Total executions: {workflow_stats['total_executions']}")
```

### **Integration Points**
- **Boss Portal**: Streamlit web interface
- **CLI Tools**: Command line automation
- **REST API**: Programmatic access
- **S3 Triggers**: Event-driven processing
- **MLflow**: Experiment tracking
- **PostgreSQL**: Metadata storage

## 📚 **References**

### **Related Documentation**
- [Workflow Registry Documentation](../WORKFLOW_REGISTRY_DOCUMENTATION.md)
- [Boss Portal User Guide](../unified_boss_portal.py)
- [CLI Quick Reference](../docs/CLI_QUICK_REFERENCE.md)
- [API Patterns](../docs/API_PATTERNS.md)

### **Configuration Files**
- **Template**: `workflow_registry/workflows/code_review/templates/code_review_template.md`
- **Criteria**: `workflow_registry/workflows/code_review/criteria/code_review_criteria.json`
- **Resources**: `workflow_registry/workflows/code_review/resources/fields.json`
- **Outputs**: `workflow_registry/workflows/code_review/outputs/`

### **Support**
- **Issues**: Report problems via Boss Portal → 🔧 Settings
- **Documentation**: [Comprehensive docs](../docs/)
- **Status**: Check system health via `tidyllm status`

---

**Generated by Boss Portal Template Creator** ⚡
**Last Updated**: 2025-09-14
**Workflow Version**: 1.0