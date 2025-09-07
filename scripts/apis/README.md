# API Scripts

REST API implementations and CLI interfaces for TidyLLM FLOW Agreement system.

## Universal Bracket Flow System

### **api_bracket_flows.py**
REST API server for FLOW agreements:
- Provides `/api/flow/execute` endpoint for bracket commands
- Handles JSON requests with context parameters
- Returns structured execution results
- Supports real-time flow execution monitoring

### **cli_bracket_flows.py**
Command-line interface for FLOW agreements:
- Direct CLI execution of `[workflow_name]` commands
- Interactive flow discovery and help
- Batch processing capabilities
- Integration with shell scripts and automation

### **ui_bracket_flows.py**
Web UI components for FLOW agreements:
- Interactive web interface for flow execution
- Visual flow status and progress indicators
- Context parameter input forms
- Real-time execution result display

## Integration Examples

### **universal_bracket_flow_examples.py**
Comprehensive examples showing bracket flow usage:
- CLI/API/UI integration patterns
- Context passing examples
- Error handling demonstrations
- Best practices for flow development

### **improved_usage_examples.py**
Advanced usage patterns and integrations:
- Complex workflow orchestration
- Multi-step flow execution
- Integration with external systems
- Performance optimization techniques

## File Upload & Processing

### **smart_file_upload_app.py**
Intelligent file upload processing system:
- Automated document classification
- Smart routing to appropriate processing workflows
- Integration with S3 storage and embedding systems
- Real-time processing status updates

## Usage

### Start API Server
```bash
python scripts/apis/api_bracket_flows.py --port 8000
```

### CLI Flow Execution
```bash
python scripts/apis/cli_bracket_flows.py "[Integration Test]"
```

### Web UI Launch
```bash
python scripts/apis/ui_bracket_flows.py
```

These scripts provide the three-interface (CLI/API/UI) access to the TidyLLM FLOW Agreement system, enabling consistent workflow execution across all interaction modes.