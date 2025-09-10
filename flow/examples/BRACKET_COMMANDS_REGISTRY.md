# Bracket Commands Registry
*Complete registry of available FLOW bracket commands with examples and usage*

## Overview
FLOW (Flexible Logic Operations Workflows) uses universal bracket syntax `[Command Name]` to trigger intelligent document processing workflows. Each bracket command maps to specific templates and processing strategies.

## Available Bracket Commands

### 📋 Quality Assurance & Compliance

#### `[Process MVR]`
**Purpose**: Process Model Validation Report through complete compliance workflow  
**Templates**: mvr_analysis + qa_control  
**Processing**: Multi-perspective analysis  
**Priority**: High  
**Example**: 
```bash
# Process MVR document with full compliance checking
[Process MVR] /path/to/mvr_document.pdf
```

#### `[Check MVS Compliance]`
**Purpose**: Validate document against MVS 5.4.3 requirements  
**Templates**: compliance_review + qa_control  
**Processing**: Single template with validation  
**Priority**: High  
**Example**:
```bash
# Check specific MVS compliance requirements
[Check MVS Compliance] /path/to/document.pdf
```

#### `[Quality Check]`
**Purpose**: Quality assurance review and validation workflow  
**Templates**: qa_control  
**Processing**: Single template  
**Priority**: Normal  
**Example**:
```bash
# Run quality assurance checks
[Quality Check] /path/to/document.pdf
```

#### `[Compliance Check]`
**Purpose**: Regulatory compliance analysis and risk assessment  
**Templates**: compliance_review + qa_control  
**Processing**: Multi-perspective  
**Priority**: High  
**Example**:
```bash
# Full regulatory compliance analysis
[Compliance Check] /path/to/regulatory_document.pdf
```

### 📊 Document Analysis

#### `[Financial Analysis]`
**Purpose**: Comprehensive financial document analysis and risk assessment  
**Templates**: financial_analysis + qa_control  
**Processing**: Single template with QA  
**Priority**: Normal  
**Example**:
```bash
# Analyze financial statements or reports
[Financial Analysis] /path/to/financial_report.pdf
```

#### `[Contract Review]`
**Purpose**: Legal contract review with compliance validation  
**Templates**: contract_analysis + compliance_review  
**Processing**: Multi-perspective  
**Priority**: High  
**Example**:
```bash
# Review legal contracts and agreements
[Contract Review] /path/to/contract.pdf
```

#### `[Data Extraction]`
**Purpose**: Structured data extraction and processing workflow  
**Templates**: data_extraction  
**Processing**: Single template  
**Priority**: Normal  
**Example**:
```bash
# Extract structured data from documents
[Data Extraction] /path/to/data_document.pdf
```

#### `[Document Section View]`
**Purpose**: Structured document section analysis for interactive browsing  
**Templates**: document_section_view  
**Processing**: Single template  
**Priority**: Normal  
**Example**:
```bash
# Create interactive document section view
[Document Section View] /path/to/document.pdf
```

### 🔬 Advanced Analysis

#### `[Peer Review]`
**Purpose**: Expert peer review and professional validation  
**Templates**: peer_review + qa_control  
**Processing**: Multi-perspective  
**Priority**: Critical  
**Example**:
```bash
# Get expert peer review for critical documents
[Peer Review] /path/to/research_paper.pdf
```

#### `[Hybrid Analysis]`
**Purpose**: Multi-dimensional document analysis combining multiple analytical perspectives  
**Templates**: hybrid_analysis + qa_control  
**Processing**: Hybrid analysis  
**Priority**: High  
**Example**:
```bash
# Complex multi-framework analysis
[Hybrid Analysis] /path/to/complex_document.pdf
```

### ⚡ System Operations

#### `[Performance Test]`
**Purpose**: Run comprehensive performance benchmark operations  
**Templates**: qa_control + data_extraction  
**Processing**: Single template  
**Priority**: Normal  
**Example**:
```bash
# Test system performance
[Performance Test] /path/to/test_document.pdf
```

#### `[Integration Test]`
**Purpose**: Test integration between components and external systems  
**Templates**: qa_control  
**Processing**: Single template  
**Priority**: Normal  
**Example**:
```bash
# Test system integration
[Integration Test] /path/to/integration_test.pdf
```

#### `[Cost Analysis]`
**Purpose**: Analyze cost patterns and optimization opportunities  
**Templates**: financial_analysis  
**Processing**: Single template  
**Priority**: Normal  
**Example**:
```bash
# Analyze costs and optimizations
[Cost Analysis] /path/to/cost_report.pdf
```

#### `[Error Analysis]`
**Purpose**: Analyze error patterns and failure modes  
**Templates**: qa_control  
**Processing**: Single template  
**Priority**: High  
**Example**:
```bash
# Analyze system errors and patterns
[Error Analysis] /path/to/error_log.pdf
```

## Usage Patterns

### CLI Usage
```bash
# Direct command line usage
tidyllm flow "[Process MVR]" /path/to/document.pdf

# With additional context
tidyllm flow "[Financial Analysis]" /path/to/report.pdf --priority critical --user-context "Q3_review"
```

### API Usage

#### New Flow Integration API
```http
POST /api/v1/flow/execute
Content-Type: application/json

{
  "bracket_command": "[Contract Review]",
  "document_path": "/path/to/contract.pdf",
  "business_priority": "high",
  "user_context": {
    "department": "legal",
    "review_type": "initial"
  }
}
```

#### Existing Chain Operations API
```http
# Document chain operations (existing implementation)
POST /chains/execute
Content-Type: application/json

{
  "operations": ["ingest", "embed", "index"],
  "domain": "legal_contracts",
  "source": "/path/to/contract.pdf",
  "execution_mode": "auto",
  "config": {
    "batch_size": 10,
    "embedding_model": "tfidf"
  }
}

# Document query (existing implementation)
POST /chains/query
Content-Type: application/json

{
  "domain": "legal_contracts",
  "question": "What are the key terms in this contract?",
  "limit": 5,
  "similarity_threshold": 0.7
}

# System status (existing implementation)
GET /chains/status
GET /chains/status/{operation_id}
```

#### Universal Bracket Flow API
```http
# Universal bracket command execution
POST /flow/execute
Content-Type: application/json

{
  "command": "[mvr_analysis]",
  "context": {
    "user_id": "api_user",
    "session_id": "sess123",
    "priority": "high"
  }
}

# Response format
{
  "execution_id": "uuid-string",
  "status": "started|running|completed|failed",
  "workflow_name": "mvr_analysis",
  "started_at": "2024-01-01T12:00:00Z",
  "estimated_completion": "2024-01-01T12:15:00Z"
}
```

### Python SDK Usage
```python
from tidyllm.infrastructure import FlowIntegrationManager

# Initialize flow manager
flow_manager = FlowIntegrationManager()
await flow_manager.initialize()

# Execute bracket command
result = await flow_manager.execute_bracket_command(
    bracket_command="[Hybrid Analysis]",
    document_path="/path/to/document.pdf",
    user_context={"analysis_type": "comprehensive"}
)

print(f"Processing result: {result.success}")
print(f"Templates used: {result.mapped_templates}")
```

## Processing Strategies

### Single Template
- Uses one primary template for processing
- Fast and efficient for specific document types
- Examples: `[Quality Check]`, `[Data Extraction]`

### Multi-Perspective  
- Combines multiple templates for comprehensive analysis
- Cross-validates findings between different analytical approaches
- Examples: `[Contract Review]`, `[Compliance Check]`

### Hybrid Analysis
- Advanced multi-framework synthesis
- Resolves conflicts between different analytical perspectives
- Example: `[Hybrid Analysis]`

## Priority Levels

### Critical
- Immediate processing with highest resource allocation
- Expert validation required
- Examples: `[Peer Review]`

### High
- Fast processing with elevated priority
- Multi-template analysis preferred
- Examples: `[Process MVR]`, `[Contract Review]`

### Normal
- Standard processing with balanced resource usage
- Most common priority level
- Examples: `[Financial Analysis]`, `[Data Extraction]`

## Security and Validation

### Bracket Command Security
- Only approved bracket commands from registry are allowed
- Commands validated against security patterns
- All executions logged with full audit trail

### Template Validation
- All templates must be pre-approved and security-reviewed
- Templates stored in secured `prompts/templates/` directory
- No dynamic template creation or modification

### LLM Gateway Enforcement
- All AI operations go through CorporateLLMGateway
- Budget controls and cost tracking applied
- Content filtering and PII detection enabled

## Custom Bracket Commands

### Registration Process
To register a new bracket command:

1. **Create Template**: Develop and security-review processing template
2. **Define Mapping**: Create flow-to-template mapping configuration
3. **Security Review**: Validate command against security constraints
4. **Testing**: Comprehensive testing of new command
5. **Registration**: Add to approved bracket commands registry

### Example Registration
```python
# Register new bracket command
mapping = FlowToTemplateMapping(
    bracket_command="[Custom Analysis]",
    flow_encoding="@custom#analysis!process@domain_specific",
    template_names=["custom_template", "qa_control"],
    processing_strategy=ProcessingStrategy.SINGLE_TEMPLATE,
    priority_level="normal",
    validation_rules=["domain_validation"]
)

success = await flow_manager.register_flow_mapping("[Custom Analysis]", mapping)
```

## Troubleshooting

### Command Not Found
- Verify bracket command exists in registry
- Check spelling and exact bracket syntax
- Use `tidyllm flow list` to see available commands

### Processing Failures
- Check document format and accessibility
- Verify user permissions for document access
- Review processing logs for specific error details

### Template Issues
- Ensure required templates are available
- Check template validation and approval status
- Verify template compatibility with document type

## Monitoring and Analytics

### Usage Tracking
- All bracket command executions tracked
- Performance metrics collected per command
- Success rates and error patterns monitored

### Quality Metrics
- Template effectiveness measured
- User satisfaction feedback collected
- Processing time and resource usage tracked

## Complete API Implementation

### Existing API Endpoints (scripts/apis/)

The system includes multiple API interfaces for bracket command execution:

#### 1. Chain Operations API (`api_bracket_flows.py`)
Complete FastAPI implementation with the following endpoints:

```python
# Backend Operations (Complex Processing)
POST /chains/ingest         # Document ingestion with S3 processing
POST /chains/embed          # Generate embeddings using tidyllm-sentence  
POST /chains/index          # Create searchable indices using tlm
POST /chains/execute        # Execute chained operations

# Frontend Operations (Simple Interface)
POST /chains/query          # Natural language query
POST /chains/search         # Keyword search

# Status and Monitoring
GET /chains/status          # Overall system status
GET /chains/status/{id}     # Specific operation status
```

**Example Chain Execution:**
```http
POST /chains/execute
{
  "operations": ["ingest", "embed", "index"],
  "domain": "compliance_docs",
  "source": "/path/to/mvr_document.pdf",
  "execution_mode": "auto",
  "config": {
    "batch_size": 10,
    "embedding_model": "tfidf",
    "parallel_workers": 3
  }
}
```

#### 2. Universal Bracket Flow API (`universal_bracket_flow_examples.py`)
Demonstrates bracket command execution across multiple interfaces:

```python
# CLI Interface
await parser.cli_execute("[mvr_analysis]", user_id="cli_user")

# API Interface  
await parser.api_execute({
    "command": "[mvr_analysis]",
    "context": {"user_id": "api_user", "priority": "high"}
})

# UI Interface (with bracket detection)
await parser.ui_execute(user_input="Please run [Contract Review] on this document")

# S3 Interface (trigger files)
# Drop [mvr_analysis].trigger file in S3 bucket
```

#### 3. Multi-Interface Support
The system supports bracket commands through:

- **CLI Interface** (`cli_bracket_flows.py`): Command-line bracket execution
- **API Interface** (`api_bracket_flows.py`): REST API endpoints
- **UI Interface** (`ui_bracket_flows.py`): Web interface with bracket detection
- **S3 Interface**: Drop zone trigger files

### API Response Models

#### Chain Operation Response
```json
{
  "operation_id": "uuid-string",
  "status": "started|running|completed|failed",
  "operation_type": "ingest|embed|index|query|search",
  "domain": "knowledge_domain",
  "started_at": "2024-01-01T12:00:00Z",
  "completed_at": "2024-01-01T12:15:00Z",
  "results": {
    "documents_processed": 50,
    "embeddings_created": 50,
    "index_entries": 1250
  },
  "errors": [],
  "metadata": {
    "batch_size": 10,
    "embedding_model": "tfidf"
  }
}
```

#### Query Response
```json
{
  "query_id": "uuid-string",
  "question": "What are the compliance requirements?",
  "matches": [
    {
      "document": "mvr_compliance_guide.pdf",
      "relevance_score": 0.95,
      "excerpt": "Compliance requirements for MVR processing...",
      "metadata": {
        "section": "Section 5.4.3",
        "page": 15
      }
    }
  ],
  "processing_time_ms": 145.7,
  "domain": "compliance_docs"
}
```

#### System Status Response
```json
{
  "domain": "compliance_docs",
  "active_operations": 3,
  "total_documents": 1250,
  "total_queries": 847,
  "avg_response_time_ms": 234.5,
  "health_status": "healthy",
  "last_updated": "2024-01-01T12:30:00Z"
}
```

### API Authentication and Permissions

The existing API implementation includes:

```python
# Permission-based access control
permissions = {
    "read": ["query", "search", "status"],
    "write": ["ingest", "embed", "index"],
    "chain": ["execute"],
    "admin": ["all"]
}

# API key validation
async def validate_api_key(api_key: str):
    """Validate API key and return permissions."""
    # Integration with existing APIKeyManager
    return {"permissions": ["read", "write", "query"]}
```

### Integration with Flow Integration Manager

The new Flow Integration Manager integrates with existing APIs:

```python
# Add Flow Integration endpoints to existing API
from tidyllm.infrastructure.api.manager_endpoints import ManagerAPI
from tidyllm.infrastructure.workers.flow_integration_manager import FlowIntegrationManager

# Initialize components
flow_manager = FlowIntegrationManager()
manager_api = ManagerAPI()

# Add to existing FastAPI app
app.include_router(manager_api.router, prefix="/api/v1")

# Bracket command execution endpoint
@app.post("/api/v1/flow/execute")
async def execute_bracket_command(request: BracketCommandRequest):
    """Execute bracket command through Flow Integration Manager."""
    result = await flow_manager.execute_bracket_command(
        bracket_command=request.bracket_command,
        document_path=request.document_path,
        user_context=request.user_context
    )
    return result
```

### API Usage Examples

#### Execute MVR Analysis via API
```bash
curl -X POST "http://localhost:8000/api/v1/flow/execute" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "bracket_command": "[Process MVR]",
    "document_path": "/path/to/mvr_document.pdf",
    "business_priority": "high",
    "user_context": {
      "department": "compliance",
      "analyst": "jane.doe",
      "request_id": "MVR-2024-001"
    }
  }'
```

#### Monitor Processing Status
```bash
curl -X GET "http://localhost:8000/api/v1/process/{processing_id}" \
  -H "Authorization: Bearer your-api-key"
```

#### Execute Chain Operations
```bash
curl -X POST "http://localhost:8000/chains/execute" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "operations": ["ingest", "embed", "index"],
    "domain": "legal_contracts",
    "source": "/path/to/documents/",
    "execution_mode": "auto"
  }'
```

---

## Related Documentation
- [Template Documentation](../templates/README.md)
- [AI Dropzone Manager](../infrastructure/workers/README.md)
- [Security Architecture](../../ARCHITECTURE.md)
- [API Reference](../api/README.md)