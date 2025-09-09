# TidyLLM Architecture Documentation

## System Overview

TidyLLM is an enterprise-grade document processing ecosystem that combines intelligent AI orchestration with secure, auditable workflows. The system processes documents through specialized gateways and workers while maintaining strict security controls and comprehensive audit trails.

## Core Architecture Principles

### 1. Security-First Design
- **Zero Trust**: All components validate inputs and enforce access controls
- **Gateway Pattern**: All external access routed through controlled gateways
- **Audit Trail**: Comprehensive logging of all operations and decisions
- **Least Privilege**: Components have minimal required permissions

### 2. Orchestration vs Execution Separation
- **AI Managers**: Intelligent orchestration and routing decisions
- **Workers**: Execute specific functions with their own security contexts
- **Gateways**: Control access and enforce enterprise policies

### 3. Template-Based Processing
- **Pre-validated Templates**: All processing templates security-reviewed
- **Structured Workflows**: Consistent, auditable processing patterns
- **Quality Assurance**: Built-in validation and peer review capabilities

## System Components

### Infrastructure Layer (`tidyllm/infrastructure/`)

#### Core Services
- **ConfigManager**: Centralized configuration management
- **UnifiedSessionManager**: AWS S3 and PostgreSQL session management
- **Logging**: Structured logging with audit capabilities
- **Adapters**: External system integration adapters

#### Worker System (`infrastructure/workers/`)
```
BaseWorker (Abstract)
├── AIDropzoneManager (Orchestration)
├── FlowIntegrationManager (FLOW Bracket Command Bridge)
├── PromptWorker (Template Processing)
├── FlowRecoveryWorker (Error Recovery)
└── CoordinatorWorker (Result Synthesis)
```

### Gateway Layer (`tidyllm/gateways/`)

#### Enterprise Gateways (Core Workflow)
```
1. CorporateLLMGateway → 2. AIProcessingGateway → 3. DatabaseGateway → 4. KnowledgeResourceGateway
                                                      ↓
5. FileStorageGateway ← 6. WorkflowOptimizerGateway (Optimization Layer)
```

#### Utility Services
```
MVRGateway (Specialized Processing)
```

#### Gateway Hierarchy
- **Foundation Layer**: CorporateLLMGateway (Enterprise LLM access control)
- **Processing Layer**: AIProcessingGateway (Document analysis orchestration)
- **Data Layer**: DatabaseGateway + FileStorageGateway (Persistence)
- **Knowledge Layer**: KnowledgeResourceGateway (Context and resources)
- **Optimization Layer**: WorkflowOptimizerGateway (Process improvement)

### Processing Templates (`prompts/templates/`)

#### Enterprise Workflow Templates
- **financial_analysis.md**: Financial document analysis and risk assessment
- **contract_analysis.md**: Legal contract review and compliance
- **compliance_review.md**: Regulatory compliance analysis
- **qa_control.md**: Quality assurance workflows
- **data_extraction.md**: Structured data extraction
- **peer_review.md**: Expert peer review processes
- **hybrid_analysis.md**: Multi-framework synthesis

## Flow Integration Architecture

### FLOW System Integration

The Flow Integration Manager bridges the existing FLOW (Flexible Logic Operations Workflows) bracket command system with the AI Dropzone Manager architecture.

#### FLOW Bracket Commands
```bash
# QA and Compliance Operations
[Process MVR]           → mvr_analysis + qa_control templates
[Check MVS Compliance]  → compliance_review + qa_control templates
[Quality Check]         → qa_control template

# Document Analysis Operations  
[Financial Analysis]    → financial_analysis + qa_control templates
[Contract Review]       → contract_analysis + compliance_review templates
[Data Extraction]       → data_extraction template

# Advanced Analysis Operations
[Peer Review]          → peer_review + qa_control templates
[Hybrid Analysis]      → hybrid_analysis + qa_control templates
```

#### Flow-to-Template Mapping System
```json
{
  "[Process MVR]": {
    "flow_encoding": "@mvr#process!extract@compliance_data",
    "template_names": ["mvr_analysis", "qa_control"],
    "processing_strategy": "multi_perspective",
    "priority_level": "high",
    "validation_rules": ["mvr_document_type", "compliance_standards"]
  }
}
```

#### Drop Zone Staged Processing
```
Input Zone → Processing Zone → AI Analysis → Completed/Failed Zone
     ↓              ↓              ↓              ↓
  Documents    In Progress     Results       Archive/Review
     
Purgatory Zone: FlowRecoveryWorker monitors and recovers stuck documents
```

### Integration Flow
```mermaid
graph TD
    A[Bracket Command: [Process MVR]] --> B[Flow Integration Manager]
    B --> C[Validate Command Security]
    C --> D[Lookup Flow Mapping]
    D --> E[Create AI Manager Task]
    E --> F[AI Dropzone Manager]
    F --> G[Template Processing]
    G --> H[Worker Coordination] 
    H --> I[Result Synthesis]
    I --> J[Drop Zone Management]
```

## AI Dropzone Manager Architecture

### Security Constraints

#### Worker Registry Only
```python
# CRITICAL: Only pre-approved worker types allowed
approved_worker_types = {
    "PromptWorker": PromptWorker,
    "FlowRecoveryWorker": FlowRecoveryWorker,
    "CoordinatorWorker": CoordinatorWorker
}
```

#### LLM Gateway Enforcement
```python
# ALL AI calls must go through CorporateLLMGateway
request = LLMRequest(
    prompt=analysis_prompt,
    audit_reason="Document intelligence analysis", 
    user_id="ai_manager_system"
)
response = self.llm_gateway.process_llm_request(request)
```

#### Template Validation
```python
def _validate_template_path(self, template_path: str) -> bool:
    """Ensure templates only loaded from approved directory."""
    if not template_file.resolve().is_relative_to(approved_templates_dir):
        logger.warning("[SECURITY] Unauthorized template rejected")
        return False
```

### Processing Flow

#### 1. Document Intelligence
```
Document Drop → Preview Analysis (8KB limit) → LLM Classification → Complexity Assessment → Template Recommendation
```

#### 2. Worker Orchestration  
```
Strategy Selection → Registry Validation → Worker Allocation → Execution Coordination → Quality Monitoring
```

#### 3. Result Synthesis
```
Worker Results → Conflict Detection → Consensus Building → Report Synthesis → Quality Assessment
```

## Gateway Integration Patterns

### CorporateLLMGateway (Foundation)
**Purpose**: Enterprise-controlled access to LLM services

**Key Features**:
- Budget controls and cost tracking
- Audit logging for compliance
- Content filtering and PII detection
- Model access controls
- MLFlow Gateway integration

**Dependencies**: Independent foundation layer

### AIProcessingGateway (Processing)
**Purpose**: Document processing workflow orchestration

**Key Features**:
- Document analysis coordination
- Multi-step workflow management
- Quality assurance integration
- Result aggregation and validation

**Dependencies**: Requires CorporateLLMGateway for LLM access

### DatabaseGateway (Data)
**Purpose**: Structured data persistence and retrieval

**Key Features**:
- PostgreSQL integration
- Query optimization
- Transaction management
- Data validation

**Dependencies**: Independent data layer

### FileStorageGateway (Storage)
**Purpose**: Document and file management

**Key Features**:
- S3 integration
- File lifecycle management
- Access controls
- Versioning support

**Dependencies**: Independent storage layer

### KnowledgeResourceGateway (Context)
**Purpose**: MCP server and knowledge resource provision

**Key Features**:
- Knowledge graph access
- Context augmentation
- Resource discovery
- Search capabilities

**Dependencies**: Requires DatabaseGateway and FileStorageGateway

### WorkflowOptimizerGateway (Optimization)
**Purpose**: Process improvement and optimization

**Key Features**:
- Workflow analysis
- Performance optimization
- Process recommendations
- Efficiency metrics

**Dependencies**: Requires all other gateways for comprehensive analysis

## Security Architecture

### Authentication and Authorization
- **User Context**: All requests tracked with user identity
- **Role-Based Access**: Permissions based on user roles
- **Audit Reasons**: Required justification for all LLM requests
- **Session Management**: Secure session handling via UnifiedSessionManager

### Data Protection
- **PII Detection**: Automatic detection and masking of sensitive data
- **Content Filtering**: Inappropriate content blocking
- **Data Residency**: Compliance with data location requirements
- **Encryption**: Data encrypted in transit and at rest

### Audit and Compliance
- **Comprehensive Logging**: All operations logged with context
- **Cost Tracking**: Budget controls and spending monitoring
- **Compliance Validation**: Regulatory requirement checking
- **Quality Metrics**: Performance and accuracy monitoring

## API Architecture

### REST Endpoints (`infrastructure/api/`)
```http
# Document Processing
POST /api/v1/process
GET /api/v1/process/{id}
GET /api/v1/process/{id}/result

# System Management
GET /api/v1/manager/status
GET /api/v1/queue
GET /api/v1/workers
GET /api/v1/templates
```

### MCP Integration
```json
{
  "method": "tools/call",
  "params": {
    "name": "ai_manager_process",
    "arguments": {
      "document_path": "/path/to/document.pdf",
      "business_priority": "high"
    }
  }
}
```

### CLI Interface (`infrastructure/cli/`)
```bash
# Document processing
tidyllm ai-manager process document.pdf --priority critical --watch

# System monitoring  
tidyllm ai-manager status
tidyllm ai-manager monitor --watch

# Resource management
tidyllm ai-manager workers
tidyllm ai-manager templates --detailed
```

## Deployment Architecture

### Component Dependencies
```
AI Dropzone Manager
├── CorporateLLMGateway (REQUIRED)
│   ├── MLFlow Gateway Client
│   ├── Cost Tracking
│   └── Audit Logging
├── UnifiedSessionManager
│   ├── AWS S3 Client
│   └── PostgreSQL Connection
├── Worker Registry
│   ├── PromptWorker
│   ├── FlowRecoveryWorker
│   └── CoordinatorWorker
└── Template Library (Pre-validated)
```

### Initialization Sequence
1. **Security Validation**: Validate worker registry and security constraints
2. **Gateway Initialization**: Initialize CorporateLLMGateway connection
3. **Session Management**: Set up UnifiedSessionManager with AWS/DB credentials
4. **Template Loading**: Load and validate approved processing templates
5. **Worker Registration**: Register approved worker instances
6. **Performance Data**: Load historical performance metrics
7. **Health Checks**: Validate all component health
8. **Service Startup**: Start API endpoints and CLI interfaces

## Monitoring and Observability

### Health Monitoring
- **Component Health**: Real-time status of all system components
- **Gateway Connectivity**: Monitor gateway connections and performance
- **Worker Status**: Track worker pool utilization and performance
- **Queue Monitoring**: Processing queue depth and throughput

### Performance Metrics
- **Processing Time**: Document processing duration by type and complexity
- **Quality Scores**: Accuracy and completeness metrics
- **Cost Tracking**: LLM usage costs and budget utilization
- **Success Rates**: Processing success/failure rates by template

### Alerting
- **Queue Backup**: Alert when processing queue exceeds capacity
- **Quality Degradation**: Alert when quality scores drop below thresholds
- **Budget Limits**: Alert when approaching spending limits
- **Component Failures**: Alert on gateway or worker failures

## Extension Points

### Future Manager Types
- **AI Batch Manager**: Large-scale batch processing orchestration
- **AI API Manager**: Real-time API document processing
- **AI Workflow Manager**: Complex multi-stage workflow orchestration
- **AI Compliance Manager**: Specialized regulatory compliance processing

### Additional Workers
- **ValidationWorker**: Document validation and verification
- **TransformationWorker**: Format conversion and transformation
- **IndexingWorker**: Search indexing and preparation
- **NotificationWorker**: Result delivery and communication

### Gateway Extensions
- **ExternalAPIGateway**: Third-party service integration
- **ReportingGateway**: Business intelligence and reporting
- **IntegrationGateway**: Enterprise system integration
- **AnalyticsGateway**: Advanced analytics and insights

## Best Practices

### Security Best Practices
- Regular security audits of worker registry
- Monitor template library for unauthorized changes
- Review audit logs for suspicious patterns
- Validate all external integrations
- Test disaster recovery procedures

### Performance Optimization
- Cache document intelligence for similar documents
- Use appropriate LLM models for different tasks
- Monitor and tune quality thresholds
- Implement load balancing across workers
- Optimize database queries and indexing

### Operational Excellence
- Maintain comprehensive system documentation
- Implement automated testing and validation
- Establish clear escalation procedures
- Regular performance reviews and optimization
- Continuous monitoring and improvement

## Migration and Upgrade Strategies

### Version Compatibility
- Backward compatibility for API endpoints
- Template version migration procedures
- Worker interface evolution strategy
- Database schema migration processes

### Rollback Procedures
- Component-level rollback capabilities
- Data consistency maintenance during rollbacks
- Service continuity during upgrades
- Emergency rollback procedures

This architecture provides a secure, scalable, and maintainable foundation for enterprise document processing with intelligent AI orchestration while maintaining strict security controls and comprehensive audit capabilities.