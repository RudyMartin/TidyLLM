# TidyLLM Corporate Onboarding Kit 🏢

## Complete 6-Section Corporate Setup Interface

This comprehensive Streamlit application provides everything needed for corporate TidyLLM deployment:

### 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements_onboarding_kit.txt

# Launch the onboarding kit
streamlit run enhanced_streamlit_onboarding_kit.py
```

### 📋 Six-Section Interface

#### 1. **Connection Config** - CorporateLLMGateway & DatabaseGateway
- **CorporateLLMGateway**: SSO/SAML authentication, corporate proxy settings
- **DatabaseGateway**: PostgreSQL database validation, S3 bucket access
- **UnifiedSessionManager**: Enhanced v1.0.4 corporate session management
- Real-time connection testing and validation

#### 2. **Chat Test** - AIProcessingGateway Live Testing
- **AIProcessingGateway**: Live AI model testing with multiple providers
- Model selection (Bedrock, OpenAI, etc.) with streaming responses
- File upload capability for document analysis
- Performance metrics and cost estimation
- Chat history with model comparison

#### 3. **DomainRAG CRUD** - Knowledge Management
- **CREATE**: New knowledge domains with document upload
- **READ**: Browse domains, semantic search, metadata viewing
- **UPDATE**: Add documents, retrain vectors, optimize storage
- **DELETE**: Archive domains, permanent deletion with safety checks
- Full S3 integration with enterprise security

#### 4. **Workflows (YAML Registry)** - WorkflowOptimizerGateway
- **Registry Conversion**: Convert `bracket_registry.py` to editable YAML
- **Ad-hoc AI Managers**: Create custom AI managers with dual RAG support
- **YAML Editor**: Live editing with validation and save functionality
- **Dual RAG System**: Domain knowledge + work history RAGs per manager

#### 5. **Test Workflow** - All 4 Gateways Live Execution
- **Workflow Testing**: Execute QA, MVR, and custom workflows with real data
- **Document Processing**: Upload and process documents through complete pipeline
- **Manager Integration**: AI managers with domain knowledge and work history
- **Results Validation**: Download and validate processing results

#### 6. **Dashboard** - UnifiedSessionManager Monitoring
- **System Health**: Connection status, resource utilization
- **Usage Analytics**: AI model usage patterns, cost tracking
- **Performance Metrics**: Processing times, confidence scores
- **Real-time Updates**: Auto-refresh capabilities with live monitoring

## 🏗️ Architecture Integration

### **4-Gateway 2-Service Architecture**
- **CorporateLLMGateway**: Corporate AI processing with compliance
- **AIProcessingGateway**: Core AI operations and model management
- **DatabaseGateway**: PostgreSQL, S3, and data operations  
- **WorkflowOptimizerGateway**: Business process optimization
- **UnifiedSessionManager**: Universal session/credential management
- **DomainRAG**: Knowledge management and AI manager support

### **Enterprise Features**
- **SSO Integration**: SAML authentication with temporary credentials
- **Corporate Proxy**: Firewall and proxy configuration support
- **Dual RAG System**: Domain knowledge + work history per AI manager
- **Work History Tracking**: Institutional memory and artifact storage
- **Ad-hoc AI Managers**: Dynamic creation for custom workflows
- **Compliance Ready**: Audit logging, security standards, data governance

## 🔧 Configuration

### **Environment Variables**
```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1

# Database Configuration  
POSTGRES_HOST=vectorqa-cluster.cluster-cu562e4m02nq.us-east-1.rds.amazonaws.com
POSTGRES_DB=vectorqa
POSTGRES_USER=vectorqa_user
POSTGRES_SSL_MODE=require

# S3 Configuration
S3_BUCKET=nsc-mvp1
S3_REGION=us-east-1

# Corporate Settings (optional)
CORPORATE_SSO_ENDPOINT=https://sso.company.com
CORPORATE_PROXY_HOST=proxy.company.com
CORPORATE_PROXY_PORT=8080
```

### **Database Setup**
The onboarding kit works with the existing TidyLLM database:
- **PostgreSQL**: Real RDS cluster for production data
- **S3**: Production bucket for document and artifact storage
- **MLflow**: PostgreSQL backend with S3 artifact storage

## 🎯 Usage Patterns

### **Corporate IT Administrator Flow**
1. **Connection Config** → Test corporate environment connectivity
2. **Chat Test** → Validate AI model access and performance  
3. **DomainRAG CRUD** → Set up knowledge domains for different departments
4. **Workflows** → Configure business-specific workflows with AI managers
5. **Test Workflow** → Validate end-to-end processing with real documents
6. **Dashboard** → Monitor system health and usage patterns

### **Business User Flow**
1. **Chat Test** → Quick AI interaction testing
2. **DomainRAG** → Search and browse organizational knowledge
3. **Test Workflow** → Process documents through established workflows  
4. **Dashboard** → View processing results and system status

### **Developer Flow**
1. **Connection Config** → Validate development environment setup
2. **Workflows** → Create and test custom AI managers
3. **DomainRAG** → Build and optimize knowledge domains
4. **Dashboard** → Monitor performance and debug issues

## 📊 Key Capabilities

### **Enterprise Integration**
- Works with existing corporate authentication systems
- Handles proxy servers and firewall configurations  
- Supports temporary/rotating credentials (AWS SSO)
- Integrates with corporate compliance requirements

### **Knowledge Management**
- Domain-specific RAG systems with S3 backend
- Semantic search across organizational knowledge
- Work history tracking for institutional memory
- Document processing with enterprise security

### **Workflow Orchestration**  
- Convert Python workflow definitions to editable YAML
- Create ad-hoc AI managers for new business processes
- Dual RAG support (domain knowledge + work history)
- End-to-end testing with real document processing

### **Monitoring & Analytics**
- Real-time system health and performance metrics
- Cost tracking and usage analytics across AI models
- Connection status monitoring for all enterprise systems
- Performance optimization recommendations

## 🔒 Security Features

- **Credential Management**: Secure handling of corporate credentials
- **Data Encryption**: S3 server-side encryption enabled by default
- **Audit Logging**: Complete audit trails for compliance requirements
- **Access Controls**: Role-based access through corporate authentication
- **Data Isolation**: Tenant-specific S3 prefixes and database schemas

## 📈 Scalability

- **Horizontal Scaling**: Worker architecture supports multiple concurrent users
- **Storage Scaling**: S3-based storage scales automatically with usage
- **Processing Scaling**: AI model access through managed services (Bedrock)
- **Database Scaling**: RDS cluster supports read replicas for high availability

## 🛠️ Troubleshooting

### **Connection Issues**
- Check corporate proxy settings in Section 1
- Validate AWS credentials and region configuration
- Test database connectivity with provided connection strings

### **Performance Issues** 
- Monitor resource utilization in Dashboard (Section 6)
- Check AI model response times in Chat Test (Section 2)
- Optimize DomainRAG vector indices in Section 3

### **Workflow Issues**
- Validate YAML syntax in Workflows section (Section 4)
- Test individual commands before full workflow execution
- Check AI manager configuration and RAG availability

## 📞 Support

For technical support and configuration assistance:
1. Check the real-time dashboard for system status
2. Review connection test results in Section 1
3. Validate workflow configurations in Section 4
4. Monitor performance metrics in Section 6

---

**TidyLLM Corporate Onboarding Kit v1.0.4** - Complete enterprise AI infrastructure setup and management.