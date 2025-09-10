# TidyLLM Corporate Onboarding System

## 🏢 **Clean, Normalized Corporate Onboarding Solution**

A comprehensive, enterprise-ready onboarding system for deploying TidyLLM in corporate environments with full security compliance and UnifiedSessionManager integration.

## 🚀 **Quick Start**

### **Single Entry Point**
```bash
# Launch the complete onboarding system
python launcher.py
```

The system will automatically:
- Configure AWS environment
- Launch Streamlit application
- Enable auto-reload for development
- Open browser to http://localhost:8501

## 📋 **System Architecture**

### **Clean Structure**
```
onboarding/
├── launcher.py              # Single entry point
├── app.py                   # Main Streamlit application
├── core/                    # Core functionality
│   ├── session_manager.py   # Unified session management
│   ├── validator.py         # Connection validation
│   └── preflight.py         # Pre-flight tests
├── config/                  # Configuration management
│   ├── manager.py           # Config management
│   └── templates.py         # Configuration templates
├── ui/                      # User interface
│   ├── components/          # Reusable UI components
│   └── pages/               # Page components
├── requirements.txt         # Dependencies
└── README.md               # Documentation
```

## 🎯 **6-Section Interface**

### **1. Connection Config** 🔗
- **AWS Services**: S3, Bedrock, STS connectivity testing
- **Database**: PostgreSQL connection validation
- **Gateways**: All 4 TidyLLM gateways testing
- **Real-time Validation**: Live connection status monitoring

### **2. Chat Test** 💬
- **AI Model Testing**: Live chat with multiple providers
- **Model Selection**: Bedrock, OpenAI, local models
- **File Upload**: Document analysis testing
- **Performance Metrics**: Response times and cost tracking

### **3. Knowledge Management** 🧠
- **DomainRAG CRUD**: Complete knowledge base management
- **Document Upload**: Multi-file document processing
- **Semantic Search**: Vector-based search capabilities
- **Metadata Management**: Comprehensive document tracking

### **4. Workflows** ⚙️
- **YAML Registry**: Convert Python workflows to editable YAML
- **AI Managers**: Create custom AI managers with dual RAG
- **Workflow Editor**: Live YAML editing with validation
- **Dual RAG System**: Domain knowledge + work history

### **5. Test Workflow** 🧪
- **End-to-End Testing**: Complete workflow execution
- **Document Processing**: Real document pipeline testing
- **All 4 Gateways**: Test complete TidyLLM ecosystem
- **Performance Monitoring**: Real-time metrics and logging

### **6. Dashboard** 📊
- **System Health**: Connection status and performance
- **Usage Analytics**: AI model usage and cost tracking
- **Performance Metrics**: Response times and throughput
- **Resource Monitoring**: CPU, memory, storage usage

## 🔒 **Security & Compliance**

### **UnifiedSessionManager Integration**
- **100% Compliance**: All AWS access through UnifiedSessionManager
- **No boto3 Fallbacks**: Complete eradication of direct boto3 calls
- **Credential Security**: Centralized credential management
- **Audit Trail**: Complete request/response logging

### **Enterprise Features**
- **SSO Integration**: Okta, Azure AD, SAML support
- **Proxy Support**: Corporate network compatibility
- **Data Encryption**: S3 server-side encryption
- **Access Controls**: Role-based permissions
- **Compliance**: GDPR, SOX, HIPAA considerations

## 🛠️ **Installation & Setup**

### **Prerequisites**
```bash
# Python 3.8+ required
python --version

# Install dependencies
pip install -r requirements.txt
```

### **Environment Setup**
```bash
# AWS credentials (automatically configured by launcher)
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-1"
```

### **Launch System**
```bash
# Single command launch
python launcher.py
```

## 📊 **Features**

### **Core Capabilities**
- **Real-time Testing**: Live connection validation
- **Document Processing**: Multi-format document support
- **AI Integration**: Multiple AI provider support
- **Knowledge Management**: Complete CRUD operations
- **Workflow Orchestration**: YAML-based workflow management
- **Performance Monitoring**: Real-time metrics and analytics

### **Enterprise Integration**
- **Corporate Networks**: Proxy and firewall support
- **Authentication**: SSO and corporate identity providers
- **Compliance**: Audit logging and data protection
- **Scalability**: Horizontal scaling support
- **Monitoring**: Comprehensive system health monitoring

## 🎯 **Usage Patterns**

### **Corporate IT Administrator**
1. **Connection Config** → Test corporate environment
2. **Chat Test** → Validate AI model access
3. **Knowledge Management** → Set up knowledge domains
4. **Workflows** → Configure business processes
5. **Test Workflow** → Validate end-to-end processing
6. **Dashboard** → Monitor system health

### **Business User**
1. **Chat Test** → Quick AI interaction
2. **Knowledge Management** → Search organizational knowledge
3. **Test Workflow** → Process documents
4. **Dashboard** → View results and status

### **Developer**
1. **Connection Config** → Validate development setup
2. **Workflows** → Create custom AI managers
3. **Knowledge Management** → Build knowledge domains
4. **Dashboard** → Monitor performance and debug

## 🆘 **Troubleshooting**

### **Common Issues**

**1. Import Errors**
```
Error: TidyLLM imports not available
```
**Solution**: Ensure you're running from the correct directory and TidyLLM is properly installed.

**2. AWS Connection Issues**
```
Error: AWS credentials not found
```
**Solution**: The launcher automatically configures AWS credentials. Check environment variables.

**3. Database Connection Issues**
```
Error: PostgreSQL connection failed
```
**Solution**: Verify database configuration and network connectivity.

### **Getting Help**
- Check system status in Dashboard
- Review connection test results in Connection Config
- Validate workflow configurations in Workflows
- Monitor performance metrics in Dashboard

## 📈 **Performance**

### **Optimization Features**
- **Auto-reload**: Development-friendly auto-refresh
- **Session Management**: Efficient resource utilization
- **Caching**: Intelligent caching for better performance
- **Parallel Processing**: Multi-threaded operations
- **Resource Monitoring**: Real-time performance tracking

## 🔄 **Development**

### **Auto-reload Development**
- **File Watching**: Automatic reload on file changes
- **Session Persistence**: Maintains state across reloads
- **Hot Reloading**: Instant updates without restart
- **Development Mode**: Optimized for development workflow

### **Code Structure**
- **Modular Design**: Clean separation of concerns
- **Single Responsibility**: Each component has one purpose
- **Reusable Components**: Shared UI components
- **Clean Architecture**: Easy to maintain and extend

---

**TidyLLM Corporate Onboarding System** - Clean, normalized, enterprise-ready solution for corporate TidyLLM deployment.