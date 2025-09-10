# 🚨 CRITICAL CONSTRAINTS - TidyLLM Onboarding System

## ⚠️ **AWS CONNECTIVITY IS MANDATORY**

### **Why Config/Connection is the First Screen**

**NOTHING WORKS WITHOUT AWS CONNECTION**

The TidyLLM system is built on AWS infrastructure. All core functionality depends on:

1. **S3** - Document storage and retrieval
2. **Bedrock** - AI model access and processing  
3. **STS** - Security token service for authentication
4. **PostgreSQL** - Vector database for knowledge management

### **Gateway Dependencies**

All TidyLLM gateways **REQUIRE** AWS connection:

- **CorporateLLMGateway** - Needs Bedrock for AI processing
- **AIProcessingGateway** - Needs Bedrock for model access
- **DatabaseGateway** - Needs S3 for document storage
- **WorkflowOptimizerGateway** - Needs S3 for workflow data

### **Onboarding Flow Requirements**

1. **FIRST**: Configure AWS connections (Connection Config page)
2. **THEN**: Initialize gateways (automatic after AWS connection)
3. **FINALLY**: Access other features (Chat, Knowledge, Workflows, etc.)

### **Error States**

If AWS connection fails:
- ❌ All gateways will fail to initialize
- ❌ Chat functionality will not work
- ❌ Knowledge management will not work
- ❌ Workflow processing will not work
- ❌ System is essentially non-functional

### **Success States**

When AWS connection succeeds:
- ✅ UnifiedSessionManager initializes
- ✅ All gateways can initialize
- ✅ Full system functionality available
- ✅ Ready for corporate demo

## 🔧 **Troubleshooting**

### **Common Issues:**
1. **Missing AWS credentials** - Check environment variables or settings file
2. **Network connectivity** - Verify internet connection to AWS
3. **Permissions** - Ensure AWS credentials have required permissions
4. **Region configuration** - Verify correct AWS region settings

### **Verification Steps:**
1. Check sidebar for "Session Manager Active" ✅
2. Check sidebar for "All Gateways Ready" ✅
3. Test connection on Connection Config page
4. Verify all services show green status

## 📋 **Implementation Notes**

- Connection Config page is intentionally first in navigation
- Sidebar shows critical status indicators
- Gateway initialization is blocked without AWS connection
- Clear error messages guide users to fix connection issues
- System gracefully handles partial failures

---

**Remember: AWS connectivity is not optional - it's the foundation of the entire system.**
