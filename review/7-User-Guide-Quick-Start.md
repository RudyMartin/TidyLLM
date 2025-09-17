# TidyLLM RAG Ecosystem - Quick Start Guide
**Time to Complete:** 15 minutes
**Difficulty:** Beginner
**Prerequisites:** Python 3.9+, AWS access configured

## 🚀 Get Running in 15 Minutes

This guide will get you from zero to running RAG systems in 15 minutes.

## 📋 Step 1: Environment Setup (5 minutes)

### Check Prerequisites
```bash
# Verify Python version
python --version  # Should be 3.9+

# Verify you're in the TidyLLM directory
ls tidyllm/  # Should see services, portals, etc.
```

### Initialize USM (AWS Credentials)
```bash
# Set up AWS environment
python -c "from tidyllm.admin.credential_loader import set_aws_environment; set_aws_environment()"
```

**Expected Output:**
```
✓ AWS credentials configured
✓ USM session manager ready
```

## 🎯 Step 2: Launch RAG Creator V3 (3 minutes)

### Start the Management Portal
```bash
# Launch the main RAG management interface
streamlit run tidyllm/knowledge_systems/migrated/portal_rag/rag_creator_v3.py
```

**Expected Result:**
- Browser opens to `http://localhost:8501`
- RAG Creator V3 interface loads
- Sidebar shows "UnifiedRAGManager Connected"

### Quick Portal Tour
1. **Browse RAG Systems Tab** - See all 6 orchestrators
2. **Health Dashboard** - Check system status
3. **Sidebar Status** - Verify "X/6 systems available"

## 📊 Step 3: Explore Available Systems (2 minutes)

### Check System Availability
In the **Browse RAG Systems** tab, you should see:

- **🤖 AI-Powered RAG** - Corporate AI analysis
- **🗃️ PostgreSQL RAG** - Authority-based compliance
- **⚖️ Judge RAG** - External system integration
- **🧠 Intelligent RAG** - Document processing
- **📚 SME RAG System** - Enterprise document lifecycle
- **✨ DSPy RAG** - Prompt optimization

**Status Indicators:**
- ✅ **Available** - System ready for use
- ❌ **Unavailable** - System needs configuration

## ➕ Step 4: Create Your First RAG System (3 minutes)

### Navigate to Create Tab
1. Click **"Create RAG System"** tab
2. Choose a system type (start with **AI-Powered RAG**)

### Configure Basic Settings
```
System ID: my_first_rag
System Name: My First RAG System
Description: Testing the RAG ecosystem
Domain: general
Priority: medium
```

### System-Specific Settings
For **AI-Powered RAG**:
- ✅ Use Corporate Gateway
- Model: claude-3-sonnet

### Create the System
1. Click **"Create RAG System"**
2. Wait for success message
3. Note the System ID for later use

**Expected Result:**
```
SUCCESS! My First RAG System (ai_powered) created successfully
System ID: my_first_rag
Status: Ready for optimization and training
```

## 💓 Step 5: Health Check (2 minutes)

### Monitor System Health
1. Go to **"Health Dashboard"** tab
2. Check overall metrics:
   - **Healthy Systems**: Should show 1 or more
   - **Avg Response Time**: Baseline measurement
   - **Success Rate**: Should be >90%

### Individual System Health
1. Expand your created system
2. Check status indicators:
   - **Status**: Should be "HEALTHY"
   - **Response Time**: Baseline measurement
   - **Error Rate**: Should be <5%

### Run Health Check
1. Click **"Refresh [system_name]"**
2. Verify status updates
3. Note any warnings or errors

## 🎯 Bonus: Try DSPy Design Assistant (Optional)

### Launch Specialized Designer
```bash
# In a new terminal/command prompt
streamlit run tidyllm/portals/rag/dspy_design_assistant_portal.py
```

### Quick DSPy Tour
1. **DSPy RAG Templates** - Pre-built optimization templates
2. **Prompt Studio** - Try optimizing a sample prompt
3. **Performance Monitor** - DSPy-specific metrics

## ✅ Verification Checklist

After completing this guide, you should have:

- [ ] **Portal Running** - RAG Creator V3 accessible at localhost:8501
- [ ] **Systems Visible** - Can see all 6 RAG orchestrator types
- [ ] **System Created** - Successfully created first RAG system
- [ ] **Health Monitoring** - Can view system health status
- [ ] **Navigation** - Comfortable with portal tabs and features

## 🚀 What's Next?

### Immediate Next Steps
1. **Explore More Systems** - Create different RAG types
2. **Test Queries** - Use the Browse tab to test functionality
3. **Performance Tuning** - Try the Tailor & Optimize features

### Advanced Features
1. **A/B Testing** - Compare different configurations
2. **DSPy Optimization** - Advanced prompt engineering
3. **Multi-System Workflows** - Orchestrate multiple RAG types

### Production Deployment
1. **Scaling** - Configure for multiple users
2. **Monitoring** - Set up alerts and dashboards
3. **Backup** - Implement data protection

## 🆘 Troubleshooting

### Common Issues

#### Portal Won't Start
```bash
# Check if port is available
netstat -an | findstr :8501

# Kill existing processes if needed
taskkill /f /im streamlit.exe

# Restart portal
streamlit run tidyllm/knowledge_systems/migrated/portal_rag/rag_creator_v3.py
```

#### "URM Not Available" Error
```bash
# Verify USM setup
python -c "from tidyllm.infrastructure.session.unified import UnifiedSessionManager; usm = UnifiedSessionManager(); print('USM OK')"

# Re-initialize credentials
python -c "from tidyllm.admin.credential_loader import set_aws_environment; set_aws_environment()"
```

#### No Systems Available
```bash
# Check database connection
python -c "from tidyllm.infrastructure.connection_pool import get_global_pool; pool = get_global_pool(); print('Database OK')"

# Verify AWS credentials
aws sts get-caller-identity
```

#### Unicode Errors (Windows)
```bash
# Set environment variable
set PYTHONIOENCODING=utf-8

# Or use Windows Terminal instead of Command Prompt
```

### Getting Help

#### Documentation
- **Complete Documentation**: `tidyllm/review/6-Complete-RAG-Ecosystem-Documentation.md`
- **Testing Results**: `tidyllm/review/5-RAG-Ecosystem-Test-Results.md`
- **Architecture Analysis**: `tidyllm/review/3-Adapter-Orchestration-Analysis.md`

#### Support Channels
- **GitHub Issues**: Report bugs and feature requests
- **Internal Documentation**: Check TidyLLM wiki
- **Team Chat**: Real-time support and questions

## 🎉 Success!

You now have a fully operational RAG ecosystem with:
- **6 RAG Orchestrators** at your disposal
- **Management Portal** for CRUD operations
- **Health Monitoring** for system oversight
- **Specialized Design Tools** for advanced optimization

**Time to Completion:** ~15 minutes
**Systems Available:** Up to 6 RAG orchestrators
**Next Goal:** Create and optimize multiple RAG systems for your specific use cases

---
**Happy RAG Building!** 🚀