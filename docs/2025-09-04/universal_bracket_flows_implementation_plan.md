# Universal Bracket Flow System Implementation Plan

## 🎯 **Executive Summary**

Implementation plan for TidyLLM's universal bracket-based flow contract system that works across CLI, API, UI, Chat, AND S3 interfaces using a single YAML workflow definition.

## 🏗️ **System Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACES                          │
├─────────────┬─────────────┬─────────────┬─────────────┬─────┤
│     CLI     │     API     │     UI      │    Chat     │ S3  │
│             │             │             │             │Evts │
├─────────────┴─────────────┴─────────────┴─────────────┴─────┤
│              Universal Bracket Parser                       │
│         [workflow_name action parameters]                   │  
├─────────────────────────────────────────────────────────────┤
│                  YAML Workflow Loader                       │
│         workflows/mvr_analysis_flow.yaml                    │
│         workflows/domainrag_robots3.yaml                    │
├─────────────────────────────────────────────────────────────┤
│               Gateway Execution Engine                      │
│         (dspy, llm, heiros gateways)                       │
├─────────────────────────────────────────────────────────────┤
│              S3-First Processing Backend                    │
│    (streaming, cloud-native, stateless)                    │
└─────────────────────────────────────────────────────────────┘
```

## 📋 **Implementation Components**

### **1. Core Components (✅ Created)**

#### **universal_flow_parser.py**
- `UniversalFlowParser`: Core bracket parsing and workflow execution
- `BracketCommand`: Parsed bracket structure
- `FlowExecution`: Execution tracking
- YAML workflow loading from `/workflows/` folder

#### **s3_flow_parser.py** 
- `S3FlowParser`: S3-integrated extension
- `S3Event`: S3 event handling
- `S3TriggerRule`: S3 trigger configuration
- Lambda handler for S3 events

#### **UNIVERSAL_BRACKET_INTEGRATION_EXAMPLES.py**
- Complete examples for all interfaces
- Cross-interface workflow demonstration
- Testing and validation examples

### **2. Interface Integrations (🔄 To Implement)**

#### **CLI Integration**
```python
# Add to existing CLI
def handle_bracket_command(args):
    """Handle bracket commands in CLI."""
    from tidyllm.s3_flow_parser import get_s3_flow_parser
    
    parser = get_s3_flow_parser()
    result = await parser.cli_execute(args.bracket_command)
    
    print(f"✅ Execution ID: {result['execution_id']}")
    print(f"📋 Status: {result['status']}")

# CLI usage: 
# tidyllm "[mvr_analysis]"
# tidyllm "[robots3 embed]"  
# tidyllm bracket "[workflow_name action]"
```

#### **API Integration**
```python
# Add to existing FastAPI server
from tidyllm.s3_flow_parser import get_s3_flow_parser

@app.post("/flow/execute")
async def execute_flow_command(request: FlowRequest):
    """Execute bracket flow command."""
    parser = get_s3_flow_parser()
    result = await parser.api_execute(request.dict())
    return result

@app.post("/flow/s3-event")
async def process_s3_event(s3_event: S3EventRequest):
    """Process S3 event for workflow triggering."""
    parser = get_s3_flow_parser()
    result = await parser.s3_process_event(s3_event.dict())
    return result

# API usage:
# POST /flow/execute {"command": "[mvr_analysis]"}
# POST /flow/s3-event {"bucket_name": "...", "object_key": "..."}
```

#### **UI Integration**
```python
# Add to existing Streamlit interface
from tidyllm.s3_flow_parser import get_s3_flow_parser

def render_bracket_input_box():
    """Render bracket command input in Streamlit."""
    parser = get_s3_flow_parser()
    
    user_input = st.text_input(
        "Enter command or bracket workflow:",
        placeholder="[mvr_analysis] or natural language with [brackets]"
    )
    
    if user_input:
        # Detect brackets
        detected = parser.ui_detect_and_execute(user_input)
        
        for detection in detected:
            if detection['valid']:
                if st.button(f"Execute {detection['bracket']}"):
                    # Execute workflow
                    result = await parser.cli_execute(detection['bracket'])
                    st.success(f"Started: {result['execution_id']}")

# UI usage: Text input box that accepts "[workflow_name]"
```

#### **Chat Integration**
```python
# Add to chat interface
from tidyllm.s3_flow_parser import get_s3_flow_parser

async def process_chat_message(message: str, context: dict):
    """Process chat message for bracket commands."""
    parser = get_s3_flow_parser()
    
    result = await parser.chat_process_message(message, context)
    
    if result['has_brackets']:
        response = f"Found {len(result['brackets_found'])} workflow commands! "
        for execution in result['executions']:
            response += f"Executing {execution['bracket']}... "
        return response
    else:
        return "No workflow commands detected."

# Chat usage: "Please [mvr_analysis] these documents"
```

#### **S3 Integration**
```python
# Lambda function for S3 triggers
from tidyllm.s3_flow_parser import lambda_handler

# Deploy lambda_handler as AWS Lambda function
# Configure S3 bucket notifications to trigger Lambda

# S3 usage:
# Drop file: s3://bucket/triggers/[mvr_analysis].trigger
# Drop file: s3://bucket/mvr_tag/document.pdf → triggers mvr_analysis
```

## 🚀 **Implementation Phases**

### **Phase 1: Core Parser (Week 1)**
✅ **COMPLETED**
- [x] `universal_flow_parser.py` - Core bracket parsing
- [x] `s3_flow_parser.py` - S3 integration
- [x] YAML workflow loading
- [x] Integration examples

**Deliverables:**
- Working bracket parser
- S3 event processing
- Example workflows

### **Phase 2: Interface Integration (Week 2)**
🔄 **IN PROGRESS**
- [ ] CLI bracket command handling
- [ ] API flow endpoints
- [ ] UI bracket input components
- [ ] Chat bracket detection

**Tasks:**
1. Add bracket command to existing CLI
2. Add `/flow/*` endpoints to API server
3. Add bracket input box to Streamlit UI
4. Integrate bracket detection in chat

### **Phase 3: S3 Deployment (Week 3)**
⏳ **PLANNED**
- [ ] Lambda function deployment
- [ ] S3 bucket configuration
- [ ] Trigger file creation
- [ ] Drop zone monitoring

**Tasks:**
1. Deploy Lambda function
2. Configure S3 bucket notifications
3. Test trigger file drops
4. Validate drop zone processing

### **Phase 4: Testing & Documentation (Week 4)**
⏳ **PLANNED**
- [ ] Cross-interface testing
- [ ] Performance optimization
- [ ] User documentation
- [ ] Deployment guides

**Tasks:**
1. Test same bracket across all interfaces
2. Optimize workflow execution
3. Write user guides
4. Create deployment documentation

## 🔧 **Integration Points**

### **Existing Components to Leverage**
1. **YAML Workflows**: `workflows/mvr_analysis_flow.yaml`, `workflows/domainrag_robots3.yaml`
2. **Gateway System**: Existing `dspy`, `llm`, `heiros` gateways
3. **S3 Session Manager**: S3-first processing infrastructure
4. **CLI Framework**: Existing command structure
5. **API Server**: FastAPI application
6. **Streamlit UI**: Dashboard interface
7. **Chat Interface**: `chat_workflow_interface.py`

### **New Components to Add**
1. **Bracket Parser**: Core parsing engine (✅ done)
2. **CLI Handler**: Bracket command handling in CLI
3. **API Endpoints**: `/flow/*` REST endpoints
4. **UI Components**: Bracket input and detection
5. **Lambda Function**: S3 event processing
6. **S3 Configuration**: Bucket notification setup

## 📊 **Usage Examples**

### **Same Workflow, All Interfaces**

**YAML Definition:** `workflows/mvr_analysis_flow.yaml`

**CLI:**
```bash
$ tidyllm "[mvr_analysis]"
✅ Execution ID: mvr_analysis_20250905_143022_123456
📋 Status: running
```

**API:**
```bash
$ curl -X POST /flow/execute -d '{"command": "[mvr_analysis]"}'
{
  "execution_id": "mvr_analysis_20250905_143022_789012", 
  "status": "running"
}
```

**UI:**
```
[Text Input Box] Please run [mvr_analysis] on these documents
[Execute Button] → Workflow Started: mvr_analysis_20250905_143022_345678
```

**Chat:**
```
User: Can you help analyze these MVR docs? [mvr_analysis]
Bot: Found 1 workflow command! Executing [mvr_analysis]... 
     Workflow started with ID: mvr_analysis_20250905_143022_901234
```

**S3:**
```
Drop file: s3://workflows/triggers/[mvr_analysis].trigger
→ Lambda triggered → Workflow executed: mvr_analysis_20250905_143022_567890
```

## 🎯 **Success Criteria**

### **Technical Success**
- [ ] Same bracket command works in all 5 interfaces
- [ ] YAML workflows execute consistently
- [ ] S3-first processing maintained
- [ ] Performance: <5s response time
- [ ] Reliability: 99%+ success rate

### **User Experience Success**
- [ ] Users can discover workflows: `tidyllm workflows list`
- [ ] Users get help: `tidyllm workflow help mvr_analysis`
- [ ] Users can monitor: `tidyllm status <execution_id>`
- [ ] Clear error messages and recovery

### **Business Success**
- [ ] 10x easier workflow execution
- [ ] Reduced support tickets
- [ ] Increased user adoption
- [ ] Cross-team collaboration (CLI=devs, UI=business, S3=automation)

## 🔄 **Monitoring & Maintenance**

### **Logging Strategy**
```python
# All interfaces log to same format
logger.info(f"Bracket executed: {bracket_command}", extra={
    "interface": "cli|api|ui|chat|s3",
    "execution_id": execution_id,
    "workflow_name": workflow_name,
    "user_id": user_id,
    "success": True|False
})
```

### **Metrics to Track**
- Bracket commands per day by interface
- Workflow success rates by type
- Response times by interface
- Error rates and types
- User adoption by interface

### **Maintenance Tasks**
- Weekly: Review workflow performance
- Monthly: Update YAML definitions
- Quarterly: Optimize parsing performance
- As needed: Add new workflows

## 📚 **Documentation Plan**

### **User Documentation**
1. **Quick Start Guide**: "5 minutes to first bracket command"
2. **Interface Guides**: Specific guides for CLI, API, UI, Chat, S3
3. **Workflow Catalog**: All available workflows with examples
4. **Bracket Syntax**: Complete syntax reference

### **Developer Documentation**
1. **Integration Guide**: How to add bracket support to new interfaces
2. **Workflow Creation**: How to create new YAML workflows
3. **Deployment Guide**: Lambda and S3 setup
4. **Troubleshooting**: Common issues and solutions

## ✅ **Next Steps**

1. **Week 1**: Complete Phase 2 - Interface Integration
2. **Week 2**: Begin Phase 3 - S3 Deployment  
3. **Week 3**: Complete Phase 4 - Testing & Documentation
4. **Week 4**: Launch and user training

---

**This universal bracket system transforms TidyLLM from a collection of separate interfaces into a truly unified platform where the same simple bracket commands work everywhere - CLI, API, UI, Chat, and S3!**