# Drop Zones Configuration Management

**Status**: 🎯 **PLANNED** - Config-driven parameterized workflow system  
**Date**: September 7, 2025  
**Purpose**: Eliminate duplicate scripts with different parameters

---

## 🚨 **Problem Identified**

We have **multiple similar scripts with different parameters** that should be **config-driven** instead of hard-coded:

### **Current Duplicate Pattern Issues:**
- ❌ `scripts/mvr_analysis_stage1.py`, `scripts/mvr_analysis_stage2.py`, etc.
- ❌ Different drop zone folders with same processing logic
- ❌ Hard-coded file paths, timeouts, gateways in scripts
- ❌ Copy-paste code with slight parameter variations
- ❌ Maintenance nightmare when changing core logic

---

## ✅ **Solution: Config-Driven Parameterized System**

### **Single Universal Script + YAML Configs**

Instead of multiple scripts, we have:
```
drop_zones/
├── config/
│   ├── workflows/              # Workflow definitions
│   │   ├── qa_control.yaml     # QA Control parameters
│   │   ├── mvr_analysis.yaml   # MVR Analysis parameters  
│   │   ├── knowledge_base.yaml # Knowledge processing parameters
│   │   └── custom_workflow.yaml
│   ├── environments/           # Environment-specific overrides
│   │   ├── development.yaml
│   │   ├── staging.yaml
│   │   └── production.yaml
│   └── templates/              # Reusable workflow templates
│       ├── standard_pipeline.yaml
│       ├── domain_rag.yaml
│       └── compliance_review.yaml
└── universal_processor.py      # Single parameterized script
```

---

## 🎯 **Implementation Plan**

### **Phase 1: Configuration System**
- [ ] Create workflow config schema
- [ ] Build config validation system
- [ ] Environment-specific parameter overrides
- [ ] Template inheritance system

### **Phase 2: Universal Processor**
- [ ] Single `universal_processor.py` script
- [ ] Dynamic workflow loading from YAML
- [ ] Parameter injection system
- [ ] Multi-environment support

### **Phase 3: Migration**
- [ ] Convert existing workflows to YAML configs
- [ ] Replace duplicate scripts with config calls
- [ ] Update documentation and examples
- [ ] Test all workflow combinations

---

## 🏗️ **Config Schema Design**

### **Workflow Config Structure:**
```yaml
# drop_zones/config/workflows/qa_control.yaml
workflow_name: "QA Control"
workflow_id: "qa_control_v1"
description: "6-stage QA pipeline with Excel processing"

# Environment-specific parameters
parameters:
  source_folder: "qa_control/incoming"
  processing_folder: "qa_control/processing" 
  output_folder: "qa_control/reports"
  file_patterns: ["*.xlsx", "*.pdf"]
  max_timeout: 300
  retry_attempts: 3

# Processing stages
stages:
  - name: "ingest"
    gateway: "heiros"
    timeout: "${parameters.max_timeout}"
    input_path: "${parameters.source_folder}"
    output_path: "qa_control/ingest"
    
  - name: "extract"  
    gateway: "dspy"
    depends_on: ["ingest"]
    # ... etc
```

### **Environment Override:**
```yaml
# drop_zones/config/environments/production.yaml
parameters:
  max_timeout: 600  # Longer timeout in production
  retry_attempts: 5 # More retries in production
  
gateways:
  primary: "production-gateway"  # Different gateway
```

---

## 💡 **Usage Examples**

### **Instead of Multiple Scripts:**
```bash
# ❌ OLD WAY - Multiple duplicate scripts
python scripts/qa_stage1_processor.py
python scripts/qa_stage2_processor.py  
python scripts/mvr_analysis_runner.py --stage=1
python scripts/mvr_analysis_runner.py --stage=2
```

### **New Config-Driven Approach:**
```bash
# ✅ NEW WAY - Single script + config
python drop_zones/universal_processor.py --workflow=qa_control
python drop_zones/universal_processor.py --workflow=mvr_analysis
python drop_zones/universal_processor.py --workflow=qa_control --env=production
python drop_zones/universal_processor.py --workflow=custom_workflow --stage=extract
```

---

## 🎯 **Benefits**

✅ **Eliminate code duplication** - One script, many configs  
✅ **Easy parameter changes** - Edit YAML, not Python code  
✅ **Environment consistency** - Same logic, different parameters  
✅ **Rapid workflow creation** - Copy template, modify parameters  
✅ **Centralized maintenance** - Fix bug once, affects all workflows  
✅ **Version control friendly** - Config changes tracked separately  
✅ **Testing simplified** - Test one script with various configs  

---

## 🔧 **Integration with Existing Systems**

### **Leverages Existing Infrastructure:**
- ✅ **YAML workflows** already exist in `tidyllm/workflows/`
- ✅ **Universal Bracket Flows** already documented
- ✅ **Gateway system** already implemented
- ✅ **Drop Zones Architecture** already established
- ✅ **S3 tiered structure** already configured

### **Builds Upon:**
- Existing `mvr_analysis_flow.yaml` structure
- Universal Flow Parser patterns
- TidyLLM gateway configuration
- AWS S3 prefix organization

---

**Next Step**: Implement Phase 1 configuration system to consolidate all parameterized workflows! 🚀