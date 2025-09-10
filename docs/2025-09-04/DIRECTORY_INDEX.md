# TidyLLM Directory Organization Index

## 📁 **Proper File Organization**

This document shows the correct organization following TidyLLM's file organization constraints.

### **✅ Scripts Directory (`/scripts/`)**
Executable files that users run directly:

```
/scripts/
├── universal_bracket_flow_examples.py    # Demo of Universal Bracket Flows
├── cli_bracket_flows.py                  # CLI interface for bracket flows  
├── api_bracket_flows.py                  # API endpoints for bracket flows
├── ui_bracket_flows.py                   # UI interface for bracket flows
├── improved_usage_examples.py            # Gateway usage examples
└── deployment/
    ├── setup_lambda.py                   # (Future) Lambda deployment
    └── configure_s3.py                   # (Future) S3 configuration
```

**Usage:**
```bash
cd scripts
python universal_bracket_flow_examples.py  # Run demo
python cli_bracket_flows.py "[mvr_analysis]"  # Test CLI
```

### **✅ Documentation Directory (`/docs/`)**
All documentation and guides:

```
/docs/
├── DIRECTORY_INDEX.md                           # This file
├── universal_bracket_flows_implementation_plan.md  # Implementation roadmap
├── chain_integration_strategy.md               # Integration strategy
└── examples/
    ├── cli_usage.md                            # (Future) CLI usage guide
    ├── api_reference.md                        # (Future) API documentation  
    └── s3_integration.md                       # (Future) S3 setup guide
```

**Usage:**
- Read implementation plans
- Follow deployment guides  
- Reference API documentation

### **✅ Library Directory (`/tidyllm/`)**
Core implementation code:

```
/tidyllm/
├── universal_flow_parser.py              # Core bracket parser
├── s3_flow_parser.py                     # S3-integrated parser
├── document_chains.py                    # Document chain operations
├── gateways/                             # Gateway system
│   ├── __init__.py
│   ├── base_gateway.py
│   ├── ai_processing_gateway.py
│   └── ...
├── knowledge_systems/                    # Knowledge resource server
└── workflows/                            # YAML workflow definitions
    ├── mvr_analysis_flow.yaml
    ├── domainrag_robots3.yaml
    └── ...
```

**Usage:**
```python
from tidyllm.universal_flow_parser import get_flow_parser
from tidyllm.s3_flow_parser import get_s3_flow_parser
```

### **✅ Tests Directory (`/tests/`)**
All test files:

```
/tests/
├── test_bracket_flows.py                 # (Future) Bracket flow tests
├── test_s3_integration.py                # (Future) S3 integration tests
├── test_universal_parser.py              # (Future) Parser unit tests
└── EVIDENCE/                             # Test evidence files
    └── ...
```

**Usage:**
```bash
cd tests
python -m pytest test_bracket_flows.py -v
```

## 🚫 **What NOT to Put in Root Directory**

The root directory should ONLY contain:
- Configuration files (`pyproject.toml`, `requirements.txt`)
- Essential documentation (`README.md`, `LICENSE`)
- Constraint files (`IMPORTANT-CONSTRAINTS-FOR-THIS-CODEBASE.md`)
- Directory organization files (this index)

**Never put in root:**
- Working code files (`.py` files with logic)
- Implementation plans
- Usage examples
- Deployment scripts
- Utility scripts

## 📋 **File Placement Decision Tree**

```
Is it executable/runnable code?
├─ YES → /scripts/
└─ NO 
   ├─ Is it documentation?
   │  ├─ YES → /docs/
   │  └─ NO
   │     ├─ Is it importable library code?
   │     │  ├─ YES → /tidyllm/
   │     │  └─ NO
   │     │     ├─ Is it a test?
   │     │     │  ├─ YES → /tests/
   │     │     │  └─ NO → Review with team
```

## ✅ **Benefits of This Organization**

### **For Users:**
- **Clear entry points**: Scripts directory shows what's runnable
- **Easy documentation**: Docs directory has all guides
- **No confusion**: Root directory isn't cluttered

### **For Developers:**
- **Predictable structure**: Know where to find files
- **Easy imports**: Clear library vs script separation  
- **Maintainable**: Clean organization scales well

### **For Teams:**
- **Onboarding**: New team members understand structure immediately
- **Collaboration**: Everyone follows same organization
- **Reviews**: Easy to verify file placement in PRs

## 🎯 **Next Steps**

1. **Existing files**: Move any remaining root files to correct directories
2. **New development**: Always place files in correct directories
3. **Documentation**: Update any file references to new locations
4. **CI/CD**: Add checks to enforce directory organization

---

**Following this organization keeps TidyLLM clean, professional, and easy to navigate for all users and contributors.**