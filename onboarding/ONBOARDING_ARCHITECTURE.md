# TidyLLM Onboarding System - Clean Architecture

## 🎯 **NORMALIZED STRUCTURE**

### **Core Files (Keep)**
```
onboarding/
├── app.py                    # Single Streamlit application
├── launcher.py              # Single entry point
├── config/
│   ├── __init__.py
│   ├── manager.py           # Configuration management
│   └── templates.py         # Configuration templates
├── core/
│   ├── __init__.py
│   ├── session_manager.py   # Unified session management
│   ├── validator.py         # Connection validation
│   └── preflight.py         # Pre-flight tests
├── ui/
│   ├── __init__.py
│   ├── pages/
│   │   ├── __init__.py
│   │   ├── connection.py    # Connection Config page
│   │   ├── chat.py          # Chat Test page
│   │   ├── knowledge.py     # DomainRAG CRUD page
│   │   ├── workflows.py     # Workflows page
│   │   ├── testing.py       # Test Workflow page
│   │   └── dashboard.py     # Dashboard page
│   └── components/
│       ├── __init__.py
│       ├── sidebar.py       # Navigation sidebar
│       └── utils.py         # UI utilities
├── requirements.txt         # Single requirements file
├── README.md               # Single documentation
└── .env.example            # Environment template
```

### **Files to Remove (Frankenstein)**
- `cli_onboarding.py` (replaced by launcher.py)
- `enhanced_cli_onboarding.py` (replaced by launcher.py)
- `streamlit_app.py` (replaced by app.py)
- `enhanced_streamlit_formatted_onboarding_kit.py` (replaced by app.py)
- `integrated_streamlit_app.py` (replaced by app.py)
- `session_validator.py` (replaced by core/validator.py)
- `enhanced_session_validator.py` (replaced by core/validator.py)
- `start_onboarding.py` (replaced by launcher.py)
- `unified_streamlit_manager.py` (replaced by core/session_manager.py)
- `universal_preflight.py` (replaced by core/preflight.py)
- `config_generator.py` (replaced by centralized settings manager)
- `tabs/` directory (replaced by ui/pages/)
- `README_ONBOARDING_KIT.md` (replaced by README.md)
- `requirements_onboarding_kit.txt` (replaced by requirements.txt)
- All test files (consolidated into core/preflight.py)

## 🏗️ **ARCHITECTURE PRINCIPLES**

### **1. Single Responsibility**
- Each file has one clear purpose
- No overlapping functionality
- Clean separation of concerns

### **2. Modular Design**
- Core functionality in `core/`
- UI components in `ui/`
- Configuration in `config/`
- Easy to maintain and extend

### **3. Single Entry Point**
- `launcher.py` - One way to start the system
- `app.py` - One Streamlit application
- No confusion about which file to run

### **4. Clean Dependencies**
- Single `requirements.txt`
- No duplicate dependencies
- Clear version management

### **5. Unified Documentation**
- Single `README.md`
- Clear setup instructions
- No conflicting documentation
