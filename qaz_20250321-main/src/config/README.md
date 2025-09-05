# Configuration Directory

This directory contains all application configuration files organized by purpose.

## 📁 Configuration Files

### **Python Configuration**
- `environments.py` - Environment-specific settings and configurations
- `setup.py` - Application setup and initialization configuration

### **YAML Configuration Files**

#### **Quality Assurance & Testing**
- `qa_criteria_full.yaml` - Complete QA validation criteria and standards
- `qa_criteria_simplified.yaml` - Simplified QA criteria for quick validation
- `test_scenarios.yaml` - Test case definitions and scenarios

#### **Risk Management**
- `control_risks.yaml` - Control risk definitions and validation requirements
- `risk_categories.yaml` - Risk categorization and classification rules

#### **Subject Matter Experts**
- `sme_experts.yaml` - SME configurations and expert system definitions

## 🔧 Usage

These configuration files are loaded by the application at runtime:

```python
# Example usage in application code
from src.config.environments import load_environment_config
from src.config import load_yaml_config

# Load environment settings
config = load_environment_config()

# Load YAML configurations
qa_criteria = load_yaml_config('qa_criteria_full.yaml')
risks = load_yaml_config('control_risks.yaml')
```

## 📝 Configuration Management

### **File Types**
- **`.py` files**: Python configuration modules with logic
- **`.yaml` files**: Declarative configuration data

### **Environment Handling**
- Development configurations are version-controlled
- Production secrets should use environment variables (see `environ_settings/`)
- No sensitive data should be hardcoded in these files

### **Validation**
All YAML files are validated for:
- Schema compliance
- Required field presence  
- Data type consistency

## 🔒 Security Notes

- ✅ **Safe to commit**: These files contain no secrets or credentials
- ✅ **Application defaults**: Provide sensible defaults for all environments
- ⚠️ **Environment-specific**: Override via environment variables in production

---
**Note**: For deployment-specific configurations with secrets, see `environ_settings/` directory.