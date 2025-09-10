# TidyLLM-X Application Template

This template provides a standardized structure for creating new TidyLLM ecosystem applications following established patterns.

## Quick Start

```bash
# 1. Copy this template
cp -r tidyllm-x-template tidyllm-{your-domain}

# 2. Rename placeholder files
# Replace {DOMAIN} with your actual domain (e.g., finance, healthcare, legal)

# 3. Install dependencies
pip install -e .[dev]

# 4. Run demo
streamlit run examples/demo.py
```

## Template Structure

```
tidyllm-{domain}/
├── tidyllm_{domain}/           # Main package
│   ├── __init__.py            # Public API exports
│   ├── core/                  # Core business logic
│   ├── analysis/              # Analysis & intelligence layer
│   ├── integration/           # External system integration
│   └── templates/             # Business templates & patterns
├── examples/                  # Demos and examples
├── tests/                     # Test suite
├── docs/                      # Documentation
├── requirements.txt           # Dependencies
├── setup.py                   # Package configuration
├── CLAUDE.md                  # Claude Code integration
└── README.md                  # Getting started guide
```

## Architecture Principles

### **1. Utility Layer Integration**
- Import from `tidyllm`, `tidyllm-sentence`, `tlm` for core functionality
- Build by composition, not duplication
- Maintain educational transparency

### **2. Business Intelligence Layer**
- Include analysis/ directory for business-friendly insights
- Provide metrics that stakeholders understand
- Enable cross-application intelligence sharing

### **3. Streamlit-First UX**
- Every application includes interactive demo
- Business-friendly interfaces over technical complexity
- Clear value demonstration from day one

### **4. Production Ready**
- Include proper error handling and logging
- Provide configuration management
- Support both demo and production deployment

## Development Checklist

### **Foundation Setup:**
- [ ] Copy and rename template files
- [ ] Update package name throughout codebase
- [ ] Configure dependencies in requirements.txt
- [ ] Set up basic imports from tidyllm utilities

### **Core Implementation:**
- [ ] Implement main business logic in core/
- [ ] Create business analysis functions in analysis/
- [ ] Add integration points in integration/
- [ ] Define business templates in templates/

### **User Experience:**
- [ ] Create Streamlit demo in examples/demo.py
- [ ] Add business-friendly interface elements
- [ ] Include portfolio/summary views
- [ ] Provide clear value demonstration

### **Quality & Documentation:**
- [ ] Write comprehensive tests
- [ ] Create CLAUDE.md for AI assistant integration
- [ ] Document API and usage patterns
- [ ] Include business value explanation

### **Ecosystem Integration:**
- [ ] Test compatibility with other tidyllm-x packages
- [ ] Follow established patterns and conventions
- [ ] Contribute shared utilities back to foundation
- [ ] Enable cross-application workflows

## Standard Dependencies

### **Required (Core):**
```python
tidyllm              # Core ML algorithms and verbs
tidyllm-sentence     # Text processing and embeddings
```

### **Common (Application Layer):**
```python
streamlit           # Interactive web interfaces
pandas             # Data manipulation
pathlib            # File system operations
typing             # Type hints
dataclasses        # Structured data
```

### **Optional (Integration):**
```python
requests           # HTTP client for APIs
click              # CLI interfaces  
pydantic          # Data validation
fastapi           # Production APIs
```

## Business Intelligence Patterns

Every tidyllm-x application should include:

1. **Efficiency Metrics** - Time/resource optimization analysis
2. **Quality Assessments** - Content/process quality scoring
3. **Business Recommendations** - Actionable insights for stakeholders
4. **Portfolio Views** - Aggregate analysis across all items
5. **Risk Indicators** - Identify potential issues or concerns

## Example Implementation Guide

See the completed `tidyllm-vectorqa` package for a reference implementation that follows these patterns:

- **Core Layer:** RAG systems and vector search
- **Analysis Layer:** Section length, signal-to-noise, links quality
- **Business Intelligence:** Investment analysis, efficiency metrics
- **Integration:** Streamlit interface with business-friendly outputs

## Contributing to Ecosystem

### **Utility Contributions:**
If you develop reusable components, consider contributing them to the utility layer:
- Abstract algorithms → `tidyllm`
- Text processing → `tidyllm-sentence`  
- ML primitives → `tlm`

### **Pattern Sharing:**
Share successful patterns with the ecosystem:
- Business analysis templates
- Integration approaches
- UX patterns
- Configuration strategies

This template ensures **consistency across the ecosystem** while **maintaining the educational transparency** and **business focus** that defines TidyLLM applications.