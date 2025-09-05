# Assets Directory

This directory contains static assets used for report generation and application branding.

## 📁 Contents

### **LaTeX Templates**
- `logo.tex` - Corporate logo template for LaTeX documents
- `control_risks_report_template.tex` - Template for control risk assessment reports

### **Prompt Templates**
- `prompts/test_prompt.md` - QA review prompt template for document processing
- `prompts/demo_document.txt` - Sample document for testing QA functionality

## 🔧 Usage

These assets are used by the report generation and QA systems:

```python
# Example usage in report generation
from pathlib import Path

assets_dir = Path(__file__).parent / 'assets'

# LaTeX templates
logo_template = assets_dir / 'logo.tex'
report_template = assets_dir / 'control_risks_report_template.tex'

# Prompt templates  
prompt_template = assets_dir / 'prompts' / 'test_prompt.md'
demo_document = assets_dir / 'prompts' / 'demo_document.txt'

# Use in LaTeX report compilation
compile_latex_report(
    template=report_template,
    logo=logo_template,
    data=report_data
)

# Use in QA processing
qa_system.process_document(
    prompt_template=prompt_template,
    test_document=demo_document
)
```

## 📝 Asset Types

### **LaTeX Templates**
- **Purpose**: Generate formatted PDF reports
- **Usage**: Loaded by report generation scripts
- **Format**: LaTeX/TikZ for professional document formatting

### **Prompt Templates**
- **Purpose**: Standardized prompts for QA document processing
- **Usage**: Loaded by QA orchestration system
- **Format**: Markdown with structured instructions and requirements

### **Branding Assets**  
- **Logo**: Corporate branding for reports
- **Colors**: Defined in LaTeX templates (corporateblue: RGB(31,78,121))
- **Styling**: Consistent formatting across all generated documents

## 🎨 Customization

To modify report appearance:
1. Edit LaTeX templates in this directory
2. Update color definitions and styling
3. Test with report generation scripts
4. Deploy updated templates

## 🔗 Related Files

- `scripts/generate_pdf_report.py` - Uses these templates
- `src/static/` - Web application static assets (separate from report assets)
- Report outputs are generated to `data/output/reports/`

---
**Note**: These are template assets for document generation, not web application assets.