# Utility Scripts

General-purpose utilities for document processing, analysis, and system building.

## Scripts

### **simple_domain_rag_builder.py**
Domain-specific RAG system builder:
- Creates specialized knowledge domains from document sets
- Builds embedding indexes for domain-specific queries
- Configures temporal resolution for conflicting documents
- Generates domain knowledge hierarchies

### **simple_pdf_sorter.py**
Document processing and organization tool:
- Sorts PDF documents by content type and date
- Extracts metadata for document classification
- Organizes files into structured directory hierarchies
- Prepares documents for embedding processing

### **submodule_analyzer.py**
Codebase analysis and dependency mapping:
- Analyzes Python module dependencies
- Maps import relationships between components
- Identifies circular dependencies
- Reports on code organization patterns
- Generates module structure documentation

## Usage

Run utilities individually for specific tasks:
```bash
python scripts/utilities/simple_domain_rag_builder.py --domain "audit_procedures"
python scripts/utilities/simple_pdf_sorter.py --input docs/ --output sorted_docs/
python scripts/utilities/submodule_analyzer.py --target tidyllm/
```

These tools support the broader TidyLLM ecosystem by providing specialized processing capabilities.