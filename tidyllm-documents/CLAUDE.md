# CLAUDE.md - tidyllm-documents

## Repository Purpose
**tidyllm-documents** provides comprehensive document processing and classification with complete algorithmic transparency. This is part of the **tidyllm-verse** ecosystem of educational ML tools.

## Core Functionality
- **Document Classification**: Multi-category classification with confidence scoring
- **Text Extraction**: PDF, DOCX, TXT processing with configurable page limits
- **Metadata Extraction**: 20+ business document patterns (invoice numbers, amounts, dates, etc.)
- **Business Templates**: Specialized processing for invoices, contracts, purchase orders, financial statements

## Architecture
```
tidyllm_documents/
├── classification/   # Document categorization and confidence scoring
├── extraction/       # Text and metadata extraction from files
├── templates/        # Business document templates and patterns
└── __init__.py       # Main API exports
```

## Key Dependencies
- **tidyllm-sentence**: Text embeddings for semantic classification (external package)
- **tlm**: Core ML algorithms for classification (external package)
- **PyPDF2**: PDF processing (optional, install with `pip install tidyllm-documents[pdf]`)
- **python-docx**: Word document processing (optional, install with `pip install tidyllm-documents[docx]`)

## Philosophy
- **Educational Transparency**: Every processing step is visible and understandable
- **Business Focus**: Optimized for common business document types
- **Pattern-Based**: Uses regex patterns with validation for reliability
- **Confidence Scoring**: Every extraction includes confidence assessment

## Common Commands
```bash
# Install basic version
pip install tidyllm-documents

# Install with full document support
pip install tidyllm-documents[full]

# Install for development
pip install -e .[dev]

# Run business processing demo
python examples/business_processing_demo.py

# Run tests
pytest tests/

# Build package
python -m build
```

## Main Classes
- **DocumentClassifier**: Multi-category document classification
- **TextExtractor**: Extract text from PDF, DOCX, TXT files
- **MetadataExtractor**: Extract structured data using business patterns
- **BusinessDocumentProcessor**: Complete business document workflow

## Business Document Patterns
The system includes 15+ extraction patterns:
- **Financial**: invoice_number, total_amount, account_number, tax_id
- **Dates**: invoice_date, due_date, contract_date
- **Contact**: email_address, phone_number
- **Legal**: contract_number, reference_number, project_code
- **Purchase Orders**: purchase_order, vendor information

## Integration with tidyllm-verse
This package works seamlessly with:
- **tidyllm-sentence**: Provides embeddings for semantic document classification
- **tlm**: Core ML algorithms for clustering and analysis
- **tidyllm-compliance**: Uses documents for compliance monitoring

## Use Cases
1. **Business Automation**: Automated invoice and contract processing
2. **Document Management**: Classify and route documents automatically
3. **Data Extraction**: Extract structured data from unstructured documents
4. **Compliance Support**: Process documents for regulatory analysis

## Performance Characteristics
- **Processing Speed**: Sub-second analysis for typical business documents
- **Memory Usage**: ~50MB with dependencies (vs 1GB+ for typical ML document systems)
- **Format Support**: PDF, DOCX, TXT with graceful fallbacks
- **Accuracy**: Template-based precision with ML enhancement

## Educational Features
- **Complete Source Transparency**: Every extraction pattern is visible
- **Business Focus**: Real-world document types and patterns
- **Confidence Scoring**: Understand reliability of each extraction
- **Extensible Design**: Easy to add custom patterns and document types

This repository demonstrates **algorithmic sovereignty** for document processing - you control every classification rule and extraction pattern.