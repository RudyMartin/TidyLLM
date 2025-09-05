# tidyllm-documents

**Document processing and classification toolkit with complete algorithmic transparency**

Part of the tidyllm-verse: Educational ML tools that compete with industrial systems while maintaining complete transparency.

## 🎯 Purpose

Comprehensive document processing for business workflows without external AI dependencies. Perfect for:
- Automated document classification
- Metadata extraction from business documents
- Text extraction from PDF, DOCX, and other formats
- Business document template processing

## 🚀 Quick Start

```python
import tidyllm_documents as td

# Document classification
classifier = td.DocumentClassifier(['invoice', 'contract', 'report'])
result = classifier.classify_document("business_document.pdf")
print(f"Category: {result.category} (confidence: {result.confidence})")

# Metadata extraction
extractor = td.MetadataExtractor()
metadata = extractor.extract("invoice.pdf")
print(f"Invoice Number: {metadata.get('invoice_number')}")
print(f"Total Amount: {metadata.get('total_amount')}")

# Business document processing
processor = td.BusinessDocumentProcessor()
analysis = processor.process_document("contract.docx", max_pages=5)
print(f"Document Type: {analysis.document_type}")
print(f"Key Fields: {analysis.extracted_fields}")
```

## 📋 Three Core Modules

### 1. Document Classification (`classification`)
- Multi-category document classification
- Confidence scoring and validation
- Custom category training
- Embedding-based similarity matching

### 2. Text Extraction (`extraction`)
- PDF text extraction (first N pages)
- DOCX document processing  
- Plain text file handling
- Metadata and structure preservation

### 3. Business Templates (`templates`)
- 20+ business document patterns
- Invoice processing (numbers, amounts, dates)
- Contract analysis (terms, values, parties)
- Legal document classification
- Financial statement processing

## 🏗️ Architecture

**Pure Python Implementation:**
- Zero external ML dependencies (optional PyPDF2, python-docx)
- Complete algorithmic transparency
- Educational-first design
- Production-ready performance

**Key Dependencies:**
- `tidyllm-sentence` for text embeddings and similarity
- `tlm` for core ML algorithms
- Optional: `PyPDF2` for PDF processing, `python-docx` for Word documents

## 📊 Business Document Support

### Invoice Processing
- Invoice numbers and reference IDs
- Total amounts and line items
- Due dates and payment terms
- Vendor and customer information

### Contract Analysis  
- Contract numbers and references
- Effective and expiration dates
- Contract values and terms
- Party identification and contacts

### Financial Documents
- Account numbers and routing information
- Transaction amounts and dates
- Statement periods and balances
- Regulatory compliance indicators

### Legal Documents
- Case numbers and references
- Court information and dates
- Attorney and party details
- Document classification and priority

## 🔗 tidyllm-verse Integration

Works seamlessly with:
- `tlm`: Core ML algorithms for classification
- `tidyllm-sentence`: Text embeddings and similarity matching
- `tidyllm-compliance`: Regulatory compliance monitoring

## 📊 Performance

- **Processing Speed**: Sub-second analysis for typical documents
- **Memory Usage**: Minimal footprint (~50MB with dependencies)
- **Accuracy**: Template-based precision with ML enhancement
- **Scalability**: Batch processing capabilities for hundreds of documents

## 🛡️ Business Features

- **Audit Trail**: Complete processing history and decisions
- **Deterministic Results**: Consistent output for same inputs
- **Explainable Classification**: Every decision point explained
- **Custom Templates**: Easy addition of new document types

---

**Built with ❤️ using [CodeBreakers](https://github.com/rudymartin/codebreakers_manifesto) / tidyllm verse**

*Where document processing meets algorithmic sovereignty*