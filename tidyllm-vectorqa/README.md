# TidyLLM Vector QA

**Comprehensive Vector QA Package - The Complete TidyLLM Ecosystem**

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Version](https://img.shields.io/badge/version-0.0.1-green.svg)]()
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 🎯 Overview

TidyLLM Vector QA consolidates the entire TidyLLM ecosystem into one comprehensive package, providing:

- **Document Processing**: Extract and analyze text from PDFs, DOCX, TXT files
- **Sentence Embeddings**: High-performance semantic similarity calculations  
- **Core ML Algorithms**: Transparent machine learning implementations
- **Y=R+S+N Framework**: Mathematical content quality analysis
- **Whitepapers Analysis**: Research paper evaluation and structure extraction

## 🚀 Quick Start

### Installation

```bash
# Basic installation
pip install tidyllm-vectorqa

# Full installation with all features
pip install tidyllm-vectorqa[full]

# Development installation
pip install -e .[dev]
```

### Usage

```python
import tidyllm_vectorqa as tlm

# Y=R+S+N Framework Analysis
framework = tlm.ResearchFramework()
analysis = framework.analyze_paper_content(
    title="Mathematical Decomposition Framework",
    abstract="This paper presents..."
)

print(f"Y Score: {analysis.y_score:.3f}")
print(f"Relevant: {analysis.relevant:.1%}")
print(f"Superfluous: {analysis.superfluous:.1%}")
print(f"Noise: {analysis.noise:.1%}")

# Document Processing
extractor = tlm.TextExtractor()
text, metadata = extractor.extract_text("document.pdf")

# Table of Contents & Bibliography
toc = tlm.extract_table_of_contents(text)
references = tlm.extract_bibliography(text)
```

## 📦 Package Structure

```
tidyllm_vectorqa/
├── documents/          # Document processing & metadata extraction
├── sentence/           # Sentence embeddings & similarity
├── core/              # Core ML algorithms & attention mechanisms  
├── yrsn/              # Y=R+S+N mathematical framework
└── whitepapers/       # Research paper analysis & Streamlit demo
```

## 🔬 Y=R+S+N Framework

The **Y=R+S+N Mathematical Decomposition Framework** is the core innovation, created by **Rudy Martin**:

- **Y = R + S + N** (where R + S + N = 1.0)
- **Y Score = R + (0.5 × S)** (relevance metric)
- **Context Collapse Risk = S + (1.5 × N)** (quality risk)

### Components

- **R (Relevant)**: Core systematic content (25-85%)
- **S (Superfluous)**: Marginally systematic content (10-45%)  
- **N (Noise)**: True noise and errors (5-35%)

## 🛠️ Features

### Document Processing
- Multi-format support: PDF, DOCX, TXT
- Metadata extraction with confidence scoring
- Business document templates

### Research Analysis
- Academic paper structure extraction
- Table of contents parsing
- Bibliography/references extraction
- Citation network analysis

### ML & Embeddings  
- Sentence similarity calculations
- Attention mechanism implementations
- Transparent algorithmic approaches

### Web Interface
- Streamlit demo application
- Interactive Y=R+S+N analysis
- PDF report generation
- LaTeX document export

## 🎯 Use Cases

### Academic Research
```python
# Analyze paper quality
framework = tlm.ResearchFramework()
paper = framework.analyze_paper_content(title, abstract)

if paper.y_score > 0.8:
    print("✅ High-quality paper - suitable for primary source")
elif paper.y_score > 0.6:
    print("🔵 Good paper - useful for supporting material")
else:
    print("⚠️ Proceed with caution")
```

### Business Documents
```python
# Process business documents
processor = tlm.BusinessDocumentProcessor()
result = processor.process_document("invoice.pdf")

print(f"Document type: {result['document_type']}")
print(f"Confidence: {result['confidence']:.3f}")
```

### Content Quality Assessment
```python
# Analyze content structure
toc = tlm.extract_table_of_contents(document_text)
references = tlm.extract_bibliography(document_text)

print(f"Sections: {len(toc)}")
print(f"References: {len(references)}")
```

## 🚀 Demo Application

Launch the interactive Streamlit demo:

```bash
tidyllm-demo
```

Or run directly:

```bash
streamlit run tidyllm_vectorqa/whitepapers/app.py
```

## 📊 Example Analysis Output

```
Paper: "Mathematical Decomposition of Signal and Noise"
Y Score: 0.863 (Excellent!)

R+S+N Decomposition:
├── R (Relevant): 78.0% - Core systematic content
├── S (Superfluous): 15.0% - Background information  
└── N (Noise): 7.0% - Minimal errors

Context Collapse Risk: 0.255 (Low)
Recommendation: ✅ Use as primary source
```

## 🔧 Development

```bash
# Clone repository
git clone https://github.com/tidyllm/tidyllm-vectorqa.git
cd tidyllm-vectorqa

# Install in development mode
pip install -e .[dev]

# Run tests
pytest

# Run demo
streamlit run tidyllm_vectorqa/whitepapers/app.py
```

## 📄 License

This project is licensed under the Creative Commons Attribution 4.0 International License.

## 👨‍💼 Author & Attribution

**Y=R+S+N Mathematical Framework**: Created by **Rudy Martin**

The Y=R+S+N framework represents a novel approach to mathematical content decomposition and context collapse prevention, developed specifically for educational ML and research integrity applications.

## 🤝 Contributing

We welcome contributions! This package represents the consolidation of multiple TidyLLM ecosystem projects into a unified, powerful toolkit for vector-based question answering and document analysis.

---

**Part of the TidyLLM-verse: Educational ML with Complete Transparency**