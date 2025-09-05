# 📄 TidyLLM-Papers

**Research paper processing for TidyLLM ecosystem with seamless LLM integration**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TidyLLM Compatible](https://img.shields.io/badge/TidyLLM-compatible-green.svg)](https://github.com/tidyllm/tidyllm)
[![ArXiv Integration](https://img.shields.io/badge/ArXiv-integrated-orange.svg)](https://arxiv.org/)

## 🎯 **Overview**

TidyLLM-Papers extends the TidyLLM ecosystem with specialized research paper discovery, processing, and analysis capabilities. It follows the same pipeline grammar as LLMData and integrates seamlessly with existing MCP infrastructure.

### **Key Features**

- 🔍 **ArXiv Discovery**: Search and retrieve papers with advanced filtering
- 📚 **Content Analysis**: Extract text, images, and tables from PDFs
- 🔗 **Citation Networks**: Analyze references and collaboration patterns  
- 📋 **Format Export**: Generate BibTeX, APA, MLA citations
- 🤖 **LLM Integration**: Direct attachment to Claude, GPT, and other models
- ⚡ **Pipeline Grammar**: Same `|` operator syntax as TidyLLM

## 🚀 **Quick Start**

### **Installation**
```bash
pip install tidyllm-papers
```

### **Basic Usage**
```python
from tidyllm_papers import papers, discover, analyze, cite

# Discover recent AI papers
research = (papers("artificial intelligence")
           | discover.arxiv(limit=10)
           | discover.recent(days=30))

print(f"Found {len(research.papers)} papers")
```

### **LLM Integration**
```python
import llmdata
from tidyllm_papers import papers, discover

# Research and analyze with Claude
insights = (papers("attention mechanisms")
           | discover.arxiv(limit=5)
           | as_attachments()
           | llm_message("Summarize key innovations")
           | chat(claude()))
```

## 📖 **Core Grammar**

### **Discovery Verbs**
```python
# ArXiv search with filters
papers("machine learning") | discover.arxiv(limit=10, sort_by="relevance")

# Recent papers (last N days)  
papers("neural networks") | discover.recent(days=7)

# Category-specific search
papers("computer vision") | discover.by_category(["cs.CV", "cs.AI"])

# Random sampling for exploration
papers("deep learning") | discover.sample(5)
```

### **Analysis Verbs**
```python
# Download PDFs
research | analyze.download("./papers", max_papers=10)

# Extract content with images/tables
research | analyze.content(extract_images=True, extract_tables=True)

# Analyze abstracts for themes
research | analyze.abstracts()

# Metadata and trends analysis
research | analyze.metadata()
```

### **Citation Verbs**
```python
# Extract references and citations
research | cite.extract_references()

# Generate BibTeX bibliography
research | cite.format_bibtex("references.bib")

# Network analysis
research | cite.network_analysis()

# Export in citation formats
research | cite.export_references("apa", "citations.txt")
```

### **LLM Attachment**
```python
# Convert to LLMData attachments
research | as_attachments(include_pdfs=True, max_papers=5)

# Direct LLM message conversion
research | to_llmdata("Analyze these papers: {query}")
```

## 🔧 **Integration with Existing Infrastructure**

### **MCP Workers**
Automatically integrates with existing workers:
- `PDFProcessingWorker` for content extraction
- `ImageProcessingWorker` for figure analysis
- `FileClassificationWorker` for document typing

### **TidyLLM Attachments**
```python
# Works with existing attachments grammar
processed = (research 
            | as_attachments()
            | load.auto()
            | present.markdown() + present.images()
            | refine.add_headers())
```

### **MLFlow Integration**
```python
import mlflow

with mlflow.start_run():
    research = (papers("transformers")
               | discover.arxiv(10)
               | analyze.content())
    
    mlflow.log_metric("papers_found", len(research.papers))
    mlflow.log_dict(research.analysis_results, "analysis.json")
```

## 📊 **Complete Example**

```python
#!/usr/bin/env python3
"""Complete research analysis workflow"""

import llmdata
from tidyllm_papers import papers, discover, analyze, cite

def analyze_attention_research():
    """Comprehensive attention mechanism analysis"""
    
    # 1. Discover and download
    research = (papers("attention mechanisms transformer")
               | discover.arxiv(limit=15, sort_by="relevance")
               | discover.recent(days=30)
               | analyze.download("./attention_papers")
               | analyze.content(extract_images=True))
    
    print(f"📥 Downloaded {research.stats['downloaded']} papers")
    
    # 2. Content analysis
    research = (research
               | analyze.abstracts()
               | analyze.metadata()
               | cite.extract_references()
               | cite.network_analysis())
    
    print(f"📚 Analyzed {research.stats['total_citations']} citations")
    
    # 3. Generate bibliography
    research = research | cite.format_bibtex("attention_refs.bib")
    
    # 4. LLM insights
    insights = (research
               | as_attachments(max_papers=10)
               | llm_message("Analyze evolution of attention mechanisms")
               | chat(claude(model="claude-3-5-sonnet")))
    
    # 5. Export results
    research.save_to_file("attention_analysis.json")
    
    with open("attention_insights.md", "w") as f:
        f.write(f"# Attention Research Analysis\n\n{insights.content}")
    
    return research, insights

if __name__ == "__main__":
    research, insights = analyze_attention_research()
    print("✅ Analysis complete!")
    print(research.summary())
```

## 🏢 **Enterprise Use Cases**

### **Model Risk Management**
```python
# Latest MRM research
mrm_papers = (papers("model risk management validation")
             | discover.arxiv(20)
             | discover.by_category(["q-fin.RM", "stat.ML"])
             | analyze.abstracts()
             | cite.extract_references())

# Integrate with compliance workflow
compliance_brief = (mrm_papers
                   | as_attachments()
                   | llm_message("Summarize regulatory implications")
                   | chat(claude()))
```

### **Technology Scouting**
```python
# Monitor emerging AI
emerging = (papers("generative AI applications")
           | discover.recent(days=30)
           | analyze.abstracts()
           | as_attachments()
           | llm_message("Identify commercial opportunities")
           | chat(openai(model="gpt-4o")))
```

### **Research Intelligence**
```python
# Track research trends
trends = (papers("large language models")
         | discover.by_category(["cs.CL", "cs.AI"], limit=50)
         | analyze.metadata()
         | cite.network_analysis())

# Export for strategic planning
trends | cite.export_references("apa", "llm_trends.txt")
```

## 🔧 **Configuration**

### **ArXiv Settings**
```python
# Rate limiting (default: 3 requests/second)
discover.arxiv.rate_limit = 2

# Download settings
analyze.download.default_dir = "./research_papers"
analyze.download.timeout = 60
```

### **Content Processing**
```python
# Content extraction limits
analyze.content.max_length = 50000
analyze.content.extract_images = True
analyze.content.extract_tables = True
```

### **Integration Settings**
```python
# Check integration status
from tidyllm_papers import ARXIV_AVAILABLE, LLMDATA_INTEGRATION
print(f"ArXiv: {ARXIV_AVAILABLE}")
print(f"LLMData: {LLMDATA_INTEGRATION}")
```

## 📋 **API Reference**

### **Core Classes**
- `Paper`: Individual research paper with metadata
- `PaperCollection`: Collection with pipeline processing
- `papers(query)`: Factory function for collections

### **Discovery Operations**
- `discover.arxiv()`: ArXiv search
- `discover.recent()`: Recent papers filter
- `discover.by_category()`: Category-specific search
- `discover.sample()`: Random sampling

### **Analysis Operations**
- `analyze.download()`: PDF download
- `analyze.content()`: Text/image/table extraction  
- `analyze.abstracts()`: Abstract analysis
- `analyze.metadata()`: Metadata trends

### **Citation Operations**
- `cite.extract_references()`: Reference extraction
- `cite.format_bibtex()`: BibTeX generation
- `cite.network_analysis()`: Citation networks
- `cite.export_references()`: Format export

### **Integration Functions**
- `as_attachments()`: Convert to LLMData attachments
- `to_llmdata()`: Direct LLMMessage conversion
- `save_for_llm_analysis()`: Export for analysis

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Follow TidyLLM grammar patterns
4. Add tests and documentation
5. Submit a pull request

## 📄 **License**

MIT License - see [LICENSE](LICENSE) file for details.

## 🔗 **Related Projects**

- [LLMData](../llmdata/): Core TidyLLM ecosystem
- [TidyLLM](https://github.com/tidyllm/tidyllm): R package inspiration
- [ArXiv API](https://arxiv.org/help/api): Paper discovery backend

---

**Built with ❤️ for the TidyLLM ecosystem**