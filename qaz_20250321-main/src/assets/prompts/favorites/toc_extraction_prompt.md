# TOC Extraction and Reference Discovery Prompt

## 🎯 **Objective**
Extract the Table of Contents (TOC) from academic papers and identify legitimate references that can be downloaded to expand our knowledge base.

## 📋 **Task Instructions**

### **Phase 1: TOC Extraction**
For each provided paper, extract the complete Table of Contents including:
- All section headings and subheadings
- Section numbers and hierarchy
- Page numbers (if available)
- Any appendices or supplementary materials

### **Phase 2: Reference Analysis**
From the extracted TOC, identify papers that are:
1. **Legitimate academic references** (not internal citations)
2. **Potentially downloadable** (from arXiv, research repositories)
3. **Relevant to our domain** (ML, AI, Data Science, Financial Modeling)

### **Phase 3: Paper Discovery**
For each identified reference:
1. **Search for the paper** using title and authors
2. **Validate availability** (PDF download, open access)
3. **Extract metadata** (title, authors, year, source, URL)
4. **Assess relevance** (domain match, citation count, impact)

## 🔍 **Search Strategy**

### **Primary Sources**
- **arXiv.org** - Primary source for ML/AI papers
- **Papers With Code** - ML papers with implementations
- **Google Scholar** - Citation tracking and availability
- **Research Gate** - Academic network
- **Institutional repositories** - University and lab papers

### **Validation Criteria**
- ✅ **Open Access** - Freely downloadable PDF
- ✅ **Domain Relevant** - ML, AI, Data Science, Finance
- ✅ **Recent/Impactful** - Published within last 10 years or highly cited
- ✅ **Quality Source** - Reputable conference/journal
- ✅ **Available** - PDF accessible via direct link

## 📊 **Output Format**

### **TOC Extraction**
```json
{
  "paper_title": "Title of the paper",
  "authors": "Author names",
  "year": 2023,
  "toc": {
    "sections": [
      {
        "number": "1",
        "title": "Introduction",
        "subsections": [
          {
            "number": "1.1",
            "title": "Background",
            "page": 1
          }
        ]
      }
    ]
  }
}
```

### **Reference Discovery**
```json
{
  "discovered_papers": [
    {
      "title": "Referenced Paper Title",
      "authors": "Author names",
      "year": 2022,
      "source": "arxiv",
      "url": "https://arxiv.org/pdf/paper_id.pdf",
      "relevance_score": 0.9,
      "availability": "downloadable",
      "domain": "machine_learning",
      "citation_count": 150,
      "validation_status": "verified"
    }
  ]
}
```

## 🎯 **Quality Criteria**

### **High Priority Papers**
- **Recent papers** (2020-2024) with high citation counts
- **Foundational papers** in ML/AI (transformers, attention, etc.)
- **Financial ML papers** (trading, risk management, time series)
- **Production/MLOps papers** (deployment, monitoring, best practices)

### **Exclusion Criteria**
- ❌ **Internal citations** (same authors, same institution)
- ❌ **Non-academic sources** (blogs, tutorials, non-peer-reviewed)
- ❌ **Outdated papers** (pre-2010 unless highly cited)
- ❌ **Non-domain papers** (unrelated to ML/AI/Data Science)
- ❌ **Paywalled papers** (no open access)

## 🔧 **Technical Requirements**

### **Search Tools**
- Use arXiv API for ML/AI papers
- Google Scholar for citation data
- Papers With Code for implementation availability
- Direct URL validation for download links

### **Metadata Extraction**
- Title normalization (remove special characters)
- Author name parsing (handle multiple authors)
- Year validation (extract from title or metadata)
- Source identification (conference, journal, repository)

### **Download Validation**
- Check HTTP status codes
- Validate PDF content (not HTML error pages)
- Verify file size (reasonable for academic papers)
- Test download accessibility

## 📈 **Success Metrics**

### **Extraction Quality**
- **TOC Completeness**: All sections and subsections captured
- **Accuracy**: Correct hierarchy and numbering
- **Metadata**: Complete paper information

### **Discovery Success**
- **Reference Coverage**: 80%+ of legitimate references found
- **Download Success**: 90%+ of discovered papers downloadable
- **Relevance**: 85%+ of papers relevant to our domain
- **Quality**: High citation count or recent impactful papers

### **Efficiency**
- **Processing Speed**: 5 papers per hour
- **Discovery Rate**: 10-20 new papers per source paper
- **Validation Time**: <2 minutes per paper

## 🚀 **Implementation Notes**

### **Automation Opportunities**
- **Batch processing** of multiple papers
- **Parallel downloads** for discovered papers
- **Automatic categorization** based on content
- **Duplicate detection** across papers

### **Integration Points**
- **Knowledge Base**: Add discovered papers to appropriate categories
- **Search System**: Index new papers for querying
- **TOC Database**: Build searchable TOC index
- **Citation Network**: Track paper relationships

### **Error Handling**
- **Network failures**: Retry with exponential backoff
- **Invalid URLs**: Skip and log for manual review
- **Metadata errors**: Use fallback extraction methods
- **Download failures**: Mark for manual verification

---

**This prompt enables systematic discovery and expansion of our knowledge base through intelligent TOC analysis and reference tracking!** 🎯
