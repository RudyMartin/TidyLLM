# 📓 Notebooks - VectorQA Sage Functionality Demonstrations

## Overview
This folder contains Jupyter notebooks that demonstrate VectorQA Sage functionality **outside of the Streamlit app**. These notebooks provide:

- **Interactive exploration** of the system capabilities
- **Step-by-step demonstrations** of key features
- **Testing and validation** of components
- **Development and debugging** tools
- **Point-in-time benchmark** demonstrations

---

## 📋 **Available Notebooks**

### **🔧 Core Functionality**
- **[01_database_exploration.ipynb](01_database_exploration.ipynb)** - Explore PostgreSQL database schema and data
- **[02_aws_bedrock_demo.ipynb](02_aws_bedrock_demo.ipynb)** - AWS Bedrock model demonstrations
- **[03_embedding_generation.ipynb](03_embedding_generation.ipynb)** - Embedding generation and storage
- **[04_s3_standards_processing.ipynb](04_s3_standards_processing.ipynb)** - S3 standards folder processing

### **🕒 Point-in-Time Benchmarks**
- **[05_point_in_time_demo.ipynb](05_point_in_time_demo.ipynb)** - Point-in-time standards retrieval
- **[06_standards_evolution.ipynb](06_standards_evolution.ipynb)** - Standards evolution tracking
- **[07_benchmark_comparison.ipynb](07_benchmark_comparison.ipynb)** - Compare standards across time periods

### **🔍 MCP Architecture**
- **[08_mcp_coordinator_demo.ipynb](08_mcp_coordinator_demo.ipynb)** - MCP coordinator demonstrations
- **[09_standards_retrieval.ipynb](09_standards_retrieval.ipynb)** - Standards retrieval worker
- **[10_similarity_search.ipynb](10_similarity_search.ipynb)** - Vector similarity search

### **📊 Analytics & Reporting**
- **[11_qa_analytics.ipynb](11_qa_analytics.ipynb)** - QA analytics and metrics
- **[12_compliance_reporting.ipynb](12_compliance_reporting.ipynb)** - Compliance reporting
- **[13_performance_monitoring.ipynb](13_performance_monitoring.ipynb)** - Performance monitoring

---

## 🚀 **Getting Started**

### **Prerequisites**
```bash
# Install Jupyter and dependencies
pip install jupyter pandas numpy matplotlib seaborn plotly

# Set up environment variables
export AWS_ONLY_MODE=true
export DATABASE_URL=postgresql://user:password@localhost:5432/database
```

### **Running Notebooks**
```bash
# Start Jupyter server
jupyter notebook

# Or run specific notebook
jupyter notebook 01_database_exploration.ipynb
```

### **Notebook Structure**
Each notebook follows this structure:
1. **Setup & Configuration** - Environment setup and imports
2. **Data Loading** - Load data from database/S3
3. **Core Demonstration** - Main functionality demonstration
4. **Analysis & Visualization** - Results analysis and charts
5. **Conclusions** - Summary and next steps

---

## 🎯 **Key Demonstrations**

### **Point-in-Time Benchmarks**
- **Standards Evolution**: Track how standards change over time
- **Review Accuracy**: Ensure reviews use correct historical standards
- **Compliance Tracking**: Maintain audit trail of standards changes

### **MCP Architecture**
- **Coordinator Patterns**: Demonstrate MCP coordinator workflows
- **Worker Integration**: Show how workers process tasks
- **Protocol Communication**: Illustrate MCP message passing

### **AWS Bedrock Integration**
- **Model Selection**: Demonstrate different Bedrock models
- **Embedding Generation**: Show real embedding creation
- **Cost Optimization**: Illustrate cost-effective model usage

### **Database Operations**
- **Vector Search**: Demonstrate pgvector similarity search
- **Categorical Filtering**: Show "Walmart-style" categorical indexing
- **Performance Monitoring**: Track query performance

---

## 📊 **Expected Outputs**

### **Visualizations**
- Standards evolution timelines
- Similarity search results
- Performance metrics charts
- Compliance tracking dashboards

### **Data Analysis**
- Point-in-time standards comparisons
- Embedding quality metrics
- Query performance analysis
- Cost optimization insights

### **Reports**
- Standards compliance reports
- Performance benchmarking
- Cost analysis reports
- Audit trail summaries

---

## 🔧 **Configuration**

### **Database Connection**
```python
# In notebook setup
import os
from backend.database.postgres_manager import PostgreSQLManager

db_manager = PostgreSQLManager(os.getenv("DATABASE_URL"))
db_manager.connect()
```

### **AWS Configuration**
```python
# AWS Bedrock setup
from backend.core.aws_llm_manager import AWSSecurityLLMManager

llm_manager = AWSSecurityLLMManager()
```

### **S3 Configuration**
```python
# S3 standards processing
from backend.mcp.coordinators.standards_processing_coordinator import StandardsProcessingCoordinator

coordinator = StandardsProcessingCoordinator()
```

---

## 📝 **Notebook Guidelines**

### **Code Style**
- Use clear, descriptive variable names
- Add comments explaining complex logic
- Include error handling and validation
- Follow PEP 8 style guidelines

### **Documentation**
- Include markdown cells explaining concepts
- Add cell outputs for verification
- Document assumptions and limitations
- Provide context for business users

### **Performance**
- Use efficient database queries
- Implement proper connection management
- Monitor memory usage for large datasets
- Include performance metrics

---

## 🎯 **Use Cases**

### **For Developers**
- **Component Testing**: Test individual components
- **Integration Testing**: Verify component interactions
- **Performance Testing**: Benchmark system performance
- **Debugging**: Isolate and fix issues

### **For Business Users**
- **Feature Demonstrations**: Show system capabilities
- **Compliance Validation**: Verify regulatory compliance
- **Performance Analysis**: Understand system performance
- **Cost Analysis**: Evaluate cost-effectiveness

### **For Data Scientists**
- **Model Evaluation**: Assess embedding quality
- **Similarity Analysis**: Analyze document similarities
- **Trend Analysis**: Identify patterns in standards evolution
- **Optimization**: Improve system performance

---

*These notebooks provide a comprehensive way to explore and validate VectorQA Sage functionality outside of the Streamlit interface.*
