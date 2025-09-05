# Scripts Directory - Organized

This directory contains all TidyLLM operational scripts organized by functionality.

## 📁 Directory Structure

### `/demos/` (19 files)
Interactive demonstrations and showcases of TidyLLM functionality:
- Streamlit demos for all major components
- Live system demonstrations
- Welcome workflows and tutorials
- Interactive examples and solutions

### `/mvr/` (8 files)
Model Validation Report (MVR) processing and workflows:
- S3-first MVR processing pipeline
- MVR monitoring and upload systems
- Human-in-the-loop MVR interfaces
- AWS terminal workflows for MVR

### `/infrastructure/` (12 files)
System setup, configuration, and service management:
- Database setup (MLflow, PostgreSQL)
- Service orchestration and management
- Credential and session management
- S3 and AWS configuration tools

### `/testing/` (7 files)
Test scripts and validation tools:
- Drop zones testing framework
- Database integrity checks
- Component integration tests
- System validation scripts

### `/apis/` (6 files)
API development and interface tools:
- FastAPI endpoint definitions
- CLI and UI interfaces
- API usage examples and patterns
- File upload systems

### `/workflows/` (9 files)
Workflow orchestration and drop zone management:
- Production drop zone systems
- Workflow automation scripts
- RAG-to-DAG conversion tools
- Research and peer review workflows

### `/rag/` (4 files)
RAG (Retrieval Augmented Generation) system variants:
- Different RAG implementations (DSPy, Polars, TidyMart, VectorQA)
- Specialized RAG configurations
- RAG system demonstrations

## 🎯 Usage

Each directory contains specialized scripts for different operational aspects of the TidyLLM ecosystem. Navigate to the appropriate directory based on your task:

- **Learning/Testing**: Start with `/demos/`
- **Production Setup**: Use `/infrastructure/`  
- **MVR Processing**: Go to `/mvr/`
- **API Development**: Check `/apis/`
- **Workflow Creation**: Explore `/workflows/`
- **System Testing**: Use `/testing/`
- **RAG Systems**: Browse `/rag/`