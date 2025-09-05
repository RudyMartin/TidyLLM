# Backend Layer

## Overview
The backend layer contains all server-side logic, API endpoints, business logic, and data processing capabilities for VectorQA Sage.

## Structure
- `api/` - API endpoints and controllers
- `core/` - Core business logic and essential components
- `models/` - Data models and schemas
- `services/` - Business services
- `utils/` - Utility functions and helpers
- `config/` - Configuration and credentials
- `cli/` - Command-line interface tools

## Key Components

### API Layer (`api/`)
- `api_status_dashboard.py` - API status monitoring and dashboard generation
- `quick_status_check.py` - Fast API health checks
- `generate_html_dashboard.py` - HTML dashboard generation utilities

### Core Logic (`core/`)
- `llm_manager.py` - LLM provider management and interactions
- `dspy_config.py` - DSPy framework configuration
- `config.py` - Application configuration management
- `gemini_client.py` - Custom Google Gemini API client
- `validator.py` - Data validation and QA validation logic
- `embedding_helper.py` - Vector embedding generation and management
- `normalize_labels.py` - Label normalization utilities
- `extraction_helper.py` - Data extraction and processing
- `qa_log_utils.py` - QA logging utilities
- `report_export.py` - Report generation and export functionality
- `dsp_pipeline_loader.py` - DSPy pipeline loading utilities
- `dspy_prompt_config.py` - DSPy prompt configuration

### Utilities (`utils/`)
- `example_manager.py` - Example data management
- `safe_cleanup.py` - Safe dependency cleanup utilities
- `check_compatibility.py` - Dependency compatibility checking
- `safe_upgrade.py` - Safe dependency upgrade utilities

### Configuration (`config/`)
- `credentials.env` - API keys and sensitive configuration
- `credentials_template.env` - Template for credential setup
- `credential_manager.py` - Credential management utilities

### CLI Tools (`cli/`)
- `run_pipeline.py` - Command-line tool to run DSPy pipelines
- `topic_generator.py` - Command-line tool to generate topics for QA analysis

## Features
- Multi-provider LLM support (OpenAI, Cohere, Google Gemini, Hugging Face)
- DSPy integration for advanced prompt optimization
- Robust error handling with fallback mechanisms
- Secure credential and configuration management
- Real-time API status and health monitoring

## Development
The backend is designed to be modular and extensible. Each component has a specific responsibility and can be developed and tested independently.
