#!/usr/bin/env python3
"""
TidyLLM - The Great Walled City of Enterprise AI
Setup script for package installation with all dependencies
"""

from setuptools import setup, find_packages
from pathlib import Path
import os

# Read the README file for long description
README_PATH = Path(__file__).parent / "README.md"
if README_PATH.exists():
    with open(README_PATH, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "TidyLLM - The Great Walled City of Enterprise AI"

# Read version from tidyllm/__init__.py
def get_version():
    init_path = Path(__file__).parent / "tidyllm" / "__init__.py"
    if init_path.exists():
        with open(init_path, "r") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"').strip("'")
    return "1.0.0"

# Core dependencies that are always required
CORE_REQUIREMENTS = [
    # Core Python data processing
    "pyyaml>=6.0",
    "requests>=2.28.0",
    "pandas>=1.5.0",
    "openpyxl>=3.0.0",  # For Excel file processing
    
    # AWS and cloud services
    "boto3>=1.26.0",
    "botocore>=1.29.0",
    
    # Database connectivity
    "psycopg2-binary>=2.9.0",
    
    # ML and AI frameworks
    "mlflow>=2.0.0",
    "dspy-ai>=2.4.0",
    "openai>=1.0.0",
    "anthropic>=0.25.0",
    
    # System monitoring
    "psutil>=5.8.0",
    
    # Type hints and utilities
    "typing-extensions>=4.0.0",
    
    # TidyLLM Ecosystem - local packages auto-installed
    "tlm @ file://./tlm",
    "tidyllm-sentence @ file://./tidyllm-sentence",
]

# Optional dependencies for enhanced features
OPTIONAL_REQUIREMENTS = {
    # Web interface and visualization
    "web": [
        "streamlit>=1.28.0",
        "plotly>=5.15.0",
        "rich>=13.0.0",
    ],
    
    # Advanced data processing
    "data": [
        "numpy>=1.24.0",
        "polars>=0.20.0",
        "pyarrow>=10.0.0",
    ],
    
    # Document processing
    "documents": [
        "PyPDF2>=3.0.0",
        "pdfplumber>=0.9.0",
        "python-docx>=0.8.11",
        "beautifulsoup4>=4.11.0",
    ],
    
    # Development and testing
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "mypy>=1.0.0",
        "pre-commit>=3.0.0",
    ],
    
    # Extended AI capabilities
    "ai": [
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "sentence-transformers>=2.2.0",
        "langchain>=0.1.0",
        "chromadb>=0.4.0",
    ],
    
    # Additional TidyLLM libraries (archived - install manually)
    "documents": [
        "tidyllm-documents @ file://./_archived/tidyllm-documents",
    ],
    "vectorqa": [
        "tidyllm-vectorqa @ file://./_archived/tidyllm-vectorqa",
    ]
}

# All optional dependencies combined
ALL_OPTIONAL = []
for deps in OPTIONAL_REQUIREMENTS.values():
    ALL_OPTIONAL.extend(deps)

OPTIONAL_REQUIREMENTS["all"] = ALL_OPTIONAL

setup(
    name="tidyllm",
    version=get_version(),
    author="TidyLLM Development Team",
    author_email="info@tidyllm.ai",
    description="The Great Walled City of Enterprise AI - A comprehensive AI/ML library for enterprise workflows",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tidyllm/tidyllm",
    project_urls={
        "Bug Tracker": "https://github.com/tidyllm/tidyllm/issues",
        "Documentation": "https://docs.tidyllm.ai",
        "Source Code": "https://github.com/tidyllm/tidyllm",
    },
    packages=find_packages(include=["tidyllm", "tidyllm.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research", 
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Office/Business",
    ],
    python_requires=">=3.8",
    install_requires=CORE_REQUIREMENTS,
    extras_require=OPTIONAL_REQUIREMENTS,
    include_package_data=True,
    package_data={
        "tidyllm": [
            "admin/*.yaml",
            "admin/*.json",
            "workflows/*.yaml",
            "workflows/*.json",
            "knowledge_systems/**/*.yaml",
            "knowledge_systems/**/*.json",
            "gateways/**/*.yaml",
            "gateways/**/*.json",
            "*.md",
            "*.txt",
            "*.yaml",
        ],
    },
    entry_points={
        "console_scripts": [
            "tidyllm=tidyllm.cli:main",
            "tidyllm-demo=tidyllm.demos.launch_demo:main",
            "tidyllm-workflow=tidyllm.workflows.cli:main",
            "qa-processor=qa_processor:main",
        ],
    },
    zip_safe=False,
    keywords=[
        "ai", "ml", "llm", "enterprise", "workflows", "automation", 
        "bedrock", "aws", "mlflow", "dspy", "knowledge-systems"
    ],
)