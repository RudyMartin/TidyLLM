#!/usr/bin/env python3
"""
TidyLLM - Core Business Logic Package
===================================
"""

from setuptools import setup, find_packages

# Read version from __init__.py
def get_version():
    try:
        with open('tidyllm/__init__.py', 'r') as f:
            for line in f:
                if line.startswith('__version__'):
                    return line.split('=')[1].strip().strip('"\'')
    except:
        pass
    return '2.0.0'

# Read README if available
def get_long_description():
    try:
        with open('README.md', 'r', encoding='utf-8') as f:
            return f.read()
    except:
        return "TidyLLM - Core business logic package for QA-Shipping 4-layer architecture"

setup(
    name="tidyllm",
    version=get_version(),
    author="TidyLLM Team",
    author_email="info@tidyllm.ai",
    description="Core TidyLLM business logic package",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/organization/qa-shipping",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pydantic>=2.0.0",
        "sqlalchemy>=2.0.0",
        "asyncio-mqtt",
        "aiofiles",
        "httpx",
        "pyyaml",
        "python-dotenv",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
        ],
    },
    entry_points={
        "console_scripts": [
            # Future CLI commands can go here
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="tidyllm, business logic, qa-shipping, architecture, clean",
    project_urls={
        "Bug Reports": "https://github.com/organization/qa-shipping/issues",
        "Source": "https://github.com/organization/qa-shipping",
        "Documentation": "https://github.com/organization/qa-shipping/wiki",
    },
)