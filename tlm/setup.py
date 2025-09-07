#!/usr/bin/env python3
"""
TLM - Transparent Learning Machines
Pure Python machine learning algorithms with zero dependencies
"""

from setuptools import setup, find_packages

# Read version from __init__.py
def get_version():
    try:
        with open('tlm/__init__.py', 'r') as f:
            for line in f:
                if line.startswith('__version__'):
                    return line.split('=')[1].strip().strip('"\'')
    except:
        pass
    return '1.0.1'

# Read README if available
def get_long_description():
    try:
        with open('README.md', 'r', encoding='utf-8') as f:
            return f.read()
    except:
        return "TLM - Transparent Learning Machines: Pure Python ML algorithms with zero dependencies"

setup(
    name="tlm",
    version=get_version(),
    author="TidyLLM Team",
    author_email="info@tidyllm.ai",
    description="Pure Python machine learning algorithms with zero dependencies",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/tidyllm/tlm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Education",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Pure Python - no external dependencies!
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
    keywords="machine learning, algorithms, pure python, educational, transparent, ml, ai",
    project_urls={
        "Bug Reports": "https://github.com/tidyllm/tlm/issues",
        "Source": "https://github.com/tidyllm/tlm",
        "Documentation": "https://docs.tidyllm.ai/tlm",
    },
)