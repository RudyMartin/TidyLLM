#!/usr/bin/env python3
"""
tidyllm-sentence: Educational sentence embeddings with complete algorithmic transparency
Pure Python implementation that competes with industrial systems
"""

from setuptools import setup, find_packages

# Read version from __init__.py if available
def get_version():
    try:
        with open('tidyllm_sentence/__init__.py', 'r') as f:
            for line in f:
                if line.startswith('__version__'):
                    return line.split('=')[1].strip().strip('"\'')
    except:
        pass
    return '0.1.1'

# Read README
def get_long_description():
    try:
        with open('README.md', 'r', encoding='utf-8') as f:
            return f.read()
    except:
        return "Educational sentence embeddings with complete algorithmic transparency"

setup(
    name="tidyllm-sentence",
    version=get_version(),
    author="TidyLLM Team",
    author_email="info@tidyllm.ai",
    description="Educational sentence embeddings that compete with industrial systems",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/tidyllm/tidyllm-sentence",
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
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Pure Python - no external dependencies!
        # Optional integration with tlm
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
        ],
        "integration": [
            "tlm>=1.0.0",  # For ML algorithms integration
        ],
    },
    entry_points={
        "console_scripts": [
            # Future CLI commands can go here
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="sentence embeddings, nlp, educational, transparent, tfidf, word2vec, pure python",
    project_urls={
        "Bug Reports": "https://github.com/tidyllm/tidyllm-sentence/issues",
        "Source": "https://github.com/tidyllm/tidyllm-sentence",
        "Documentation": "https://docs.tidyllm.ai/sentence",
    },
)