# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
TLM is a foundational component of the **tidyllm verse** - a paradigm shift toward simplicity in data science and machine learning infrastructure. 

### The tidyllm Philosophy
- **Simplicity as Strategy**: Revealing essential core concepts rather than hiding complexity
- **ML Infrastructure Sovereignty**: Complete independence from Big Tech ML frameworks
- **Transparency-First**: Every operation is readable, traceable, and modifiable
- **Data Liberation**: Maximum portability with zero vendor lock-in
- **Composable Architecture**: Simple components that work together predictably

TLM demonstrates that sophisticated machine learning doesn't require complex tooling. By implementing algorithms in pure Python with only standard library dependencies, it proves viable alternatives to the dominant NumPy/TensorFlow/PyTorch ecosystem exist.

### Strategic Positioning
TLM isn't just educational - it's **paradigm-shifting architecture** that:
- Enables complete user ownership of ML pipelines
- Demonstrates feasible paths away from corporate ML infrastructure
- Creates foundation for truly open, transparent ML ecosystems
- Shows how simplicity enables rather than constrains innovation

This positions TLM as part of a movement toward **democratized, sovereign ML infrastructure** where users maintain complete control and understanding of their tools.

## Development Commands

### Testing
- Run all tests: `python -m pytest tlm/tests/ -v`
- Run specific test file: `python -m pytest tlm/tests/test_algos.py -v`
- Run single test function: `python -m pytest tlm/tests/test_algos.py::test_softmax_row_sums -v`

### Installation
- Install in editable mode: `pip install -e .`

### Package Structure
- No build/lint commands - this is a simple pure Python package
- Tests use standard pytest framework
- No external dependencies beyond Python standard library

## Architecture

### Module Organization
The codebase follows a scikit-learn-inspired modular structure:

```
tlm/
├── __init__.py          # Main API exports, imports from all modules
├── pure/ops.py          # Core list-based operations (array, dot, transpose, etc.)
├── core/                # Fundamental ML components
│   ├── activations.py   # sigmoid, relu, softmax
│   ├── losses.py        # mse, cross_entropy, etc.
│   └── metrics.py       # accuracy, confusion_matrix
├── linear_models/       # Linear classification/regression
├── cluster/             # K-means clustering
├── svm/                 # Linear SVM implementation
├── decomp/              # PCA with power iteration, whitening
├── mixture/             # Gaussian Mixture Models (EM algorithm)
├── anomaly/             # Gaussian anomaly detection
├── naive_bayes/         # Multinomial Naive Bayes
├── mf/                  # Matrix factorization
├── model_selection/     # K-fold cross validation
└── tests/               # Test suite
```

### Key Design Principles (tidyllm verse)
- **Pure Python**: No external dependencies, uses only standard library - enables true portability
- **List-based operations**: All data structures are Python lists (1D/2D) - maximum interoperability
- **Transparency over Performance**: Every algorithm step is readable and modifiable
- **Functional API**: Each module exports fit/predict/transform functions - composable design
- **Consistent naming**: Functions follow pattern `module_operation` (e.g., `kmeans_fit`, `svm_predict`)
- **Zero Vendor Lock-in**: Can interface with any data source (pandas, databases, APIs) via simple list conversion
- **Algorithmic Sovereignty**: Users own and understand their entire ML pipeline

### API Conventions
- All algorithms follow fit/predict pattern where applicable
- Data format: lists of lists for 2D data, lists for 1D
- Functions are imported and re-exported through `tlm/__init__.py`
- Return values typically include fitted parameters and training history
- Seeds are used for reproducibility in stochastic algorithms

### Testing Strategy
- Simple assertion-based tests in `tlm/tests/`
- Tests focus on basic functionality and shape validation
- No complex test fixtures or mocking - keeps tests readable
- Each test file covers related algorithm groups