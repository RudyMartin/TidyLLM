# CLAUDE.md - tidyllm-compliance

## Repository Purpose
**tidyllm-compliance** provides automated regulatory compliance monitoring with complete algorithmic transparency. This is part of the **tidyllm-verse** ecosystem of educational ML tools.

## Core Functionality
- **Model Risk Standards**: Federal Reserve SR 11-7, OCC guidance compliance checking
- **Evidence Validation**: Document authenticity and completeness assessment
- **Argument Consistency**: Review scope determination based on logical analysis
- **Research Papers**: Whitepaper and academic document analysis for regulatory compliance

## Architecture
```
tidyllm_compliance/
├── model_risk/       # Model risk development standards
├── evidence/         # Evidence validation and authenticity 
├── consistency/      # Argument consistency analysis
├── research_papers/  # Whitepaper and academic document analysis
└── __init__.py       # Main API exports
```

## Key Dependencies
- **tidyllm-sentence**: Text embeddings and similarity (external package)
- **tlm**: Core ML algorithms (external package)
- **Standard library only** for core functionality

## Philosophy
- **Educational Transparency**: Every algorithm step is visible and auditable
- **Pure Python**: No external ML dependencies beyond tidyllm-verse
- **Regulatory Ready**: Designed for financial services compliance
- **Deterministic**: Same input always produces same output

## Common Commands
```bash
# Install for development
pip install -e .

# Run model risk demo
python examples/model_risk_demo.py

# Run evidence validation demo  
python examples/evidence_validation_demo.py

# Run consistency analysis demo
python examples/consistency_analysis_demo.py

# Run all tests
pytest tests/

# Build package
python -m build
```

## Integration with tidyllm-verse
This package works seamlessly with:
- **tlm**: Core ML algorithms for classification and analysis
- **tidyllm-sentence**: Text embeddings for semantic similarity
- **tidyllm-documents**: Document processing and extraction

## Use Cases
1. **Financial Services**: SR 11-7 model risk compliance
2. **Audit & Investigation**: Evidence authenticity validation  
3. **Legal Review**: Argument consistency and scope determination
4. **Regulatory Reporting**: Automated compliance monitoring
5. **Research Analysis**: Whitepaper analysis for regulatory compliance backing

## Academic Validation
- Rule-based precision with configurable thresholds
- Complete audit trail for regulatory review
- Explainable decisions with detailed scoring
- Benchmarked against manual compliance processes

## Educational Features
- Complete source code transparency
- Detailed algorithm explanations
- Regulatory framework implementations
- Production-ready examples and demonstrations

This repository represents **algorithmic sovereignty** for compliance workflows - you own and understand every decision rule.