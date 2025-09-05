# Domain RAG System

Built: 2025-09-05 11:01:11.509325
Total Documents: 25

## Folder Structure:
- checklist/: 3 files (Authoritative - highest precedence)
- sop/: 14 files (Standard procedures)  
- modeling/: 8 files (Technical guidance)

## Usage:

```bash
cd domain_rag_system
python demo.py
```

## Architecture:

The system implements hierarchical precedence:
1. Checklist (Authoritative) - Regulatory requirements
2. SOP (Standard) - Operating procedures  
3. Modeling (Technical) - Methods and algorithms

Queries are processed with precedence ranking - authoritative sources returned first.
