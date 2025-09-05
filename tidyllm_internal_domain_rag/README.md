# TidyLLM Internal Domain RAG

Self-referential domain RAG for resolving conflicts in TidyLLM's own documentation.

## Purpose
- Resolve conflicts between different documentation sources
- Establish precedence hierarchy for architectural decisions
- Provide authoritative answers about TidyLLM implementation

## Hierarchy (Precedence Order)
1. **CRITICAL_DECISIONS** (1.0) - Critical design decisions and constraints
2. **ARCHITECTURE** (0.9) - System architecture and integration docs
3. **CURRENT** (0.8) - Latest documentation (2025-09-05)
4. **RECENT** (0.7) - Recent documentation (2025-09-04, 2025-09-03)
5. **HISTORICAL** (0.6) - Historical documentation (2025-09-01)
6. **EXAMPLES** (0.5) - Examples and templates

## Usage
```bash
python demo.py
```

## Naming Convention
- **external_domain_rag**: Model validation compliance (knowledge_base/)
- **internal_domain_rag**: TidyLLM self-documentation (docs/)
- **tidyllm_self_rag**: This system (conflict resolution)

Generated: 2025-09-05T12:30:37.494573
Total Documents: 12
