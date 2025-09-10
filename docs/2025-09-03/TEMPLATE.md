# Design Decision Template

Use this template for documenting new design decisions:

## Decision ID: DD-XXX

**Date**: YYYY-MM-DD  
**Context**: Describe the problem/question that triggered this decision

### Decision

Clear statement of what was decided.

### Rationale

Why this decision was made:
1. **Reason 1**: Explanation
2. **Reason 2**: Explanation  
3. **Reason 3**: Explanation

### Consequences

**For Tests**:
- Impact on test implementation
- What tests need to account for

**For Implementation**:
- Impact on code structure
- What developers need to know

### Examples

```python
# Code examples showing the correct pattern
example_usage = "demonstrating the decision"
```

```python
# Anti-patterns (what NOT to do)
wrong_usage = "showing incorrect approach"  # WRONG
```

## Decision Numbering

- DD-001 to DD-099: Core architecture decisions
- DD-100 to DD-199: API and interface decisions  
- DD-200 to DD-299: Testing strategy decisions
- DD-300 to DD-399: Data format and structure decisions
- DD-400 to DD-499: Provider and integration decisions