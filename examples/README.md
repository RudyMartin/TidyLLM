# TidyLLM Reasoning Examples - Tensor Logic Orchestration

This folder contains examples demonstrating the **TensorLogicService** - the orchestration layer that unifies symbolic and analogical reasoning with temperature control.

## What's New

The `packages/tidyllm/reasoning/` module provides **Tensor Logic** orchestration based on Pedro Domingos's framework for temperature-controlled reasoning.

### New Module: `reasoning/`

Components:
- `TensorLogicService` - Main orchestration service
- `TemperatureRouter` - Routes queries based on temperature
- `ReasoningMode` - Enum for reasoning modes (SYMBOLIC, HYBRID, ANALOGICAL)
- `create_reasoner()` - Convenience factory function

## Examples

### 01_temperature_routing.py

Demonstrates temperature-based routing between reasoning modes:
- Understanding the temperature router
- Symbolic reasoning (T≈0) - certifiable, rule-based
- Analogical reasoning (T≥0.5) - case-based, similarity
- Hybrid reasoning (0<T<0.5) - weighted combination
- Temperature sweep analysis
- Custom threshold configuration

**Run:**
```bash
cd examples
python 01_temperature_routing.py
```

**Key Concepts:**
- Temperature determines reasoning mode
- Modes: SYMBOLIC (T≤0.05), HYBRID (0.05<T<0.5), ANALOGICAL (T≥0.5)
- Hybrid mode uses weighted averaging

### 02_complete_workflow.py

End-to-end compliance checking workflow:
- Setting up knowledge base (rules + cases)
- Symbolic mode for formal verification
- Analogical mode for case-based reasoning
- Hybrid mode for balanced reasoning
- Temperature sweep for decision analysis
- Batch compliance checking
- Evidence and component examination
- Best practices guide

**Run:**
```bash
cd examples
python 02_complete_workflow.py
```

**Use Cases:**
- Compliance checking
- Regulatory analysis
- Risk assessment
- Policy interpretation
- Requirement validation

## Architecture

The TensorLogicService is the **top layer** of the TidyLLM Tensor Logic ecosystem:

```
┌─────────────────────────────────────────┐
│  tidyllm/reasoning/                     │
│  TensorLogicService (Orchestration)     │
│  - Temperature routing                  │
│  - Mode selection                       │
│  - Result combination                   │
└──────────────┬──────────────────────────┘
               │
    ┌──────────┴──────────┐
    │                     │
    ▼                     ▼
┌─────────┐       ┌──────────────┐
│ Symbolic│       │  Analogical  │
│ Engine  │       │   Engine     │
│ (Rules) │       │  (Cases)     │
└─────────┘       └──────┬───────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ tidyllm-sentence    │
              │ (Embeddings)        │
              └──────────┬──────────┘
                         │
                         ▼
                    ┌────────┐
                    │  tlm   │
                    │ (Math) │
                    └────────┘
```

## Temperature Guide

### When to Use Each Mode

**Symbolic (T = 0.0)**
```python
result = reasoner.infer("Is X required?", temperature=0.0)
# Certifiable: True
# Uses: Rule checking, formal verification
```

**Hybrid (T = 0.3)**
```python
result = reasoner.infer("How should X be done?", temperature=0.3)
# Certifiable: False
# Uses: Balanced reasoning, risk assessment
```

**Analogical (T = 0.7)**
```python
result = reasoner.infer("What approaches exist for X?", temperature=0.7)
# Certifiable: False
# Uses: Case finding, exploratory search
```

## API Reference

### Creating a Reasoner

```python
from reasoning import create_reasoner

# Basic reasoner
reasoner = create_reasoner()

# With symbolic rules
rules = [{"rule": "validation_required", "condition": "all_data"}]
reasoner = create_reasoner(rules=rules)

# With analogical cases
cases = ["Example 1", "Example 2", "Example 3"]
reasoner = create_reasoner(cases=cases)

# With both
reasoner = create_reasoner(rules=rules, cases=cases, embedding_method='lsa')
```

### Running Inference

```python
# Basic inference
result = reasoner.infer("Query?", temperature=0.0)

# With context
result = reasoner.infer("Query?", context={"domain": "compliance"}, temperature=0.3)

# With trustworthiness scoring (requires YRSN scorer)
result = reasoner.infer("Query?", temperature=0.0, score_trustworthiness=True)
```

### Result Structure

```python
{
    'answer': str,              # Inferred answer
    'confidence': float,        # 0-1 confidence score
    'reasoning_mode': str,      # 'symbolic', 'hybrid', or 'analogical'
    'certifiable': bool,        # True only for pure symbolic
    'trustworthiness': float,   # YRSN score (if requested)
    'evidence': list,           # Supporting evidence
    'components': dict          # Component details (for hybrid)
}
```

## Integration

This reasoning module is designed to integrate with the compliance-qa application:

1. **Symbolic Engine**: Can be connected to existing rule engines
2. **Analogical Engine**: Uses tidyllm-sentence for case retrieval
3. **YRSN Scorer**: Can be connected to trustworthiness validators

Currently, placeholder implementations are provided. Replace them with actual adapters from compliance-qa when ready.

## Performance Tips

1. **Pre-compute embeddings** for large case bases
2. **Choose appropriate temperature** for your use case
3. **Use temperature sweeps** for critical decisions
4. **Cache reasoner instances** with pre-loaded knowledge
5. **Batch similar queries** at the same temperature

## Best Practices

1. **Start symbolic** (T=0) for compliance checks
2. **Use hybrid** (T=0.3) for balanced reasoning
3. **Use analogical** (T=0.7) for exploration
4. **Run sweeps** when decision is critical
5. **Examine components** in hybrid mode to understand reasoning
6. **Check certifiable flag** when proof is needed

## Next Steps

- Integrate symbolic engine from compliance-qa
- Connect YRSN trustworthiness scoring
- Add domain-specific temperature presets
- Implement result caching
- Add batch processing utilities

See the main TidyLLM documentation for complete API reference.
