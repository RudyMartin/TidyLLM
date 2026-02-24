# YRSN API

## Quick Start

```python
from yrsn_tools import YRSN

y = YRSN("Your content here", query="optional query")

# Decomposition
y.R, y.S, y.N          # Components (0-1 each)

# Quality
y.quality              # α_ω (reliability-adjusted)
y.collapse_risk        # S + 1.5·N
y.temperature          # τ = 1/α_ω

# Actions
y.gate(0.7)            # True if quality >= 0.7
y.route()              # RoutingDecision (tier + temp)
y.explain()            # Print human-readable summary

# Different backends (all offline, no API calls)
y = YRSN(content, backend="tfidf", query="my query")
y = YRSN(content, backend="robust_pca", embed_fn=my_embedder)
```

---

## Backends

All backends are **fully offline** - no external API calls or model downloads. This ensures compatibility with corporate firewalls.

```python
from yrsn_tools import list_backends
print(list_backends())  # ['heuristic', 'tfidf', 'robust_pca', 'signal']
```

### Backend Comparison

| Backend | Best For | Dependencies | Speed | Accuracy |
|---------|----------|--------------|-------|----------|
| `heuristic` | General text (default) | None | ⚡ Fast | Good |
| `tfidf` | Query-content similarity | sklearn | ⚡ Fast | Better for queries |
| `robust_pca` | Numeric/matrix data | numpy | Medium | Best for embeddings |
| `signal` | Time series, signals | numpy | Medium | Best for sequences |
| `quantum` | Research, hybrid workflows | pennylane | Slow | Experimental |

### `heuristic` Backend (Default)

Rule-based decomposition using pattern matching. Zero dependencies.

**How it works:**
- **R (Relevant)**: Query term overlap + factual indicators + data presence
- **S (Superfluous)**: Filler words, excessive length
- **N (Noise)**: Off-topic content, spam indicators, too short

```python
y = YRSN("The mitochondria is the powerhouse of the cell.")
# Uses regex patterns to estimate R, S, N
```

**When to use:** General-purpose analysis, quick assessments, environments with no optional dependencies.

---

### `tfidf` Backend

TF-IDF vectorization with cosine similarity scoring.

**How it works:**
- Builds TF-IDF vectors for content, query, and context
- **R**: Cosine similarity between content and query/context
- **S**: Length-based verbosity estimate
- **N**: Residual (1 - R - S)

```python
y = YRSN(
    "Python uses indentation for blocks.",
    query="How does Python structure code?",
    backend="tfidf"
)
```

**When to use:** When you have a query and want better relevance scoring than heuristics.

**Requires:** `scikit-learn` (falls back to heuristic if not installed)

---

### `robust_pca` Backend

Robust Principal Component Analysis for matrix decomposition.

**How it works:**
- Decomposes matrix M into low-rank (L) + sparse (S) components
- **R**: Energy in low-rank component (systematic signal)
- **S**: Energy in sparse component (outliers, anomalies)
- **N**: Residual reconstruction error

```python
import numpy as np

# With custom embedding function
def embed(text):
    # Your embedding logic (e.g., sentence transformers)
    vec = my_encoder.encode(text)
    return vec.reshape(32, 32)  # Must be 2D matrix

y = YRSN(content, backend="robust_pca", embed_fn=embed)

# Or with raw matrix
matrix = np.random.randn(100, 50)
y = YRSN("", backend="robust_pca", embed_fn=lambda _: matrix)
```

**When to use:**
- Analyzing document embeddings
- Finding anomalies in structured data
- When you have numeric representations

**Requires:** `numpy`, `yrsn-context` (falls back to heuristic if not available)

**Algorithm options:**
```python
from yrsn_tools import RobustPCABackend, register_backend

# Use different RPCA algorithm
backend = RobustPCABackend(method="godec")  # 'ialm', 'pcp', or 'godec'
register_backend("rpca_godec", backend)
y = YRSN(content, backend="rpca_godec", embed_fn=embed)
```

---

### `signal` Backend

Fast Iterative Filtering (FIF) for signal decomposition.

**How it works:**
- Decomposes signal into Intrinsic Mode Functions (IMFs)
- **R**: Variance in low-frequency trend (last IMF)
- **S**: Variance in mid-frequency components (middle IMFs)
- **N**: Variance in high-frequency oscillations (first IMF)

```python
import numpy as np

# Time series data
time_series = np.sin(np.linspace(0, 10*np.pi, 1000)) + np.random.randn(1000)*0.1

y = YRSN("", backend="signal", signal=time_series)
print(f"Signal quality: R={y.R:.2f}, S={y.S:.2f}, N={y.N:.2f}")
```

**When to use:**
- Analyzing time series quality
- Audio/sensor signal decomposition
- Sequential data with frequency components

**Requires:** `numpy`, `yrsn-context` (falls back to heuristic if not available)

---

### `quantum` Backend

Quantum kernel-based decomposition using `YRSNQuantumKernel` from yrsn-context.

Uses PennyLane for quantum circuit simulation. All computation runs **locally** - no cloud QPU required.

**How it works:**
- Maps content to quantum feature space via parameterized circuits
- Computes quantum kernel values (state overlap) for similarity
- **R**: Kernel similarity to query/context (relevance)
- **S**: Self-coherence minus reference match (coherent but off-topic)
- **N**: Incoherence from low self-kernel values (noise)

```python
# Basic usage (uses angle embedding, 8 qubits)
y = YRSN("Content here", backend="quantum")

# With query for relevance comparison
y = YRSN(
    "Python uses indentation for blocks.",
    query="How does Python structure code?",
    backend="quantum"
)

# With pre-computed feature vectors
import numpy as np
features = np.random.rand(8)  # 8 features for 8 qubits
y = YRSN("", backend="quantum", features=features)
```

**Feature maps:**

| Feature Map | Description | Best For |
|-------------|-------------|----------|
| `angle` | AngleEmbedding (RY rotations) - default | Fast, simple experiments |
| `zz` | ZZFeatureMap with CNOT entanglement | Capturing correlations |

```python
from yrsn_tools import QuantumBackend, register_backend

# ZZ feature map (more expressive, entangling layers)
backend = QuantumBackend(feature_map="zz", n_qubits=8, reps=3)
register_backend("quantum_zz", backend)
y = YRSN(content, backend="quantum_zz")

# Angle embedding (simpler, faster)
backend = QuantumBackend(feature_map="angle", n_qubits=8)
register_backend("quantum_angle", backend)
y = YRSN(content, backend="quantum_angle")
```

**YRSN ↔ Quantum mapping:**

| YRSN Concept | Quantum Analog |
|--------------|----------------|
| R (Relevant) | High kernel similarity to reference |
| S (Superfluous) | High self-coherence, low reference match |
| N (Noise) | Low self-kernel (incoherent state) |

**Installation (all optional):**

```bash
# Basic (CPU, pure Python - slowest but no compilation)
pip install pennylane

# Fast CPU (C++ backend - recommended)
pip install pennylane pennylane-lightning

# GPU (NVIDIA CUDA - fastest for large circuits)
pip install pennylane pennylane-lightning-gpu

# TPU (via PennyLane JAX plugin)
pip install pennylane pennylane-jax jax[tpu]
```

**Device selection:**

The `device` parameter accepts any valid [PennyLane device string](https://docs.pennylane.ai/en/stable/introduction/circuits.html#supported-devices). Common options:

| Device | Hardware | Install |
|--------|----------|---------|
| `default.qubit` | CPU (pure Python) | Base pennylane |
| `default.qubit.jax` | CPU/TPU (JAX) | `pennylane-jax` |
| `lightning.qubit` | CPU (fast C++) | `pennylane-lightning` |
| `lightning.gpu` | NVIDIA GPU | `pennylane-lightning-gpu` |
| `lightning.kokkos` | Multi-backend | `pennylane-lightning-kokkos` |
| `qiskit.aer` | IBM Qiskit | `pennylane-qiskit` |

```python
from yrsn_tools import QuantumBackend, register_backend

# CPU (default, no extra install needed)
backend = QuantumBackend(device="default.qubit")

# Fast CPU (requires pennylane-lightning)
backend = QuantumBackend(device="lightning.qubit")

# GPU (requires pennylane-lightning-gpu + CUDA)
backend = QuantumBackend(device="lightning.gpu")

# JAX for TPU/GPU via JAX
backend = QuantumBackend(device="default.qubit.jax")

# Kokkos for flexible backend (OpenMP, CUDA, HIP, SYCL)
backend = QuantumBackend(device="lightning.kokkos")

register_backend("quantum_fast", backend)
y = YRSN(content, backend="quantum_fast")
```

**Falls back to heuristic** if pennylane or yrsn-context quantum module is not installed.

**For full quantum ML workflows** (QSVM, VQC), use yrsn-context directly:
```python
from yrsn_context.models.quantum_qsvm import YRSNQSVM
from yrsn_context.quantum.kernels import YRSNQuantumKernel
```

---

### Custom Backends

Create your own backend by implementing the `YRSNBackend` protocol:

```python
from yrsn_tools import register_backend, YRSNBackend

class MyDomainBackend:
    """Custom backend for domain-specific decomposition."""

    name = "my_domain"

    def decompose(self, content, query="", context="", **kwargs):
        """
        Decompose content into (R, S, N) ratios.

        Args:
            content: Text to analyze
            query: Optional query for relevance
            context: Optional grounding context
            **kwargs: Your custom parameters

        Returns:
            Tuple of (R, S, N) where R + S + N ≈ 1.0
        """
        # Your decomposition logic
        R = self._compute_relevance(content, query)
        S = self._compute_superfluous(content)
        N = self._compute_noise(content)

        # Normalize
        total = R + S + N
        return (R/total, S/total, N/total)

    def _compute_relevance(self, content, query):
        # Domain-specific relevance logic
        return 0.6

    def _compute_superfluous(self, content):
        return 0.2

    def _compute_noise(self, content):
        return 0.2

# Register and use
register_backend("my_domain", MyDomainBackend())
y = YRSN(content, backend="my_domain")
```

---

### Backend Selection Guide

```
┌─────────────────────────────────────────────────────────────┐
│                    What's your input?                        │
└─────────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
       Plain Text      Embeddings/      Time Series
                       Matrices
            │               │               │
            ▼               ▼               ▼
    ┌───────────────┐ ┌───────────┐ ┌───────────────┐
    │ Have a query? │ │robust_pca │ │    signal     │
    └───────────────┘ └───────────┘ └───────────────┘
      │         │
      Yes       No
      │         │
      ▼         ▼
   tfidf    heuristic
```

---

## Theory

### Y = R + S + N

| Component | Meaning | Goal |
|-----------|---------|------|
| **R** | Relevant signal | Maximize |
| **S** | Superfluous | Accept |
| **N** | Noise | Eliminate |

### Metrics

| Metric | Formula | Access |
|--------|---------|--------|
| Y_Score | `R + 0.5·S` | `.y_score` |
| α | `R / (R + S + N)` | `.alpha` |
| ω | OOD reliability | `.omega` |
| α_ω | `ω·α + (1-ω)·prior` | `.quality` |
| τ | `1 / α_ω` | `.temperature` |
| Risk | `S + 1.5·N` | `.collapse_risk` |
| c(n;K) | `n / (n + K)` | `.credibility` |

### Collapse Types

| Type | Condition |
|------|-----------|
| POISONING | High N |
| DISTRACTION | High S |
| CONFUSION | High S + N |
| CLASH | Variable S |

---

## YRSN Class Reference

### Constructor

```python
YRSN(content, query="", context="", omega=1.0, n=None, K=10, backend="heuristic", **backend_kwargs)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `content` | str | Text to analyze |
| `query` | str | Optional query for relevance |
| `context` | str | Optional grounding context |
| `omega` | float | OOD reliability (0-1), default 1.0 |
| `n` | float | Sample size for credibility (optional) |
| `K` | float | Half-saturation for credibility, default 10 |
| `backend` | str | Decomposition algorithm: `heuristic`, `tfidf`, `robust_pca`, `signal` |
| `**backend_kwargs` | dict | Backend-specific params (e.g., `embed_fn` for robust_pca) |

### Properties

```python
y = YRSN(content)

# Backend
y.backend              # Name of decomposition backend

# Decomposition
y.R                    # Relevant ratio (0-1)
y.S                    # Superfluous ratio (0-1)
y.N                    # Noise ratio (0-1)

# Quality metrics
y.alpha                # α = R / (R + S + N)
y.omega                # ω (input OOD reliability)
y.quality              # α_ω = ω·α + (1-ω)·prior
y.y_score              # R + 0.5·S
y.temperature          # τ = 1/α_ω
y.collapse_risk        # S + 1.5·N

# Credibility (if n provided)
y.credibility          # c(n;K) = n/(n+K)
y.effective_omega      # ω_OOD · c(n;K)

# Collapse detection
y.collapse             # CollapseResult
y.is_collapsed         # bool
```

### Methods

```python
# Gate by quality threshold
y.gate(threshold=0.5)  # → bool

# Get routing decision
y.route()              # → RoutingDecision
                       #   .tier (FAST/BALANCED/QUALITY)
                       #   .temperature

# Print human-readable summary
y.explain()            # prints and returns str
y.explain(verbose=True)  # includes formulas
```

---

## Examples

### Basic Usage

```python
from yrsn_tools import YRSN

y = YRSN("The mitochondria is the powerhouse of the cell.")
print(f"Quality: {y.quality:.2f}")
# Quality: 0.65

if y.gate(0.5):
    print("Safe to use")
```

### With Query

```python
y = YRSN(
    "Python uses indentation for code blocks.",
    query="How does Python handle code structure?"
)
y.explain()
```

### With Credibility

```python
# When you have sample size information
y = YRSN(content, omega=0.9, n=50, K=10)
print(f"Credibility: {y.credibility:.2f}")
print(f"Effective ω: {y.effective_omega:.2f}")
```

### Collapse Detection

```python
y = YRSN(noisy_content)
if y.is_collapsed:
    print(f"⚠ {y.collapse.collapse_type.name}")
    print(f"  {y.collapse.recommendation}")
```

### Routing

```python
y = YRSN(context, query=user_query)
decision = y.route()
print(f"Use {decision.tier.value} model at temp={decision.temperature:.2f}")
```

---

## Package Structure

| Package | Purpose |
|---------|---------|
| **yrsn-tools** | User-facing API (this package) |
| **yrsn-context** | Core algorithms (research) |

Backend algorithms from yrsn-context are wrapped as pluggable backends:
- `robust_pca` → `yrsn_context.core.decomposition.robust_pca`
- `signal` → `yrsn_context.core.filtering.fif`

---

## References

- Bühlmann, H. (1967). "Experience Rating and Credibility" - credibility formula
- See `docs/theory/YRSN_definition.md` for full theory
