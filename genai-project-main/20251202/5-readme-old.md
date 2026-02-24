# YRSN Context: Adaptive Context Engineering Framework

A Python library for **context quality analysis** - diagnosing why context fails and how to fix it.

## What is YRSN?

YRSN is a framework for analyzing context quality. It separates context into components:

| Component | Meaning | Goal |
|-----------|---------|------|
| **Y** (Yield) | What you get - total output | Measure it |
| **R** (Relevant) | What you need - useful signal | Maximize it |
| **S** (Superfluous) | What you tolerate - redundant but harmless | Accept it |
| **N** (Noise) | What hurts you - harmful interference | Eliminate it |

## What Makes YRSN Different?

Most libraries optimize for accuracy. YRSN asks a different question: **"Did we learn the signal or the bias?"**

### The Problem YRSN Solves

When context quality degrades, models fail in predictable ways:

| Collapse Type | Condition | Effect |
|--------------|-----------|--------|
| **Poisoning** | High N | Model hallucinates from bad data |
| **Distraction** | High S | Model loses focus on what matters |
| **Confusion** | High S + N | Model can't distinguish signal from noise |
| **Clash** | Variable S | Sources conflict with each other |

YRSN detects these failures and maps each to a specific remedy.

### The Core Innovation: YRSN-Guided Attention

Neural models can't distinguish signal from noise on their own. They attend uniformly - a board state with magnitude 10.0 drowns out rules with magnitude 2.0. The model learns bias, not logic.

**The YRSN framework solves this by controlling the neural model:**

```
YRSN Classification         Attention Guidance
───────────────────         ──────────────────
R (Relevant)        →       Boost (rules, constraints)
S (Superfluous)     →       Baseline (board state, context)
N (Noise)           →       Filter out (contradictions, errors)
```

Attention weights are derived from YRSN decomposition (not hardcoded):

```python
from yrsn_context.applications.sudoku import YRSNIntegratedEncoder

encoder = YRSNIntegratedEncoder(context_dim=64)
result = encoder.encode(rules, visual, board, 9, 9, task)

# Weights computed from L/S/N norms - not arbitrary multipliers
print(result['yrsn_weights'])
# {'rules_weight': 0.70, 'board_weight': 0.30, 'noise_weight': 0.0}
```

**Know what matters. Know when to stop.**

YRSN controls both:
- **What to attend to**: R/S/N classification → attention boosting
- **When to stop thinking**: Observe CTM certainty + correctness → adjust computation depth

The self-calibration loop connects them: YRSN observes what CTM actually attended to, whether it was correct, and adjusts boost factors and precision thresholds until the model attends to R and stops at the right time.

**Models don't know what matters or when they're done. YRSN tells them.**

### What's Genuinely Novel

- **YRSN-Guided Attention**: Use R/S/N classification to steer where models look
- **Collapse Detection**: Classifies context failures into types with remediation guidance
- **Self-Calibration**: Automatically adjusts boost factors based on attention patterns
- **Strategy Mapping**: Each collapse type maps to a specific fix

### What's Established Techniques (Honestly)

- **Robust PCA**: Standard algorithm (Candès 2011) - one technique for R/S/N separation
- **Adaptive Computation**: CTM, PonderNet, ACT - prior art we build on
- **Retrieval Strategies**: Common patterns reframed around context quality

## Installation

```bash
pip install yrsn-context              # Core (NumPy only)
pip install yrsn-context[neural]      # + PyTorch models
pip install yrsn-context[quantum]     # + PennyLane kernels
pip install yrsn-context[full]        # Everything
```

## Quick Start

### 1. Detect Context Collapse

```python
from yrsn_context import detect_collapse

# Check if context quality is degraded
analysis = detect_collapse(R=0.3, S=0.2, N=0.5)

print(analysis.collapse_type)    # CollapseType.POISONING
print(analysis.severity)         # Severity.HIGH
print(analysis.recommendation)   # "Reduce noise sources..."
print(analysis.paradigm_remedy)  # "Paradigm 1: Iterative Refinement..."
```

### 2. Decompose Data into YRSN Components

```python
from yrsn_context import decompose_to_yrsn
import numpy as np

# Decompose a matrix into low-rank (R), sparse (S), noise (N)
M = np.random.randn(50, 40)
R, S, N = decompose_to_yrsn(M)

print(f"Relevant: {R:.2f}")
print(f"Superfluous: {S:.2f}")
print(f"Noise: {N:.2f}")
```

### 3. Load Sample Data (sklearn-style)

```python
from yrsn_context import load_sample_signals, load_yrsn_examples

# Pre-built signals with known R/S/N ground truth
signals = load_sample_signals()
noisy = signals['noisy_sine']
print(f"True R={noisy.true_R}, S={noisy.true_S}, N={noisy.true_N}")

# Pre-built collapse examples
examples = load_yrsn_examples()
print(examples['poisoning']['description'])
```

## Context Engineering Strategies

Five patterns for managing context quality:

### Iterative Refinement
Best for: **Poisoning** (high noise)

Coarse retrieval → analyze gaps → precise refinement → repeat

```python
from yrsn_context import IterativeContextEngine
engine = IterativeContextEngine(coarse, fine, analyzer, max_iterations=5)
result = engine.retrieve("your query")
```

### Bit-Slicing
Best for: **Distraction** (information overload)

Organize context by importance level (core → detail → background)

```python
from yrsn_context import SlicedContext
sliced = SlicedContext()
sliced.add_slice(level=0, content="Core fact", source="doc1")
sliced.add_slice(level=1, content="Supporting detail", source="doc1")
```

### Hierarchical Decomposition
Best for: **Confusion** (contradictory signals)

Break large documents into processable blocks with dependencies

```python
from yrsn_context import HierarchicalContextDecomposer
decomposer = HierarchicalContextDecomposer(base_block_size=1000)
blocks = decomposer.decompose(large_document, query)
```

### Layered Architecture
Best for: **Clash** (source conflicts)

Stack context layers where foundation quality propagates upward

```python
from yrsn_context import LayeredContextStack
stack = LayeredContextStack(max_layers=6)
stack.add_layer(level=0, name="foundation", content=[...], quality_score=0.95)
```

### Self-Calibration
Best for: **Continuous optimization** (long-running systems)

Learn optimal parameters from execution traces - compare what YRSN predicted vs. what actually worked

```python
from yrsn_context import YRSNSelfCalibrator

calibrator = YRSNSelfCalibrator(learning_rate=0.1)

# Record execution traces over time
calibrator.record_trace(trace)

# Calibrate based on false positive/negative rates
result = calibrator.calibrate(current_weights, current_boost)
# Adjusts bit-slice weights, instruction boost automatically
```

## Applications

### Sudoku: A Proving Ground for Truth Algorithms

Why Sudoku? Because it makes the distinction between **learning truth** vs. **learning bias** objectively measurable.

**The Problem with Accuracy**

Two models can both achieve 90% accuracy on Sudoku:
- Model A learned the rules: "if 5 is in row 1, it can't appear elsewhere in row 1"
- Model B learned the bias: "position 65 is usually empty in training data"

They look identical by accuracy metrics. But Model A generalizes to new puzzles while Model B fails. YRSN makes this distinction explicit.

**What YRSN Measures in Sudoku**

| Component | Sudoku Meaning | How to Detect |
|-----------|---------------|---------------|
| **R** (Relevant) | Valid constraint logic | Predictions follow Sudoku rules |
| **S** (Superfluous) | Dataset patterns | Predictions correlate with training distribution |
| **N** (Noise) | Invalid deductions | Contradictory or impossible moves |

**Why This Matters for AI Truth**

The Sudoku application lets you:
1. **Compare approaches fairly** - Classical solver vs. neural vs. hybrid on equal ground
2. **Measure prediction diversity** - Did the model give varied answers or always guess the same thing?
3. **Detect bias exploitation** - High accuracy + low diversity = learned bias, not rules
4. **Test generalization** - Train on easy puzzles, test on hard ones

```python
from yrsn_context.applications.sudoku import (
    SudokuBoard,
    ClassicalSudokuSolver,
    HybridSudokuSolver,
    YRSNSudokuReasoner,
    # YRSN-based encoder (weights from decomposition, not hardcoded)
    YRSNIntegratedEncoder,
    compute_yrsn_weights,
)

# Compare approaches
classical = ClassicalSudokuSolver()
hybrid = HybridSudokuSolver()
reasoner = YRSNSudokuReasoner()

# Measure not just accuracy, but *how* it got there
```

This is the core YRSN thesis: **accuracy without understanding is fragile**. Sudoku provides a controlled environment where "understanding" (learning R) vs. "memorization" (learning S) has clear, testable meaning.

**Why This Generalizes to Creative Thinking**

Sudoku is a stepping stone, not the destination. The same R/S/N distinction applies to any domain where we want models to reason rather than pattern-match:

| Domain | R (Truth) | S (Bias) |
|--------|-----------|----------|
| Sudoku | Constraint logic | Position patterns in training data |
| Code Generation | Language semantics | Copy-paste from similar examples |
| Scientific Reasoning | Causal mechanisms | Correlation in literature |
| Creative Writing | Narrative coherence | Stylistic clichés |

A model that truly understands constraints can solve *novel* Sudoku puzzles, write *original* code, propose *new* hypotheses, or create *fresh* narratives. A model that learned bias can only recombine what it's seen.

YRSN's diagnostic framework - detecting when a model relies on S instead of R - is the first step toward building systems that genuinely reason rather than cleverly interpolate.

## Command Line Interface

YRSN includes a CLI for quick analysis:

```bash
# Analyze a text file for context quality
yrsn analyze document.txt

# Decompose a matrix using Robust PCA
yrsn decompose data.csv --method ialm

# Detect collapse from R/S/N ratios
yrsn detect --r 0.3 --s 0.4 --n 0.3

# Show package info
yrsn info
```

## Visualization

Visualize YRSN decomposition results (requires matplotlib):

```python
from yrsn_context.viz import (
    plot_yrsn_decomposition,  # Stacked bar + quality meter
    plot_yrsn_pie,            # Pie chart with interpretation
    plot_collapse_analysis,   # Collapse type visualization
    plot_rpca_components,     # L/S/N matrix heatmaps
    plot_yrsn_comparison,     # Compare multiple sources
)

# Plot decomposition results
R, S, N = decompose_to_yrsn(M)
plot_yrsn_decomposition(R, S, N, title="Document Analysis")

# Visualize collapse analysis
analysis = detect_collapse(R=0.3, S=0.4, N=0.3)
plot_collapse_analysis(analysis)
```

## Architecture

```
src/yrsn_context/
├── core/                    # Core YRSN framework
│   ├── decomposition/       # Robust PCA, collapse detection
│   ├── filtering/           # Fast Iterative Filtering (FIF) for signals
│   ├── optimization/        # Pluggable backends (Gradient, EGGROLL, Adaptive)
│   └── reservoir/           # Echo State Networks for temporal decomposition
├── strategies/              # Five paradigms:
│   ├── iterative_refinement/
│   ├── bit_slicing/
│   ├── hierarchical_decomposition/
│   ├── layered_stack/
│   └── self_calibration/
├── applications/            # Proving grounds
│   └── sudoku/              # Constraint satisfaction testbed
├── models/                  # Complete end-to-end models
│   ├── hrm.py               # Hierarchical Reasoning Model
│   ├── quantum.py           # Quantum-enhanced models
│   └── quantum_qsvm.py      # Quantum SVM for classification
├── datasets/                # sklearn-style data loaders
├── benchmarks/              # Evaluation utilities
├── quality/                 # Consistency checking
├── viz/                     # Visualization (matplotlib)
├── cli.py                   # Command-line interface
├── neural/                  # Optional: PyTorch CTM models
└── quantum/                 # Optional: PennyLane kernels
```

## Examples

See `examples/` for working code:

- `hierarchical_decomposition.py` - Large document processing
- `layered_stack_demo.py` - Layered architecture demo

## Advanced Features

YRSN includes powerful optional components for specialized use cases.

### Signal Decomposition Methods

Three complementary approaches for Y=R+S+N separation:

| Method | Input Type | Best For | Module |
|--------|------------|----------|--------|
| **Robust PCA** | Matrices | Document embeddings, static data | `core.decomposition` |
| **Fast Iterative Filtering** | 1D/2D signals | Audio, images, oscillatory data | `core.filtering` |
| **Reservoir Computing** | Time series | Streaming data, temporal patterns | `core.reservoir` |

```python
# Robust PCA (matrices)
from yrsn_context import decompose_to_yrsn
R, S, N = decompose_to_yrsn(matrix)

# Fast Iterative Filtering (signals)
from yrsn_context.core.filtering import fif_decompose, classify_imfs_to_yrsn
imfs = fif_decompose(signal)
result = classify_imfs_to_yrsn(imfs)

# Echo State Networks (time series)
from yrsn_context.core.reservoir import temporal_decompose
result = temporal_decompose(time_series)
```

### Optimization Backends

Pluggable optimization strategies - the framework auto-selects based on task:

| Backend | Type | When Used |
|---------|------|-----------|
| `GradientBackend` | Gradient-based | Neural networks, smooth objectives |
| `EggrollBackend` | Evolution-guided | Discrete decisions, non-differentiable |
| `AdaptiveBackend` | Auto-select | Default - picks optimal backend |

```python
from yrsn_context.core.optimization import AdaptiveBackend, TaskProfile, Paradigm

profile = TaskProfile(paradigm=Paradigm.BIT_SLICING)
backend = AdaptiveBackend(profile)  # Auto-selects EGGROLL for discrete decisions
```

### Neural Components (requires PyTorch)

CTM-inspired models with adaptive computation:

| Component | Purpose |
|-----------|---------|
| `CTMContextRetriever` | Main neural retrieval model |
| `NeuronLevelModel` | Private-weight temporal neurons |
| `NeuralSynchronization` | Multi-signal fusion |
| `CertaintyBasedLoss` | Uncertainty-aware training |
| `IterativeContextRefinement` | HP-INV style iterative improvement |
| `SnApTracker` | Efficient gradient tracking |

```python
from yrsn_context.neural import CTMContextRetriever, CertaintyBasedLoss

model = CTMContextRetriever(input_dim=64, hidden_dim=128, output_dim=64)
criterion = CertaintyBasedLoss(base_loss='mse', certainty_weight=0.1)
```

See [docs/neural/GETTING-STARTED.md](docs/neural/GETTING-STARTED.md) for full guide.

### Quantum Components (requires PennyLane)

Quantum kernel methods for context similarity:

| Component | Purpose |
|-----------|---------|
| `ZZFeatureMap` | Entangling feature map |
| `AngleEmbeddingFeatureMap` | Rotation-based encoding |
| `YRSNQuantumKernel` | YRSN-aware quantum kernel |
| `SelfCalibratingVQC` | Auto-tuning variational circuits |

```python
from yrsn_context.quantum import create_yrsn_quantum_kernel
from sklearn.svm import SVC

kernel = create_yrsn_quantum_kernel(n_qubits=4, feature_map='zz')
clf = SVC(kernel=kernel.sklearn_kernel)
```

See [docs/quantum/KERNEL-USAGE.md](docs/quantum/KERNEL-USAGE.md) for full guide.

### Hierarchical Reasoning Model (HRM)

Two-level reasoning architecture for complex multi-step problems:

```python
from yrsn_context.models.hrm import HRMReasoningModule

model = HRMReasoningModule(
    hidden_size=256,
    num_layers=4,
    h_update_interval=3  # H-level updates every 3 L-level steps
)
```

## Use Cases: Adaptive in Action

YRSN adapts strategy based on context characteristics. The same framework handles different scenarios:

### Legal Document Analysis
**Problem**: 200-page contracts requiring high accuracy with full provenance.

**YRSN Approach**: Hierarchical decomposition (handles size) + Iterative refinement (high precision) + Layered stack (quality tracking)

```python
# Decompose large document into blocks
blocks = decomposer.decompose(contract_text, question)

# Iterative refinement for precision
result = engine.retrieve(question)

# Layer by verification level
stack.add_layer(level=0, name="verified_clauses", content=[...], quality_score=0.98)
stack.add_layer(level=1, name="supporting_sections", content=[...], quality_score=0.90)
```

### Real-Time Customer Support
**Problem**: Fast responses (< 500ms), good-enough accuracy, personalization needed.

**YRSN Approach**: Bit-slicing only (speed priority), low precision acceptable

```python
# Fast bit-sliced retrieval - just core facts
sliced = retrieval.adaptive_precision_retrieve(query, target_precision_bits=6)

# Quick response with escalation if confidence low
if stack.compute_stack_quality() < 0.70:
    escalate_to_human()
```

### Medical Literature Review
**Problem**: 100+ papers, must detect contradictions, evidence hierarchy matters.

**YRSN Approach**: Hierarchical (many papers) + Layered (evidence levels) + Self-Calibration (learn optimal weights)

```python
# Layer by evidence strength
stack.add_layer(level=0, name="systematic_reviews", quality_score=0.98)  # Highest evidence
stack.add_layer(level=1, name="rcts", quality_score=0.95)               # RCTs
stack.add_layer(level=2, name="cohort_studies", quality_score=0.88)     # Observational

# Detect contradictions (CONFUSION collapse type)
analysis = detect_collapse(R=computed_R, S=computed_S, N=computed_N)
if analysis.collapse_type == CollapseType.CONFUSION:
    flag_contradictions()

# Self-calibrate evidence weights over time
calibrator.record_trace(trace)
if calibrator.get_statistics()['needs_calibration']:
    new_weights = calibrator.calibrate(current_weights, current_boost)
```

### Strategy Selection Guide

| Scenario | Primary Strategy | Why |
|----------|-----------------|-----|
| Simple fact lookup | Bit-Slicing (low precision) | Fast, sufficient |
| Complex multi-part question | Iterative Refinement | Handles residual queries |
| Large document (> 100KB) | Hierarchical Decomposition | Required for efficiency |
| Quality tracking needed | Layered Architecture | Provenance and scoring |
| Long-running system | Self-Calibration | Learns optimal params over time |
| Mission-critical | All combined | Maximum quality |

### The Adaptive Pattern

```
Query arrives
    │
    ├─ Estimate complexity
    │   ├─ Simple → Bit-slicing (fast)
    │   ├─ Complex → Iterative (thorough)
    │   └─ Large doc → Hierarchical (scalable)
    │
    ├─ Check for collapse
    │   ├─ Poisoning → More refinement iterations
    │   ├─ Distraction → Aggressive bit-slicing (Level 0 only)
    │   ├─ Confusion → Hierarchical isolation
    │   └─ Clash → Layered with quality gates
    │
    ├─ Return with quality score
    │
    └─ Self-Calibrate (background)
        └─ Record trace, adjust weights if FP/FN rates drift
```

See `docs/use-cases.md` and `docs/when-to-use-what.md` for detailed examples.

## Why Choose YRSN?

| Alternative | What it does | YRSN difference |
|-------------|-------------|-----------------|
| sklearn/scipy | Optimize accuracy | No context quality diagnosis |
| LangChain/LlamaIndex | Retrieve more context | No mechanism to distinguish signal from bias |
| YRSN | Diagnose failures, prescribe fixes | Tells you *why* context fails and *how* to fix it |

### Feature Comparison

| Feature | YRSN | LangChain | LlamaIndex | sklearn |
|---------|------|-----------|------------|---------|
| **Context Quality Diagnosis** | ✅ Collapse detection | ❌ | ❌ | ❌ |
| **Multiple Decomposition Methods** | ✅ RPCA, FIF, ESN | ❌ | ❌ | Partial |
| **Adaptive Strategy Selection** | ✅ 5 paradigms | ❌ | ❌ | ❌ |
| **Self-Calibration** | ✅ From traces | ❌ | ❌ | ❌ |
| **Neural Models** | ✅ CTM, HRM | ❌ | ❌ | ❌ |
| **Quantum Kernels** | ✅ PennyLane | ❌ | ❌ | ❌ |
| **Optimization Backends** | ✅ Gradient + EGGROLL | ❌ | ❌ | Partial |
| **Time Series Support** | ✅ Reservoir computing | ❌ | ❌ | ❌ |
| **Image Decomposition** | ✅ Multidimensional FIF | ❌ | ❌ | ❌ |
| **CLI Tool** | ✅ | ❌ | ❌ | ❌ |
| **Visualization** | ✅ Built-in | ❌ | ❌ | ❌ |

### Full Feature List

**Core Framework:**
- Y=R+S+N decomposition with multiple methods
- Collapse detection and remediation mapping
- Five context engineering paradigms
- Self-calibration from execution traces

**Decomposition Methods:**
- Robust PCA (IALM, APG algorithms)
- Fast Iterative Filtering (1D, 2D, multivariate)
- Echo State Networks for temporal data
- IMF-to-YRSN classification

**Optimization:**
- GradientBackend (Adam, SGD, RMSprop)
- EggrollBackend (evolution-guided, gradient-free)
- AdaptiveBackend (auto-selection)
- Phase transitions for escaping local optima

**Neural (PyTorch):**
- CTMContextRetriever
- NeuronLevelModel with private weights
- NeuralSynchronization for multi-signal fusion
- CertaintyBasedLoss for uncertainty-aware training
- IterativeContextRefinement (HP-INV inspired)
- SnApTracker for efficient gradients
- HRM (Hierarchical Reasoning Model)

**Quantum (PennyLane):**
- ZZFeatureMap, AngleEmbeddingFeatureMap
- YRSNQuantumKernel for sklearn integration
- SelfCalibratingVQC
- AWS Braket support for real hardware

**Applications:**
- Sudoku constraint satisfaction testbed
- Full encoder pipeline (YRSN-based attention weights)
- CTC puzzle parser
- Classical, hybrid, and neural solvers

**Tools:**
- CLI (`yrsn analyze`, `decompose`, `detect`, `info`)
- Visualization (`plot_yrsn_decomposition`, `plot_collapse_analysis`, etc.)
- sklearn-style data loaders
- Benchmarking utilities

## Acknowledgments

This library draws inspiration from multiple research areas:

**Hardware Architecture** (Strategies):
- Nature Electronics research on 3D integrated circuits and RRAM-based computing
- Demonstrates how physical computing principles inform software design patterns

**Neural Continuous Thought** (Neural Components):
- Sakana AI's Continuous Thought Machine (CTM) research
- HP-INV analog computing from Nature Electronics
- DeepMind's NoisyNet exploration
- Efficient gradient tracking (SnAp, RTRL variants)

## Requirements

- Python >= 3.11
- NumPy >= 1.20.0
- Optional: torch >= 2.0.0, pennylane >= 0.35.0

## Development

```bash
git clone https://github.com/RudyMartin/yrsn-context.git
cd yrsn-context
pip install -e ".[dev]"
pytest
```

## Author

Rudy Martin - [ORCID 0000-0002-6023-8526](https://orcid.org/0000-0002-6023-8526)

## License

MIT License - see LICENSE file.

## Citation

```bibtex
@software{yrsn_context,
  title = {YRSN Context: Adaptive Context Engineering Framework},
  author = {Martin, Rudy},
  year = {2025},
  url = {https://github.com/RudyMartin/yrsn-context},
  note = {ORCID: 0000-0002-6023-8526}
}
```
