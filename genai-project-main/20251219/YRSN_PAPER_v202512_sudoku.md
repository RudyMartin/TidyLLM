# YRSN: A Theoretical Framework for Quality-Adaptive Context Engineering

**Theory + Reference Implementation (Sudoku Domain)**

**Version:** v2025121h-OOD
**Date:** 2025-12-16
**Authors:** Rudy Martin et al.

---

## Version History

### v2025121h-OOD (Current)
Hardware-faithful R/S/N integration across all strategies + memristor crossbar projection.

**Key Changes:**
- ContextBlock now carries explicit R, S, N, ω fields with derived α, α_ω, τ
- Iterative Refinement: precision uses α × (1-N) weighting, residual = high-N blocks
- Bit-Slicing: slice assignment via R/S/N thresholds, not retriever precision
- Layered Stack: roughness mapped to N component, quality auto-computed from R/S/N
- Added Section 4.1 (ContextBlock) and Section 4.2 (Strategy Integration)
- Backward compatibility maintained via property aliases

**Memristor Integration (Option B deeper):**
- NEW: `YRSNMemristorProjection` with three crossbar arrays (R, S, N)
- NEW: HP Labs dynamics with τ-modulated plasticity
- NEW: True bit-sliced quantization (3-bit, not metaphorical)
- NEW: 23 tests validating hardware-faithful behavior
- Added Section 4.7 (Memristor Crossbar Projection)

### v2025121g-OOD
Expanded collapse taxonomy to 10 types across 3 domains (Quality, Reliability, Representation).

**Key Changes:**
- Collapse taxonomy expanded from 6 types to 10 types
- Added Representation Domain collapses: POSTERIOR_COLLAPSE, MODE_COLLAPSE, RSN_COLLAPSE
- Clarified CONFUSION = "High N AND high S" (not "contradictory R")
- Clarified CLASH = "Source S variance" (not "tier conflict")
- Updated all code examples with full 10-type enum
- Aligned with implementation in yrsn_context.core.decomposition.collapse_detection

### v2025121f-OOD
Integration of Out-of-Distribution (OOD) detection as meta-dimension extension.

**Merged Content:**
- YRSN_PAPER_v2025121e (core framework)
- yrsn_ood_context_analysis.md (OOD extension)

**Key Additions:**
- Extended decomposition: Y = R + S + N + O (where O is meta-dimension)
- Distributional-adjusted quality score: α_ω = ω·α + (1-ω)·α_prior
- Temperature-Quality-Distribution Triad: τ = 1/α_ω
- OOD-aware projection heads (Pillar 1 extension)
- Distributional phase transitions (Pillar 2 extension)
- 10 collapse types across 3 domains: Quality, Reliability, Representation (Pillar 3)
- OOD-hardened curriculum learning (Pillar 4 extension)
- Complete implementation code

### v2025121e
[Previous version - authenticity corrections and unique contributions clarification]

### Prior Versions
[See previous version history in original document]

---

## Abstract

Context quality determines model behavior. We introduce **YRSN**, a framework that decomposes context into three components—**Relevant (R)**, **Superfluous (S)**, and **Noise (N)**—and uses this decomposition to detect when context quality has collapsed.

**The Core Mechanism:**

Context is projected onto learned R/S/N subspaces using variational (VAE-style) heads that produce both means and variances. From these projections:

- **Quality (α)** = R / (R + S + N) — how much of the context is useful signal
- **Reliability (ω)** = emergent from projection variance — can we trust the R/S/N scores?
- **Temperature (τ)** = 1 / α — controls downstream reasoning (deterministic vs. exploratory)

The key insight: **ω emerges naturally from variance**. High variance in the projections indicates the input is out-of-distribution—no external OOD detector needed. This follows from Domingos' Tensor Logic: temperature is a thermodynamic variable that controls the sharpness of inference.

**The Danger Zone:**

High α + Low ω = **Hallucination Risk**. The model thinks the context is good (high quality score), but the projection is unreliable (high variance). This is the most dangerous state—confident but wrong.

**The Primary Output: COLLAPSE ALERT**

YRSN detects 10 collapse types across 3 domains:

| Domain | Collapses |
|--------|-----------|
| **Quality** | POISONING (too much N), DISTRACTION (too much S), CONFUSION (high N+S), CLASH (source conflicts) |
| **Reliability** | HALLUCINATION (high α + low ω), O_POISONING (OOD as signal), DRIFT (ω declining) |
| **Representation** | POSTERIOR_COLLAPSE, MODE_COLLAPSE, RSN_COLLAPSE |

Each collapse type has a specific detection criterion and remediation strategy.

**Implementation:**

**The Theory** (architecture-agnostic):
- Three domains: Quality (α), Reliability (ω), Representation (ρ)
- 10 collapse types with detection criteria
- Temperature-quality coupling: τ = f(α, ω, ρ)

**The Reference Implementation** (swappable):
- VAE projection heads (could be: attention, linear, etc.)
- Mahalanobis OOD detection (could be: energy-based, ensemble, etc.)
- 4-layer memristive memory (could be: RAG, Hopfield, transformer, etc.)
- Curriculum learning (could be: self-paced, RL, etc.)

**First Application**: Sudoku constraint satisfaction—demonstrating YRSN theory on a domain where ground truth is verifiable.

The system answers one question: **"Is this context safe to use?"**

**Keywords:** Context engineering, collapse detection, hallucination prevention, variational inference, out-of-distribution detection

---

## 1. Introduction

### 1.1 The Tale of Two Models

Consider two language models evaluated on Sudoku puzzles. Both achieve **91.4% accuracy** on the benchmark:

| Model | Accuracy | Unique Predictions | Behavior |
|-------|----------|-------------------|----------|
| Model A | 91.4% | 1 | Predicts position 65 for every puzzle |
| Model B | 91.4% | 64+ | Reasons from constraint rules |

Traditional metrics declare these models equivalent. Yet Model B exhibits the reasoning we actually want, while Model A exploited a dataset bias: position 65 is empty in 64 of 70 benchmark puzzles (91.4%).

This paradox—**high accuracy masking poor generalization**—motivates our work.

### 1.2 The Core Problem

Accuracy conflates three fundamentally different mechanisms:
- **Memorization:** Recognizing training patterns
- **Shortcut learning:** Exploiting dataset biases
- **Genuine reasoning:** Actually understanding the problem

Current evaluation cannot distinguish these mechanisms. When your model scores 95% accuracy, you cannot answer: *"Did it learn the task, or did it learn the dataset?"*

### 1.3 The OOD Challenge

Production systems face an additional challenge beyond R/S/N classification: **Out-of-Distribution (OOD) context**. Content from unknown distributions can:

1. **Masquerade as R**: High apparent relevance, but semantically wrong domain
2. **Be ignored as S**: Missing critical novel information
3. **Be quarantined as N**: Over-rejection of valid novel signals

The danger: R/S/N projections trained on in-distribution data produce **confident but unreliable** classifications for OOD content. This is the hallucination problem at its root.

### 1.4 Problem Statement

We address four interconnected challenges:

1. **Signal Quality vs. Quantity**: More context is not always better. What matters is the *quality* of information—how much helps (R), how much is neutral (S), and how much actively hurts (N).

2. **Distributional Alignment**: Can we trust our R/S/N classifications? OOD content renders them unreliable.

3. **Context Collapse**: Under certain conditions, model behavior degrades catastrophically. We need to detect and prevent these failure modes—including OOD-induced collapses.

4. **Adaptive Computation**: Different quality regimes demand different computational strategies—deterministic reasoning for high-quality, in-distribution context; exploratory search for uncertain or OOD situations.

### 1.5 Contributions

We make the following contributions:

**Theoretical Contributions:**

1. **Y=R+S+N Decomposition**: A framework that decomposes context into Relevant, Superfluous, and Noise components, with S as a distinct category (not merely "not N").

2. **OOD as Meta-Dimension (O)**: Extension to Y = R + S + N + O where O indicates R/S/N reliability, not a fourth quality type.

3. **Temperature-Quality-Distribution Triad**: A coupling τ = 1/α_ω connecting context quality AND distributional alignment to computational temperature—grounded in thermodynamic principles (Section 2.4).

4. **Two Independent Threshold Systems**:
   - **BBP thresholds** (from random matrix theory): Determine decomposition rank
   - **Curriculum thresholds** (empirical, tunable): Determine constraint staging
   - **Distributional thresholds** (ω_critical): Determine R/S/N reliability

5. **Extended Collapse Taxonomy**: Ten collapse types across three domains (Quality, Reliability, Representation) with detection criteria and remediation strategies.

6. **Input-Layer Safety**: Complementary to output-layer approaches (RLHF, Constitutional AI), YRSN assesses context trustworthiness *before* processing (Section 2.5).

7. **Information Bottleneck Extension**: Three-way decomposition as principled extension of Tishby's IB, with explicit modeling of the superfluous dimension S (Section 2.6).

**Reference Implementation (Sudoku Domain):**

8. **Sudoku as Proof-of-Concept**: Constraint satisfaction domain where ground truth is verifiable, demonstrating YRSN theory with:
   - VAE-style projection heads (swappable)
   - Mahalanobis OOD detection (swappable)
   - Progressive curriculum learning (swappable)

9. **Convergence with Nested Learning**: Independent development paths arriving at similar multi-scale principles—while YRSN addresses gaps NL does not (three-way decomposition, OOD detection, external orchestration, collapse taxonomy).

### 1.6 The Product Era Context: Why YRSN Matters Now

The AI research community finds itself at an inflection point. Following the release of GPT-5 in August 2025, Yannic Kilcher—an AI researcher and influential commentator—offered a sobering assessment:

> "The era of boundary-breaking advancements is over... AGI is not coming and we can be reasonably sure about that. We're in the Samsung Galaxy era of LLMs—where each new generation has marginally better features but no groundbreaking capabilities."

This analysis, echoed by Ilya Sutskever's November 2025 observation that LLMs "generalize dramatically worse than people," marks what MIT Technology Review calls "the great AI hype correction of 2025" [Douglas Heaven, 2025].

**Why YRSN-OOD Addresses This**:

YRSN-OOD is specifically designed for the product era:

1. **Principled Context Management**: The R/S/N decomposition offers measurable quality metrics—not black-box accuracy, but interpretable component scores.

2. **OOD-Aware Reliability**: The ω-score prevents confident failures by detecting when classifications are unreliable.

3. **Failure Mode Detection**: The 10-type collapse taxonomy across 3 domains transforms opaque failures into actionable diagnostics.

4. **Integration-Ready Architecture**: The modular design enables incremental adoption with existing systems.

5. **Bridging Demo-to-Production Gaps**: The temperature-quality-distribution triad provides principled guidance for when models should be confident versus exploratory.

### 1.7 Paper Organization

Section 2 reviews related work, including thermodynamic foundations, safety paradigms, and information theory. Section 3 presents the extended Y=R+S+N+O framework and temperature-quality-distribution triad. Section 4 details the reference implementation (Sudoku domain). Section 5 describes the architecture and algorithms. Section 6 presents applications and evaluation. Section 7 discusses limitations and implications. Section 8 concludes.

---

## 2. Background and Related Work

YRSN draws on concepts from diverse fields—signal processing, thermodynamics, information theory, constraint satisfaction, and OOD detection. This section traces those conceptual lineages.

### 2.1 Signal Processing & Decomposition

#### 2.1.1 Robust PCA and Matrix Decomposition

**Robust PCA** [Candès et al., 2011] decomposes matrices into low-rank and sparse components:

$$\min_{L,S} \|L\|_* + \lambda\|S\|_1 \quad \text{subject to} \quad M = L + S$$

**YRSN extends** this to three components with semantic meaning:

$$\min_{R,S,N} \|R\|_* + \lambda_S\|S\|_1 + \lambda_N\|N\|_F^2 \quad \text{subject to} \quad Y = R + S + N$$

Unlike classic Robust PCA which is purely structural, YRSN adds semantic interpretation, quality metrics, temperature mapping, and OOD detection.

#### 2.1.2 Phase Transitions and Spectral Analysis

**The BBP Transition** [Baik, Ben Arous, Péché, 2005] identifies critical thresholds where signal emerges from noise:

$$\alpha_\ell = \frac{\ell - 1}{2\ell}$$

| Order (ℓ) | Critical α | Interpretation |
|-----------|-----------|----------------|
| 1 | 0.000 | Linear structure always visible |
| 2 | 0.250 | Quadratic patterns detectable |
| 3 | 0.333 | Cubic structure recoverable |
| ∞ | 0.500 | Maximum useful depth |

**YRSN-OOD extends** this with a distributional phase transition for OOD detection.

### 2.2 Out-of-Distribution Detection

OOD detection has a rich literature that YRSN-OOD integrates:

#### 2.2.1 Distance-Based Methods

**Mahalanobis Distance** [Lee et al., 2018]: Distance from training centroid in feature space:

$$d_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$$

Well-understood and interpretable, but assumes Gaussian distribution. YRSN uses this as default for Decomposition OOD detection.

#### 2.2.2 Energy-Based Methods

**Energy-Based OOD Detection** [Liu et al., 2020]: Free energy of input under model:

$$E(x; f) = -\tau \cdot \log \sum_i \exp(f_i(x)/\tau)$$

No distributional assumptions, but requires calibration. YRSN uses this as backup detector.

#### 2.2.3 Ensemble Methods

**Deep Ensembles** [Lakshminarayanan et al., 2017]: Variance across model ensemble captures epistemic uncertainty. Computationally expensive but catches model uncertainty. YRSN uses for high-stakes contexts.

#### 2.2.4 OOD Detection Methods Comparison

| Method | Mechanism | Strengths | Weaknesses | YRSN Integration |
|--------|-----------|-----------|------------|------------------|
| **Mahalanobis Distance** | Distance from training centroid | Well-understood, interpretable | Assumes Gaussian | Default for Pillar 1 |
| **Energy-Based** | Free energy of input | No distributional assumptions | Requires calibration | Backup detector |
| **Ensemble Disagreement** | Variance across ensemble | Catches epistemic uncertainty | Computationally expensive | High-stakes contexts |
| **Gradient-Based** | Gradient magnitude w.r.t. input | Sensitive to adversarial OOD | Can be fooled | Adversarial detection |
| **Reconstruction Error** | Autoencoder reconstruction loss | Catches structural OOD | Needs auxiliary model | Multimodal contexts |
| **Spectral Analysis** | Eigenvalue structure deviation | Connects to BBP thresholds | Requires batch processing | Phase detection integration |

### 2.3 Nested Learning and Multi-Level Optimization

The 2025 NeurIPS proceedings introduced the **Nested Learning** framework [Behrouz et al., 2025], revealing that deep learning architectures are integrated systems of **nested optimization problems**.

YRSN was developed independently (beginning November 2024) through a different path: extending Robust PCA to three-component decomposition, then discovering that curriculum thresholds naturally produce multi-scale behavior. The convergence suggests the underlying principles are robust.

#### 2.3.1 The Anterograde Amnesia Problem

Nested Learning's most striking insight concerns the **static nature of LLMs after pre-training**:

> "Current LLMs suffer from a pattern similar to anterograde amnesia—their knowledge is limited to either the immediate context that fits into their context window, or the knowledge in MLP layers that stores long-past, before the onset of 'end of pre-training.'"

**OOD context exacerbates this**: Novel domain knowledge absent from pre-training cannot be properly assessed for R/S/N classification because the projection heads themselves are out-of-distribution.

This creates a chicken-and-egg problem: we need to classify context quality to process it, but classification accuracy depends on distributional alignment we cannot guarantee.

**YRSN-OOD's Response**: Rather than solving anterograde amnesia architecturally, YRSN provides **principled context management within the constraint**, with explicit detection of when that management becomes unreliable (low ω).

#### 2.3.2 Gaps Addressed by YRSN

| Capability | YRSN | NL/Titans |
|------------|------|-----------|
| Three-way decomposition | R/S/N with explicit dilution | Binary (signal/noise) |
| Superfluous modeling | Distinct S component | Not addressed |
| **OOD detection** | **ω-score with multi-method detection** | **Not addressed** |
| Tunable thresholds | Curriculum (configurable) | Fixed by architecture |
| External orchestration | Works with existing models | Requires new architecture |
| Collapse taxonomy | **10 types across 3 domains** + remediation | Capacity overflow only |
| Multi-source calibration | Hierarchical per-source R/S/N + ω | Single memory module |
| Production focus | Deployment-ready framework | Research architecture |

#### 2.3.3 YRSN vs Titans: Computed vs Learned Surprise

The **Titans** architecture (Google, NeurIPS 2025) introduces a "surprise metric" to decide what information to store in long-term memory. When the model encounters unexpected information—measured by prediction error—it commits that to memory. This approach handles 2M+ token contexts effectively.

**YRSN takes the opposite approach**: rather than *learning* what's surprising, we *compute* it directly via statistical OOD detection.

```
YRSN = YRSN + OOD
       (quality)  (surprise/shift)

                    Changed Context (e.g., new diagnosis)
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    OOD DETECTION                            │
│  Components:                                                │
│  ├── VGFA    (Variance-Guided Feature Analysis)            │
│  ├── CVJF    (Cross-Validation Jacobian Flow)              │
│  ├── SEDA    (Spectral Energy Distribution Analysis)       │
│  ├── MSDF    (Multi-Source Distribution Fusion)            │
│  └── Shift Detection                                        │
│                                                             │
│  Output: ω (omega) = distributional reliability             │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                 YRSN DECOMPOSITION                          │
│                                                             │
│  α (alpha) = quality = R/(R+S+N)                           │
│  ω (omega) = reliability (from OOD)                        │
│                                                             │
│  Combined: τ = f(α, ω)                                     │
│            Low ω → OOD → HIGH τ → Conservative             │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   ROUTING DECISION                          │
│                                                             │
│  High α, High ω → GREEN (confident, in-distribution)       │
│  High α, Low ω  → YELLOW (quality ok, but OOD - verify)    │
│  Low α, any ω   → RED (low quality - human review)         │
└─────────────────────────────────────────────────────────────┘
```

**The Fundamental Difference:**

| Aspect | Titans | YRSN |
|--------|--------|-------|
| **Surprise signal** | Learned (prediction error: s = ||pred - actual||) | Computed (ω from OOD detection) |
| **What it measures** | "Did the model predict this?" | "Is this in-distribution?" + "Is quality good?" |
| **Failure mode** | Model may be bad at predicting → wrong surprise | Statistical measure on DATA, not model |
| **Interpretability** | Black-box neural score | Explicit: R=0.3, S=0.4, N=0.3, ω=0.6 |
| **Memory decision** | Binary (store if surprising) | Graded (τ controls plasticity continuously) |
| **Debuggability** | "Why did it remember X?" | "ω=0.3 because Mahalanobis distance exceeded threshold" |

**The Critical Insight:**

Titans asks: *"Did I fail to predict this?"* — but prediction failure conflates model inadequacy with genuine novelty.

YRSN asks: *"Is this statistically different from what I've seen?"* — a property of the DATA, independent of model quality.

**Practical Example: Medical Diagnosis Change**

Consider a patient whose diagnosis changes from routine diabetes management to a new cancer diagnosis:

| Scenario | ω (OOD) | α (Quality) | τ | Route | Behavior |
|----------|---------|-------------|---|-------|----------|
| Routine diabetes visit | 0.9 (in-dist) | 0.8 | 1.1 | GREEN | Confident automation |
| **New cancer diagnosis** | **0.3 (OOD!)** | 0.6 | 2.8 | **RED** | Human review required |
| 2nd cancer visit | 0.7 (learning) | 0.7 | 1.5 | YELLOW | Assisted processing |
| 10th cancer visit | 0.9 (normal now) | 0.8 | 1.1 | GREEN | Back to automation |

The system **naturally adapts** without storing patient "backstory":
- Distribution shift (ω drops) triggers conservative routing
- As similar cases accumulate, ω recovers
- No explicit memory of "Patient X has cancer"—the distribution IS the memory

**Why This Matters for Titans Comparison:**

| Scenario | Titans Response | YRSN Response |
|----------|-----------------|----------------|
| Rare disease, model poorly trained | High surprise (prediction error) ✓ | High ω shift (OOD detected) ✓ |
| Rare disease, model well-trained | Low surprise (good prediction) ✗ | High ω shift (still OOD!) ✓ |
| Common disease, model broken | High surprise (prediction error) ✗ | Low ω shift (in-distribution) ✓ |

YRSN correctly identifies distribution shift **regardless of model quality**. Titans conflates the two.

**Temperature Coupling (Unique to YRSN):**

Titans has no notion of temperature—memory decisions are binary. YRSN couples everything through τ = 1/α_ω:

- Low ω (OOD detected) → High τ → Conservative routing + aggressive learning
- High ω (in-distribution) → Low τ → Confident routing + EWC memory protection

This produces **emergent adaptation**: the system naturally becomes more exploratory when facing novel inputs and more conservative when operating in familiar territory—without explicit rules.

**The Publication Differentiator:**

YRSN provides **provable** OOD detection via statistical methods (Mahalanobis distance, spectral analysis) with **interpretable** quality decomposition (R/S/N). Titans provides impressive scale (2M+ tokens) but sacrifices interpretability for learned heuristics. For safety-critical applications requiring auditability, YRSN's explicit computation is essential.


### 2.4 Thermodynamic Foundations: Temperature as Physics

The temperature parameter τ appears throughout deep learning—in softmax, knowledge distillation, sampling, and attention. Most treat it as a hyperparameter to tune. YRSN treats it as a **physical quantity** that emerges from data quality.

#### 2.4.1 The Boltzmann-Hinton-Domingos Lineage

The intellectual lineage runs through three key insights:

**Boltzmann (1870s)**: The probability of a system being in state i with energy E_i follows:

$$P(i) = \frac{e^{-E_i / k_B T}}{\sum_j e^{-E_j / k_B T}}$$

Temperature T controls the sharpness of the distribution. Low T → system concentrates on lowest-energy states. High T → uniform exploration.

**Hinton's Knowledge Distillation (2015)**: Temperature controls knowledge transfer between models:

$$q_i = \frac{e^{z_i / \tau}}{\sum_j e^{z_j / \tau}}$$

Hinton showed that "dark knowledge"—the relative probabilities of wrong answers—transfers better at high τ. The temperature isn't arbitrary; it determines what information flows.

**Domingos' Tensor Logic (2024)**: Pedro Domingos unified logic and probability through tensor networks, showing that temperature is the bridge between deterministic inference (τ→0) and probabilistic inference (τ→∞). In his framework:

> "Temperature is not a hyperparameter—it is a thermodynamic variable that controls the sharpness of inference."

#### 2.4.2 YRSN's Contribution: Emergent Temperature

YRSN makes temperature **emergent from data quality** rather than a tuning knob:

$$\tau = \frac{1}{\alpha_\omega} = \frac{1}{\omega \cdot \alpha + (1-\omega) \cdot \alpha_{prior}}$$

This is not arbitrary. It follows from the physics:

| Data State | Quality (α) | Reliability (ω) | Temperature (τ) | Physical Interpretation |
|------------|-------------|-----------------|-----------------|------------------------|
| High quality, in-distribution | 0.9 | 0.9 | 1.2 | Low entropy → deterministic |
| High quality, OOD | 0.9 | 0.3 | 2.0 | Unreliable → exploratory |
| Low quality, in-distribution | 0.3 | 0.9 | 3.0 | Poor signal → explore |
| Low quality, OOD | 0.3 | 0.3 | 5.0+ | Maximum uncertainty |

The system "heats up" when uncertain and "cools down" when confident—exactly as thermodynamic systems do. This isn't metaphor; it's the same mathematics.

#### 2.4.3 Why This Matters

Most ML systems treat temperature as a dial to turn. YRSN says: **you're turning the wrong dial**.

The right approach:
1. Measure context quality (α from R/S/N decomposition)
2. Measure distributional alignment (ω from OOD detection)
3. Let temperature emerge: τ = 1/α_ω

This connects YRSN to fundamental physics rather than ad-hoc engineering. When reviewers ask "why τ = 1/α?", the answer is: "Because that's what thermodynamics says."

### 2.5 Input-Layer vs Output-Layer Safety

Most AI safety research focuses on constraining model outputs. YRSN operates at a different layer: assessing input quality before processing.

#### 2.5.1 The Output-Layer Paradigm

**RLHF** [Ouyang et al., 2022]: Train models to produce outputs humans prefer.

**Constitutional AI** [Bai et al., 2022]: Train models to follow explicit principles in outputs.

**Output Filtering**: Post-hoc detection of harmful/incorrect outputs.

These approaches share an assumption: the model receives context, processes it, and we constrain what comes out.

#### 2.5.2 The Problem with Output-Only Safety

Consider a medical AI receiving patient records:

```
Input: [Valid patient history] + [Injected misinformation]
       ↓
    Model processes everything
       ↓
Output: [Confident but wrong diagnosis]
       ↓
    RLHF/Constitutional AI: "Output looks professional" ✓
```

Output-layer safety cannot catch this because:
- The output format is correct
- The reasoning appears sound
- The confidence is appropriate for the (corrupted) context

**The failure mode**: Garbage in, polished garbage out.

#### 2.5.3 YRSN's Input-Layer Approach

YRSN intervenes before processing:

```
Input: [Valid patient history] + [Injected misinformation]
       ↓
    YRSN Assessment:
    - R = 0.4 (some relevant content)
    - S = 0.2 (neutral filler)
    - N = 0.4 (detected noise/inconsistency)
    - ω = 0.6 (partial OOD characteristics)
    - α = 0.4 → τ = 2.5
       ↓
    COLLAPSE ALERT: CONFUSION (high N + S)
       ↓
    Route to human review (RED stream)
```

The system catches the problem **before** the model can produce confident-but-wrong output.

#### 2.5.4 Complementary, Not Competing

| Layer | What it catches | YRSN | RLHF/Constitutional |
|-------|-----------------|------|---------------------|
| Input | Poisoned context | ✓ | ✗ |
| Input | OOD content | ✓ | ✗ |
| Input | Source conflicts | ✓ | ✗ |
| Output | Harmful language | ✗ | ✓ |
| Output | Policy violations | ✗ | ✓ |
| Output | Format compliance | ✗ | ✓ |

YRSN and output-layer safety are **complementary**. A complete system needs both:
- YRSN ensures the model receives trustworthy context
- RLHF/Constitutional AI ensures the model produces appropriate outputs

But input-layer safety must come first. As Russell [2019] argues in *Human Compatible*: provably beneficial AI requires knowing what information to trust, not just what outputs to produce.

### 2.6 Information Bottleneck and the S Dimension

Tishby's **Information Bottleneck (IB)** principle [Tishby et al., 2000] provides a theoretical foundation for representation learning: compress input X to representation T while preserving information about target Y.

$$\min_{p(t|x)} I(X;T) - \beta \cdot I(T;Y)$$

YRSN can be understood as a specific IB instantiation with explicit structure.

#### 2.6.1 Standard IB: Signal vs Noise

Traditional IB produces a compressed representation that:
- Retains information predictive of Y (signal)
- Discards information not predictive of Y (noise)

This is a **two-way decomposition**: useful vs not-useful.

#### 2.6.2 YRSN's Three-Way Extension

YRSN adds a critical third category:

| Component | IB Interpretation | Effect on Y |
|-----------|-------------------|-------------|
| **R (Relevant)** | I(T;Y) maximized | Improves prediction |
| **S (Superfluous)** | I(T;Y) ≈ 0, but I(X;T) > 0 | Neutral—neither helps nor hurts |
| **N (Noise)** | Negative transfer | Degrades prediction |

The **S dimension** is YRSN's key extension. Standard IB lumps S with either R or N, but they're fundamentally different:

- **S is not noise**: It doesn't hurt prediction
- **S is not signal**: It doesn't help prediction
- **S consumes resources**: Attention, memory, compute

#### 2.6.3 Why S Matters in Practice

**RAG Systems**: Retrieval returns documents ranked by relevance. But "relevant" documents contain:
- Directly useful passages (R)
- Background context (S)
- Outdated/contradictory information (N)

Without explicit S modeling, systems either:
- Include everything (wasting context window on S)
- Aggressively filter (losing R along with S)

**Long Context Models**: A 100K token context might be:
- 10K tokens of R (actual signal)
- 80K tokens of S (related but not needed)
- 10K tokens of N (noise, contradictions)

Standard attention treats all tokens similarly. YRSN weights by R/S/N classification.

#### 2.6.4 The Dilution Problem

S doesn't hurt—but it **dilutes**. Consider:

$$\alpha = \frac{R}{R + S + N}$$

Even with N = 0, high S drives down α:

| R | S | N | α | Interpretation |
|---|---|---|---|----------------|
| 100 | 0 | 0 | 1.0 | Pure signal |
| 100 | 100 | 0 | 0.5 | 50% diluted |
| 100 | 900 | 0 | 0.1 | 90% diluted |

This is the **long context problem**: more context often means more S, not more R. YRSN makes this explicit and measurable.

#### 2.6.5 IB with Explicit S

YRSN's objective can be written as an extended IB:

$$\min_{p(r,s,n|x)} I(X;S) + \lambda_N \cdot I(X;N) - \beta \cdot I(R;Y)$$

Where:
- Minimize information in S (reduce dilution)
- Minimize information in N (reduce noise)
- Maximize information in R about Y (preserve signal)

This provides theoretical grounding for the R/S/N decomposition as a principled extension of information-theoretic representation learning.


---

## 3. The Y=R+S+N+O Framework

Not all context is created equal. Some helps, some hurts, some just gets in the way—and some comes from distributions we've never seen. Traditional approaches distinguish only "signal" from "noise." We argue this is insufficient—and that both the third category (superfluous information) and the meta-category (distributional alignment) are precisely what cause production failures.

### 3.1 Core Decomposition

We decompose any context signal Y into three orthogonal quality components:

$$Y = R + S + N$$

Where:
- **R (Relevant)**: Information that directly contributes to task performance
- **S (Superfluous)**: Information that neither helps nor hurts—neutral context
- **N (Noise)**: Information that actively degrades performance

### 3.2 The OOD Meta-Dimension

We extend the decomposition with a meta-dimension:

$$Y = R + S + N + O$$

Where **O (Out-of-Distribution)** represents content that:
- Cannot be reliably projected onto learned R/S/N subspaces
- Exhibits statistical properties outside the training manifold
- Requires detection **before** (not during) R/S/N classification

**Key Insight**: O is not a quality dimension like R/S/N—it's a **meta-dimension** that indicates whether R/S/N scoring is reliable.

### 3.3 OOD Context Categories

| OOD Type | Description | Example | Risk |
|----------|-------------|---------|------|
| **Semantic OOD** | Known words, unknown meaning | Domain jargon from unfamiliar field | Misclassified as R |
| **Structural OOD** | Unfamiliar document formats | New API response schema | Misclassified as S |
| **Temporal OOD** | Previously valid, now stale | Pre-merger company data | Misclassified as R |
| **Adversarial OOD** | Intentionally crafted edge cases | Prompt injection patterns | Misclassified as anything |
| **Domain Drift OOD** | Gradual distribution shift | Evolving user terminology | Slow R→O transition |

### 3.4 The O-Score: Distributional Alignment Metric

Define the **O-score (ω)** as a confidence measure for R/S/N classification:

$$\omega = P(\text{in-distribution} | Y)$$

Low ω indicates high O-content, signaling that R/S/N scores are unreliable.

### 3.5 Distributional-Adjusted Quality Score

The original quality score:
$$\alpha = \frac{R}{R + S + N}$$

Becomes the **distributional-adjusted quality score**:
$$\alpha_{\omega} = \omega \cdot \alpha + (1 - \omega) \cdot \alpha_{\text{prior}}$$

Where $\alpha_{\text{prior}}$ is a conservative prior (typically 0.5, triggering exploratory behavior).

### 3.6 Temperature-Quality-Distribution Triad

The temperature-quality duality (τ = 1/α) extends to:

$$\tau = \frac{1}{\alpha_{\omega}} = \frac{1}{\omega \cdot \alpha + (1 - \omega) \cdot \alpha_{\text{prior}}}$$

| ω (O-score) | α | τ | Behavior |
|-------------|---|---|----------|
| High (0.9) | High (0.8) | Low (1.39) | Confident, deterministic |
| High (0.9) | Low (0.3) | High (3.03) | Exploratory within known space |
| Low (0.3) | Any | Very High (≥2.86) | Maximum uncertainty, flag for review |
| Very Low (0.1) | Any | Extreme (≥5.26) | Refuse or escalate |

### 3.7 Distributional Phase Transitions

The BBP transition thresholds determine when signal emerges from noise. For OOD content, we need a **distributional phase transition**:

$$\omega_{\text{critical}} = 1 - \sqrt{\frac{1}{n}}$$

Where n is the effective sample size of the projection training data. Below this threshold:
- R/S/N projections become unreliable
- System should switch to OOD-handling protocols
- Temperature locks to high values regardless of apparent α

### 3.8 Multi-Source OOD Calibration

Extend the multi-source calibration hierarchy to track per-source distributional alignment:

| Source Tier | Expected ω | OOD Handling |
|-------------|------------|--------------|
| **ACTUAL** (ground truth) | ~1.0 | Should never be OOD; if detected, critical alert |
| **LEARNED** (validated) | >0.7 | Monitor for drift, retrain on detection |
| **EMERGENT** (unvalidated) | Variable | Default to suspicious, require confirmation |
| **EXTERNAL** (third-party) | Unknown | Always run OOD detection, never trust blindly |

---

## 4. Reference Implementation: Sudoku Domain

> **Note**: This section describes ONE implementation of YRSN theory. The architectural choices (VAE heads, Mahalanobis OOD, memristive memory) are swappable. The theory in Section 3 is what we defend; this implementation demonstrates feasibility.

Section 3 defined *what* YRSN measures. This section describes *how* we implement those measurements in practice, with full OOD awareness.

| Component | Question | Timescale | Implementation Choice |
|--------|----------|----------|---------------|
| **Decomposition** | How do we classify R/S/N? | Fast | VAE projection (swappable) |
| **Detection** | How do we detect OOD? | Fast | Mahalanobis + CVJF (swappable) |
| **Collapse** | How do we detect failures? | Medium | 10-type taxonomy (theory) |
| **Memory** | How do we learn over time? | Slow | 4-layer memristor (swappable) |

### 4.1 Core Data Structure: ContextBlock with R/S/N

The fundamental unit of context in YRSN is the `ContextBlock`, which carries explicit R/S/N decomposition:

```python
@dataclass
class ContextBlock:
    """
    Represents a retrieved context segment with YRSN decomposition.

    YRSN Decomposition:
        R (Relevant): Information that directly contributes to task performance
        S (Superfluous): Information that neither helps nor hurts—neutral context
        N (Noise): Information that actively degrades performance

    Derived Metrics:
        alpha (α): Quality score = R / (R + S + N)
        alpha_omega (α_ω): Distribution-adjusted quality
        tau (τ): Temperature = 1 / α_ω
    """
    content: str
    source: str = "unknown"

    # YRSN decomposition
    R: float = 0.5  # Relevant
    S: float = 0.3  # Superfluous
    N: float = 0.2  # Noise
    omega: float = 1.0  # OOD reliability (1.0 = in-distribution)

    # Metadata
    precision_level: int = 0

    @property
    def alpha(self) -> float:
        """Quality score α = R / (R + S + N)."""
        total = self.R + self.S + self.N
        return self.R / total if total > 0 else 0.0

    @property
    def alpha_omega(self) -> float:
        """Distribution-adjusted quality α_ω = ω·α + (1-ω)·α_prior."""
        alpha_prior = 0.5  # Conservative prior for OOD content
        return self.omega * self.alpha + (1 - self.omega) * alpha_prior

    @property
    def tau(self) -> float:
        """Temperature τ = 1 / α_ω (clamped to avoid division by zero)."""
        return 1.0 / max(self.alpha_omega, 0.01)

    # Backward compatibility aliases
    @property
    def relevance_score(self) -> float:
        """Backward compatible alias for α (quality score)."""
        return self.alpha

    @property
    def quality_score(self) -> float:
        """Backward compatible alias for α_ω (distribution-adjusted quality)."""
        return self.alpha_omega
```

**Key Design Decisions:**

1. **Explicit R/S/N fields**: Not a single quality score, but three semantic components
2. **Derived properties**: α, α_ω, τ computed on access, always consistent
3. **Backward compatibility**: Legacy code using `relevance_score` continues to work
4. **Validation**: R, S, N, ω must be in [0, 1]

### 4.2 Strategy Integration: Hardware-Faithful R/S/N Usage

The hardware-inspired strategies now use R/S/N decomposition directly, not generic quality scores.

#### 4.2.1 Iterative Refinement (HP-INV Algorithm)

**YRSN Mapping:**
- Residual = content with high N (noise to eliminate)
- Refinement target = increase R, decrease N
- Convergence = α exceeds threshold

```python
class IterativeContextEngine:
    """HP-INV inspired iterative refinement using YRSN decomposition."""

    def _compute_precision(self, query: str, context: List[ContextBlock]) -> float:
        """
        Uses YRSN decomposition:
        - α = R/(R+S+N) as base quality score
        - Weight by (1-N) to penalize noisy content
        """
        if not context:
            return 0.0

        # Weight by precision level AND noise penalty
        weights = [2**block.precision_level * (1 - block.N) for block in context]
        scores = [block.alpha for block in context]  # α = R/(R+S+N)

        return sum(w * s for w, s in zip(weights, scores)) / sum(weights)

    def _identify_residual_blocks(self, context: List[ContextBlock],
                                   noise_threshold: float = 0.3) -> List[ContextBlock]:
        """YRSN mapping: Residual = content with high N (noise to eliminate)."""
        return [b for b in context if b.N > noise_threshold]

    def _merge_contexts(self, existing, coarse, fine) -> List[ContextBlock]:
        """Sort by (precision_level, α quality score, -N noise)."""
        all_blocks = existing + fine + coarse
        all_blocks.sort(key=lambda b: (b.precision_level, b.alpha, -b.N), reverse=True)
        return all_blocks
```

#### 4.2.2 Bit-Slicing (RRAM Algorithm)

**YRSN Mapping:**
- Slice 0 (Core) = high R content
- Slice 1 (Detail) = medium R, low N
- Slice 2 (Background) = high S content
- Slice 3 (Tangential) = high N content (candidate for removal)

```python
class BitSlicedRetrieval:
    """RRAM-inspired bit-slicing using YRSN decomposition."""

    def assign_slice_from_rsn(self, R: float, S: float, N: float) -> int:
        """
        Assign slice level based on YRSN decomposition.
        Content drives slice assignment, not retriever precision.
        """
        if R > 0.7:
            return 0  # Core - most significant slice
        elif R > 0.4 and N < 0.2:
            return 1  # Detail - supporting information
        elif S > 0.5:
            return 2  # Background - neutral context
        else:
            return 3  # Tangential/Noise - least significant
```

#### 4.2.3 Layered Stack (3D Integration Algorithm)

**YRSN Mapping:**
- Layer quality = α of layer content
- Interface quality = min(lower_α, upper_α) × ω
- Roughness = N component (noise propagates like roughness)

```python
class LayeredContextStack:
    """3D-stacking inspired layered architecture using YRSN decomposition."""

    def add_layer(self, level: int, name: str,
                  content: Union[List[str], List[ContextBlock]],
                  quality_score: Optional[float] = None) -> bool:
        """
        YRSN mapping:
        - Layer quality = α of layer content (auto-computed if ContextBlocks)
        - Roughness = N component (noise propagates like roughness)
        """
        if content and isinstance(content[0], ContextBlock):
            blocks = content
            # Use YRSN metrics: α × ω for distribution-adjusted quality
            layer_alpha = np.mean([b.alpha for b in blocks])
            layer_omega = np.mean([b.omega for b in blocks])
            layer_N = np.mean([b.N for b in blocks])
            computed_quality = layer_alpha * layer_omega
            # Roughness = noise component (noise propagates like roughness)
            layer_roughness = layer_N * 2.0
        else:
            computed_quality = quality_score or 0.5
            layer_roughness = (1 - computed_quality) * 2.0

        # Roughness compounds from lower layers (like 3D IC interface roughness)
        if level > 0 and (level - 1) in self.layers:
            lower_roughness = self.layers[level - 1].roughness
            layer_roughness = np.sqrt(layer_roughness**2 + lower_roughness**2)

        # ... store layer with computed metrics
```

#### 4.2.4 Before/After Comparison

| Component | Before (Generic) | After (YRSN) |
|-----------|------------------|--------------|
| **ContextBlock** | `relevance_score: float` | `R, S, N, ω` + derived `α, α_ω, τ` |
| **Iterative Refinement** | `weight × relevance` | `weight × α × (1-N)` |
| **Bit-Slicing** | Slice = retriever precision | Slice = f(R, S, N) |
| **Layered Stack** | Explicit quality param | Auto from R/S/N |
| **Residual Detection** | None | Blocks with N > threshold |
| **Roughness** | `(1-quality) × 2` | `N × 2` (physics mapping) |

**Concrete Example:**

```python
# Two blocks with same "old" relevance but different R/S/N
block_a = ContextBlock(content="Python created by Guido in 1991",
                       R=0.85, S=0.10, N=0.05, omega=1.0)
# α = 0.85, τ = 1.18 (low temperature = stable)

block_b = ContextBlock(content="Python is amazing! Best ever!",
                       R=0.85, S=0.05, N=0.60, omega=0.7)
# α = 0.57, τ = 1.64 (high temperature = unstable)

# OLD: Both scored ~0.85 relevance
# NEW: Block B penalized for noise

# Iterative Refinement precision:
precision_a = weight × 0.85 × (1 - 0.05) = weight × 0.81  ✓
precision_b = weight × 0.57 × (1 - 0.60) = weight × 0.23  ✗

# Bit-Slicing assignment:
slice_a = 0  # Core (R > 0.7)
slice_b = 3  # Tangential (high N)

# Layered Stack roughness:
roughness_a = 0.05 × 2.0 = 0.10  # Smooth
roughness_b = 0.60 × 2.0 = 1.20  # Rough
```

### 4.3 VAE Projection Heads with OOD Detection (Swappable)

*Alternative implementations: attention-based decomposition, linear probes, contrastive methods*

The standard R/S/N projection heads are extended with parallel OOD detection:

```python
class OODAwareProjectionHeads(nn.Module):
    """
    YRSN Decomposition with OOD detection extension.
    Computes R/S/N projections AND distributional alignment (ω).
    """
    def __init__(self, embed_dim, rsn_dim, ood_methods=['mahalanobis', 'energy', 'ensemble']):
        super().__init__()
        # Standard R/S/N projections
        self.W_R = nn.Linear(embed_dim, rsn_dim)
        self.W_S = nn.Linear(embed_dim, rsn_dim)
        self.W_N = nn.Linear(embed_dim, rsn_dim)
        
        # OOD detection components
        self.ood_methods = ood_methods
        
        # Mahalanobis distance components
        self.register_buffer('mu_train', torch.zeros(embed_dim))
        self.register_buffer('cov_inv', torch.eye(embed_dim))
        self.mahal_threshold = 10.0  # Calibrated on validation set
        
        # Energy-based detection
        self.energy_head = nn.Linear(embed_dim, 1)
        self.energy_threshold = 0.0
        
        # Ensemble disagreement
        self.ensemble_heads = nn.ModuleList([
            nn.Linear(embed_dim, rsn_dim) for _ in range(5)
        ])
        self.disagree_threshold = 0.5
        
        # ω threshold for R/S/N reliability
        self.omega_threshold = 0.5
    
    def fit_training_distribution(self, train_embeddings):
        """Fit Mahalanobis parameters on training data."""
        self.mu_train = train_embeddings.mean(dim=0)
        cov = torch.cov(train_embeddings.T)
        # Add small regularization for numerical stability
        cov = cov + 1e-5 * torch.eye(cov.shape[0])
        self.cov_inv = torch.linalg.inv(cov)
    
    def compute_omega(self, x):
        """Compute distributional alignment score (O-score)."""
        scores = []
        
        if 'mahalanobis' in self.ood_methods:
            # Mahalanobis distance to training distribution
            delta = x - self.mu_train
            mahal = torch.sqrt(torch.sum(delta @ self.cov_inv * delta, dim=-1))
            scores.append(torch.sigmoid(-mahal + self.mahal_threshold))
        
        if 'energy' in self.ood_methods:
            # Energy-based OOD detection (Liu et al., 2020)
            energy = -self.energy_head(x).squeeze(-1)
            scores.append(torch.sigmoid(-energy + self.energy_threshold))
        
        if 'ensemble' in self.ood_methods:
            # Ensemble disagreement as OOD signal
            preds = torch.stack([head(x) for head in self.ensemble_heads])
            disagreement = preds.std(dim=0).mean(dim=-1)
            scores.append(torch.sigmoid(-disagreement + self.disagree_threshold))
        
        # Aggregate O-score (conservative: use minimum)
        omega = torch.stack(scores).min(dim=0).values
        return omega
    
    def forward(self, x):
        # First: detect OOD
        omega = self.compute_omega(x)
        
        # Second: compute R/S/N projections
        R = self.W_R(x)
        S = self.W_S(x)
        N = self.W_N(x)
        
        # Compute raw quality score
        R_norm = torch.norm(R, dim=-1)
        S_norm = torch.norm(S, dim=-1)
        N_norm = torch.norm(N, dim=-1)
        alpha = R_norm / (R_norm + S_norm + N_norm + 1e-8)
        
        # Compute distributional-adjusted quality
        alpha_prior = 0.5  # Conservative default
        alpha_omega = omega * alpha + (1 - omega) * alpha_prior
        
        # Compute temperature
        tau = 1.0 / (alpha_omega + 1e-8)
        
        return {
            'R': R, 'S': S, 'N': N,
            'omega': omega,
            'alpha': alpha,
            'alpha_omega': alpha_omega,
            'tau': tau,
            'rsn_reliable': omega > self.omega_threshold
        }
```

### 4.4 OOD Detection: Mahalanobis + CVJF (Swappable)

*Alternative implementations: energy-based, ensemble disagreement, reconstruction error*

BBP thresholds determine decomposition rank. We add distributional phase transitions:

```python
class OODAwarePhaseDetector:
    """
    OOD Detection with OOD-aware phase transitions.
    """
    def __init__(self, curriculum_thresholds=[0.4, 0.6, 0.8], n_train=10000):
        self.curriculum_thresholds = curriculum_thresholds
        # Distributional phase transition threshold
        self.omega_critical = 1 - np.sqrt(1/n_train)
    
    def detect_phase(self, alpha_omega, omega):
        """Determine processing phase based on quality AND distributional alignment."""
        
        # First check: is R/S/N classification reliable?
        if omega < self.omega_critical:
            return 'OOD_HANDLING', {
                'reason': 'Below distributional threshold',
                'omega': omega,
                'omega_critical': self.omega_critical,
                'action': 'Force exploratory mode, flag for review'
            }
        
        # Standard phase detection based on adjusted quality
        if alpha_omega < self.curriculum_thresholds[0]:
            return 'FOUNDATION', {'alpha_omega': alpha_omega}
        elif alpha_omega < self.curriculum_thresholds[1]:
            return 'EXPANSION', {'alpha_omega': alpha_omega}
        elif alpha_omega < self.curriculum_thresholds[2]:
            return 'REFINEMENT', {'alpha_omega': alpha_omega}
        else:
            return 'MASTERY', {'alpha_omega': alpha_omega}
```

### 4.5 Collapse Detection: 10-Type Taxonomy (Theory)

*This is part of the YRSN theory—the detection criteria are fixed, not swappable*

The collapse taxonomy spans three domains with 10 total collapse types:

#### 4.5.1 Extended Collapse Taxonomy (10 Types, 3 Domains)

**Quality Domain** (detected from R/S/N decomposition):

| Collapse Type | Cause | Detection | Remediation |
|---------------|-------|-----------|-------------|
| **POISONING** | Too much N | N > 0.30 | Filter noise sources, re-score |
| **DISTRACTION** | Too much S | S/R > 2.0 | Compress S-content, focus attention |
| **CONFUSION** | High N AND high S | N > 0.20 AND S > 0.25 | Decompose hierarchically |
| **CLASH** | Source S variance | std(S_sources) > 0.15 | Reconcile sources, weight by reliability |

**Reliability Domain** (detected from omega/OOD):

| Collapse Type | Cause | Detection | Remediation |
|---------------|-------|-----------|-------------|
| **HALLUCINATION** | High α + Low ω | α > 0.70 AND ω < 0.50 | Force high τ, human review |
| **O_POISONING** | OOD looks like R | R_norm > 0.50 AND ω < 0.50 | Quarantine source, re-run filtered |
| **DRIFT** | ω declining over time | d(ω)/dt < -threshold | Trigger projection retraining |

**Representation Domain** (VAE-specific, detected by SCP):

| Collapse Type | Cause | Detection | Remediation |
|---------------|-------|-----------|-------------|
| **POSTERIOR_COLLAPSE** | Variance → 0 | var_total < 0.01 | Increase β in L_kl |
| **MODE_COLLAPSE** | No diversity | output_entropy < threshold | Add diversity loss |
| **RSN_COLLAPSE** | R ≈ S ≈ N | cos_sim(R,S,N) > 0.90 | Add diversity constraint |

#### 4.5.2 Collapse Detection Implementation

```python
from enum import Enum
from collections import deque
import numpy as np

class CollapseType(Enum):
    """10 collapse types across 3 domains."""
    NONE = "none"

    # Quality Domain (from R/S/N decomposition)
    POISONING = "poisoning"           # Too much N
    DISTRACTION = "distraction"       # Too much S
    CONFUSION = "confusion"           # High N AND high S
    CLASH = "clash"                   # Source S variance

    # Reliability Domain (from omega/OOD)
    HALLUCINATION = "hallucination"   # High α + Low ω
    O_POISONING = "o_poisoning"       # OOD looks like R
    DRIFT = "drift"                   # ω declining over time

    # Representation Domain (VAE-specific)
    POSTERIOR_COLLAPSE = "posterior"  # Variance → 0
    MODE_COLLAPSE = "mode"            # No diversity
    RSN_COLLAPSE = "rsn"              # R ≈ S ≈ N

class OODCollapseDetector:
    """
    Collapse detection extended with OOD collapse detection.
    """
    def __init__(self, 
                 n_threshold=0.3, 
                 s_ratio_threshold=2.0,
                 omega_threshold=0.5,
                 alpha_threshold=0.7,
                 drift_window=1000,
                 drift_threshold=0.1):
        # Original thresholds
        self.n_threshold = n_threshold
        self.s_ratio_threshold = s_ratio_threshold
        
        # OOD thresholds
        self.omega_threshold = omega_threshold
        self.alpha_threshold = alpha_threshold
        
        # Drift monitoring
        self.omega_history = deque(maxlen=drift_window)
        self.drift_threshold = drift_threshold
    
    def detect_hallucination_risk(self, alpha, omega):
        """Hallucination: high apparent quality but low distributional alignment."""
        hallucination_risk = (alpha > self.alpha_threshold) & (omega < self.omega_threshold)
        risk_score = np.where(
            hallucination_risk,
            (alpha - self.alpha_threshold) * (self.omega_threshold - omega),
            0.0
        )
        return hallucination_risk, risk_score
    
    def update_drift_monitor(self, omega):
        """Track ω over time for drift detection."""
        if isinstance(omega, (np.ndarray, list)):
            self.omega_history.extend(omega if isinstance(omega, list) else omega.tolist())
        else:
            self.omega_history.append(float(omega))
    
    def detect_drift(self):
        """Detect distribution drift via declining ω trend."""
        if len(self.omega_history) < 100:
            return False, 0.0
        
        recent = np.mean(list(self.omega_history)[-100:])
        historical = np.mean(list(self.omega_history)[:-100])
        drift = historical - recent  # Positive = declining omega
        
        return drift > self.drift_threshold, drift
    
    def detect(self, rsn_result, source_omegas=None):
        """
        Comprehensive collapse detection including OOD types.
        
        Args:
            rsn_result: Output from OODAwareProjectionHeads
            source_omegas: Optional per-source ω values
        
        Returns:
            List of detected collapses with metadata
        """
        collapses = []
        
        R_norm = np.linalg.norm(rsn_result['R'], axis=-1)
        S_norm = np.linalg.norm(rsn_result['S'], axis=-1)
        N_norm = np.linalg.norm(rsn_result['N'], axis=-1)
        omega = rsn_result['omega']
        alpha = rsn_result['alpha']
        
        # Update drift monitor
        self.update_drift_monitor(omega)
        
        # --- Original collapse types ---
        
        # POISONING: High N
        if np.any(N_norm > self.n_threshold):
            collapses.append({
                'type': CollapseType.POISONING,
                'severity': float(np.max(N_norm)),
                'indices': np.where(N_norm > self.n_threshold)[0].tolist(),
                'remediation': 'Remove high-N sources, re-score'
            })
        
        # DISTRACTION: High S/R ratio
        s_r_ratio = S_norm / (R_norm + 1e-8)
        if np.any(s_r_ratio > self.s_ratio_threshold):
            collapses.append({
                'type': CollapseType.DISTRACTION,
                'severity': float(np.max(s_r_ratio)),
                'indices': np.where(s_r_ratio > self.s_ratio_threshold)[0].tolist(),
                'remediation': 'Compress S-content, focus attention'
            })
        
        # --- OOD collapse types ---
        
        # HALLUCINATION: High α + Low ω
        hall_risk, hall_score = self.detect_hallucination_risk(alpha, omega)
        if np.any(hall_risk):
            collapses.append({
                'type': CollapseType.HALLUCINATION,
                'severity': float(np.max(hall_score)),
                'indices': np.where(hall_risk)[0].tolist(),
                'remediation': 'Force high temperature, require human verification'
            })
        
        # O-POISONING: OOD content with high apparent R
        o_poison_risk = (omega < self.omega_threshold) & (R_norm > 0.5)
        if np.any(o_poison_risk):
            collapses.append({
                'type': CollapseType.O_POISONING,
                'severity': float(np.mean(R_norm[o_poison_risk])),
                'indices': np.where(o_poison_risk)[0].tolist(),
                'remediation': 'Quarantine OOD content, re-run with filtered context'
            })
        
        # DISTRIBUTION_DRIFT: Gradual ω decline
        drift_detected, drift_rate = self.detect_drift()
        if drift_detected:
            collapses.append({
                'type': CollapseType.DISTRIBUTION_DRIFT,
                'severity': float(drift_rate),
                'indices': [],  # Systemic, not per-item
                'remediation': 'Trigger projection head retraining'
            })
        
        return collapses if collapses else [{'type': CollapseType.NONE}]
```

### 4.6 Memory & Learning (Swappable)

*Alternative implementations: RAG retrieval, transformer memory, Hopfield networks, external databases*

The curriculum progression incorporates progressive OOD exposure:

#### 4.6.1 OOD-Hardened Curriculum Stages

| Stage | Original Focus | OOD Extension |
|-------|----------------|---------------|
| **Foundation** | Basic constraint rules | In-distribution exemplars only |
| **Expansion** | Local constraints | Introduce near-OOD examples (5%) |
| **Refinement** | Global constraints | Handle semantic OOD (15%) |
| **Mastery** | Full integration | Robust to all OOD types (25%) |
| **Hardening** (NEW) | — | Adversarial OOD, edge cases (50%) |

#### 4.6.2 OOD-Hardened Curriculum Implementation

```python
class OODHardenedCurriculum:
    """
    Memory system with progressive OOD exposure.
    """
    def __init__(self, base_curriculum, ood_generator):
        self.base_curriculum = base_curriculum
        self.ood_generator = ood_generator
        self.ood_exposure_schedule = {
            'foundation': 0.0,    # No OOD
            'expansion': 0.05,    # 5% near-OOD
            'refinement': 0.15,   # 15% mixed OOD
            'mastery': 0.25,      # 25% hard OOD
            'hardening': 0.50     # 50% adversarial OOD
        }
    
    def get_batch(self, stage, batch_size):
        ood_fraction = self.ood_exposure_schedule[stage]
        n_ood = int(batch_size * ood_fraction)
        n_id = batch_size - n_ood
        
        # In-distribution from base curriculum
        id_batch = self.base_curriculum.get_batch(stage, n_id)
        
        # OOD examples with labels
        if n_ood > 0:
            ood_batch = self.ood_generator.generate(
                n_ood, 
                difficulty=self._stage_to_ood_difficulty(stage)
            )
            # OOD examples have omega_target = low
            ood_batch['omega_target'] = torch.full((n_ood,), 0.2)
            return self._merge_batches(id_batch, ood_batch)
        
        return id_batch
    
    def _stage_to_ood_difficulty(self, stage):
        mapping = {
            'foundation': 'none',
            'expansion': 'easy',      # Slight distribution shift
            'refinement': 'medium',   # Domain shift
            'mastery': 'hard',        # Significant semantic shift
            'hardening': 'adversarial' # Intentionally crafted
        }
        return mapping[stage]
    
    def _merge_batches(self, id_batch, ood_batch):
        """Merge in-distribution and OOD batches."""
        merged = {}
        for key in id_batch:
            if key in ood_batch:
                merged[key] = torch.cat([id_batch[key], ood_batch[key]], dim=0)
            else:
                merged[key] = id_batch[key]
        return merged


class MemristorOODLearning:
    """
    Memristor-inspired weight updates with OOD-resistant plasticity.
    """
    def __init__(self, base_lr=0.01, ood_dampening=True):
        self.base_lr = base_lr
        self.ood_dampening = ood_dampening
    
    def compute_update(self, gradient, omega):
        """
        Compute OOD-aware weight update.
        
        Low ω reduces learning rate, preventing OOD examples from
        corrupting learned projections.
        
        ΔW = η · ω · ∇L
        """
        if self.ood_dampening:
            effective_lr = self.base_lr * omega
        else:
            effective_lr = self.base_lr
        
        return effective_lr * gradient
```

### 4.7 Memristor Crossbar Projection (Hardware-Faithful)

The hardware-inspired strategies are now backed by actual memristor crossbar arrays for R/S/N projection. This bridges "hardware-inspired naming" to "hardware-faithful implementation."

#### 4.7.1 YRSNMemristorProjection

Three separate memristor crossbar arrays learn to project embeddings onto R/S/N subspaces:

```python
class YRSNMemristorProjection:
    """
    YRSN R/S/N projection using memristor crossbar arrays.

    Conductance G = 1/R encodes learned projection weights.
    Temperature (τ) modulates plasticity via HP Labs dynamics.
    """

    def __init__(self, config: YRSNMemristorConfig):
        # Three crossbar arrays for R/S/N projection
        self.R_array = MemristorArray(config.embed_dim, config.rsn_dim)
        self.S_array = MemristorArray(config.embed_dim, config.rsn_dim)
        self.N_array = MemristorArray(config.embed_dim, config.rsn_dim)

    def forward(self, embedding: np.ndarray) -> Dict[str, Any]:
        """Project through crossbar (O(1) in hardware!)."""
        R_proj = self.R_array.forward(embedding)  # y = G.T @ x
        S_proj = self.S_array.forward(embedding)
        N_proj = self.N_array.forward(embedding)

        # Normalize to R/S/N scores
        R = norm(R_proj) / total
        S = norm(S_proj) / total
        N = norm(N_proj) / total

        return {'R': R, 'S': S, 'N': N, 'alpha': R, 'tau': 1/R}

    def learn(self, embedding, target_R, target_S, target_N, tau):
        """
        Temperature-modulated Hebbian learning.

        High τ (low quality/OOD) → aggressive learning
        Low τ (high quality) → protect learned projections
        """
        self.R_array.apply_hebbian_update(embedding, R_error, tau=tau)
        self.S_array.apply_hebbian_update(embedding, S_error, tau=tau)
        self.N_array.apply_hebbian_update(embedding, N_error, tau=tau)
```

#### 4.7.2 Memristor Dynamics (HP Labs Model)

The virtual memristor implements HP Labs dynamics:

```python
class VirtualMemristor:
    """
    Core equation: dR/dt = μ_v × (R_on/D²) × I × f(R)

    Temperature τ modulates plasticity:
    - High τ (low quality) = more plasticity
    - Low τ (high quality) = less plasticity (EWC-like protection)
    """

    def apply_current(self, I: float, tau: float = 1.0) -> float:
        # Temperature modulates learning rate
        effective_rate = self.drift_rate * tau * self.temperature_coupling

        # HP Labs sinh model with Biolek window
        dR = effective_rate * np.sinh(I) * window_function() * dt

        # Passive decay + noise
        self.R = np.clip(self.R + dR, R_min, R_max)
        return self.R
```

#### 4.7.3 True Bit-Sliced Quantization

Unlike metaphorical "bit-slicing", this implementation uses actual multi-bit quantization:

```python
def quantize_embedding(embedding: np.ndarray, bits: int = 3):
    """
    True bit-slicing: quantize to n-bit representation.
    Like RRAM paper: multiple precision levels for efficiency.
    """
    levels = 2 ** bits  # 3-bit = 8 levels
    scale = max_val / (levels // 2)
    quantized = np.round(embedding / scale)
    return quantized.astype(np.int8)

class BitSlicedMemristorProjection(YRSNMemristorProjection):
    """Project through bit-sliced crossbar."""

    def forward_quantized(self, embedding, slice_weights=[4, 2, 1]):
        quantized, meta = quantize_embedding(embedding, bits=3)

        # Decompose into bit slices and project each
        for i, (bit_slice, weight) in enumerate(zip(slices, slice_weights)):
            R_total += weight * self.R_array.forward(bit_slice)
            S_total += weight * self.S_array.forward(bit_slice)
            N_total += weight * self.N_array.forward(bit_slice)

        return {'R': R, 'S': S, 'N': N, 'quantization': meta}
```

#### 4.7.4 Hardware Claim Update

| Claim | Before Phase 2 | After Memristor Integration |
|-------|---------------|----------------------------|
| **"Memristor dynamics"** | Real but disconnected | **Functional**: HP Labs model drives R/S/N learning |
| **"Crossbar projection"** | Not implemented | **Implemented**: G.T @ x for R/S/N projection |
| **"τ-modulated plasticity"** | Concept only | **Implemented**: `apply_current(I, tau=tau)` |
| **"Bit-slicing"** | Metaphor | **Actual**: 3-bit quantization with slice weights |

The implementation lives in `src/yrsn_context/hardware/memristor_projection.py` with 23 passing tests.

---

## 5. Complete YRSN-OOD Implementation

### 5.1 Unified YRSN-OOD Module

```python
"""
YRSN-OOD: Complete Out-of-Distribution Aware Context Engineering Module
"""

import torch
import torch.nn as nn
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from collections import deque


class OODMethod(Enum):
    MAHALANOBIS = "mahalanobis"
    ENERGY = "energy"
    ENSEMBLE = "ensemble"
    GRADIENT = "gradient"


@dataclass
class YRSNOODConfig:
    """Configuration for YRSN-OOD module."""
    embed_dim: int = 768
    rsn_dim: int = 256
    ood_methods: List[OODMethod] = None
    omega_threshold: float = 0.5
    alpha_prior: float = 0.5
    n_ensemble: int = 5
    curriculum_thresholds: List[float] = None
    drift_window: int = 1000
    drift_threshold: float = 0.1
    
    def __post_init__(self):
        if self.ood_methods is None:
            self.ood_methods = [OODMethod.MAHALANOBIS, OODMethod.ENERGY]
        if self.curriculum_thresholds is None:
            self.curriculum_thresholds = [0.4, 0.6, 0.8]


class YRSN_OOD(nn.Module):
    """
    Complete YRSN-OOD implementation integrating all four pillars
    with OOD detection.
    """
    
    def __init__(self, config: YRSNOODConfig):
        super().__init__()
        self.config = config
        
        # Pillar 1: R/S/N Projections with OOD Detection
        self.W_R = nn.Linear(config.embed_dim, config.rsn_dim)
        self.W_S = nn.Linear(config.embed_dim, config.rsn_dim)
        self.W_N = nn.Linear(config.embed_dim, config.rsn_dim)
        
        # OOD Detection Components
        self.register_buffer('mu_train', torch.zeros(config.embed_dim))
        self.register_buffer('cov_inv', torch.eye(config.embed_dim))
        self.mahal_threshold = nn.Parameter(torch.tensor(10.0))
        
        if OODMethod.ENERGY in config.ood_methods:
            self.energy_head = nn.Linear(config.embed_dim, 1)
            self.energy_threshold = nn.Parameter(torch.tensor(0.0))
        
        if OODMethod.ENSEMBLE in config.ood_methods:
            self.ensemble_heads = nn.ModuleList([
                nn.Linear(config.embed_dim, config.rsn_dim) 
                for _ in range(config.n_ensemble)
            ])
            self.disagree_threshold = nn.Parameter(torch.tensor(0.5))
        
        # Pillar 3: Collapse Detection State
        self.omega_history = deque(maxlen=config.drift_window)
    
    def fit_training_distribution(self, train_embeddings: torch.Tensor):
        """Fit OOD detector on training distribution."""
        with torch.no_grad():
            self.mu_train.copy_(train_embeddings.mean(dim=0))
            cov = torch.cov(train_embeddings.T)
            cov = cov + 1e-5 * torch.eye(cov.shape[0], device=cov.device)
            self.cov_inv.copy_(torch.linalg.inv(cov))
    
    def compute_omega(self, x: torch.Tensor) -> torch.Tensor:
        """Compute distributional alignment score (ω)."""
        scores = []
        
        if OODMethod.MAHALANOBIS in self.config.ood_methods:
            delta = x - self.mu_train
            mahal = torch.sqrt(torch.sum(delta @ self.cov_inv * delta, dim=-1))
            scores.append(torch.sigmoid(-mahal + self.mahal_threshold))
        
        if OODMethod.ENERGY in self.config.ood_methods:
            energy = -self.energy_head(x).squeeze(-1)
            scores.append(torch.sigmoid(-energy + self.energy_threshold))
        
        if OODMethod.ENSEMBLE in self.config.ood_methods:
            preds = torch.stack([head(x) for head in self.ensemble_heads])
            disagreement = preds.std(dim=0).mean(dim=-1)
            scores.append(torch.sigmoid(-disagreement + self.disagree_threshold))
        
        # Conservative aggregation: minimum score
        omega = torch.stack(scores).min(dim=0).values
        return omega
    
    def forward(self, x: torch.Tensor, return_details: bool = False) -> Dict[str, Any]:
        """
        Forward pass computing R/S/N decomposition with OOD awareness.
        
        Args:
            x: Input embeddings [batch_size, embed_dim]
            return_details: Whether to return full diagnostic information
        
        Returns:
            Dictionary with R, S, N projections, omega, alpha_omega, tau, etc.
        """
        # Step 1: OOD Detection
        omega = self.compute_omega(x)
        
        # Step 2: R/S/N Projections
        R = self.W_R(x)
        S = self.W_S(x)
        N = self.W_N(x)
        
        # Step 3: Quality Scores
        R_norm = torch.norm(R, dim=-1)
        S_norm = torch.norm(S, dim=-1)
        N_norm = torch.norm(N, dim=-1)
        
        alpha = R_norm / (R_norm + S_norm + N_norm + 1e-8)
        
        # Step 4: Distributional-Adjusted Quality
        alpha_omega = omega * alpha + (1 - omega) * self.config.alpha_prior
        
        # Step 5: Temperature
        tau = 1.0 / (alpha_omega + 1e-8)
        
        # Step 6: Update drift monitor
        self.omega_history.extend(omega.detach().cpu().numpy().tolist())
        
        result = {
            'R': R,
            'S': S,
            'N': N,
            'omega': omega,
            'alpha': alpha,
            'alpha_omega': alpha_omega,
            'tau': tau,
            'rsn_reliable': omega > self.config.omega_threshold
        }
        
        if return_details:
            result['R_norm'] = R_norm
            result['S_norm'] = S_norm
            result['N_norm'] = N_norm
        
        return result
    
    def detect_collapses(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect collapse conditions including OOD types."""
        collapses = []
        
        omega = result['omega'].detach().cpu().numpy()
        alpha = result['alpha'].detach().cpu().numpy()
        R_norm = result.get('R_norm', torch.norm(result['R'], dim=-1)).detach().cpu().numpy()
        N_norm = result.get('N_norm', torch.norm(result['N'], dim=-1)).detach().cpu().numpy()
        S_norm = result.get('S_norm', torch.norm(result['S'], dim=-1)).detach().cpu().numpy()
        
        # POISONING
        if np.any(N_norm > 0.3):
            collapses.append({
                'type': 'POISONING',
                'severity': float(np.max(N_norm)),
                'remediation': 'Remove high-N sources, re-score'
            })
        
        # DISTRACTION
        s_r_ratio = S_norm / (R_norm + 1e-8)
        if np.any(s_r_ratio > 2.0):
            collapses.append({
                'type': 'DISTRACTION',
                'severity': float(np.max(s_r_ratio)),
                'remediation': 'Compress S-content, focus attention'
            })
        
        # HALLUCINATION
        hall_risk = (alpha > 0.7) & (omega < self.config.omega_threshold)
        if np.any(hall_risk):
            collapses.append({
                'type': 'HALLUCINATION',
                'severity': float(np.max((alpha - 0.7) * (self.config.omega_threshold - omega))),
                'remediation': 'Force high temperature, require human verification'
            })
        
        # O-POISONING
        o_poison = (omega < self.config.omega_threshold) & (R_norm > 0.5)
        if np.any(o_poison):
            collapses.append({
                'type': 'O_POISONING',
                'severity': float(np.mean(R_norm[o_poison])),
                'remediation': 'Quarantine OOD content, re-run filtered'
            })
        
        # DISTRIBUTION_DRIFT
        if len(self.omega_history) >= 100:
            recent = np.mean(list(self.omega_history)[-100:])
            historical = np.mean(list(self.omega_history)[:-100]) if len(self.omega_history) > 100 else recent
            drift = historical - recent
            if drift > self.config.drift_threshold:
                collapses.append({
                    'type': 'DISTRIBUTION_DRIFT',
                    'severity': float(drift),
                    'remediation': 'Trigger projection head retraining'
                })
        
        return collapses if collapses else [{'type': 'NONE'}]
    
    def get_processing_recommendation(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Get processing recommendations based on analysis."""
        recommendations = []
        collapses = self.detect_collapses(result)
        
        for collapse in collapses:
            if collapse['type'] != 'NONE':
                recommendations.append({
                    'type': 'COLLAPSE_DETECTED',
                    'collapse': collapse['type'],
                    'action': collapse['remediation'],
                    'severity': collapse.get('severity', 0)
                })
        
        # Temperature-based compute allocation
        tau = result['tau'].mean().item()
        omega = result['omega'].mean().item()
        
        if omega < self.config.omega_threshold:
            recommendations.append({
                'type': 'OOD_DETECTED',
                'action': 'Flag for human review, increase exploration',
                'reason': f'Low distributional alignment (ω={omega:.2f})'
            })
        
        if tau > 3.0:
            recommendations.append({
                'type': 'COMPUTE_ALLOCATION',
                'action': 'Use exploratory search, multiple samples',
                'reason': f'High uncertainty (τ={tau:.2f})'
            })
        elif tau < 1.5:
            recommendations.append({
                'type': 'COMPUTE_ALLOCATION',
                'action': 'Use deterministic inference',
                'reason': f'High confidence (τ={tau:.2f})'
            })
        
        return {
            'omega': omega,
            'tau': tau,
            'alpha_omega': result['alpha_omega'].mean().item(),
            'rsn_reliable': result['rsn_reliable'].all().item(),
            'recommendations': recommendations
        }


# =============================================================================
# Example Usage
# =============================================================================

def example_usage():
    """Demonstrate YRSN-OOD module usage."""
    
    # Configuration
    config = YRSNOODConfig(
        embed_dim=768,
        rsn_dim=256,
        ood_methods=[OODMethod.MAHALANOBIS, OODMethod.ENERGY],
        omega_threshold=0.5,
    )
    
    # Initialize module
    yrsn_ood = YRSN_OOD(config)
    
    # Simulate training distribution fitting
    train_embeddings = torch.randn(1000, 768)
    yrsn_ood.fit_training_distribution(train_embeddings)
    
    # Process in-distribution input
    id_input = torch.randn(32, 768)
    id_result = yrsn_ood(id_input, return_details=True)
    print("In-distribution results:")
    print(f"  ω (omega): {id_result['omega'].mean():.3f}")
    print(f"  α_ω: {id_result['alpha_omega'].mean():.3f}")
    print(f"  τ (temperature): {id_result['tau'].mean():.3f}")
    
    # Process OOD input
    ood_input = torch.randn(32, 768) * 5 + 10  # Different distribution
    ood_result = yrsn_ood(ood_input, return_details=True)
    print("\nOut-of-distribution results:")
    print(f"  ω (omega): {ood_result['omega'].mean():.3f}")
    print(f"  α_ω: {ood_result['alpha_omega'].mean():.3f}")
    print(f"  τ (temperature): {ood_result['tau'].mean():.3f}")
    
    # Get recommendations
    recommendations = yrsn_ood.get_processing_recommendation(ood_result)
    print("\nRecommendations for OOD input:")
    for rec in recommendations['recommendations']:
        print(f"  [{rec['type']}] {rec['action']}")


if __name__ == "__main__":
    example_usage()
```

---

## 6. YRSN-OOD Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         YRSN-OOD Processing Pipeline                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input Context Y                                                            │
│        │                                                                    │
│        ▼                                                                    │
│  ┌─────────────────────┐                                                    │
│  │  OOD Detection      │ ──────────────────────────┐                        │
│  │   (compute ω)       │                           │                        │
│  │   • Mahalanobis     │                           ▼                        │
│  │   • Energy-based    │              ┌────────────────────────┐            │
│  │   • Ensemble        │              │   OOD Handling Path    │            │
│  └─────────┬───────────┘              │   (if ω < threshold)   │            │
│            │                          │   • Flag for review    │            │
│            │ ω                        │   • Force high τ       │            │
│            │                          │   • Log & alert        │            │
│            ▼                          │   • Surface reasoning  │            │
│  ┌─────────────────────┐              └────────────────────────┘            │
│  │  R/S/N Decomposition│                           │                        │
│  │  (Pillar 1)         │                           │                        │
│  │   • W_R projection  │                           │                        │
│  │   • W_S projection  │                           │                        │
│  │   • W_N projection  │                           │                        │
│  └─────────┬───────────┘                           │                        │
│            │                                       │                        │
│            ▼                                       │                        │
│  ┌─────────────────────┐                           │                        │
│  │ Quality Score       │◄──────────────────────────┘                        │
│  │  α = R/(R+S+N)      │                                                    │
│  │  α_ω = ω·α +        │                                                    │
│  │       (1-ω)·α_prior │                                                    │
│  └─────────┬───────────┘                                                    │
│            │                                                                │
│            ▼                                                                │
│  ┌─────────────────────┐                                                    │
│  │ Temperature         │                                                    │
│  │  τ = 1/α_ω          │                                                    │
│  └─────────┬───────────┘                                                    │
│            │                                                                │
│            ├──────────────────┬──────────────────┐                          │
│            ▼                  ▼                  ▼                          │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────────┐          │
│  │ Phase Detection │  │ Collapse Check   │  │ Drift Monitoring  │          │
│  │ (Pillar 2)      │  │ (Pillar 3)       │  │                   │          │
│  │ • Foundation    │  │ Quality Domain:  │  │ • ω trend         │          │
│  │ • Expansion     │  │  POISONING       │  │ • Retraining      │          │
│  │ • Refinement    │  │  DISTRACTION     │  │   triggers        │          │
│  │ • Mastery       │  │  CONFUSION       │  │                   │          │
│  │ • OOD_HANDLING  │  │  CLASH           │  │                   │          │
│  └────────┬────────┘  │ Reliability:     │  └───────────────────┘          │
│           │           │  HALLUCINATION   │                                  │
│           │           │  O_POISONING     │                                  │
│           │           │  DRIFT           │                                  │
│           │           │ Representation:  │                                  │
│           │           │  POSTERIOR_COLL  │                                  │
│           │           │  MODE_COLLAPSE   │                                  │
│           │           │  RSN_COLLAPSE    │                                  │
│           │           └────────┬─────────┘                                  │
│           │                    │                                            │
│           ▼                    ▼                                            │
│  ┌──────────────────────────────────────────────────────────────┐          │
│  │              Adaptive Processing & Output                     │          │
│  │  • Strategy selection (5 strategies + OOD handling)          │          │
│  │  • Collapse remediation (10 types across 3 domains)          │          │
│  │  • Compute allocation (based on τ)                           │          │
│  │  • Output confidence calibration (based on ω)                │          │
│  │  • Recommendations for downstream processing                  │          │
│  └──────────────────────────────────────────────────────────────┘          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Experimental Design for Validation

### 7.1 OOD Detection Accuracy

**Metrics:**
- AUROC for ω-based OOD detection
- FPR95: False positive rate at 95% true positive rate
- Detection delay for distribution drift

**Datasets:**
1. In-distribution: Domain-specific training corpus
2. Semantic OOD: Adjacent domains (e.g., medical → legal)
3. Structural OOD: Format variations (JSON → XML → natural language)
4. Adversarial OOD: Crafted prompt injections

### 7.2 Hallucination Prevention

**Hypothesis:** YRSN-OOD reduces hallucination rate by detecting when R/S/N scores are unreliable.

**Experiment:**
1. Baseline: Standard YRSN without OOD detection
2. Treatment: YRSN-OOD with ω-based confidence adjustment
3. Measure: Hallucination rate on held-out OOD test set

### 7.3 Curriculum Learning Improvement

**Hypothesis:** OOD-hardened curriculum improves robustness without sacrificing in-distribution performance.

**Experiment:**
1. Train YRSN with standard curriculum
2. Train YRSN-OOD with OOD-hardened curriculum
3. Evaluate on both in-distribution and OOD test sets

---

## 8. Connection to Broader Research

### 8.1 Relationship to Uncertainty Quantification

YRSN-OOD extends beyond point estimates to provide calibrated uncertainty:

| Approach | Uncertainty Type | YRSN-OOD Integration |
|----------|------------------|---------------------|
| MC Dropout | Epistemic | Ensemble disagreement in ω |
| Deep Ensembles | Epistemic | Multi-head variance |
| Temperature Scaling | Calibration | τ = 1/α_ω |
| Conformal Prediction | Distribution-free | ω as conformity score |

### 8.2 Relationship to Robust ML

YRSN-OOD provides a **context-level** robustness layer complementing model-level approaches:

| Model-Level | Context-Level (YRSN-OOD) |
|-------------|-------------------------|
| Adversarial training | OOD-hardened curriculum |
| Certified defenses | ω-based input filtering |
| Robust optimization | Quality-adjusted loss |

### 8.3 Connection to LLM Safety

OOD detection is critical for LLM safety:
- Prompt injection often manifests as OOD context
- Jailbreaks exploit distributional blind spots
- YRSN-OOD provides a principled detection layer

---

## 9. Discussion

### 9.1 Unique YRSN-OOD Contributions

| Capability | YRSN-OOD | NL/Titans | Standard YRSN |
|------------|----------|-----------|---------------|
| Three-way decomposition | ✓ R/S/N | ✗ Binary | ✓ R/S/N |
| OOD detection | ✓ ω-score | ✗ | ✗ |
| Distributional-adjusted quality | ✓ α_ω | ✗ | ✗ |
| Hallucination prevention | ✓ Explicit | ✗ | ✗ |
| Collapse taxonomy | ✓ 10 types / 3 domains | ✗ (1 type) | ✓ (4 types) |
| OOD-hardened curriculum | ✓ | ✗ | ✗ |
| External orchestration | ✓ | ✗ | ✓ |
| Production focus | ✓ | ✗ | ✓ |

### 9.2 Limitations

1. **Computational overhead**: OOD detection adds latency (~15-30% increase)
2. **Calibration requirement**: Mahalanobis detector requires training distribution fit
3. **Threshold sensitivity**: ω_threshold requires domain-specific tuning
4. **Drift detection delay**: Distribution drift detected after ~100 samples

### 9.3 Positioning in the Post-Hype Era

YRSN-OOD addresses the critical gap between impressive demos and reliable production:

| Demo Success | Production Challenge | YRSN-OOD Solution |
|--------------|---------------------|-------------------|
| High benchmark accuracy | OOD hallucinations | ω-based detection |
| Clean evaluation data | Noisy real-world input | R/S/N + O decomposition |
| Static test sets | Distribution drift | Drift monitoring |
| Single-source context | Multi-source chaos | Per-source ω calibration |

---

## 10. Conclusion

We have presented YRSN-OOD, an extension of the YRSN framework that addresses the critical challenge of Out-of-Distribution context in production AI systems.

### 10.1 Summary of Contributions

1. **Extended decomposition** Y = R + S + N + O with OOD as meta-dimension
2. **Distributional-adjusted quality** α_ω accounting for R/S/N reliability
3. **Multi-method OOD detection** integrated with Pillar 1
4. **10 collapse types across 3 domains** (Quality, Reliability, Representation) with remediation strategies
5. **OOD-hardened curriculum** for Pillar 4
6. **Complete implementation** with production-ready code

### 10.2 The Core Insight

> We don't treat uncertainty as noise—we treat it as a thermodynamic variable.
> And we don't assume our uncertainty estimates are reliable—we measure their distributional alignment.

### 10.3 Future Work

1. **Theoretical guarantees** for ω-based detection
2. **Adaptive thresholds** that learn from feedback
3. **Integration with Titans/ATLAS** memory architectures
4. **Real-time OOD monitoring** in production systems
5. **Benchmark creation** for OOD context detection

---

## References

### Core YRSN Citations

- Candès, E., et al. (2011). "Robust Principal Component Analysis?" *JACM*.
- Baik, J., Ben Arous, G., Péché, S. (2005). "Phase transition of the largest eigenvalue." *Annals of Probability*.

### OOD Detection Citations

- Liu, W., et al. (2020). "Energy-based Out-of-distribution Detection." *NeurIPS*.
- Hendrycks, D., & Gimpel, K. (2017). "A Baseline for Detecting Misclassified and Out-of-Distribution Examples." *ICLR*.
- Lee, K., et al. (2018). "A Simple Unified Framework for Detecting Out-of-Distribution Samples." *NeurIPS*.
- Lakshminarayanan, B., et al. (2017). "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." *NeurIPS*.

### Nested Learning & Memory Citations

- Behrouz, A., Razaviyayn, M., Zhong, P., & Mirrokni, V. (2025). "Nested Learning: The Illusion of Deep Learning Architectures." *NeurIPS 2025*.
- Behrouz, A., Zhong, P., & Mirrokni, V. (2024). "Titans: Learning to Memorize at Test Time." *arXiv:2501.00663*.
- Behrouz, A., et al. (2025). "ATLAS: Learning to Optimally Memorize the Context at Test Time." *arXiv:2505.23735*.

### Product Era Context Citations

- Douglas Heaven, W. (2025). "The Great AI Hype Correction of 2025." *MIT Technology Review*.
- Kilcher, Y. (2025). "GPT-5 Analysis." *YouTube*.
- Sutskever, I. (2025). "Interview with Dwarkesh Patel." *YouTube*.

### Additional Citations

[Additional citations from original documents preserved]

---

## Appendix A: Mathematical Details

### A.1 Distributional Phase Transition Derivation

The critical ω threshold is derived from concentration inequalities on the projection error. For a training set of size n, the expected estimation error for the Mahalanobis distance is O(1/√n), leading to:

$$\omega_{\text{critical}} = 1 - \sqrt{\frac{1}{n}}$$

### A.2 Temperature-Quality-Distribution Triad Properties

The extended coupling τ = 1/α_ω satisfies:
- Monotonicity: τ increases as either α or ω decreases
- Boundedness: τ ∈ [1/max(α), 1/α_prior] for typical ranges
- Continuity: Smooth interpolation between in-distribution and OOD regimes

---

## Appendix B: Implementation Details

### B.1 Hyperparameter Recommendations

| Parameter | Default | Range | Tuning Guidance |
|-----------|---------|-------|-----------------|
| omega_threshold | 0.5 | [0.3, 0.7] | Higher for safety-critical |
| alpha_prior | 0.5 | [0.3, 0.6] | Lower for conservative |
| mahal_threshold | 10.0 | [5.0, 20.0] | Calibrate on validation |
| drift_window | 1000 | [500, 5000] | Larger for stability |
| drift_threshold | 0.1 | [0.05, 0.2] | Smaller for sensitivity |

### B.2 Computational Complexity

| Component | Complexity | Notes |
|-----------|------------|-------|
| R/S/N projection | O(d·r) | d=embed_dim, r=rsn_dim |
| Mahalanobis | O(d²) | Dominated by cov_inv |
| Energy | O(d) | Single linear layer |
| Ensemble | O(k·d·r) | k=n_ensemble |
| Drift detection | O(w) | w=window_size |

---

## Appendix C: Document Sources

This merged paper synthesizes content from:
- `YRSN_PAPER_v2025121e.md` (core framework, 983 lines)
- `yrsn_ood_context_analysis.md` (OOD extension, 1126 lines)

Primary source repositories:
- yrsn-context (primary)
- yrsn-omnibus (theory)
- yrsn-1210 (curriculum learning)

---

## Appendix D: Convergence Analysis

### D.1 Independent Development Paths

| Aspect | YRSN Path | Nested Learning Path | OOD Extension Path |
|--------|-----------|---------------------|-------------------|
| **Origin** | Robust PCA extension (Nov 2024) | Neurophysiology + optimization | Production deployment failures |
| **Core insight** | Three-way decomposition (R/S/N) | Nested optimization levels | Distributional reliability |
| **Temperature** | Memristor coupling (Dec 2024) | Inverse update frequency | Quality-distribution triad |
| **Primary focus** | External orchestration | Internal architecture | Hallucination prevention |

The convergence on multi-scale, quality-modulated, OOD-aware learning suggests these principles are fundamental to reliable AI deployment.

---

*This document represents version v2025121f-OOD of the YRSN framework, integrating the core paper with OOD detection extensions for production-ready context engineering.*
