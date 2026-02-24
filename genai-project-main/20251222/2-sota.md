et me clarify each one:

  1. Per-Sample Uncertainty: What Is It?

  Current state: The system outputs a single confidence score per prediction (softmax max). This tells you "how sure the model is" but doesn't distinguish why it's unsure.

  What uncertainty adds: Two distinct signals per sample:

  | Type      | Meaning              | Source                | When High           |
  |-----------|----------------------|-----------------------|---------------------|
  | Epistemic | "Model doesn't know" | Lack of training data | Novel/OOD inputs    |
  | Aleatoric | "Data is ambiguous"  | Inherent noise        | Overlapping classes |

  Already exists in your codebase (supervised_heads.py:573-596):
  # MC Dropout for epistemic uncertainty
  uncertainty = mc_probs.var(dim=0).sum(dim=-1)  # Variance across samples
  entropy = -(mean_probs * torch.log(mean_probs)).sum(dim=-1)  # Predictive entropy

  Why it matters for YRSN:
  - High epistemic uncertainty → memristor hasn't learned this region → train more
  - High aleatoric uncertainty → R/S/N boundaries genuinely overlap → accept confusion
  - Could modulate τ: τ_effective = τ * (1 + epistemic_uncertainty)

  It IS just an extra output - but one that enables smarter decisions downstream.

  ---
  2. Mining Curriculum vs Current Curriculum

  These are two different things operating at different levels:

  Current Curriculum (constraint-based)

  What it does: Controls which constraints are active during training
  Stage 1: Foundation  → Only EMERGENT constraints active
  Stage 2: Expansion   → Add CONSOLIDATED constraints
  Stage 3: Refinement  → Add LEARNED constraints
  Stage 4: Mastery     → All constraints including ACTUAL
  Level: Task/constraint level - "what rules apply"

  Mining Curriculum (triplet-based)

  What it does: Controls which negative examples are selected for contrastive learning
  Stage 1: EASY       → Pick negatives far from anchor (easy to distinguish)
  Stage 2: SEMI_HARD  → Pick negatives within margin (challenging but learnable)
  Stage 3: HARD       → Pick closest negatives (hardest cases)
  Level: Sample/batch level - "what examples to compare"

  Why Both Matter

  | Aspect       | Constraint Curriculum          | Mining Curriculum                  |
  |--------------|--------------------------------|------------------------------------|
  | Controls     | Which rules to enforce         | Which triplets to form             |
  | Metaphor     | "What subjects to teach"       | "What practice problems to assign" |
  | Progression  | Relaxed → Strict rules         | Easy → Hard comparisons            |
  | Failure mode | Too strict early = can't learn | Hard negatives early = collapse    |

  The gap: Your codebase has constraint curriculum but triplet mining is fixed at SEMI_HARD. A mining curriculum would schedule:
  Epoch 0-10:  MiningStrategy.EASY      (warm-up)
  Epoch 10-30: MiningStrategy.SEMI_HARD (main learning)
  Epoch 30+:   MiningStrategy.HARD      (fine-tuning)

  This is additive, not replacement - they're orthogonal dimensions of curriculum.

  ---
  3. WHY Replace vs ADD for Drift Detection?

  You're right to question this. Let me reconsider:

  Current Heuristic Detection (3 mechanisms)

  # 1. Muddy: All R/S/N near 0.33
  is_muddy = max(|R-0.33|, |S-0.33|, |N-0.33|) < 0.15

  # 2. Confusion streak: Repeated CONFUSION states
  confusion_triggered = confusion_streak >= 5

  # 3. Low variance: Stale predictions
  low_variance = variance(recent_values) < 0.02

  Statistical Tests (Kolmogorov-Smirnov)

  # Compare recent distribution to reference distribution
  ks_statistic, p_value = ks_2samp(recent_R_values, reference_R_values)
  drift_detected = p_value < 0.05  # Significant distribution shift

  The Real Answer: COMPLEMENT, Not Replace

  | Heuristic        | Strength                   | Weakness                              |
  |------------------|----------------------------|---------------------------------------|
  | Muddy detection  | Fast, intuitive            | Arbitrary threshold (0.15)            |
  | Confusion streak | Catches sustained problems | Misses gradual drift                  |
  | Low variance     | Catches stale model        | False positives on stable good states |

  | Statistical Test | Strength                    | Weakness                           |
  |------------------|-----------------------------|------------------------------------|
  | KS test          | Mathematically principled   | Requires sufficient samples (~50+) |
  | Wasserstein      | Captures magnitude of shift | Computationally heavier            |
  | CUSUM            | Good for gradual drift      | Needs tuning                       |

  Better architecture:
  Drift Detection
  ├── Fast Path (heuristics) ← Keep these, they're cheap
  │   ├── Muddy check (O(1))
  │   ├── Streak check (O(1))
  │   └── Variance check (O(n) where n=20)
  │
  └── Slow Path (statistical) ← ADD this as confirmation
      ├── KS test on R distribution
      ├── KS test on S distribution
      └── KS test on N distribution

  Hybrid strategy:
  1. Heuristics trigger suspicion (cheap, always running)
  2. Statistical tests confirm drift (expensive, only when suspicious)
  3. Only recalibrate if BOTH agree

  This avoids:
  - False positives from heuristics alone
  - Computational cost of running KS on every sample
  - The "replace everything" trap

  Why I Initially Said "Replace"

  That was sloppy. The NeurIPS 2025 trend is toward principled uncertainty quantification, which made me overcorrect. But your heuristics are:
  - Computationally free
  - Domain-specific (tuned for R/S/N)
  - Battle-tested in your codebase

  The right answer is layered detection: heuristics for speed, statistics for confirmation.

  ---
  Summary

  | Gap               | Resolution                                                    | Complexity                    |
  |-------------------|---------------------------------------------------------------|-------------------------------|
  | Uncertainty       | Already implemented, just expose it                           | Low - wire up existing code   |
  | Mining curriculum | Orthogonal to constraint curriculum, schedule mining strategy | Medium - add scheduler        |
  | Drift detection   | ADD statistical confirmation, don't replace heuristics        | Medium - layer the approaches |

  Want me to go deeper on any of these?

  Todos
  [ ] Update test_synthetic_routing_memristor.py to use existing SOTA modules
  [ ] Add per-sample uncertainty (epistemic/aleatoric scores)
  [ ] Implement mining curriculum (EASY -> SEMI_HARD -> HARD)
  [ ] Replace heuristic drift detection with statistical tests (KS)
