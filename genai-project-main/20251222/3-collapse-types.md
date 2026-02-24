 Current Collapse Types (10 total, 3 domains)

  | Domain         | Type               | Detection Signal           |
  |----------------|--------------------|----------------------------|
  | Quality        | POISONING          | N > 0.30                   |
  |                | DISTRACTION        | S > 0.40                   |
  |                | CONFUSION          | N > 0.20 AND S > 0.25      |
  |                | CLASH              | variance(S_sources) > 0.15 |
  | Reliability    | HALLUCINATION      | high α + low ω             |
  |                | O_POISONING        | high R + low ω             |
  |                | DRIFT              | ω declining over time      |
  | Representation | POSTERIOR_COLLAPSE | variance → 0               |
  |                | MODE_COLLAPSE      | no diversity               |
  |                | RSN_COLLAPSE       | R ≈ S ≈ N                  |

  ---
  NEW Collapse Types from Proposed Additions

  1. From Per-Sample Uncertainty (Epistemic/Aleatoric)

  | New Type              | Detection Signal                                  | Meaning                                        |
  |-----------------------|---------------------------------------------------|------------------------------------------------|
  | OVERCONFIDENCE        | low uncertainty + low α                           | Model is certain but wrong                     |
  | EPISTEMIC_SPIKE       | sudden epistemic ↑                                | Novel/OOD input arrived                        |
  | ALEATORIC_DOMINANCE   | aleatoric >> epistemic                            | Data inherently ambiguous, not learnable       |
  | UNCERTAINTY_INVERSION | high aleatoric + low epistemic + wrong prediction | Model thinks it knows, but data says otherwise |

  Why these matter:
  Current HALLUCINATION: high α + low ω
      → "Confident but unreliable source"

  New OVERCONFIDENCE: low uncertainty + low α
      → "Model is certain despite poor quality"
      → This is WORSE - model doesn't know it doesn't know

  Relationship to existing:
  - HALLUCINATION detects source unreliability (ω)
  - OVERCONFIDENCE detects model unreliability (uncertainty)
  - They're orthogonal - you can have both simultaneously

  ---
  2. From Mining Curriculum (EASY → SEMI_HARD → HARD)

  | New Type             | Detection Signal              | Meaning                        |
  |----------------------|-------------------------------|--------------------------------|
  | TRIPLET_COLLAPSE     | no valid triplets formable    | All embeddings too similar     |
  | EMBEDDING_DEGENERACY | all mining strategies fail    | Representation space collapsed |
  | CURRICULUM_STALL     | can't progress to HARD mining | Learning has plateaued         |
  | MARGIN_EROSION       | margin violations increasing  | Subspace boundaries dissolving |

  Why these matter:
  Current RSN_COLLAPSE: R ≈ S ≈ N (output-level)
      → Detected AFTER collapse happens

  New TRIPLET_COLLAPSE: no valid triplets (embedding-level)
      → Detected BEFORE R/S/N outputs equalize
      → Early warning system

  Relationship to existing:
  - RSN_COLLAPSE is output collapse (R/S/N values converge)
  - TRIPLET_COLLAPSE is embedding collapse (representations converge)
  - Embedding collapse CAUSES output collapse - earlier detection

  Curriculum stall as diagnostic:
  If EASY mining works but SEMI_HARD fails:
      → Embeddings separated but not refined
      → Prescribes more training, not recalibration

  If even EASY mining fails:
      → Fundamental embedding collapse
      → Prescribes architecture change

  ---
  3. From Statistical Drift Detection (KS Tests)

  | New Type               | Detection Signal                              | Meaning                         |
  |------------------------|-----------------------------------------------|---------------------------------|
  | DISTRIBUTIONAL_SHIFT   | KS p-value < 0.05                             | Significant distribution change |
  | GRADUAL_DRIFT          | Wasserstein distance increasing monotonically | Slow distribution migration     |
  | MULTI_MODAL_SPLIT      | bimodality test positive                      | Single mode → multiple modes    |
  | REFERENCE_INVALIDATION | all R/S/N distributions shifted               | Reference distribution obsolete |

  Why these matter:
  Current DRIFT: ω declining over time
      → Heuristic detection of reliability decay
      → Can miss gradual distributional changes

  New DISTRIBUTIONAL_SHIFT: KS test on R/S/N
      → Statistically principled detection
      → Catches shifts heuristics miss

  Layered detection architecture:
  Level 1 (heuristic, always on):
      - Muddy detection: max(|R-0.33|, ...) < 0.15
      - Confusion streak: confusion_count >= 5
      - Low variance: var(recent) < 0.02

  Level 2 (statistical, when Level 1 triggers):
      - KS test on R distribution vs reference
      - KS test on S distribution vs reference
      - KS test on N distribution vs reference

  Detection = Level1_suspicion AND Level2_confirmation

  New composite collapse:
  CONFIRMED_DRIFT = (heuristic_drift OR heuristic_muddy) AND (KS_p_value < 0.05)
      → Higher confidence than either alone
      → Fewer false positives

  ---
  Collapse Type Taxonomy (Extended)

  CollapseType (Extended)
  ├── Quality Domain (from R/S/N)
  │   ├── POISONING
  │   ├── DISTRACTION
  │   ├── CONFUSION
  │   └── CLASH
  │
  ├── Reliability Domain (from ω)
  │   ├── HALLUCINATION
  │   ├── O_POISONING
  │   └── DRIFT
  │
  ├── Representation Domain (VAE)
  │   ├── POSTERIOR_COLLAPSE
  │   ├── MODE_COLLAPSE
  │   └── RSN_COLLAPSE
  │
  ├── Uncertainty Domain (NEW)          ← From per-sample uncertainty
  │   ├── OVERCONFIDENCE
  │   ├── EPISTEMIC_SPIKE
  │   ├── ALEATORIC_DOMINANCE
  │   └── UNCERTAINTY_INVERSION
  │
  ├── Embedding Domain (NEW)            ← From mining curriculum
  │   ├── TRIPLET_COLLAPSE
  │   ├── EMBEDDING_DEGENERACY
  │   ├── CURRICULUM_STALL
  │   └── MARGIN_EROSION
  │
  └── Distributional Domain (NEW)       ← From statistical tests
      ├── DISTRIBUTIONAL_SHIFT
      ├── GRADUAL_DRIFT
      ├── MULTI_MODAL_SPLIT
      └── REFERENCE_INVALIDATION

  ---
  Cross-Domain Collapse Detection

  The new types enable composite detection that wasn't possible before:

  | Composite      | Components                            | Detection                             |
  |----------------|---------------------------------------|---------------------------------------|
  | SILENT_FAILURE | low uncertainty + CONFUSION           | Model confident during confusion      |
  | LEARNING_DEATH | CURRICULUM_STALL + RSN_COLLAPSE       | Can't learn AND outputs collapsed     |
  | TOPIC_AMNESIA  | DISTRIBUTIONAL_SHIFT + high epistemic | Context shifted AND model knows it    |
  | FALSE_RECOVERY | DRIFT resolved + OVERCONFIDENCE       | Drift "fixed" but model overconfident |

  ---
  Summary: New Detection Capabilities

  | Addition          | New Collapse Types | Detection Improvement                          |
  |-------------------|--------------------|------------------------------------------------|
  | Uncertainty       | 4 new types        | Detect MODEL failures (not just DATA failures) |
  | Mining Curriculum | 4 new types        | EARLIER detection at embedding level           |
  | Statistical Tests | 4 new types        | CONFIRMED detection with fewer false positives |

  Total: 12 new collapse types, extending the taxonomy from 10 to 22.

  ---
  Shall I proceed with implementing these as code changes with detailed comments explaining each new collapse type?

  Todos
  [ ] Update test_synthetic_routing_memristor.py to use existing SOTA modules
  [ ] Add per-sample uncertainty (epistemic/aleatoric scores)
  [ ] Implement mining curriculum (EASY -> SEMI_HARD -> HARD)
  [ ] Add statistical drift detection (complement heuristics with KS)
