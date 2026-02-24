  MAST → YRSN Coverage Matrix

  | MAST Failure                 | Required YRSN Collapse              | Current Status      | Implementation Required         |
  |------------------------------|-------------------------------------|---------------------|---------------------------------|
  | FM-1.1 Disobey Task          | DISTRACTION, O_POISONING            | ⚠️ DISTRACTION only | Add O_POISONING (omega)         |
  | FM-1.2 Disobey Role          | DISTRACTION                         | ✅ Works            | -                               |
  | FM-1.3 Step Repetition       | MODE_COLLAPSE                       | ❌ Not implemented  | Add MODE_COLLAPSE (diversity)   |
  | FM-1.4 Loss History          | DRIFT, DISTRIBUTIONAL_SHIFT         | ❌ Not implemented  | Add DRIFT (omega trend)         |
  | FM-1.5 Unaware Termination   | CONFUSION, OVERCONFIDENCE           | ⚠️ CONFUSION only   | Add OVERCONFIDENCE (epistemic)  |
  | FM-2.1 Conversation Reset    | DRIFT, RSN_COLLAPSE                 | ❌ Not implemented  | Add DRIFT + RSN_COLLAPSE        |
  | FM-2.2 Fail Clarify          | OVERCONFIDENCE, ALEATORIC_DOMINANCE | ❌ Not implemented  | Add both (epistemic/aleatoric)  |
  | FM-2.3 Task Derailment       | DISTRACTION, CONFUSION              | ✅ Works            | -                               |
  | FM-2.4 Info Withholding      | CLASH                               | ✅ Works            | -                               |
  | FM-2.5 Ignored Input         | POISONING, CLASH                    | ✅ Works            | -                               |
  | FM-2.6 Reasoning Mismatch    | HALLUCINATION                       | ❌ Not implemented  | Add HALLUCINATION (omega+alpha) |
  | FM-3.1 Premature Termination | EPISTEMIC_SPIKE                     | ❌ Not implemented  | Add EPISTEMIC_SPIKE             |
  | FM-3.2 Incomplete Verify     | OVERCONFIDENCE                      | ❌ Not implemented  | Add OVERCONFIDENCE              |
  | FM-3.3 Incorrect Verify      | HALLUCINATION                       | ❌ Not implemented  | Add HALLUCINATION               |

  Critical Path for MAST Compliance

  Must Implement (P0 - Blocks MAST Testing):
  1. HALLUCINATION - Required for FM-2.6, FM-3.3 (alpha > 0.7 AND omega < 0.5)
  2. DRIFT - Required for FM-1.4, FM-2.1 (omega < 0.35)
  3. OVERCONFIDENCE - Required for FM-1.5, FM-2.2, FM-3.2 (epistemic < 0.1 AND alpha < 0.4)
  4. RSN_COLLAPSE - Required for FM-2.1 (|R-S| < 0.1 AND |S-N| < 0.1)

  Should Implement (P1 - Improves Coverage):
  5. O_POISONING - Required for FM-1.1 (R > 0.5 AND omega < 0.5)
  6. MODE_COLLAPSE - Required for FM-1.3 (diversity < threshold)
  7. **EPISTEMIC_SPIKE** - Required for FM-3.1 (epistemic > 0.5)
  8. ALEATORIC_DOMINANCE - Required for FM-2.2 (aleatoric > 2 × epistemic)

  Optional (P2 - Full Patent Compliance):
  9. DISTRIBUTIONAL_SHIFT - KS p-value < 0.05
  10. GRADUAL_DRIFT - Wasserstein increasing
  11. CONFIRMED_DRIFT - Heuristic + KS
  12. POSTERIOR_COLLAPSE - VAE-specific

  ---
  REFINED IMPLEMENTATION PLAN

  Architecture: detect_collapse_full()

  def detect_collapse_full(
      # Core R/S/N (always required)
      R: float,
      S: float,
      N: float,
      # Reliability Domain (P0)
      omega: Optional[float] = None,
      # Uncertainty Domain (P1)
      epistemic: Optional[float] = None,
      aleatoric: Optional[float] = None,
      was_correct: Optional[bool] = None,
      # Representation Domain (P1)
      diversity: Optional[float] = None,
      # Distributional Domain (P2)
      drift_heuristic: Optional[bool] = None,
      ks_p_value: Optional[float] = None,
      # Multi-source (existing)
      source_S_values: Optional[List[float]] = None,
  ) -> CollapseAnalysis:

  Detection Priority Order (Patent-Aligned)

  Priority 1 (Distributional - if signals available):
  ├── CONFIRMED_DRIFT:     drift_heuristic AND ks_p_value < 0.05
  ├── DISTRIBUTIONAL_SHIFT: ks_p_value < 0.05
  └── GRADUAL_DRIFT:        wasserstein_trend > 0

  Priority 2 (Uncertainty - if signals available):
  ├── OVERCONFIDENCE:      epistemic < 0.1 AND alpha < 0.4 AND was_correct=False
  ├── EPISTEMIC_SPIKE:     epistemic > 0.5
  └── ALEATORIC_DOMINANCE: aleatoric > 2 × epistemic

  Priority 3 (Reliability - if omega available):
  ├── HALLUCINATION:       alpha > 0.7 AND omega < 0.5
  ├── O_POISONING:         R > 0.5 AND omega < 0.5
  └── DRIFT:               omega < 0.35

  Priority 4 (Representation - always checked):
  ├── POSTERIOR_COLLAPSE:  latent_variance < 0.01 (if available)
  ├── MODE_COLLAPSE:       diversity < 0.2 (if available)
  └── RSN_COLLAPSE:        |R-S| < 0.1 AND |S-N| < 0.1

  Priority 5 (Quality - always checked, fallback):
  ├── CONFUSION:           N > 0.20 AND S > 0.25
  ├── CLASH:               var(source_S) > 0.15
  ├── POISONING:           N > 0.30
  └── DISTRACTION:         S > 0.40

  Thresholds (Patent-Specified)

  @dataclass
  class CollapseSignature:
      # Quality Domain (existing)
      POISONING_N_THRESHOLD: float = 0.30
      DISTRACTION_S_THRESHOLD: float = 0.40
      CONFUSION_N_THRESHOLD: float = 0.20
      CONFUSION_S_THRESHOLD: float = 0.25
      CLASH_S_VARIANCE_THRESHOLD: float = 0.15
      COLLAPSE_RISK_THRESHOLD: float = 0.60

      # Reliability Domain (NEW - Patent §5-7)
      HALLUCINATION_ALPHA_MIN: float = 0.70    # alpha > this
      HALLUCINATION_OMEGA_MAX: float = 0.50    # omega < this
      O_POISONING_R_MIN: float = 0.50          # R > this
      O_POISONING_OMEGA_MAX: float = 0.50      # omega < this
      DRIFT_OMEGA_MAX: float = 0.35            # omega < this

      # Representation Domain (NEW - Patent §8-10)
      RSN_COLLAPSE_DELTA: float = 0.10         # |R-S| and |S-N| < this
      MODE_COLLAPSE_DIVERSITY_MIN: float = 0.20 # diversity < this
      POSTERIOR_COLLAPSE_VARIANCE_MIN: float = 0.01

      # Uncertainty Domain (NEW - Patent §11-13)
      OVERCONFIDENCE_EPISTEMIC_MAX: float = 0.10
      OVERCONFIDENCE_ALPHA_MAX: float = 0.40
      EPISTEMIC_SPIKE_MIN: float = 0.50
      ALEATORIC_DOMINANCE_RATIO: float = 2.0

      # Distributional Domain (NEW - Patent §14-16)
      KS_P_VALUE_MAX: float = 0.05
