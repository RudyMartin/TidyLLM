The Actual Novel Contribution

  It's not "YRSN + Adaptive Computation" as two separate things.

  It's: YRSN-informed attention guidance for neural models

  The Mechanism (from the code):

  1. Problem: CTM (or any neural model) attends uniformly - it can't tell signal from noise
    - Board state has magnitude ~10.0
    - Rules have magnitude ~2.0
    - CTM drowns rules in board state → learns bias, not logic
  2. YRSN Solution: Use the R/S/N decomposition to boost what matters:
    - instruction_boost = 1.5x for rules (R)
    - task_boost = 1.8x for task description (R)
    - entity_boost = 1.0-2.0x for constraint keywords (R)
    - Board state gets 1.0x (S - tolerated but not emphasized)
  3. Self-Calibration Loop: Adjust boosts based on execution:
    - If CTM attention to instructions < 15% → increase boost
    - If false positives high → flatten bit-slice weights
    - If false negatives high → sharpen bit-slice weights

  The Better Framing:

  "YRSN guides attention."

  Neural models can't distinguish signal from noise on their own. YRSN's R/S/N decomposition identifies what matters (R), and this informs
  attention boosting - scaling embeddings so the model actually looks at the important parts.

  The self-calibration loop closes the feedback: if the model isn't attending to R, increase the boost until it does.

  ---
  Corrected Thesis Statement:

  "One contribution: YRSN-guided attention.

  YRSN decomposes context into signal (R), redundancy (S), and noise (N). This decomposition directly informs attention mechanisms: R is
  boosted, S is tolerated, N is filtered. A self-calibration loop adjusts boost factors based on whether the model actually attends to R.

  The insight: Models don't know what matters. YRSN tells them."

  ---
  Does this match what you intended? The key innovation isn't adaptive computation (that's prior art) - it's using YRSN to steer where
  attention goes.
