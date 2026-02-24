

● The model is still showing severe digit bias. The issue is likely that training on 9x9 data didn't have enough time and the model
  collapsed to predicting a single digit. Let me update the todo list with a summary:

● Summary of Session Progress

  Completed Sudoku-Bench Improvements:

  1. Constraint Parser (constraint_parser.py) - Parses rules into structured constraint types
  2. Candidate Tracker (candidate_tracker.py) - Pencil marks and constraint propagation
  3. Curriculum Trainer (curriculum_trainer.py) - 4x4 → 6x6 → 9x9 progression
  4. Multi-Turn Evaluator (multi_turn_evaluator.py) - Matches Sudoku-Bench protocol

  Data Augmentation Added:

  - Partial fills: 10%, 20%, 30%, 40% of cells pre-filled from solution
  - Geometric transforms: rotate 90°/180°/270°, flip H/V (only for rotation-safe puzzles)
  - Rotation-safe check: Skips transforms for directional constraints (thermometer, arrow, little killer, etc.)
  - Dataset grew from 80 → 1,760 puzzles

  Current Status:

  The models still need more training. Key issues:
  - Digit bias: Models predict same digit (4 or 9) regardless of puzzle
  - Small training set: Even with augmentation, need more diverse examples
  - Training time: CPU training is slow, would benefit from GPU

  Next Steps (for future sessions):

  1. Train longer with more epochs
  2. Use GPU acceleration if available
  3. Add digit permutation augmentation (swap 1↔2, etc.)
  4. Consider pre-training on standard Sudoku before variant puzzles

  5. ---------------

  6. ● The model IS learning now!

  - Loss decreased from 2.85 → 0.02 (excellent!)
  - Predictions are now correct: pred=3 actual=3, pred=4 actual=4

  The encoder fix resolved the bug. The model now:
  1. Produces unique per-cell embeddings ✓
  2. Can learn to make different predictions ✓
  3. Successfully learns from training data ✓

● Summary

  Root Cause Found and Fixed: The digit bias was caused by the align_rules_to_positions method in both rules_encoder.py and
  rules_encoder_enhanced.py. The attention mechanism was using the single rules vector as key/value, which produced nearly identical outputs
   for all 81 cells regardless of their actual positions or contents.

  Fix Applied: Changed the align_rules_to_positions method from attention-based alignment to element-wise addition:

  # OLD (buggy): Attention with single vector → all cells get same output
  aligned, _ = self.rule_to_position_attention(
      query=position_kv,  # [1, 81, dim]
      key=rules_query,    # [1, 1, dim] - SAME for all queries!
      value=rules_query   # [1, 1, dim] - Returns same value!
  )

  # NEW (fixed): Direct combination preserves unique per-cell info
  rules_broadcast = rules_vector.expand(num_cells, -1)  # [81, dim]
  combined = rules_broadcast + position_features         # Each cell unique!

  Verification:
  1. ✅ Different cells now have different embeddings (diff ~0.07)
  2. ✅ Changed cells produce different outputs
  3. ✅ Model successfully learns (loss: 2.85 → 0.02)
  4. ✅ Predictions become correct after training (pred=3 actual=3)

  The model can now be trained properly without digit bias.


    Available Architectures for Variant Sudoku

  | Architecture  | File                            | Approach                        | Rules Handling                    |
  |---------------|---------------------------------|---------------------------------|-----------------------------------|
  | HRM           | models/hrm.py                   | Hierarchical Refinement Machine | Rules encoder → context           |
  | Hybrid Solver | applications/sudoku/hybrid.py   | Classical CSP + Neural break-in | Uses constraints encoder          |
  | YRSN Reasoner | applications/sudoku/reasoner.py | Full YRSN: Bit-sliced + CTM     | 100% neural, preserves rules text |
  | CTM/NLM       | neural/ctm/nlm.py               | Continuous Thought Machine      | Temporal reasoning                |
  | Quantum QSVM  | models/quantum_qsvm.py          | Quantum kernel classification   | Feature encoding                  |

  Key Options:

  1. Hybrid Solver - Classical CSP handles 70-90% deterministically, neural only for "break-ins". Most data-efficient!
  2. YRSN Reasoner - Pure neural but uses bit-sliced context to preserve rules at different levels. More aligned with YRSN thesis.
  3. Continue HRM - Current approach, but struggling with small data.

  Recommendation: The Hybrid Solver makes the most sense for variant Sudoku because:
  - Classical CSP can parse and apply the variant rules deterministically
  - Neural only needed for creative break-ins (rare)
  - Much less training data needed
  - Matches the d1 paper insight: "hybrid approaches outperform pure neural"

  Should I pivot to the Hybrid Solver approach?

