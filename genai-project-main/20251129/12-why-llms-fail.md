Key Findings from Sudoku-Bench

  Why LLMs Fail (<15% solve rate)

  1. "Break-in" Problem: Variants require finding novel logical breakthroughs unique to each puzzle. LLMs resort to brute-force search
  instead of identifying these insights.
  2. Constraint Interaction: The key challenge isn't individual constraints but their unique combinations. Even familiar constraints
  (arrows, knight's move) create novel interactions.
  3. Meta-Constraints: Some puzzles require deducing puzzle parameters (e.g., "cage sums to unknown value to be determined").
  4. Empty Grids: Variants often start with zero given digits, making the search space too large for guessing.

  Failure Categories (Figure 4)

  - Incorrect Solution (most common): Wrong digit with confidence
  - Missing Information: Claims puzzle is incomplete
  - Claimed Contradiction: False contradiction in rules
  - Surrender: Explicitly gives up

  ---
  Improvement Ideas for YRSN+HRM

  1. Constraint Interaction Encoding

  The paper emphasizes that constraint interactions matter most. Our current YRSN encodes constraints separately.

  Idea: Add a constraint interaction layer that explicitly models relationships between constraints:
  # Encode pairwise constraint interactions
  # e.g., "knight's move + arrow" creates long-range dependency chains
  constraint_pairs = encode_constraint_interactions(rules)

  2. Break-in Detection Module

  Train a separate module to identify the "break-in point" - where logical deduction should start.

  Idea: Instead of predicting all cells equally, first predict which cells form the break-in chain:
  # Two-stage prediction:
  # 1. Identify break-in cells (where constraints interact most tightly)
  # 2. Solve those first, propagate to rest
  break_in_mask = self.break_in_detector(context)

  3. Hierarchical Reasoning for Variants

  The paper shows that 4×4 puzzles are much easier (40-73% solve rate) than 9×9 (0%). Our HRM hierarchy could mirror puzzle structure:
  - H-level: Global constraint reasoning (row/column/box interactions)
  - L-level: Local constraint propagation (individual cell candidates)

  4. Candidate Tracking (Pencil Marks)

  Human solvers use pencil marks to track candidate digits. Current model doesn't maintain this state.

  Idea: Add explicit candidate representation:
  # Instead of just board state, track:
  # - Current board (filled digits)
  # - Candidate sets per cell (possible digits)
  # - Eliminated candidates with reason

  5. Multi-Step Reasoning with Memory

  Paper notes: "solving over 100s of steps will require a form of memory, like a scratchpad"

  Idea: Extend HRM state to include:
  - Deduction history
  - Constraint satisfaction status
  - Working hypotheses

  6. Training on Expert Traces

  The paper provides Cracking the Cryptic transcripts (thousands of hours):
  - HuggingFace: SakanaAI/Sudoku-CTC-Reasoning

  Idea: Imitation learning from expert reasoning:
  # Train to predict:
  # 1. Which cell to reason about next
  # 2. What deduction technique to apply
  # 3. What digit to place (with justification)

  7. Constraint-Specific Encoders

  Different constraint types need different reasoning:
  - Sum constraints (killer, arrows): Arithmetic reasoning
  - Ordering constraints (thermometer, whispers): Sequence reasoning
  - Distance constraints (knight, king): Spatial reasoning

  Idea: Specialized encoder heads per constraint type:
  class ConstraintTypeEncoder(nn.Module):
      def __init__(self):
          self.sum_encoder = SumConstraintEncoder()
          self.order_encoder = OrderConstraintEncoder()
          self.distance_encoder = DistanceConstraintEncoder()

  8. Constraint Satisfaction Verification

  Add a verification head that checks if placements satisfy all constraints:
  # Before committing a digit, verify:
  verify_logits = self.constraint_verifier(proposed_placement, all_constraints)
  # Only place if verification passes

  ---
  Quick Wins to Implement

  1. Better constraint parsing in YRSN encoder - extract structured constraint types
  2. Candidate elimination tracking - which digits are impossible for each cell
  3. Difficulty-aware training - start with 4×4, curriculum to 9×9
  4. Multi-turn evaluation - match Sudoku-Bench's multi-step protocol

● All 4 Sudoku-Bench improvements are now complete. Here's a summary:

  Completed Quick Wins (1-4)

| Component            | File                    | Key Features    |
|----------------------|-------------------------|----------------------------------------------------------------------------------|
| Constraint Parser    | constraint_parser.py    | Parses rules → structured types (SUM, ORDER, DISTANCE, etc.), detects constraint interactions, generates break in hints |
| Candidate Tracker    | candidate_tracker.py    | Pencil marks, constraint propagation, naked/hidden singles detection, MRV heuristic, tensor encoding for HRM |   
| Curriculum Trainer   | curriculum_trainer.py   | 5-stage progression (4x4→9x9), weighted grid sampling, accuracy-based advancement, validation      |            
| Multi-Turn Evaluator | multi_turn_evaluator.py | Sudoku-Bench protocol: model predicts → verify → update board → repeat until solved or error. Tracks solve rate, correct placements, termination reasons |

  All modules are exported from yrsn_context.__init__.py and tested. The multi-turn evaluator implements the exact protocol from
  Sudoku-Bench:
  - Model provides one digit placement per turn
  - Evaluator verifies against solution
  - Updates board and continues or terminates on first error
  - Tracks: solve rate, avg correct placements, placement rate, termination distribution
