 The YRSN Reasoner is complete but needs:
  1. A trained neural model (CTMContextRetriever) to generate insights
  2. Training script for the LearnedConstraintEncoder

  Let me create a training script for the YRSN Reasoner:

● Write(examples\train_yrsn_reasoner.py)
  ⎿  Wrote 1 lines to examples\train_yrsn_reasoner.py
     """
     Train YRSN Sudoku Reasoner on Variant Puzzles
     Pure neural approach using bit-sliced context:
     - Level 0: Core rules (standard Sudoku)
     - Level 1: Visual constraints (Killer, Thermo, etc.)
     - Level 2: Deduction history
     Key insight: Rules are PRESERVED at Level 0, never compressed away.
     """
