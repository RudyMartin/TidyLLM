  Sudoku Augmentation Strategies

  | Vision Analog     | Sudoku Equivalent                              | Applied?       |
  |-------------------|------------------------------------------------|----------------|
  | Random crop       | Partial fill states                            | ✅ Just created |
  | Rotation/flip     | Grid symmetry transforms (rotate 90°, reflect) | Can add        |
  | Color jitter      | Digit permutation (swap 1↔2, 3↔4, etc.)        | Can add        |
  | Text augmentation | Rule paraphrasing                              | Can add        |
  | Noise injection   | Constraint masking (hide some constraints)     | Advanced       |

  The partial fill augmentation I just created is the most important - it ensures the model sees realistic test conditions. Now let me
  update the training script to use the augmented data:


  ● Now let's run the augmentation with geometric transforms:

● Bash(python examples/augment_training_data.py 2>&1)
  ⎿  Augmenting 4x4...
     Original: 12 puzzles from data/train_4x4.json
       Rotation-safe puzzles: 7/12
     … +24 lines (ctrl+o to expand)

● Excellent! We went from 80 original puzzles to 1,760 augmented examples:
  - 4x4: 12 → 261 (7 rotation-safe)
  - 6x6: 12 → 231 (5 rotation-safe)
  - 9x9: 56 → 1,268 (36 rotation-safe)

  The geometric transforms increased data by ~70% on top of the fill-level augmentation. Now let's retrain with this larger dataset:
54 +      # Directional constraints that break with rotation
       155 +      unsafe_patterns = [
       156 +          'thermometer', 'thermo',
       157 +          'arrow',
       158 +          'little killer', 'littlekiller',
       159 +          'sandwich',  # Row/col specific
       160 +          'skyscraper',  # Direction matters
       161 +          'between',
       162 +          'nabner',  # Oriented lines
       163 +          'indexer', 'indexing',  # Row/col references
       164 +      ]
       165 +
       166 +      for pattern in unsafe_patterns:
       167 +          if pattern in rules or pattern in visual:
       168 +              return False
       169 +
       170 +      # Check visual elements for directional types
       171 +      if 'arrow' in visual or 'direction' in visual:
       172 +          return False
       173 +
       174 +      return True
