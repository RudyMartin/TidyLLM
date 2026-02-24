● Great question! Let me clarify the architecture differences:

  Model Variants - Different Architectures

  | Model    | Architecture    | Key Difference                             |
  |----------|-----------------|--------------------------------------------|
  | hrmv     | HRMV            | Visual encoder + text encoder (multimodal) |
  | yrsn_hrm | YRSNEnhancedHRM | Text-only HRM + YRSN projection heads      |
  | hrm      | SimplifiedHRM   | Basic text-only HRM                        |
  | hrmv_sig | HRMVSig         | SigLIP visual encoder (pretrained)         |

  Each model has different layers/weights - they are NOT interchangeable:

  HRMV:
    encoder.word_embeddings
    encoder.cell_embeddings
    encoder.visual_type_embeddings  ← visual encoding
    H_level, L_level, pos_head, digit_head

  YRSNEnhancedHRM:
    input_proj
    H_level, L_level
    pos_head, digit_head
    yrsn_projection  ← YRSN metrics head (R/S/N)

  You MUST Train Each Variant Separately

  # Train HRMV
  python benchmark/benchmark_1201HRM.py --epochs 10 --models hrmv
  # Saves: models/1201_hrmv.pt (HRMV weights)

  # Train YRSN-enhanced HRM
  python benchmark/benchmark_1201HRM.py --epochs 10 --models yrsn_hrm
  # Saves: models/1201_yrsn_hrm.pt (YRSNEnhancedHRM weights)

  Why Checkpoints Aren't Interchangeable

  Loading yrsn_hrm weights into HRMV fails because:
  - HRMV expects encoder.* keys
  - YRSNEnhancedHRM has input_proj.* keys
  - Different layer counts/dimensions

  The strict=False in load_checkpoint masks this by silently ignoring mismatched keys → model uses random weights →
  mode collapse!

  Summary

  YES, you need separate checkpoints for each model variant. The fix I made ensures:
  1. Checkpoint includes model_class metadata
  2. load_model() auto-detects and creates the correct architecture
  3. No more silent mismatches
