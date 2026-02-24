HRM Model Feature Comparison (with Paper Reference)

  | Feature                    | Paper HRM        | HRMV (Baseline) | HRMV-Sig       | YRSN-HRM       |
  |----------------------------|------------------|-----------------|----------------|----------------|
  | Source                     | arXiv:2506.21734 | Our baseline    | Ablation       | Full YRSN      |
  | HRM Architecture           | ✅                | ✅               | ✅              | ✅              |
  | H/L Hierarchy              | ✅ H=3, L=5       | ✅ H=3, L=5      | ✅ H=3, L=5     | ✅ H=3, L=5     |
  | SwiGLU Activation          | ✅                | ✅               | ✅              | ✅              |
  | RMS Normalization          | ✅                | ✅               | ✅              | ✅              |
  | Post-Norm                  | ✅                | ✅               | ✅              | ✅              |
  | Truncated Normal Init      | ✅                | ✅               | ✅              | ✅              |
  | Flash Attention            | ✅                | ✅ SDPA          | ✅ SDPA         | ✅ SDPA         |
  | ACT (Adaptive Computation) | ✅ Q-learning     | ❌ Fixed         | ❌ Fixed        | ✅ Optional     |
  | RoPE                       | ❌                | ❌               | ❌              | ✅              |
  | Sparse Puzzle Embeddings   | ✅                | ❌ Learned       | ❌ Learned      | ❌ Pre-trained  |
  | Visual Encoder             | N/A (ARC)        | Sparse          | SigLIP         | SigLIP         |
  | Text Encoder               | N/A              | Learned         | Learned        | MiniLM         |
  | Instruction Boosting       | ❌                | ❌               | ❌              | 3.73x          |
  | Parameters                 | ~10M             | ~11M            | ~11M+          | ~31M           |
  | Task                       | ARC puzzles      | Variant Sudoku  | Variant Sudoku | Variant Sudoku |

  Notes:
  - Paper HRM was designed for ARC (Abstract Reasoning Corpus), not Sudoku
  - HRMV is our faithful adaptation of Paper HRM for variant Sudoku
  - SDPA = F.scaled_dot_product_attention (PyTorch's optimized attention, equivalent to Flash)
  - ACT Q-learning removed in HRMV for simplicity (fixed iterations instead)
