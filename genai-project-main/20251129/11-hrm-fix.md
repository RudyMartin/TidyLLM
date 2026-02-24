  HRM Fix Summary

  Problem identified: The mock HRM only used position 0 for output:
  output_repr = z_H[:, 0, :]  # WRONG - only position 0

  Fixed to match official sapientinc/HRM:
  all_logits = self.lm_head(z_H)  # [batch, seq_len, vocab_size] - ALL positions

  Key Changes:

  | Aspect       | Before (Mock)                           | After (Fixed)               |
  |--------------|-----------------------------------------|-----------------------------|
  | Output shape | [batch, 81] position + [batch, 9] digit | [batch, 81, 10] all cells   |
  | Output head  | Separate position_head + digit_head     | Single lm_head              |
  | Task framing | "Predict which cell + which digit"      | "Predict all cells at once" |
  | vocab_size   | 81 (positions)                          | 10 (digits 0-9)             |

  What This Means:

  1. Training: Target is now the full solution (81 digits), not just one cell
  2. Loss: Cross-entropy over all 81 positions simultaneously
  3. Inference: Model predicts entire solution, then we compare to ground truth

  The model now matches the official HRM architecture. The benchmark and training scripts will need updates to use return_all_positions=True
   and compare full solutions.

  Sources:
  - https://github.com/sapientinc/HRM
  - https://arxiv.org/abs/2506.21734
  - https://github.com/lucidrains/HRM

─────────────────────────────────────────────────────────
