 Calibration Deployment Summary

  You now have two calibration systems:

  1. Temperature Scaling (just added) - Fixes Overconfidence at Inference

  When to deploy: Before running benchmarks
  How: --calibrate flag

  python examples/benchmark_hrm_yrsn_comparison.py --mode first_move --calibrate

  What it does:
  - Collects model logits on calibration puzzles
  - Learns optimal temperature T via LBFGS (minimizes NLL)
  - T > 1 → reduces confidence (fixes overconfidence)
  - T < 1 → increases confidence (fixes underconfidence)
  - Applies scaling at inference: softmax(logits / T)

  Current results:
  - YRSN: Position T=0.875, Digit T=1.112
  - Vanilla: Position T=0.803, Digit T=1.213

  ---
  2. YRSNSelfCalibrator (existing) - Optimizes YRSN Parameters During Training

  Location: src/yrsn_context/strategies/self_calibration/calibrator.py

  When to deploy: During/after training when you have CTM attention weights
  How: Integrate into training loop

  from yrsn_context.strategies.self_calibration import YRSNSelfCalibrator

  calibrator = YRSNSelfCalibrator(learning_rate=0.15, min_traces_for_calibration=20)

  # During training/validation:
  for batch in dataloader:
      trace = YRSNExecutionTrace(
          cell_importance=yrsn_output.cell_importance,
          ctm_attention_weights=ctm_attention,
          deduction_correct=is_correct,
          ...
      )
      calibrator.record_trace(trace)

  # Periodically calibrate:
  result = calibrator.calibrate(current_bit_slice_weights, current_instruction_boost)
  # Use result['new_bit_slice_weights'] and result['new_instruction_boost']

  What it adjusts:
  - Bit-slice weights (0.5, 0.3, 0.2 → learned values)
  - Instruction boost factor (3.73 → learned value)
  - Based on false positive/negative rates between YRSN predictions and CTM attention

  ---
  Key Insight from Diagnostics

  The 95% confidence on wrong answers and 100% position bias indicate the models need more training with diverse data, not just
  inference-time calibration. Temperature scaling helps but the root cause is:

  1. Position bias → Training data always has first empty cell at similar positions
  2. Digit bias → Training data not balanced across digits
  3. Overconfidence → Model hasn't seen enough failure cases

  Would you like me to integrate the YRSNSelfCalibrator into the training script next?

  Todos
  [×] Add temperature scaling for confidence calibration
  [ ] Integrate YRSNSelfCalibrator into training loop

  ----

  
● Now it's working correctly. The adaptive calibration:

  1. Detects overconfidence: YRSN was 90% confident on wrong answers
  2. Computes temperature: T=1.66 based on the diagnostic heuristic
  3. Applies scaling: Re-runs benchmark with calibrated model
  4. Shows impact: Confidence dropped from 90% → 46%

  The solve rate didn't change because temperature scaling doesn't change the argmax prediction - it just makes the model less confident.
  The real fix needs to happen in training.

● ---
  Summary: Calibration Systems Now Connected

  You now have diagnostic-driven adaptive calibration:

  # Run with adaptive calibration (uses failure diagnostics)
  python examples/benchmark_hrm_yrsn_comparison.py --mode first_move --adaptive

  # Or manual calibration (uses NLL optimization on validation data)
  python examples/benchmark_hrm_yrsn_comparison.py --mode first_move --calibrate

  How --adaptive Works:

  1. Run initial benchmark → get results
  2. Analyze failures → detect overconfidence, position bias, etc.
  3. Compute temperature from diagnostics:
    - 90% confidence on failures → T = 1.66 (reduce confidence)
    - 69% confidence → T = 1.0 (no change needed)
    - 30%+ position errors → T *= 1.2 (extra penalty for confusion)
  4. Re-run with calibrated models
  5. Show before/after comparison

  Key Results:

  | Model    | Before Cal.          | After Cal. | Temperature |
  |----------|----------------------|------------|-------------|
  | YRSN+HRM | 90% conf on failures | 46%        | T=1.66      |
  | Vanilla  | 69% conf on failures | 69%        | T=1.00      |

  The diagnostic system now informs the calibration - no manual tuning needed.

  Todos
  [×] Connect failure diagnostics to temperature scaling
  [ ] Integrate YRSNSelfCalibrator into training loop

-----------------
    Can safely calibrate (external to model):
  - YRSN encoding parameters (bit-slice weights, instruction boost) - these are input preprocessing parameters
  - Temperature scaling - applied after model output, doesn't change model

  The current implementation is fine in that regard - we're only adjusting:
  1. bit_slice_weights - how we weight different precision levels in YRSN encoding
  2. instruction_boost - how much we emphasize rules text in encoding

  These affect how we prepare input to HRM, not HRM itself. The HRM model weights stay frozen.

