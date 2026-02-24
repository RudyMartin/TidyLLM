

● Exactly. To summarize what we now have:

  Calibration Architecture (Model Weights Untouched)

  ┌─────────────────────────────────────────────────────────────────────────┐
  │                        CALIBRATION POINTS                               │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                                                                         │
  │  Raw Puzzle ──► [YRSN Encoding] ──► [HRM Model] ──► [Temp Scaling] ──► Output
  │                      │                   │               │              │
  │                      ▼                   │               ▼              │
  │              ┌──────────────┐            │        ┌──────────────┐      │
  │              │ CALIBRATED:  │            │        │ CALIBRATED:  │      │
  │              │ - bit_slice  │            │        │ - temperature│      │
  │              │   weights    │      FROZEN│        │   (T > 1     │      │
  │              │ - instruction│            │        │   reduces    │      │
  │              │   boost      │            │        │   confidence)│      │
  │              └──────────────┘            │        └──────────────┘      │
  │                                          │                              │
  │                              ┌───────────┴───────────┐                  │
  │                              │ HRM WEIGHTS FROZEN:   │                  │
  │                              │ - Attention params    │                  │
  │                              │ - Layer weights       │                  │
  │                              │ - All learned params  │                  │
  │                              └───────────────────────┘                  │
  └─────────────────────────────────────────────────────────────────────────┘

  Two Calibration Systems

  | System             | What it Adjusts                      | When            | How                            |
  |--------------------|--------------------------------------|-----------------|--------------------------------|
  | YRSNSelfCalibrator | bit_slice_weights, instruction_boost | During training | --self_calibrate flag          |
  | TemperatureScaler  | Output confidence (logits / T)       | At inference    | --adaptive or --calibrate flag |

  Both operate outside the HRM model - one on input encoding, one on output scaling. The HRM model itself is only modified by gradient
  descent during training, never by calibration.
