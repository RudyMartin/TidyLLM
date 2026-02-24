Humanoid VLA YRSN Integration Complete

  All 29 tests pass and the code has been pushed to main (commit 8c525f6).

  What was built:

  | Layer    | Files                 | Purpose                                                      |
  |----------|-----------------------|--------------------------------------------------------------|
  | Ports    | humanoid.py, phase.py | IHumanoidController interface, VLAPhase enum                 |
  | Core     | core/humanoid/        | Phase detector (hysteresis), impedance selector, gait scaler |
  | Adapters | mock_humanoid.py      | Mock robot for testing (H1/G1 support)                       |
  | Nodes    | vla_quality_node.py   | Main orchestrator with lifecycle                             |
  | Tests    | test_humanoid_vla.py  | 29 test cases                                                |

  Key YRSN features integrated:

  - Phase-aware quality thresholds: CONTACT allows α ≥ 0.3, FREE_MOTION requires α ≥ 0.7
  - Quality-scaled impedance: Higher α → higher stiffness (K), lower α → compliant
  - Quality-scaled gait: α controls walking speed/stride (α < 0.4 → stop)
  - Control barriers: Balance checks, safety threshold violations
  - Collapse detection: Maps R/S/N to safety responses (DRIFT, CONFUSION, SATURATION)
  - Telemetry: Real-time α, phase, mode emission for predictive alerts

  Architecture follows hexagonal pattern:
  VLA Model → [Port] → VLAQualityNode → [Port] → Robot
                  ↓
           YRSN decompose()
           ControlBarrier
           FeasibilityFilter
