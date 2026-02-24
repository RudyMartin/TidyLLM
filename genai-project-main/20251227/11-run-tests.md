That's real code. Here's what just ran:

  | Test             | What It Proves                                                                     |
  |------------------|------------------------------------------------------------------------------------|
  | Phase Impedance  | CONTACT phase uses 188 N/m stiffness vs 1000 N/m for FREE_MOTION (5.3x difference) |
  | Gait Scaling     | α=0.9 walks at 1.2m/s, α<0.4 stops completely                                      |
  | YRSN Decompose   | "maybe grab the thing" → α=0.29 (low), "Pick up red cube" → α=0.63 (higher)        |
  | Control Barriers | α=0.8 allowed, α=0.2 blocked                                                       |
  | Robot Physics    | Joint moved 0.48 rad toward 0.5 rad target in 50 steps                             |
  | Joint Mapping    | Real H1 robot has 12 leg, 6 arm, 1 torso joints                                    |

  The key insight: YRSN quality signals (α, R, S, N) directly control robot behavior:
  - Low α → compliant impedance (safe contact)
  - Low α → slower gait (cautious movement)
  - Low α → blocked predictions (safety barrier)

  The fuzzy instruction "maybe grab the thing" scores α=0.29, which would:
  - Stop the robot from walking (gait=0)
  - Block the prediction (ControlBarrier returns False)
  - Use minimum stiffness if forced to move

  vs "Pick up the red cube" at α=0.63 allows normal operation.

The test specification is now at docs/HUMANOID_VLA_TEST_SPEC.md and pushed.

  What it documents:

  | Section                  | Shows                                   |
  |--------------------------|-----------------------------------------|
  | 1. Quality Decomposition | Input instruction → R/S/N/alpha mapping |
  | 2. Phase Detection       | Force + Distance → Phase + threshold    |
  | 3. Impedance Selection   | Phase + alpha → K/D values              |
  | 4. Gait Scaling          | alpha → velocity/stride/mode            |
  | 5. Control Barriers      | alpha → ALLOW/BLOCK decision            |
  | 6. End-to-End Tick       | Full pipeline with real outputs         |

  Key tables for quick reference:

  Instruction Quality:
    "Pick up red cube"     → alpha=0.63 (good)
    "maybe grab something" → alpha=0.29 (bad)

  Phase Thresholds:
    FREE_MOTION → needs alpha >= 0.70
    CONTACT     → needs alpha >= 0.30

  Impedance by Phase:
    FREE_MOTION → K=1000 N/m (stiff)
    CONTACT     → K=50-200 N/m (compliant)

  Gait by Quality:
    alpha=0.90 → 1.2 m/s (confident)
    alpha<0.40 → 0.0 m/s (STOP)

  Run pytest tests/test_humanoid_vla.py -v to verify all 29 tests pass.
