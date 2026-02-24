Unitree's ecosystem is perfect for this. Let me update the design to specifically target their tools:

● Write(docs\HUMANOID_UNITREE_INTEGRATION.md)
  ⎿  Wrote 1 lines to docs\HUMANOID_UNITREE_INTEGRATION.md
     # YRSN + Unitree Integration Plan
     **Target Robots:** G1, H1, H1_2
     **Simulation:** unitree_mujoco, Isaac Lab (unitree_rl_gym)
     **SDK:** unitree_sdk2_python
     **Date:** 2025-12-29
     ---
     ## 1. Unitree Ecosystem Mapping

● Summary: Complete Design Documentation Created

  I've created three design documents:

  1. docs/HUMANOID_VLA_YRSN_DESIGN.md (Comprehensive)

  - Full architecture with Layers 0-4 (Safety → VLA → Subsystems → Coordinator → Telemetry)
  - LEGS: TrajOpt-style with FeasibilityFilter, quality-scaled gait
  - ARMS/HANDS: OmniVIC-style with phase-aware impedance (K,D)
  - TORSO: Balance CBF with h_balance barrier function
  - Non-invasive telemetry: Trend detection, predictive alerts ("weeks ahead" warnings)
  - Detailed collapse type mapping per subsystem
  - 8-week implementation roadmap with 50+ specific tasks

  2. docs/HUMANOID_VLA_YRSN_HEXAGONAL.md (Simplified/Practical)

  - Proper hexagonal architecture mapping
  - Reuses existing ports: IVLAModel, ISkillMemory, IEventBus
  - Minimal new ports: IHumanoidController, IPhaseDetector
  - Simple VLAQualityNode with dependency injection
  - 4-week simplified roadmap
  - Philosophy: "Start simple, measure, add complexity only where needed"

  3. docs/HUMANOID_UNITREE_INTEGRATION.md (Specific)

  - Maps to Unitree ecosystem: unitree_mujoco, unitree_sdk2_python, unitree_IL_lerobot
  - H1 (19 DOF) vs G1 (23+ DOF) joint mappings
  - Adapter code sketches for MuJoCo and real robot
  - LeRobot → VLAExperience converter for RAG
  - Unitree-specific dashboard metrics

  ---
  Key Insights That Shaped the Plan

  | Insight                            | Impact                                                                  |
  |------------------------------------|-------------------------------------------------------------------------|
  | Phase-aware α expectations         | CONTACT phase: α≥0.3 OK (noisy), FREE_MOTION: α≥0.7 expected            |
  | Different failure modes            | Legs=catastrophic (fall), Arms=recoverable (drop) → different responses |
  | Non-invasive is the differentiator | YRSN's trend detection before threshold crossing is unique              |
  | RAG enables skill transfer         | commit_if_improved() stores only successful (instruction, K, D) tuples  |
  | Hexagonal keeps it clean           | Inject ports, swap adapters (sim↔real) without changing core            |

  ---
  Revised Priority Order

  MUST HAVE (Simulation V1):
  1. IHumanoidController + IPhaseDetector ports
  2. UnitreeMuJoCoAdapter
  3. VLAQualityNode with phase detection
  4. Basic telemetry emission

  SHOULD HAVE (Simulation V2):
  5. Quality-scaled gait (legs)
  6. Phase-aware impedance (arms)
  7. Balance barrier (torso)
  8. Trend detection alerts

  NICE TO HAVE (Future):
  9. RAG experience memory
  10. Predictive alerts
  11. Real robot (unitree_sdk2)

  Ready to start implementation when you are!

> proceed with full plan test debug redesign and debug TEST YRSN features throughout !!

* Creating humanoid port interface… (esc to interrupt · ctrl+t to hide todos · thought for 2s)
  ⎿  [ ] Create ports/humanoid.py (IHumanoidController)
     [ ] Create ports/phase.py (IPhaseDetector, VLAPhase)
     [ ] Create adapters/inbound/unitree_mujoco.py
     [ ] Create core/humanoid/phase_detector.py
     [ ] Create nodes/vla_quality_node.py
     [ ] Wire to existing decompose() and ControlBarrier
     [ ] Add legs quality routing (α → gait params)
     [ ] Add arms phase-aware impedance (phase → K,D)
     [ ] Add torso balance barrier check
     [ ] Connect telemetry for predictive alerts
