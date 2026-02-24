All three tasks completed:

  Summary

  1. Committed New Files

  45eaeae Add unified checkpointing and humanoid control components
  - ConsolidationManager, serialization, MuJoCo adapter
  - Interaction mesh, kinematic retargeting
  - VLAQualityNode lifecycle + checkpointing

  2. Trajectory Optimization (kinematic_retargeting.py)

  def optimize_trajectory(self, kinematic_traj, physical_params):
      # Iterative LQR-style optimization:
      # 1. Inverse dynamics: τ = M(q)q̈ + b*q̇
      # 2. Forward simulate with torque limits
      # 3. Feedback correction: Δτ = Kp*e + Kd*ė
      # 4. Enforce joint/torque limits
  - Joint inertia/damping models for H1 humanoid
  - Physical parameter perturbation (mass_scale, friction_scale)
  - Outputs: positions, velocities, torques, tracking_error

  3. OOD Detection (base_learner.py)

  def compute_omega(self, **kwargs):
      # Multi-signal OOD detection:
      # 1. SDM read confidence (low = far from stored patterns)
      # 2. Hopfield retrieval energy (high = poor prototype match)
      # 3. Replay buffer distance (far = unlike past experiences)
      # -> Geometric mean for ω ∈ [0, 1]
  - Returns ω=1.0 for in-distribution, ω<0.5 for OOD
  - Enables compute_omega=True in config to activate

  Commits

  edf39be Add trajectory optimization and OOD detection
  45eaeae Add unified checkpointing and humanoid control components

  All 130+ core tests pass.

    Summary of MORE completed work:

  1. Trajectory optimization - Implemented iterative LQR-style optimization in kinematic_retargeting.py with inverse/forward dynamics, joint inertia/damping/torque enforcement
  2. OOD detection - Implemented multi-signal approach in base_learner.py using SDM confidence, Hopfield retrieval energy, and replay buffer similarity
  3. Command-level NLP module - Created yrsn.core.nlp with:
    - IntentParser: Extract structured intent from natural language
    - CommandDecomposer: R/S/N at command level (not word level)
    - SkillMatcher: Match intents to robot skill catalog
    - GroundingValidator: Resolve language references to scene objects
