deep analysis of the switching functionality.

● Bash(git -C "C:\Users\marti\github\yrsn" push)
  ⎿  To https://github.com/NextShiftConsulting/yrsn.git
        6571a1d..16fd3a3  main -> main

● Commits pushed successfully. Now let me provide the deep analysis of YRSN's switching functionality.

  ---
  Deep Research: YRSN Switching Functionality - Fixed vs. Dynamic

  Based on my analysis of 186 files containing threshold/switch logic, here's what I found:

  FIXED (Hard-Coded) Switching Thresholds

  | Location                      | Parameter                         | Fixed Value                        | Purpose                  |
  |-------------------------------|-----------------------------------|------------------------------------|--------------------------|
  | command_decomposer.py:146-147 | alpha_threshold, safety_threshold | 0.6, 0.8                           | Command execution gate   |
  | _defaults.py:14-18            | ALPHA_HIGH, ALPHA_LOW             | 0.70, 0.40                         | Quality phase boundaries |
  | adaptive_config.py:83-96      | Routing thresholds                | 0.70, 0.40, 0.95, 0.70, 0.50, 0.30 | Stream routing           |
  | collapse.py:181-221           | All collapse thresholds           | 0.30, 0.40, 0.20, 0.25, etc.       | Collapse detection       |
  | vla_control_center.py:68-77   | VLA routing thresholds            | 0.7, 0.4, 0.3, 0.95                | Action acceptance        |
  | base_learner.py:94-98         | Safety constraints                | 0.4, 0.05, 0.5                     | Memory safety gates      |
  | temperature/__init__.py:65-67 | Phase thresholds                  | 0.70, 0.40                         | Temperature mapping      |
  | grounding.py:59-60            | Grounding confidence              | 0.8                                | Reference resolution     |

  Key Fixed Decision Logic (from command_decomposer.py:200-206):

  # CURRENT: Binary threshold-based switching
  threshold = self.safety_threshold if self._is_safety_critical(intent) else self.alpha_threshold
  quality.can_execute = (
      quality.alpha >= threshold and      # FIXED comparison
      quality.execution_risk < 0.5 and    # FIXED threshold
      len(quality.issues) == 0            # Binary check
  )

  What IS Dynamic (Learned/Adaptive)

  | Location                         | Mechanism             | How It Adapts                             |
  |----------------------------------|-----------------------|-------------------------------------------|
  | adaptive_routing.py:145-148      | Target error rates    | Calibrated from observed data             |
  | adaptive_config.py:329-332       | Dynamic thresholds    | Adjusted by τ at runtime                  |
  | ensemble_optimizer.py:181-227    | Ensemble weights      | Updated based on layer accuracy           |
  | base_learner.py:349-417          | ω (omega) computation | Computed from SDM/Hopfield/replay signals |
  | pillar_configs.py:252-254        | Adaptation rate       | EMA smoothing of quality                  |
  | curriculum_learning.py:2283-2385 | Synaptic weights      | STDP-like plasticity                      |

  Gap Analysis: Fixed vs. Paper's Learned Switching

  Paper's approach (Emergent Temporal Abstractions):
  # LEARNED switching unit
  β_t = f_switch(e_t, h_{t-1}, z_{t-1})  # Neural network output ∈ [0,1]
  z_t = β_t ⊙ z̃_t + (1 - β_t) ⊙ z_{t-1}  # Soft temporal integration

  YRSN's current approach:
  # FIXED binary decision
  if alpha >= 0.6:
      execute()
  else:
      clarify()

  Specific Recommendations

  1. CommandDecomposer (command_decomposer.py:200-206):
    - Replace quality.can_execute = quality.alpha >= threshold with:
  β = self.switching_network(intent_embedding, prev_state, quality_signals)
  quality.execution_confidence = β * quality.alpha + (1 - β) * prev_confidence
  quality.can_execute = quality.execution_confidence >= self.dynamic_threshold
  2. VLA Control Center (vla_control_center.py:263-281):
    - The _compute_alpha method already computes α dynamically
    - Add learned β_t to modulate action confidence over time
  3. Base Learner (base_learner.py:94-98):
    - The alpha_min = 0.4 is fixed as a safety barrier
    - Could add learned switching for non-safety-critical paths

  Files Requiring Changes for Learned Switching

  1. src/yrsn/core/nlp/command_decomposer.py - Add SwitchingUnit class
  2. src/yrsn/core/control/vla_control_center.py - Add temporal β_t
  3. src/yrsn/core/_defaults.py - Make thresholds configurable vs. constants
  4. src/yrsn/core/memory/base_learner.py - Add learned switching for non-safety paths

  Summary

  | Category               | Count | Examples                                       |
  |------------------------|-------|------------------------------------------------|
  | Fixed thresholds       | ~50+  | 0.6, 0.7, 0.4, 0.3, 0.8, 0.5                   |
  | Adaptive mechanisms    | ~10   | ω computation, ensemble weights, EMA smoothing |
  | True learned switching | 0     | No neural β_t function exists yet              |

  The codebase has extensive fixed thresholds with some adaptive mechanisms (ω, ensemble weights), but lacks the paper's learned switching unit (β_t) that could:
  - Replace hard thresholds with soft, learned transitions
  - Enable temporal integration of decisions
  - Allow gradient-based optimization of switching behavior

> what will true learned switching break?

● Great question. Let me analyze what could break:

  What True Learned Switching Would Break

  1. Safety Guarantees (CRITICAL)

  Current fixed approach:
  # command_decomposer.py:147 - DETERMINISTIC
  safety_threshold: float = 0.8  # Always blocks unsafe commands below this

  With learned β_t:
  # β_t could drift, outputting 0.9 when it should output 0.2
  # A trained network could "learn" to approve dangerous actions

  Risk: Safety-critical actions (press, close, rotate, pour) currently require α > 0.8. A learned switching unit could:
  - Overfit to training distribution
  - Fail on adversarial inputs
  - Drift during deployment

  2. Interpretability / Auditability

  Current: "Command rejected because α=0.58 < 0.60 threshold"

  With learned switching: "Command rejected because β_t=0.42 from 256-dim hidden state"

  Breaks:
  - clarification_prompt generation in command_decomposer.py:447-455
  - Debugging why a command failed
  - Regulatory compliance (explainable AI requirements)

  3. Cold Start Problem

  Current: Works immediately with zero training data

  With learned switching: Needs supervised examples of when to switch

  Breaks:
  - Out-of-box functionality
  - New robot deployments
  - Novel task domains

  4. Existing Code Assumptions

  Files that assume fixed threshold semantics:

  | File                     | Assumption                                  | Would Break                                 |
  |--------------------------|---------------------------------------------|---------------------------------------------|
  | grounding.py:59          | grounding_confidence > 0.8 = fully grounded | Soft β could accept partial grounding       |
  | skill_matcher.py:259-261 | Missing params → confidence *= 0.5          | Multiplicative penalties become meaningless |
  | collapse.py:181-221      | Collapse thresholds are safety bounds       | Learned switching could mask collapses      |
  | base_learner.py:94       | alpha_min = 0.4 is a hard barrier           | Barrier function guarantees violated        |

  5. Test Suite Expectations

  Current tests likely assert:
  assert quality.can_execute == (alpha >= 0.6)  # Deterministic

  With learned switching:
  # Non-deterministic - depends on network weights
  # Tests become flaky or require mocking

  6. Latency Budget

  Current: Threshold comparison is O(1)

  With neural β_t:
  β = self.switching_network(e_t, h_{t-1}, z_{t-1})  # Forward pass ~1-10ms

  Breaks: Real-time constraints in vla_control_center.py:259 (timescale_ms=10.0)

  ---
  Safe Migration Path

  Instead of replacing fixed thresholds, layer learned switching on top:

  class HybridSwitchingGate:
      """Learned switching with safety overrides."""

      def __init__(self, safety_threshold: float = 0.8):
          self.safety_threshold = safety_threshold  # FIXED - never learned
          self.switching_net = LearnedSwitchingUnit(...)

      def decide(self, intent, quality, prev_state) -> tuple[bool, float]:
          # Hard safety gate (FIXED)
          if self._is_safety_critical(intent):
              if quality.alpha < self.safety_threshold:
                  return False, 0.0  # Deterministic rejection

          # Learned switching for non-critical paths
          β = self.switching_net(intent.embedding, prev_state)
          confidence = β * quality.alpha + (1 - β) * prev_state.confidence

          # Learned threshold with floor
          threshold = max(0.3, self.learned_threshold)  # Floor prevents catastrophic drift

          return confidence >= threshold, β

  What Stays Fixed (Safety Layer)

  - safety_threshold = 0.8 for critical actions
  - alpha_min = 0.4 barrier in base_learner.py
  - Collapse detection thresholds in collapse.py

  What Becomes Learned (Non-Safety)

  - Soft switching between VLA models
  - Temporal integration of confidence
  - Ensemble weight adaptation

  This preserves safety guarantees while gaining the paper's temporal abstraction benefits.

> i DONT WNAT TO BREAK working code - seems to me that  β_t will "need" support for processing that may not exist. Lets check that
 aspect DEEP RESEARCH how it is calculated and if ALL the items are always there. Fixed metrics are there to fill the gaps.

● Excellent point. Let me deep-dive into what β_t actually needs and whether YRSN has those inputs available.

● Search(pattern: "def decompose|def predict|def decide|def route|def execute", path: "src\yrsn\core", glob: "*.py", output_mode:
        "content")
  ⎿  Found 124 lines (ctrl+o to expand)

● Read(src\yrsn\core\memory\base_learner.py)
  ⎿  Read 150 lines

● Read(src\yrsn\core\control\vla_control_center.py)
  ⎿  Read 200 lines

● Read(src\yrsn\core\memory\base_learner.py)
  ⎿  Read 150 lines

● Read(src\yrsn\core\control\vla_control_center.py)
  ⎿  Read 150 lines

● Search(pattern: "prev_|previous|h_t|z_t|hidden|state|history", path: "src\yrsn\core", glob: "*.py", output_mode: "content",
        head_limit: 80)
  ⎿  Found 80 lines (ctrl+o to expand)

● Read(src\yrsn\core\control\barrier_control.py)
  ⎿  Read 33 lines

● Search(pattern: "embedding|encode|latent|repr", path: "src\yrsn\core", glob: "*.py", output_mode: "content", head_limit: 60)
  ⎿  Found 60 lines (ctrl+o to expand)

● Read(src\yrsn\core\memory\base_learner.py)
  ⎿  Read 150 lines

● Read(src\yrsn\core\memory\base_learner.py)
  ⎿  Read 150 lines

● Now let me check what state the CommandDecomposer actually has access to:

● Read(src\yrsn\core\nlp\command_decomposer.py)
  ⎿  Read 70 lines

● ---
  Deep Research: β_t Input Requirements vs. YRSN State Availability

  What β_t Needs (from paper)

  β_t = f_switch(e_t, h_{t-1}, z_{t-1})  # Learned switching unit

  # Inputs required:
  # e_t     = current embedding/observation (256-1024 dim vector)
  # h_{t-1} = previous hidden state from controller (persistent)
  # z_{t-1} = previous subgoal/latent (persistent)

  What YRSN Actually Has at Each Switching Point

  | Component          | e_t (embedding)     | h_{t-1} (prev hidden) | z_{t-1} (prev subgoal)  |
  |--------------------|---------------------|-----------------------|-------------------------|
  | CommandDecomposer  | ❌ NO               | ❌ NO                 | ❌ NO                   |
  | VLAControlCenter   | ❌ NO               | ❌ NO                 | ⚠️ _action_history only |
  | BaseLearner        | ✅ encode_context() | ⚠️ _last_alpha only   | ❌ NO                   |
  | GroundingValidator | ❌ NO               | ⚠️ _discourse_history | ❌ NO                   |

  Detailed Gap Analysis

  1. CommandDecomposer (command_decomposer.py)

  # CURRENT - Stateless! No embedding, no history
  def decompose(self, intent: ParsedIntent, scene_objects=None):
      # e_t: Not computed - just parses intent slots
      # h_{t-1}: MISSING - no instance state tracked
      # z_{t-1}: MISSING - no subgoal representation

      # Falls back to FIXED threshold
      threshold = self.safety_threshold if critical else self.alpha_threshold

  Why fixed threshold exists: There's no e_t embedding to feed into a neural network. Intent is just slot dictionaries (action="pick", object="cup").

  2. VLAControlCenter (vla_control_center.py:125)

  # CURRENT - Partial state tracking
  self._action_history: deque  # ✅ Exists
  # But:
  # - No embedding of current state
  # - No persistent hidden state h_t
  # - No subgoal representation z_t

  Why fixed threshold exists: _estimate_alpha() at line 244 uses heuristic keyword matching, not learned embeddings:

  def _estimate_alpha(self, vision, language, context):
      alpha = 0.5  # FIXED base
      if "pick" in language.lower():  # Keyword heuristic
          alpha += 0.1

  3. BaseLearner (base_learner.py:168-175)

  # CURRENT - Best state tracking in YRSN
  self.step = 0
  self._last_alpha = self.config.quality_prior      # Previous α
  self._last_alpha_omega = self.config.quality_prior

  # encode_context() returns (context, binary_addr)
  # BUT: context is recomputed each call, not accumulated

  Partial support: Has _last_alpha but missing:
  - Persistent hidden state h_t
  - Subgoal representation z_t
  - Temporal integration mechanism

  4. GroundingValidator (grounding.py:91)

  # CURRENT - Discourse history only
  self._discourse_history: List[str] = []  # Just object IDs!

  # This is NOT an embedding - just ["cup_123", "block_456"]

  Why fixed threshold exists: grounding_confidence > 0.8 check exists because there's no learned embedding of discourse state.

  ---
  Summary: Why Fixed Thresholds ARE Necessary

  | β_t Input         | Status in YRSN       | Gap Filled By             |
  |-------------------|----------------------|---------------------------|
  | e_t (embedding)   | MISSING at NLP layer | Fixed slot-based scoring  |
  | h_{t-1} (hidden)  | MISSING everywhere   | Single scalar _last_alpha |
  | z_{t-1} (subgoal) | MISSING everywhere   | No temporal integration   |

  What Would Need to Exist for Learned β_t

  class CommandDecomposerWithSwitch:
      def __init__(self, ...):
          # NEW: State for β_t
          self.embedding_model = SentenceEncoder(dim=256)  # e_t
          self.hidden_state = np.zeros(128)                 # h_{t-1}
          self.subgoal_state = np.zeros(64)                 # z_{t-1}
          self.switching_network = MLP(256+128+64, 1)       # β_t

      def decompose(self, intent, scene_objects=None):
          # Compute e_t - REQUIRES embedding model
          e_t = self.embedding_model(intent.raw_text)  # ❌ Not available

          # Compute β_t
          β = self.switching_network(
              concat(e_t, self.hidden_state, self.subgoal_state)
          )

          # Update states - REQUIRES temporal tracking
          self.hidden_state = f(e_t, self.hidden_state)
          if β > 0.5:
              self.subgoal_state = g(e_t)  # Switch to new subgoal

  Conclusion: Fixed Thresholds Fill Real Gaps

  Fixed thresholds are NOT just "lazy engineering" - they fill these actual gaps:

  1. No embedding model at NLP layer → Can't compute e_t
  2. No persistent hidden state → Can't compute h_{t-1}
  3. No subgoal representation → Can't compute z_{t-1}
  4. No training data for β_t network
  5. No temporal integration architecture

  Recommendation

  Don't replace fixed thresholds until these exist:

  | Prerequisite               | Status     | Effort                           |
  |----------------------------|------------|----------------------------------|
  | Embedding model for intent | ❌ Missing | Medium (add SentenceTransformer) |
  | Hidden state tracking      | ❌ Missing | Low (add RNN cell)               |
  | Subgoal representation     | ❌ Missing | High (define subgoal space)      |
  | Training data for β_t      | ❌ Missing | High (collect switch examples)   |
  | Integration architecture   | ❌ Missing | Medium                           |

  Current fixed thresholds are the RIGHT choice given the infrastructure gaps.
ot
