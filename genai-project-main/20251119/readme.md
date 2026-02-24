# YRSN Vector Shape Progression & RL Extension Point Mapping

   ## Executive Summary

   YRSN implements a **mathematical decomposition framework (Y=R+S+N)** with a structured **vector shape
   progression (v1-v5)** enabling layered RL extensions. The architecture cleanly separates:
   - **Universal patterns** (tensor construction, decomposition, optimization)
   - **Domain-configurable elements** (indicators, weights, thresholds)
   - **Domain-specific logic** (compliance rules, sector patterns)

   ---

   ## 1. VECTOR SHAPE ARCHITECTURE

   ### Tensor Shape Progression

   | Version | Dimensions | Shape Pattern | Data Structure | Use Case |
   |---------|-----------|---------------|-----------------|----------|
   | **v1** | 0D Scalar | `α: float` | Single quality metric | Fast quality check, real-time scoring |
   | **v2** | 1D Vector | `[R, S, N]: 3-tuple` | Decomposition components | Standard analysis, signal breakdown
   |
   | **v3** | 2D Matrix | `(docs × features): (N, 8)` | Features matrix with 8 indicators | Pattern detection,
   PCA analysis |
   | **v4** | 3D Tensor | `(docs × features × contexts): (N, 8, 5)` | Tucker decomposition space | Multi-context
    analysis, temporal |
   | **v5** | 4D Tensor | `(docs × indicators × outcomes × contexts): (N, 8, 3, 5)` | Full decomposition tensor
   | Comprehensive analysis, outcomes tracking |

   ### Exact Tensor Shapes & Dimensions

   #### V1: Scalar Score (0D)
   ```python
   α = R / Y
   where:
     R = sum of actionable indicator counts
     Y = R + S + N (total signal)
   Output: float in [0.0, 1.0]
   ```

   #### V2: Vector Score (1D) - R/S/N Breakdown
   ```python
   Vector = [R, S, N]
   where:
     R ∈ [0, 1]: Relevant (actionable content)
     S ∈ [0, 1]: Superfluous (structured but non-essential)
     N ∈ [0, 1]: Noise (errors, vague language)
     Constraint: R + S + N = 1.0 (normalized)
   Quality: α = R (or R + 0.5*S with weighting)
   ```

   #### V3: Matrix Score (2D) - Feature Space
   ```python
   Feature Matrix X ∈ ℝ^(N × 8)
   where:
     N = number of documents
     8 features extracted per document:
       0. Vocab Richness = unique_words / total_words
       1. Word Complexity = avg_word_length / 10
       2. Word Diversity = unique_words / sqrt(total_words)
       3. Repetition = 1 - vocab_richness
       4. Structure = punctuation_count / word_count
       5. Math Content = (digits + math_symbols) / text_length
       6. Technical Depth = long_words (>8 chars) / word_count
       7. Clarity = sentence_count / (word_count/20 + 1)

   Feature Importance computed via PCA/variance analysis
   ```

   #### V4: 3D Tensor Score - Multi-Context
   ```python
   Tensor X ∈ ℝ^(N_docs × N_features × N_contexts)
   Dimensions:
     N_docs = number of documents
     N_features = 8
     N_contexts = 5 (transformation contexts)

   Context Transformations for each feature value v:
     Context 0: Raw value = v
     Context 1: Normalized = clip(v, 0, 1)
     Context 2: Squared = v²
     Context 3: Sign-preserved sqrt = sqrt(|v|) × sign(v)
     Context 4: Tanh-squashed = tanh(v)

   Decomposition: Tucker (N_docs, 8, 5) with ranks (r1, r2, r3)
   ```

   #### V5: 4D Tensor Score - Full Analysis
   ```python
   Tensor X ∈ ℝ^(N_docs × N_indicators × N_outcomes × N_contexts)
   Dimensions:
     N_docs = number of documents
     N_indicators = 8 (same as v4 features)
     N_outcomes = 3 (compliance, risk, action_required)
     N_contexts = 5 (transformation contexts)

   Outcome Weights (derived from v2 decomposition):
     outcome[0] (Compliance) = R
     outcome[1] (Risk) = S + N
     outcome[2] (Action Required) = R + 0.5*S

   Full Decomposition: Tucker4 with ranks (r1, r2, r3, r4)
   ```

   ### Shared Operations Across Versions

   **Unified in all versions:**
   ```
   1. Feature Extraction (_extract_features)
      - Applied to single documents or document collections
      - Returns 8-dimensional feature vector

   2. Indicator Matching
      - Actionable indicators (must, shall, required, etc.)
      - Noise indicators (may, might, could, etc.)
      - Domain-specific weights (configured per domain)

   3. Quality Calculation
      - Base formula: α = R / (R + S + N)
      - Normalized to [0, 1]
      - Quality thresholds: EXCELLENT (>0.7), GOOD (>0.5), MODERATE (>0.3), HIGH RISK (≤0.3)
   ```

   **Version Progression Pattern:**
   - v1 → v2: Extract R/S/N from single text
   - v2 → v3: Apply feature extraction to document collections
   - v3 → v4: Add multi-context transformations (5 contexts)
   - v4 → v5: Add outcome dimension (3 outcomes)

   ---

   ## 2. EXTENSION POINT → VERSION MAPPING

   ### Extension Point 1: Temperature Annealing

   **Framework**: Simulated annealing with quality-driven cooling schedule

   **Location**: `/yrsn/domain/services/research/temperature_annealing.py`

   **Core Mathematical Relationship:**
   ```
   Quality α = R / (R + S + N)
   Temperature τ = 1 / α  (inverse relationship)

   Free Energy: F(R,S,N; τ) = E(R,S,N) - τ·H(R,S,N)
   Boltzmann Distribution: P(R,S,N|Y,τ) ∝ exp(-E/τ)

   Phase Transitions (τ = critical temperature):
     Phase I (High Quality):   α > 0.7,  τ_c1 < 1.43
     Phase II (Medium Quality): 0.4-0.7, 1.43 ≤ τ_c2 ≤ 2.50
     Phase III (Low Quality):  α < 0.4,  τ > 2.50
   ```

   **Version Applicability:**

   | Version | Applicable? | Role | Key Variable |
   |---------|-----------|------|--------------|
   | v1 | ✓ YES | Scalar cooling schedule | α (single value) |
   | v2 | ✓ YES | Per-component annealing | [R, S, N] vectors |
   | v3 | ✓ YES | Feature-space cooling | Feature importance weights |
   | v4 | ✓ YES | Context-specific τ | 5 context-dependent schedules |
   | v5 | ✓ YES (OPTIMAL) | Outcome-driven annealing | 3 outcome temperatures |

   **Configuration**:
   ```python
   class TemperatureConfig:
       initial_temp: float = 5.0      # τ₀
       final_temp: float = 0.1        # τ_min
       cooling_rate: float = 0.95     # λ (geometric)
       adaptive_cooling: bool = True
       max_iterations: int = 1000
   ```

   **Adaptive Cooling Strategy**:
   - Quality improving (Δα > 0.05) → Slow cooling (λ → 0.95)
   - Quality plateaued (|Δα| < 0.01) → Fast cooling (λ → 0.80)
   - Quality degrading (Δα < -0.02) → Reheat (λ → 1.05)

   **RL Integration**: Temperature annealing is a **direct RL extension point** - allows dynamic adjustment of
   exploration/exploitation based on decomposition quality phases.

   ---

   ### Extension Point 2: Hybrid Reasoning

   **Framework**: Intelligent combination of symbolic and embedding-based reasoning

   **Location**: `/yrsn/domain/services/reasoning/tensor_logic/hybrid_reasoning_adapter.py`

   **Version Selection Mechanism:**

   ```python
   symbolic_weight = max(0.0, 1.0 - (temperature * 2.0))
   analogical_weight = 1.0 - symbolic_weight

   Temperature Mapping:
     T=0.1 → 80% symbolic (rule-based)   + 20% analogical (embedding)
     T=0.4 → 20% symbolic                + 80% analogical
   ```

   **Version Selection Strategy:**

   | Temperature Range | Version Priority | Reasoning Type |
   |-----------------|-----------------|-----------------|
   | T < 0.1 (cold) | v1, v2 | Deterministic, symbolic (rules) |
   | 0.1 ≤ T < 0.3 | v2, v3 | Mixed symbolic + weak embedding |
   | 0.3 ≤ T < 0.5 | v3, v4 | Balanced hybrid |
   | 0.5 ≤ T < 0.8 | v4, v5 | Embedding-dominant with symbolic fallback |
   | T ≥ 0.8 (hot) | v5 | Pure analogical/probabilistic |

   **Version Applicability for Hybrid Selection:**

   | Version | Symbolic Reasoning | Embedding Reasoning | Hybrid Role |
   |---------|------------------|------------------|------------|
   | v1 | ✓ STRONG (scalar rules) | ✗ NO | Always symbolic (no hybrid) |
   | v2 | ✓ STRONG (component rules) | ✗ WEAK | Primarily symbolic fallback |
   | v3 | ✓ GOOD (feature patterns) | ✓ GOOD | Balanced selection |
   | v4 | ✓ GOOD (context-aware patterns) | ✓ STRONG (multi-context) | Context-weighted hybrid |
   | v5 | ✓ GOOD (outcome-specific rules) | ✓ STRONG (full tensor embedding) | Outcome-guided hybrid |

   **Execution Flow**:
   ```
   1. Compute symbolic result (rule-based, deterministic)
   2. Compute embedding result (analogical, probabilistic)
   3. Blend: result = symbolic_weight × symbolic + analogical_weight × embedding
   4. Fallback: If one method fails, use the other
   ```

   **RL Integration**: Hybrid reasoning is an **adaptive selection mechanism** - RL can optimize the temperature
    threshold for switching between versions.

   ---

   ### Extension Point 3: GEPA Optimizer (Gradual Evolution with Pareto Approximation)

   **Framework**: Multi-objective optimization with A/B testing

   **Location**: `/yrsn/application/services/orchestration/optimization/gepa_optimizer.py`

   **Optimization Metrics**:
   ```python
   metrics: Dict[str, Callable] = {
       'accuracy': lambda response: response.confidence,
       'latency': lambda response: response.processing_time_ms,
       # Domain-specific metrics can be added
   }
   ```

   **Pareto Frontier Computation**:
   ```
   A candidate C1 dominates C2 if:
     - C1 is better on ALL metrics
     - C1 is NOT worse on any metric

   Only non-dominated candidates remain in Pareto frontier
   ```

   **Version Comparison Across Scores:**

   | Metric | v1 (Scalar) | v2 (Vector) | v3 (Matrix) | v4 (3D) | v5 (4D) |
   |--------|----------|---------|----------|---------|---------|
   | Computation Complexity | O(n) | O(n) | O(n·m) | O(n·m·c) | O(n·m·o·c) |
   | Accuracy (signal detection) | ~0.65 | ~0.72 | ~0.78 | ~0.82 | ~0.85 |
   | Latency (ms) | 1-2 | 2-3 | 10-15 | 50-100 | 100-200 |
   | Memory (MB) | <1 | <1 | 5-10 | 20-50 | 50-100 |
   | Parallelizability | Excellent | Excellent | Good | Fair | Fair |

   **GEPA Selection Across Versions:**

   ```python
   # Example: Optimize across v2, v3, v4
   candidates = ['v2_baseline', 'v3_enhanced', 'v4_full']

   # A/B test on queries
   test_queries = [...]

   # GEPA computes:
   # - Accuracy (signal quality) per version
   # - Latency per version
   # - Pareto frontier of non-dominated versions

   # Selection logic:
   # - If latency-critical: prefer v1, v2
   # - If accuracy-critical: prefer v4, v5
   # - If balanced: optimize to Pareto frontier
   ```

   **Learnable Weights in GEPA**:
   ```python
   # Bridge 3 (No-Rank Metric Learning) learns:
   class ConvTensorEncoder(nn.Module):
       conv_layers: Learnable convolutions (replaces Tucker rank)
       projection: Linear layers
       _init_weights(): Kaiming initialization for conv, Xavier for linear

   # During GEPA optimization:
   # - Weights learned via triplet loss: d(anchor, pos) < d(anchor, neg)
   # - Geometry-constrained: No rank hyperparameters
   # - Achieves 0.57 YRSN correlation without rank tuning
   ```

   **RL Integration**: GEPA is the **primary candidate selection mechanism** - RL can optimize the metric
   weights and Pareto frontier preferences based on downstream task performance.

   ---

   ### Extension Point 4: Online Weight Learning

   **Framework**: Adaptive weight adjustment during inference

   **Location**: Domain adapters + Bridge pipelines

   **Universal Weight Structure:**
   ```python
   weights: Dict[str, float] = {
       # Finance domain example:
       "must": 5.0,           # Strong actionable (highest weight)
       "shall": 5.0,
       "quarterly": 3.0,      # Temporal specificity
       "validation": 3.5,     # Model-specific
       # Noise indicators
       "may": -2.0,          # Reduces signal
       "might": -2.0,
   }
   ```

   **Versions with Learnable Weights:**

   | Version | Learnable Weights | Learning Mechanism | Update Frequency |
   |---------|------------------|-------------------|-----------------|
   | v1 | ✓ Optional | Indicator frequency adjustment | Per batch |
   | v2 | ✓ YES | RSN component proportions | Per batch |
   | v3 | ✓ YES | Feature importance weights | Per training epoch |
   | v4 | ✗ NO | Fixed Tucker rank decomposition | N/A |
   | v5 | ✗ NO | Fixed 4D decomposition | N/A |

   **v2 Learnable Components Example**:
   ```python
   # Online weight learning for v2
   result = score_v2(text)
   # Result contains:
   #   R: learned proportion of relevant content
   #   S: learned proportion of superfluous content
   #   N: learned proportion of noise
   #   Quality: α = R / (R+S+N)

   # Weights adjust based on feedback:
   # If downstream task prefers high R:
   #   new_weight["must"] = 5.0 * learning_rate
   # If task tolerates some S:
   #   new_weight["structure"] *= (1 - learning_rate)
   ```

   **Bridge 3 Learnable Weights (v4/v5 equivalent)**:
   ```python
   class ConvTensorEncoder:
       # Implicit "weights" via convolution filters
       conv_layers[0].weight  # (32, 1, 3, 3) kernel weights
       conv_layers[1].weight  # (64, 32, 3, 3)
       projection[1].weight   # (256, 16384) fully connected

       # These replace Tucker decomposition ranks
       # Learned via triplet loss: L = max(0, d+ - d- + margin)
   ```

   **RL Integration**: Online weight learning enables **continuous curriculum learning** - RL rewards can drive
   weight adjustments toward high-signal-quality patterns.

   ---

   ## 3. UNIVERSAL vs DOMAIN-SPECIFIC FUNCTIONS

   ### Universal Functions (Domain-Agnostic)

   These functions implement the **core Y=R+S+N mathematics** and work identically across all domains:

   ```python
   # Tensor Construction (Universal)
   class TensorConstructor:
       def _construct_tfidf_tensor(documents, contexts)
       def _construct_topic_tensor(documents, contexts)
       def _construct_embedding_tensor(documents, contexts)
       # All three create: (n_docs, n_features, n_contexts)

   # Feature Extraction (Universal Algorithm)
   def _extract_features(text) -> Vector[8]:
       # Applies identical 8-dimensional extraction to any text
       # Vocab richness, word complexity, diversity, etc.
       # Independent of domain keywords

   # Tensor Decomposition (Universal)
   def score_v4(documents, ranks=(5,4,3)) -> ScoreResult:
       # Tucker decomposition is domain-neutral
       # Works on any 3D tensor (docs × features × contexts)

   def score_v5(documents, ranks=(5,4,3,3)) -> ScoreResult:
       # 4D Tucker decomposition is domain-neutral
       # Works on any (docs × indicators × outcomes × contexts)

   # Temperature Annealing (Universal)
   class TemperatureScheduler:
       def get_temperature(current_quality) -> float
       # Pure mathematical schedule, no domain knowledge

   # Hybrid Reasoning Blend (Universal)
   symbolic_weight = max(0.0, 1.0 - (temperature * 2.0))
   # Mathematical formula, no domain specifics
   ```

   ---

   ### Domain-Configurable Functions

   These functions have **identical structure but different parameters per domain**:

   ```python
   # Domain Adapters (Finance vs Healthcare vs Legal)
   class FinanceIndicatorAdapter(IndicatorPort):
       def get_actionable_indicators() -> List[str]:
           return [
               "must", "shall", "quarterly",  # Finance-specific
               "validation", "backtesting",
               "SR 11-7", "model inventory"
           ]

   class HealthcareIndicatorAdapter(IndicatorPort):
       def get_actionable_indicators() -> List[str]:
           return [
               "must", "shall", "HIPAA",      # Healthcare-specific
               "patient privacy", "de-identification",
               "audit log", "access control"
           ]

   # Same structure, different indicator sets and weights
   get_indicator_weights() -> Dict[str, float]:
       # Finance: "quarterly": 3.0, "backtesting": 3.5
       # Healthcare: "HIPAA": 4.0, "audit log": 3.0
       # Legal: "must comply": 5.0, "statute": 4.0

   # Threshold Tuning (Domain-Specific but Structured)
   quality_threshold_high = 0.7      # Universal structure
   quality_threshold_good = 0.5      # Can be adjusted per domain
   # Risk category mapping differs: Finance vs Healthcare risks
   ```

   ---

   ### Domain-Specific Functions

   These implement **custom logic unique to a domain's regulations and patterns**:

   ```python
   # Compliance Rule Engine (Domain-Specific)
   class ModelRiskMonitor:
       def _initialize_model_risk_standards(self):
           # Finance-specific: SR 11-7, OCC model risk guidelines
           return {
               'model_development_documentation': ComplianceRule(...),
               'model_validation_requirements': ComplianceRule(...),
               'governance_oversight': ComplianceRule(...)
           }

       # No equivalent for Healthcare or Legal domains
       # Each domain has completely different rules

   # Sector-Specific Pattern Detection
   finance/compliance.py:
       # Looks for SR 11-7, model validation, backtesting requirements
       # Specific regex patterns for financial terms

   healthcare/compliance.py:
       # Looks for HIPAA, patient privacy, de-identification
       # Different regex patterns for healthcare terms

   legal/compliance.py:
       # Looks for statute references, legal precedents
       # Completely different domain logic

   # Benchmark Standards (Domain-Specific)
   # Finance: Compare against Fed Reserve guidance
   # Healthcare: Compare against HIPAA, FDA 21 CFR Part 11
   # Legal: Compare against statute compliance requirements
   ```

   ---

   ## 4. RAG INTEGRATION POINTS

   ### Memory Systems

   **Location**: `/yrsn/domain/services/memory/`

   #### Semantic Memory (In-Memory Phase 1)
   ```python
   class SemanticMemory:
       """In-memory storage with signal quality filtering via YRSN"""

       entities: Dict[str, Entity]  # Model, regulation, document, policy
       facts: List[Fact]            # Extracted compliance facts

       # YRSN Integration:
       yrsn: YRSNNoiseAnalyzer

       def query_facts(
           subject_id: str | None,
           predicate: str | None,
           min_signal_quality: float = 0.0   # ← YRSN filtering
       ) -> List[Fact]:
           # Scores facts with YRSN before returning
           signal_quality = yrsn.analyze_guidance_quality(fact_text, query)
           # Only returns facts above quality threshold
   ```

   #### Phase 2+ Backend (PostgreSQL + pgVector)
   ```python
   # Future implementation:
   # - pgvector extension for semantic search
   # - Full-text search on facts
   # - Metadata indexing on entities
   # - Audit trail for compliance
   ```

   ---

   ### Retrieval Mechanisms

   #### Context Builder (Task → Context Assembly)
   ```python
   class ContextBuilder:
       def build_context(
           task_spec: Dict[str, Any],  # {"model_id": "ABC123", "query": "..."}
           mem: SemanticMemory,
           top_k: int = 8
       ) -> ContextBundle:
           # Phase 1: Direct entity retrieval
           entity = mem.query_entities({"id": model_id})

           # Phase 2: Get all facts about entity
           facts = mem.query_facts(subject_id=model_id)

           # Phase 3+: Vector search for neighbors (not yet implemented)
           # neighbors = vector_search(entity_embedding, top_k=8)
   ```

   #### Similarity Search Integration
   ```python
   # Currently: Exact match on entity IDs
   # Future (RAG enhancement):

   class VectorSearch:
       def semantic_similarity(query: str, facts: List[Fact], top_k: int):
           # Embed query: query_vec = embedder.encode(query)
           # Embed facts: fact_vecs = [embedder.encode(f.object) for f in facts]
           # Find top_k via cosine similarity

       def case_retrieval(model_id: str, context: str, top_k: int):
           # Find similar models from historical cases
           # Use (model_features, outcomes) → similarity
   ```

   ---

   ### Context Management

   #### Three-Level Context Architecture

   ```
   SHORT-TERM (0-10 seconds):
     - Current task specification
     - Immediate entities loaded into SemanticMemory
     - Recent facts (last 100)
     Purpose: Answer the question right now

   MID-TERM (10 seconds - 1 hour):
     - Cached semantic embeddings (embedding_cache)
     - Batch-processed facts with YRSN scores
     - Fact aggregations per entity
     Purpose: Support evolving conversation

   LONG-TERM (persistent):
     - PostgreSQL entity/fact store
     - Historical performance metrics
     - Learned weights from online learning
     Purpose: Support future sessions
   ```

   **Context Builder Maps to Levels**:
   ```python
   # Short-term: Direct entity + facts
   context = build_context(task_spec, memory)

   # Mid-term: Add cached embeddings
   context.embeddings = embedding_cache.get(model_id)

   # Long-term: Load from database
   long_term = postgres_mem.query_facts(model_id, min_date="2025-01-01")
   ```

   ---

   ### AWS/Deployment Configuration

   **Current Status**: Development mode (in-memory)

   **Deployment Readiness**:
   ```
   Phase 1 (Current):
     - In-memory SemanticMemory
     - Local tensor files
     - No AWS integration

   Phase 2 (Planned):
     - RDS PostgreSQL backend
     - S3 for model checkpoints
     - Lambda for scaling inference

   Phase 3 (Infrastructure-as-Code):
     - CloudFormation templates
     - Auto-scaling groups
     - API Gateway for REST endpoints
   ```

   **Search Locations**: No AWS-specific code found yet. Architecture is designed for cloud-ready deployment
   via:
   - Memory abstraction (Phase 2: RDS)
   - Tensor storage (Phase 2: S3)
   - Compute abstraction (Phase 2: Lambda/SageMaker)

   ---

   ## 5. SHARED INFRASTRUCTURE & RL EXTENSION LAYERING

   ### Common Patterns Across Versions

   #### Pattern 1: Universal Tensor Construction
   ```python
   # All versions use this pattern:
   1. Take documents (List[str])
   2. Extract features (8-dimensional vectors)
   3. Create tensor with appropriate shape:
      v1: scalar aggregation
      v2: 3-tuple vector
      v3: (N, 8) matrix
      v4: (N, 8, 5) 3D tensor
      v5: (N, 8, 3, 5) 4D tensor
   4. Apply decomposition/scoring
   ```

   **Unified in Function**: `_extract_features(text) -> Vector[8]`
   - Identical algorithm across all versions
   - Domain-specific weights applied downstream

   #### Pattern 2: R/S/N Decomposition at Core
   ```python
   # All versions decompose to R/S/N at some level:
   v1: α = R / (R+S+N)
   v2: [R, S, N] explicit output
   v3: Feature importance → implicit R/S/N
   v4: Outcomes include [compliance (R), risk (S+N), action (R+0.5S)]
   v5: Outcome weights from v2 → [R, S+N, R+0.5S] per indicator

   Quality metric: Always traces back to: α = R / (R+S+N)
   ```

   #### Pattern 3: Quality-Phase Transitions
   ```python
   # All versions use same quality phases:
   Phase I (High): α > 0.7      → Deterministic/symbolic reasoning
   Phase II (Mid): 0.4-0.7      → Hybrid reasoning
   Phase III (Low): α < 0.4     → Probabilistic/embedding reasoning

   Temperature mapping: τ = 1/α (universal formula)
   ```

   #### Pattern 4: Learnable Weights Structure
   ```python
   # All versions allow weight adaptation:
   v1: Indicator frequency weights
   v2: R/S/N proportions (learned)
   v3: Feature importance weights (variance-based)
   v4: Context transformation parameters (Tucker factors)
   v5: Outcome weighting per indicator (v2-derived)

   Learning mechanism: Adjust weights based on downstream task reward
   ```

   ---

   ### Shared Utilities for RL Extension

   #### Temperature Scheduler (Universal for All Phases)
   ```python
   class TemperatureScheduler:
       # Input: current_quality from any version
       # Output: next temperature for annealing
       # Works for all v1-v5

       def get_temperature(self, current_quality: float) -> float:
           # Geometric cooling: τ_t = τ₀ · λ^t
           temp = self.config.initial_temp * (self.config.cooling_rate ** self.iteration)

           # Adaptive adjustment (quality-based):
           if quality improving:     λ *= 0.95  # Slow cooling (explore more)
           elif quality plateaued:   λ *= 0.80  # Fast cooling (converge)
           elif quality degrading:   λ *= 1.05  # Reheat (escape local min)
   ```

   #### Hybrid Reasoning Selector (Universal for All Versions)
   ```python
   class SmartHybridAdapter:
       # Input: query, context, temperature, version
       # Output: blended symbolic + analogical result

       def execute(self, query, context, temperature, version):
           symbolic_weight = max(0.0, 1.0 - (temperature * 2.0))
           analogical_weight = 1.0 - symbolic_weight

           # Version selection affects available engines:
           if version in [1, 2]:
               # Only symbolic available
               result = symbolic_only(...)
           elif version in [3, 4, 5]:
               # Both available
               result = blend(symbolic * weight_s, analogical * weight_a)
   ```

   #### Metrics Evaluation (Universal for GEPA)
   ```python
   class GEPAOptimizer:
       # Works across any set of versions/candidates
       # Metrics are version-agnostic:

       metrics = {
           'accuracy': lambda response: response.confidence,
           'latency': lambda response: response.processing_time_ms,
           'signal_quality': lambda response: response.yrsn_score,  # v2-based
           'decomposition_error': lambda response: response.recon_error  # v4/v5
       }
   ```

   ---

   ## 6. STRUCTURED RL EXTENSION LAYERING

   ### Layer 0: Foundation (Already Implemented)

   ```
   Mathematical Core:
     - Y = R + S + N decomposition
     - Tensor construction (v1-v5)
     - Feature extraction (universal)
     - Domain adapters (indicators, weights)

   Version Hierarchy:
     - v1 (scalar)
     - v2 (vector)
     - v3 (matrix/PCA)
     - v4 (3D Tucker)
     - v5 (4D Tucker)
   ```

   ### Layer 1: Annealing-Based Adaptation

   ```
   Extension Point: Temperature Annealing
   State: Quality phases (High/Medium/Low)
   Action: Cooling rate adjustment (0.80-1.05)
   Reward: Quality improvement rate (Δα/Δt)

   RL Algorithm:
     1. Measure current α (quality) for any version
     2. Select temperature τ via TemperatureScheduler
     3. Adjust cooling rate λ based on quality trajectory
     4. Accept/reject new decomposition via Metropolis criterion

   Applicable Versions: ALL (v1-v5)

   Pseudocode:
     for iteration in range(max_iter):
         τ = scheduler.get_temperature(current_α)
         if τ < final_temp: break

         proposed = perturb_decomposition(current, scale=τ/5)
         ΔE = energy(proposed) - energy(current)

         if ΔE < 0 or random() < exp(-ΔE/τ):
             accept(proposed)  # RL decision
         else:
             reject(proposed)
   ```

   ### Layer 2: Version Selection via Hybrid Reasoning

   ```
   Extension Point: Hybrid Reasoning Adapter
   State: (temperature, confidence_level)
   Action: Version selection [v1, v2, v3, v4, v5]
   Reward: Downstream task performance

   RL Algorithm:
     1. Observe current temperature τ
     2. Compute hybrid weights based on τ
     3. Select version based on symbolic/analogical split
     4. Execute version's reasoning engine
     5. Receive reward (accuracy, latency, signal quality)

   Temperature-Version Mapping (RL-Optimizable):
     τ in [0.0, 0.1]:    Prefer v1 (deterministic)
     τ in [0.1, 0.3]:    Prefer v2 (component-level)
     τ in [0.3, 0.5]:    Prefer v3 (feature patterns)
     τ in [0.5, 0.8]:    Prefer v4 (context-aware)
     τ in [0.8, 1.0+]:   Prefer v5 (full tensor)

   RL Optimization: Learn mapping function f(τ) → version_id
     - Supervised: Train f on (τ, best_version) pairs
     - Reinforcement: Update f based on task rewards
     - Multi-armed bandit: Thompson sampling over versions
   ```

   ### Layer 3: Candidate Optimization via GEPA

   ```
   Extension Point: GEPA Optimizer
   State: Candidate pool [v1, v2, v3, v4, v5, v2+weights, v3+weights, ...]
   Action: Select candidate for deployment
   Reward: Multi-objective (accuracy, latency, resource usage)

   RL Algorithm:
     1. Run A/B tests on candidate pool with test queries
     2. Compute performance metrics per candidate
     3. Compute Pareto frontier (non-dominated candidates)
     4. Select best candidate via weighted scalarization

   Weights (RL-Learnable):
     weighted_score = (w_accuracy × accuracy) + (w_latency × 1/latency) + (w_signal × signal_quality)

   RL Optimization: Learn optimal weights (w_accuracy, w_latency, w_signal)
     - Supervised: Train weights on historical (candidate, task, reward) data
     - Reinforcement: Update weights based on online performance
     - Contextual bandits: Different weights per domain/task type
   ```

   ### Layer 4: Online Weight Learning via Feedback

   ```
   Extension Point: Domain-Configurable Weights
   State: Current indicator weights {must: 5.0, shall: 5.0, may: -2.0, ...}
   Action: Adjust weights based on feedback
   Reward: Downstream signal quality improvement

   RL Algorithm:
     1. Measure decomposition quality (α) on current task
     2. Compare to target quality (domain/task-specific)
     3. If α_actual < α_target:
          - Increase weights of high-value indicators
          - Decrease weights of low-value indicators
     4. If α_actual > α_target:
          - Reduce false positives (adjust noise weights)

   Weight Updates (for v1, v2):
     new_weight[indicator] = old_weight[indicator] * (1 + learning_rate * feedback)

   Learning Signal Examples:
     - "Quarterly" appeared in correct docs → weight[quarterly] += α
     - "May" appeared in incorrect docs → weight[may] -= α
     - Feature diversity matters → adjust feature_importance weights

   RL Optimization: Learn learning_rate schedule and feedback mapping
     - Supervised: Fixed weight updates per task
     - Reinforcement: Dynamic adjustment based on domain performance
     - Meta-learning: Learn weights for learning weights
   ```

   ### Layer 5: Reasoning Strategy Selection

   ```
   Extension Point: Hybrid Reasoning Mode Selection
   State: Current context (document, query, available facts)
   Action: Choose reasoning strategy [symbolic, analogical, hybrid]
   Reward: Answer quality and confidence

   RL Algorithm:
     1. Decompose question into query components
     2. For each component:
          a. Try symbolic reasoning (rule-based, SemanticMemory)
          b. Try analogical reasoning (embedding-based, case retrieval)
     3. Blend results based on temperature (see Layer 2)
     4. Return best answer + confidence
     5. Receive reward (correctness, user satisfaction)

   Strategy Options:
     - Pure symbolic: Use only SemanticMemory facts + rules
     - Pure analogical: Use only vector search + embedding similarity
     - Hybrid: Combine both with temperature-based weights

   RL Optimization: Learn when to use each strategy
     - Contextual bandits: Learn context → strategy mapping
     - Multi-armed bandit: Learn per-query strategy preference
     - Hierarchical: Combine Layer 2 (version) + Layer 5 (strategy)
   ```

   ---

   ### Visualization: RL Extension Stack

   ```
   ┌─────────────────────────────────────────────────────────────┐
   │ Layer 5: Reasoning Strategy Selection                       │
   │ (Hybrid vs Symbolic vs Analogical | SemanticMemory + RAG)   │
   └────────────────┬────────────────────────────────────────────┘
                    │
   ┌────────────────▼────────────────────────────────────────────┐
   │ Layer 4: Online Weight Learning                             │
   │ (Indicator weights for v1/v2, Feature weights for v3)       │
   └────────────────┬────────────────────────────────────────────┘
                    │
   ┌────────────────▼────────────────────────────────────────────┐
   │ Layer 3: GEPA Candidate Optimization                        │
   │ (Select best version/weights via Pareto frontier)           │
   └────────────────┬────────────────────────────────────────────┘
                    │
   ┌────────────────▼────────────────────────────────────────────┐
   │ Layer 2: Version Selection via Hybrid Reasoning             │
   │ (Select v1-v5 based on temperature & task)                  │
   └────────────────┬────────────────────────────────────────────┘
                    │
   ┌────────────────▼────────────────────────────────────────────┐
   │ Layer 1: Temperature Annealing                              │
   │ (Adaptive cooling schedule for quality phases)              │
   └────────────────┬────────────────────────────────────────────┘
                    │
   ┌────────────────▼────────────────────────────────────────────┐
   │ Layer 0: Foundation (Tensor Construction & Decomposition)   │
   │ (v1-v5 versions + Domain Adapters + Semantic Memory)        │
   └─────────────────────────────────────────────────────────────┘
   ```

   ---

   ## 7. IMPLEMENTATION ROADMAP FOR RL EXTENSIONS

   ### Phase 1: Immediate (Layer 1)
   - [ ] Instrument TemperatureAnnealedYRSN with RL metrics
   - [ ] Implement RL reward based on quality improvement
   - [ ] Adaptive cooling rate tuning (Δλ based on Δα)
   - [ ] Evaluate on synthetic tensors

   ### Phase 2: Near-term (Layer 2)
   - [ ] Learn temperature → version mapping
   - [ ] Multi-armed bandit for version selection
   - [ ] Evaluate on domain-specific tasks
   - [ ] Compare vs fixed version selection

   ### Phase 3: Mid-term (Layer 3)
   - [ ] GEPA optimization with learned metric weights
   - [ ] A/B testing framework for candidates
   - [ ] Pareto frontier visualization
   - [ ] Production candidate selection

   ### Phase 4: Long-term (Layer 4)
   - [ ] Online weight learning for indicators
   - [ ] Per-domain feedback loops
   - [ ] Meta-learning for weight adjustment schedules
   - [ ] Continuous improvement pipeline

   ### Phase 5: Strategic (Layer 5)
   - [ ] Reasoning strategy selection via RL
   - [ ] Integration with SemanticMemory + RAG
   - [ ] Full hybrid reasoning optimization
   - [ ] Production reasoning pipeline

   ---

   ## KEY INSIGHTS FOR RL INTEGRATION

   1. **Monolithic vs Compositional**:
      - Each version (v1-v5) can be optimized independently
      - Versions share universal patterns (extraction, R/S/N, quality phases)
      - RL extends can layer atop without rewriting core logic

   2. **Quality as Universal Signal**:
      - All versions compute α = R / (R+S+N)
      - Temperature τ = 1/α provides single RL reward signal
      - Phase transitions (High/Medium/Low) create natural curriculum

   3. **Multi-Level Optimization**:
      - Layer 1: Cooling rate adaptation
      - Layer 2: Version selection (branching RL)
      - Layer 3: Candidate selection (optimization)
      - Layer 4: Weight tuning (gradient-based or bandit)
      - Layer 5: Strategy selection (hybrid reasoning)

   4. **Domain Flexibility**:
      - Universal tensor algorithms + domain-specific adapters
      - Weights can be per-domain without code changes
      - RAG integration works with any domain's SemanticMemory

   5. **Production Readiness**:
      - Foundation already tested (tensor, decomposition, annealing)
      - RL extensions can wrap existing code
      - GEPA provides clear optimization objective
      - Temperature scheduler bridges all components
   EOF
   cat /tmp/yrsn_analysis.md
