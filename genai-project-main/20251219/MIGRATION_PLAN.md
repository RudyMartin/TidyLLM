# YRSN Migration Plan

## Current Status

**Migration Progress:** 23% complete (39 of ~169 files)

### Completed Modules
- ✅ Core temperature mapping
- ✅ Core decomposition (basic: collapse.py, pca.py)
- ✅ Core memory (SDM, Hopfield, EWC, Replay)
- ✅ Core filtering (FIF, MdFIF, MvFIF)
- ✅ Core reservoir (ESN, SignalSeparator)
- ✅ Core optimization (EGGROLL, Gradient, Adaptive)
- ✅ Contracts (Observation, YrsnState, YrsnEvent, ContextBlock)
- ✅ Ports (ILLMClient, IStorage, IEventBus)
- ✅ Adapters (MockLLM, Anthropic, OpenAI)
- ✅ Runtime (LifecycleNode)

### Missing Modules (by priority)
1. **Core/Decomposition** - 16 more files needed
2. **Neural** - 16 files (0% complete)
3. **Strategies** - 15 files (0% complete)
4. **OOD Detection** - 10 files (0% complete)
5. **Models** - 19 files (0% complete)
6. **Hardware** - 3 files (0% complete)
7. **Reasoning** - 4 files (0% complete)

---

## Goal 1: Complete Core Decomposition Migration

**Objective:** Migrate all 18 decomposition files to provide complete Y=R+S+N analysis.

**Source:** `/Users/rudy/github/yrsn-context/src/yrsn_context/core/decomposition/`
**Target:** `/Users/rudy/github/yrsn/src/yrsn/core/decomposition/`

### Files to Migrate

| File | Purpose | Priority | Dependencies |
|------|---------|----------|--------------|
| `yrsn_decomposition.py` | Main decomposition framework | P0 | None |
| `collapse_detection.py` | Full collapse detection | P0 | None |
| `robust_pca.py` | Full Robust PCA | P0 | numpy |
| `phase_transitions.py` | BBP thresholds, phase detection | P1 | numpy |
| `spectral_analysis.py` | Marchenko-Pastur, eigenvalue analysis | P1 | numpy, scipy |
| `adaptive_scaling.py` | Adaptive scaling manager | P1 | phase_transitions |
| `calibration_manager.py` | Drift-reset mechanism | P1 | None |
| `triplet_loss.py` | Contrastive learning losses | P2 | torch (optional) |
| `supervised_heads.py` | Learnable projection heads | P2 | torch (optional) |
| `outcome_supervision.py` | Weak supervision from outcomes | P2 | None |
| `semantic_labeler.py` | Semantic labeling for R/S/N | P2 | None |
| `rsn_trainer.py` | Training loop for projections | P2 | torch (optional) |
| `validation.py` | Validation and ground truth | P2 | None |
| `constraint_taxonomy.py` | Three-tier constraint system | P3 | None |
| `collapse_monitor.py` | Audit trail, early warning | P3 | collapse_detection |
| `curriculum_learning.py` | Memristor-inspired curriculum | P3 | None |
| `pillar_configs.py` | Unified pillar configurations | P3 | All pillars |
| `pillar_adapters.py` | Pillar integration pipeline | P3 | All pillars |

### Detailed Todos

#### Phase 1.1: Core Decomposition (P0)
- [ ] **1.1.1** Copy `yrsn_decomposition.py` to decomposition/
- [ ] **1.1.2** Update imports from `yrsn_context` to `yrsn`
- [ ] **1.1.3** Merge with existing `__init__.py` (keep DecompositionScore, add ResearchFramework)
- [ ] **1.1.4** Verify `DecompositionScore`, `PaperAnalysis`, `ResearchFramework` exports
- [ ] **1.1.5** Test: `from yrsn.core.decomposition import ResearchFramework`

#### Phase 1.2: Enhanced Collapse Detection (P0)
- [ ] **1.2.1** Compare existing `collapse.py` with source `collapse_detection.py`
- [ ] **1.2.2** Add missing classes: `HiveMindDetector`, `HiveMindAnalysis`
- [ ] **1.2.3** Add missing function: `generate_with_r_boost()`
- [ ] **1.2.4** Update `__init__.py` exports
- [ ] **1.2.5** Test HiveMind detection functionality

#### Phase 1.3: Phase Transitions - BBP Thresholds (P1)
- [ ] **1.3.1** Copy `phase_transitions.py` to decomposition/
- [ ] **1.3.2** Update imports
- [ ] **1.3.3** Add exports to `__init__.py`:
  - `compute_bbp_threshold`
  - `compute_bbp_threshold_inverse`
  - `quality_supports_rank`
  - `get_rank_headroom`
  - `BBPThresholdTable`
  - `PhaseTransitionDetector`
  - `TransitionType`
  - `TransitionEvent`
  - `RankRecommendation`
- [ ] **1.3.4** Test: BBP threshold computation
- [ ] **1.3.5** Test: Phase transition detection

#### Phase 1.4: Spectral Analysis - Marchenko-Pastur (P1)
- [ ] **1.4.1** Copy `spectral_analysis.py` to decomposition/
- [ ] **1.4.2** Update imports
- [ ] **1.4.3** Add exports:
  - `compute_mp_bounds`
  - `mp_density`
  - `EigenvalueSnapshot`
  - `EigenvalueTracker`
  - `MPValidationResult`
  - `MarchenkoPasturValidator`
  - `SNRAnalysisResult`
  - `SignalNoiseAnalyzer`
  - `quick_spectral_analysis`
  - `estimate_quality_from_hidden_states`
- [ ] **1.4.4** Test: MP bounds computation
- [ ] **1.4.5** Test: Eigenvalue tracking

#### Phase 1.5: Adaptive Scaling (P1)
- [ ] **1.5.1** Copy `adaptive_scaling.py` to decomposition/
- [ ] **1.5.2** Update imports (depends on phase_transitions)
- [ ] **1.5.3** Add exports:
  - `AdaptiveScalingConfig`
  - `AdaptiveState`
  - `AdaptiveResult`
  - `AdaptiveScalingManager`
  - `create_adaptive_manager`
  - `get_adaptive_temperature`
  - `get_adaptive_rank`
  - `quick_adaptive_analysis`
- [ ] **1.5.4** Test: Adaptive temperature computation
- [ ] **1.5.5** Test: Integration with phase transitions

#### Phase 1.6: Calibration Manager (P1)
- [ ] **1.6.1** Copy `calibration_manager.py` to decomposition/
- [ ] **1.6.2** Update imports
- [ ] **1.6.3** Add exports:
  - `CalibrationManager`
  - `CalibrationConfig`
  - `CalibrationState`
  - `CalibrationMetrics`
  - `create_calibrated_projections`
  - `SupervisedCalibrationManager`
  - `create_supervised_calibration`
- [ ] **1.6.4** Test: Calibration state management
- [ ] **1.6.5** Test: Drift-reset mechanism

#### Phase 1.7: Supervised Learning Components (P2)
- [ ] **1.7.1** Copy `triplet_loss.py` to decomposition/
- [ ] **1.7.2** Copy `supervised_heads.py` to decomposition/
- [ ] **1.7.3** Copy `outcome_supervision.py` to decomposition/
- [ ] **1.7.4** Copy `semantic_labeler.py` to decomposition/
- [ ] **1.7.5** Copy `rsn_trainer.py` to decomposition/
- [ ] **1.7.6** Copy `validation.py` to decomposition/
- [ ] **1.7.7** Update all imports
- [ ] **1.7.8** Add torch as optional dependency in pyproject.toml
- [ ] **1.7.9** Add exports with lazy loading for torch dependencies
- [ ] **1.7.10** Test: Import without torch installed (should not fail)
- [ ] **1.7.11** Test: RSNLabel, DistanceFunction, MiningStrategy enums

#### Phase 1.8: Constraint System (P3)
- [ ] **1.8.1** Copy `constraint_taxonomy.py` to decomposition/
- [ ] **1.8.2** Copy `collapse_monitor.py` to decomposition/
- [ ] **1.8.3** Copy `curriculum_learning.py` to decomposition/
- [ ] **1.8.4** Update imports
- [ ] **1.8.5** Add exports:
  - `ConstraintTier`
  - `MemristorModel`
  - `TIER_TO_MEMRISTOR`
  - `TieredConstraint`
  - `ConstraintTaxonomy`
  - `create_yrsn_taxonomy`
  - `check_yrsn_constraints`
  - `VirtualMemristor`
  - `CurriculumManager`
  - `CollapseMonitor`
  - `EarlyWarning`
- [ ] **1.8.6** Test: Constraint checking
- [ ] **1.8.7** Test: Curriculum progression

#### Phase 1.9: Pillar Integration (P3)
- [ ] **1.9.1** Copy `pillar_configs.py` to decomposition/
- [ ] **1.9.2** Copy `pillar_adapters.py` to decomposition/
- [ ] **1.9.3** Update imports
- [ ] **1.9.4** Add exports:
  - `Pillar1Config`, `Pillar2Config`, `Pillar3Config`
  - `YRSNPillarConfig`
  - `create_default_configs`
  - `IntegratedPipeline`
  - `quick_integrated_analysis`
- [ ] **1.9.5** Test: Full integrated pipeline

#### Phase 1.10: Final Integration
- [ ] **1.10.1** Update `yrsn/core/decomposition/__init__.py` with all exports
- [ ] **1.10.2** Update `yrsn/core/__init__.py` to re-export key items
- [ ] **1.10.3** Run full import test
- [ ] **1.10.4** Commit with message: "feat(decomposition): Complete Pillar 1-4 migration"

---

## Goal 2: Migrate Neural Modules

**Objective:** Migrate all neural processing components for learned R/S/N separation.

**Source:** `/Users/rudy/github/yrsn-context/src/yrsn_context/neural/`
**Target:** `/Users/rudy/github/yrsn/src/yrsn/neural/`

### Files to Migrate

| File | Purpose | Priority | Dependencies |
|------|---------|----------|--------------|
| `__init__.py` | Module exports | P0 | All neural modules |
| `retriever.py` | Neural retriever | P0 | torch, transformers |
| `latent_combiner.py` | Latent space combination | P0 | torch |
| `strategy_router.py` | Strategy routing | P1 | torch |
| `ctc_strategy_classifier.py` | CTC classifier | P1 | torch |
| `refinement/__init__.py` | Refinement package | P1 | None |
| `refinement/iterative.py` | Iterative refinement | P1 | torch |
| `refinement/noisy.py` | Noisy refinement | P1 | torch |
| `ctm/__init__.py` | CTM package | P2 | None |
| `ctm/loss.py` | CTM loss functions | P2 | torch |
| `ctm/nlm.py` | Neural language model | P2 | torch, transformers |
| `ctm/synapse.py` | Synapse management | P2 | torch |
| `ctm/synchronization.py` | Synchronization | P2 | torch |
| `training/__init__.py` | Training package | P2 | None |
| `training/budget.py` | Training budget | P2 | None |
| `training/snap.py` | SNAP training | P2 | torch |

### Detailed Todos

#### Phase 2.1: Module Structure Setup
- [ ] **2.1.1** Create directory structure:
  ```
  src/yrsn/neural/
  ├── __init__.py
  ├── retriever.py
  ├── latent_combiner.py
  ├── strategy_router.py
  ├── ctc_strategy_classifier.py
  ├── refinement/
  │   ├── __init__.py
  │   ├── iterative.py
  │   └── noisy.py
  ├── ctm/
  │   ├── __init__.py
  │   ├── loss.py
  │   ├── nlm.py
  │   ├── synapse.py
  │   └── synchronization.py
  └── training/
      ├── __init__.py
      ├── budget.py
      └── snap.py
  ```
- [ ] **2.1.2** Update pyproject.toml with neural dependencies:
  ```toml
  [project.optional-dependencies]
  neural = [
      "torch>=2.0.0",
      "transformers>=4.30.0",
      "sentence-transformers>=2.2.0",
  ]
  ```

#### Phase 2.2: Core Neural Components (P0)
- [ ] **2.2.1** Copy `retriever.py` to neural/
- [ ] **2.2.2** Update imports from `yrsn_context` to `yrsn`
- [ ] **2.2.3** Add lazy imports for torch to handle missing dependency gracefully
- [ ] **2.2.4** Copy `latent_combiner.py` to neural/
- [ ] **2.2.5** Update imports
- [ ] **2.2.6** Create neural/__init__.py with exports:
  ```python
  __all__ = [
      "NeuralRetriever",
      "LatentCombiner",
      "YRSNLatentCombiner",
  ]
  ```
- [ ] **2.2.7** Test: Import without torch (should warn, not fail)
- [ ] **2.2.8** Test: Import with torch (full functionality)

#### Phase 2.3: Strategy Components (P1)
- [ ] **2.3.1** Copy `strategy_router.py` to neural/
- [ ] **2.3.2** Update imports
- [ ] **2.3.3** Copy `ctc_strategy_classifier.py` to neural/
- [ ] **2.3.4** Update imports
- [ ] **2.3.5** Add exports:
  - `StrategyRouter`
  - `CTCStrategyClassifier`
  - `StrategyPrediction`
- [ ] **2.3.6** Test: Strategy routing logic

#### Phase 2.4: Refinement Module (P1)
- [ ] **2.4.1** Copy `refinement/__init__.py`
- [ ] **2.4.2** Copy `refinement/iterative.py`
- [ ] **2.4.3** Copy `refinement/noisy.py`
- [ ] **2.4.4** Update all imports
- [ ] **2.4.5** Add exports:
  - `IterativeRefinement`
  - `NoisyRefinement`
  - `RefinementConfig`
  - `RefinementResult`
- [ ] **2.4.6** Test: Iterative refinement loop
- [ ] **2.4.7** Test: Noisy refinement with temperature

#### Phase 2.5: CTM Module (P2)
- [ ] **2.5.1** Copy entire `ctm/` directory
- [ ] **2.5.2** Update all imports in:
  - `ctm/__init__.py`
  - `ctm/loss.py`
  - `ctm/nlm.py`
  - `ctm/synapse.py`
  - `ctm/synchronization.py`
- [ ] **2.5.3** Add exports:
  - `CTMLoss`
  - `NeuralLanguageModel`
  - `SynapseManager`
  - `SynchronizationModule`
- [ ] **2.5.4** Test: CTM loss computation
- [ ] **2.5.5** Test: NLM forward pass

#### Phase 2.6: Training Module (P2)
- [ ] **2.6.1** Copy entire `training/` directory
- [ ] **2.6.2** Update imports
- [ ] **2.6.3** Add exports:
  - `TrainingBudget`
  - `BudgetConfig`
  - `SNAPTrainer`
  - `SNAPConfig`
- [ ] **2.6.4** Test: Budget computation
- [ ] **2.6.5** Test: SNAP training loop

#### Phase 2.7: Integration
- [ ] **2.7.1** Update neural/__init__.py with complete exports
- [ ] **2.7.2** Add `yrsn.neural` to main `yrsn/__init__.py`
- [ ] **2.7.3** Create integration test: retriever → combiner → router
- [ ] **2.7.4** Test: Full neural pipeline without torch
- [ ] **2.7.5** Test: Full neural pipeline with torch
- [ ] **2.7.6** Commit: "feat(neural): Add neural processing modules"

---

## Goal 3: Migrate Strategies

**Objective:** Migrate all 4 strategy implementations for context processing.

**Source:** `/Users/rudy/github/yrsn-context/src/yrsn_context/strategies/`
**Target:** `/Users/rudy/github/yrsn/src/yrsn/strategies/`

### Strategy Overview

| Strategy | Purpose | Files | Key Classes |
|----------|---------|-------|-------------|
| **Bit-Slicing** | Multi-precision context levels | 3 | BitSlicingDecomposer, SlicedContext |
| **Hierarchical Decomposition** | Tree-based decomposition | 3 | HierarchicalDecomposer, HierarchyNode |
| **Iterative Refinement** | Progressive refinement | 3 | IterativeEngine, ResidualTracker |
| **Layered Stack** | Complementary layers | 3 | LayeredStack, ComplementaryLayer |
| **Self-Calibration** | Auto-calibration | 1 | SelfCalibrator |

### Detailed Todos

#### Phase 3.1: Module Structure Setup
- [ ] **3.1.1** Create directory structure:
  ```
  src/yrsn/strategies/
  ├── __init__.py
  ├── base.py                    # Abstract strategy base
  ├── bit_slicing/
  │   ├── __init__.py
  │   ├── decomposer.py
  │   ├── retrieval.py
  │   └── sliced_context.py
  ├── hierarchical/
  │   ├── __init__.py
  │   ├── decomposer.py
  │   ├── processor.py
  │   └── query.py
  ├── iterative/
  │   ├── __init__.py
  │   ├── engine.py
  │   ├── residual.py
  │   └── retrievers.py
  ├── layered/
  │   ├── __init__.py
  │   ├── complementary.py
  │   ├── interface.py
  │   └── stack.py
  └── calibration/
      ├── __init__.py
      └── calibrator.py
  ```

#### Phase 3.2: Strategy Base Class
- [ ] **3.2.1** Create `strategies/base.py` with:
  ```python
  from abc import ABC, abstractmethod
  from typing import List, Any
  from yrsn.contracts import ContextBlock

  class BaseStrategy(ABC):
      """Abstract base for all YRSN strategies."""

      @abstractmethod
      def process(self, query: str, context: List[ContextBlock]) -> Any:
          """Process context using this strategy."""
          pass

      @abstractmethod
      def get_quality_metrics(self) -> dict:
          """Get strategy-specific quality metrics."""
          pass
  ```
- [ ] **3.2.2** Test: Base class can be subclassed

#### Phase 3.3: Bit-Slicing Strategy (Paradigm 2)
- [ ] **3.3.1** Copy `bit_slicing/decomposer.py`
- [ ] **3.3.2** Copy `bit_slicing/retrieval.py`
- [ ] **3.3.3** Copy `bit_slicing/sliced_context.py`
- [ ] **3.3.4** Update all imports
- [ ] **3.3.5** Create `bit_slicing/__init__.py`:
  ```python
  from .decomposer import BitSlicingDecomposer, BitSlicingConfig
  from .retrieval import BitSlicingRetriever
  from .sliced_context import SlicedContext, PrecisionLevel

  __all__ = [
      "BitSlicingDecomposer",
      "BitSlicingConfig",
      "BitSlicingRetriever",
      "SlicedContext",
      "PrecisionLevel",
  ]
  ```
- [ ] **3.3.6** Test: Level 0/1/2 decomposition
- [ ] **3.3.7** Test: Precision-based retrieval
- [ ] **3.3.8** Test: Context slicing

#### Phase 3.4: Hierarchical Decomposition Strategy (Paradigm 3)
- [ ] **3.4.1** Copy `hierarchical_decomposition/decomposer.py` → `hierarchical/decomposer.py`
- [ ] **3.4.2** Copy `hierarchical_decomposition/processor.py` → `hierarchical/processor.py`
- [ ] **3.4.3** Copy `hierarchical_decomposition/query.py` → `hierarchical/query.py`
- [ ] **3.4.4** Update all imports
- [ ] **3.4.5** Create `hierarchical/__init__.py`:
  ```python
  from .decomposer import HierarchicalDecomposer, HierarchyConfig
  from .processor import HierarchicalProcessor, HierarchyNode
  from .query import HierarchicalQuery, QueryResult

  __all__ = [
      "HierarchicalDecomposer",
      "HierarchyConfig",
      "HierarchicalProcessor",
      "HierarchyNode",
      "HierarchicalQuery",
      "QueryResult",
  ]
  ```
- [ ] **3.4.6** Test: Tree construction
- [ ] **3.4.7** Test: Hierarchical query resolution
- [ ] **3.4.8** Test: Node-level R/S/N computation

#### Phase 3.5: Iterative Refinement Strategy (Paradigm 1)
- [ ] **3.5.1** Copy `iterative_refinement/engine.py` → `iterative/engine.py`
- [ ] **3.5.2** Copy `iterative_refinement/residual.py` → `iterative/residual.py`
- [ ] **3.5.3** Copy `iterative_refinement/retrievers.py` → `iterative/retrievers.py`
- [ ] **3.5.4** Update all imports
- [ ] **3.5.5** Create `iterative/__init__.py`:
  ```python
  from .engine import IterativeRefinementEngine, RefinementConfig
  from .residual import ResidualTracker, ResidualQuery
  from .retrievers import RefinementRetriever, MultiStageRetriever

  __all__ = [
      "IterativeRefinementEngine",
      "RefinementConfig",
      "ResidualTracker",
      "ResidualQuery",
      "RefinementRetriever",
      "MultiStageRetriever",
  ]
  ```
- [ ] **3.5.6** Test: Refinement loop convergence
- [ ] **3.5.7** Test: Residual query generation
- [ ] **3.5.8** Test: Multi-stage retrieval

#### Phase 3.6: Layered Stack Strategy (Paradigm 4)
- [ ] **3.6.1** Copy `layered_stack/complementary.py` → `layered/complementary.py`
- [ ] **3.6.2** Copy `layered_stack/interface.py` → `layered/interface.py`
- [ ] **3.6.3** Copy `layered_stack/stack.py` → `layered/stack.py`
- [ ] **3.6.4** Update all imports
- [ ] **3.6.5** Create `layered/__init__.py`:
  ```python
  from .complementary import ComplementaryLayer, LayerConfig
  from .interface import LayerInterface, LayerProtocol
  from .stack import LayeredStack, StackConfig

  __all__ = [
      "ComplementaryLayer",
      "LayerConfig",
      "LayerInterface",
      "LayerProtocol",
      "LayeredStack",
      "StackConfig",
  ]
  ```
- [ ] **3.6.6** Test: Layer stacking
- [ ] **3.6.7** Test: Inter-layer communication
- [ ] **3.6.8** Test: Quality gates between layers

#### Phase 3.7: Self-Calibration Strategy
- [ ] **3.7.1** Copy `self_calibration/calibrator.py` → `calibration/calibrator.py`
- [ ] **3.7.2** Update imports
- [ ] **3.7.3** Create `calibration/__init__.py`:
  ```python
  from .calibrator import SelfCalibrator, CalibrationConfig, CalibrationResult

  __all__ = [
      "SelfCalibrator",
      "CalibrationConfig",
      "CalibrationResult",
  ]
  ```
- [ ] **3.7.4** Test: Auto-calibration loop
- [ ] **3.7.5** Test: Threshold adjustment

#### Phase 3.8: Strategy Registry
- [ ] **3.8.1** Create `strategies/__init__.py` with registry:
  ```python
  from typing import Dict, Type
  from .base import BaseStrategy
  from .bit_slicing import BitSlicingDecomposer
  from .hierarchical import HierarchicalDecomposer
  from .iterative import IterativeRefinementEngine
  from .layered import LayeredStack
  from .calibration import SelfCalibrator

  STRATEGY_REGISTRY: Dict[str, Type[BaseStrategy]] = {
      "bit_slicing": BitSlicingDecomposer,
      "hierarchical": HierarchicalDecomposer,
      "iterative": IterativeRefinementEngine,
      "layered": LayeredStack,
      "calibration": SelfCalibrator,
  }

  def get_strategy(name: str) -> Type[BaseStrategy]:
      """Get strategy class by name."""
      if name not in STRATEGY_REGISTRY:
          raise ValueError(f"Unknown strategy: {name}")
      return STRATEGY_REGISTRY[name]

  __all__ = [
      # Base
      "BaseStrategy",
      # Registry
      "STRATEGY_REGISTRY",
      "get_strategy",
      # Strategies
      "BitSlicingDecomposer",
      "HierarchicalDecomposer",
      "IterativeRefinementEngine",
      "LayeredStack",
      "SelfCalibrator",
  ]
  ```
- [ ] **3.8.2** Test: Registry lookup
- [ ] **3.8.3** Test: All strategies implement BaseStrategy

#### Phase 3.9: Integration with Core
- [ ] **3.9.1** Add strategies to `yrsn/__init__.py`:
  ```python
  from yrsn.strategies import get_strategy, STRATEGY_REGISTRY
  ```
- [ ] **3.9.2** Create integration example:
  ```python
  from yrsn.core import YRSN
  from yrsn.strategies import get_strategy

  # Use strategy by name
  Strategy = get_strategy("iterative")
  strategy = Strategy(config=...)
  result = strategy.process(query, context)
  ```
- [ ] **3.9.3** Test: Strategy integration with YRSN class
- [ ] **3.9.4** Test: Strategy chaining (hierarchical → bit_slicing)
- [ ] **3.9.5** Commit: "feat(strategies): Add all 5 strategy implementations"

---

## Verification Checklist

### After Goal 1 Completion
- [ ] All 18 decomposition files migrated
- [ ] `from yrsn.core.decomposition import *` works
- [ ] Phase transitions compute correctly
- [ ] Spectral analysis runs on test matrices
- [ ] Calibration manager tracks drift
- [ ] Constraint taxonomy validates

### After Goal 2 Completion
- [ ] All 16 neural files migrated
- [ ] Neural imports work without torch (graceful degradation)
- [ ] Neural imports work with torch (full functionality)
- [ ] Retriever → Combiner → Router pipeline works
- [ ] CTM loss computes gradients
- [ ] SNAP training runs

### After Goal 3 Completion
- [ ] All 15 strategy files migrated
- [ ] All strategies implement BaseStrategy
- [ ] Registry returns correct strategy classes
- [ ] Bit-slicing produces 3 precision levels
- [ ] Hierarchical builds valid trees
- [ ] Iterative converges within max_iterations
- [ ] Layered stack passes quality gates
- [ ] Calibration adjusts thresholds

---

## Timeline Estimate

| Goal | Files | Complexity | Effort |
|------|-------|------------|--------|
| Goal 1 | 16 files | Medium-High | Significant |
| Goal 2 | 16 files | High (torch deps) | Significant |
| Goal 3 | 15 files | Medium | Moderate |

**Total:** 47 additional files to migrate

---

## Next Steps After Goals 1-3

1. **Goal 4:** Migrate OOD Detection (10 files)
2. **Goal 5:** Migrate Models/Architectures (19 files)
3. **Goal 6:** Migrate Hardware Integration (3 files)
4. **Goal 7:** Migrate Reasoning (4 files)
5. **Goal 8:** Migrate remaining modules (datasets, benchmarks, apps)

---

## Notes

- All torch dependencies should be optional (lazy imports)
- Maintain backwards compatibility with existing API
- Each phase should have its own commit
- Run tests after each phase before proceeding
