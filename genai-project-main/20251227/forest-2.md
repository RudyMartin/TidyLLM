● Bash(python -m apps.stress_test.test_curriculum_hybrid) timeout: 10m 0s
  ⎿  Running in the background (↓ to manage)

● Task Output b7fa4a1
  ⎿  ======================================================================
     CURRICULUM LEARNING TEST
     Testing if CurriculumManager adds benefit to HybridYRSNClassifier
     … +68 lines (ctrl+o to expand)

● Curriculum Learning IS Additive

  | Method     | Val Acc | Test Acc | Delta |
  |------------|---------|----------|-------|
  | Baseline   | 77.4%   | 67.1%    | -     |
  | Curriculum | 77.0%   | 77.0%    | +9.9% |

  Key Findings

  1. Test accuracy jumped +9.9% (67.1% → 77.0%)
  2. Prevented overfitting - baseline had 10% val/test gap, curriculum had 0% gap
  3. Better routing distribution:

  BASELINE (overfit):          CURRICULUM (balanced):
    abstain: 35%                 abstain: 28%     (-7%)
    ensemble: 38%                ensemble: 2%     (-36%)
    deep: 22%                    deep: 49%        (+27%)
    shallow: 5%                  shallow: 21%     (+16%)

  Why It Worked

  The τ multiplier progression:
  - Foundation (τ×1.5): Higher LR → explored more, found better features
  - Refinement (τ×1.0): Normal LR → consolidated learning
  - Result: Model learned to trust deep path more, abstain less

  Conclusion

  CDL/MOE architecture IS naturally additive:
  - HybridYRSNClassifier provides quality-based routing (per-sample)
  - CurriculumManager provides training schedule (per-epoch)
  - Combined: +9.9% test accuracy, better generalization

  The curriculum wrapper worked without modifying model code - pure natural settings.
