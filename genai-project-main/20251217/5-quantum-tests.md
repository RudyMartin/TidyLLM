⏺ All tests pass. Here's a summary:

  Quantum Backend Tests - PASSED

  | Test                           | Result                                 |
  |--------------------------------|----------------------------------------|
  | Basic quantum backend          | R=0.625, S=0.312, N=0.062              |
  | With query comparison          | R=0.154, S=0.769, N=0.077              |
  | ZZ feature map                 | R=0.154, S=0.769, N=0.077              |
  | Pre-computed features          | Working                                |
  | Similar vs dissimilar features | R=0.210 vs R=0.154 (correct direction) |
  | Fallback to heuristic          | Working                                |

  Issues Fixed:
  1. Added None check for YRSNQuantumKernel (when dependencies unavailable)
  2. Fixed import conflict: kernels.py vs kernels/ directory in yrsn-context by loading the .py file directly

  Note: The yrsn-context package has a naming conflict (kernels.py and kernels/ both exist) which should be fixed there. The yrsn-tools workaround loads the actual .py file directly.

| Item                                                | Status |
  |-----------------------------------------------------|--------|
  | QuantumBackend uses robust YRSNQuantumKernel        | ✓      |
  | Flexible device parameter (CPU/GPU/TPU)             | ✓      |
  | Graceful fallback to heuristic                      | ✓      |
  | Documentation updated                               | ✓      |
  | Import conflict workaround (kernels.py vs kernels/) | ✓      |
  | All tests pass                                      | ✓      |

  The quantum backend works with:
  - YRSN(content, backend="quantum") - basic usage
  - YRSN(content, query=q, backend="quantum") - with query comparison
  - QuantumBackend(feature_map="zz", device="lightning.gpu") - custom config
  - Pre-computed feature vectors
  - Falls back to heuristic when PennyLane unavailable
