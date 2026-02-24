   5    Core algorithms for the Y=R+S+N decomposition framework.
        6    
        7    This package contains:
        8 -  - yrsn: Core Y=R+S+N mathematical framework and decomposition algorithms
        8 +  - yrsn: Core Y=R+S+N mathematical framework and semantic decomposition
        9    - optimization: Pluggable optimization backends (gradient-based, evolution-based)
       10 -  - (future) context: Context engineering algorithms
       11 -  - (future) neural: Neural network components (CTM, NoisyNet, etc.)
       10 +  - decomposition: Matrix/tensor decomposition (Robust PCA, L+S)
       11    
       12    Framework Architecture:
       13        5 Core Paradigms (what) + Optimization Backends (how)
     ...
       23        Backends:
       24            - GradientBackend: Standard backprop optimization
       25            - EggrollBackend: Evolution-guided low-rank optimization (gradient-free)
       26 +          - AdaptiveBackend: Auto-selects optimal backend based on task
       27    
       28 +      Decomposition:
       29 +          - RobustPCA: Low-rank + Sparse decomposition (L+S→YRSN)
       30 +  
       31    Usage:
       32        from algorithms.yrsn import ResearchFramework, TensorConstructor
       33 -      from algorithms.yrsn import YRSNEvaluator, evaluate_yrsn_embeddings
       34 -      from algorithms.optimization import GradientBackend, EggrollBackend
       33 +      from algorithms.yrsn import detect_collapse, CollapseType
       34 +      from algorithms.optimization import AdaptiveBackend, QualityMetrics
       35 +      from algorithms.decomposition import robust_pca, decompose_to_yrsn
       36    """
