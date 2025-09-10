# Mathematical Formulations from Y=R+S+N Framework Papers

## Paper 1: Signal-noise separation using unsupervised reservoir computing (2404.04870)

### Core Mathematical Framework

**Signal Model:**
- Additive noise: `xi = qi + ξi, E[ξi] = 0` (Eq. 1)
- Multiplicative noise: `xi = qiξi, E[ξi] = 1` (Eq. 2)

Where:
- `xi` = observed signal at sampling instant i
- `qi` = deterministic signal component  
- `ξi` = noise component

**Reservoir Computing Predictor:**
- State evolution: `r(i+1) = (1−α)r(i)+α tanh(Ar(i)+Winxi−1)` (Eq. 3)
- Readout optimization: `Wout = arg min Σ ||xi+1 − WT r(i)||² + λ||W||²F` (Eq. 4)
- Signal reconstruction: `q̂i = Woutr(i)` (Eq. 5)

**Noise Type Identification:**
- Additive noise: `E[||ψi|| | q̂i] = const.` (flat graph)
- Multiplicative noise: `E[||xi − qi|| | qi] ≈ C ||qi||` (V-shaped graph)

**Key Performance Metrics:**
- Training cost: `ΣK i=m ||xi − P(xi−1, xi−2, ..., xi−m)||²`
- Validation error: `ΣK i=m ||xi − q̂i||²`
- RMSE for reconstruction quality
- Jensen-Shannon divergence (JSD) for noise distribution estimation

### Y=R+S+N Relevance
This paper provides direct mathematical formulation for separating signal (R=Relevant) from noise (N=Noise) components, with sophisticated methods to handle both additive and multiplicative noise scenarios.

---

## Paper 2: A novel algorithm for the decomposition of non-stationary multidimensional and multivariate signals (2412.00553)

### Multidimensional Fast Iterative Filtering (MdMvFIF)

**Signal Decomposition:**
- Decomposes complex signals into Intrinsic Mode Functions (IMFs)
- Handles signals varying simultaneously in space and time
- Extends traditional EMD/FIF to multidimensional cases

**Core Innovation:**
- Breaks down complex signals into simpler oscillatory components
- Enables efficient data representation, noise reduction, feature extraction
- No predefined basis functions required

### Y=R+S+N Relevance  
MdMvFIF directly addresses the R (relevant signal components) extraction while filtering out N (noise) through iterative decomposition, applicable to multidimensional data.

---

## Paper 3: Signal-Plus-Noise Decomposition of Nonlinear Spiked Random Matrix Models (2405.18274)

### Matrix-Based Signal-Noise Framework

**Model Structure:**
- Nonlinear spiked random matrix model
- Nonlinear function applied element-wise to noise matrix perturbed by rank-one signal
- Signal-plus-noise decomposition: M = L + S + W

Where:
- M = observed matrix
- L = low-rank signal component
- S = sparse component  
- W = unstructured noise

**Phase Transitions:**
- Critical thresholds of signal strength affect matrix structure
- Identifies phase transitions in signal components

### Y=R+S+N Relevance
Direct mathematical framework for Y=L+S+W decomposition where L represents relevant signal information, S captures structured but superfluous information, and W represents pure noise.

---

## Paper 4: Short-time Variational Mode Decomposition (2501.09174)

### STVMD Mathematical Framework

**Optimization Objective:**
- Minimizes sum of bandwidths across windowed data segments
- Incorporates Short-Time Fourier Transform (STFT)
- Uses alternating direction method of multipliers

**Signal Processing:**
- Segments signals into short time windows
- Converts segments to frequency domain
- Extracts band-limited modes with narrow bandwidths

**Two Variants:**
- Dynamic STVMD: Better for non-stationary signals
- Non-dynamic STVMD: Standard windowed approach

### Y=R+S+N Relevance
STVMD provides mathematical framework for time-windowed signal decomposition, separating relevant signal modes (R) from noise (N) while handling temporal variations in signal characteristics.

---

## Paper 5: Empirical Wavelet Transform (2410.23534)

### Adaptive Wavelet Framework

**Core Mathematical Approach:**
- Builds adaptive wavelets based on signal's inherent frequency characteristics
- Fourier spectrum segmentation into contiguous frequency supports
- Parameter γ controls filter transitions ensuring tight frame property

**Wavelet Construction:**
- Empirical scaling functions and wavelets via Fourier transform definitions
- Bandpass filters based on detected frequency supports
- Creates wavelet filter bank from signal spectrum information

**Signal Decomposition:**
- Detects frequency supports empirically from processed signal spectrum
- More consistent than EMD for signal component separation
- Extensible to 2D signals (images)

### Y=R+S+N Relevance
EWT provides mathematical framework for adaptive separation of relevant signal components (R) from noise (N) by building wavelets tailored to signal's frequency characteristics.

---

## Paper 6: Gabor-based learnable sparse representation for self-supervised denoising (2308.03077)

### Self-Supervised Sparse Framework

**Mathematical Innovation:**
- Deep unfolding methods transform optimization into neural networks
- Gabor filters with parameter constraints during training
- Physics-informed filter embedding

**Sparse Representation:**
- Self-supervised noise suppression without clean training labels
- Learnable dictionary with Gabor basis functions
- Constraint-based parameter optimization

### Y=R+S+N Relevance
Provides framework for learning sparse representations where relevant signal components (R) are captured by constrained Gabor filters while noise (N) is suppressed through self-supervised learning.

---

## Paper 7: Tailored Low-Rank Matrix Factorization for Similarity Matrix Completion (2409.19550)

### Low-Rank Completion Framework

**Mathematical Model:**
- Utilizes Positive Semi-definiteness (PSD) property
- Nonconvex low-rank regularizer
- Similarity matrix completion: M = L + S where L is low-rank, S is sparse

**Optimization:**
- Rank-minimization regularizer for optimal low-rank solution
- Reduces computational complexity vs SVD approaches
- SMCNN and SMCNmF algorithms

### Y=R+S+N Relevance
Direct L+S decomposition framework where L captures relevant structural information (R) and missing/corrupted entries represent noise (N) to be completed/corrected.

---

## Paper 8: Blind Source Separation Based on Sparsity (2504.19124)

### Sparse BSS Framework

**Mathematical Foundation:**
- Morphological Component Analysis (MCA): signal = linear combination of components with distinct geometries
- Sparsity-based decomposition in predefined dictionaries
- SAC+BK-SVD algorithm for block-sparsifying dictionary learning

**Key Algorithms:**
- Block coordinate relaxation MCA
- Multichannel MCA (MMCA)
- Generalized MCA (GMCA)
- Dictionary learning with K-SVD extensions

### Y=R+S+N Relevance
Provides sparse representation framework where relevant signal sources (R) are separated from noise (N) using sparsity constraints and morphological component analysis.

---

## Paper 9: Learned Robust PCA: A Scalable Deep Unfolding Approach (2110.05649)

### Deep Unfolding RPCA Framework

**Mathematical Model:**
- Low-rank + sparse decomposition: M = L + S
- Deep unfolding extends finite to infinite iterations
- Feedforward-recurrent-mixed neural network architecture

**Optimization:**
- Minimizes rank of low-rank component L
- Minimizes ℓ0 norm of sparse component S
- Recovery guarantees under mild assumptions

### Y=R+S+N Relevance
Direct mathematical implementation of Y=L+S decomposition using deep learning, where L represents relevant low-rank structure (R) and S captures sparse anomalies/noise (N).

---

## Paper 10: Noisy Nonnegative Tucker Decomposition (2208.08287)

### Tensor Decomposition Framework

**Mathematical Model:**
- Tensor decomposition: T = core ×₁ U₁ ×₂ U₂ ×₃ U₃
- ℓ0 norm sparsity constraints on factor matrices
- Error bounds for Gaussian, Laplace, and Poisson noise

**Noise Handling:**
- Handles missing data and various noise distributions
- Nonnegative constraints preserve physical interpretability
- Sparse factor matrices for improved decomposition

### Y=R+S+N Relevance
Tensor framework for decomposing multidimensional data where core tensor and factors capture relevant structure (R) while noise handling addresses N component through robust optimization.

---

## Summary of Mathematical Approaches for Y=R+S+N Framework

1. **Signal Models:** xi = qi + ξi (additive), xi = qiξi (multiplicative)
2. **Matrix Decomposition:** M = L + S + W (low-rank + sparse + noise)
3. **Tensor Framework:** Multi-dimensional decomposition with sparsity constraints
4. **Optimization Methods:** Variational, alternating direction multipliers, deep unfolding
5. **Validation Criteria:** RMSE, Jensen-Shannon divergence, reconstruction error
6. **Noise Identification:** Conditional expectation analysis, correlation studies

All papers provide mathematical foundations for decomposing signals/data into relevant components (R), superfluous but structured components (S), and pure noise (N), supporting the Y=R+S+N theoretical framework.