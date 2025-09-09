from __future__ import annotations
from typing import List, Sequence, Tuple, Callable, Union

# Pure built-in math functions (no imports)
def _pure_sqrt(x: float) -> float:
    """Square root using Newton's method"""
    if x < 0:
        raise ValueError("sqrt of negative number")
    if x == 0:
        return 0.0
    
    # Newton's method: x_new = (x + n/x) / 2
    guess = x
    for _ in range(20):
        new_guess = (guess + x / guess) / 2
        if abs(new_guess - guess) < 1e-15:
            return new_guess
        guess = new_guess
    return guess

def _pure_max(*args):
    """Pure max function"""
    if len(args) == 1 and hasattr(args[0], '__iter__') and not isinstance(args[0], str):
        args = args[0]
    
    if not args:
        raise ValueError("max() arg is an empty sequence")
    
    result = args[0]
    for item in args[1:]:
        if item > result:
            result = item
    return result

def _pure_min(*args):
    """Pure min function"""
    if len(args) == 1 and hasattr(args[0], '__iter__') and not isinstance(args[0], str):
        args = args[0]
    
    if not args:
        raise ValueError("min() arg is an empty sequence")
    
    result = args[0]
    for item in args[1:]:
        if item < result:
            result = item
    return result

Scalar = Union[int, float]
Vector = List[Scalar]
Matrix = List[Vector]

# -----------------------
# Shapes & constructors
# -----------------------

def shape(x: Union[Scalar, Vector, Matrix]) -> Tuple[int, ...]:
    """Get shape of scalar, vector, or matrix."""
    if not isinstance(x, list):
        return ()
    if len(x) == 0:
        return (0,)
    if isinstance(x[0], list):
        return (len(x), len(x[0]))
    return (len(x),)

def array(obj: Union[Scalar, Sequence, Sequence[Sequence]]) -> Union[Scalar, Vector, Matrix]:
    """Convert sequence to array format."""
    if isinstance(obj, list):
        return [array(e) for e in obj]
    if isinstance(obj, tuple):
        return [array(e) for e in list(obj)]
    if isinstance(obj, (int, float)):
        return float(obj)
    return obj

def zeros(sz: Union[int, Tuple[int, int]]) -> Union[Vector, Matrix]:
    """Create array filled with zeros."""
    if isinstance(sz, int):
        return [0.0] * sz
    n, m = sz
    return [[0.0 for _ in range(m)] for _ in range(n)]

def ones(sz: Union[int, Tuple[int, int]]) -> Union[Vector, Matrix]:
    """Create array filled with ones."""
    if isinstance(sz, int):
        return [1.0] * sz
    n, m = sz
    return [[1.0 for _ in range(m)] for _ in range(n)]

def eye(n: int) -> Matrix:
    """Create identity matrix."""
    I = zeros((n, n))
    for i in range(n):
        I[i][i] = 1.0
    return I

# -----------------------
# Elementwise helpers
# -----------------------

def _is_vec(x):
    return isinstance(x, list) and (len(x) == 0 or not isinstance(x[0], list))

def _is_mat(x):
    return isinstance(x, list) and (len(x) > 0 and isinstance(x[0], list))

def _is_scalar(x):
    return not isinstance(x, list)

def _apply1(x, f: Callable[[Scalar], Scalar]):
    if isinstance(x, list):
        return [_apply1(e, f) for e in x]
    return f(float(x))

def _apply2(x, y, f: Callable[[Scalar, Scalar], Scalar]):
    if not isinstance(x, list) and not isinstance(y, list):
        return f(float(x), float(y))
    if not isinstance(x, list):
        return [_apply2(x, e, f) for e in y]
    if not isinstance(y, list):
        return [_apply2(e, y, f) for e in x]
    assert len(x) == len(y), "shape mismatch"
    return [_apply2(a, b, f) for a, b in zip(x, y)]

# public elementwise ops
def add(x, y):
    """Add two arrays elementwise."""
    return _apply2(x, y, lambda a, b: a + b)

def sub(x, y):
    """Subtract two arrays elementwise."""
    return _apply2(x, y, lambda a, b: a - b)

def mul(x, y):
    """Multiply two arrays elementwise."""
    return _apply2(x, y, lambda a, b: a * b)

def div(x, y):
    """Divide two arrays elementwise."""
    return _apply2(x, y, lambda a, b: a / b)

def exp(x):
    """Apply exponential function elementwise."""
    return _apply1(x, _pure_exp)

def log(x):
    """Apply natural logarithm elementwise."""
    return _apply1(x, lambda z: -1e30 if z <= 0 else _pure_ln(z))

def sqrt(x):
    """Apply square root elementwise."""
    return _apply1(x, _pure_sqrt)

def clip(x, lo, hi):
    """Clip values to range [lo, hi]."""
    return _apply1(x, lambda z: hi if z > hi else (lo if z < lo else z))

# -----------------------
# Reductions
# -----------------------

def sum(x, axis: int | None = None) -> Union[Scalar, Vector]:
    """Compute sum along axis or all elements."""
    if axis is None:
        if _is_vec(x):
            return float(_sum1d(x))
        elif _is_mat(x):
            total = 0.0
            for row in x:
                total += _sum1d(row)
            return total
        else:
            return float(x)
    else:
        if axis == 0 and _is_mat(x):
            n, m = shape(x)
            result = []
            for j in range(m):
                col_sum = 0.0
                for i in range(n):
                    col_sum += x[i][j]
                result.append(col_sum)
            return result
        if axis == 1 and _is_mat(x):
            return [float(_sum1d(row)) for row in x]
        raise ValueError("axis out of range or invalid for input")

def _sum1d(v: Vector) -> float:
    s = 0.0
    for a in v:
        s += float(a)
    return s

def mean(x, axis: int | None = None):
    """Compute mean along axis or all elements."""
    if axis is None:
        if _is_vec(x):
            return _sum1d(x) / len(x)
        elif _is_mat(x):
            n, m = shape(x)
            total = 0.0
            for row in x:
                total += _sum1d(row)
            return total / (n * m)
        else:
            return float(x)
    if axis == 0:
        n, m = shape(x)
        return [_sum1d([x[i][j] for i in range(n)]) / n for j in range(m)]
    if axis == 1:
        return [_sum1d(row) / len(row) for row in x]
    raise ValueError("axis out of range")

def var(x, axis: int | None = None, ddof: int = 0):
    """Compute variance along axis or all elements - handles scalars, vectors, matrices."""
    if axis is None:
        mu = mean(x)
        if _is_scalar(x):
            return 0.0  # Scalar has no variance from itself
        elif _is_vec(x):
            return sum([(a - mu) ** 2 for a in x]) / (len(x) - ddof)
        elif _is_mat(x):
            n, m = shape(x)
            return sum([(a - mu) ** 2 for row in x for a in row]) / (n * m - ddof)
    else:
        if axis == 0:
            mu = mean(x, axis=0)
            n, m = shape(x)
            return [sum([(x[i][j] - mu[j]) ** 2 for i in range(n)]) / (n - ddof) for j in range(m)]
        if axis == 1:
            mus = mean(x, axis=1)
            return [sum([(a - mus[i]) ** 2 for a in row]) / (len(row) - ddof) for i, row in enumerate(x)]
    raise ValueError("axis out of range")

def std(x, axis: int | None = None, ddof: int = 0):
    """Compute standard deviation along axis or all elements."""
    if axis is None:
        return _pure_sqrt(var(x, axis=None, ddof=ddof))
    v = var(x, axis=axis, ddof=ddof)
    if isinstance(v, list):
        return [_pure_sqrt(a) for a in v]
    return _pure_sqrt(v)

def max(x, axis: int | None = None):
    """Compute maximum along axis or all elements."""
    if axis is None:
        if _is_vec(x):
            return float(_pure_max(x))
        elif _is_mat(x):
            return float(_pure_max([_pure_max(row) for row in x]))
        return float(x)
    if axis == 0:
        n, m = shape(x)
        return [float(_pure_max([x[i][j] for i in range(n)])) for j in range(m)]
    if axis == 1:
        return [float(_pure_max(row)) for row in x]
    raise ValueError("axis out of range")

def min(x, axis: int | None = None):
    """Compute minimum along axis or all elements."""
    if axis is None:
        if _is_vec(x):
            return float(_pure_min(x))
        elif _is_mat(x):
            return float(_pure_min([_pure_min(row) for row in x]))
        return float(x)
    if axis == 0:
        n, m = shape(x)
        return [float(_pure_min([x[i][j] for i in range(n)])) for j in range(m)]
    if axis == 1:
        return [float(_pure_min(row)) for row in x]
    raise ValueError("axis out of range")

def argmax(x, axis: int = -1):
    """Compute indices of maximum values along axis - handles scalars, vectors, matrices."""
    # Handle scalar case
    if _is_scalar(x):
        return 0  # Only one element, so index is 0
    
    # Handle vector case
    if _is_vec(x):
        if len(x) == 0:
            raise ValueError("argmax of empty array")
        max_idx = 0
        max_val = x[0]
        for i in range(1, len(x)):
            if x[i] > max_val:
                max_val = x[i]
                max_idx = i
        return max_idx
    
    # Handle matrix case
    if _is_mat(x):
        if axis in (-1, 1):  # argmax along rows
            return [argmax(row) for row in x]
        if axis == 0:  # argmax along columns
            n, m = shape(x)
            result = []
            for j in range(m):
                max_idx = 0
                max_val = x[0][j]
                for i in range(1, n):
                    if x[i][j] > max_val:
                        max_val = x[i][j]
                        max_idx = i
                result.append(max_idx)
            return result
    
    raise ValueError("axis out of range or invalid for input")

# -----------------------
# Linear algebra basics
# -----------------------

def transpose(A):
    """Transpose a matrix, vector, or scalar - handles all input types."""
    if _is_scalar(A):
        return A  # Scalar transpose is itself
    elif _is_vec(A):
        # Convert vector to column matrix
        return [[x] for x in A]
    elif _is_mat(A):
        n, m = shape(A)
        return [[A[i][j] for i in range(n)] for j in range(m)]
    else:
        raise ValueError("transpose: unsupported input type")

def dot(a, b):
    """Compute dot product - handles vectors or matrices like numpy.dot
    
    For vectors: returns scalar dot product
    For matrices: delegates to matmul for matrix multiplication
    """
    # Check if both are vectors (1D)
    if _is_vec(a) and _is_vec(b):
        assert len(a) == len(b), "shape mismatch for dot"
        return sum([a[i] * b[i] for i in range(len(a))])
    
    # If either is a matrix (2D), use matrix multiplication
    elif _is_mat(a) or _is_mat(b):
        return matmul(a, b)
    
    # Handle scalar cases
    else:
        return float(a) * float(b)

def matmul(A: Matrix, B: Matrix) -> Matrix:
    """Compute matrix multiplication."""
    n, k = shape(A)
    k2, m = shape(B)
    assert k == k2, "shape mismatch for matmul"
    BT = transpose(B)
    C = zeros((n, m))
    for i in range(n):
        for j in range(m):
            C[i][j] = dot(A[i], BT[j])
    return C

def norm(v, ord: int = 2) -> float:
    """Compute norm - handles scalars, vectors (vector norm), matrices (Frobenius norm)."""
    if _is_scalar(v):
        return abs(float(v))  # Scalar norm is absolute value
    elif _is_vec(v):
        # Vector norm
        if ord == 2:
            return _pure_sqrt(sum([x*x for x in v]))
        if ord == 1:
            return sum([abs(x) for x in v])
        raise NotImplementedError
    elif _is_mat(v):
        # Matrix Frobenius norm (default for matrices)
        total = 0.0
        for row in v:
            for x in row:
                total += x * x
        return _pure_sqrt(total)
    else:
        raise ValueError("norm: unsupported input type")

def l2_normalize(X, axis: int = -1, eps: float = 1e-12):
    """Apply L2 normalization along axis."""
    if _is_vec(X):
        nrm = _pure_max(norm(X), eps)
        return [x / nrm for x in X]
    out = []
    for row in X:
        nrm = _pure_max(norm(row), eps)
        out.append([x / nrm for x in row])
    return out

# -----------------------
# Shape manipulation
# -----------------------

def flatten(x) -> Vector:
    """Flatten array to 1D - handles scalars, vectors, matrices."""
    if _is_scalar(x):
        return [float(x)]  # Scalar becomes single-element list
    elif _is_vec(x):
        return x[:]  # Vector copy
    elif _is_mat(x):
        result = []
        for row in x:
            result.extend(row)
        return result
    else:
        raise ValueError("flatten: unsupported input type")

def reshape(x: Union[Vector, Matrix], new_shape: Union[int, Tuple[int, ...]]) -> Union[Vector, Matrix]:
    """Reshape array to new dimensions."""
    # Flatten first
    flat = flatten(x) if _is_mat(x) else x[:]
    
    # Handle single dimension
    if isinstance(new_shape, int):
        if new_shape == -1:
            return flat
        if new_shape != len(flat):
            raise ValueError(f"Cannot reshape array of size {len(flat)} into shape ({new_shape},)")
        return flat
    
    # Handle 2D reshape
    if len(new_shape) == 1:
        if new_shape[0] == -1:
            return flat
        if new_shape[0] != len(flat):
            raise ValueError(f"Cannot reshape array of size {len(flat)} into shape {new_shape}")
        return flat
    
    if len(new_shape) == 2:
        n, m = new_shape
        total = len(flat)
        
        # Handle -1 (infer dimension)
        if n == -1:
            if total % m != 0:
                raise ValueError(f"Cannot reshape array of size {total} into shape ({n}, {m})")
            n = total // m
        elif m == -1:
            if total % n != 0:
                raise ValueError(f"Cannot reshape array of size {total} into shape ({n}, {m})")
            m = total // n
        
        if n * m != total:
            raise ValueError(f"Cannot reshape array of size {total} into shape ({n}, {m})")
        
        result = []
        for i in range(n):
            result.append(flat[i*m:(i+1)*m])
        return result
    
    raise NotImplementedError("Only 1D and 2D reshapes supported")

def concatenate(arrays: List[Union[Vector, Matrix]], axis: int = 0) -> Union[Vector, Matrix]:
    """Concatenate arrays along axis."""
    if not arrays:
        return []
    
    # All vectors
    if all(_is_vec(a) for a in arrays):
        if axis != 0:
            raise ValueError("Can only concatenate 1D arrays along axis 0")
        result = []
        for arr in arrays:
            result.extend(arr)
        return result
    
    # All matrices
    if all(_is_mat(a) for a in arrays):
        if axis == 0:
            # Stack rows
            result = []
            for arr in arrays:
                result.extend(arr)
            return result
        elif axis == 1:
            # Stack columns
            if not all(len(a) == len(arrays[0]) for a in arrays):
                raise ValueError("All arrays must have same number of rows for axis=1 concatenation")
            result = []
            for i in range(len(arrays[0])):
                row = []
                for arr in arrays:
                    row.extend(arr[i])
                result.append(row)
            return result
        else:
            raise ValueError(f"Invalid axis {axis} for 2D concatenation")
    
    raise ValueError("All arrays must have same dimensionality")

# -----------------------
# Classic Random Number Generators (Historical Definitions)
# -----------------------
#
# HISTORICAL CONTEXT & ALGORITHM CHOICE:
#
# 1. Linear Congruential Generator (LCG):
#    - Invented by D.H. Lehmer (1951) 
#    - Park & Miller "Minimal Standard" (1988) refined parameters
#    - Original: X(n+1) = (16807 * X(n)) mod (2^31 - 1)
#    - Why this variant: Well-studied, passes basic statistical tests
#    - Historical significance: First practical PRNG, used in early computers
#
# 2. Xorshift:
#    - Invented by George Marsaglia (2003)
#    - Original 32-bit version: x ^= x << 13; x ^= x >> 17; x ^= x << 5
#    - Why this variant: Simple, fast, good statistical properties
#    - Modern alternative to flawed early generators like Middle Square
#
# 3. Box-Muller Transform:
#    - Invented by Box & Muller (1958), refined by Marsaglia (1964)
#    - Converts uniform random to Gaussian using polar coordinates
#    - Classic formula: Z₁ = √(-2 ln U₁) cos(2π U₂), Z₂ = √(-2 ln U₁) sin(2π U₂)
#    - Why this method: Mathematically exact, widely recognized standard
#
# PURE IMPLEMENTATION RATIONALE:
# - No imports (even math/random) maintains TLM's zero-dependency sovereignty
# - Hand-coded Taylor series for sin/cos/ln/exp preserves algorithmic transparency
# - Classic historical algorithms chosen for educational & reference value
# - Provides complete NumPy.random replacement without external dependencies

# Global state for random generators (modernized)
_rng_state = {'lcg': None, 'xorshift': None, 'initialized': False}

def _get_entropy_seed():
    """Get high-entropy seed using only built-ins (modern practice)"""
    # Use current time in microseconds + memory address hash for entropy
    # This provides much better randomness than fixed seeds
    import time
    time_seed = int((time.time() * 1000000) % (2**31))
    
    # Mix in object memory addresses for additional entropy
    obj_hash = hash(str(id([]))) & 0x7FFFFFFF
    
    # Simple mixing function (modern technique)
    mixed = time_seed ^ (obj_hash << 13) ^ (obj_hash >> 19)
    mixed = mixed & 0x7FFFFFFF  # Keep positive 31-bit
    
    return mixed if mixed != 0 else 1

def _mix_seed(seed):
    """Mix seed for better distribution (modern practice)"""
    # Simple hash mixing to improve seed quality
    seed = seed & 0x7FFFFFFF  # Ensure positive
    seed ^= seed >> 16
    seed = (seed * 0x85ebca6b) & 0x7FFFFFFF
    seed ^= seed >> 13
    seed = (seed * 0xc2b2ae35) & 0x7FFFFFFF
    seed ^= seed >> 16
    return seed if seed != 0 else 1

def seed_rng(seed=None):
    """
    Seed all random number generators (modernized).
    
    Args:
        seed: Integer seed value, or None for high-entropy auto-seeding
              
    Modern improvements over historical practice:
    - None default uses system entropy (not fixed 1)
    - Seed mixing improves distribution quality
    - Separate initialization prevents bad defaults
    """
    global _rng_state
    
    if seed is None:
        # Modern practice: High-entropy default seeding
        entropy_seed = _get_entropy_seed()
        mixed_seed = _mix_seed(entropy_seed)
    else:
        # User-provided seed with modern mixing
        mixed_seed = _mix_seed(abs(int(seed)))
    
    _rng_state['lcg'] = mixed_seed
    _rng_state['xorshift'] = mixed_seed ^ 0x12345678  # Different initial states
    _rng_state['initialized'] = True

def _ensure_initialized():
    """Ensure RNG is initialized with modern entropy (lazy initialization)"""
    if not _rng_state['initialized']:
        seed_rng()  # Auto-seed with entropy

def lcg_random() -> float:
    """Linear Congruential Generator - Park & Miller "Minimal Standard" (1988)
    
    Modern enhancement: Entropy-based auto-seeding on first use
    Historical algorithm: Lehmer (1951) / Park & Miller (1988) parameters
    """
    _ensure_initialized()
    # Classic parameters: a=16807, m=2^31-1 (Lehmer, 1951 / Park & Miller, 1988)
    _rng_state['lcg'] = (16807 * _rng_state['lcg']) % (2**31 - 1)
    return _rng_state['lcg'] / (2**31 - 1)

def xorshift_random() -> float:
    """Classic Xorshift - Marsaglia (2003) original 32-bit version
    
    Modern enhancement: Entropy-based auto-seeding on first use
    Historical algorithm: Marsaglia (2003) original bit shifts
    """
    _ensure_initialized()
    x = _rng_state['xorshift']
    x ^= x << 13
    x ^= x >> 17
    x ^= x << 5
    x &= 0xFFFFFFFF  # Keep 32-bit
    _rng_state['xorshift'] = x
    return x / (2**32)

def random_uniform(low: float = 0.0, high: float = 1.0, size: Union[int, Tuple[int, int], None] = None) -> Union[float, Vector, Matrix]:
    """Generate uniform random numbers using LCG
    
    Modern enhancement: Entropy-based auto-seeding on first use
    Compatible with numpy.random.uniform API
    """
    _ensure_initialized()
    def _single():
        return low + (high - low) * lcg_random()
    
    if size is None:
        return _single()
    elif isinstance(size, int):
        return [_single() for _ in range(size)]
    else:
        n, m = size
        return [[_single() for _ in range(m)] for _ in range(n)]

def random_normal(mean: float = 0.0, std: float = 1.0, size: Union[int, Tuple[int, int], None] = None) -> Union[float, Vector, Matrix]:
    """Generate normal random numbers using Box-Muller Transform (1958)
    
    Modern enhancement: Entropy-based auto-seeding on first use
    Historical algorithm: Box & Muller (1958) polar transform
    Compatible with numpy.random.normal API
    """
    _ensure_initialized()
    # Classic Box-Muller polar method
    def _pair():
        u1 = lcg_random()
        u2 = lcg_random()
        # Avoid log(0) 
        u1 = _pure_max(u1, 1e-10)
        
        # Box-Muller transform using pure math (no imports)
        r = (-2 * _pure_ln(u1)) ** 0.5
        theta = 2 * 3.14159265358979323846 * u2
        z1 = r * _pure_cos(theta)
        z2 = r * _pure_sin(theta)
        return mean + std * z1, mean + std * z2
    
    if size is None:
        return _pair()[0]
    elif isinstance(size, int):
        result = []
        for i in range(0, size, 2):
            z1, z2 = _pair()
            result.append(z1)
            if len(result) < size:
                result.append(z2)
        return result[:size]
    else:
        n, m = size
        total = n * m
        flat = []
        for i in range(0, total, 2):
            z1, z2 = _pair()
            flat.append(z1)
            if len(flat) < total:
                flat.append(z2)
        flat = flat[:total]
        return [flat[i*m:(i+1)*m] for i in range(n)]

def random_choice(array: Vector, size: Union[int, None] = None) -> Union[Scalar, Vector]:
    """Random choice from array
    
    Modern enhancement: Entropy-based auto-seeding on first use
    Compatible with numpy.random.choice API
    """
    _ensure_initialized()
    if size is None:
        idx = int(lcg_random() * len(array))
        return array[idx]
    else:
        return [array[int(lcg_random() * len(array))] for _ in range(size)]

# Pure math functions (no imports)
def _pure_ln(x: float) -> float:
    """Natural logarithm using Newton's method"""
    if x <= 0:
        raise ValueError("ln undefined for x <= 0")
    if x == 1:
        return 0.0
    
    # Newton's method: y = ln(x), e^y = x
    # f(y) = e^y - x, f'(y) = e^y
    # y_new = y - f(y)/f'(y) = y - (e^y - x)/e^y = y - 1 + x/e^y
    y = x - 1  # Initial guess
    for _ in range(20):  # Usually converges quickly
        ey = _pure_exp(y)
        y_new = y - 1 + x / ey
        if abs(y_new - y) < 1e-15:
            break
        y = y_new
    return y

def _pure_exp(x: float) -> float:
    """Exponential function using Taylor series"""
    if x > 700:  # Prevent overflow
        return float('inf')
    if x < -700:
        return 0.0
    
    result = 1.0
    term = 1.0
    for i in range(1, 150):  # Taylor series
        term *= x / i
        result += term
        if abs(term) < 1e-15:
            break
    return result

def _pure_cos(x: float) -> float:
    """Cosine using Taylor series"""
    # Reduce to [0, 2π]
    pi = 3.14159265358979323846
    x = x % (2 * pi)
    
    result = 1.0
    term = 1.0
    x_sq = x * x
    
    for i in range(1, 50):
        term *= -x_sq / ((2*i-1) * (2*i))
        result += term
        if abs(term) < 1e-15:
            break
    return result

def _pure_sin(x: float) -> float:
    """Sine using Taylor series"""
    # Reduce to [0, 2π]
    pi = 3.14159265358979323846
    x = x % (2 * pi)
    
    result = x
    term = x
    x_sq = x * x
    
    for i in range(1, 50):
        term *= -x_sq / ((2*i) * (2*i+1))
        result += term
        if abs(term) < 1e-15:
            break
    return result
