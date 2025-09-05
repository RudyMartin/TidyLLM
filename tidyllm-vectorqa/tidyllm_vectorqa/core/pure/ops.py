from __future__ import annotations
import math, random, builtins
from typing import List, Sequence, Tuple, Callable, Union

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
    return _apply1(x, math.exp)

def log(x):
    """Apply natural logarithm elementwise."""
    return _apply1(x, lambda z: -1e30 if z <= 0 else math.log(z))

def sqrt(x):
    """Apply square root elementwise."""
    return _apply1(x, math.sqrt)

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
            return float(sum(_sum1d(row) for row in x))
        else:
            return float(x)
    else:
        if axis == 0 and _is_mat(x):
            n, m = shape(x)
            return [float(sum(x[i][j] for i in range(n))) for j in range(m)]
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
    """Compute variance along axis or all elements."""
    if axis is None:
        mu = mean(x)
        if _is_vec(x):
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
        return math.sqrt(var(x, axis=None, ddof=ddof))
    v = var(x, axis=axis, ddof=ddof)
    if isinstance(v, list):
        return [math.sqrt(a) for a in v]
    return math.sqrt(v)

def max(x, axis: int | None = None):
    """Compute maximum along axis or all elements."""
    if axis is None:
        if _is_vec(x):
            return float(builtins.max(x))
        elif _is_mat(x):
            return float(builtins.max([builtins.max(row) for row in x]))
        return float(x)
    if axis == 0:
        n, m = shape(x)
        return [float(builtins.max([x[i][j] for i in range(n)])) for j in range(m)]
    if axis == 1:
        return [float(builtins.max(row)) for row in x]
    raise ValueError("axis out of range")

def min(x, axis: int | None = None):
    """Compute minimum along axis or all elements."""
    if axis is None:
        if _is_vec(x):
            return float(builtins.min(x))
        elif _is_mat(x):
            return float(builtins.min([builtins.min(row) for row in x]))
        return float(x)
    if axis == 0:
        n, m = shape(x)
        return [float(builtins.min([x[i][j] for i in range(n)])) for j in range(m)]
    if axis == 1:
        return [float(builtins.min(row)) for row in x]
    raise ValueError("axis out of range")

def argmax(x, axis: int = -1):
    """Compute indices of maximum values along axis."""
    if _is_vec(x):
        return int(max(range(len(x)), key=lambda i: x[i]))
    if _is_mat(x):
        if axis in (-1, 1):
            return [int(max(range(len(row)), key=lambda j: row[j])) for row in x]
        if axis == 0:
            n, m = shape(x)
            return [int(max(range(n), key=lambda i: x[i][j])) for j in range(m)]
    raise ValueError("axis out of range or invalid for input")

# -----------------------
# Linear algebra basics
# -----------------------

def transpose(A: Matrix) -> Matrix:
    """Transpose a matrix."""
    n, m = shape(A)
    return [[A[i][j] for i in range(n)] for j in range(m)]

def dot(a: Vector, b: Vector) -> float:
    """Compute dot product of two vectors."""
    assert len(a) == len(b), "shape mismatch for dot"
    return sum([a[i] * b[i] for i in range(len(a))])

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

def norm(v: Vector, ord: int = 2) -> float:
    """Compute vector norm."""
    if ord == 2:
        return math.sqrt(sum([x*x for x in v]))
    if ord == 1:
        return sum([abs(x) for x in v])
    raise NotImplementedError

def l2_normalize(X, axis: int = -1, eps: float = 1e-12):
    """Apply L2 normalization along axis."""
    if _is_vec(X):
        nrm = builtins.max(norm(X), eps)
        return [x / nrm for x in X]
    out = []
    for row in X:
        nrm = builtins.max(norm(row), eps)
        out.append([x / nrm for x in row])
    return out

# -----------------------
# Shape manipulation
# -----------------------

def flatten(x: Union[Vector, Matrix]) -> Vector:
    """Flatten array to 1D."""
    if _is_vec(x):
        return x[:]
    result = []
    for row in x:
        result.extend(row)
    return result

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
