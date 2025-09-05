import math
from typing import List, Union
from ..pure.ops import _apply1, max as amax, sum as asum

# Type definitions for consistency
Scalar = float
Vector = List[Scalar]
Matrix = List[Vector]

__all__ = ['sigmoid','relu','leaky_relu','softmax']

def sigmoid(x: Union[Scalar, Vector, Matrix]) -> Union[Scalar, Vector, Matrix]:
    """Apply sigmoid activation function."""
    def s(z):
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        ez = math.exp(z)
        return ez / (1.0 + ez)
    return _apply1(x, s)

def relu(x: Union[Scalar, Vector, Matrix]) -> Union[Scalar, Vector, Matrix]:
    """Apply ReLU activation function."""
    return _apply1(x, lambda z: z if z > 0 else 0.0)

def leaky_relu(x: Union[Scalar, Vector, Matrix], alpha: float = 0.01) -> Union[Scalar, Vector, Matrix]:
    """Apply Leaky ReLU activation function."""
    return _apply1(x, lambda z: z if z > 0 else alpha*z)

def softmax(X: Union[Vector, Matrix], axis: int = -1) -> Union[Vector, Matrix]:
    """Apply softmax activation function."""
    # supports 1D or 2D lists
    if not isinstance(X, list) or (len(X)>0 and not isinstance(X[0], list)):
        m = amax(X)
        ex = [math.exp(z - m) for z in X]
        s = asum(ex)
        return [e/s for e in ex]
    out = []
    for row in X:
        m = amax(row)
        ex = [math.exp(z - m) for z in row]
        s = asum(ex)
        out.append([e/s for e in ex])
    return out
