import math
from typing import List
from ..pure.ops import sum as asum

# Type definitions for consistency
Scalar = float
Vector = List[Scalar]
Matrix = List[Vector]

__all__ = ['mse','mae','binary_cross_entropy','cross_entropy']

def mse(y_true: List[float], y_pred: List[float]) -> float:
    """Calculate mean squared error."""
    n = len(y_true)
    return asum([(y_true[i]-y_pred[i])**2 for i in range(n)]) / n

def mae(y_true: List[float], y_pred: List[float]) -> float:
    """Calculate mean absolute error."""
    n = len(y_true)
    return asum([abs(y_true[i]-y_pred[i]) for i in range(n)]) / n

def binary_cross_entropy(p: List[float], y: List[int]) -> float:
    """Calculate binary cross entropy loss."""
    eps = 1e-12
    n = len(y)
    total = 0.0
    for i in range(n):
        pi = min(max(p[i], eps), 1.0-eps)
        yi = y[i]
        total += -(yi*math.log(pi) + (1-yi)*math.log(1-pi))
    return total / n

def cross_entropy(logits: Matrix, y_idx: List[int]) -> float:
    """Calculate categorical cross entropy loss."""
    # logits: (n,K) list; y_idx: (n) ints
    n = len(logits)
    total = 0.0
    for i in range(n):
        row = logits[i]
        m = max(row)
        ex = [math.exp(z - m) for z in row]
        s = sum(ex)
        pi = ex[y_idx[i]] / s
        total += -math.log(max(pi, 1e-12))
    return total / n
