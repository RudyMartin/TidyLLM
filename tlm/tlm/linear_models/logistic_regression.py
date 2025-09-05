import random, math
from typing import List, Tuple, Optional
from ..core.activations import sigmoid
from ..pure.ops import dot

# Type definitions for consistency
Scalar = float
Vector = List[Scalar]
Matrix = List[Vector]

__all__ = ['fit','predict_proba','predict']

def fit(X: Matrix, y: List[int], lr: float = 1e-2, epochs: int = 200, reg: float = 0.0, fit_intercept: bool = True, seed: Optional[int] = None) -> Tuple[Vector, float, List[float]]:
    """Binary logistic regression via GD (pure Python).
    Complexity: per epoch O(n·d). Returns (w, b, history_ce).
    """
    n, d = len(X), len(X[0])
    w = [0.0]*d
    b = 0.0
    rng = random.Random(seed)
    history = []
    for _ in range(epochs):
        idx = list(range(n)); rng.shuffle(idx)
        # full batch update
        grad_w = [0.0]*d
        grad_b = 0.0
        ce = 0.0
        for i in idx:
            z = dot(X[i], w) + (b if fit_intercept else 0.0)
            p = sigmoid(z)
            err = p - y[i]
            for j in range(d):
                grad_w[j] += X[i][j] * err
            if fit_intercept:
                grad_b += err
            pi = min(max(p, 1e-12), 1-1e-12)
            ce += -(y[i]*math.log(pi) + (1-y[i])*math.log(1-pi))
        for j in range(d):
            grad_w[j] = grad_w[j]/n + reg*w[j]
            w[j] -= lr * grad_w[j]
        if fit_intercept:
            grad_b = grad_b/n
            b -= lr * grad_b
        history.append(ce/n)
    return w, b, history

def predict_proba(X: Matrix, w: Vector, b: float = 0.0) -> List[float]:
    """Predict class probabilities."""
    return [sigmoid(dot(x, w) + b) for x in X]

def predict(X: Matrix, w: Vector, b: float = 0.0, threshold: float = 0.5) -> List[int]:
    """Make binary predictions."""
    return [1 if p >= threshold else 0 for p in predict_proba(X, w, b)]
