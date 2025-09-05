from __future__ import annotations
import random, math
from typing import List, Tuple, Optional
from ..pure.ops import dot

# Type definitions for consistency
Scalar = float
Vector = List[Scalar]
Matrix = List[Vector]

__all__ = ['hinge_loss', 'fit', 'predict']

def hinge_loss(X: Matrix, y: List[int], w: Vector, b: float, C: float) -> float:
    """Objective: 0.5||w||^2 + C * mean(max(0, 1 - y*(Xw+b)))."""
    n = len(X)
    sq = sum(wj*wj for wj in w)
    total = 0.0
    for i in range(n):
        margin = y[i] * (dot(X[i], w) + b)
        total += max(0.0, 1.0 - margin)
    return 0.5 * sq + C * (total / n)

def fit(X: Matrix, y: List[int], lr: float = 1e-2, epochs: int = 200, C: float = 1.0, fit_intercept: bool = True, seed: Optional[int] = None) -> Tuple[Vector, float, List[float]]:
    """Linear SVM via subgradient SGD (pure Python). Returns (w, b, history_obj)."""
    n, d = len(X), len(X[0])
    w = [0.0]*d
    b = 0.0
    rng = random.Random(seed)
    history = []
    for _ in range(epochs):
        idx = list(range(n)); rng.shuffle(idx)
        for i in idx:
            xi = X[i]; yi = y[i]
            margin = yi * (dot(xi, w) + (b if fit_intercept else 0.0))
            if margin >= 1.0:
                for j in range(d):
                    w[j] -= lr * (w[j])
                if fit_intercept:
                    b -= 0.0
            else:
                for j in range(d):
                    w[j] -= lr * (w[j] - C * yi * xi[j])
                if fit_intercept:
                    b -= lr * (-C * yi)
        history.append(hinge_loss(X, y, w, b, C))
    return w, b, history

def predict(X: Matrix, w: Vector, b: float = 0.0) -> List[int]:
    """Make binary predictions with trained SVM."""
    return [1 if (dot(x, w) + b) >= 0 else -1 for x in X]
