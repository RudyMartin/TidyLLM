import random, math
from typing import List, Tuple, Optional
from ..core.activations import softmax

# Type definitions for consistency
Scalar = float
Vector = List[Scalar]
Matrix = List[Vector]

__all__ = ['fit','predict_proba','predict']

def fit(X: Matrix, y_idx: List[int], lr: float = 1e-2, epochs: int = 200, reg: float = 0.0, fit_intercept: bool = True, seed: Optional[int] = None) -> Tuple[Matrix, Vector, List[float]]:
    """Multiclass softmax regression (pure Python).
    Complexity: per epoch O(n·d·K). Returns (W, b, history_ce).
    """
    n, d = len(X), len(X[0])
    K = max(y_idx) + 1
    W = [[0.0 for _ in range(K)] for _ in range(d)]  # d×K
    b = [0.0 for _ in range(K)] if fit_intercept else [0.0]*K
    rng = random.Random(seed)
    hist = []
    for _ in range(epochs):
        idx = list(range(n)); rng.shuffle(idx)
        Z = []
        for i in idx:
            zi = [b[k] for k in range(K)]
            for k in range(K):
                for j in range(d):
                    zi[k] += X[i][j] * W[j][k]
            Z.append(zi)
        P = softmax(Z, axis=1)  # n×K
        grad_W = [[0.0 for _ in range(K)] for _ in range(d)]
        grad_b = [0.0 for _ in range(K)]
        ce = 0.0
        for t, i in enumerate(idx):
            yi = y_idx[i]
            for k in range(K):
                pk = P[t][k]
                err = pk - (1.0 if k == yi else 0.0)
                for j in range(d):
                    grad_W[j][k] += X[i][j] * err
                grad_b[k] += err
            ce += -math.log(max(P[t][yi], 1e-12))
        for j in range(d):
            for k in range(K):
                grad_W[j][k] = grad_W[j][k]/n + reg*W[j][k]
                W[j][k] -= lr * grad_W[j][k]
        if fit_intercept:
            for k in range(K):
                b[k] -= lr * (grad_b[k]/n)
        hist.append(ce/n)
    return W, b, hist

def predict_proba(X: Matrix, W: Matrix, b: Vector) -> Matrix:
    """Predict class probabilities."""
    Z = []
    K = len(W[0])
    for i in range(len(X)):
        zi = [b[k] for k in range(K)]
        for k in range(K):
            for j in range(len(W)):
                zi[k] += X[i][j] * W[j][k]
        Z.append(zi)
    return softmax(Z, axis=1)

def predict(X: Matrix, W: Matrix, b: Vector) -> List[int]:
    """Predict class labels."""
    P = predict_proba(X, W, b)
    return [max(range(len(P[0])), key=lambda k: P[i][k]) for i in range(len(P))]
