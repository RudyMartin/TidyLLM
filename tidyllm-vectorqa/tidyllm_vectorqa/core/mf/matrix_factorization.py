from __future__ import annotations
import random

__all__ = ['fit_sgd', 'predict_full', 'mse']

def _dot(a, b):
    return sum(a[i]*b[i] for i in range(len(a)))

def fit_sgd(R, k: int, epochs: int = 50, lr: float = 0.01, reg: float = 0.02, seed=None):
    """Matrix factorization via SGD on observed entries. Returns (P, Q, history)."""
    rng = random.Random(seed)
    n = len(R); m = len(R[0])
    P = [[(rng.random()-0.5)*0.1 for _ in range(k)] for _ in range(n)]
    Q = [[(rng.random()-0.5)*0.1 for _ in range(k)] for _ in range(m)]
    obs = [(i,j) for i in range(n) for j in range(m) if R[i][j] is not None]
    hist = []
    for _ in range(epochs):
        rng.shuffle(obs)
        se = 0.0; cnt = 0
        for (i,j) in obs:
            rij = R[i][j]
            pred = _dot(P[i], Q[j])
            err = rij - pred
            for f in range(k):
                pif = P[i][f]
                qjf = Q[j][f]
                P[i][f] += lr * (err * qjf - reg * pif)
                Q[j][f] += lr * (err * pif - reg * qjf)
            se += err*err; cnt += 1
        hist.append(se / max(cnt,1))
    return P, Q, hist

def predict_full(P, Q):
    """Reconstruct full rating matrix."""
    n, k = len(P), len(P[0])
    m = len(Q)
    M = [[0.0]*m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            s = 0.0
            for f in range(k):
                s += P[i][f]*Q[j][f]
            M[i][j] = s
    return M

def mse(R, P, Q):
    """Calculate MSE on observed entries."""
    se = 0.0; cnt = 0
    for i in range(len(R)):
        for j in range(len(R[0])):
            if R[i][j] is not None:
                pred = 0.0
                for f in range(len(P[0])):
                    pred += P[i][f]*Q[j][f]
                err = R[i][j] - pred
                se += err*err; cnt += 1
    return se / max(cnt,1)
