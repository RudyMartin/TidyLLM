from __future__ import annotations
import math

__all__ = ['fit', 'score_logpdf', 'predict']

def fit(X):
    """Estimate diagonal Gaussian parameters per feature. Returns (mu, var)."""
    n, d = len(X), len(X[0])
    mu = [0.0]*d
    for i in range(n):
        for j in range(d):
            mu[j] += X[i][j]
    for j in range(d):
        mu[j] /= n
    var = [0.0]*d
    for i in range(n):
        for j in range(d):
            diff = X[i][j] - mu[j]
            var[j] += diff*diff
    for j in range(d):
        var[j] = max(var[j] / n, 1e-12)
    return mu, var

def score_logpdf(X, mu, var):
    """Compute log probability density scores."""
    n, d = len(X), len(mu)
    out = [0.0]*n
    cst = [0.5*math.log(2*math.pi*v) for v in var]
    for i in range(n):
        s = 0.0
        for j in range(d):
            vj = var[j]
            s += cst[j] + 0.5 * ((X[i][j]-mu[j])**2) / vj
        out[i] = -s
    return out

def predict(X, mu, var, eps: float | None = None, percentile: float | None = None):
    """Predict anomalies using threshold."""
    scores = score_logpdf(X, mu, var)
    if percentile is not None:
        srt = sorted(scores)
        k = max(0, min(len(srt)-1, int(len(srt) * (percentile/100.0))))
        thr = srt[k]
    elif eps is not None:
        thr = math.log(max(eps, 1e-300))
    else:
        srt = sorted(scores)
        k = max(0, int(len(srt) * 0.01))
        thr = srt[k]
    return [1 if s < thr else 0 for s in scores]
