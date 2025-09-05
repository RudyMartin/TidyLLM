from __future__ import annotations
import random, math
from ..pure.ops import mean

__all__ = ['fit', 'transform', 'fit_transform']

def _matvec_rows(X, v):
    return [sum(X[i][j]*v[j] for j in range(len(v))) for i in range(len(X))]

def _tmatvec_rows(X, t):
    d = len(X[0])
    out = [0.0]*d
    for i in range(len(X)):
        for j in range(d):
            out[j] += X[i][j] * t[i]
    return out

def _norm(v):
    return math.sqrt(sum(vi*vi for vi in v))

def _scale(v, s):
    return [vi/s for vi in v]

def _center_scale(X, center=True, scale=False):
    n, d = len(X), len(X[0])
    mu = [0.0]*d
    if center:
        mu = mean(X, axis=0)
    sc = [1.0]*d
    if scale:
        var = [0.0]*d
        for i in range(n):
            for j in range(d):
                var[j] += (X[i][j] - mu[j])**2
        for j in range(d):
            sc[j] = math.sqrt(var[j]/n) if var[j] > 0 else 1.0
    Xc = [[(X[i][j]-mu[j]) / sc[j] for j in range(d)] for i in range(n)]
    return Xc, mu, sc

def _apply_deflation(Xc, v):
    scores = _matvec_rows(Xc, v)
    d = len(v)
    for i in range(len(Xc)):
        si = scores[i]
        for j in range(d):
            Xc[i][j] -= si * v[j]

def _power_once(Xc, d, n, rng, iters, tol):
    v = [rng.random() - 0.5 for _ in range(d)]
    nv = _norm(v); v = _scale(v, nv if nv>0 else 1.0)
    prev = None
    for _ in range(iters):
        t = _matvec_rows(Xc, v)
        Av = _tmatvec_rows(Xc, t)
        if n > 1:
            Av = _scale(Av, (n-1))
        nAv = _norm(Av)
        if nAv == 0:
            break
        v_new = _scale(Av, nAv)
        if prev is not None:
            diff = _norm([v_new[i]-prev[i] for i in range(d)])
            if diff < tol:
                v = v_new
                break
        prev = v
        v = v_new
    t = _matvec_rows(Xc, v)
    lam = sum(ti*ti for ti in t) / max(n-1, 1)
    return v, lam

def fit(X, k=None, iters=1000, tol=1e-6, center=True, scale=False, n_init: int = 3, seed=None):
    """PCA via power iteration with `n_init` restarts per component (pure Python)."""
    Xc, mu, sc = _center_scale(X, center=center, scale=scale)
    n, d = len(Xc), len(Xc[0])
    if k is None:
        k = min(n, d)
    rng = random.Random(seed)
    comps, eigvals = [], []

    total_var = 0.0
    for j in range(d):
        total_var += sum(Xc[i][j]*Xc[i][j] for i in range(n)) / max(n-1,1)

    for _ in range(k):
        best_v, best_lam = None, -1.0
        for _start in range(max(1, n_init)):
            v_try, lam_try = _power_once(Xc, d, n, rng, iters, tol)
            if lam_try > best_lam:
                best_v, best_lam = v_try, lam_try
        comps.append(best_v[:])
        eigvals.append(best_lam)
        _apply_deflation(Xc, best_v)

    explained_ratio = [ (ev/total_var) if total_var>0 else 0.0 for ev in eigvals ]
    return {
        'components': comps,
        'mean': mu,
        'scale': sc,
        'explained_var': eigvals,
        'explained_ratio': explained_ratio,
    }

def transform(X, pca):
    """Transform data using fitted PCA."""
    mu, sc, comps = pca['mean'], pca['scale'], pca['components']
    Z = []
    for i in range(len(X)):
        rowc = [(X[i][j]-mu[j])/(sc[j]) for j in range(len(mu))]
        Z.append([sum(rowc[t]*comps[c][t] for t in range(len(rowc))) for c in range(len(comps))])
    return Z

def fit_transform(X, k=None, iters=1000, tol=1e-6, center=True, scale=False, n_init: int = 3, seed=None):
    """Fit PCA and transform data in one step."""
    p = fit(X, k=k, iters=iters, tol=tol, center=center, scale=scale, n_init=n_init, seed=seed)
    Z = transform(X, p)
    return Z, p
