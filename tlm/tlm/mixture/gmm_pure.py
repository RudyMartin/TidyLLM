from __future__ import annotations
import math, random

__all__ = ['fit', 'predict_proba', 'predict']

def _logsumexp(vec):
    m = max(vec)
    s = 0.0
    for v in vec:
        s += math.exp(v - m)
    return m + math.log(s)

def _log_gauss_diag(x, mu, var_diag):
    s = 0.0
    for j in range(len(x)):
        vj = max(var_diag[j], 1e-12)
        s += (x[j]-mu[j])**2 / vj + math.log(2*math.pi*vj)
    return -0.5 * s

def _log_gauss_spherical(x, mu, var_scalar):
    v = max(var_scalar, 1e-12)
    d = len(x)
    dist2 = 0.0
    for j in range(d):
        dist2 += (x[j]-mu[j])**2
    return -0.5 * (dist2 / v + d * math.log(2*math.pi*v))

def fit(X, k, max_iter=100, tol=1e-3, cov='diag', seed=None):
    """Pure‑Python EM for GMM. Returns params dict {'pi','mu','var','cov','loglik'}."""
    assert cov in ('diag','spherical')
    rng = random.Random(seed)
    n = len(X); d = len(X[0])

    mu = [X[i][:] for i in rng.sample(range(n), k)]

    global_var = [0.0]*d
    mean_all = [0.0]*d
    for i in range(n):
        for j in range(d):
            mean_all[j] += X[i][j]
    for j in range(d):
        mean_all[j] /= n
    for i in range(n):
        for j in range(d):
            global_var[j] += (X[i][j]-mean_all[j])**2
    for j in range(d):
        global_var[j] = max(global_var[j]/max(n-1,1), 1e-3)

    if cov == 'diag':
        var = [global_var[:] for _ in range(k)]
    else:
        g = sum(global_var)/d
        var = [g for _ in range(k)]

    pi = [1.0/k for _ in range(k)]
    loglik_hist = []
    for _ in range(max_iter):
        log_r = []
        ll = 0.0
        for i in range(n):
            row = []
            for c in range(k):
                if cov == 'diag':
                    row.append(math.log(pi[c]) + _log_gauss_diag(X[i], mu[c], var[c]))
                else:
                    row.append(math.log(pi[c]) + _log_gauss_spherical(X[i], mu[c], var[c]))
            lse = _logsumexp(row)
            ll += lse
            row_prob = [math.exp(v - lse) for v in row]
            log_r.append(row_prob)
        loglik_hist.append(ll)
        if len(loglik_hist) > 1 and abs(loglik_hist[-1] - loglik_hist[-2]) < tol:
            break

        Nk = [0.0]*k
        mu_new = [[0.0]*d for _ in range(k)]
        for i in range(n):
            ri = log_r[i]
            for c in range(k):
                rc = ri[c]
                Nk[c] += rc
                for j in range(d):
                    mu_new[c][j] += rc * X[i][j]
        for c in range(k):
            denom = max(Nk[c], 1e-12)
            for j in range(d):
                mu_new[c][j] /= denom

        if cov == 'diag':
            var_new = [[1e-9]*d for _ in range(k)]
            for i in range(n):
                ri = log_r[i]
                for c in range(k):
                    rc = ri[c]
                    for j in range(d):
                        diff = X[i][j] - mu_new[c][j]
                        var_new[c][j] += rc * diff * diff
            for c in range(k):
                denom = max(Nk[c], 1e-12)
                for j in range(d):
                    var_new[c][j] = max(var_new[c][j] / denom, 1e-9)
        else:
            var_new = [1e-9 for _ in range(k)]
            for i in range(n):
                ri = log_r[i]
                for c in range(k):
                    rc = ri[c]
                    dist2 = 0.0
                    for j in range(d):
                        diff = X[i][j] - mu_new[c][j]
                        dist2 += diff*diff
                    var_new[c] += rc * dist2
            for c in range(k):
                denom = max(Nk[c], 1e-12)
                var_new[c] = max( (var_new[c] / denom) / d, 1e-9)

        pi = [Nk[c]/n for c in range(k)]
        mu, var = mu_new, var_new

    return {'pi': pi, 'mu': mu, 'var': var, 'cov': cov, 'loglik': loglik_hist}

def predict_proba(X, params):
    """Predict cluster probabilities."""
    k = len(params['pi'])
    cov = params['cov']
    out = []
    for i in range(len(X)):
        row = []
        for c in range(k):
            if cov == 'diag':
                row.append(math.log(params['pi'][c]) + _log_gauss_diag(X[i], params['mu'][c], params['var'][c]))
            else:
                row.append(math.log(params['pi'][c]) + _log_gauss_spherical(X[i], params['mu'][c], params['var'][c]))
        lse = _logsumexp(row)
        out.append([math.exp(v - lse) for v in row])
    return out

def predict(X, params):
    """Predict cluster assignments."""
    P = predict_proba(X, params)
    return [ max(range(len(P[0])), key=lambda c: P[i][c]) for i in range(len(P)) ]
