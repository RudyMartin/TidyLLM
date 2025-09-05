from __future__ import annotations
import math

__all__ = ['fit', 'predict_log_proba', 'predict']

def fit(X_counts, y, alpha: float = 1.0):
    """Fit Multinomial NB (log-space). Returns model dict."""
    n, d = len(X_counts), len(X_counts[0])
    K = max(y) + 1
    Nc = [0]*K
    for yi in y:
        Nc[yi] += 1
    pi_log = [math.log((Nc[c] + 1.0) / (n + K)) for c in range(K)]
    wc = [[0.0]*d for _ in range(K)]
    for i in range(n):
        yi = y[i]
        xi = X_counts[i]
        for j in range(d):
            wc[yi][j] += xi[j]
    log_theta = [[0.0]*d for _ in range(K)]
    for c in range(K):
        total_c = sum(wc[c][j] + alpha for j in range(d))
        for j in range(d):
            log_theta[c][j] = math.log((wc[c][j] + alpha) / total_c)
    return {
        'pi_log': pi_log,
        'log_theta': log_theta,
        'K': K,
        'd': d,
    }

def predict_log_proba(X_counts, model):
    """Predict log probabilities."""
    pi_log = model['pi_log']
    log_theta = model['log_theta']
    K, d = model['K'], model['d']
    out = []
    for i in range(len(X_counts)):
        xi = X_counts[i]
        scores = [pi_log[c] for c in range(K)]
        for c in range(K):
            s = scores[c]
            for j in range(d):
                if xi[j] != 0:
                    s += xi[j] * log_theta[c][j]
            scores[c] = s
        m = max(scores)
        Z = m + float(math.log(sum(math.exp(sc - m) for sc in scores)))
        out.append([sc - Z for sc in scores])
    return out

def predict(X_counts, model):
    """Predict class labels."""
    logp = predict_log_proba(X_counts, model)
    return [max(range(model['K']), key=lambda c: logp[i][c]) for i in range(len(X_counts))]
