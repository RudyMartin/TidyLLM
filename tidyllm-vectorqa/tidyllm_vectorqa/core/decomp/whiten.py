from __future__ import annotations
import math

__all__ = ['whiten']

def whiten(Z, pca, eps: float = 1e-12):
    """Whiten PCA coordinates: Z_white[:,j] = Z[:,j] / sqrt(explained_var[j])."""
    ev = pca['explained_var']
    scales = [1.0/max(math.sqrt(v), eps) for v in ev]
    ZW = []
    for i in range(len(Z)):
        ZW.append([Z[i][j] * scales[j] for j in range(len(scales))])
    return ZW
