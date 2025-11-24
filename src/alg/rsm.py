# src/alg/rsm.py
"""
Prosty wariant metody Reference Set Method (RSM).
- Jeśli brak oddzielnych punktów odniesienia, tworzymy klasy przez kolejne fronty Pareto:
  A0 = P(U), A1 = P(U\A0), ..., ostatnia klasa = najgorsze punkty.
- Dla prostoty agregujemy R1 jako 'najlepszą' klasę (A0), R-1 jako 'najgorszą' (ostatnia klasa)
- score(u) = normalized( dist(u, R-1) - dist(u, R1) ) -> większe = lepsze
"""
import numpy as np
from src.alg.topsis import _pareto_front, _distance, _normalize

def _pareto_layers(X):
    """Zwraca listę list indeksów kolejnych frontów Pareto (A0,A1,...)."""
    X = np.asarray(X, dtype=float)
    remaining = np.arange(X.shape[0])
    layers = []
    M = X.copy()
    while remaining.size > 0:
        idx = _pareto_front(M)
        if idx.size == 0:
            break
        # map local idx to global indices
        layers.append(remaining[idx])
        mask = np.ones(M.shape[0], dtype=bool)
        mask[idx] = False
        M = M[mask]
        remaining = remaining[mask]
    return layers

def compute_rsm_scores(data, weights=None, norm='l2'):
    """
    data: array-like m x n
    returns list of (index, score) sorted descending
    """
    A = np.asarray(data, dtype=float)
    m, n = A.shape
    if weights is None:
        weights = np.ones(n, dtype=float)

    layers = _pareto_layers(A)
    if len(layers) == 0:
        # fallback: simple average score
        return [(i, float(np.mean(A[i]))) for i in range(m)]

    R1_idx = layers[0]             # najlepsza klasa
    Rminus_idx = layers[-1]        # najgorsza klasa

    # aggregate points: use centroid of each set
    R1 = np.mean(A[R1_idx, :], axis=0)
    Rm = np.mean(A[Rminus_idx, :], axis=0)

    # normalize same way as TOPSIS (so distances are consistent)
    V = _normalize(A, norm=norm, weights=weights)
    R1n = _normalize(R1.reshape((1,-1)), norm=norm, weights=weights).reshape(-1)
    Rmn = _normalize(Rm.reshape((1,-1)), norm=norm, weights=weights).reshape(-1)

    metric_map = {'l1':'l1','l2':'l2','linf':'linf', 'range':'l2'}
    metric = metric_map.get(norm, 'l2')

    raw_scores = []
    for i in range(m):
        d_to_bad = _distance(V[i], Rmn, p=metric)
        d_to_good = _distance(V[i], R1n, p=metric)
        # want bigger scores for better alternatives: larger d_to_bad and smaller d_to_good
        raw = d_to_bad - d_to_good
        raw_scores.append(raw)
    # normalize raw_scores to [0,1]
    raw_arr = np.array(raw_scores, dtype=float)
    minv, maxv = raw_arr.min(), raw_arr.max()
    if maxv - minv == 0:
        normed = np.ones_like(raw_arr) * 0.5
    else:
        normed = (raw_arr - minv) / (maxv - minv)
    results = [(i, float(normed[i])) for i in range(m)]
    results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
    return results_sorted
