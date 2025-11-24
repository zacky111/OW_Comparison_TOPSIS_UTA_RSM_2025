# src/alg/spcs.py
"""
Prosty wariant SP-CS (safety principle + compromise selection)
- budujemy łamaną szkieletową jako linię łączącą centroid najlepszych i najgorszych klas
  (można potem rozszerzyć o wszystkie pary z sąsiednich klas)
- parametryzacja t: 0 (początek) -> 1 (koniec)
- metryka rzutów: używamy projekcji w sensie zwykłego rzutu (następnie odległość L_inf)
- score = t + distance(F(u), gamma) (odległość znormalizowana), a następnie skalujemy, by większy == lepszy
"""
import numpy as np
from src.alg.topsis import _pareto_front, _normalize, _distance

def compute_spcs_scores(data, weights=None, norm='linf'):
    A = np.asarray(data, dtype=float)
    m, n = A.shape
    if weights is None:
        weights = np.ones(n, dtype=float)

    # stworzyć warstwy Pareto
    remaining = A.copy()
    idx_map = np.arange(m)
    layers = []
    while remaining.shape[0] > 0:
        pf = _pareto_front(remaining)
        if pf.size == 0:
            layers.append(idx_map)
            break
        layers.append(idx_map[pf])
        mask = np.ones(remaining.shape[0], dtype=bool)
        mask[pf] = False
        remaining = remaining[mask]
        idx_map = idx_map[mask]

    if len(layers) < 2:
        # fallback: użyj min/max globalnych
        a = np.min(A, axis=0)
        b = np.max(A, axis=0)
    else:
        a = np.mean(A[layers[0], :], axis=0)     # aspiracje (najlepsze)
        b = np.mean(A[layers[-1], :], axis=0)    # status-quo (najgorsze)

    # znormalizuj wszystkie wektory tą samą procedurą
    V = _normalize(A, norm=norm, weights=weights)
    an = _normalize(a.reshape((1,-1)), norm=norm, weights=weights).reshape(-1)
    bn = _normalize(b.reshape((1,-1)), norm=norm, weights=weights).reshape(-1)

    # parametryzacja linii L(t) = an + t*(bn - an)
    dir_vec = bn - an
    dir_norm_sq = np.dot(dir_vec, dir_vec)
    scores_raw = []
    distances = []
    ts = []
    metric_map = {'l1':'l1','l2':'l2','linf':'linf', 'range':'l2'}
    metric = metric_map.get(norm, 'linf')

    for i in range(m):
        x = V[i]
        if dir_norm_sq == 0:
            t = 0.0
            proj = an
        else:
            t = np.dot(x - an, dir_vec) / dir_norm_sq
            t = float(np.clip(t, 0.0, 1.0))
            proj = an + t * dir_vec
        dist = _distance(x, proj, p=metric)
        ts.append(t)
        distances.append(dist)
        scores_raw.append(t + dist)  # założenie: mniejsza suma = bliżej aspiracji i linii
    # Mamy raw = t + dist (mniejsze lepsze) -> zamienimy na score większe lepsze
    raw = np.array(scores_raw, dtype=float)
    # inwersja: nscore = 1 - (raw - min)/(max-min)
    minr, maxr = raw.min(), raw.max()
    if maxr - minr == 0:
        nscore = np.ones_like(raw) * 0.5
    else:
        nscore = 1.0 - (raw - minr) / (maxr - minr)

    results = [(i, float(nscore[i])) for i in range(m)]
    results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
    return results_sorted
