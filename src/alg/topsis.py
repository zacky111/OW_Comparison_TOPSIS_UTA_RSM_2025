# src/alg/topsis.py
"""
Rozszerzona implementacja TOPSIS zgodnie z instrukcją:
- wszystkie kryteria traktujemy jako MIN (jeżeli któraś trzeba maksymalizować ->
  użytkownik powinien najpierw przekształcić dane)
- normalizacja: 'l1', 'l2', 'linf' lub 'range' (range = max-min)
- odległości liczone tą samą metryką, którą użyto do normalizacji
- ideal = min over P(U) (pareto front), nadir = max over P(U)
- wynik: lista (index, score) gdzie score ∈ [0,1] (im większy lepszy)
"""
import numpy as np

def _pareto_front(matrix):
    """Zwraca indeksy elementów nie-dominowanych (minimalizacja wszystkich kryteriów)."""
    m, n = matrix.shape
    is_dominated = np.zeros(m, dtype=bool)
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            # j dominates i if j <= i for all criteria and < for at least one (minimization)
            if np.all(matrix[j] <= matrix[i]) and np.any(matrix[j] < matrix[i]):
                is_dominated[i] = True
                break
    return np.where(~is_dominated)[0]

def _normalize(matrix, norm='l2', weights=None):
    """Normalizes columns of matrix according to chosen norm. Returns normalized matrix."""
    X = matrix.astype(float).copy()
    m, n = X.shape
    if weights is None:
        weights = np.ones(n, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if norm == 'l2':
        denom = np.sqrt(np.sum(X**2, axis=0))
        denom[denom == 0] = 1.0
        V = X / denom
    elif norm == 'l1':
        denom = np.sum(np.abs(X), axis=0)
        denom[denom == 0] = 1.0
        V = X / denom
    elif norm == 'linf':
        denom = np.max(np.abs(X), axis=0)
        denom[denom == 0] = 1.0
        V = X / denom
    elif norm == 'range':
        denom = np.max(X, axis=0) - np.min(X, axis=0)
        denom[denom == 0] = 1.0
        X_shift = X - np.min(X, axis=0)
        V = X_shift / denom
    else:
        raise ValueError("Unsupported norm: choose 'l1','l2','linf' or 'range'")
    # apply weights as compensatory multipliers (per instrukcji PDF)
    V = V * weights.reshape((1, -1))
    return V

def _distance(a, b, p='l2'):
    """Distance between vectors a and b: p can be 'l1','l2','linf'."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    if p == 'l2':
        return np.sqrt(np.sum(diff**2))
    if p == 'l1':
        return np.sum(np.abs(diff))
    if p == 'linf':
        return np.max(np.abs(diff))
    raise ValueError("Unsupported metric")

def calculate_topsis_score(data, weights=None, norm='l2'):
    """
    data: list-of-lists or numpy array (m x n), all criteria assumed to be minimized
    weights: array-like length n (weights for scaling, not preference weights)
    norm: 'l1','l2','linf','range' - determines normalization and distance metric
    returns: list of (index, score) where higher score == better
    """
    A = np.asarray(data, dtype=float)
    if A.ndim != 2:
        raise ValueError("Data must be 2D array-like")
    m, n = A.shape

    if weights is None:
        weights = np.ones(n, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.size != n:
        raise ValueError("weights length must equal number of criteria")

    # 1) wyznacz P(U) (pareto front)
    pf_idx = _pareto_front(A)
    if pf_idx.size == 0:
        pf_idx = np.arange(m)  # na wszelki wypadek

    # 2) oblicz ideal i nadir tylko na podstawie punktów niezdominowanych (PF)
    PF = A[pf_idx, :]
    ideal = np.min(PF, axis=0)   # dla minimizacji
    nadir = np.max(PF, axis=0)

    # 3) normalizacja - stosujemy tą samą metrykę do odległości
    V = _normalize(A, norm=norm, weights=weights)

    # normalizowane ideal i nadir (należy normalizować tak samo)
    # Aby to zrobić, normalizujemy wektor poprzez tę samą procedurę
    ideal_norm = _normalize(ideal.reshape((1, -1)), norm=norm, weights=weights).reshape(-1)
    nadir_norm = _normalize(nadir.reshape((1, -1)), norm=norm, weights=weights).reshape(-1)

    # 4) oblicz odległości i wynik c* = d_nadir / (d_ideal + d_nadir)
    metric_map = {'l1':'l1','l2':'l2','linf':'linf', 'range':'l2'}
    metric = metric_map.get(norm, 'l2')
    results = []
    for i in range(m):
        d_star = _distance(V[i], ideal_norm, p=metric)
        d_nadir = _distance(V[i], nadir_norm, p=metric)
        denom = d_star + d_nadir
        if denom == 0:
            score = 1.0  # iw przypadku idealnego dopasowania, dajemy maksymalny wynik
        else:
            score = d_nadir / denom
        results.append( (i, float(score)) )
    # sort descending by score
    results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
    return results_sorted
