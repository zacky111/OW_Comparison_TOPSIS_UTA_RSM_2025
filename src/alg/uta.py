# src/alg/uta.py
"""
Prosty wariant UTA-DIS (heurystyczny):
- normalizujemy wszystkie kryteria do [0,1] (minimization assumed)
- zakładamy addytywną funkcję użyteczności z równymi krokami (prosty wielomian)
- dopasowujemy proste piecewise-linear utility przez ranking domyślny (tu: równy weighting)
Ten moduł daje wynik porównywalny do innych metod - można później podmienić na pełen solver UTA-Star.
"""
import numpy as np

def compute_uta_results(data):
    A = np.asarray(data, dtype=float)
    m, n = A.shape
    # normalizacja do [0,1] (min -> 0, max ->1)
    mins = np.min(A, axis=0)
    maxs = np.max(A, axis=0)
    denom = maxs - mins
    denom[denom == 0] = 1.0
    X = (A - mins) / denom  # 0..1, ale przypominam: oryginalne były MIN -> lepsze gdy mniejsze
    # ponieważ mniejsze lepsze, transformujemy: util_j = 1 - X_j
    U = 1.0 - X
    # proste wagi równomierne (można w przyszłości zastąpić estymacją)
    w = np.ones(n) / n
    scores = U.dot(w)
    results = [(i, float(scores[i])) for i in range(m)]
    results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
    return results_sorted
