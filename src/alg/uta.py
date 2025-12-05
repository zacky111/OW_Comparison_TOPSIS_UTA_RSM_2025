
import numpy as np

def compute_uta_results(data):
    A = np.asarray(data, dtype=float)
    m, n = A.shape
    mins = np.min(A, axis=0)
    maxs = np.max(A, axis=0)
    denom = maxs - mins
    denom[denom == 0] = 1.0
    X = (A - mins) / denom
    U = 1.0 - X
    w = np.ones(n) / n
    scores = U.dot(w)
    results = [(i, float(scores[i])) for i in range(m)]
    results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
    return results_sorted
