import numpy as np

def compute_rsm_scores(data):
    X = np.array(data, dtype=float)

    # Punkt idealny (najlepsze wartości w każdym kryterium)
    RP = X.max(axis=0)

    # Odległość euklidesowa od punktu idealnego
    distances = np.linalg.norm(X - RP, axis=1)

    # Normalizacja odległości — im bliżej RP, tym wyższy score
    max_d = distances.max()
    scores = 1 - distances / max_d

    return [(i, float(round(scores[i], 4))) for i in range(len(scores))]
