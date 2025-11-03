def compute_rsm_scores(data):
    return [(i, round(sum(row) / len(row), 3)) for i, row in enumerate(data)]