def calculate_topsis_score(data):
    return [(i, round(sum(row), 3)) for i, row in enumerate(data)]