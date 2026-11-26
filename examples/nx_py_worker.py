def mean(values):
    return float(sum(values) / len(values))


def label(values):
    return f"count={len(values)}"


def doubles(values):
    return [float(value * 2) for value in values]
