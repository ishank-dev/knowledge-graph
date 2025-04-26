import math


def get_triplet_string(triplet: tuple[str, str, str]) -> str:
    return f"({triplet[0]}, {triplet[1]}, {triplet[2]})"


def process_types(allowed_types: list[str]) -> str:
    quoted = [f"'{t}'" for t in allowed_types]
    types_string = "\n-"
    types_string += '\n-'.join(quoted)
    types_string += '\n'
    return types_string


def rescale_figsize(base_figsize: float,
                    n_nodes: int,
                    max_figsize: float = 100,
                    min_figsize: float = 4) -> tuple[float, float]:
    multiplier = math.sqrt(n_nodes)
    x = max(base_figsize * multiplier, min_figsize)
    x = min(x, max_figsize)
    y = max(base_figsize * multiplier, min_figsize)
    y = min(y, max_figsize)
    return (x, y)
