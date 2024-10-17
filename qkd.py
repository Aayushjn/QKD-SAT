import enum
import sys
from functools import cache
from itertools import combinations
from typing import Callable
from typing import Collection

import numpy as np
import numpy.typing as npt

from network import NetworkPath


class RiskFunction(enum.Enum):
    MAX = "max"
    SUM = "sum"
    MEAN = "mean"


def risk_function(fn_type: RiskFunction) -> Callable[[Collection[float | np.float64]], float | np.float64]:
    if fn_type == RiskFunction.MAX:
        return max
    elif fn_type == RiskFunction.SUM:
        return sum
    elif fn_type == RiskFunction.MEAN:
        return lambda x: sum(x) / len(x)
    else:
        raise ValueError("Unsupported risk function")


@cache
def gathering_probability(node: int, paths: list[NetworkPath], collaboration_matrix: tuple[tuple[np.float64]]) -> float:
    """
    P_A -> probability that node A gathers all parts (collaboration)
    """
    collab = np.array(collaboration_matrix)
    return np.prod([1.0 - np.prod([1 - collab[node, j] for j in path[1:-1]]) for path in paths])


@cache
def path_breaking_probability(
    path: NetworkPath,
    paths: tuple[NetworkPath],
    num_shares: int,
    curiosity_matrix: tuple[np.float64],
    collaboration_matrix: tuple[tuple[np.float64, ...]],
    risk_fn: RiskFunction,
) -> float | np.float64:
    return risk_function(risk_fn)(
        [
            (curiosity_matrix[node] ** num_shares) * gathering_probability(node, paths, collaboration_matrix)
            for node in path[1:-1]
        ]
    )


@cache
def compute_mean_q_values(
    chosen_paths: tuple[NetworkPath, ...],
    m: int,
    norm_factor: int,
    curiosity_matrix: tuple[np.float64],
    collaboration_matrix: tuple[tuple[np.float64]],
    risk_fn: RiskFunction,
) -> npt.NDArray[np.float64]:
    mean_q_values = np.zeros(m)
    for _ in range(norm_factor):
        for path_no, path in enumerate(chosen_paths):
            pbp = path_breaking_probability(
                path,
                chosen_paths[:path_no] + chosen_paths[path_no + 1 :],
                m,
                curiosity_matrix,
                collaboration_matrix,
                risk_fn,
            )
            # print(pbp)
            mean_q_values[path_no] += pbp
    return mean_q_values / norm_factor


@cache
def optimality(
    paths: tuple[NetworkPath],
    curiosity_matrix: tuple[np.float64],
    collaboration_matrix: tuple[tuple[np.float64, ...]],
    risk_fn: RiskFunction = RiskFunction.MAX,
) -> tuple[int, tuple[NetworkPath, ...]]:
    optimal_parts = 0
    optimal_paths: tuple[NetworkPath, ...] = (paths[0],)

    if risk_fn == RiskFunction.SUM:
        max_q_value = sys.maxsize
    else:
        max_q_value = 1
    norm_factor = int(0.25 * len(paths))

    for m in range(2, len(paths) + 1):
        path_combinations = combinations(paths, m)
        for chosen_paths in path_combinations:
            mean_q_values = compute_mean_q_values(
                chosen_paths, m, norm_factor, curiosity_matrix, collaboration_matrix, risk_fn
            )
            if np.max(mean_q_values) >= max_q_value:
                return optimal_parts, optimal_paths

            max_q_value = max(mean_q_values)
            optimal_parts = m
            optimal_paths = chosen_paths

            break

    return optimal_parts, optimal_paths
