import enum
from functools import cache
from itertools import combinations
from math import ceil
from math import comb
from typing import Callable
from typing import Collection

import numpy as np
import numpy.typing as npt

from network import NetworkPath


class RiskFunction(enum.Enum):
    MOST_RISKY_NODE = "most-risky-node"
    ANY_NODE = "any-node"


def risk_function(fn_type: RiskFunction) -> Callable[[Collection[float | np.float64]], float | np.float64]:
    if fn_type == RiskFunction.MOST_RISKY_NODE:
        return max
    elif fn_type == RiskFunction.ANY_NODE:
        return lambda x: 1.0 - np.prod([1 - p for p in x])
    else:
        raise ValueError("Unsupported risk function")


@cache
def gathering_probability(node: int, paths: list[NetworkPath], collaboration_matrix: tuple[tuple[np.float64]]) -> float:
    """
    P_A -> probability that node A gathers all parts (collaboration)
    """
    reduced_paths = list(filter(lambda path: node not in path, paths))
    m, k = len(paths), len(reduced_paths)
    power_factor = k / m

    # if a node lies on all paths, then it doesn't need to "gather" collaboratively
    if power_factor == 1:
        return 1.0

    collab = np.array(collaboration_matrix)
    return np.prod(
        [(1.0 - np.prod([1 - collab[node, j] for j in path[1:-1]])) ** power_factor for path in reduced_paths]
    )


@cache
def path_breaking_probability(
    # path: NetworkPath,
    paths: tuple[NetworkPath],
    curiosity_matrix: tuple[np.float64],
    collaboration_matrix: tuple[tuple[np.float64, ...]],
    risk_fn: RiskFunction,
) -> float | np.float64:
    return risk_function(risk_fn)(
        [
            curiosity_matrix[node] * gathering_probability(node, paths, collaboration_matrix)
            for node in range(1, len(curiosity_matrix) - 1)
        ]
    )


@cache
def compute_q_values(
    chosen_paths: tuple[NetworkPath, ...],
    curiosity_matrix: tuple[np.float64],
    collaboration_matrix: tuple[tuple[np.float64]],
    risk_fn: RiskFunction,
) -> npt.NDArray[np.float64]:
    return np.array(
        [
            path_breaking_probability(chosen_paths, curiosity_matrix, collaboration_matrix, risk_fn)
            for _ in range(len(chosen_paths))
        ]
    )


@cache
def optimality(
    paths: tuple[NetworkPath],
    curiosity_matrix: tuple[np.float64],
    collaboration_matrix: tuple[tuple[np.float64, ...]],
    risk_fn: RiskFunction = RiskFunction.MOST_RISKY_NODE,
) -> tuple[int, tuple[NetworkPath, ...]]:
    optimal_parts = 1
    optimal_paths: tuple[NetworkPath, ...] = (paths[0],)

    max_q_value = 1.0
    tested_combinations = 0

    for m in range(2, len(paths) + 1):
        mean_q_values = np.zeros(m)
        norm_factor = ceil(0.25 * comb(len(paths), m))
        path_combinations = combinations(paths, m)
        for chosen_paths in path_combinations:
            q_values = compute_q_values(chosen_paths, curiosity_matrix, collaboration_matrix, risk_fn)
            mean_q_values += q_values
            tested_combinations += 1
            if tested_combinations >= norm_factor:
                mean_q_values /= norm_factor
                break

        mq = np.max(mean_q_values)
        if mq >= max_q_value:
            return optimal_parts, optimal_paths

        max_q_value = mq
        optimal_parts = m
        optimal_paths = chosen_paths

    return optimal_parts, optimal_paths
