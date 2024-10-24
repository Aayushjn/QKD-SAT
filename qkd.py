import enum
import math
import sys
from functools import cache
from itertools import combinations
from typing import Callable
from typing import Collection

import numpy as np
import numpy.typing as npt

from network import NetworkPath


class RiskFunction(enum.Enum):
    MOST_RISKY_NODE_BREAKS_SECRET = "most-risky"
    ATLEAST_ONE_NODE_BREAKS_SECRET = "any-node"


def risk_function(fn_type: RiskFunction) -> Callable[[Collection[float | np.float64]], float | np.float64]:
    if fn_type == RiskFunction.MOST_RISKY_NODE_BREAKS_SECRET:
        return max
    elif fn_type == RiskFunction.ATLEAST_ONE_NODE_BREAKS_SECRET:
        return lambda x: 1 - np.prod([1 - p for p in x])
    else:
        raise ValueError("Unsupported risk function")


@cache
def gathering_probability(node: int, paths: list[NetworkPath], collaboration_matrix: tuple[tuple[np.float64]]) -> float:
    """
    P_A -> probability that node A gathers all parts (collaboration)
    """
    collab = np.array(collaboration_matrix)
    total_parts = len(paths)
    reduced_paths = list(filter(lambda path: node not in path, paths))

    if len(reduced_paths) == 0:
        return 1.0
    
    return np.prod([(1.0 - np.prod([1 - collab[node, j] for j in path[1:-1]])) ** (len(reduced_paths) / total_parts) for path in reduced_paths])


@cache
def node_secret_breaking_probability(
    # path: NetworkPath,
    paths: tuple[NetworkPath],
    curiosity_matrix: tuple[np.float64],
    collaboration_matrix: tuple[tuple[np.float64, ...]],
    risk_fn: RiskFunction,
) -> float | np.float64:
    return risk_function(risk_fn)(
        [(curiosity_matrix[node]) * (gathering_probability(node, paths, collaboration_matrix)) for node in range(1, len(curiosity_matrix) - 1)]
    )


@cache
def compute_q_values(
    chosen_paths: tuple[NetworkPath, ...],
    # m: int,
    curiosity_matrix: tuple[np.float64],
    collaboration_matrix: tuple[tuple[np.float64]],
    risk_fn: RiskFunction,
) -> npt.NDArray[np.float64]:
    # q_values = np.zeros(m, dtype=np.float64)
    # for path_no, path in enumerate(chosen_paths):
    sbp = node_secret_breaking_probability(
        # path,
        chosen_paths,
        curiosity_matrix,
        collaboration_matrix,
        risk_fn,
    )
        # print(pbp)
        # q_values[path_no] += pbp
    return sbp


@cache
def optimality(
    paths: tuple[NetworkPath],
    curiosity_matrix: tuple[np.float64],
    collaboration_matrix: tuple[tuple[np.float64, ...]],
    risk_fn: RiskFunction = RiskFunction.MOST_RISKY_NODE_BREAKS_SECRET,
) -> tuple[int, tuple[NetworkPath, ...]]:
    optimal_parts = 1
    optimal_paths: tuple[NetworkPath, ...] = (paths[0],)

    max_q_value = 1.0
    

    for m in range(2, len(paths) + 1):
        # mean_q_values = np.zeros(m, dtype=np.float64)
        mean_sbp = 0.0
        path_combinations = combinations(paths, m)
        num_combinations = math.comb(len(paths), m)
        norm_factor = int(math.ceil(0.10 * num_combinations))
        tested_combinations = 0
        for chosen_paths in path_combinations:
            sbp = compute_q_values(
                chosen_paths, curiosity_matrix, collaboration_matrix, risk_fn
            )

            tested_combinations += 1
            if tested_combinations > norm_factor:
                
                mean_sbp /= norm_factor
                # mq = np.max(mean_q_values)
                if mean_sbp >= max_q_value:
                    return optimal_parts, optimal_paths

                max_q_value = mean_sbp
                optimal_parts = m
                optimal_paths = chosen_paths
                break
            else:
                mean_sbp += sbp

            

    return optimal_parts, optimal_paths
