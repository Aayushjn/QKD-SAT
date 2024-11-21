import enum
import math
import sys
from concurrent.futures import as_completed
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import cache
from itertools import combinations
from itertools import islice
from multiprocessing import cpu_count
from multiprocessing import Value
from typing import Callable
from typing import Collection

import numpy as np
from tqdm import tqdm
from tqdm import trange

from network import NetworkPath


@dataclass
class OptimalResult:
    shares: int
    paths: tuple[NetworkPath, ...]
    breaking_probability: np.float64


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
    reduced_paths = list(filter(lambda path: node not in path, paths))
    power_factor = len(reduced_paths) / len(paths)

    if power_factor == 0:
        return 1.0

    path_probs = [(1.0 - np.prod([1 - collab[node, j] for j in path[1:-1]])) ** power_factor for path in reduced_paths]

    return np.prod(path_probs)


@cache
def node_secret_breaking_probability(
    paths: tuple[NetworkPath],
    curiosity_matrix: tuple[np.float64],
    collaboration_matrix: tuple[tuple[np.float64, ...]],
    risk_fn: RiskFunction,
) -> float | np.float64:
    return risk_function(risk_fn)(
        [
            (curiosity_matrix[node]) * (gathering_probability(node, paths, collaboration_matrix))
            for node in range(1, len(curiosity_matrix) - 1)
        ]
    )


@cache
def optimality(
    paths: tuple[NetworkPath],
    curiosity_matrix: tuple[np.float64],
    collaboration_matrix: tuple[tuple[np.float64, ...]],
    risk_fn: RiskFunction = RiskFunction.MOST_RISKY_NODE_BREAKS_SECRET,
) -> OptimalResult:
    optimal_parts = 0
    optimal_paths = tuple()
    max_q_value = Value("d", sys.maxsize)

    num_workers = (cpu_count() or 2) - 1
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for m in trange(2, len(paths) + 1, desc="Optimizing", leave=False, unit="parts", dynamic_ncols=True):
            optimized = False
            path_combinations = combinations(paths, m)
            num_combinations = math.comb(len(paths), m)
            # sample_size = math.ceil(0.10 * num_combinations)
            # sampled_combinations = islice(path_combinations, sample_size)

            futures = {
                executor.submit(
                    node_secret_breaking_probability, chosen_paths, curiosity_matrix, collaboration_matrix, risk_fn
                ): chosen_paths
                for chosen_paths in path_combinations
            }
            for future in tqdm(
                as_completed(futures.keys()),
                desc="Testing paths",
                total=num_combinations,
                unit="paths",
                leave=False,
                dynamic_ncols=True,
            ):
                with max_q_value.get_lock():
                    if (res := future.result()) < max_q_value.value:
                        max_q_value.value = res
                        optimized = True
                        optimal_paths = tuple(futures[future])

            if not optimized:
                return OptimalResult(shares=optimal_parts, paths=optimal_paths, breaking_probability=max_q_value.value)

            optimal_parts = m

        return OptimalResult(shares=0, paths=(), breaking_probability=np.float64(0.0))
