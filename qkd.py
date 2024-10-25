import enum
import math
from concurrent.futures import as_completed
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from itertools import combinations
from itertools import islice
from multiprocessing import cpu_count
from typing import Callable
from typing import Collection

import numpy as np
from tqdm import tqdm
from tqdm import trange
from tqdm.contrib.concurrent import process_map

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
) -> int:
    optimal_parts = 1
    optimal_paths: tuple[NetworkPath, ...] = (paths[0],)

    max_q_value = 1.0

    num_workers = (cpu_count() or 2) - 1
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for m in trange(2, len(paths) + 1, desc="Optimizing", leave=False, unit="parts", dynamic_ncols=True):
            path_combinations = combinations(paths, m)
            num_combinations = math.comb(len(paths), m)
            sample_size = math.ceil(0.10 * num_combinations)
            sampled_combinations = islice(path_combinations, sample_size)

            # mean_sbp = np.mean(executor.map(node_secret_breaking_probability, ((chosen_paths, curiosity_matrix, collaboration_matrix, risk_fn) for chosen_paths in sampled_combinations), chunksize=round(sample_size / num_workers)))
            futures = (
                executor.submit(
                    node_secret_breaking_probability, chosen_paths, curiosity_matrix, collaboration_matrix, risk_fn
                )
                for chosen_paths in sampled_combinations
            )
            mean_sbp = np.mean(
                np.fromiter(
                    (
                        future.result()
                        for future in tqdm(
                            as_completed(futures),
                            desc="Testing paths",
                            total=sample_size,
                            unit="paths",
                            leave=False,
                            dynamic_ncols=True,
                        )
                    ),
                    np.float64,
                )
            )
            # mean_sbp = 0.0
            # for future in tqdm(as_completed(futures), total=len(futures), dynamic_ncols=True):
            #     mean_sbp += future.result()
            # for chosen_paths in sampled_combinations:
            #     mean_sbp += node_secret_breaking_probability(chosen_paths, curiosity_matrix, collaboration_matrix, risk_fn)
            # mean_sbp /= sample_size

            if mean_sbp >= max_q_value:
                return optimal_parts

            max_q_value = mean_sbp
            optimal_parts = m

        return optimal_parts
