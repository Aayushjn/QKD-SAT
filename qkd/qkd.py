from itertools import combinations_with_replacement, combinations
from network import NetworkPath
import numpy as np
from functools import cache


def gathering_probability(node: int, paths: list[NetworkPath], collaboration_matrix: np.ndarray[np.float64]) -> float:
    """
    P_A -> probability that node A gathers all parts (collaboration)
    """
    return np.prod([1.0 - np.prod([1 - collaboration_matrix[node, j] for j in path[1:-1]]) for path in paths])

def decoding_probabiliity(node: int, num_shares: int, curiosity_matrix: np.ndarray[np.float64]) -> float:
    return curiosity_matrix[node] ** num_shares

def breaking_probabilities(gathering_probability: np.ndarray[np.float64], decoding_probability: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    return gathering_probability * decoding_probability

def no_node_breaks_secret(breaking_probabilities: np.ndarray[np.float64]) -> float:
    return np.prod(1 - breaking_probabilities[1:-1])

def some_node_breaks_secret(breaking_probabilities: np.ndarray[np.float64]) -> float:
    return 1 - max(breaking_probabilities[1:-1])

def objective_value(num_nodes: int, curiosity_matrix: np.ndarray[np.float64], collaboration_matrix: np.ndarray[np.float64], paths: list[NetworkPath]) -> float:
    num_parts = len(paths)
    gathering_probabilities = np.array([gathering_probability(i, paths, collaboration_matrix) for i in range(num_nodes)])
    decoding_probabiliities = np.array([decoding_probabiliity(i, num_parts, curiosity_matrix) for i in range(num_nodes)])

    return no_node_breaks_secret(gathering_probabilities * decoding_probabiliities)

# def optimality(num_nodes: int, curiosity_matrix: np.ndarray[np.float64], collaboration_matrix: np.ndarray[np.float64], paths: list[NetworkPath]) -> int:
#     num_shares = 0

#     for _ in range(20):
#         optimal_paths = None
#         max_value = -1
#         for path in combinations_with_replacement(paths, num_shares):
#             value = objective_value(num_nodes, curiosity_matrix, collaboration_matrix, path)
#             if value > max_value:
#                 max_value = value
#                 optimal_paths = path
        
#         if optimal_paths is not None:
#             value = objective_value(num_nodes, curiosity_matrix, collaboration_matrix, optimal_paths)
#             print(value, num_shares)
#         num_shares += 1

@cache
def path_breaking_probability(path: NetworkPath, paths: tuple[NetworkPath], num_shares: int, curiosity_matrix: tuple[np.float64], collaboration_matrix: tuple[tuple[np.float64, ...]]) -> float:
    return max((curiosity_matrix[node] ** num_shares) * gathering_probability(node, paths, np.array(collaboration_matrix)) for node in path[1:-1])


def optimality(paths: list[NetworkPath], curiosity_matrix: tuple[np.float64], collaboration_matrix: tuple[tuple[np.float64, ...]]) -> int:
    optimal_paths = None
    optimal_parts = -1
    max_q_value = 1
    for m in range(2, len(paths) + 1):
        path_combinations = combinations(paths, m)
        mean_q_values = np.zeros(m)
        norm_factor = int(0.25 * len(paths))
        print(f"Testing {m=} {norm_factor=}", end="\r")
        for chosen_paths in path_combinations:
            for _ in range(norm_factor):
                for path_no, path in enumerate(chosen_paths):
                    pbp = path_breaking_probability(path, chosen_paths[:path_no] + chosen_paths[path_no+1:], m, curiosity_matrix, collaboration_matrix)
                    # print(pbp)
                    mean_q_values[path_no] += pbp
            else:
                mean_q_values /= norm_factor
                if np.max(mean_q_values) < max_q_value:
                    max_q_value = max(mean_q_values)
                    optimal_parts = m
                    # print(chosen_paths, end="\r")
                else:
                    return optimal_parts, chosen_paths
                break

    return optimal_parts

