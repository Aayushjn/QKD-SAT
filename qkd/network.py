from itertools import pairwise
import os
import random
import sys
from typing import Iterable

import networkx as nx
import numpy as np


class NetworkPath(tuple):
    risk: float

    def __new__(cls, path: Iterable[int], risk: float = 0.0):
        val = super().__new__(cls, path)
        val.risk = risk
        return val
    
    def __str__(self):
        return super().__repr__()

    def __repr__(self):
        return f"({super().__repr__()}, risk={self.risk})"


class Network(nx.Graph):
    disjoint_paths: list[NetworkPath]
    curiosity_matrix: np.ndarray[np.float64]
    collaboration_matrix: np.ndarray[np.float64]

    def __init__(
        self, graph: nx.Graph, curiosity_matrix: np.ndarray[np.float64], collaboration_matrix: np.ndarray[np.float64]
    ):
        super().__init__(graph)
        self.curiosity_matrix = curiosity_matrix
        self.collaboration_matrix = collaboration_matrix

    def determine_disjoint_paths(self):
        self.disjoint_paths = [
            NetworkPath(path, risk=self.path_risk(path))
            for path in nx.node_disjoint_paths(self, 0, self.number_of_nodes() - 1)
        ]
        self.disjoint_paths.sort(key=lambda path: path.risk)

    def node_risk(self, node: int) -> float:
        return self.curiosity_matrix[node] * (
            1.0 - np.prod([1 - self.collaboration_matrix[node, j] for j in range(1, self.number_of_nodes() - 1)])
        )

    def edge_risk(self, edge: tuple[int, int]) -> float:
        if edge[1] == self.number_of_nodes() - 1:
            return 0.0
        return 0.5 * self.curiosity_matrix[edge[1]] + 0.5 * np.mean(self.collaboration_matrix[edge[1]])

    def edge_capacity(self, edge: tuple[int, int]) -> float:
        return 1.0 - self.edge_risk(edge)

    def path_risk(self, path: list[tuple[int, ...]]) -> float:
        return np.sum([self.edge_risk(edge) for edge in pairwise(path)]) / (len(path) - 1)

    def optimality(self, threshold: float = 0.8) -> int:
        candidates = []
        for m in range(len(self.disjoint_paths), 0, -1):
            q_values = [self.node_risk(node) ** m for node in self.nodes]
            print(q_values)
            if all(q_value < threshold for q_value in q_values):
                candidates.append(m)
        return min(candidates)

    @classmethod
    def random(cls, num_nodes: int) -> "Network":
        graph = nx.generators.hnm_harary_graph(num_nodes, 3 * num_nodes)

        rng = np.random.default_rng(seed=int.from_bytes(os.urandom(4), sys.byteorder))
        curiosity = rng.random(num_nodes)
        curiosity[0] = curiosity[num_nodes - 1] = 1.0
        collaboration = np.diag(np.full(num_nodes, 1.0))
        for i in range(1, num_nodes - 1):
            for j in range(1, i):
                collaboration[i, j] = collaboration[j, i] = rng.random()

        if graph.has_edge(0, graph.number_of_nodes() - 1):
            graph.remove_edge(0, graph.number_of_nodes() - 1)

        if not nx.has_path(graph, 0, graph.number_of_nodes() - 1):
            intermediate_node = random.choice(range(1, graph.number_of_nodes() - 1))
            graph.add_edge(0, intermediate_node)
            graph.add_edge(intermediate_node, graph.number_of_nodes() - 1)

        assert nx.is_connected(graph)
        assert nx.has_path(graph, 0, graph.number_of_nodes() - 1)

        g = cls(graph, curiosity, collaboration)
        nx.set_edge_attributes(graph, {e: g.edge_capacity(e) for e in graph.edges}, "capacity")
        g.determine_disjoint_paths()
        return g
