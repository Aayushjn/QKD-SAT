from operator import itemgetter
import os
import sys
from functools import cache
from itertools import pairwise
from typing import Iterable

import networkx as nx
import numpy as np

from point import Point


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
    reduced_paths: list[NetworkPath]
    curiosity_matrix: np.ndarray[np.float64]
    collaboration_matrix: np.ndarray[np.float64]

    def __init__(
        self, graph: nx.Graph, curiosity_matrix: np.ndarray[np.float64], collaboration_matrix: np.ndarray[np.float64]
    ):
        super().__init__(graph)
        self.curiosity_matrix = curiosity_matrix
        self.collaboration_matrix = collaboration_matrix

    def determine_reduced_paths(self):
        reduced_paths = [
            NetworkPath(path, risk=self.path_risk(tuple(path)))
            for path in nx.all_simple_paths(self, 0, self.number_of_nodes() - 1)
        ]
        reduced_paths.sort(key=lambda path: path.risk)
        
        last_path = reduced_paths[0]
        self.reduced_paths = [last_path]
        for path in reduced_paths[1:]:
            if path.risk != last_path.risk:
                self.reduced_paths.append(path)
                last_path = path

    def node_risk(self, node: int) -> float:
        return self.curiosity_matrix[node] * (
            1.0 - np.prod([1 - self.collaboration_matrix[node, j] for j in range(1, self.number_of_nodes() - 1)])
        )

    def edge_risk(self, edge: tuple[int, int]) -> float:
        if edge[1] == self.number_of_nodes() - 1:
            return 0.0
        return self.curiosity_matrix[edge[1]] * np.mean(self.collaboration_matrix[edge[1]])

    def edge_capacity(self, edge: tuple[int, int]) -> float:
        return 1.0 - self.edge_risk(edge)

    @cache
    def path_risk(self, path: tuple[int, ...]) -> float:
        return np.sum([self.edge_risk(edge) for edge in pairwise(path)]) / (len(path) - 1)

    @classmethod
    def random(cls, num_nodes: int) -> "Network":
        graph = nx.Graph()
        for node in range(num_nodes):
            graph.add_node(node, point=Point.random())
        distances = [
            (graph.nodes[i]["point"].euclid_distance(graph.nodes[j]["point"]), i, j)
            for i in range(num_nodes)
            for j in range(i)
        ]
        links = list(map(itemgetter(1, 2), sorted(distances, key=itemgetter(0))))
        for edge in links:
            if {0, num_nodes - 1} == set(edge):
                continue
            graph.add_edge(*edge)
            if nx.is_connected(graph):
                break
        else:
            raise RuntimeError("Unexpected inability to construct graph")

        rng = np.random.default_rng(seed=int.from_bytes(os.urandom(4), sys.byteorder))
        curiosity = rng.random(num_nodes)
        curiosity[0] = curiosity[num_nodes - 1] = 1.0
        collaboration = np.diag(np.full(num_nodes, 1.0))
        for i in range(1, num_nodes - 1):
            for j in range(1, i):
                collaboration[i, j] = collaboration[j, i] = rng.random()

        assert nx.is_connected(graph)
        assert nx.has_path(graph, 0, graph.number_of_nodes() - 1)

        g = cls(graph, curiosity, collaboration)
        nx.set_edge_attributes(graph, {e: g.edge_risk(e) for e in graph.edges}, "weight")
        g.determine_reduced_paths()
        return g
