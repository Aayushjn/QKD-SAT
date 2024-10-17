import random
from functools import cache
from itertools import pairwise
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import numpy.typing as npt

from rng import random_collaboration_matrix
from rng import random_curiosity_matrix


class NetworkPath(tuple):
    weight: float

    def __new__(cls, path: Iterable[int], weight: float = 0.0):
        val = super().__new__(cls, path)
        val.weight = weight
        return val

    def __str__(self):
        return super().__repr__()

    def __repr__(self):
        return f"({super().__repr__()}, weight={self.weight})"


class Network(nx.Graph):
    simple_paths: tuple[NetworkPath]
    curiosity_matrix: npt.NDArray[np.float64]
    collaboration_matrix: npt.NDArray[np.float64]

    def __init__(
        self,
        graph: nx.Graph,
        curiosity_matrix: npt.NDArray[np.float64] | None = None,
        collaboration_matrix: npt.NDArray[np.float64] | None = None,
    ):
        super().__init__(graph)

        if curiosity_matrix is None:
            curiosity_matrix = random_curiosity_matrix(self.number_of_nodes())
        if collaboration_matrix is None:
            collaboration_matrix = random_collaboration_matrix(self.number_of_nodes())

        self.curiosity_matrix = curiosity_matrix
        self.collaboration_matrix = collaboration_matrix

        nx.set_edge_attributes(graph, {e: self.edge_weight(e) for e in graph.edges}, "weight")
        self.determine_simple_paths()

    def determine_simple_paths(self):
        reduced_paths = [
            NetworkPath(path, weight=round(self.path_weight(tuple(path)), 5))
            for path in nx.all_simple_paths(self, 0, self.number_of_nodes() - 1)
        ]
        reduced_paths.sort(key=lambda path: path.weight)

        last_path = reduced_paths[0]
        simple_paths = [last_path]
        for path in reduced_paths[1:]:
            if path.weight != last_path.weight:
                simple_paths.append(path)
                last_path = path

        self.simple_paths = tuple(simple_paths)

    def node_risk(self, node: int) -> float:
        return self.curiosity_matrix[node] * (
            1.0 - np.prod([1 - self.collaboration_matrix[node, j] for j in range(1, self.number_of_nodes() - 1)])
        )

    @cache
    def edge_weight(self, edge: tuple[int, int]) -> float:
        if edge[1] == self.number_of_nodes() - 1:
            return 0.0
        return self.curiosity_matrix[edge[1]] * np.mean(self.collaboration_matrix[edge[1]])

    @cache
    def path_weight(self, path: tuple[int, ...]) -> float:
        return np.sum([self.edge_weight(edge) for edge in pairwise(path)]) / (len(path) - 1)

    def to_dir(self, path: Path):
        nx.write_adjlist(self, str(path.joinpath("graph.txt")))
        np.save(path.joinpath("curiosity.npy"), self.curiosity_matrix)
        np.save(path.joinpath("collaboration.npy"), self.collaboration_matrix)

    @classmethod
    def from_dir(cls, path: Path) -> "Network":
        graph = nx.read_adjlist(path.joinpath("graph.txt"))
        graph = nx.relabel_nodes(graph, {node: int(node) for node in graph.nodes})
        curiosity = np.load(path.joinpath("curiosity.npy"))
        collaboration = np.load(path.joinpath("collaboration.npy"))
        return cls(graph, curiosity, collaboration)

    @classmethod
    def random(cls, num_nodes: int) -> "Network":
        graph = nx.generators.harary_graph.hnm_harary_graph(
            num_nodes, random.randrange(int(1.5 * num_nodes), 2 * num_nodes)
        )

        if graph.has_edge(0, num_nodes - 1):
            graph.remove_edge(0, num_nodes - 1)
            # Ensure the graph remains connected after removing the edge
            if not nx.has_path(graph, 0, num_nodes - 1):
                intermediate = random.randint(1, num_nodes - 2)
                graph.add_edge(0, intermediate)
                graph.add_edge(intermediate, num_nodes - 1)

        assert nx.is_connected(graph)
        assert nx.has_path(graph, 0, graph.number_of_nodes() - 1)

        return cls(graph)
