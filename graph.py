from datetime import timedelta
from operator import itemgetter

import networkx as nx

from point import Point
from util import limit_function_execution


class NetworkGraph(nx.Graph):
    num_nodes: int
    start_node: int
    end_node: int
    num_adversaries: int
    links: list[tuple[int, int]]
    reduced_paths: set[tuple[int, ...]]

    def __init__(self, num_nodes: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.start_node = 0
        self.end_node = num_nodes - 1
        self.num_adversaries = num_nodes - 2

    def _add_random_nodes(self):
        self.clear()
        for i in range(self.num_nodes):
            self.add_node(i, point=Point.random())

    def _calculate_distances(self):
        distances = [
            (self.nodes[i]["point"].euclid_distance(self.nodes[j]["point"]), i, j)
            for i in range(self.num_nodes)
            for j in range(i)
        ]
        self.links = list(map(itemgetter(1, 2), sorted(distances, key=itemgetter(0))))

    def _generate_graph(self):
        for edge in self.links:
            if {self.start_node, self.end_node} == set(edge):
                continue
            self.add_edge(*edge)
            if nx.is_connected(self):
                break
        else:
            raise RuntimeError("Unexpected inability to construct graph")

    def _calculate_paths(self):
        simple_paths = nx.all_simple_paths(self, self.start_node, self.end_node)
        self.reduced_paths = set()
        for path in simple_paths:
            append_path = True
            for some_path in self.reduced_paths.copy():
                path_set = set(path)
                if path_set.issuperset(some_path):
                    append_path = False
                elif path_set.issubset(some_path):
                    self.reduced_paths.add(tuple(some_path))
            if append_path:
                self.reduced_paths.add(tuple(path))

    def construct_connected_graph(self):
        self._add_random_nodes()
        self._calculate_distances()
        self._generate_graph()
        self._calculate_paths()

    def ensure_at_least_n_paths(self, n: int, retries: int = 20, timeout: timedelta = timedelta(seconds=45)) -> bool:
        for _ in range(retries):
            try:
                limit_function_execution(10, self.construct_connected_graph)
            except TimeoutError:
                continue
            # self.construct_connected_graph()
            if len(self.reduced_paths) >= n:
                return True
        return False
