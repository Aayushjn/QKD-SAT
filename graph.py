from math import sqrt
from operator import itemgetter

import networkx as nx

from point import Point


class NetworkGraph(nx.Graph):
    num_nodes: int
    start_node: int
    end_node: int
    num_adversaries: int
    links: list[tuple[int, int]]
    reduced_paths: list[list[int]]

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
            (self.nodes[i]["point"].distance(self.nodes[j]["point"]), i, j)
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
            raise RuntimeError("Unexpected in-ability to construct graph")

    def _calculate_paths(self):
        simple_paths = nx.all_simple_paths(self, self.start_node, self.end_node)
        self.reduced_paths = []
        for path in simple_paths:
            append_path = True
            for some_path in self.reduced_paths.copy():
                path_set = set(path)
                if path_set.issuperset(some_path):
                    append_path = False
                elif path_set.issubset(some_path):
                    self.reduced_paths.append(some_path)
            if append_path:
                self.reduced_paths.append(path)

    def construct_connected_graph(self):
        self._add_random_nodes()
        self._calculate_distances()
        self._generate_graph()
        self._calculate_paths()

    def ensure_at_least_n_paths(self, n: int, retries: int = 20) -> bool:
        for _ in range(retries):
            self.construct_connected_graph()
            if len(self.reduced_paths) >= n:
                return True
        return False

    def calculate_layout_points(self):
        initial_positions = {i: self.nodes[i]["point"].to_tuple() for i in range(self.num_nodes)}
        spring_positions = nx.spring_layout(self, pos=initial_positions, k=1 / sqrt(sqrt(self.num_nodes)))
        x_max, x_min = max(map(itemgetter(0), spring_positions.values())), min(
            map(itemgetter(0), spring_positions.values())
        )
        y_max, y_min = max(map(itemgetter(1), spring_positions.values())), min(
            map(itemgetter(1), spring_positions.values())
        )
        for i in range(self.num_nodes):
            self.nodes[i]["layout_point"] = Point.normalize(
                *spring_positions[i], x_range=(x_min, x_max), y_range=(y_min, y_max)
            )
