from pathlib import Path

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from draw import draw_network_graph
from network import DeterministicNetwork
from qkd import optimality
from qkd import RiskFunction

np.set_printoptions(linewidth=120)

graph = nx.Graph()
graph.add_nodes_from(range(10))
edges = (
    (0, 2),
    (0, 8),
    (0, 7),
    (8, 6),
    (8, 5),
    (8, 2),
    (8, 7),
    (2, 5),
    (2, 1),
    (2, 7),
    (2, 4),
    (7, 5),
    (7, 1),
    (7, 4),
    (7, 3),
    (7, 9),
    (6, 5),
    (5, 1),
    (1, 4),
    (1, 9),
    (4, 3),
    (4, 9),
    (3, 9),
)
graph.add_edges_from(edges)
curiosity = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0])
collaboration = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]
)
net_graph = DeterministicNetwork(graph, curiosity, collaboration, exact=True)

print(net_graph)
print(net_graph.curiosity_matrix)
print(net_graph.collaboration_matrix)

print(
    optimality(
        net_graph.simple_paths,
        tuple(net_graph.curiosity_matrix),
        tuple(tuple(row) for row in net_graph.collaboration_matrix),
        RiskFunction.ATLEAST_ONE_NODE_BREAKS_SECRET,
    )
)

draw_network_graph(net_graph, window_title="n9")

plt.show()
