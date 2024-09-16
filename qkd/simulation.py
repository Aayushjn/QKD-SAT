from itertools import pairwise
from operator import itemgetter
import random

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from network import Network

n = 10

np.set_printoptions(linewidth=120)

net_graph = Network.random(n)
print(net_graph.disjoint_paths)
# print(net_graph.optimality())

fig, axes = plt.subplots(2, 2)
# pos = nx.bfs_layout(network_graph, 0)
pos = nx.arf_layout(net_graph)

node_color = ["lightgreen" if i == 0 or i == net_graph.number_of_nodes() - 1 else "lightblue" for i in net_graph.nodes]
node_size = 1000 * net_graph.curiosity_matrix
nx.draw_networkx_nodes(
    net_graph, pos=pos, ax=axes[0, 0], nodelist=net_graph.nodes, node_color=node_color, node_size=node_size
)
nx.draw_networkx_edges(net_graph, pos=pos, ax=axes[0, 0], edgelist=net_graph.edges, edge_color="gray")
nx.draw_networkx_labels(net_graph, pos=pos, ax=axes[0, 0], font_size=10)
axes[0, 0].set_title("Network Graph")

edge_list = [
    (i, j)
    for i in range(net_graph.number_of_nodes())
    for j in range(net_graph.number_of_nodes())
    if net_graph.collaboration_matrix[i, j] > 0.0 and i != j
]
cmap = cm.Blues
norm = colors.Normalize(vmin=0.0, vmax=1.0)
nx.draw_networkx_nodes(
    net_graph,
    pos=pos,
    ax=axes[0, 1],
    nodelist=range(1, net_graph.number_of_nodes() - 1),
    node_color="lightblue",
    node_size=500,
)
nx.draw_networkx_edges(
    net_graph,
    pos=pos,
    ax=axes[0, 1],
    edgelist=edge_list,
    edge_color=tuple(map(lambda t: cmap(norm(net_graph.collaboration_matrix[t[0]][t[1]])), edge_list)),
    width=tuple(map(lambda t: net_graph.collaboration_matrix[t[0]][t[1]] * 8, edge_list)),
    alpha=tuple(map(lambda t: net_graph.collaboration_matrix[t[0]][t[1]], edge_list)),
)
nx.draw_networkx_labels(
    net_graph,
    pos=pos,
    ax=axes[0, 1],
    labels={i: str(i) for i in range(1, net_graph.number_of_nodes() - 1)},
    font_size=10,
)
axes[0, 1].set_title("Collaboration Graph")

edge_list = {}
color_values = list(colors.TABLEAU_COLORS.keys())
for i, path in enumerate(net_graph.disjoint_paths):
    color = color_values[i % len(color_values)]
    for edge in pairwise(path):
        edge_list[edge] = color
node_list = {edge[0] for edge in edge_list}
node_list.add(net_graph.number_of_nodes() - 1)
node_color = ["lightgreen" if i == 0 or i == net_graph.number_of_nodes() - 1 else "lightblue" for i in node_list]
node_size = [1000 * net_graph.curiosity_matrix[i] for i in node_list]
nx.draw_networkx_nodes(
    net_graph, pos=pos, ax=axes[1, 0], nodelist=node_list, node_color=node_color, node_size=node_size
)
nx.draw_networkx_edges(net_graph, pos=pos, ax=axes[1, 0], edgelist=edge_list.keys(), edge_color=edge_list.values(), width=2)
nx.draw_networkx_labels(net_graph, pos=pos, ax=axes[1, 0], labels={node: str(node) for node in node_list}, font_size=10)
axes[1, 0].set_title("Disjoint Paths")


x_axis = [str(path) for path in net_graph.disjoint_paths]
y_axis = [path.risk for path in net_graph.disjoint_paths]
bar = axes[1, 1].bar(
    x_axis,
    y_axis,
    color="lightblue",
)
axes[1, 1].bar_label(bar, fmt="%.2f")
axes[1, 1].set_title("Path Risks")


fig.tight_layout()
plt.show()
