import random
import sys
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from network import Network

from qkd import optimality

np.set_printoptions(linewidth=120)

# num_nodes = random.randint(15, 20)
net_graph = Network.random(25)
net_graph.to_file(Path(".", "graphs", "n25"))
sys.exit(0)
net_graph = Network.from_file(Path(".", "graphs"))
num_nodes = net_graph.number_of_nodes()

fig, axes = plt.subplots(1, 2)
# pos = nx.bfs_layout(net_graph, 0)
pos = nx.arf_layout(net_graph)

node_color = ["lightgreen" if i == 0 or i == net_graph.number_of_nodes() - 1 else "lightblue" for i in net_graph.nodes]
node_size = 1000 * net_graph.curiosity_matrix
nx.draw_networkx_nodes(
    net_graph, pos=pos, ax=axes[0], nodelist=net_graph.nodes, node_color=node_color, node_size=node_size
)
nx.draw_networkx_edges(net_graph, pos=pos, ax=axes[0], edgelist=net_graph.edges, edge_color="gray")
nx.draw_networkx_labels(net_graph, pos=pos, ax=axes[0], font_size=10)
axes[0].set_title("Network Graph")

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
    ax=axes[1],
    nodelist=range(1, net_graph.number_of_nodes() - 1),
    node_color="lightblue",
    node_size=500,
)
nx.draw_networkx_edges(
    net_graph,
    pos=pos,
    ax=axes[1],
    edgelist=edge_list,
    edge_color=tuple(map(lambda t: cmap(norm(net_graph.collaboration_matrix[t[0]][t[1]])), edge_list)),
    width=tuple(map(lambda t: net_graph.collaboration_matrix[t[0]][t[1]] * 8, edge_list)),
    alpha=tuple(map(lambda t: net_graph.collaboration_matrix[t[0]][t[1]], edge_list)),
)
nx.draw_networkx_labels(
    net_graph,
    pos=pos,
    ax=axes[1],
    labels={i: str(i) for i in range(1, net_graph.number_of_nodes() - 1)},
    font_size=10,
)
axes[1].set_title("Collaboration Graph")

# edge_list = {}
# color_values = list(colors.TABLEAU_COLORS.keys())
# for i, path in enumerate(net_graph.reduced_paths):
#     color = color_values[i % len(color_values)]
#     for edge in pairwise(path):
#         edge_list[edge] = color
# node_list = {edge[0] for edge in edge_list}
# node_list.add(net_graph.number_of_nodes() - 1)
# node_color = ["lightgreen" if i == 0 or i == net_graph.number_of_nodes() - 1 else "lightblue" for i in node_list]
# node_size = [1000 * net_graph.curiosity_matrix[i] for i in node_list]
# nx.draw_networkx_nodes(
#     net_graph, pos=pos, ax=axes[1, 0], nodelist=node_list, node_color=node_color, node_size=node_size
# )
# nx.draw_networkx_edges(
#     net_graph, pos=pos, ax=axes[1, 0], edgelist=edge_list.keys(), edge_color=edge_list.values(), width=2
# )
# nx.draw_networkx_labels(net_graph, pos=pos, ax=axes[1, 0], labels={node: str(node) for node in node_list}, font_size=10)
# axes[1, 0].set_title("Disjoint Paths")


# x_axis = [str(path) for path in net_graph.reduced_paths]
# y_axis = [path.risk for path in net_graph.reduced_paths]
# bar = axes[1, 1].bar(
#     x_axis,
#     y_axis,
#     color="lightblue",
# )
# axes[1, 1].bar_label(bar, fmt="%.2f")
# axes[1, 1].set_title("Path Risks")


fig.tight_layout()

print(net_graph)
print("Average path length: ", sum(len(path) for path in net_graph.simple_paths) / len(net_graph.simple_paths))
# print("Simple Paths:", net_graph.simple_paths)
# print(net_graph.curiosity_matrix)
# print(net_graph.collaboration_matrix)

opt = optimality(
    net_graph.simple_paths,
    tuple(net_graph.curiosity_matrix),
    tuple(tuple(row) for row in net_graph.collaboration_matrix),
)
print("\r\n", opt, end="\r\n")

plt.show()


# rpl

# num paths above threshold
# paths determined by path breaking probability
# vary curiosity matrix and collaboration for the same graph
