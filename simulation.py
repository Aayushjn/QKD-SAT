from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from network import Network
from draw import draw_network_graph

np.set_printoptions(linewidth=120)

graph_dir = Path.cwd().joinpath("graphs")

for path in graph_dir.iterdir():
    if path.is_dir():
        net_graph = Network.from_dir(path)
        
        draw_network_graph(net_graph, window_title=path.name)

plt.show()


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

# rpl

# num paths above threshold
# paths determined by path breaking probability
# vary curiosity matrix and collaboration for the same graph
