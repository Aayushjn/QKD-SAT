from typing import Mapping

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import networkx as nx

from network import Network


def draw_network_graph(net_graph: Network, pos: Mapping | None = None, window_title: str = "Figure"):
    fig, axes = plt.subplots(nrows=1, ncols=2)

    if pos is None:
        pos = nx.spring_layout(net_graph)
    node_color = [
        "lightgreen" if i == 0 or i == net_graph.number_of_nodes() - 1 else "lightblue" for i in net_graph.nodes
    ]
    cmap = cm.Blues
    norm = colors.Normalize(vmin=0.0, vmax=1.0)

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
        edge_color=tuple(cmap(norm(net_graph.collaboration_matrix[t[0]][t[1]])) for t in edge_list),
        width=tuple(net_graph.collaboration_matrix[t[0]][t[1]] * 8 for t in edge_list),
        alpha=tuple(net_graph.collaboration_matrix[t[0]][t[1]] for t in edge_list),
    )
    nx.draw_networkx_labels(
        net_graph,
        pos=pos,
        ax=axes[1],
        labels={i: str(i) for i in range(1, net_graph.number_of_nodes() - 1)},
        font_size=10,
    )
    axes[1].set_title("Collaboration Graph")
    fig.tight_layout()
    plt.get_current_fig_manager().set_window_title(window_title)
