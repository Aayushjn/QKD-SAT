from typing import Mapping

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import networkx as nx

from network import ProbabilisticNetwork


def draw_network_graph(net_graph: ProbabilisticNetwork, pos: Mapping | None = None, window_title: str = "Figure"):
    fig, axes = plt.subplots()

    if pos is None:
        pos = nx.arf_layout(net_graph)
    node_color = [
        "lightgreen" if i == 0 or i == net_graph.number_of_nodes() - 1 else "lightblue" for i in net_graph.nodes
    ]
    cmap = cm.Blues
    norm = colors.Normalize(vmin=0.0, vmax=1.0)

    node_size = 1000 * net_graph.curiosity_matrix
    nx.draw_networkx_nodes(
        net_graph, pos=pos, ax=axes, nodelist=net_graph.nodes, node_color=node_color, node_size=node_size
    )
    nx.draw_networkx_edges(net_graph, pos=pos, ax=axes, edgelist=net_graph.edges, edge_color="gray")
    nx.draw_networkx_labels(net_graph, pos=pos, ax=axes, font_size=10)
    # axes.set_title("Network Graph")
    plt.get_current_fig_manager().set_window_title(window_title)

    fig, axes = plt.subplots()
    edge_list = [
        (i, j)
        for i in range(net_graph.number_of_nodes())
        for j in range(net_graph.number_of_nodes())
        if net_graph.collaboration_matrix[i, j] > 0.0 and i != j
    ]
    nx.draw_networkx_nodes(
        net_graph,
        pos=pos,
        ax=axes,
        nodelist=range(1, net_graph.number_of_nodes() - 1),
        node_color="lightblue",
        node_size=500,
    )
    nx.draw_networkx_edges(
        net_graph,
        pos=pos,
        ax=axes,
        edgelist=edge_list,
        edge_color=tuple(cmap(norm(net_graph.collaboration_matrix[t[0]][t[1]])) for t in edge_list),
        width=tuple(net_graph.collaboration_matrix[t[0]][t[1]] * 8 for t in edge_list),
        alpha=tuple(net_graph.collaboration_matrix[t[0]][t[1]] for t in edge_list),
    )
    nx.draw_networkx_labels(
        net_graph,
        pos=pos,
        ax=axes,
        labels={i: str(i) for i in range(1, net_graph.number_of_nodes() - 1)},
        font_size=10,
    )
    # axes.set_title("Collaboration Graph")
    fig.tight_layout()
    plt.get_current_fig_manager().set_window_title(window_title)
