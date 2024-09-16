import networkx as nx
import matplotlib.pyplot as plt

from graph import NetworkGraph
from model import ProbabilisticModel, simulator

if __name__ == "__main__":
    NUM_NODES = 12
    MIN_PATHS = 5

    g = NetworkGraph(NUM_NODES)
    g.ensure_at_least_n_paths(MIN_PATHS)
    print(f"Reduced paths: {g.reduced_paths}")
    
    nx.draw(g, with_labels=True)
    plt.show()

    curiosity_matrix = ProbabilisticModel.random_curiosity(NUM_NODES)
    collaboration_matrix = ProbabilisticModel.random_collaboration(NUM_NODES)
    print(f"{curiosity_matrix=}")
    print(f"{collaboration_matrix=}")

    obj_fn = ProbabilisticModel.objective_function(NUM_NODES, curiosity_matrix, collaboration_matrix)
    print(obj_fn(g.reduced_paths, ProbabilisticModel.ObjectFunction.SOME_NODE_BREAK_SECRET))
    print(obj_fn(g.reduced_paths, ProbabilisticModel.ObjectFunction.NO_NODE_BREAK_SECRET))

    optimal_choice = ProbabilisticModel.optimal_choice(NUM_NODES, curiosity_matrix, collaboration_matrix)
    print(optimal_choice(g.reduced_paths, 4))

    sim = simulator(NUM_NODES, g.reduced_paths, curiosity_matrix, collaboration_matrix)
    for _ in range(20):
        print(next(sim))
