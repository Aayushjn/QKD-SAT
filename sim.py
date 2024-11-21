import sys
from argparse import ArgumentParser
from itertools import product
from math import ceil
from math import floor
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from network import ProbabilisticNetwork
from qkd import optimality
from qkd import RiskFunction
from rng import random_collaboration_matrix
from rng import random_curiosity_matrix

parser = ArgumentParser()

parser.add_argument("--num-nodes", default=10, type=int)
parser.add_argument("--new-graph", default=False, action="store_true")
parser.add_argument("--num-runs", default=10, type=int)
parser.add_argument("--bin-size", default=0.2, type=float)

args = parser.parse_args(sys.argv[1:])

np.set_printoptions(linewidth=120)

graph_dir = Path.cwd().joinpath("graphs").joinpath(f"n{args.num_nodes}")
graph_dir.mkdir(parents=True, exist_ok=True)

res_path = Path.cwd().joinpath("results")
res_path.mkdir(parents=True, exist_ok=True)
outfile = res_path.joinpath("results.csv")

if args.new_graph:
    net_graph = ProbabilisticNetwork.random(args.num_nodes)
    net_graph.to_dir(graph_dir)
    completed_idx = set()
else:
    net_graph = ProbabilisticNetwork.from_dir(graph_dir)
    if not outfile.exists() or outfile.stat().st_size == 0:
        completed_idx = set()
    else:
        existing_data = pd.read_csv(str(outfile))
        completed_idx = set(existing_data.loc[existing_data["graph_size"] == args.num_nodes]["bins"].apply(eval))

num_bins = floor(1.0 / args.bin_size)
num_nodes = net_graph.number_of_nodes()
pos = nx.spring_layout(net_graph)
print(net_graph)


def run_simulation(idx: tuple[int, int]) -> tuple[tuple[int, int], int, np.float64]:
    i, j = idx  # i for curiosity, j for collaboration

    bin_total = 0
    breaking_prob = 0.0
    for _ in range(args.num_runs):
        ng = ProbabilisticNetwork(
            net_graph,
            curiosity_matrix=random_curiosity_matrix(
                num_nodes, low=i * args.bin_size, high=(i * args.bin_size) + args.bin_size
            ),
            collaboration_matrix=random_collaboration_matrix(
                num_nodes, low=j * args.bin_size, high=(j * args.bin_size) + args.bin_size
            ),
        )

        # Optimality calculation (remains unchanged)
        opt = optimality(
            ng.simple_paths,
            tuple(ng.curiosity_matrix),
            tuple(tuple(row) for row in ng.collaboration_matrix),
            RiskFunction.ATLEAST_ONE_NODE_BREAKS_SECRET,
        )

        bin_total += opt.shares
        breaking_prob += opt.breaking_probability

    return idx, ceil(bin_total / args.num_runs), breaking_prob / args.num_runs


def batch_simulation(simulations: list[tuple[int, int]], batch_size: int = 10):
    results = []

    # with ProcessPoolExecutor(max_workers=(cpu_count() or 2) - 1) as executor:
    # futures = [executor.submit(run_simulation, idx) for idx in simulations]
    for idx in simulations:
        results.append(run_simulation(idx))
        if len(results) >= batch_size:
            yield results
            results = []

    if results:
        yield results


simulations = [(i, j) for i, j in product(range(num_bins), repeat=2) if (i, j) not in completed_idx]

for batch in batch_simulation(simulations):
    df = pd.DataFrame.from_records(batch, columns=["bins", "optimal_parts", "breaking_probability"])
    df["graph_size"] = [num_nodes] * len(df)
    df = df[["graph_size", "bins", "optimal_parts", "breaking_probability"]]
    df.to_csv(str(outfile), mode="a+", header=not outfile.exists() or outfile.stat().st_size == 0, index=False)
