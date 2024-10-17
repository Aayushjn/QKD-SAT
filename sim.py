import sys
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from math import ceil
from math import floor
from os import cpu_count
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import trange

from network import Network
from qkd import optimality
from qkd import RiskFunction
from rng import random_collaboration_matrix
from rng import random_curiosity_matrix

parser = ArgumentParser()

parser.add_argument("--num-nodes", default=10, type=int)
parser.add_argument("--new-graph", default=False, action="store_true")
parser.add_argument("--num-runs", default=10, type=int)
parser.add_argument("--bin-size", default=0.2, type=float)
parser.add_argument("--vary", required=True, choices=("both", "curiosity", "collaboration"))

args = parser.parse_args(sys.argv[1:])

np.set_printoptions(linewidth=120)

graph_dir = Path.cwd().joinpath("graphs").joinpath(f"n{args.num_nodes}")
graph_dir.mkdir(parents=True, exist_ok=True)

if args.new_graph:
    net_graph = Network.random(args.num_nodes)
    net_graph.to_dir(graph_dir)
else:
    net_graph = Network.from_dir(graph_dir)

num_bins = floor(1.0 / args.bin_size)
num_nodes = net_graph.number_of_nodes()
pos = nx.spring_layout(net_graph)
print(net_graph)


def run_simulation(idx: tuple[int, int]) -> tuple[tuple[int, int], int]:
    i, j = idx

    bin_total = 0
    for _ in trange(args.num_runs, dynamic_ncols=True, leave=True):
        if args.vary == "curiosity":
            ng = Network(
                net_graph,
                curiosity_matrix=random_curiosity_matrix(
                    num_nodes, low=i * args.bin_size, high=(i * args.bin_size) + args.bin_size
                ),
                collaboration_matrix=net_graph.collaboration_matrix,
            )
        elif args.vary == "collaboration":
            ng = Network(
                net_graph,
                curiosity_matrix=net_graph.curiosity_matrix,
                collaboration_matrix=random_collaboration_matrix(
                    num_nodes, low=j * args.bin_size, high=(j * args.bin_size) + args.bin_size
                ),
            )
        else:
            ng = Network(
                net_graph,
                curiosity_matrix=random_curiosity_matrix(
                    num_nodes, low=i * args.bin_size, high=(i * args.bin_size) + args.bin_size
                ),
                collaboration_matrix=random_collaboration_matrix(
                    num_nodes, low=j * args.bin_size, high=(j * args.bin_size) + args.bin_size
                ),
            )

        opt = optimality(
            ng.simple_paths,
            tuple(ng.curiosity_matrix),
            tuple(tuple(row) for row in ng.collaboration_matrix),
            RiskFunction.MEAN,
        )

        bin_total += opt[0]

    return idx, ceil(bin_total / args.num_runs)


if args.vary == "curiosity":
    curiosities = list(range(num_bins))
    collaborations = [-1]
elif args.vary == "collaboration":
    curiosities = [-1]
    collaborations = list(range(num_bins))
else:
    curiosities = list(range(num_bins))
    collaborations = list(range(num_bins))

with ProcessPoolExecutor(max_workers=(cpu_count() or 2) - 1) as executor:
    results = executor.map(run_simulation, product(curiosities, collaborations))

df = pd.DataFrame.from_records(results, columns=["bins", "optimal_parts"])
df["graph_size"] = [num_nodes] * len(df)
df = df[["graph_size", "bins", "optimal_parts"]]

res_path = Path.cwd().joinpath("results")
res_path.mkdir(parents=True, exist_ok=True)
file = res_path.joinpath(f"{args.vary}.csv")

df.to_csv(str(file), mode="a+", header=not file.exists() or file.stat().st_size == 0, index=False)
