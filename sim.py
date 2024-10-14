import csv
import sys
from argparse import ArgumentParser
from copy import deepcopy
from itertools import product
from math import ceil
from math import floor
from pathlib import Path

import networkx as nx
import numpy as np
from network import Network
from rng import random_collaboration_matrix
from rng import random_curiosity_matrix
from tqdm import tqdm
from tqdm import trange

from qkd import optimality

parser = ArgumentParser()

parser.add_argument("--num-nodes", default=10, type=int)
parser.add_argument("--num-runs", default=10, type=int)
parser.add_argument("--bin-size", default=0.2, type=float)
parser.add_argument("--vary", required=True, choices=("both", "curiosity", "collaboration"))

args = parser.parse_args(sys.argv[1:])

np.set_printoptions(linewidth=120)


num_bins = floor(1.0 / args.bin_size)
net_graph = Network.from_file(Path(".", "graphs", f"n{args.num_nodes}"))
num_nodes = net_graph.number_of_nodes()
print(net_graph)

data = {}

if args.vary == "curiosity":
    curiosities = list(range(num_bins))
    collaborations = [-1]
elif args.vary == "collaboration":
    curiosities = [-1]
    collaborations = list(range(num_bins))
else:
    curiosities = list(range(num_bins))
    collaborations = list(range(num_bins))

for bin_count, (i, j) in tqdm(enumerate(product(curiosities, collaborations))):
    bin_total = 0
    path_length_total = 0

    for _ in trange(args.num_runs):
        ng = deepcopy(net_graph)
        low_prob = i * args.bin_size
        high_prob = (i * args.bin_size) + args.bin_size
        if args.vary == "curiosity":
            ng.curiosity_matrix = random_curiosity_matrix(
                num_nodes, low=i * args.bin_size, high=(i * args.bin_size) + args.bin_size
            )
        elif args.vary == "collaboration":
            ng.collaboration_matrix = random_collaboration_matrix(
                num_nodes, low=j * args.bin_size, high=(j * args.bin_size) + args.bin_size
            )
        else:
            ng.curiosity_matrix = random_curiosity_matrix(
                num_nodes, low=i * args.bin_size, high=(i * args.bin_size) + args.bin_size
            )
            ng.collaboration_matrix = random_collaboration_matrix(
                num_nodes, low=j * args.bin_size, high=(j * args.bin_size) + args.bin_size
            )

        nx.set_edge_attributes(net_graph, {e: ng.edge_weight(e) for e in net_graph.edges}, "weight")
        ng.determine_simple_paths()

        opt = optimality(
            net_graph.simple_paths,
            tuple(net_graph.curiosity_matrix),
            tuple(tuple(row) for row in net_graph.collaboration_matrix),
        )

        bin_total += opt[0]
        path_length_total += sum(len(path) for path in ng.simple_paths) / len(ng.simple_paths)

    data[(i, j)] = (ceil(bin_total / args.num_runs), path_length_total / args.num_runs)

with Path(".", "results", f"{args.vary}.csv").open("a+") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["n", "bins", "parts"])
    if csvfile.tell() == 0:
        writer.writeheader()
    for bin, (parts, _) in data.items():
        writer.writerow({"n": num_nodes, "bins": bin, "parts": parts})
