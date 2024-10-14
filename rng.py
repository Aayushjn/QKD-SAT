import os
import sys

import numpy as np


def random_curiosity_matrix(num_nodes: int, low: float = 0.0, high: float = 1.0) -> np.ndarray[np.float64]:
    rng = np.random.default_rng(seed=int.from_bytes(os.urandom(4), sys.byteorder))
    curiosity = rng.uniform(low=low, high=high, size=num_nodes)
    curiosity[0] = curiosity[num_nodes - 1] = 1.0
    return curiosity


def random_collaboration_matrix(num_nodes: int, low: float = 0.0, high: float = 1.0) -> np.ndarray[np.float64]:
    rng = np.random.default_rng(seed=int.from_bytes(os.urandom(4), sys.byteorder))
    collaboration = np.diag(np.full(num_nodes, 1.0))
    for i in range(1, num_nodes - 1):
        for j in range(1, i):
            collaboration[i, j] = collaboration[j, i] = rng.uniform(low=low, high=high)
    return collaboration
