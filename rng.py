import os
import sys

import numpy as np
import numpy.typing as npt


def random_curiosity_matrix(
    num_nodes: int, low: float = 0.0, high: float = 1.0, exact: bool = False
) -> npt.NDArray[np.float64]:
    rng = np.random.default_rng(seed=int.from_bytes(os.urandom(4), sys.byteorder))
    if not exact:
        curiosity = rng.uniform(low=low, high=high, size=num_nodes)
    else:
        curiosity = rng.choice((0.0, 1.0), size=num_nodes)
    curiosity[0] = curiosity[num_nodes - 1] = 1.0
    return curiosity


def random_collaboration_matrix(
    num_nodes: int, low: float = 0.0, high: float = 1.0, exact: bool = False
) -> npt.NDArray[np.float64]:
    rng = np.random.default_rng(seed=int.from_bytes(os.urandom(4), sys.byteorder))
    collaboration = np.diag(np.full(num_nodes, 1.0))
    for i in range(1, num_nodes - 1):
        for j in range(1, i):
            if not exact:
                collaboration[i, j] = collaboration[j, i] = rng.uniform(low=low, high=high)
            else:
                collaboration[i, j] = collaboration[j, i] = rng.choice((low, high))
    return collaboration
