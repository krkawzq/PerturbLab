"""Type stubs for Cython-accelerated bipartite graph operations."""

import numpy as np
import scipy.sparse as sparse
from typing import Union


def bipartite_graph_query(
    graph: sparse.spmatrix,
    queries: Union[np.ndarray, list[int]],
    check_bounds: bool = True,
) -> list[np.ndarray]:
    """Perform high-performance bipartite graph queries.

    Args:
        graph: Scipy CSR sparse matrix (M × N).
        queries: Array of source node IDs to query.
        check_bounds: Whether to check index bounds.

    Returns:
        list[np.ndarray]: Ragged list where each item contains neighbors
            of the corresponding query node.
    """
    ...
