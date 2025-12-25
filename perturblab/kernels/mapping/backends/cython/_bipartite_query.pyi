"""Type stubs for Cython bipartite query module."""

import numpy as np
import scipy.sparse

def bipartite_graph_query(
    graph: scipy.sparse.spmatrix,
    queries: np.ndarray | list[int],
    check_bounds: bool = True,
) -> list[np.ndarray]: ...
