"""Type stubs for Cython bipartite query module."""

from typing import Union

import numpy as np
import scipy.sparse

def bipartite_graph_query(
    graph: scipy.sparse.spmatrix,
    queries: Union[np.ndarray, list[int]],
    check_bounds: bool = True,
) -> list[np.ndarray]: ...
