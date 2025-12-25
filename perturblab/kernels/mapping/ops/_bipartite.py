"""Bipartite graph query operations with automatic backend selection.

This module provides high-performance bipartite graph queries with
automatic backend selection performed at import time.

Backend priority: Cython > Python
"""

from __future__ import annotations

import numpy as np
import scipy.sparse

from perturblab.utils import get_logger

logger = get_logger()

__all__ = ["bipartite_graph_query"]


# =============================================================================
# Backend Selection (performed at import time)
# =============================================================================

_BACKEND_NAME = None
_bipartite_graph_query_impl = None


def _select_backend():
    """Select backend at import time."""
    global _BACKEND_NAME, _bipartite_graph_query_impl

    # Try Cython backend first
    try:
        from ..backends.cython._bipartite_query import bipartite_graph_query as cython_impl

        _BACKEND_NAME = "Cython"
        _bipartite_graph_query_impl = cython_impl
        logger.debug("Bipartite query backend: Cython (2-3x speedup)")
        return
    except ImportError:
        pass

    # Fall back to Python
    from ..backends.python._bipartite import bipartite_graph_query as python_impl

    _BACKEND_NAME = "Python"
    _bipartite_graph_query_impl = python_impl
    logger.debug(
        "Bipartite query backend: Python (consider compiling Cython for large-scale queries)"
    )


# Select backend on module import
_select_backend()


# =============================================================================
# Public API
# =============================================================================


def bipartite_graph_query(
    graph: scipy.sparse.spmatrix,
    queries: np.ndarray | list[int],
    check_bounds: bool = True,
) -> list[np.ndarray]:
    """Perform efficient batch queries on bipartite graph.

    Leverages CSR sparse matrix format for O(1) row slicing to quickly
    retrieve node neighbors in a bipartite graph.

    Parameters
    ----------
    graph : scipy.sparse.spmatrix
        Sparse matrix (M × N) representing the bipartite graph.
        Rows represent source nodes, columns represent target nodes.
        Non-zero entries indicate edges between nodes.
    queries : np.ndarray or list[int]
        Array of source node IDs to query.
    check_bounds : bool, default=True
        Whether to check index bounds to avoid errors.
        Set to False for maximum performance if you're sure all queries are valid.

    Returns
    -------
    list[np.ndarray]
        Ragged list where item i contains all target node IDs
        corresponding to queries[i].

    Examples
    --------
    >>> import scipy.sparse as sp
    >>> import numpy as np
    >>>
    >>> # Create a bipartite graph: 4 genes × 5 GO terms
    >>> # Gene 0 -> GO terms [1, 2]
    >>> # Gene 1 -> GO term [0]
    >>> # Gene 2 -> GO terms [2, 3, 4]
    >>> rows = [0, 0, 1, 2, 2, 2]
    >>> cols = [1, 2, 0, 2, 3, 4]
    >>> data = [1, 1, 1, 1, 1, 1]
    >>> graph = sp.csr_matrix((data, (rows, cols)), shape=(4, 5))
    >>>
    >>> # Query neighbors for genes 0, 2
    >>> queries = [0, 2]
    >>> neighbors = bipartite_graph_query(graph, queries)
    >>> print(neighbors[0])  # GO terms for gene 0
    [1 2]
    >>> print(neighbors[1])  # GO terms for gene 2
    [2 3 4]

    Notes
    -----
    - Time complexity: O(n + k) where n is number of queries, k is total edges queried
    - Space complexity: O(k) for the output arrays
    - The graph is automatically converted to CSR format if needed
    - Returns zero-copy array views when possible; modifications affect original matrix
    - For small queries (< 1000), Python and Cython have similar performance
    - For large queries (> 10000), Cython provides 2-3x speedup

    See Also
    --------
    scipy.sparse.csr_matrix : Compressed Sparse Row matrix format
    """
    return _bipartite_graph_query_impl(graph, queries, check_bounds)
