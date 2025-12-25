"""Pure Python implementation of bipartite graph queries."""


import numpy as np
import scipy.sparse as sparse


def bipartite_graph_query(
    graph: sparse.spmatrix,
    queries: np.ndarray | list[int],
    check_bounds: bool = True,
) -> list[np.ndarray]:
    """Perform efficient batch queries on bipartite graph.

    Leverages CSR sparse matrix format for O(1) row slicing to quickly
    retrieve node neighbors.

    Args:
        graph: Scipy sparse matrix (M × N). Rows represent source nodes,
            columns represent target nodes.
        queries: Array of source node IDs to query.
        check_bounds: Whether to check index bounds to avoid segfault.

    Returns:
        list[np.ndarray]: Ragged list where item i contains all target node
            IDs corresponding to queries[i].

    Notes:
        - For small queries (< 1000), Python version is sufficient.
        - For large queries (> 10000), use Cython-accelerated version.
        - Returns zero-copy array views; modifications affect original matrix.
    """
    # Convert to CSR format (only format supporting O(1) row slicing)
    if not sparse.isspmatrix_csr(graph):
        graph = graph.tocsr()

    # Convert queries to numpy array for better indexing
    if not isinstance(queries, np.ndarray):
        queries = np.asarray(queries, dtype=np.int32)

    # Cache C arrays to avoid repeated getattr calls
    c_indptr = graph.indptr  # Row pointer array
    c_indices = graph.indices  # Column index array
    n_rows = graph.shape[0]

    # Pre-allocate result list
    n_queries = len(queries)
    results = [None] * n_queries

    # Core query loop
    if check_bounds:
        for i, u in enumerate(queries):
            if u < 0 or u >= n_rows:
                results[i] = np.array([], dtype=c_indices.dtype)
            else:
                start = c_indptr[u]
                end = c_indptr[u + 1]
                results[i] = c_indices[start:end]
    else:
        for i, u in enumerate(queries):
            start = c_indptr[u]
            end = c_indptr[u + 1]
            results[i] = c_indices[start:end]

    return results
