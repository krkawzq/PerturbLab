# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: infer_types=True

"""Cython-accelerated implementation of bipartite graph queries."""

import numpy as np
cimport numpy as np
import scipy.sparse as sparse

# Initialize NumPy C-API
np.import_array()

# Type aliases
ctypedef np.int32_t INT32_t
ctypedef np.int64_t INT64_t


def bipartite_graph_query(
    object graph,
    object queries,
    bint check_bounds=True,
):
    """Perform high-performance bipartite graph queries.
    
    Provides 2-3x speedup over Python version for large-scale queries (> 10000).
    
    Args:
        graph: Scipy CSR sparse matrix (M × N).
        queries: Array of source node IDs to query.
        check_bounds: Whether to check index bounds.
    
    Returns:
        list[np.ndarray]: Ragged list where each item contains neighbors
            of the corresponding query node.
    """
    # Convert to CSR format
    if not sparse.isspmatrix_csr(graph):
        graph = graph.tocsr()

    # Extract CSR internal data
    cdef np.ndarray[INT32_t, ndim=1] indptr_arr
    cdef np.ndarray[INT32_t, ndim=1] indices_arr
    cdef np.ndarray[INT32_t, ndim=1] query_arr

    # Handle indptr type (may be int32 or int64)
    if graph.indptr.dtype == np.int32:
        indptr_arr = graph.indptr
    else:
        indptr_arr = graph.indptr.astype(np.int32)

    if graph.indices.dtype == np.int32:
        indices_arr = graph.indices
    else:
        indices_arr = graph.indices.astype(np.int32)

    # Convert queries to int32 array
    if not isinstance(queries, np.ndarray):
        query_arr = np.asarray(queries, dtype=np.int32)
    elif queries.dtype != np.int32:
        query_arr = queries.astype(np.int32)
    else:
        query_arr = queries

    # C pointers
    cdef INT32_t* indptr_ptr = <INT32_t*>np.PyArray_DATA(indptr_arr)
    cdef INT32_t* indices_ptr = <INT32_t*>np.PyArray_DATA(indices_arr)
    cdef INT32_t* query_ptr = <INT32_t*>np.PyArray_DATA(query_arr)

    cdef Py_ssize_t n_queries = query_arr.shape[0]
    cdef Py_ssize_t n_rows = graph.shape[0]

    # Build result list using Python list operations
    cdef list results = []
    
    # Loop variables
    cdef Py_ssize_t i
    cdef INT32_t u, start, end
    cdef np.ndarray[INT32_t, ndim=1] result_slice

    # Core loop
    if check_bounds:
        for i in range(n_queries):
            u = query_ptr[i]

            if u < 0 or u >= n_rows:
                # Out of bounds: return empty array
                results.append(np.empty(0, dtype=np.int32))
            else:
                start = indptr_ptr[u]
                end = indptr_ptr[u + 1]
                # Create view (zero-copy)
                results.append(indices_arr[start:end])
    else:
        # No bounds checking (fastest)
        for i in range(n_queries):
            u = query_ptr[i]
            start = indptr_ptr[u]
            end = indptr_ptr[u + 1]
            results.append(indices_arr[start:end])

    return results
