# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: infer_types=True

"""Cython-accelerated implementation of dictionary lookup operations."""

import numpy as np
cimport numpy as np

# Initialize NumPy C-API
np.import_array()

# Type aliases
ctypedef np.int32_t INT32_t


def lookup_indices(
    dict mapping, 
    object queries, 
    int fallback_value=-1
) -> np.ndarray:
    """Perform high-performance vectorized dictionary lookup (str -> int).
    
    Uses optimized Cython code for fast dictionary access.
    
    Args:
        mapping: Dictionary mapping strings to integers.
        queries: Iterable of string keys to query.
        fallback_value: Value to return when key is not found.
    
    Returns:
        np.ndarray: int32 array of mapped integers.
    """
    # Ensure input is a list
    cdef list query_list = list(queries) if not isinstance(queries, list) else queries
    cdef Py_ssize_t n = len(query_list)
    
    # Pre-allocate output array
    cdef np.ndarray[INT32_t, ndim=1] results = np.empty(n, dtype=np.int32)
    cdef INT32_t* res_ptr = <INT32_t*>np.PyArray_DATA(results)
    
    cdef Py_ssize_t i
    cdef object key
    cdef INT32_t fallback = fallback_value
    
    for i in range(n):
        key = query_list[i]
        if key in mapping:
            res_ptr[i] = <INT32_t>mapping[key]
        else:
            res_ptr[i] = fallback
            
    return results


def lookup_tokens(
    object indices, 
    object vocabulary,
    str fallback_value='<unk>'
) -> list:
    """Perform vectorized index-to-string mapping (int -> str).
    
    Optimized for list and NumPy array vocabularies.
    
    Args:
        indices: Integer indices to query.
        vocabulary: String vocabulary (list[str] or np.ndarray[object]).
        fallback_value: String to return when index is out of bounds.
    
    Returns:
        list[str]: List of strings corresponding to the indices.
    """
    # Prepare index array
    cdef np.ndarray[INT32_t, ndim=1] idx_arr = np.ascontiguousarray(indices, dtype=np.int32)
    cdef Py_ssize_t n = idx_arr.shape[0]
    cdef INT32_t* idx_ptr = <INT32_t*>np.PyArray_DATA(idx_arr)
    
    # Convert vocabulary to list for uniform access
    cdef list vocab_list
    if isinstance(vocabulary, list):
        vocab_list = vocabulary
    else:
        vocab_list = list(vocabulary)
    
    cdef Py_ssize_t vocab_size = len(vocab_list)
    
    # Build result list
    cdef list results = []
    cdef Py_ssize_t i
    cdef INT32_t idx
    
    for i in range(n):
        idx = idx_ptr[i]
        
        if idx < 0 or idx >= vocab_size:
            results.append(fallback_value)
        else:
            results.append(vocab_list[idx])
    
    return results
