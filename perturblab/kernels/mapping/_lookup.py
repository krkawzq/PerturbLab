"""Pure Python implementation of dictionary lookup operations."""

import numpy as np
from typing import Iterable


def lookup_indices(
    mapping: dict[str, int], 
    queries: Iterable[str], 
    fallback_value: int = -1
) -> np.ndarray:
    """Perform vectorized dictionary lookup (str -> int).

    Args:
        mapping: Dictionary mapping strings to integers.
        queries: Iterable of string keys to query.
        fallback_value: Value to return when key is not found.
        
    Returns:
        np.ndarray: int32 array of mapped integers or fallback_value.
    """
    query_list = list(queries)
    n = len(query_list)
    
    results = np.empty(n, dtype=np.int32)
    
    for i, key in enumerate(query_list):
        results[i] = mapping.get(key, fallback_value)
    
    return results


def lookup_tokens(
    indices: list | np.ndarray,
    vocabulary: list[str],
    fallback_value: str = '<unk>'
) -> list[str]:
    """Perform vectorized index-to-string mapping (int -> str).

    Args:
        indices: Integer indices to query.
        vocabulary: List of strings to gather from.
        fallback_value: String to return when index is out of bounds.

    Returns:
        list[str]: List of strings corresponding to the indices.
    """
    idx_arr = np.ascontiguousarray(indices, dtype=np.int32)
    vocab_size = len(vocabulary)
    
    results = []
    for idx in idx_arr:
        if 0 <= idx < vocab_size:
            results.append(vocabulary[idx])
        else:
            results.append(fallback_value)
    
    return results
